/**
 * The shared machinery behind the Postgres and MySQL backends: one `Store`
 * implementation over any bounded SQL connection, plus the backend base class
 * that connects lazily and creates the schema if it is missing.
 *
 * Everything genuinely dialect-specific — the DDL's column TYPES, the block-dedup
 * upsert syntax, how a connection is opened and bounded, how a driver spells
 * "deadlock" — lives in a small `Dialect` object supplied by
 * `postgres.ts`/`mysql.ts`. Everything else (the logical model, the statement
 * text, the transaction shape, the fail-open retry, the reconnect) is written
 * ONCE here, so the two adapters cannot drift apart in semantics: the
 * conformance suite runs the same assertions against both.
 *
 * PLACEHOLDERS. Python could share one statement text verbatim because psycopg
 * and PyMySQL both use `%s`; the Node drivers do not agree (`pg` wants `$1`,
 * `mysql2` wants `?`). Rather than maintain two copies of every statement, the
 * shared text is written with `?` — mysql2's own spelling — and the Postgres
 * connection rewrites it to `$1..$n` on the way out (`SqlConn.query` is where a
 * driver's quirks belong). One statement text, no drift.
 *
 * Naming: every table is prefixed `ctxdiff_` because, unlike a private `.ctrace`
 * file, these tables live in a database the user shares with their own
 * application — the prefix keeps ctxdiff's four tables obviously ctxdiff's and
 * avoids colliding with a `run`/`call`/`block` table that is already there.
 * Three columns are also renamed from their SQLite spellings (`usage` ->
 * `usage_json`, `position` -> `pos`, `text` -> `body`) because `usage` is a
 * RESERVED word in MySQL and the other two are keyword-adjacent; picking
 * non-reserved names beats quoting identifiers in every statement. These names
 * match the Python adapters exactly, so a database written by either SDK is
 * readable by the other.
 *
 * Session ordering: SQLite orders sessions by physical `rowid` — their INSERT
 * order. The network stores reproduce that with an explicit `insert_order`
 * column, spelled `BIGSERIAL` on Postgres and `BIGINT AUTO_INCREMENT` on MySQL.
 * Ordering by `(started_at, id)` instead would be wrong twice over, and both
 * ways matter for the case these backends exist to serve — several containers
 * writing into one database:
 *
 * - clocks disagree. A container a few seconds behind writes a session that
 *   sorts BEFORE one written minutes earlier, so `openReader()` binds — and
 *   `ctxdiff diff` analyzes — somebody else's run;
 * - timestamps tie. Two sessions starting in the same millisecond fall back to
 *   comparing random uuid hex: a coin flip that can reorder them between one
 *   read and the next, where SQLite was deterministic.
 *
 * Insert order has neither problem: it is assigned by the database, is total,
 * and means the same thing on all three backends. `started_at` is still stored
 * and still required to be non-empty — it is what the UI DISPLAYS — it is just
 * no longer what anything sorts by.
 */
import { randomUUID } from "node:crypto";
import type {
  Call,
  CallBlock,
  OpenSessionArgs,
  RecordCallArgs,
  Run,
  Session,
  Store,
  StoreBackend,
} from "./base.js";
import { EmptyStoreError } from "./base.js";
import type { Block } from "../models.js";
import { SCHEMA_VERSION } from "./schema.js";
import { VERSION } from "../version.js";

// Bounded write retry (mirrors the SQLite store's philosophy). A network DB can
// transiently refuse a write — a deadlock between two agents' writers, a
// serialization failure under a strict isolation level — where an immediate
// retry succeeds. Bounded on both axes so a genuinely broken DB gives up fast
// and lets the caller's fail-open guard drop the call, rather than holding the
// writer queue (and therefore `close()`) indefinitely.
const WRITE_MAX_ATTEMPTS = 3;
const WRITE_BACKOFF_START_MS = 50;
const WRITE_BACKOFF_MAX_MS = 200;

// Bounded retry for the auto-create DDL when two backends race to create the
// same table, or when the server fails one of them transiently — a MySQL
// deadlock on the data dictionary is the observed case (see
// `SQLBackend.ensureSchema`). Three attempts because the race is between
// processes starting together and one of them wins each round; the delay is a
// fixed, tiny pause — just long enough for the winner's catalog insert to
// commit, and short enough that a cold start is not noticeably slower.
const SCHEMA_MAX_ATTEMPTS = 3;
const SCHEMA_RETRY_DELAY_MS = 100;

/** Default seconds to wait for a TCP connect/handshake before giving up. Small
 * on purpose: capture must degrade quickly, never stall an agent's first LLM
 * call waiting on a database that is not there. */
export const DEFAULT_CONNECT_TIMEOUT = 5;

/** Default seconds any single statement may run before it is aborted. The writer
 * queue is shared by the whole run and is drained by `close()`, so an unbounded
 * query would turn a slow DB into a hung shutdown. */
export const DEFAULT_STATEMENT_TIMEOUT = 10;

/** Seconds added to the statement bound to get the CLIENT-side deadline (see
 * `withDeadline`). The client bound must sit ABOVE the server-side one so the
 * server gets the chance to report its own timeout as a proper, classifiable
 * error; the client deadline is the backstop for a server that stops answering
 * entirely, where no server-side knob can ever fire. */
export const DEADLINE_MARGIN = 2;

export const RUN_TABLE = "ctxdiff_run";
export const CALL_TABLE = "ctxdiff_call";
export const BLOCK_TABLE = "ctxdiff_block";
export const CALL_BLOCK_TABLE = "ctxdiff_call_block";

// --- statements shared by every dialect (written with `?` placeholders) -------

const INSERT_RUN =
  `INSERT INTO ${RUN_TABLE} ` +
  "(id, project, started_at, provider, models, ctxdiff_version, schema_version) " +
  "VALUES (?, ?, ?, ?, ?, ?, ?)";

const SELECT_NEWEST_RUN =
  `SELECT id, schema_version FROM ${RUN_TABLE} ORDER BY insert_order DESC LIMIT 1`;

const SELECT_MAX_SCHEMA = `SELECT MAX(schema_version) FROM ${RUN_TABLE}`;

const SELECT_SESSIONS =
  `SELECT id, project, started_at, provider, models FROM ${RUN_TABLE} ORDER BY insert_order`;

const SELECT_TURN_COUNTS = `SELECT run_id, COUNT(*) FROM ${CALL_TABLE} GROUP BY run_id`;

const SELECT_AGENTS =
  `SELECT run_id, agent FROM ${CALL_TABLE} WHERE agent IS NOT NULL ORDER BY run_id, seq`;

const SELECT_RUN =
  "SELECT id, project, started_at, provider, models, ctxdiff_version " +
  `FROM ${RUN_TABLE} WHERE id = ?`;

const SELECT_CALLS =
  "SELECT id, run_id, seq, params, usage_json, latency_ms, error, " +
  `agent, step, provider FROM ${CALL_TABLE} WHERE run_id = ? ORDER BY seq`;

const INSERT_CALL =
  `INSERT INTO ${CALL_TABLE} ` +
  "(id, run_id, seq, params, usage_json, latency_ms, error, agent, step, provider) " +
  "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";

const INSERT_CALL_BLOCK =
  `INSERT INTO ${CALL_BLOCK_TABLE} ` +
  "(call_id, block_id, pos, label, label_source) VALUES (?, ?, ?, ?, ?)";

const SELECT_CALL_BLOCKS =
  "SELECT cb.pos, cb.label, cb.label_source, b.content_hash, b.role, " +
  "       b.kind, b.body, b.token_count, b.token_method " +
  `FROM ${CALL_BLOCK_TABLE} cb JOIN ${BLOCK_TABLE} b ` +
  "  ON b.content_hash = cb.block_id " +
  "WHERE cb.call_id = ? ORDER BY cb.pos";

const UPDATE_MODELS = `UPDATE ${RUN_TABLE} SET models = ? WHERE id = ?`;

// --- the connection seam ------------------------------------------------------

/**
 * One bounded SQL connection, as the shared layer needs it: run a statement with
 * `?` placeholders and get rows back as plain arrays, say whether the connection
 * is still usable, and close. Deliberately tiny — a driver's `pg.Client` /
 * `mysql2` connection is wrapped in one of these by its adapter, which is where
 * placeholder syntax, row shape, timeouts and error plumbing get normalized so
 * nothing below this line is driver-aware.
 *
 * Rows are arrays (not objects) for the same reason Python materializes tuples:
 * positional access means the two dialects' column-name casing/quirks can never
 * change what a shared SELECT reads back.
 */
export interface SqlConn {
  /** Run one statement and resolve its rows as positional arrays. Rejects on
   * error, and — crucially — within a bounded time even when the server has
   * stopped answering (see `withDeadline`). */
  query(sql: string, params?: unknown[]): Promise<unknown[][]>;
  /** Whether this connection is known to be unusable (the driver's own liveness
   * flag, plus anything the adapter has observed). Consulted before a retry, so
   * a dead socket is REOPENED rather than written to again. */
  isDead(): boolean;
  /**
   * Mark this connection DETACHED: while it sits IDLE its socket is removed
   * from the event loop's liveness accounting (`socket.unref()`), so an open
   * connection can never by itself stop the host program from exiting.
   *
   * This is ctxdiff's equivalent of Python's DAEMON writer thread, and it exists
   * for the same reason: a program that finishes its work and forgets
   * `tracer.close()` must still exit. In Node a live TCP socket is a referenced
   * libuv handle, so without this the tracing connection would hold the process
   * open FOREVER — a debugging tool preventing a script from ending is a far
   * worse failure than losing the last few queued writes.
   *
   * IDLE is the crucial word. Unref'ing unconditionally would be a worse bug
   * than the one it fixes: with no other referenced handle, Node does not wait
   * for an unref'd socket, so a `await store.recordCall(...)` — a supported,
   * exported, documented call — would simply never settle and the process would
   * exit mid-script. So the socket is re-referenced for the duration of every
   * statement and every close, and released again the moment the connection
   * goes quiet (see `DetachableSocket`).
   *
   * Applied to the WRITER's connection only (see `SQLBackend.openSession`). A
   * reader is something the caller is explicitly awaiting from start to finish,
   * so its socket simply stays referenced.
   */
  unref(): void;
  /** Release the socket. Best-effort: never rejects. */
  close(): Promise<void>;
}

/** Everything about one SQL dialect the shared layer cannot assume. */
export interface Dialect {
  /** Short name, used in messages and tests. */
  readonly name: string;
  /** `CREATE TABLE IF NOT EXISTS` statements, parents first. */
  ddl(): string[];
  /** Insert a block, or no-op when this content hash is already stored — the
   * content-addressed dedup primitive. */
  insertBlockDedup(): string;
  /** True when a failed write is worth ONE bounded retry (deadlock,
   * serialization failure, lock wait). */
  isTransient(err: unknown): boolean;
  /** True when the error means THIS CONNECTION is gone, so a retry must reopen
   * before it can mean anything. */
  isConnectionLost(err: unknown): boolean;
  /** True when the auto-create DDL lost a race to another process creating the
   * SAME table. */
  isSchemaRace(err: unknown): boolean;
}

/** A client-side deadline expired: the server accepted the connection and then
 * stopped answering, so no server-side timeout could ever fire. Carries its own
 * type because it is always treated as connection loss — the adapter destroys
 * the socket when it fires, so the next attempt must reopen. */
export class DeadlineExceededError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DeadlineExceededError";
  }
}

/**
 * Bound `work` to `ms` milliseconds. THE bound that matters in Node.
 *
 * Why it is not redundant with the drivers' own knobs: a connect timeout covers
 * connect and authentication and nothing after them, and a SERVER-side
 * `statement_timeout`/`max_execution_time` cannot fire when the packets carrying
 * its verdict are being dropped. A wedged box, a hung pooler or a network
 * partition therefore leaves a query pending forever — and while that does not
 * freeze Node's event loop the way a synchronous read would, it does leave
 * `close()` waiting on a promise that will never settle, which is the shutdown
 * hang this exists to prevent.
 *
 * On expiry `onExpire` runs (the adapter destroys the socket, so the connection
 * is unambiguously dead and the next attempt reopens) and the returned promise
 * rejects with `DeadlineExceededError`. The losing promise's own eventual
 * rejection is swallowed, so a late failure can never surface as an unhandled
 * rejection in the host's process. The timer is `unref`'d so a pending deadline
 * never keeps a program alive.
 */
export function withDeadline<T>(
  work: Promise<T>,
  ms: number,
  label: string,
  onExpire?: () => void,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      // The work promise is now abandoned: attach a no-op handler so its late
      // rejection (the socket erroring after we destroyed it) stays contained.
      void work.then(
        () => undefined,
        () => undefined,
      );
      try {
        onExpire?.();
      } catch {
        /* destroying an already-dead socket must not mask the deadline */
      }
      reject(new DeadlineExceededError(`ctxdiff: ${label} exceeded ${ms}ms`));
    }, ms);
    timer.unref?.();
    work.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (err) => {
        clearTimeout(timer);
        reject(err);
      },
    );
  });
}

/**
 * Keeps a DETACHED connection's socket referenced exactly while work is in
 * flight on it, and unreferenced whenever it is idle.
 *
 * The two requirements pull in opposite directions and both are absolute:
 *
 * - an IDLE tracing connection must not hold the host process open, or a script
 *   that forgets `tracer.close()` never exits (Node counts a live TCP socket as
 *   a reason to keep running);
 * - a BUSY one must, or an `await` on it can never settle — with no other
 *   referenced handle Node does not wait for an unref'd socket, it fires
 *   `beforeExit` and terminates, silently truncating the caller's script.
 *
 * Reference counting satisfies both: `during()` refs before the work and
 * unrefs after the last concurrent operation finishes. Until `detach()` is
 * called it does nothing at all, so a reader — which a caller awaits from start
 * to finish — is left exactly as the driver made it.
 */
export class DetachableSocket {
  private readonly stream: () => { ref?(): void; unref?(): void } | undefined;
  private detached = false;
  private inFlight = 0;

  /** `stream` is read lazily rather than captured, because a driver may only
   * expose its socket once connected — and may replace it. */
  constructor(stream: () => { ref?(): void; unref?(): void } | undefined) {
    this.stream = stream;
  }

  /** Start detaching this socket when idle. Idempotent; applies immediately, so
   * a connection detached while quiet is released at once. */
  detach(): void {
    this.detached = true;
    this.apply();
  }

  /** Run `work` with the socket referenced, releasing it again once every
   * concurrent operation has finished. The result (or rejection) passes through
   * untouched. */
  async during<T>(work: Promise<T>): Promise<T> {
    this.inFlight += 1;
    this.apply();
    try {
      return await work;
    } finally {
      this.inFlight -= 1;
      this.apply();
    }
  }

  /** Reference or release the socket to match the current state. Guarded: a
   * driver that exposes no socket, or one already destroyed, simply has nothing
   * to adjust — and failing to adjust it must never break a query. */
  private apply(): void {
    try {
      const socket = this.stream();
      if (!socket) return;
      if (this.detached && this.inFlight === 0) socket.unref?.();
      else socket.ref?.();
    } catch {
      /* nothing to (un)reference */
    }
  }
}

/**
 * Sleep `ms`, holding the event loop open only when `keepAlive` says to.
 *
 * The default is an UNREFERENCED timer: a pause that is not part of any work in
 * progress must never be the reason a finished program cannot exit.
 *
 * `keepAlive` is for the one shape where the opposite is true — a RETRY backoff,
 * which sits in the middle of a half-finished job. Both retry sites run on the
 * writer's connection, whose socket is detached while idle (`SqlConn.unref`), so
 * during the pause the process can hold NO referenced handle at all: Node then
 * fires `beforeExit` and terminates, abandoning a session that was opened but
 * never written to, or a call that was rolled back and not yet re-run — with no
 * warning, because nothing failed. The retry has to be able to say "I am still
 * working", and a referenced timer is the only thing that says it.
 *
 * Safe because it is BOUNDED at both sites: at most two schema pauses of
 * `SCHEMA_RETRY_DELAY_MS` and at most two write pauses capped at
 * `WRITE_BACKOFF_MAX_MS`, so the worst case a forgotten `close()` can add to a
 * program's exit is a few hundred milliseconds — not the indefinite hold the
 * unref was guarding against.
 */
function sleep(ms: number, keepAlive = false): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms);
    if (!keepAlive) timer.unref?.();
  });
}

/** A driver error's message, defensively — an adapter may reject with something
 * that is not an `Error` at all. */
function messageOf(err: unknown): string {
  const message = (err as { message?: unknown } | null)?.message;
  return typeof message === "string" && message !== "" ? message : String(err);
}

/**
 * The error a reader raises when it could neither create ctxdiff's tables nor
 * read them — the one case `openReader`'s "ignore a refused ensureSchema" rule
 * has to explain rather than swallow.
 *
 * Both causes are named, in the order they happened, because either one alone is
 * misleading: the CREATE failure on its own reads as "you cannot read this
 * database" when ctxdiff only wanted to create tables the reader may not even
 * need, and the SELECT failure on its own hides that ctxdiff tried a DDL
 * statement at all — which is the fact that explains a permissions ticket. The
 * remedy line is included because there are exactly two, and a reader hitting
 * this is an operator who wants to know which one they need.
 */
function readerOpenError(schemaError: unknown, readError: unknown): Error {
  return new Error(
    "ctxdiff: cannot read this store — creating ctxdiff's tables failed " +
      `(${messageOf(schemaError)}) and reading them failed ` +
      `(${messageOf(readError)}). A role that can only SELECT needs the ` +
      `${RUN_TABLE}/${CALL_TABLE}/${BLOCK_TABLE}/${CALL_BLOCK_TABLE} tables to ` +
      "exist already: create them once with a role that can CREATE, or grant " +
      "SELECT on them to this one.",
    { cause: readError },
  );
}

/** Coerce a driver's integer column to a JS number. Needed because `pg` returns
 * `BIGINT` (and `COUNT(*)`) as a STRING to avoid precision loss, while `mysql2`
 * and `node:sqlite` return numbers — without this, `latencyMs` would read back
 * as "1234" from Postgres and as 1234 everywhere else, and the conformance
 * suite's cross-backend equality would be a lie. */
function toInt(value: unknown): number {
  return typeof value === "number" ? value : Number(value);
}

/** `toInt` for a nullable column: null/undefined stay null. */
function toIntOrNull(value: unknown): number | null {
  return value === null || value === undefined ? null : toInt(value);
}

/** Read a TEXT column as a string. `mysql2` can hand back a Buffer for a
 * column whose charset it does not consider textual; normalizing here means no
 * caller ever sees a Buffer where the SQLite store gives a string. */
function toText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value instanceof Buffer) return value.toString("utf8");
  return String(value);
}

/** `toText` for a nullable column. */
function toTextOrNull(value: unknown): string | null {
  if (value === null || value === undefined) return null;
  return toText(value);
}

/**
 * A `Store` handle bound to ONE session in a networked SQL database.
 *
 * Owns a single connection, exactly like `CTrace` owns a single `node:sqlite`
 * connection, and is used the same way: the tracer's ONE writer queue does every
 * write through it (so the connection is never used concurrently — a SQL
 * connection tolerates one statement at a time), and a separate read handle is
 * opened by the CLI/viewer.
 *
 * Construct via `SQLBackend.openSession()`/`openReader()`, never directly —
 * those apply the schema, run the version gate, and bind the run id.
 */
export class SQLStore implements Store {
  private conn: SqlConn;
  private readonly dialect: Dialect;
  private readonly runId: string;
  private models: string[];
  private modelsSeen: Set<string>;
  private readonly reconnect: (() => Promise<SqlConn>) | null;
  private closed = false;

  /**
   * Wrap an already-connected, already-migrated connection for one session. How:
   * stores the connection, the dialect (for the one piece of SQL that differs —
   * the block dedup upsert — and for error classification), and the session id
   * every write lands against, then seeds the in-memory mirror of the run's
   * `models` list so `noteModel` can gate on a set lookup instead of a SELECT
   * per call — the same trick, and the same first-seen ordering, as the SQLite
   * store.
   *
   * `reconnect` returns a FRESH connection to the same database — in practice
   * the owning backend's `connect`. It is what makes the retry in
   * `withWriteRetry` mean anything when the failure was the connection itself
   * dying: without it, "retry" re-runs the write against the same dead socket.
   * Optional so a handle can be built without one (a reader, a unit test), in
   * which case a lost connection simply fails.
   */
  constructor(
    conn: SqlConn,
    dialect: Dialect,
    runId: string,
    models: string[] = [],
    reconnect: (() => Promise<SqlConn>) | null = null,
  ) {
    this.conn = conn;
    this.dialect = dialect;
    this.runId = runId;
    this.models = [...models];
    this.modelsSeen = new Set(this.models);
    this.reconnect = reconnect;
  }

  // --- internals -------------------------------------------------------------

  /** Run one read query and resolve its rows. A thin passthrough that exists so
   * every read in this class reads the same way and a future cross-cutting
   * concern (metrics, logging) has one place to live. */
  private query(sql: string, params: unknown[] = []): Promise<unknown[][]> {
    return this.conn.query(sql, params);
  }

  /**
   * Run a write `fn` (a thunk wrapping ONE transaction), retrying a bounded
   * number of times when the failure was transient (deadlock/serialization) or
   * the connection itself died. Why: unlike a local file, a networked DB
   * legitimately refuses a write that would succeed moments later, and losing a
   * call to that would be needless capture loss. Bounded on both axes — fixed
   * attempt cap, capped backoff — so a genuinely broken DB re-raises fast into
   * the recorder's fail-open guard instead of stalling the writer queue. Each
   * attempt rolls back before re-running, so a retry can never double-write.
   *
   * The connection-loss case is handled by REOPENING first (see `reopen`). This
   * is the difference between a retry that works and one that only looks like it
   * does: a killed backend, a recycled pooler connection or a database restarted
   * mid-deploy leaves this handle's socket permanently dead, so re-running the
   * same statement on it fails identically every time — capture for the whole
   * process would end at the first blip, with one warning, even though the
   * database is healthy again a second later.
   */
  private async withWriteRetry<T>(fn: () => Promise<T>): Promise<T> {
    let delay = WRITE_BACKOFF_START_MS;
    for (let attempt = 0; ; attempt++) {
      try {
        return await fn();
      } catch (err) {
        const lost = this.connectionLost(err);
        await this.rollbackQuietly();
        const lastAttempt = attempt === WRITE_MAX_ATTEMPTS - 1;
        if (lastAttempt || !(lost || this.dialect.isTransient(err))) throw err;
        // Nothing to retry ON: fail now rather than spin against a dead socket.
        if (lost && !(await this.reopen())) throw err;
        // KEEP-ALIVE: this pause is the middle of a write, not idle time. The
        // connection it will retry on is detached (and a just-reopened one is
        // detached and idle), so without a referenced timer here the process has
        // nothing holding it open and exits — dropping this call and every call
        // after it, silently. Bounded by WRITE_BACKOFF_MAX_MS and the attempt
        // cap, so it can never become a hang.
        await sleep(delay, true);
        delay = Math.min(delay * 2, WRITE_BACKOFF_MAX_MS);
      }
    }
  }

  /**
   * True when the failure means THIS CONNECTION is gone, rather than the
   * statement having been refused.
   *
   * Three signals, because none alone is enough. A client-side deadline always
   * counts (the adapter destroyed the socket when it fired). The dialect
   * classifies the error itself (Postgres reports a kill as SQLSTATE 57P01 or
   * the 08xxx class; mysql2 as `PROTOCOL_CONNECTION_LOST`/`ECONNRESET`/errno
   * 2006/2013) — but only for the FIRST failure after the kill: every later
   * attempt raises a driver-level "connection is closed" carrying no code at
   * all. So the connection's own liveness flag is consulted too, which stays
   * authoritative afterwards.
   */
  private connectionLost(err: unknown): boolean {
    if (err instanceof DeadlineExceededError) return true;
    if (this.dialect.isConnectionLost(err)) return true;
    return this.conn.isDead();
  }

  /**
   * Replace this handle's dead connection with a fresh one, reporting whether it
   * worked. How: closes the old connection (best-effort — it is already gone;
   * this just releases the local socket) and swaps in whatever the backend's
   * `connect` returns, which re-applies the same timeouts/session settings the
   * original was opened with. A handle built without a `reconnect`, or a
   * reconnect that itself fails (the server is still down), returns false so the
   * caller re-raises into the fail-open guard instead of retrying against
   * nothing.
   */
  private async reopen(): Promise<boolean> {
    if (this.reconnect === null || this.closed) return false;
    await this.conn.close();
    try {
      this.conn = await this.reconnect();
    } catch {
      return false; // still unreachable; fail open and try again next call
    }
    return true;
  }

  /** Abandon the current transaction, swallowing any failure. Guarded because
   * rollback is itself a network round-trip: on a dropped connection it rejects,
   * and that must not mask the ORIGINAL write error the caller is about to see
   * (or fail a retry that would otherwise have worked). */
  private async rollbackQuietly(): Promise<void> {
    try {
      await this.conn.query("ROLLBACK");
    } catch {
      /* best-effort; the real error is the caller's */
    }
  }

  // --- writing ---------------------------------------------------------------

  /**
   * Persist one call and its ordered blocks in a single transaction — the same
   * contract, and the same on-the-wire content, as the SQLite store's
   * `recordCall`. Each block is upserted by content_hash (stored once, ignored
   * if already present: content-addressed dedup, via the dialect's
   * no-op-on-duplicate insert); each membership row records the block's position
   * and label within THIS call. Resolves to the new call id.
   *
   * Params/usage are stored as JSON TEXT (not a native JSON column) so both
   * dialects, and the SQLite store, hold byte-identical values and no driver
   * type-mapping can reinterpret them on the way back out.
   */
  async recordCall(args: RecordCallArgs): Promise<string> {
    const callId = randomUUID().replace(/-/g, "");
    const {
      seq,
      params,
      usage,
      latencyMs,
      error,
      callBlocks,
      agent = null,
      step = null,
      provider = null,
    } = args;

    await this.withWriteRetry(async () => {
      await this.conn.query("BEGIN");
      try {
        await this.conn.query(INSERT_CALL, [
          callId,
          this.runId,
          seq,
          JSON.stringify(params),
          usage !== null ? JSON.stringify(usage) : null,
          latencyMs,
          error,
          agent,
          step,
          provider,
        ]);
        for (const cb of callBlocks) {
          const b: Block = cb.block;
          await this.conn.query(this.dialect.insertBlockDedup(), [
            b.contentHash,
            b.role,
            b.kind,
            b.text,
            b.tokenCount,
            b.tokenMethod,
          ]);
          await this.conn.query(INSERT_CALL_BLOCK, [
            callId,
            b.contentHash,
            cb.position,
            cb.label,
            cb.labelSource,
          ]);
        }
        await this.conn.query("COMMIT");
      } catch (err) {
        await this.rollbackQuietly();
        throw err;
      }
    });

    // Best-effort model roll-up onto the run row, mirroring the SQLite store
    // exactly: the call is ALREADY COMMITTED, so a failure here must not
    // propagate (the caller would log "failed to record" for a call that WAS
    // saved). It is self-healing — `noteModel` only marks a model seen once its
    // UPDATE commits, so the next call carrying it retries.
    try {
      // `||`, not `??`: Python spells this `params.get("model") or
      // params.get("modelId")`, so an EMPTY model falls through to the Bedrock
      // spelling rather than being rolled up as a blank.
      const model =
        (params["model"] as string | undefined) ||
        (params["modelId"] as string | undefined) ||
        null;
      await this.noteModel(model);
    } catch {
      /* roll-up is best-effort; the call is saved */
    }
    return callId;
  }

  /**
   * Append `model` to the run's `models` list the first time it is seen,
   * preserving first-seen order and deduping repeats; ignores null/empty so a
   * call with no model param never stores a blank entry. The common case (a
   * model already known) costs one set lookup and no round-trip.
   *
   * COMMIT-THEN-MARK ordering, same as the SQLite store: the in-memory mirror is
   * updated only AFTER the UPDATE commits, so a write that exhausts its retry
   * budget leaves nothing marked and the next call carrying that model tries
   * again.
   */
  async noteModel(model: string | null | undefined): Promise<void> {
    if (!model || this.modelsSeen.has(model)) return;
    const next = [...this.models, model];
    await this.withWriteRetry(async () => {
      await this.conn.query("BEGIN");
      try {
        await this.conn.query(UPDATE_MODELS, [JSON.stringify(next), this.runId]);
        await this.conn.query("COMMIT");
      } catch (err) {
        await this.rollbackQuietly();
        throw err;
      }
      this.models = next;
      this.modelsSeen.add(model);
    });
  }

  // --- reading ---------------------------------------------------------------

  /**
   * Every session in this database, OLDEST FIRST, each summarized with its
   * models, the distinct agents seen on its calls (first-appearance order), and
   * its turn count. Assembled from three cheap queries rather than one wide join
   * — identical to the SQLite reader, so the two backends' `Session` lists are
   * indistinguishable.
   */
  async listSessions(): Promise<Session[]> {
    const runs = await this.query(SELECT_SESSIONS);

    const counts = new Map<string, number>();
    for (const [runId, n] of await this.query(SELECT_TURN_COUNTS)) {
      counts.set(toText(runId), toInt(n));
    }

    const agents = new Map<string, string[]>();
    for (const [runId, agent] of await this.query(SELECT_AGENTS)) {
      const key = toText(runId);
      const seen = agents.get(key) ?? [];
      const name = toText(agent);
      if (!seen.includes(name)) seen.push(name);
      agents.set(key, seen);
    }

    return runs.map((r) => ({
      id: toText(r[0]),
      project: toText(r[1]),
      startedAt: toText(r[2]),
      provider: toText(r[3]),
      models: JSON.parse(toText(r[4])) as string[],
      agents: agents.get(toText(r[0])) ?? [],
      turnCount: counts.get(toText(r[0])) ?? 0,
    }));
  }

  /**
   * One session's row as a `Run`, decoding the models JSON array. `sessionId`
   * selects a session in a multi-session database; it defaults to the handle's
   * bound session (the newest, for a reader) so the common single-session read
   * needs no argument.
   */
  async getRun(sessionId?: string): Promise<Run> {
    const rows = await this.query(SELECT_RUN, [sessionId ?? this.runId]);
    if (rows.length === 0) {
      throw new Error(`ctxdiff: no such session: ${sessionId ?? this.runId}`);
    }
    const r = rows[0];
    return {
      id: toText(r[0]),
      project: toText(r[1]),
      startedAt: toText(r[2]),
      provider: toText(r[3]),
      models: JSON.parse(toText(r[4])) as string[],
      ctxdiffVersion: toText(r[5]),
    };
  }

  /**
   * All calls for ONE session, ordered by turn sequence. `sessionId` defaults to
   * the bound session, so sessions read in isolation from each other even though
   * they share tables. The attribution columns (agent/step/provider) always
   * exist here — unlike SQLite, which may be reading a physically-v1 file — so
   * there is no column-presence branch.
   */
  async getCalls(sessionId?: string): Promise<Call[]> {
    const rows = await this.query(SELECT_CALLS, [sessionId ?? this.runId]);
    return rows.map((r) => ({
      id: toText(r[0]),
      runId: toText(r[1]),
      seq: toInt(r[2]),
      params: JSON.parse(toText(r[3])) as Record<string, unknown>,
      usage:
        r[4] === null || r[4] === undefined
          ? null
          : (JSON.parse(toText(r[4])) as Record<string, unknown>),
      latencyMs: toIntOrNull(r[5]),
      error: toTextOrNull(r[6]),
      agent: toTextOrNull(r[7]),
      step: toTextOrNull(r[8]),
      provider: toTextOrNull(r[9]),
    }));
  }

  /**
   * Reconstruct one call's blocks in position order by joining the membership
   * table to the deduped block table — rebuilding the same `CallBlock`/`Block`
   * objects the analyzers consume from any backend.
   */
  async getCallBlocks(callId: string): Promise<CallBlock[]> {
    const rows = await this.query(SELECT_CALL_BLOCKS, [callId]);
    return rows.map((r) => ({
      block: {
        contentHash: toText(r[3]),
        role: toText(r[4]),
        kind: toText(r[5]),
        text: toText(r[6]),
        tokenCount: toInt(r[7]),
        tokenMethod: toText(r[8]),
      },
      position: toInt(r[0]),
      label: toText(r[1]),
      labelSource: toText(r[2]),
    }));
  }

  /**
   * Close the connection, best-effort and never rejecting — it runs on the
   * writer's way out and inside the CLI's `finally`, where a rejection would be
   * worse than a leaked socket the OS reclaims anyway. A rollback first discards
   * any transaction left open by a failed write, so a server-side lock is
   * released immediately rather than at timeout. Idempotent.
   */
  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    await this.rollbackQuietly();
    await this.conn.close();
  }
}

/**
 * Base for the networked backends (`PostgresStore`, `MySQLStore`): inert until
 * used, then connect-and-create-if-missing.
 *
 * A subclass supplies the dialect (DDL types + dedup syntax + error
 * classification) and `connect()`. Everything about the lifecycle — creating the
 * tables on first connect, gating on schema version, inserting the session row,
 * binding a reader to the newest session — is shared, so the two adapters behave
 * identically by construction.
 */
export abstract class SQLBackend implements StoreBackend {
  abstract readonly dialect: Dialect;
  /** Seconds any single statement may take. Published (rather than private) so
   * `Tracer.close()` can bound its flush by the store's own bound — see
   * `statementTimeoutOf` in `store/base.ts`. */
  abstract readonly statementTimeout: number;

  /** Open one bounded connection. Implemented per adapter (each driver takes its
   * own connect arguments); must import its driver lazily and throw a clear
   * install hint if the optional peer dependency is missing. */
  protected abstract connect(): Promise<SqlConn>;

  /** The client-side deadline for one statement, in milliseconds — the store's
   * statement bound plus the margin that lets a server-side timeout report
   * itself first. */
  protected deadlineMs(): number {
    return (this.statementTimeout + DEADLINE_MARGIN) * 1000;
  }

  /**
   * Create ctxdiff's four tables if they are not already there — the "no manual
   * migration step" promise. How: runs the dialect's `CREATE TABLE IF NOT
   * EXISTS` statements one per query (neither driver runs multi-statement
   * scripts by default), so running this on every connect — which is what makes
   * auto-create work without a "have I set up yet?" flag — costs one cheap
   * no-op round trip against an existing database.
   *
   * Retried on a CATALOG RACE, which is the one case `IF NOT EXISTS` does not
   * cover. `IF NOT EXISTS` is a check, not a lock: two Postgres backends that
   * pass it in the same instant both insert into `pg_type`/`pg_class`, and the
   * loser dies with a duplicate-key error (SQLSTATE 23505) on a system index —
   * or, at a slightly different interleaving, 42P07 "relation already exists".
   * This is not a rare corner: it is the SHAPE OF A COLD START. A service
   * scaling up points every replica at the same empty database at the same
   * moment, and un-retried, all but one of them record nothing for their entire
   * run — the deployment where capture matters most is the one that has none.
   * Retrying is trivially safe because the winner has by then created the table,
   * so the retry's `IF NOT EXISTS` statements are pure no-ops.
   *
   * Retried on a TRANSIENT failure too, which is a DIFFERENT thing and is why it
   * is a separate condition rather than more errnos in `isSchemaRace`: a race is
   * two clients creating the same object, while MySQL raising 1213 (deadlock
   * found) or 1205 (lock wait timeout) on `CREATE TABLE IF NOT EXISTS` is the
   * server picking a victim among concurrent transactions touching the data
   * dictionary — nothing about the schema is wrong, and the same statement
   * simply works the next time. It is observed on a real MySQL 9.x cold start
   * where several containers create the tables at once, and un-retried it costs
   * that container its ENTIRE run's capture, which is exactly the outcome the
   * catalog-race retry exists to prevent. Re-running the DDL after a deadlock is
   * unambiguously safe because every statement is `IF NOT EXISTS` (idempotent by
   * construction) and the loop is attempt-capped, so a server that keeps failing
   * still re-raises after `SCHEMA_MAX_ATTEMPTS`. A connection that DIED is also
   * transient by this test but cannot be repaired here — the retry re-raises it
   * one attempt later, since reopening is the write path's job (`SQLStore.reopen`)
   * and there is no store yet at schema time.
   *
   * Each attempt rolls back first: on Postgres the failing statement leaves the
   * transaction aborted, where every subsequent statement would fail with 25P02
   * until it is.
   */
  protected async ensureSchema(conn: SqlConn): Promise<void> {
    for (let attempt = 0; ; attempt++) {
      try {
        for (const statement of this.dialect.ddl()) await conn.query(statement);
        return;
      } catch (err) {
        try {
          await conn.query("ROLLBACK");
        } catch {
          /* best-effort between attempts; the real error is re-raised below */
        }
        const retryable = this.dialect.isSchemaRace(err) || this.dialect.isTransient(err);
        if (attempt === SCHEMA_MAX_ATTEMPTS - 1 || !retryable) {
          throw err;
        }
        // KEEP-ALIVE: the session is half-open — the tables are not there yet
        // and no row has been written. The writer's socket is detached and idle
        // for the whole pause, so an unreferenced timer would let the process
        // exit right here, and a replica that lost the cold-start race would
        // record NOTHING for its entire run without a single warning. Bounded to
        // at most two pauses of SCHEMA_RETRY_DELAY_MS.
        await sleep(SCHEMA_RETRY_DELAY_MS, true);
      }
    }
  }

  /**
   * Connect, create the schema if missing, and INSERT one new session row — the
   * write entry point `trace.init()` uses on its first `wrap()` (from the
   * writer, never from the host's call path — see `trace.ts`).
   *
   * `models` starts empty when `model` is falsy: the run's model is really a
   * per-CALL fact that `wrap()` does not know yet, so seeding a placeholder
   * would store a permanent blank; `noteModel()` backfills it from real call
   * params. `startedAt` is required in substance — an empty value is replaced
   * with 'now' so a session always carries the timestamp the UI displays; it is
   * deliberately NOT what sessions are ordered by (see the module docstring).
   *
   * A database already holding rows from a NEWER ctxdiff schema is refused
   * rather than mixed, mirroring the SQLite store's gate. On ANY failure the
   * connection is closed before the error escapes, so a rejected open never
   * leaks a socket.
   */
  async openSession(args: OpenSessionArgs): Promise<SQLStore> {
    const conn = await this.connectDetached();
    let runId: string;
    let models: string[];
    try {
      await this.ensureSchema(conn);
      const existing = await this.maxSchemaVersion(conn);
      if (existing !== null && existing > SCHEMA_VERSION) {
        throw new Error(
          `ctxdiff: store schema version ${existing} is newer than supported ` +
            `${SCHEMA_VERSION} — upgrade ctxdiff to write here`,
        );
      }
      runId = randomUUID().replace(/-/g, "");
      models = args.model ? [args.model] : [];
      await conn.query("BEGIN");
      await conn.query(INSERT_RUN, [
        runId,
        args.project,
        args.startedAt || new Date().toISOString(),
        args.provider,
        JSON.stringify(models),
        VERSION,
        SCHEMA_VERSION,
      ]);
      await conn.query("COMMIT");
    } catch (err) {
      await conn.close();
      throw err;
    }
    // `connectDetached` (not a closure over `conn`) is handed over as the
    // reconnect callable, so a connection killed mid-run is replaced by a fresh
    // one carrying the same timeouts AND the same detachment — see
    // `SQLStore.reopen`.
    return new SQLStore(conn, this.dialect, runId, models, () => this.connectDetached());
  }

  /**
   * Open a connection that will never keep the host process alive: the ordinary
   * `connect()` plus `SqlConn.unref()`.
   *
   * Used for every WRITER connection, because a writer runs in the background
   * and nobody is awaiting it — so its socket must not be the reason a finished
   * program hangs (see `SqlConn.unref` for the full argument). Readers
   * deliberately do NOT use this: the caller is blocked on the read, so its
   * socket SHOULD hold the loop open until the answer arrives.
   */
  private async connectDetached(): Promise<SqlConn> {
    const conn = await this.connect();
    conn.unref();
    return conn;
  }

  /**
   * Connect and bind a handle to the NEWEST session in this database — what the
   * CLI/viewer read when the user names no session. Newest means the greatest
   * `insert_order`, i.e. the session written LAST, which is the SQLite reader's
   * `ORDER BY rowid DESC` and is immune to a writer whose clock is behind. A
   * database with no sessions throws a clear `EmptyStoreError` instead of
   * returning an empty handle that would fail obscurely later. The schema is
   * ensured first so reading a database ctxdiff has never written reports "no
   * sessions", not "no such table".
   *
   * BUT ONLY AS A CONVENIENCE — never at the cost of the read itself. A
   * SELECT-only role is exactly the right credential to hand somebody pointing
   * `ctxdiff diff` at a production database, and it cannot CREATE: making
   * `ensureSchema` a hard prerequisite turned that into "permission denied for
   * schema public" against a database whose ctxdiff tables were sitting right
   * there — with nothing in the message even hinting that ctxdiff had tried to
   * CREATE anything, so the error read as "you cannot read this" rather than
   * "the setup step you do not need was refused".
   *
   * So HERE — and only here, never on the write path, where a missing table is
   * fatal and must say so — a failed `ensureSchema` is remembered rather than
   * raised and the read is attempted anyway. If the read works, the refusal was
   * irrelevant. If it does not, both halves are reported together (see
   * `readerOpenError`): what ctxdiff tried, what the database said to each
   * attempt, and what an operator can do about it.
   */
  async openReader(): Promise<SQLStore> {
    const conn = await this.connect();
    let runId: string;
    try {
      // Deliberately catching ANY failure rather than sniffing for a permission
      // code: classifying the refusal is the database's job, and the read that
      // follows is the only test of whether it actually mattered.
      let schemaError: unknown = null;
      try {
        await this.ensureSchema(conn);
      } catch (err) {
        schemaError = err;
      }
      let rows: unknown[][];
      try {
        rows = await conn.query(SELECT_NEWEST_RUN);
      } catch (err) {
        throw schemaError === null ? err : readerOpenError(schemaError, err);
      }
      if (rows.length === 0) throw new EmptyStoreError();
      runId = toText(rows[0][0]);
      const version = toInt(rows[0][1]);
      if (version > SCHEMA_VERSION) {
        throw new Error(
          `ctxdiff: store schema version ${version} is newer than supported ` +
            `${SCHEMA_VERSION} — upgrade ctxdiff to read here`,
        );
      }
    } catch (err) {
      await conn.close();
      throw err;
    }
    return new SQLStore(conn, this.dialect, runId);
  }

  /** Highest `schema_version` any session in this database was written with, or
   * null when there are none yet. Its own helper so `openSession` reads as a
   * sequence of intentions rather than cursor bookkeeping. */
  private async maxSchemaVersion(conn: SqlConn): Promise<number | null> {
    const rows = await conn.query(SELECT_MAX_SCHEMA);
    const value = rows.length ? rows[0][0] : null;
    return value === null || value === undefined ? null : toInt(value);
  }
}
