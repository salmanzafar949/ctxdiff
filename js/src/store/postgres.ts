/**
 * The PostgreSQL backend — on `pg` (node-postgres), an OPTIONAL peer dependency.
 *
 *     import { configure, PostgresStore } from "ctxdiff";
 *     configure({ store: new PostgresStore({ dsn: "postgresql://user@host/agents" }) });
 *
 * ...or `CTXDIFF_STORE=postgresql://user@host/agents` with no code change at
 * all. Tables are created on first connect if they are not already there; there
 * is no migration step.
 *
 * The driver is imported LAZILY, inside `connect()`, for two reasons: ctxdiff's
 * core must stay dependency-light (one runtime dependency, `gpt-tokenizer`), and
 * a user who has configured Postgres but not installed `pg` must get a clear
 * one-line install hint at connect time — inside the tracer's fail-open guard —
 * rather than a module-resolution crash at `import "ctxdiff"`.
 *
 * The `pg` types are described structurally right here rather than imported from
 * `@types/pg`, so neither the build nor the published `.d.ts` depends on a
 * package a user may not have installed.
 */
import {
  BLOCK_TABLE,
  CALL_BLOCK_TABLE,
  CALL_TABLE,
  DEFAULT_CONNECT_TIMEOUT,
  DEFAULT_STATEMENT_TIMEOUT,
  RUN_TABLE,
  SQLBackend,
  DetachableSocket,
  withDeadline,
  type Dialect,
  type SqlConn,
} from "./sql.js";

// SQLSTATE classes worth one bounded retry rather than dropping the call:
// serialization failure and deadlock (two agents' writers racing on the same
// block rows), lock-not-available, and the whole 08xxx connection-exception
// class (a connection recycled by a pooler/proxy mid-write).
const TRANSIENT_SQLSTATES = new Set(["40001", "40P01", "55P03"]);

// SQLSTATEs meaning THIS CONNECTION is gone and the retry must reopen before it
// can mean anything: the 08xxx connection-exception class, plus the 57Pxx
// operator-intervention codes a kill produces — `pg_terminate_backend` reports
// 57P01 (admin_shutdown), a crash-restart 57P02, an idle-session timeout 57P03.
const CONNECTION_LOST_SQLSTATES = new Set(["57P01", "57P02", "57P03"]);

// Node-level socket errors `pg` surfaces with an errno-style `code` instead of a
// SQLSTATE — what a killed/partitioned peer actually produces client-side.
const CONNECTION_LOST_CODES = new Set([
  "ECONNRESET",
  "EPIPE",
  "ETIMEDOUT",
  "ECONNREFUSED",
  "EHOSTUNREACH",
  "ENETUNREACH",
  "ENOTFOUND",
]);

// SQLSTATEs raised when two backends create the same table at the same instant.
// `CREATE TABLE IF NOT EXISTS` checks the catalog without locking it, so both
// proceed and the loser fails on a system index — as a duplicate key (23505,
// e.g. `pg_type_typname_nsp_index` for the table's row type, or
// `pg_class_relname_nsp_index` for a serial column's sequence) or, at a slightly
// different interleaving, as an outright duplicate_table/duplicate_object.
const SCHEMA_RACE_SQLSTATES = new Set(["23505", "42P07", "42710"]);

// TCP keepalive probing for a connection sitting IDLE between an agent's calls.
// Chosen so a silently-dead peer is detected in well under a minute rather than
// the OS default of two hours. Deliberately more aggressive than a
// general-purpose pool would be: this connection is a debugging tool's, and
// noticing it is dead is worth far more than the handful of packets.
const KEEPALIVE_INITIAL_DELAY_MS = 10_000;

/** The two `pg` shapes this module uses, described structurally so `@types/pg`
 * is never required to build or consume ctxdiff. */
interface PgQueryConfig {
  text: string;
  values?: unknown[];
  rowMode?: "array";
}
interface PgClient {
  connect(): Promise<void>;
  query(config: PgQueryConfig): Promise<{ rows: unknown[][] }>;
  end(): Promise<void>;
  on(event: "error" | "end", handler: (err?: unknown) => void): void;
  connection?: { stream?: { destroy(): void; destroyed?: boolean; unref?(): void } };
}
interface PgModule {
  Client: new (config: Record<string, unknown>) => PgClient;
}

/** The driver's error `code`, which is a SQLSTATE for a server error and an
 * errno string (`ECONNRESET`) for a socket failure. */
function codeOf(err: unknown): string | null {
  const code = (err as { code?: unknown } | null)?.code;
  return typeof code === "string" ? code : null;
}

// Failures `pg` raises ITSELF, with no SQLSTATE and no errno to classify them
// by, all of which mean this connection is finished:
//
// - "Client has encountered a connection error..." / "Connection terminated" —
//   what every statement after a killed backend rejects with (the driver
//   attaches a code exactly ONCE, to the first one);
// - "Query read timeout" — `query_timeout`, the driver's own client-side timer.
//   It is the nastiest of the three because it leaves the socket UP and the
//   client still executing that query, so the connection looks healthy while
//   being unusable: the next statement rejects with "already executing a query"
//   after burning its whole deadline. Classifying it as connection loss is what
//   makes the write retry REOPEN instead of hammering a busy client;
// - "timeout expired" — the connect timeout, which destroys the stream itself.
const BARE_DRIVER_FAILURES =
  /connection (terminated|error)|client has encountered a connection error|server closed the connection|query read timeout|timeout expired/i;

/** True when `err` is one of the driver's own code-less failures (see
 * `BARE_DRIVER_FAILURES`) — the only way to recognize them is the message. */
function isBareDriverFailure(err: unknown): boolean {
  const message = (err as { message?: unknown } | null)?.message;
  return typeof message === "string" && BARE_DRIVER_FAILURES.test(message);
}

/** True when `err` came from `pg`'s OWN client-side query timer, which — unlike
 * every other failure here — leaves the socket connected and the client BUSY.
 * Its own predicate because the connection must be destroyed when it fires, not
 * merely classified: nothing else will ever free that client. */
function isQueryReadTimeout(err: unknown): boolean {
  const message = (err as { message?: unknown } | null)?.message;
  return typeof message === "string" && /query read timeout/i.test(message);
}

/**
 * Everything about Postgres that the shared SQL layer cannot assume.
 *
 * Types: Postgres' `TEXT` is unbounded AND indexable, so the DDL is a nearly
 * literal transcription of the `.ctrace` schema — no VARCHAR lengths to invent,
 * which is the single biggest structural difference from MySQL. `latency_ms` is
 * the one deliberate widening: `INTEGER` milliseconds are a 32-bit value that
 * overflows after ~24 days, and an overflow REJECTS the whole call rather than
 * storing an odd number.
 *
 * Ordering: `insert_order` is a `BIGSERIAL` — the portable stand-in for SQLite's
 * implicit `rowid`, giving sessions a monotonic write order independent of the
 * `started_at` clock. It is the ordering key for `listSessions()` and for
 * binding a reader to the newest session, so containers whose clocks disagree
 * (the reason to share one database in the first place) still agree on which run
 * is the latest.
 *
 * Dedup: `ON CONFLICT (content_hash) DO NOTHING` — the standard-flavoured
 * upsert. It names the conflicting column explicitly, so it can only ever no-op
 * on a duplicate CONTENT HASH; any other constraint violation still raises,
 * which is what content-addressed dedup should mean.
 */
export const PostgresDialect: Dialect = {
  name: "postgres",

  /** The `CREATE TABLE IF NOT EXISTS` statements, in dependency order (parents
   * before the tables whose foreign keys reference them), one per statement
   * because `pg` does not run multi-statement scripts with parameters. Re-running
   * them against an existing database is a no-op, which is what makes
   * auto-create safe on every connect. */
  ddl(): string[] {
    return [
      `CREATE TABLE IF NOT EXISTS ${RUN_TABLE} (
  id              TEXT PRIMARY KEY,
  insert_order    BIGSERIAL NOT NULL UNIQUE,
  project         TEXT NOT NULL,
  started_at      TEXT NOT NULL,
  provider        TEXT NOT NULL,
  models          TEXT NOT NULL,
  ctxdiff_version TEXT NOT NULL,
  schema_version  INTEGER NOT NULL
)`,
      `CREATE TABLE IF NOT EXISTS ${CALL_TABLE} (
  id          TEXT PRIMARY KEY,
  run_id      TEXT NOT NULL REFERENCES ${RUN_TABLE}(id),
  seq         INTEGER NOT NULL,
  params      TEXT NOT NULL,
  usage_json  TEXT,
  latency_ms  BIGINT,
  error       TEXT,
  agent       TEXT,
  step        TEXT,
  provider    TEXT,
  UNIQUE (run_id, seq)
)`,
      `CREATE TABLE IF NOT EXISTS ${BLOCK_TABLE} (
  content_hash TEXT PRIMARY KEY,
  role         TEXT NOT NULL,
  kind         TEXT NOT NULL,
  body         TEXT NOT NULL,
  token_count  INTEGER NOT NULL,
  token_method TEXT NOT NULL
)`,
      `CREATE TABLE IF NOT EXISTS ${CALL_BLOCK_TABLE} (
  call_id      TEXT NOT NULL REFERENCES ${CALL_TABLE}(id),
  block_id     TEXT NOT NULL REFERENCES ${BLOCK_TABLE}(content_hash),
  pos          INTEGER NOT NULL,
  label        TEXT NOT NULL,
  label_source TEXT NOT NULL,
  PRIMARY KEY (call_id, pos)
)`,
    ];
  },

  /** Insert a block, or do nothing if this content hash is already stored — the
   * content-addressed dedup primitive, in Postgres syntax. */
  insertBlockDedup(): string {
    return (
      `INSERT INTO ${BLOCK_TABLE} ` +
      "(content_hash, role, kind, body, token_count, token_method) " +
      "VALUES (?, ?, ?, ?, ?, ?) " +
      "ON CONFLICT (content_hash) DO NOTHING"
    );
  },

  /** True when a failed write is worth ONE bounded retry. How: reads the
   * SQLSTATE `pg` attaches to every server error and matches it against the
   * serialization/deadlock/lock codes plus the whole `08xxx` connection-exception
   * class. Anything without one (a bug in ctxdiff, a bad value) is NOT transient
   * and re-raises immediately — retrying it would only delay the fail-open
   * drop. */
  isTransient(err: unknown): boolean {
    const code = codeOf(err);
    if (code === null) return false;
    return TRANSIENT_SQLSTATES.has(code) || code.startsWith("08");
  },

  /** True when the error means the CONNECTION died, not just the statement — the
   * case a retry can only survive by reopening first. How: matches the 08xxx
   * connection-exception class and the 57Pxx operator-intervention codes
   * (`pg_terminate_backend`, a crash restart, an idle-session timeout), plus the
   * socket-level errnos `pg` reports when the peer vanishes. The driver reports a
   * kill this way exactly ONCE; every later statement rejects with a plain
   * "Client has encountered a connection error" carrying no code at all, which is
   * why the caller ALSO consults the connection's own liveness flag. */
  isConnectionLost(err: unknown): boolean {
    const code = codeOf(err);
    if (code !== null) {
      if (CONNECTION_LOST_SQLSTATES.has(code) || code.startsWith("08")) return true;
      if (CONNECTION_LOST_CODES.has(code)) return true;
    }
    // `pg` rejects post-mortem statements with this exact message and no code —
    // and its OWN timers (`query_timeout`, the connect timeout) reject with a
    // bare Error carrying no code either. A read timeout in particular leaves
    // the client still executing that query, so the connection is finished as
    // far as ctxdiff is concerned even though the socket is still up: treating
    // it as anything less means the retry lands on a busy client.
    return isBareDriverFailure(err);
  },

  /** True when the auto-create DDL lost a race to another process creating the
   * SAME table (see `SCHEMA_RACE_SQLSTATES`), which the caller retries because
   * the winner has by then made the retry a no-op. Deliberately narrow: an
   * ordinary 23505 on a ctxdiff table is a real constraint violation and must not
   * be retried, which is why this is consulted ONLY from `ensureSchema` and never
   * from the write path. */
  isSchemaRace(err: unknown): boolean {
    const code = codeOf(err);
    return code !== null && SCHEMA_RACE_SQLSTATES.has(code);
  },
};

/** Rewrite the shared statements' `?` placeholders into Postgres' `$1..$n`.
 * ctxdiff's SQL contains no string literals and no JSON operators, so a
 * positional scan is exact — and keeping the rewrite here means the shared layer
 * holds ONE statement text for both dialects (see `store/sql.ts`). */
export function toDollarPlaceholders(sql: string): string {
  let n = 0;
  return sql.replace(/\?/g, () => `$${++n}`);
}

/** A `pg.Client` wrapped as the shared layer's `SqlConn`: placeholder rewriting,
 * array rows, a hard client-side deadline per statement, and a liveness flag the
 * retry path can trust. */
class PgConn implements SqlConn {
  private readonly client: PgClient;
  private readonly deadline: number;
  // Referenced while a statement is in flight, released when idle — see
  // `DetachableSocket` and `SqlConn.unref`.
  private readonly socket: DetachableSocket;
  private dead = false;

  constructor(client: PgClient, deadlineMs: number) {
    this.client = client;
    this.deadline = deadlineMs;
    this.socket = new DetachableSocket(() => client.connection?.stream);
    // A `pg` Client emits 'error' asynchronously when the backend goes away; an
    // unhandled one would crash the HOST process, which is the opposite of
    // fail-open. Latching `dead` here is also what makes the reopen path work
    // after the driver stops attaching SQLSTATEs to its rejections.
    client.on("error", () => {
      this.dead = true;
    });
    client.on("end", () => {
      this.dead = true;
    });
  }

  /**
   * Run one statement, and DESTROY this connection if `pg`'s own read timer is
   * what ended it.
   *
   * Why the extra branch: `query_timeout` is enforced by a plain `setTimeout` in
   * the driver, which rejects with a bare `Error: Query read timeout` and then
   * walks away — it does not cancel the query, does not touch the socket, and
   * does not clear the client's active query. So the connection is left BUSY and
   * looking perfectly healthy: `isDead()` stays false, the shared layer's
   * rollback and retry both land on that busy client ("Calling client.query()
   * when the client is already executing a query"), and each one burns its full
   * deadline before failing. The call is dropped against a database that is
   * fine.
   *
   * Destroying is the honest response, and the same one `withDeadline` already
   * takes on expiry: there is no client-side way to reclaim that client, so the
   * connection is finished and the next attempt must reopen. Every other failure
   * passes through untouched — a rejected statement is not a dead connection.
   */
  async query(sql: string, params: unknown[] = []): Promise<unknown[][]> {
    const work = this.client.query({
      text: toDollarPlaceholders(sql),
      values: params,
      rowMode: "array",
    });
    try {
      const result = await this.socket.during(
        withDeadline(work, this.deadline, "postgres query", () => this.destroy()),
      );
      return result.rows;
    } catch (err) {
      if (isQueryReadTimeout(err)) this.destroy();
      throw err;
    }
  }

  isDead(): boolean {
    return this.dead || this.client.connection?.stream?.destroyed === true;
  }

  /** Release the socket whenever this connection is idle — see `SqlConn.unref`
   * and `DetachableSocket`. */
  unref(): void {
    this.socket.detach();
  }

  /** Rip the socket down without waiting for a protocol-level goodbye — used
   * when a deadline fires, precisely because the server is not answering and a
   * graceful `end()` would wait for a reply that is never coming. */
  private destroy(): void {
    this.dead = true;
    this.client.connection?.stream?.destroy();
  }

  async close(): Promise<void> {
    this.dead = true;
    try {
      // Bounded: `end()` waits for the server to acknowledge, which a wedged
      // server never will — and close() must not be the thing that hangs a
      // shutdown. Falls back to destroying the socket outright.
      await this.socket.during(
        withDeadline(this.client.end(), this.deadline, "postgres close", () =>
          this.client.connection?.stream?.destroy(),
        ),
      );
    } catch {
      /* closing is best-effort on the way out */
    }
  }
}

/** Options for `PostgresStore`. */
export interface PostgresStoreOptions {
  /** libpq-style DSN, e.g. `postgresql://user:pw@host:5432/db`. */
  dsn: string;
  /** Seconds for the TCP connect/handshake (default 5). */
  connectTimeout?: number;
  /** Seconds any single statement may run (default 10), applied both
   * server-side and client-side. */
  statementTimeout?: number;
}

/**
 * Points ctxdiff at a PostgreSQL database. Connection-less until used: this
 * constructor validates nothing and opens nothing, so
 * `configure({ store: new PostgresStore({ dsn }) })` at module import is free
 * and a dead database surfaces inside the tracer's fail-open guard.
 */
export class PostgresStore extends SQLBackend {
  readonly dialect = PostgresDialect;
  readonly dsn: string;
  readonly connectTimeout: number;
  readonly statementTimeout: number;

  /**
   * Record where to connect and the two bounds that keep a slow/dead database
   * from ever delaying the host: `connectTimeout` seconds for the TCP
   * connect/handshake, and `statementTimeout` seconds for any single statement
   * once connected. Both default small (see `store/sql.ts`) because capture must
   * degrade quickly rather than stall an agent.
   */
  constructor(opts: PostgresStoreOptions) {
    super();
    this.dsn = opts.dsn;
    this.connectTimeout = opts.connectTimeout ?? DEFAULT_CONNECT_TIMEOUT;
    this.statementTimeout = opts.statementTimeout ?? DEFAULT_STATEMENT_TIMEOUT;
  }

  /**
   * Open one bounded `pg` connection.
   *
   * FOUR bounds, because in Node no single one of them is enough:
   * - `connectionTimeoutMillis` — the TCP connect/handshake, so an unreachable
   *   host fails in seconds rather than at the OS default;
   * - `statement_timeout` — SERVER-side, so a lock held by another session can
   *   never pin the writer (and therefore `close()`) indefinitely;
   * - `query_timeout` — CLIENT-side, and the one that covers the case the
   *   server-side knob structurally cannot: a box that completes the handshake
   *   and then stops answering, where the packets carrying the server's own
   *   timeout verdict are the packets being dropped. This is Node's equivalent
   *   of the `tcp_user_timeout` the Python adapter sets;
   * - `keepAlive` — TCP keepalive probing for a connection sitting IDLE between
   *   an agent's calls, so a silently-dead peer is noticed rather than
   *   discovered at the next write.
   *
   * On top of those, every statement is ALSO raced against `withDeadline` (see
   * `PgConn`), which destroys the socket if the driver's own timer somehow does
   * not fire — the guarantee that `close()` can never wait forever.
   *
   * Autocommit is left as `pg`'s default (implicit per statement) and
   * transactions are explicit `BEGIN`/`COMMIT`, so `recordCall`'s multi-statement
   * write is one real transaction.
   */
  protected async connect(): Promise<SqlConn> {
    const pg = await loadPg();
    const client = new pg.Client({
      connectionString: this.dsn,
      connectionTimeoutMillis: this.connectTimeout * 1000,
      statement_timeout: this.statementTimeout * 1000,
      query_timeout: this.statementTimeout * 1000,
      keepAlive: true,
      keepAliveInitialDelayMillis: KEEPALIVE_INITIAL_DELAY_MS,
      application_name: "ctxdiff",
    });
    const conn = new PgConn(client, this.deadlineMs());
    try {
      // Bounded a second time: `connectionTimeoutMillis` is the driver's own
      // promise, and a blackholed peer must not be able to outlive it.
      await withDeadline(
        client.connect(),
        (this.connectTimeout + 1) * 1000,
        "postgres connect",
        () => client.connection?.stream?.destroy(),
      );
    } catch (err) {
      await conn.close();
      throw err;
    }
    return conn;
  }
}

/**
 * Import `pg` lazily, turning a missing optional peer dependency into an
 * actionable one-line install hint instead of a module-resolution crash at
 * `import "ctxdiff"`. The error surfaces at CONNECT time, which is inside the
 * tracer's fail-open guard, so a user who configured Postgres without installing
 * the driver gets a warning and an untouched host program.
 */
async function loadPg(): Promise<PgModule> {
  try {
    const mod = (await import("pg")) as unknown as { default?: PgModule } & PgModule;
    return mod.default ?? mod;
  } catch (err) {
    throw new Error(
      "ctxdiff: PostgresStore needs the 'pg' driver — install it with " +
        "`npm install pg`",
      { cause: err },
    );
  }
}
