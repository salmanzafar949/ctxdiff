/**
 * Read/write access to a `.ctrace` file — a plain SQLite database that is a
 * PROJECT db holding one OR MANY sessions (each session is one `run` row), their
 * calls, and content-addressed blocks. Writers dedup blocks by hash; readers
 * reconstruct ordered CallBlocks and can list/select sessions. No analysis lives
 * here.
 *
 * Project-scoped model (parity with Python 2311181): one file per PROJECT,
 * appended to over time. Each `init()` opens the project db (creating it if
 * absent) and inserts a NEW `run` row — one session. A single-session file (one
 * `run` row) is the degenerate case and reads identically to before.
 *
 * Storage is `node:sqlite`'s built-in `DatabaseSync` (zero runtime deps). The
 * file format is byte-compatible with the Python SDK's `CTrace`: same DDL, same
 * SCHEMA_VERSION, same JSON column encodings, same dedup + per-run model roll-up
 * semantics — so a project file written by either SDK reads in the other.
 * `startedAt` is passed in by the caller (the tracer) rather than read from the
 * clock here, keeping the store a pure I/O layer.
 *
 * Multi-writer knobs. Multiple Tracers in one process AND multiple Node
 * processes may write the same project db concurrently. WAL (set at create/open
 * time) lets readers run while one writer commits; `busy_timeout` makes SQLite
 * BLOCK an incoming writer for up to this long waiting for the lock instead of
 * failing instantly; the bounded retry loop below is the last line of defence on
 * top of that, so brief contention degrades fail-open (the tracer drops + warns)
 * rather than raising into the host or hanging. Unlike the Python port there is
 * NO background writer thread — Node is single-threaded and `node:sqlite` writes
 * stay synchronous — but the cross-process lock contention is real, hence the
 * busy_timeout-first + retry.
 *
 * THE RETRY BUDGET IS DELIBERATELY SMALL — BECAUSE IT RUNS ON THE MAIN THREAD.
 * `node:sqlite` blocks inside busy_timeout synchronously and the backoff sleep
 * is `Atomics.wait`, so `busy_timeout x attempts + backoff` is time Node's ONE
 * event loop is FROZEN: no timers, no I/O, no incoming requests. Python can
 * afford a generous budget because it offloads writes to a dedicated writer
 * thread; here a 5000ms timeout x 6 attempts measured a 32-second freeze on a
 * single contended `wrap()`, which for a server is worse than the exception the
 * budget exists to avoid. 250ms x 3 caps the worst case near a second, and a
 * 32-racing-process append test never needed more than ONE attempt — so the
 * headroom that matters in practice is fully preserved.
 */
import { DatabaseSync } from "node:sqlite";
import { randomUUID } from "node:crypto";
import type { Block, Call, CallBlock, Run, Session } from "../models.js";
import { DDL, SCHEMA_VERSION } from "./schema.js";
import { VERSION } from "../version.js";

const BUSY_TIMEOUT_MS = 250; // kept small: this BLOCKS the Node event loop
const WRITE_MAX_ATTEMPTS = 3; // total tries before giving up (fail-open upstream)
const WRITE_BACKOFF_START_MS = 50; // doubles each retry, capped by _MAX
const WRITE_BACKOFF_MAX_MS = 500;

/** The v2 attribution columns on `call`. A file physically written by a v1
 * ctxdiff lacks them; appending a v2 session ADDs them (see `upgradeCallTable`). */
const V2_CALL_COLUMNS = ["agent", "step", "provider"] as const;

/** A uuid4 hex (32 chars, no dashes) — matches Python's `uuid.uuid4().hex`. */
function uuidHex(): string {
  return randomUUID().replace(/-/g, "");
}

/**
 * Block the current thread for `ms` milliseconds SYNCHRONOUSLY. `node:sqlite`
 * writes are synchronous, so a retry backoff can't `await`; `Atomics.wait` on a
 * throwaway SharedArrayBuffer is the standard way to sleep the main thread
 * without spinning. Used only on the rare contended-lock retry path.
 */
function sleepSync(ms: number): void {
  if (ms <= 0) return;
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
}

/**
 * Is `err` SQLite reporting the database is locked/busy? node:sqlite surfaces a
 * contended lock as an error whose message contains "database is locked" (or
 * "busy"); we also check the numeric SQLite codes (SQLITE_BUSY=5,
 * SQLITE_LOCKED=6) defensively. Only these get retried — every other error
 * propagates immediately, unretried.
 */
function isLockedError(err: unknown): boolean {
  if (err == null || typeof err !== "object") return false;
  const e = err as { message?: string; errstr?: string; errcode?: number };
  const msg = `${e.message ?? ""} ${e.errstr ?? ""}`.toLowerCase();
  if (msg.includes("locked") || msg.includes("busy")) return true;
  return e.errcode === 5 || e.errcode === 6;
}

/**
 * A stored `started_at` split into its date-time part and its zone designator.
 * The zone is recognized only AFTER a time: the `-04` ending a bare date like
 * `2026-07-04` is a day, not an offset.
 */
const ZONED = /^(\d{4}-?\d{2}-?\d{2}[T ][\d:.]+)([Zz]|[+-]\d{2}:?\d{2}|[+-]\d{2})$/;

/** The date-time part in EXTENDED form (`YYYY-MM-DDTHH:MM:SS`), whether it was
 * written that way or in the separator-free ISO BASIC form (`20260704T100000`).
 * Anything that is not a recognizable ISO date-time is returned untouched, so
 * `Date` still gets its say — and still rejects it. */
function expandBasicIso(text: string): string {
  const m = /^(\d{4})-?(\d{2})-?(\d{2})(?:([T ])(\d{2}):?(\d{2})(?::?(\d{2})(\.\d+)?)?)?$/
    .exec(text);
  if (!m) return text;
  const [, y, mo, d, sep, hh, mi, ss, frac] = m;
  if (!sep) return `${y}-${mo}-${d}`;
  return `${y}-${mo}-${d}T${hh}:${mi}:${ss ?? "00"}${frac ?? ""}`;
}

/**
 * Parse a session's stored `started_at` into a Date, tolerant of every format a
 * file may carry: the canonical UTC string new sessions write (a trailing `Z`),
 * an offset string, and a legacy naive string (no zone). A naive value is
 * assumed to be UTC — the store always writes UTC — so downstream local-time
 * rendering is unambiguous regardless of which format produced the row. Throws
 * on a value that can't be parsed.
 *
 * Acceptance deliberately matches Python's `parse_started_at`, which is
 * `datetime.fromisoformat` and therefore takes two spellings `Date` alone does
 * not: an HOUR-ONLY offset (`+05`) and the ISO BASIC form (`20260704T100000Z`).
 * Neither is anything ctxdiff writes, but a foreign or hand-edited database can
 * hold them — and a row one CLI renders as a local time while the other echoes
 * it raw is the kind of silent disagreement the two SDKs exist to prevent. So
 * the zone is peeled off first, normalized to the `±HH:MM`/`Z` spelling `Date`
 * understands (a missing zone meaning UTC), and the date-time part expanded to
 * extended form.
 */
export function parseStartedAt(value: string): Date {
  const text = value.trim();
  const m = ZONED.exec(text);
  const body = expandBasicIso(m ? m[1] : text);
  let zone = "Z";
  if (m) {
    const raw = m[2];
    if (/^[Zz]$/.test(raw)) zone = "Z";
    else if (raw.length === 3) zone = `${raw}:00`; // +05  -> +05:00
    else if (raw.length === 5) zone = `${raw.slice(0, 3)}:${raw.slice(3)}`; // +0530
    else zone = raw; // already +HH:MM
  }
  const d = new Date(body + zone);
  if (Number.isNaN(d.getTime())) {
    throw new Error(`invalid started_at: ${JSON.stringify(value)}`);
  }
  return d;
}

export class CTrace {
  private db: DatabaseSync;
  private runId: string;
  private hasV2Cols: boolean;
  private models: string[];
  private modelsSeen: Set<string>;
  // One-way latch flipped by close(), making close idempotent — see close().
  private closed = false;

  /**
   * Wrap an already-open, already-initialized database bound to ONE session.
   * Snapshots the `call` table's column set ONCE (via PRAGMA table_info) so
   * reads/writes can adapt to a v1 file (which lacks the agent/step/provider
   * columns) without a PRAGMA per query, and seeds an in-memory mirror of the
   * run's `models` column so `noteModel` gates on a cheap set lookup instead of a
   * SELECT per call. Construct via `create()`/`openOrCreateSession()`/`open()` —
   * never directly.
   */
  private constructor(db: DatabaseSync, runId: string) {
    this.db = db;
    this.runId = runId;
    const cols = new Set(
      (db.prepare("PRAGMA table_info(call)").all() as { name: string }[]).map(
        (r) => r.name,
      ),
    );
    this.hasV2Cols = V2_CALL_COLUMNS.every((c) => cols.has(c));
    const row = db
      .prepare("SELECT models FROM run WHERE id = ?")
      .get(runId) as { models: string } | undefined;
    this.models = row ? (JSON.parse(row.models) as string[]) : [];
    this.modelsSeen = new Set(this.models);
  }

  // --- construction --------------------------------------------------------

  /**
   * Run a write `fn` (a zero-arg thunk wrapping one transaction), retrying a
   * bounded number of times when SQLite reports the database is locked. Why on
   * top of `busy_timeout`: busy_timeout already blocks the writer while another
   * connection holds the lock, but under heavy multi-writer contention SQLite can
   * still surface "database is locked" (e.g. a WAL write-write conflict); a few
   * short exponential-backoff retries clear those transients. Bounded on both
   * axes — a fixed attempt cap and a capped per-sleep — so a genuinely stuck lock
   * gives up (re-raising for the caller's fail-open guard to drop+warn) rather
   * than hanging. Non-lock errors propagate immediately, unretried. Because each
   * `fn` is a self-contained transaction that rolls back on failure, a retry
   * re-runs it cleanly with no partial-write risk. Mirrors Python
   * `_with_write_retry`.
   */
  private static withWriteRetry<T>(fn: () => T): T {
    let delay = WRITE_BACKOFF_START_MS;
    let lastErr: unknown;
    for (let attempt = 0; attempt < WRITE_MAX_ATTEMPTS; attempt++) {
      try {
        return fn();
      } catch (err) {
        lastErr = err;
        if (!isLockedError(err) || attempt === WRITE_MAX_ATTEMPTS - 1) throw err;
        sleepSync(delay);
        delay = Math.min(delay * 2, WRITE_BACKOFF_MAX_MS);
      }
    }
    throw lastErr; // unreachable — the loop either returns or throws
  }

  /**
   * Configure `db` for safe multi-writer access and insert ONE fresh session
   * (`run`) row, returning its id. The single code path both creates a brand-new
   * project file and APPENDS to an existing one, because the DDL is all
   * `CREATE TABLE IF NOT EXISTS` (a harmless no-op on a populated db).
   *
   * Ordering is the crux of the concurrent-creation fix: `busy_timeout` is set
   * FIRST, before ANY lock-taking statement. `PRAGMA journal_mode = WAL` and the
   * `IF NOT EXISTS` DDL both acquire a write lock, and when several processes
   * race to create the SAME project file at once, whichever lose the lock would
   * otherwise fail INSTANTLY with "database is locked" — straight out into the
   * host. Setting busy_timeout ahead of them makes those statements block-and-
   * wait for the lock instead. The ENTIRE setup (WAL enable + DDL + schema gate +
   * session INSERT) then runs under `withWriteRetry`, because busy_timeout alone
   * doesn't cover every transient. Made safe to re-run: a leading rollback clears
   * any partial state from a prior locked attempt, WAL is idempotent, the DDL is
   * IF NOT EXISTS, the version gate is a pure read, and the INSERT is wrapped in
   * its own transaction so a failed attempt leaves no half-written run row.
   *
   * The one thing `IF NOT EXISTS` canNOT do is widen an existing table, so a
   * physically-v1 file (a `call` table with no agent/step/provider) gets an
   * explicit in-place column upgrade first — see `upgradeCallTable`.
   */
  private static setupSession(
    db: DatabaseSync,
    path: string,
    project: string,
    provider: string,
    model: string,
    startedAt: string,
  ): string {
    db.exec(`PRAGMA busy_timeout = ${BUSY_TIMEOUT_MS}`);
    const runId = uuidHex();
    const models = model ? [model] : [];

    CTrace.withWriteRetry(() => {
      // Drop any partial state left by a prior locked attempt. On the first
      // attempt there is no active transaction, so the rollback is a harmless
      // no-op we swallow.
      try {
        db.exec("ROLLBACK");
      } catch {
        /* no active transaction — expected on the first attempt */
      }
      db.exec("PRAGMA journal_mode = WAL");
      db.exec("PRAGMA foreign_keys = ON");
      db.exec(DDL); // IF NOT EXISTS — a no-op against an existing project db
      // ...and because it IS a no-op, an existing PHYSICALLY-v1 `call` table
      // survives the DDL unchanged. Widen it before writing, or this session's
      // agent/step/provider would be silently dropped under a v2 run row.
      CTrace.upgradeCallTable(db);
      // Refuse to append into a file written by a NEWER ctxdiff (mirrors open()).
      const row = db
        .prepare("SELECT MAX(schema_version) AS v FROM run")
        .get() as { v: number | null } | undefined;
      const existing = row?.v ?? null;
      if (existing !== null && existing > SCHEMA_VERSION) {
        throw new Error(
          `${path}: schema version ${existing} is newer than supported ` +
            `${SCHEMA_VERSION} — upgrade ctxdiff to write this file`,
        );
      }
      db.exec("BEGIN");
      try {
        db.prepare("INSERT INTO run VALUES (?,?,?,?,?,?,?)").run(
          runId,
          project,
          startedAt,
          provider,
          JSON.stringify(models),
          VERSION,
          SCHEMA_VERSION,
        );
        db.exec("COMMIT");
      } catch (err) {
        try {
          db.exec("ROLLBACK");
        } catch {
          /* nothing to roll back */
        }
        throw err;
      }
    });
    return runId;
  }

  /**
   * Bring an existing `call` table up to the v2 layout by ADDing whichever of
   * `agent`/`step`/`provider` are missing, on the WRITE path only.
   *
   * Why it's needed: `CREATE TABLE IF NOT EXISTS` no-ops against a table that
   * already exists, so appending a session into a file some older ctxdiff wrote
   * physically as v1 left `hasV2Cols` false — every new call was written in the
   * 7-column v1 shape and lost its attribution, even though the new run row was
   * stamped `schema_version = 2`. Silent data loss, and newly reachable now that
   * the default path is a STABLE `./<project>.ctrace` that can land on a
   * pre-existing user file. Upgrading in place is preferred over downgrading the
   * stamp because the columns are nullable: pre-existing v1 calls simply read
   * back NULL, nothing is rewritten or reinterpreted, and the file stays
   * readable by both SDKs.
   *
   * Read-only `open()` deliberately does NOT call this — a debugger must not
   * rewrite the evidence it inspects; only a writer that is about to append its
   * own v2 rows upgrades the layout.
   */
  private static upgradeCallTable(db: DatabaseSync): void {
    const cols = new Set(
      (db.prepare("PRAGMA table_info(call)").all() as { name: string }[]).map(
        (r) => r.name,
      ),
    );
    for (const col of V2_CALL_COLUMNS) {
      // ADD COLUMN of a nullable TEXT is an O(1) header-only change in SQLite:
      // existing rows are not rewritten, they just read back NULL.
      if (!cols.has(col)) db.exec(`ALTER TABLE call ADD COLUMN ${col} TEXT`);
    }
  }

  /**
   * Start a NEW session in the project db at `path`, creating the file if it
   * doesn't exist yet — the project-scoped write entry point the tracer uses on
   * its first `wrap()`. Opens (or creates) the connection with the full multi-
   * writer write configuration, ensures the schema is present, then inserts ONE
   * fresh `run` row (a new session with its own uuid id and `startedAt`) and
   * returns a CTrace bound to it. Every call this Tracer records lands against
   * that session's run id, so many sessions coexist in one file. Mirrors Python
   * `open_or_create_session`.
   */
  static openOrCreateSession(
    path: string,
    project: string,
    provider: string,
    model: string,
    startedAt = "",
  ): CTrace {
    const db = new DatabaseSync(path);
    try {
      const runId = CTrace.setupSession(db, path, project, provider, model, startedAt);
      return new CTrace(db, runId);
    } catch (err) {
      // Setup failed after the connection opened; close it so we don't leak a
      // file handle/lock on the way out.
      db.close();
      throw err;
    }
  }

  /**
   * Create a session in the `.ctrace` at `path` — an alias for
   * `openOrCreateSession` kept for the direct callers (tests, the Recorder
   * conformance harness) that predate the project-scoped rename. Identical
   * behavior: creates the file if absent, appends a new session if present. Both
   * apply the full multi-writer configuration (busy_timeout-first + WAL + bounded
   * retry). `models` starts as `[model]` when a real model id is passed, else as
   * an EMPTY list: the run's model is really a per-CALL fact (`wrap()` doesn't
   * know it yet at run-creation time), so seeding a placeholder would just store
   * a bogus blank forever. `noteModel()` populates `models` from real call params
   * as they arrive.
   */
  static create(
    path: string,
    project: string,
    provider: string,
    model: string,
    startedAt = "",
  ): CTrace {
    return CTrace.openOrCreateSession(path, project, provider, model, startedAt);
  }

  /**
   * Open an existing project `.ctrace` read/write, accepting ANY schema version
   * this build understands. A v1 file (Python-written, pre-attribution) is read
   * as-is and never migrated — a debugger must not rewrite the evidence it
   * inspects; its missing agent/step/provider columns surface as null. Only a
   * file whose version is NEWER than supported is rejected, with a clear error.
   *
   * A project db may hold MANY sessions; the returned handle is bound to the
   * NEWEST session (last-inserted `run` row) so a session-less read analyzes the
   * run the user JUST made, not the first one ever recorded into this
   * accumulating file. A single-session/v1 file is unaffected — newest == the
   * only row. `busy_timeout` is set FIRST so reads don't fail instantly against a
   * file a live writer is concurrently committing to. Mirrors Python `open`.
   */
  static open(path: string): CTrace {
    const db = new DatabaseSync(path);
    try {
      db.exec(`PRAGMA busy_timeout = ${BUSY_TIMEOUT_MS}`);
      db.exec("PRAGMA foreign_keys = ON");
      const r = db
        .prepare("SELECT id, schema_version FROM run ORDER BY rowid DESC LIMIT 1")
        .get() as { id: string; schema_version: number } | undefined;
      if (r === undefined) {
        throw new Error(`${path}: not a ctrace file (no run row)`);
      }
      if (r.schema_version > SCHEMA_VERSION) {
        throw new Error(
          `${path}: schema version ${r.schema_version} is newer than supported ` +
            `${SCHEMA_VERSION} — upgrade ctxdiff to read this file`,
        );
      }
      if (r.schema_version < 1) {
        throw new Error(
          `${path}: schema version ${r.schema_version} is not a recognized ctrace version`,
        );
      }
      return new CTrace(db, r.id);
    } catch (err) {
      db.close();
      throw err;
    }
  }

  // --- writing -------------------------------------------------------------

  /**
   * Persist one call and its ordered blocks in a single transaction. Each block
   * is upserted by content_hash (stored once, ignored if already present — the
   * dedup mechanism); each membership is written to call_block with its position
   * and label. `agent`/`step`/`provider` are the v2 attribution fields
   * (nullable). Returns the new call id. The transaction is retried on a
   * transient lock under concurrent multi-writer access (see `withWriteRetry`); a
   * rolled-back attempt re-runs cleanly, so no partial write can result. Mirrors
   * Python `record_call`, including the per-call model roll-up onto the run.
   */
  recordCall(args: {
    seq: number;
    params: Record<string, unknown>;
    usage: Record<string, unknown> | null;
    latencyMs: number | null;
    error: string | null;
    callBlocks: CallBlock[];
    agent?: string | null;
    step?: string | null;
    provider?: string | null;
  }): string {
    const callId = uuidHex();
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

    CTrace.withWriteRetry(() => {
      // All-or-nothing transaction, matching Python's `with self._conn:`.
      this.db.exec("BEGIN");
      try {
        if (this.hasV2Cols) {
          this.db
            .prepare("INSERT INTO call VALUES (?,?,?,?,?,?,?,?,?,?)")
            .run(
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
            );
        } else {
          this.db
            .prepare("INSERT INTO call VALUES (?,?,?,?,?,?,?)")
            .run(
              callId,
              this.runId,
              seq,
              JSON.stringify(params),
              usage !== null ? JSON.stringify(usage) : null,
              latencyMs,
              error,
            );
        }
        const insBlock = this.db.prepare(
          "INSERT OR IGNORE INTO block VALUES (?,?,?,?,?,?)",
        );
        const insCallBlock = this.db.prepare(
          "INSERT INTO call_block VALUES (?,?,?,?,?)",
        );
        for (const cb of callBlocks) {
          const b = cb.block;
          // INSERT OR IGNORE: first writer of a hash wins; repeats are no-ops —
          // exactly content-addressed dedup.
          insBlock.run(
            b.contentHash,
            b.role,
            b.kind,
            b.text,
            b.tokenCount,
            b.tokenMethod,
          );
          insCallBlock.run(
            callId,
            b.contentHash,
            cb.position,
            cb.label,
            cb.labelSource,
          );
        }
        this.db.exec("COMMIT");
      } catch (err) {
        try {
          this.db.exec("ROLLBACK");
        } catch {
          /* nothing to roll back */
        }
        throw err;
      }
    });

    // Roll this call's model up onto the run — `model` covers openai/anthropic/
    // gemini; `modelId` covers bedrock's Converse shape.
    // `||`, not `??`: Python spells this `params.get("model") or
    // params.get("modelId")`, so an EMPTY model falls through to the Bedrock
    // spelling rather than being rolled up as a blank. (`noteModel` ignores
    // falsy models either way, so no stored file changes.)
    const model =
      (params["model"] as string | undefined) ||
      (params["modelId"] as string | undefined) ||
      null;
    // Guarded: the call above is ALREADY COMMITTED, so a failure of the
    // best-effort roll-up must not propagate out of recordCall — that would make
    // the caller log "failed to record call" for a call that WAS persisted. The
    // roll-up is self-healing anyway: noteModel only marks a model seen once its
    // UPDATE commits, so the next call carrying the same model retries it.
    try {
      this.noteModel(model);
    } catch {
      /* best-effort model roll-up; retried by the next call with this model */
    }
    return callId;
  }

  /**
   * Append `model` to the run's `models` list the first time it's seen,
   * preserving first-seen order and deduping repeats; ignores null/empty so a
   * call with no model never pollutes the list with a blank entry. The JSON is
   * re-serialized and the run row UPDATEd only on an actual new model, not every
   * call — and that write is retried on a transient lock (multi-writer). Mirrors
   * Python `note_model`.
   *
   * COMMIT-THEN-MARK ordering matters: the in-memory `modelsSeen`/`models`
   * mirror is mutated INSIDE the retried thunk, only AFTER the UPDATE succeeds.
   * Marking the model seen up front (the obvious ordering) meant a write that
   * failed its retry budget left the model permanently "already known", so no
   * later call ever retried it and `run.models` stayed `[]` for the life of the
   * run. Deferring the mutation makes the roll-up self-healing: a failed attempt
   * changes nothing, and the next call carrying that model tries again.
   */
  noteModel(model: string | null | undefined): void {
    if (!model || this.modelsSeen.has(model)) return;
    // Candidate list built without touching the mirror, so a failed (or
    // retried) attempt can never leave a half-applied state behind.
    const next = [...this.models, model];
    CTrace.withWriteRetry(() => {
      this.db
        .prepare("UPDATE run SET models = ? WHERE id = ?")
        .run(JSON.stringify(next), this.runId);
      this.models = next;
      this.modelsSeen.add(model);
    });
  }

  // --- reading -------------------------------------------------------------

  /**
   * List every session (every `run` row) in this project db, OLDEST first, each
   * as a `Session` summary: id, project, startedAt, provider, models, the set of
   * agents seen on its calls (first-appearance order), and its turn (call) count.
   * The reader half of the project-scoped model — a single-session file returns a
   * one-element list. Three cheap queries assembled in JS (rather than one big
   * join) keep turn counts and agent sets easy to reason about; agent sets are
   * gathered only when the v2 `agent` column exists (a v1 file reports `[]`).
   * Mirrors Python `list_sessions`.
   */
  listSessions(): Session[] {
    const runs = this.db
      .prepare(
        "SELECT id, project, started_at, provider, models FROM run ORDER BY rowid",
      )
      .all() as {
      id: string;
      project: string;
      started_at: string;
      provider: string;
      models: string;
    }[];

    const counts = new Map<string, number>();
    for (const r of this.db
      .prepare("SELECT run_id, COUNT(*) AS n FROM call GROUP BY run_id")
      .all() as { run_id: string; n: number }[]) {
      counts.set(r.run_id, r.n);
    }

    const agents = new Map<string, string[]>();
    if (this.hasV2Cols) {
      for (const r of this.db
        .prepare(
          "SELECT run_id, agent FROM call WHERE agent IS NOT NULL ORDER BY run_id, seq",
        )
        .all() as { run_id: string; agent: string }[]) {
        const seen = agents.get(r.run_id) ?? [];
        if (!seen.includes(r.agent)) seen.push(r.agent);
        agents.set(r.run_id, seen);
      }
    }

    return runs.map((r) => ({
      id: r.id,
      project: r.project,
      startedAt: r.started_at,
      provider: r.provider,
      models: JSON.parse(r.models) as string[],
      agents: agents.get(r.id) ?? [],
      turnCount: counts.get(r.id) ?? 0,
    }));
  }

  /**
   * Return one session's `run` row as a Run, decoding the models JSON array.
   * `sessionId` selects which session in a multi-session project db; it defaults
   * to the handle's bound session (the newest — see `open()`), so a single-
   * session file needs no argument and reads exactly as before. Mirrors Python
   * `get_run`.
   */
  getRun(sessionId?: string): Run {
    const runId = sessionId ?? this.runId;
    const r = this.db
      .prepare(
        "SELECT id, project, started_at, provider, models, ctxdiff_version " +
          "FROM run WHERE id = ?",
      )
      .get(runId) as {
      id: string;
      project: string;
      started_at: string;
      provider: string;
      models: string;
      ctxdiff_version: string;
    };
    return {
      id: r.id,
      project: r.project,
      startedAt: r.started_at,
      provider: r.provider,
      models: JSON.parse(r.models) as string[],
      ctxdiffVersion: r.ctxdiff_version,
    };
  }

  /**
   * Return all calls for ONE session ordered by turn sequence. `sessionId`
   * selects the session (defaults to the bound one — the newest), so a project
   * db's sessions read in isolation while a single-session file is unchanged.
   * Selects the v2 attribution columns only when they exist (a v1 file lacks
   * them); for a v1 file those three fields surface as null on every Call.
   * Mirrors Python `get_calls`.
   */
  getCalls(sessionId?: string): Call[] {
    const runId = sessionId ?? this.runId;
    const base = "id, run_id, seq, params, usage, latency_ms, error";
    if (this.hasV2Cols) {
      const rows = this.db
        .prepare(
          `SELECT ${base}, agent, step, provider FROM call WHERE run_id = ? ORDER BY seq`,
        )
        .all(runId) as Record<string, unknown>[];
      return rows.map((r) => this.rowToCall(r, true));
    }
    const rows = this.db
      .prepare(`SELECT ${base} FROM call WHERE run_id = ? ORDER BY seq`)
      .all(runId) as Record<string, unknown>[];
    return rows.map((r) => this.rowToCall(r, false));
  }

  private rowToCall(r: Record<string, unknown>, v2: boolean): Call {
    return {
      id: r.id as string,
      runId: r.run_id as string,
      seq: r.seq as number,
      params: JSON.parse(r.params as string) as Record<string, unknown>,
      usage:
        r.usage !== null && r.usage !== undefined
          ? (JSON.parse(r.usage as string) as Record<string, unknown>)
          : null,
      latencyMs: (r.latency_ms as number | null) ?? null,
      error: (r.error as string | null) ?? null,
      agent: v2 ? ((r.agent as string | null) ?? null) : null,
      step: v2 ? ((r.step as string | null) ?? null) : null,
      provider: v2 ? ((r.provider as string | null) ?? null) : null,
    };
  }

  /**
   * Reconstruct one call's blocks in position order by joining call_block to
   * block. Mirrors Python `get_call_blocks`.
   */
  getCallBlocks(callId: string): CallBlock[] {
    const rows = this.db
      .prepare(
        "SELECT cb.position, cb.label, cb.label_source, " +
          "b.content_hash, b.role, b.kind, b.text, b.token_count, b.token_method " +
          "FROM call_block cb JOIN block b ON b.content_hash = cb.block_id " +
          "WHERE cb.call_id = ? ORDER BY cb.position",
      )
      .all(callId) as {
      position: number;
      label: string;
      label_source: string;
      content_hash: string;
      role: string;
      kind: string;
      text: string;
      token_count: number;
      token_method: string;
    }[];
    return rows.map((r) => {
      const block: Block = {
        contentHash: r.content_hash,
        role: r.role,
        kind: r.kind,
        text: r.text,
        tokenCount: r.token_count,
        tokenMethod: r.token_method,
      };
      return {
        block,
        position: r.position,
        label: r.label,
        labelSource: r.label_source,
      };
    });
  }

  /**
   * Checkpoint the WAL back into the main database file (best-effort), then close
   * the connection. The explicit `wal_checkpoint(TRUNCATE)` guarantees a fresh
   * reader that opens the file right after close — including the Python reader in
   * a cross-language conformance run — sees every committed write immediately
   * (and shrinks the -wal sidecar), rather than relying on SQLite's automatic
   * on-close checkpoint. The checkpoint is swallowed on failure (e.g. WAL not in
   * use) so it never stops the connection from closing. Mirrors Python `close`.
   *
   * IDEMPOTENT. `node:sqlite`'s `db.close()` throws `ERR_INVALID_STATE:
   * database is not open` on a second call, and double-close is an ordinary JS
   * shape (a `finally` block plus a `process.on('exit')` safety net). On a
   * library whose entire contract is fail-open, close is the one public method
   * that must never throw into the host — so a one-way `closed` latch short-
   * circuits repeat calls and the close itself is guarded too. Python is
   * naturally idempotent here (`sqlite3.Connection.close()` is a no-op when
   * already closed), so this restores parity.
   */
  close(): void {
    if (this.closed) return;
    this.closed = true;
    try {
      this.db.exec("PRAGMA wal_checkpoint(TRUNCATE)");
    } catch {
      /* checkpoint is best-effort; always close */
    }
    try {
      this.db.close();
    } catch {
      /* already closed / never opened — closing must never throw */
    }
  }
}
