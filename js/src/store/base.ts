/**
 * The storage contract every ctxdiff backend implements — the seam that lets a
 * user point ctxdiff at their own database instead of a local `.ctrace` file.
 *
 * Two protocols, deliberately small, both derived from what the existing
 * `node:sqlite` store (`CTrace`) already does and what the rest of the codebase
 * already calls:
 *
 * - `Store` — a handle bound to ONE session (one `run` row). The recorder writes
 *   through it; the analyzers/CLI/viewer read through it. Nothing else.
 * - `StoreBackend` — the configurable, connection-less DESCRIPTION of where data
 *   lives (`SQLiteStore({path})`, `PostgresStore({dsn})`, `MySQLStore({dsn})`).
 *   `trace.init()` asks it to open a new session; the CLI asks it for a reader.
 *   A backend holds NO connection until one of those is called, which is what
 *   makes `configure({ store: new PostgresStore(...) })` cheap, side-effect free
 *   and safe to evaluate at import time in a user's module.
 *
 * WHY EVERY METHOD IS `Awaitable<T>` (`T | Promise<T>`) — the one real shape
 * difference from the Python port. Python's DB-API drivers are synchronous, so
 * its `Store` is a plain synchronous protocol and concurrency is bought with a
 * writer THREAD. Node has no such option: `node:sqlite` is synchronous and every
 * network driver (`pg`, `mysql2`) is promise-based, so a single protocol has to
 * admit both. `Awaitable` is what makes that honest rather than a lie in either
 * direction:
 *
 * - the SQLite default stays literally synchronous — same call, same tick, same
 *   `.ctrace` bytes, no promise allocated anywhere on the existing path;
 * - the network stores return promises, and the ONE caller that must handle both
 *   (`trace.ts`'s writer) awaits — `await` on a non-promise is a no-op contract-
 *   wise and never yields control on the SQLite path;
 * - the read side, which has to stay synchronous because every analyzer is
 *   (`analyze/*`, `viewer/export.ts`), consumes `ReadableStore` instead — a
 *   strictly synchronous subset that `CTrace` implements directly and a network
 *   store reaches via `snapshotStore()` (see `snapshot.ts`).
 *
 * The row types (`Run`/`Session`/`Call`) are re-exported here — they live in
 * `models.ts` and stay there so no existing import path changes, but they are
 * the backend-independent shape every store returns, so this module is where a
 * backend author should look for them (mirroring Python, where they were MOVED
 * into `store/base.py` and re-exported from `store/ctrace.py`).
 */
import type { Call, CallBlock, Run, Session } from "../models.js";

export type { Block, Call, CallBlock, Run, Session } from "../models.js";

/** A value a backend may return either directly (SQLite) or as a promise
 * (Postgres/MySQL). See the module docstring for why the protocol is written
 * this way instead of being uniformly async. */
export type Awaitable<T> = T | Promise<T>;

/**
 * A store that exists and is readable but holds NO sessions yet.
 *
 * Its own type so the one caller that must tell "nothing recorded yet" apart
 * from "this store is broken" — `ctxdiff runs`, which prints an empty listing
 * for the former and an error for the latter — can do so without string-matching
 * the message. Mirrors Python's `EmptyStoreError`.
 */
export class EmptyStoreError extends Error {
  constructor(message = "ctxdiff: no sessions recorded in this store") {
    super(message);
    this.name = "EmptyStoreError";
  }
}

/** The arguments of one `Store.recordCall` — one LLM turn plus its ordered
 * blocks. An object rather than a positional list because `CTrace.recordCall`
 * already took one, and keeping it identical is what lets `CTrace` satisfy
 * `Store` with no change at all to the SQLite write path. */
export interface RecordCallArgs {
  seq: number;
  params: Record<string, unknown>;
  usage: Record<string, unknown> | null;
  latencyMs: number | null;
  error: string | null;
  callBlocks: CallBlock[];
  agent?: string | null;
  step?: string | null;
  provider?: string | null;
}

/**
 * A live handle to ONE session in a ctxdiff store — the only surface the
 * recorder and the readers ever touch. Seven methods, each earning its place
 * because something in the codebase already calls it:
 *
 * Write side (the capture path, `capture/recorder.ts`):
 * - `recordCall` — persist one turn plus its ordered blocks atomically, with
 *   content-hash dedup on the blocks and call->block membership carrying
 *   position/label. Returns the new call id.
 * - `noteModel` — roll a per-call model id up onto the session's `models` list
 *   (first-seen order, deduped). Called by `recordCall` and directly by tests;
 *   part of the write contract because a run's model is only ever really known
 *   per call.
 *
 * Read side (`analyze/*`, `cli.ts`, `viewer/export.ts`):
 * - `listSessions` — every session in the store, oldest first, with agents and
 *   turn counts (the session-picker query).
 * - `getRun` — one session's metadata; no argument means the session this handle
 *   is bound to (the newest, for a reader).
 * - `getCalls` — one session's calls in turn order.
 * - `getCallBlocks` — one call's blocks in position order, rehydrated.
 *
 * Lifecycle:
 * - `close` — release the connection; must never throw.
 *
 * Anything a specific backend needs beyond this (SQLite's schema-version gate,
 * Postgres' statement timeout) stays private to that backend.
 */
export interface Store {
  /** Persist one call and its ordered blocks in a single transaction, deduping
   * blocks by content hash. Resolves to the new call id. */
  recordCall(args: RecordCallArgs): Awaitable<string>;
  /** Append `model` to this session's `models` list the first time it is seen;
   * ignore null/empty and repeats. */
  noteModel(model: string | null | undefined): Awaitable<void>;
  /** Every session in this store, oldest first. */
  listSessions(): Awaitable<Session[]>;
  /** One session's run row (default: this handle's bound session). */
  getRun(sessionId?: string): Awaitable<Run>;
  /** One session's calls ordered by turn sequence. */
  getCalls(sessionId?: string): Awaitable<Call[]>;
  /** One call's blocks in position order. */
  getCallBlocks(callId: string): Awaitable<CallBlock[]>;
  /** Release the underlying connection. Best-effort; never throws. */
  close(): Awaitable<void>;
}

/**
 * The synchronous READ subset every analyzer consumes (`analyze/diff.ts`,
 * `analyze/tokens.ts`, `analyze/cache.ts`, `viewer/export.ts`).
 *
 * Why it exists separately from `Store`: those analyzers are pure synchronous
 * functions that walk a run call-by-call and block-by-block, and making them
 * async would ripple through every caller and every test for no gain — the data
 * they need is bounded and already fully materialized by the time analysis
 * starts. So a networked store is read ONCE, up front, into an in-memory
 * snapshot that implements this interface (`snapshotStore()`), and the analyzers
 * keep taking a plain synchronous reader. `CTrace` implements it natively.
 */
export interface ReadableStore {
  getRun(sessionId?: string): Run;
  getCalls(sessionId?: string): Call[];
  getCallBlocks(callId: string): CallBlock[];
}

/** The arguments of `StoreBackend.openSession` — everything a new session row
 * needs that the backend does not already know. */
export interface OpenSessionArgs {
  project: string;
  provider: string;
  /** Usually "": a run's model is a per-CALL fact `wrap()` does not know yet, so
   * seeding a placeholder would store a permanent blank. `noteModel()` backfills
   * it from real call params. */
  model?: string;
  /** Canonical UTC-with-offset timestamp stamped by the CALLER, so a session
   * records when tracing began rather than when the store finished connecting.
   * Empty means "now", filled in by the backend. */
  startedAt?: string;
}

/**
 * A connection-less description of WHERE traces live, and the factory for
 * `Store` handles onto it. This is what `configure({ store })` takes and what
 * `CTXDIFF_STORE` resolves to.
 *
 * Constructing a backend must not connect, create tables, or touch the network —
 * all of that happens in `openSession()`/`openReader()`, so a misconfigured or
 * unreachable database surfaces inside `Tracer.wrap()`'s fail-open guard
 * (degrade capture, warn once) rather than exploding at import time in the
 * user's module.
 */
export interface StoreBackend {
  /** Create the schema if missing, start a NEW session in this store, and return
   * a `Store` bound to it. The write entry point `trace.init()` uses on the
   * first `wrap()`. */
  openSession(args: OpenSessionArgs): Awaitable<Store>;
  /** Open a read handle bound to the NEWEST session in this store — the entry
   * point the CLI/viewer use when no explicit session is named. */
  openReader(): Awaitable<Store>;
}

/**
 * A backend whose sessions live in a local FILE and open synchronously — today
 * exactly `SQLiteStore`. Two things key off this shape rather than off a class
 * check, mirroring how Python asks a backend for `path_for` instead of running
 * an isinstance chain:
 *
 * - `Tracer.path` (public API, asserted by existing tests) reports the concrete
 *   `.ctrace` a run writes to; a networked backend simply has none, and reports
 *   null.
 * - `Tracer.wrap()` opens a file session INLINE (no promise, no writer queue),
 *   which is what keeps the local path byte-for-byte the code it has always
 *   been, while a networked session is deferred off the host's call path
 *   entirely (see `trace.ts`).
 */
export interface FileStoreBackend extends StoreBackend {
  /** The concrete file this backend uses for `project`. */
  pathFor(project: string): string;
  openSession(args: OpenSessionArgs): Store;
}

/** Whether `backend` is file-backed (and therefore synchronous), asked by
 * capability — the presence of `pathFor` — rather than by class identity, so a
 * user-supplied backend that behaves like a file store is treated like one. */
export function isFileBackend(backend: StoreBackend): backend is FileStoreBackend {
  return typeof (backend as Partial<FileStoreBackend>).pathFor === "function";
}

/** A backend that publishes its own per-statement bound in SECONDS. Read by
 * capability so `Tracer.close()` can wait exactly as long as the store could
 * legitimately still be busy, and no longer — see `closeTimeoutFor` in
 * `trace.ts`. */
export interface BoundedBackend {
  statementTimeout: number;
}

/** The per-statement bound `backend` publishes, or null when it publishes none
 * (the local SQLite store, a test double). */
export function statementTimeoutOf(backend: unknown): number | null {
  const value = (backend as Partial<BoundedBackend> | null)?.statementTimeout;
  return typeof value === "number" && value > 0 ? value : null;
}
