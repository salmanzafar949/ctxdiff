/**
 * Read a networked store ONCE, up front, into memory — the bridge between an
 * async `Store` and the strictly synchronous analyzers.
 *
 * Why this exists: every analyzer (`analyze/diff.ts`, `analyze/tokens.ts`,
 * `analyze/cache.ts`) and the dashboard builder (`viewer/export.ts`) is a pure
 * synchronous function that walks a run call-by-call and block-by-block,
 * re-reading the same call's blocks several times (a diff of adjacent turns
 * reads each turn twice, the cache profiler reads every turn again). Making them
 * async would ripple `await` through every caller and every test and buy nothing
 * — the data is bounded and is fully known before analysis starts.
 *
 * So a networked read is done eagerly here, in a fixed number of round trips (one
 * for the run, one for its calls, one per call for its blocks), and handed to the
 * analyzers as an object satisfying the same synchronous `ReadableStore` surface
 * `CTrace` implements. The analyzers cannot tell the difference — which is the
 * point: `npx ctxdiff diff/tokens/cache/view/export` works against Postgres and
 * MySQL with no analyzer aware that a database exists.
 */
import type { Call, CallBlock, ReadableStore, Run, Session, Store } from "./base.js";

/**
 * An in-memory, synchronous view of ONE session in a store: its run row, its
 * calls in turn order, and each call's blocks in position order — and, only when
 * it was asked for, the store's full session list.
 *
 * The session list is OPTIONAL because it is the one piece of a snapshot whose
 * cost is set by the DATABASE rather than by the run being read (see
 * `snapshotStore`), and no analyzer wants it.
 *
 * `close()` exists so callers can treat a snapshot exactly like a `CTrace` in a
 * `finally` block; it is a no-op because the underlying connection is already
 * closed by the time the snapshot is handed over (see `snapshotStore`).
 */
export class StoreSnapshot implements ReadableStore {
  private readonly run: Run;
  private readonly calls: Call[];
  private readonly blocks: Map<string, CallBlock[]>;
  /** null when this snapshot was taken WITHOUT the session list — distinct from
   * an empty list, which means the store genuinely holds no sessions. */
  private readonly sessions: Session[] | null;

  constructor(
    run: Run,
    calls: Call[],
    blocks: Map<string, CallBlock[]>,
    sessions: Session[] | null = null,
  ) {
    this.run = run;
    this.calls = calls;
    this.blocks = blocks;
    this.sessions = sessions;
  }

  /** The snapshotted session's run row. A `sessionId` naming a DIFFERENT session
   * is rejected rather than silently answered with this one's data — a snapshot
   * covers exactly the session it was taken of. */
  getRun(sessionId?: string): Run {
    if (sessionId !== undefined && sessionId !== this.run.id) {
      throw new Error(`ctxdiff: session ${sessionId} is not in this snapshot`);
    }
    return this.run;
  }

  /** The snapshotted session's calls, in turn order. */
  getCalls(sessionId?: string): Call[] {
    if (sessionId !== undefined && sessionId !== this.run.id) {
      throw new Error(`ctxdiff: session ${sessionId} is not in this snapshot`);
    }
    return this.calls;
  }

  /** One call's blocks, in position order; an unknown call id reads as empty,
   * matching what a join would return. */
  getCallBlocks(callId: string): CallBlock[] {
    return this.blocks.get(callId) ?? [];
  }

  /** Every session in the store this snapshot came from, oldest first —
   * available only when the snapshot was taken with `{ sessions: true }`.
   * Asking without it is a caller bug, and says so: answering "no sessions" for
   * a database full of them would be a far worse failure. */
  listSessions(): Session[] {
    if (this.sessions === null) {
      throw new Error(
        "ctxdiff: this snapshot was taken without the store's session list — " +
          "take it with snapshotStore(store, { sessions: true }) if you need it",
      );
    }
    return this.sessions;
  }

  /** Present for symmetry with `CTrace` so CLI code can `finally { r.close() }`
   * regardless of where the trace came from. The connection is already gone. */
  close(): void {
    /* nothing to release: the snapshot owns no connection */
  }
}

/**
 * An in-memory, synchronous view of a WHOLE project: every session's row and
 * calls, and the blocks of only those sessions whose turn-by-turn detail the
 * dashboard will embed.
 *
 * Why a second snapshot class rather than a flag on `StoreSnapshot`: the two
 * answer different questions and have different costs, and conflating them
 * would make every single-session command pay a project-wide read. A
 * `StoreSnapshot` is ONE session, materialized completely, for the analyzers.
 * This is the whole project materialized ASYMMETRICALLY — calls for everything
 * (cheap, and what levels 1 and 2 aggregate) but blocks only for the capped set
 * (expensive, and what level 3 needs). Nothing but the dashboard wants that
 * shape, and the dashboard wants nothing else.
 *
 * A session outside the detail set answers `getCallBlocks` with `[]` rather than
 * throwing: its level-3 view was never going to be built, and the page renders
 * it as a listing row marked "detail not embedded".
 */
export class ProjectSnapshot implements ReadableStore {
  constructor(
    private readonly sessions: Session[],
    private readonly runs: Map<string, Run>,
    private readonly calls: Map<string, Call[]>,
    private readonly blocks: Map<string, CallBlock[]>,
    private readonly focusId: string,
  ) {}

  /** One session's run row; no argument means the focus session. Only the
   * sessions with embedded detail have one — the aggregate levels read the
   * session LIST, which carries the same facts without a per-session query. */
  getRun(sessionId?: string): Run {
    const id = sessionId ?? this.focusId;
    const run = this.runs.get(id);
    if (run === undefined) {
      throw new Error(`ctxdiff: session ${id} is not in this project snapshot`);
    }
    return run;
  }

  /** One session's calls, in turn order; every session in the project has them. */
  getCalls(sessionId?: string): Call[] {
    return this.calls.get(sessionId ?? this.focusId) ?? [];
  }

  /** One call's blocks, or `[]` for a call in a session outside the detail set. */
  getCallBlocks(callId: string): CallBlock[] {
    return this.blocks.get(callId) ?? [];
  }

  /** Every session in the project, oldest first — what levels 1 and 2 list. */
  listSessions(): Session[] {
    return this.sessions;
  }

  /** Present for symmetry with `CTrace`/`StoreSnapshot` so CLI code can
   * `finally { r.close() }`. The connection is already gone. */
  close(): void {
    /* nothing to release: the snapshot owns no connection */
  }
}

/** Which project `snapshotProject` reads, and how much of it. */
export interface ProjectSnapshotOptions {
  /** The session whose detail the dashboard focuses on; also the default of
   * every no-argument read. */
  focusId: string;
  /** The session list the caller already fetched — the CLI always has one,
   * because it needed it to resolve `--session` at all. */
  sessionList: Session[];
  /** The sessions whose BLOCKS to read: the dashboard's detail set, decided by
   * `detailSessionIds` in the viewer so the cap lives in exactly one place. */
  detailIds: string[];
}

/**
 * Materialize a whole project into a `ProjectSnapshot` and CLOSE the store —
 * the networked-store path behind `ctxdiff view`/`export`.
 *
 * The read is deliberately asymmetric, because the two halves of the dashboard
 * have completely different costs against a shared database:
 *
 * - CALLS for every session: one query per session, no joins to the block
 *   tables. This is what levels 1 and 2 aggregate, and it is the same read
 *   `ctxdiff agents` already does across the project, so listing every agent is
 *   affordable even against a database holding thousands of sessions.
 * - BLOCKS for the detail set only: one query per CALL of those sessions, which
 *   is the expensive axis and the reason the detail set is capped at all.
 *
 * Closing here rather than leaving it to the caller matches `snapshotStore`:
 * after this returns, every byte the exporter will read is in memory, and
 * holding a connection open for the duration of an HTML render is exactly the
 * liability a debugging tool should not add to someone's production database.
 */
export async function snapshotProject(
  store: Store,
  opts: ProjectSnapshotOptions,
): Promise<ProjectSnapshot> {
  try {
    const detail = new Set(opts.detailIds);
    const runs = new Map<string, Run>();
    const calls = new Map<string, Call[]>();
    const blocks = new Map<string, CallBlock[]>();
    for (const s of opts.sessionList) {
      const sessionCalls = await store.getCalls(s.id);
      calls.set(s.id, sessionCalls);
      if (!detail.has(s.id)) continue;
      runs.set(s.id, await store.getRun(s.id));
      for (const call of sessionCalls) blocks.set(call.id, await store.getCallBlocks(call.id));
    }
    // A focus session missing from the listing (a store that lists nothing but
    // is bound to a run) still has to answer `getRun`, or the export cannot even
    // name the project.
    if (!runs.has(opts.focusId)) runs.set(opts.focusId, await store.getRun(opts.focusId));
    return new ProjectSnapshot(opts.sessionList, runs, calls, blocks, opts.focusId);
  } finally {
    try {
      await store.close();
    } catch {
      /* the data is already read; a failed goodbye must not lose it */
    }
  }
}

/** Which session `snapshotStore` reads, and what it may read BEYOND it. */
export interface SnapshotOptions {
  /** Also materialize every session in the store (`ctxdiff sessions`' listing).
   * Off by default: it is the one read whose cost grows with the DATABASE
   * rather than with the run — see `snapshotStore`. */
  sessions?: boolean;
  /** Snapshot THIS session instead of the one the handle is bound to (the
   * newest) — how `--session` reaches a networked store, where a handle cannot
   * be rebound but every read already accepts a session id. */
  sessionId?: string;
  /** A session list the caller ALREADY fetched. The CLI must fetch one to
   * resolve `--session` at all, and passing it here means the expensive listing
   * query (see below) is paid once per command rather than twice. Takes
   * precedence over `sessions`. */
  sessionList?: Session[];
}

/**
 * Materialize `store`'s bound session into a `StoreSnapshot` and CLOSE the
 * store.
 *
 * Reads exactly the run it was pointed at, in a fixed number of round trips:
 * one for the run row, one for its calls, one per call for its blocks. The
 * store's SESSION LIST is deliberately not among them unless
 * `{ sessions: true }` asks for it, because it is the one read whose cost is set
 * by the database instead of by the run: `listSessions()` is a COUNT and a GROUP
 * BY across every call row anyone has ever written to that database — the shared
 * database these backends exist to serve. Taking it unconditionally meant
 * `diff`, `tokens`, `cache` and `export`, every one of which analyzes a SINGLE
 * run, each paid for the whole fleet's history — measured against a real
 * PostgreSQL holding 2001 sessions / 200k calls, a 40-turn run snapshots in
 * 11ms and the listing added 156ms on top, growing with the database forever.
 * Only `ctxdiff runs` wants the listing, and it asks for it.
 *
 * Closing here rather than leaving it to the caller is deliberate: after this
 * returns, nothing can possibly need the connection again (every byte the
 * analyzers will read is already in memory), and holding a database connection
 * open for the duration of an HTML export is exactly the kind of avoidable
 * liability a debugging tool should not add to someone's production database.
 * The close is best-effort — a snapshot that was read successfully is not lost
 * because the goodbye packet failed.
 */
export async function snapshotStore(
  store: Store,
  opts: SnapshotOptions = {},
): Promise<StoreSnapshot> {
  try {
    // `getRun`/`getCalls` take the session id when one was chosen and fall back
    // to the handle's own binding (the newest) when it wasn't — the same
    // default every store already implements, so nothing changes for a caller
    // that names no session.
    const run = await store.getRun(opts.sessionId);
    const calls = await store.getCalls(opts.sessionId);
    const blocks = new Map<string, CallBlock[]>();
    for (const call of calls) blocks.set(call.id, await store.getCallBlocks(call.id));
    const sessions =
      opts.sessionList ?? (opts.sessions === true ? await store.listSessions() : null);
    return new StoreSnapshot(run, calls, blocks, sessions);
  } finally {
    try {
      await store.close();
    } catch {
      /* the data is already read; a failed goodbye must not lose it */
    }
  }
}
