#!/usr/bin/env node
/**
 * ctxdiff's command-line entry point (`npx ctxdiff <cmd>`). Matches the Python
 * CLI's command surface — `diff`, `tokens`, `cache`, `check` (analysis), `sessions`,
 * `agents` (discovery, with `runs` kept as a hidden alias of `sessions`), plus
 * `view`, `export`, `demo` (the HTML dashboard) — with the same flags
 * (`--project`/`--run`, `--session`, `--agent`, `--turn`, `--out`, `--no-open`,
 * `--keep`), the same output, and the same exit codes. Uses Node's built-in
 * `util.parseArgs`; no CLI framework dependency. As an ergonomic extra over the
 * Python CLI, a positional `.ctrace` path is accepted as an alias for
 * `--project`.
 *
 * The surface is SESSION- and AGENT-aware because a project store holds many
 * sessions and each session many agents. Which session/agent a command reads is
 * resolved by `selectors.ts` — shared, identical rules for every command, and a
 * byte-identical port of Python's `cli/select.py` down to the error text.
 */
import { parseArgs } from "node:util";
import { readdirSync, statSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import { platform } from "node:process";
import { CTrace } from "./store/ctrace.js";
import {
  EmptyStoreError,
  isFileBackend,
  type Call,
  type CallBlock,
  type ReadableStore,
  type Run,
  type Session,
  type Store,
  type StoreBackend,
} from "./store/base.js";
import { fromDsn, resolve as resolveConfigured } from "./store/config.js";
import { SQLiteStore } from "./store/sqlite.js";
import { snapshotProject, snapshotStore } from "./store/snapshot.js";
import { diffCalls, diffTurns } from "./analyze/diff.js";
import { analyzeRun, registeredToolNames, usageTotals } from "./analyze/tokens.js";
import { analyzeCache } from "./analyze/cache.js";
import { analyzeEvictions } from "./analyze/evictions.js";
import {
  CONTEXT_WINDOW_ENV,
  ContextWindowError,
  resolveContextWindow,
} from "./analyze/window.js";
import { analyzeCheck, anyRequested, checkPassed, type Thresholds } from "./analyze/check.js";
import { pyComma as comma } from "./analyze/pyround.js";
import { isTraceFile, listFileCalls, listFileSessions } from "./analyze/sessions.js";
import { detailSessionIds, exportHtml, exportStore, type ProjectReader } from "./viewer/export.js";
import { buildDemoTrace } from "./demo.js";
import {
  renderTurnDiff,
  renderRunTokens,
  renderCacheReport,
  renderCheckReport,
  renderSessionsList,
  renderAgentsList,
  renderUsageSummary,
  renderAgentSummary,
  type AgentRow,
  type SessionRow,
} from "./render.js";
import {
  agentsText,
  chooseSession,
  diffScopeLine,
  distinctAgentNames,
  formatLocal,
  parseSelector,
  requireAgent,
  requireSingleAgent,
  SelectionError,
  shortId,
  turnArg,
  turnList,
  UNLABELED,
  type DiffSide,
  type Selector,
  type TurnArg,
} from "./selectors.js";

/** Most recently modified `*.ctrace` in `dir`, or null. Backs the `--project`
 * default so the common case (one project DB in the cwd) needs no flag. Mirrors
 * Python `_find_default_run`, including its `glob("*.ctrace")` blindness to
 * dot-prefixed files (see `isTraceFile`) — this one picks the file every later
 * number is computed from, so the two CLIs choosing differently here means they
 * report on different runs entirely. */
function findDefaultRun(dir: string): string | null {
  let entries: string[];
  try {
    entries = readdirSync(dir);
  } catch {
    return null;
  }
  const candidates = entries.filter(isTraceFile).map((f) => join(dir, f));
  if (!candidates.length) return null;
  return candidates.reduce((best, p) =>
    statSync(p).mtimeMs > statSync(best).mtimeMs ? p : best,
  );
}

/**
 * A trace opened for READING and pinned to ONE session, whichever backend it
 * came from: a `SessionView` over a local `CTrace`, or an in-memory snapshot of
 * one session of a Postgres/MySQL store. Both are synchronous and both close the
 * same way, so every command below reads one of these without knowing which it
 * got.
 */
interface Reader extends ReadableStore {
  close(): void;
}

/**
 * A read handle PINNED to one session of a (possibly many-session) `CTrace`.
 *
 * Why it exists: every analyzer and the dashboard exporter call
 * `getRun()`/`getCalls()` with no arguments and get the store's default binding
 * — the NEWEST session. Once `--session` can name any session, those calls have
 * to answer for the CHOSEN one instead. Rebinding a `CTrace` is not an option
 * (it binds its run id at open time), so this thin forwarder substitutes the
 * chosen session id on the two reads that take one and passes everything else
 * straight through. Mirrors Python `_SessionView`.
 *
 * `close()` is a NO-OP: the underlying `CTrace` is owned by the project handle
 * that created this view (see `openProject`), which may hand out two views for a
 * cross-session diff and must outlive both.
 */
class SessionView implements Reader {
  private readonly store: CTrace;
  readonly sessionId: string;

  constructor(store: CTrace, sessionId: string) {
    this.store = store;
    this.sessionId = sessionId;
  }

  /** The pinned session's run row (an explicit `sessionId` still wins, so a
   * caller that knows exactly what it wants is never overridden). */
  getRun(sessionId?: string): Run {
    return this.store.getRun(sessionId ?? this.sessionId);
  }

  /** The pinned session's calls, in turn order. */
  getCalls(sessionId?: string): Call[] {
    return this.store.getCalls(sessionId ?? this.sessionId);
  }

  /** One call's blocks — call ids are globally unique, so no pinning. */
  getCallBlocks(callId: string): CallBlock[] {
    return this.store.getCallBlocks(callId);
  }

  /** Every session in the underlying file (pinning scopes READS, not
   * discovery) — what the dashboard's level-1/level-2 listings are built from. */
  listSessions(): Session[] {
    return this.store.listSessions();
  }

  close(): void {
    /* the project handle owns the CTrace; see the class docstring */
  }
}

/** No trace to read: no `--project`, no configured backend, and no `*.ctrace` in
 * the working directory. Its own type so callers can print the friendly "did the
 * run capture?" line for this case while still reporting a genuine open FAILURE
 * (corrupt file, unreachable database) as an error. Mirrors Python
 * `_NoTraceFound`. */
class NoTraceFound extends Error {}

/**
 * The ONE project store a command should read, or null meaning "there is no
 * single project here — scan the working directory instead".
 *
 * Resolution mirrors the write side's explicit-beats-ambient rule:
 * 1. `--project VALUE` (or the positional) — always that, whether it is a
 *    `.ctrace` path or a database DSN. `fromDsn` already knows both spellings (a
 *    value with no recognizable scheme is a filesystem path), so one flag covers
 *    both and a path still beats an ambient database, as it always has.
 * 2. A configured NETWORKED backend (`configure()` / `CTXDIFF_STORE`) — detected
 *    by the ABSENCE of `pathFor`, the same file-backend capability check
 *    `Tracer` uses, rather than an instanceof chain over every backend class.
 * 3. A configured SQLite backend pointing at a concrete FILE — that file.
 * 4. Anything else (nothing configured, or a backend naming a whole directory of
 *    traces, which is not one project) — null.
 *
 * Returning null rather than throwing is what lets the discovery commands
 * (`sessions`, `agents`) list every `.ctrace` in the cwd while the analysis
 * commands narrow that to the newest one — see `resolveBackend`.
 */
function discoveryBackend(project: string | undefined): StoreBackend | null {
  if (project) return fromDsn(project);
  const backend = resolveConfigured();
  if (backend !== null) {
    if (!isFileBackend(backend)) return backend;
    const configuredPath = (backend as SQLiteStore).path;
    if (configuredPath && !isDirectory(configuredPath)) return backend;
  }
  return null;
}

/** The store an analysis command will read, plus the path it was DISCOVERED at
 * (null whenever the user named the project themselves). `discovered` is carried
 * purely so selector errors can name the file — see `sessionWhere`. */
interface ResolvedBackend {
  backend: StoreBackend;
  discovered: string | null;
}

/** The store an ANALYSIS command should read: whatever `discoveryBackend`
 * resolves, else the most recently modified `*.ctrace` in the cwd — the
 * zero-config default that has always made `ctxdiff diff` work with no flags in
 * a directory holding one project DB. Throws `NoTraceFound` when even that finds
 * nothing. Mirrors Python `_resolve_backend`.
 *
 * Only the cwd fall-through reports a `discovered` path: it is the one branch
 * where the user never said which project this is, so it is the one branch whose
 * errors have to. */
function resolveBackend(project: string | undefined): ResolvedBackend {
  const backend = discoveryBackend(project);
  if (backend !== null) return { backend, discovered: null };
  const path = findDefaultRun(process.cwd());
  if (path === null) throw new NoTraceFound();
  return { backend: new SQLiteStore({ path }), discovered: path };
}

/**
 * The scope label a selector error uses for one session: `session 4f3a2b1c9d8e`
 * normally, and `one.ctrace (session 4f3a2b1c9d8e)` when the project was
 * DISCOVERED by scanning the working directory rather than named with
 * `--project`.
 *
 * Why the filename earns its place exactly there: `ctxdiff agents` lists agents
 * from EVERY .ctrace in the directory, so the obvious next command —
 * `ctxdiff tokens --agent alpha` — can name an agent that really exists, just
 * not in the one file the no-flag default happened to pick. Unqualified, the
 * error then names a session short id that appears NOWHERE in the `sessions`
 * listing (whose rows are labeled by filename), leaving no hint that a different
 * project was chosen or that `--project` is the way out. Naming the file turns a
 * dead end into a pointer. Mirrors Python `_session_where`.
 */
function sessionWhere(sessionId: string, discovered: string | null): string {
  const where = `session ${shortId(sessionId)}`;
  return discovered ? `${basename(discovered)} (${where})` : where;
}

/** Whether `path` is an existing directory — used to tell a configured
 * `SQLiteStore` that names ONE file from one that names a directory of them. */
function isDirectory(path: string): boolean {
  const st = statSync(path, { throwIfNoEntry: false });
  return st !== undefined && st.isDirectory();
}

/** The dashboard path `export`/`view` default to for `backend`:
 * `<trace-stem>.html` beside a file-backed store, and null for a networked one
 * (a database has no filename to borrow, so those commands ask for `--out`). */
function htmlDefaultForBackend(backend: StoreBackend): string | null {
  if (!isFileBackend(backend)) return null;
  const path = (backend as SQLiteStore).path;
  if (path && !isDirectory(path)) return defaultHtmlFor(path);
  return null;
}

/** The dashboard path `export`/`view` default to for a file-backed trace:
 * `<trace-stem>.html` right beside the trace — unchanged from when the CLI
 * passed the path straight to `exportHtml`. */
function defaultHtmlFor(ctracePath: string): string {
  const abs = resolve(ctracePath);
  const stem = basename(abs).replace(/\.[^.]*$/, "");
  return join(dirname(abs), `${stem}.html`);
}

/**
 * An opened project: everything a command needs to choose WITHIN it, plus a way
 * to materialize any one of its sessions as a synchronous `Reader`.
 *
 * The session list is fetched eagerly because ambiguity has to be DETECTED
 * before anything is analyzed — a command cannot know whether it may default to
 * the newest session without knowing how many there are. That is one extra query
 * against a networked store, paid once per command.
 */
interface ProjectHandle {
  /** Every session in the project, oldest first. */
  sessions: Session[];
  /** The session the store binds to by default: the newest. */
  defaultId: string;
  /** Where `export`/`view` write with no `--out`, or null for a database. */
  htmlDefault: string | null;
  /** The path this project was DISCOVERED at, or null when the user named it.
   * Selector errors name it — see `sessionWhere`. */
  discovered: string | null;
  /** Materialize one session as a synchronous reader. Called twice (with two
   * different ids) by a cross-session diff, which is exactly why this is a
   * factory rather than a single pre-opened reader. */
  read(sessionId: string): Promise<Reader>;
  /** Materialize the WHOLE project as a synchronous reader focused on
   * `sessionId` — what the three-level dashboard needs and no analysis command
   * does. Separate from `read` because the reads are genuinely different: this
   * one pulls every session's calls for the aggregate levels and only the capped
   * set's blocks (see `store/snapshot.ts`). */
  readProject(sessionId: string): Promise<ProjectReaderHandle>;
  /** Release anything still held open — safe to call whether or not `read` was
   * ever reached (a selector error aborts before it is). */
  close(): Promise<void>;
}

/** A whole-project reader: the synchronous analyzer surface plus the session
 * listing, closable like every other reader. */
interface ProjectReaderHandle extends ProjectReader {
  close(): void;
}

/**
 * Open the resolved project store. Two shapes, because the two families of
 * backend have genuinely different costs:
 *
 * - A FILE backend is synchronous and cheap to keep open, so one `CTrace` is
 *   opened and every `read()` is a `SessionView` over it. Two sessions of one
 *   file cost one file handle.
 * - A NETWORKED backend is read into an in-memory snapshot (the analyzers are
 *   synchronous — see `store/snapshot.ts`), and `snapshotStore` closes the
 *   connection it was handed. So the connection opened for the session listing
 *   is HANDED TO the first `read()` rather than closed and reopened, and only a
 *   genuine second session (a cross-session diff) pays for a second connection.
 *   The already-fetched session list is passed along so the expensive listing
 *   query is never run twice.
 */
async function openProject(project: string | undefined): Promise<ProjectHandle> {
  const { backend, discovered } = resolveBackend(project);
  const htmlDefault = htmlDefaultForBackend(backend);

  if (isFileBackend(backend)) {
    const ct = backend.openReader() as CTrace;
    let sessions: Session[];
    let defaultId: string;
    try {
      sessions = ct.listSessions();
      defaultId = ct.getRun().id;
    } catch (err) {
      // Never leak the handle when the follow-up reads fail: the caller only
      // gets a project it can close on success.
      ct.close();
      throw err;
    }
    return {
      sessions,
      defaultId,
      htmlDefault,
      discovered,
      read: async (sessionId: string) => new SessionView(ct, sessionId),
      // A file store is synchronous and already open, so the "project reader"
      // is the same pinned view — it can read any session of the file directly.
      readProject: async (sessionId: string) => new SessionView(ct, sessionId),
      close: async () => ct.close(),
    };
  }

  const first = await backend.openReader();
  let sessions: Session[];
  let defaultId: string;
  try {
    sessions = await first.listSessions();
    defaultId = (await first.getRun()).id;
  } catch (err) {
    await first.close();
    throw err;
  }
  let pending: Store | null = first;
  return {
    sessions,
    defaultId,
    htmlDefault,
    discovered,
    read: async (sessionId: string) => {
      const store = pending ?? (await backend.openReader());
      pending = null;
      return snapshotStore(store, { sessionId, sessionList: sessions });
    },
    readProject: async (sessionId: string) => {
      const store = pending ?? (await backend.openReader());
      pending = null;
      return snapshotProject(store, {
        focusId: sessionId,
        sessionList: sessions,
        detailIds: detailSessionIds(sessions, sessionId),
      });
    },
    close: async () => {
      if (pending) {
        const store = pending;
        pending = null;
        await store.close();
      }
    },
  };
}

/** Open the resolved project bound to ONE session — the entry point every
 * single-session command (`tokens`, `cache`, `export`, `view`, and the ordinary
 * `diff`) uses. Throws `SelectionError` when the session cannot be resolved
 * (unknown id, ambiguous prefix, or several sessions and no `--session`), always
 * closing the handle on the way out. Mirrors Python `_open_session`. */
async function openSession(
  project: string | undefined,
  session: string | null,
): Promise<{ reader: Reader; sessionId: string; htmlDefault: string | null; handle: ProjectHandle }> {
  const handle = await openProject(project);
  try {
    const sessionId = chooseSession(handle.sessions, session, handle.defaultId);
    const reader = await handle.read(sessionId);
    return { reader, sessionId, htmlDefault: handle.htmlDefault, handle };
  } catch (err) {
    await handle.close();
    throw err;
  }
}

/** Every NAMED agent anywhere in the project, in first-appearance order, read
 * from the session rows the listing already carries — so validating `--agent`
 * for the dashboard costs no extra query. Mirrors Python
 * `_project_agent_names`. */
function projectAgentNames(sessions: Session[]): string[] {
  const names: string[] = [];
  for (const s of sessions) for (const n of s.agents) if (!names.includes(n)) names.push(n);
  return names;
}

/**
 * Open a project for `export`/`view`, focused on the session whose detail the
 * dashboard shows.
 *
 * Why this is NOT `openSession`: every other read command analyzes exactly one
 * session, so several sessions and no `--session` is a question it must refuse
 * to guess at. The dashboard is the opposite — it exists to show the WHOLE
 * project, and it opens on the agent listing precisely when there is more than
 * one thing to choose from. So a bare `ctxdiff view` on a many-session project
 * is not ambiguous any more; it focuses the newest session and lands the user on
 * level 1, where picking is what the page is for.
 *
 * `--session` still resolves through `chooseSession` (same errors, same
 * listing), and `--agent` is checked against the whole project so a typo is
 * caught here rather than silently opening on a level scoped to nobody. Mirrors
 * Python `_open_dashboard`.
 */
async function openDashboard(
  project: string | undefined,
  session: string | null,
  agent: string | null,
): Promise<{
  reader: ProjectReaderHandle;
  htmlDefault: string | null;
  sessionSelected: boolean;
  handle: ProjectHandle;
}> {
  const handle = await openProject(project);
  try {
    const sessionId =
      session !== null
        ? chooseSession(handle.sessions, session, handle.defaultId)
        : handle.defaultId;
    if (agent !== null) requireAgent(agent, projectAgentNames(handle.sessions), "this project");
    const reader = await handle.readProject(sessionId);
    return { reader, htmlDefault: handle.htmlDefault, sessionSelected: session !== null, handle };
  } catch (err) {
    await handle.close();
    throw err;
  }
}

/**
 * Print the right message for a failed store open and return the exit code, so
 * every read command reports identically: the friendly "did the run capture?"
 * line when there was nothing to open, and `ctxdiff: <error>` for a genuine
 * failure — both exit 1.
 *
 * The prefix is added only when the message does not ALREADY carry it: most
 * errors ctxdiff raises itself are spelled `ctxdiff: ...` so they read correctly
 * wherever they surface, and blindly prepending here produced "ctxdiff: ctxdiff:
 * no sessions recorded", which reads like a bug in the tool.
 */
function reportOpenFailure(err: unknown): number {
  if (err instanceof NoTraceFound) {
    process.stderr.write("no .ctrace here — did the run capture?\n");
    return 1;
  }
  const message = (err as Error).message ?? String(err);
  process.stderr.write(
    (message.startsWith("ctxdiff:") ? message : `ctxdiff: ${message}`) + "\n",
  );
  return 1;
}

/**
 * Print an unresolvable `--session`/`--agent` and return exit code 2.
 *
 * Exit 2 (not 1) because this is a USAGE error in argparse's sense — the flags
 * as given cannot be acted on — and the message is printed verbatim because
 * `selectors.ts` already built it complete with the listing of what could be
 * picked instead. Mirrors Python `_report_selection_error`.
 */
function reportSelectionError(err: SelectionError): number {
  process.stderr.write(err.message + "\n");
  return 2;
}

/** Route any error out of a command to the right reporter: a selector problem is
 * a usage error (exit 2, with its listing), anything else is an open/read
 * failure (exit 1). One helper so all five read commands agree. */
function reportCommandFailure(err: unknown): number {
  if (err instanceof SelectionError) return reportSelectionError(err);
  return reportOpenFailure(err);
}

interface ParsedArgs {
  /** Every `--turn` value, in order, each carrying the text the user typed. */
  turns: TurnArg[];
  /** Every `--agent` value, in order. Non-repeatable commands see at most one. */
  agents: string[];
  /** Every `--session` value, in order. */
  sessions: string[];
  /** `--project`, its `--run` alias, or a leading positional path. */
  project: string | undefined;
  /** Every parsed flag, verbatim — how a command reads the extra options it
   * registered through `parseCommon`'s `extra` (see `cmdCheck`), without every
   * command's private flags having to grow a field on this shared shape. */
  values: Record<string, unknown>;
}

/** A usage error (bad flags/values). Carries the one-line `ctxdiff: error: …`
 * message to print to stderr; `main` turns it into exit code 2, matching the
 * Python CLI's argparse convention. */
class UsageError extends Error {}

/** Format an int array like Python's list repr: `[1, 2, 3]` (space after each
 * comma), unlike `JSON.stringify` which omits the spaces. For lists holding a
 * turn the USER typed, use `turnList` instead — it echoes the digits rather than
 * a double's rendering of them. */
function pyIntList(arr: number[]): string {
  return "[" + arr.join(", ") + "]";
}

/** Python `int(str)` acceptance: optional surrounding whitespace, optional
 * sign, then decimal digits. Rejects "abc", "1.5", "", "0x10" — matching
 * argparse's `type=int`. */
const INT_RE = /^\s*[+-]?\d+\s*$/;

/** The `--max-*-pct` grammar: optional whitespace, optional sign, then ASCII
 * decimal digits with an optional fractional part (or a bare fraction).
 * Narrower than `Number()`/`float()` on purpose — both also accept `Infinity`,
 * `NaN`, `1e4` and (in Python) non-ASCII decimal digits, none of which is a
 * percentage anyone means to type, and each of which the two CLIs would then
 * have to agree about. Mirrors Python `_FLOAT_RE`. */
const FLOAT_RE = /^\s*[+-]?(\d+(\.\d*)?|\.\d+)\s*$/;

/** Reformat a `node:util` parseArgs error into a Python-style
 * `ctxdiff: error: …` message. Unknown options mirror argparse's top-level
 * "unrecognized arguments: <opt>"; a missing option value mirrors the
 * subparser's "argument <opt>: expected one argument". */
function usageErrorFromParse(err: unknown): UsageError {
  const e = err as { code?: string; message?: string };
  if (e.code === "ERR_PARSE_ARGS_UNKNOWN_OPTION") {
    const m = /Unknown option '([^']+)'/.exec(e.message ?? "");
    const opt = m ? m[1] : (e.message ?? "");
    return new UsageError(`ctxdiff: error: unrecognized arguments: ${opt}`);
  }
  if (e.code === "ERR_PARSE_ARGS_INVALID_OPTION_VALUE") {
    const m = /Option '(--?[\w-]+)/.exec(e.message ?? "");
    const opt = m ? m[1] : "argument";
    return new UsageError(`ctxdiff: error: argument ${opt}: expected one argument`);
  }
  return new UsageError(`ctxdiff: error: ${e.message ?? "invalid arguments"}`);
}

/** The three selectors a command may register beyond `--project`/`--run`. */
type SelectorFlag = "turn" | "agent" | "session";
const ALL_SELECTORS: SelectorFlag[] = ["turn", "agent", "session"];

/** argparse's own `_negative_number_matcher`. Reproduced rather than
 * approximated so `--turn -1` and `--turn -1.5` are classified exactly as
 * Python classifies them — see `joinNegativeTurnValues`. */
const NEGATIVE_NUMBER = /^-\d+$|^-\d*\.\d+$/;

/**
 * Rewrite `--turn -1` (and `--max-growth -5`, and every other numeric flag
 * named in `flags`) into the `--flag=value` form before parsing.
 *
 * Why: `parseArgs` treats ANY token starting with `-` as an option, so a
 * negative value fails with "expected one argument" (exit 2) where argparse —
 * which, having no options that look like negative numbers, applies its
 * negative-number heuristic — accepts `-1` as the VALUE and goes on to report
 * what is actually wrong with it ("turn -1 not found", "--max-growth cannot be
 * negative"). The `=` form is the same thing `parseArgs` would have done had
 * the value not begun with a dash. Using argparse's own matcher also routes
 * `--turn -1.5` to the int check, so it fails with Python's "invalid int
 * value: '-1.5'" rather than a parser error about a missing value.
 *
 * Every numeric flag gets this treatment rather than just `--turn`, because a
 * negative number is exactly as typable — and exactly as worth a real error
 * message — on `--max-context -1` as it is on `--turn -1`.
 *
 * Scanning stops at a bare `--`, exactly as argparse's does.
 */
function joinNegativeNumberValues(args: string[], flags: string[]): string[] {
  const out: string[] = [];
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--") {
      out.push(...args.slice(i));
      break;
    }
    if (flags.includes(args[i]) && i + 1 < args.length && NEGATIVE_NUMBER.test(args[i + 1])) {
      out.push(`${args[i]}=${args[i + 1]}`);
      i++;
      continue;
    }
    out.push(args[i]);
  }
  return out;
}

/**
 * Parse the flags shared by the analysis commands. `--turn` is always
 * `multiple` so `diff` can pass it twice (the callers apply their own count
 * rules); `--session`/`--agent` are `multiple` ONLY for `diff`, which is the one
 * command that names two sides — everywhere else a repeated flag means
 * last-wins, exactly as argparse behaves without `action="append"`. `--project`
 * and its hidden `--run` alias write the same value, and a leading positional is
 * treated as that path too. Throws `UsageError` (→ exit 2) on an unknown flag, a
 * missing option value, or a non-integer `--turn`, matching the Python CLI's
 * exit code and message text.
 *
 * `only` names the selectors THIS command actually acts on, and defaults to all
 * three. It is not cosmetic: Python registers each flag on the one subparser
 * that honors it, so `ctxdiff agents --agent researcher` is exit 2 there.
 * Registering every flag everywhere and reading a subset made that command print
 * EVERY agent and exit 0 — which reads as "the filter matched everything", so a
 * script grepping the output is wrong forever and never finds out. Unregistered
 * flags now reach `ERR_PARSE_ARGS_UNKNOWN_OPTION` and the shared
 * `usageErrorFromParse`, instead of each command growing its own parser.
 */
function parseCommon(
  rest: string[],
  opts: {
    repeatable?: boolean;
    only?: SelectorFlag[];
    /** Flags THIS command adds on top of the shared selectors (`check`'s
     * thresholds). Registered here rather than in a private parser so those
     * commands still get `--project`/`--run`/positional resolution and the
     * shared unknown-flag → exit 2 path for free. */
    extra?: Record<string, { type: "string" | "boolean" }>;
    /** Which of `extra`'s flags take a NUMBER, and so may legitimately be
     * given a negative value — see `joinNegativeNumberValues`. `--turn` is
     * added automatically when this command registers it. */
    numeric?: string[];
    /** Whether a leading positional is accepted as the project path. Default
     * true, matching the commands whose Python subparser registers one.
     *
     * `check` sets it false because Python's `check` subparser registers NO
     * positional, so `ctxdiff check --require-stable-prefix false` is exit 2
     * there ("unrecognized arguments: false"). Adopting the stray token as
     * `--project` instead — which is what a shared positional rule does —
     * makes the two CLIs disagree on both the exit code and the report, and
     * silently swallows a typo'd boolean in exactly the command whose job is to
     * not swallow things. */
    positional?: boolean;
  } = {},
): ParsedArgs {
  const repeatable = opts.repeatable === true;
  const only = opts.only ?? ALL_SELECTORS;
  const options: Record<string, { type: "string" | "boolean"; multiple?: boolean }> = {
    project: { type: "string" },
    run: { type: "string" },
    ...(opts.extra ?? {}),
  };
  if (only.includes("turn")) options.turn = { type: "string", multiple: true };
  if (only.includes("agent")) options.agent = { type: "string", multiple: repeatable };
  if (only.includes("session")) options.session = { type: "string", multiple: repeatable };

  const numericFlags = [
    ...(only.includes("turn") ? ["--turn"] : []),
    ...(opts.numeric ?? []).map((f) => `--${f}`),
  ];
  let parsed;
  try {
    parsed = parseArgs({
      args: numericFlags.length ? joinNegativeNumberValues(rest, numericFlags) : rest,
      options,
      allowPositionals: true,
    });
  } catch (err) {
    throw usageErrorFromParse(err);
  }
  if (opts.positional === false && parsed.positionals.length) {
    // argparse's own wording for a token no parser claimed, joined the way
    // argparse joins several of them.
    throw new UsageError(
      `ctxdiff: error: unrecognized arguments: ${parsed.positionals.join(" ")}`,
    );
  }
  const rawTurns = (parsed.values.turn as string[] | undefined) ?? [];
  const turns: TurnArg[] = [];
  for (const raw of rawTurns) {
    if (!INT_RE.test(raw)) {
      throw new UsageError(`ctxdiff: error: argument --turn: invalid int value: '${raw}'`);
    }
    turns.push(turnArg(raw));
  }
  const many = (v: unknown): string[] =>
    v === undefined ? [] : Array.isArray(v) ? (v as string[]) : [v as string];
  return {
    turns,
    agents: many(parsed.values.agent),
    sessions: many(parsed.values.session),
    project:
      (parsed.values.project as string | undefined) ??
      (parsed.values.run as string | undefined) ??
      (parsed.positionals[0] as string | undefined),
    values: parsed.values as Record<string, unknown>,
  };
}

/** Reject an `--agent` naming nobody in the session under analysis, listing the
 * agents that ARE there.
 *
 * Why this is worth a read: without it a typo'd or stale agent name filters every
 * call away and the command cheerfully reports "no calls in this run" —
 * technically true, actively misleading, and exit 0, so a CI check built on it
 * would pass forever. Throwing `SelectionError` turns it into what it is: a bad
 * flag (exit 2) with the correct values printed underneath. Mirrors Python
 * `_check_agent_filter`. */
function checkAgentFilter(
  reader: Reader,
  sessionId: string,
  agent: string | null,
  discovered: string | null,
): void {
  if (agent === null) return;
  requireAgent(agent, distinctAgentNames(reader.getCalls()), sessionWhere(sessionId, discovered));
}

// --- diff ----------------------------------------------------------------------

/** Which turn one side of a cross diff should read, in priority order: the
 * side's own `:TURN` suffix; else the matching one of two `--turn` flags; else a
 * single `--turn` applied to BOTH sides (the common regression shape — "turn 8
 * in the good run vs turn 8 in the bad one"). null means the user named no turn
 * for this side, which is a usage error the caller reports. Mirrors Python
 * `_side_turn`. */
function sideTurn(selector: Selector, index: number, turns: TurnArg[]): TurnArg | null {
  if (selector.turn !== null) return selector.turn;
  if (turns.length === 2) return turns[index];
  if (turns.length === 1) return turns[0];
  return null;
}

/** Assemble one `DiffSide`, turning "no turn was named for this side" into a
 * `SelectionError` (exit 2) that spells out both ways to name one. */
function makeSide(
  sessionId: string,
  agent: string | null,
  selector: Selector,
  index: number,
  turns: TurnArg[],
  axis: string,
): DiffSide {
  const turn = sideTurn(selector, index, turns);
  if (turn === null) {
    throw new SelectionError(
      `ctxdiff: each side of a cross-${axis} diff needs a turn — pass ` +
        `--${axis} VALUE:TURN twice, or --turn N --turn M`,
    );
  }
  return { sessionId, agent, turn };
}

/** Load the ordered blocks of the ONE call a diff side names, validating that
 * the turn exists (and belongs to the named agent) first. Throws with a
 * session-qualified message — a cross diff spans two sessions, so "turn 8 not
 * found" without saying WHERE is unusable. Mirrors Python `_side_blocks`. */
function sideBlocks(
  reader: Reader,
  side: DiffSide,
  discovered: string | null,
): CallBlock[] {
  const calls = reader.getCalls(side.sessionId);
  const where = sessionWhere(side.sessionId, discovered);
  let match: Call | undefined;
  if (side.agent !== null) {
    const owned = calls.filter((c) => c.agent === side.agent);
    match = owned.find((c) => c.seq === side.turn.value);
    if (!match) {
      throw new Error(
        `${where}: turn ${side.turn.text} is not a call of agent '${side.agent}' ` +
          `(that agent's turns: ${pyIntList(owned.map((c) => c.seq))})`,
      );
    }
  } else {
    match = calls.find((c) => c.seq === side.turn.value);
    if (!match) {
      throw new Error(
        `${where}: turn ${side.turn.text} not found (available turns: ` +
          `${pyIntList(calls.map((c) => c.seq).sort((a, b) => a - b))})`,
      );
    }
  }
  return reader.getCallBlocks(match.id);
}

/** `ctxdiff diff` in its three shapes, dispatched purely on how many times
 * `--session`/`--agent` were passed:
 *
 * - twice `--session` -> CROSS-SESSION (same agent, two runs);
 * - twice `--agent`   -> CROSS-AGENT (two agents, one run);
 * - otherwise         -> the ordinary two-turn diff within one session.
 *
 * Everything `parseArgs` cannot express is checked here: at most two of each
 * flag, never both axes at once, and a `:TURN` suffix only where it means
 * something. All of those are usage errors (exit 2). Mirrors Python `_cmd_diff`.
 */
async function cmdDiff(rest: string[]): Promise<number> {
  const args = parseCommon(rest, { repeatable: true });
  const sessionSels = args.sessions.map(parseSelector);
  const agentSels = args.agents.map(parseSelector);

  if (sessionSels.length > 2 || agentSels.length > 2) {
    process.stderr.write(
      "ctxdiff diff compares exactly two sides: pass --session (or --agent) " +
        "at most twice\n",
    );
    return 2;
  }
  if (sessionSels.length === 2 && agentSels.length === 2) {
    process.stderr.write(
      "ctxdiff diff compares along ONE axis: pass two --session values " +
        "(cross-session) or two --agent values (cross-agent), not both\n",
    );
    return 2;
  }

  const crossSession = sessionSels.length === 2;
  const crossAgent = agentSels.length === 2;
  if (!crossSession && !crossAgent) {
    if ([...sessionSels, ...agentSels].some((s) => s.turn !== null)) {
      process.stderr.write(
        "ctxdiff diff: a ':TURN' suffix only means something on a " +
          "cross-session or cross-agent diff — use --turn N --turn M here\n",
      );
      return 2;
    }
    if (args.turns.length !== 2) {
      process.stderr.write(
        "ctxdiff diff requires exactly two --turn flags, e.g. " +
          "'ctxdiff diff --turn 7 --turn 8'\n",
      );
      return 2;
    }
    return await diffWithinSession(args, sessionSels, agentSels);
  }

  if (args.turns.length > 2) {
    process.stderr.write("ctxdiff diff accepts at most two --turn flags\n");
    return 2;
  }
  return await diffCross(args, sessionSels, agentSels, crossSession);
}

/** The ordinary diff: two turns of ONE session, optionally filtered to one
 * agent. Unchanged in behavior from before sessions existed — same ownership
 * check, same messages, same exit codes — except that the session it reads is
 * now chosen rather than assumed. */
async function diffWithinSession(
  args: ParsedArgs,
  sessionSels: Selector[],
  agentSels: Selector[],
): Promise<number> {
  const wanted = sessionSels.length ? sessionSels[0].name : null;
  const agent = agentSels.length ? agentSels[0].name : null;
  let opened;
  try {
    opened = await openSession(args.project, wanted);
  } catch (err) {
    return reportCommandFailure(err);
  }
  const { reader: ct, sessionId, handle } = opened;
  try {
    try {
      checkAgentFilter(ct, sessionId, agent, handle.discovered);
    } catch (err) {
      return reportCommandFailure(err);
    }
    const [turnOld, turnNew] = args.turns;
    // --agent: validate BOTH turns belong to that agent before diffing. Turns
    // stay global seq numbers (what every view shows); we only check ownership,
    // then diffTurns resolves them against the whole session as usual.
    if (agent !== null) {
      const agentSeqs = ct.getCalls().filter((c) => c.agent === agent).map((c) => c.seq);
      const bad = [turnOld, turnNew].filter((t) => !agentSeqs.includes(t.value));
      if (bad.length) {
        process.stderr.write(
          `ctxdiff: turn(s) ${turnList(bad)} are not calls of agent ` +
            `'${agent}' (that agent's turns: ${pyIntList(agentSeqs)})\n`,
        );
        return 1;
      }
    }
    let diff;
    try {
      diff = diffTurns(ct, turnOld.value, turnNew.value, [turnOld.text, turnNew.text]);
    } catch (err) {
      process.stderr.write(`ctxdiff: ${(err as Error).message}\n`);
      return 1;
    }
    process.stdout.write(renderTurnDiff(diff) + "\n");
    return 0;
  } finally {
    ct.close();
    await handle.close();
  }
}

/**
 * The cross-session / cross-agent diff. Reuses `diffCalls` verbatim — the differ
 * aligns two ordered block lists by content hash and has never cared where they
 * came from, so comparing turn 8 of a good run against turn 8 of a bad one is
 * the SAME operation as comparing two turns of one run. All this adds is
 * resolving which two calls those are, and a scope header naming them (without
 * which two `turn 8`s in one header would be indistinguishable).
 */
async function diffCross(
  args: ParsedArgs,
  sessionSels: Selector[],
  agentSels: Selector[],
  crossSession: boolean,
): Promise<number> {
  let handle: ProjectHandle;
  try {
    handle = await openProject(args.project);
  } catch (err) {
    return reportOpenFailure(err);
  }
  try {
    let left: DiffSide;
    let right: DiffSide;
    let leftReader: Reader;
    let rightReader: Reader;
    try {
      const built = crossSession
        ? await crossSessionSides(handle, sessionSels, agentSels, args.turns)
        : await crossAgentSides(handle, sessionSels, agentSels, args.turns);
      [left, right] = built.sides;
      [leftReader, rightReader] = built.readers;
    } catch (err) {
      return reportCommandFailure(err);
    }
    let oldBlocks: CallBlock[];
    let newBlocks: CallBlock[];
    try {
      oldBlocks = sideBlocks(leftReader, left, handle.discovered);
      newBlocks = sideBlocks(rightReader, right, handle.discovered);
    } catch (err) {
      process.stderr.write(`ctxdiff: ${(err as Error).message}\n`);
      return 1;
    }
    const diff = diffCalls(oldBlocks, newBlocks, left.turn.value, right.turn.value);
    const scope = diffScopeLine(left, right);
    if (scope) process.stdout.write(scope + "\n");
    process.stdout.write(renderTurnDiff(diff) + "\n");
    return 0;
  } finally {
    await handle.close();
  }
}

/** Both sides of a CROSS-SESSION diff — the regression case: the same agent, two
 * runs. Resolves each `--session` value to a real session id, then settles the
 * agent ONCE for both sides, since "the same agent in two runs" is the entire
 * premise: an explicit `--agent` is validated against both sessions, and with
 * none given the agent is inferred only when it cannot be wrong (exactly one
 * agent, or none at all) — otherwise the user is asked. Mirrors Python
 * `_cross_session_sides`. */
async function crossSessionSides(
  handle: ProjectHandle,
  sessionSels: Selector[],
  agentSels: Selector[],
  turns: TurnArg[],
): Promise<{ sides: DiffSide[]; readers: Reader[] }> {
  const ids = sessionSels.map((s) => chooseSession(handle.sessions, s.name, handle.defaultId));
  const first = await handle.read(ids[0]);
  // Two `--session` values that resolve to the SAME session need only one
  // reader — and against a database that is one fewer connection.
  const second = ids[1] === ids[0] ? first : await handle.read(ids[1]);
  const readers = [first, second];
  const perSessionAgents = readers.map((r) => distinctAgentNames(r.getCalls()));

  let agent: string | null = agentSels.length ? agentSels[0].name : null;
  if (agent === null) {
    const combined: string[] = [];
    for (const names of perSessionAgents) {
      for (const n of names) if (!combined.includes(n)) combined.push(n);
    }
    requireSingleAgent(combined, "these sessions");
    agent = combined.length ? combined[0] : null;
  } else {
    for (let i = 0; i < ids.length; i++) {
      requireAgent(agent, perSessionAgents[i], sessionWhere(ids[i], handle.discovered));
    }
  }

  return {
    sides: [0, 1].map((i) => makeSide(ids[i], agent, sessionSels[i], i, turns, "session")),
    readers,
  };
}

/** Both sides of a CROSS-AGENT diff: two agents, one session — "why does the
 * writer see a different context than the researcher?". The session is resolved
 * exactly as for any single-session command (so it may be omitted when the
 * project holds one), and both agent names are validated against that session
 * before any turn is looked up. Mirrors Python `_cross_agent_sides`. */
async function crossAgentSides(
  handle: ProjectHandle,
  sessionSels: Selector[],
  agentSels: Selector[],
  turns: TurnArg[],
): Promise<{ sides: DiffSide[]; readers: Reader[] }> {
  const wanted = sessionSels.length ? sessionSels[0].name : null;
  const sessionId = chooseSession(handle.sessions, wanted, handle.defaultId);
  const reader = await handle.read(sessionId);
  const names = distinctAgentNames(reader.getCalls());
  for (const sel of agentSels) {
    requireAgent(sel.name, names, sessionWhere(sessionId, handle.discovered));
  }
  return {
    sides: [0, 1].map((i) =>
      makeSide(sessionId, agentSels[i].name, agentSels[i], i, turns, "agent"),
    ),
    readers: [reader, reader],
  };
}

// --- tokens / cache -------------------------------------------------------------

/** `ctxdiff tokens [--turn N] [--project P] [--session S] [--agent A]`. */
async function cmdTokens(rest: string[]): Promise<number> {
  const args = parseCommon(rest, {
    extra: { "context-window": { type: "string" } },
    numeric: ["context-window"],
  });
  // Resolved BEFORE any store is opened (same order Python uses), so a bad
  // `CTXDIFF_CONTEXT_WINDOW` fails identically with or without a trace present.
  let contextWindow: number | null;
  try {
    contextWindow = resolveWindow(args.values);
  } catch (err) {
    if (err instanceof SelectionError) return reportSelectionError(err);
    throw err;
  }
  // --session/--agent are single here: last value wins, matching argparse
  // without action="append".
  const session = args.sessions.length ? args.sessions[args.sessions.length - 1] : null;
  const agent = args.agents.length ? args.agents[args.agents.length - 1] : null;
  let opened;
  try {
    opened = await openSession(args.project, session);
  } catch (err) {
    return reportCommandFailure(err);
  }
  const { reader: ct, sessionId, handle } = opened;
  try {
    try {
      checkAgentFilter(ct, sessionId, agent, handle.discovered);
    } catch (err) {
      return reportCommandFailure(err);
    }
    // tokens' --turn is single: last value wins (matching argparse without append).
    const turn = args.turns.length ? args.turns[args.turns.length - 1] : null;
    const runTokens = analyzeRun(ct, agent);
    const bySeq = new Map(runTokens.calls.map((c) => [c.seq, c]));

    let selected;
    if (turn !== null) {
      if (!bySeq.has(turn.value)) {
        const where = agent !== null ? `agent '${agent}'` : "this run";
        const available = [...bySeq.keys()].sort((a, b) => a - b);
        process.stderr.write(
          `ctxdiff: turn ${turn.text} not found in ${where} ` +
            `(available turns: ${pyIntList(available)})\n`,
        );
        return 1;
      }
      selected = [bySeq.get(turn.value)!];
    } else {
      selected = runTokens.calls;
    }

    let totalTools: number | null = null;
    if (runTokens.bloat !== null && runTokens.bloat.unusedTools.length) {
      const allBlocks = ct.getCalls().map((c) => ct.getCallBlocks(c.id));
      totalTools = registeredToolNames(allBlocks).size;
    }

    const usageSummary = renderUsageSummary(runTokens.usage, agent);
    const agentSummary = renderAgentSummary(runTokens);
    // The detector always runs over the whole session — an eviction is a fact
    // about a PAIR of turns, so it cannot be computed from one turn's blocks —
    // but under `--turn N` the report shown beside it covers exactly that turn,
    // and a stanza saying "…was evicted at turn 3" under a turn-1 report names
    // something the reader did not ask about and cannot see above it. So the
    // events are filtered to the turn that LOST the block.
    const allEvictions = analyzeEvictions(ct, agent);
    const evictions =
      turn === null
        ? allEvictions
        : {
            ...allEvictions,
            evictions: allEvictions.evictions.filter((e) => e.evictedSeq === turn.value),
          };
    process.stdout.write(
      renderRunTokens(
        selected,
        runTokens.bloat,
        totalTools,
        agentSummary,
        usageSummary,
        contextWindow,
        evictions,
      ) + "\n",
    );
    return 0;
  } finally {
    ct.close();
    await handle.close();
  }
}

/** `ctxdiff cache [--project P] [--session S] [--agent A]`. No `--turn`: the
 * profiler works on consecutive-call PAIRS, and a single turn has no pair to be
 * stable against. */
async function cmdCache(rest: string[]): Promise<number> {
  // No `--turn`: the profiler works on consecutive-call PAIRS, so Python's
  // subparser never registered one, and neither does this.
  const args = parseCommon(rest, { only: ["agent", "session"] });
  const session = args.sessions.length ? args.sessions[args.sessions.length - 1] : null;
  const agent = args.agents.length ? args.agents[args.agents.length - 1] : null;
  let opened;
  try {
    opened = await openSession(args.project, session);
  } catch (err) {
    return reportCommandFailure(err);
  }
  const { reader: ct, sessionId, handle } = opened;
  try {
    try {
      checkAgentFilter(ct, sessionId, agent, handle.discovered);
    } catch (err) {
      return reportCommandFailure(err);
    }
    process.stdout.write(renderCacheReport(analyzeCache(ct, agent)) + "\n");
    return 0;
  } finally {
    ct.close();
    await handle.close();
  }
}

// --- check ------------------------------------------------------------------------

// The assertion flags, in the order the "you asked for nothing" error lists
// them — the same order the report prints its rows in, so the message reads as
// a menu of what the command can do rather than an arbitrary list. Mirrors
// Python `_ASSERTION_FLAGS`.
const ASSERTION_FLAGS = [
  "--max-context",
  "--max-context-pct",
  "--require-stable-prefix",
  "--no-dead-schemas",
  "--no-tagged-eviction",
  "--max-growth",
  "--max-growth-pct",
];

/** Read one integer threshold flag, or null when it was not passed. Rejects
 * anything Python's narrowed `_turn_int` rejects, with the same wording, so a
 * typo fails the same way in both CLIs. */
function intFlag(values: Record<string, unknown>, name: string): number | null {
  const raw = values[name] as string | undefined;
  if (raw === undefined) return null;
  if (!INT_RE.test(raw)) {
    throw new UsageError(`ctxdiff: error: argument --${name}: invalid int value: '${raw}'`);
  }
  return parseInt(raw.trim(), 10);
}

/**
 * The context window this invocation should render percentages against:
 * `--context-window` if typed, else `CTXDIFF_CONTEXT_WINDOW`, else null. The ONE
 * resolution path for `tokens`, `check`, `view` and `export`, so no two commands
 * can disagree about the denominator on the same machine. Mirrors Python
 * `_resolve_window`.
 *
 * An unusable environment value is re-thrown as a `SelectionError` so it reports
 * as the usage error it is (exit 2, message printed verbatim) rather than as a
 * failure to read a trace — the variable is part of how the command was invoked,
 * not part of what it was pointed at.
 */
function resolveWindow(values: Record<string, unknown>): number | null {
  const flag = intFlag(values, "context-window");
  try {
    return resolveContextWindow(flag);
  } catch (err) {
    if (err instanceof ContextWindowError) throw new SelectionError(err.message);
    throw err;
  }
}

/** Read one percentage threshold flag, or null when it was not passed. Mirrors
 * Python's `_pct_float` narrowing (see `FLOAT_RE`). */
function pctFlag(values: Record<string, unknown>, name: string): number | null {
  const raw = values[name] as string | undefined;
  if (raw === undefined) return null;
  if (!FLOAT_RE.test(raw)) {
    throw new UsageError(`ctxdiff: error: argument --${name}: invalid float value: '${raw}'`);
  }
  return parseFloat(raw.trim());
}

/**
 * Turn `check`'s flags into a `Thresholds`, rejecting every combination that
 * cannot mean what it says. Throws `SelectionError` (→ exit 2, message printed
 * verbatim) — these are usage errors: the flags as given cannot be acted on.
 * Mirrors Python `_check_thresholds`, message for message.
 *
 * The four rules, and why each is an error rather than a shrug:
 *
 * 1. NO assertion at all is refused. `ctxdiff check` with no thresholds would
 *    otherwise exit 0 having verified nothing — a green tick in CI that means
 *    "nobody asked a question", which is the exact failure this command exists
 *    to prevent.
 * 2. `--max-context-pct` with NO window at all is refused: the percentage has no
 *    denominator. ctxdiff deliberately ships no model→window table (windows
 *    differ per model and per provider and change under you), so the window is
 *    the user's to supply — as `--context-window N` or as
 *    `CTXDIFF_CONTEXT_WINDOW`, resolved by the ONE `resolveWindow` path
 *    `ctxdiff tokens` and the dashboard use.
 * 3. The `--context-window` FLAG without `--max-context-pct` is refused too —
 *    nothing would consume it, and silently ignoring a flag someone typed is how
 *    a CI gate ends up asserting less than its author believes. Only the flag: an
 *    inherited `CTXDIFF_CONTEXT_WINDOW` is ambient configuration for the whole
 *    shell, and failing a `--max-context` check because the environment happens
 *    to know the window would be absurd.
 * 4. A limit outside its legal range is refused: the two absolute budgets and
 *    the percentage must be POSITIVE (a zero window is a division by zero),
 *    while the two growth limits may legitimately be zero — "this context must
 *    not grow at all" is a real thing to assert.
 */
function checkThresholds(values: Record<string, unknown>): Thresholds {
  // Built first so "was anything asked for?" is answered by the analyzer's own
  // `anyRequested` — the single definition of that question, rather than a
  // second copy here that could fall out of step the next time a flag is added.
  // The FLAG on its own is kept beside the RESOLVED window: rule 3 below is
  // about what the user typed, while every other use is about what was resolved.
  const contextWindowFlag = intFlag(values, "context-window");
  // Rule 4 for the WINDOW FLAG is stated here, before the shared resolver gets a
  // chance to state it in its own words. `resolveContextWindow` refuses a
  // non-positive flag too (so `tokens`/`view`/`export` inherit the rule), but its
  // message is the generic `ctxdiff:` one; every usage error `check` emits is
  // prefixed `ctxdiff check:`, which is what tells a CI log which step spoke.
  // Only the flag needs the early guard: an environment-supplied window is
  // already positive by the time `parseContextWindow` returns it.
  if (contextWindowFlag !== null && contextWindowFlag <= 0) {
    throw new SelectionError(
      `ctxdiff check: --context-window must be greater than 0 (got ${contextWindowFlag})`,
    );
  }
  const thresholds: Thresholds = {
    maxContext: intFlag(values, "max-context"),
    contextWindow: resolveWindow(values),
    maxContextPct: pctFlag(values, "max-context-pct"),
    requireStablePrefix: values["require-stable-prefix"] === true,
    noDeadSchemas: values["no-dead-schemas"] === true,
    noTaggedEviction: values["no-tagged-eviction"] === true,
    maxGrowth: intFlag(values, "max-growth"),
    maxGrowthPct: pctFlag(values, "max-growth-pct"),
  };
  const { maxContext, contextWindow, maxContextPct, maxGrowth, maxGrowthPct } = thresholds;

  if (!anyRequested(thresholds)) {
    throw new SelectionError(
      "ctxdiff check: nothing to assert — pass at least one of " + ASSERTION_FLAGS.join(", "),
    );
  }
  if (maxContextPct !== null && contextWindow === null) {
    throw new SelectionError(
      "ctxdiff check: --max-context-pct needs a denominator — pass " +
        "--context-window N (ctxdiff ships no model→window table by design, " +
        "so the window is yours to state)",
    );
  }
  if (contextWindowFlag !== null && maxContextPct === null) {
    throw new SelectionError(
      "ctxdiff check: --context-window is only used by --max-context-pct " +
        "— pass that too, or use --max-context N for an absolute budget",
    );
  }
  // Integer and percentage limits are reported back in their own spelling — a
  // bare integer for the token budgets, one decimal for the percentages, the
  // same way every percentage in the report itself is written.
  if (maxContext !== null && maxContext <= 0) {
    throw new SelectionError(`ctxdiff check: --max-context must be greater than 0 (got ${maxContext})`);
  }
  if (maxContextPct !== null && maxContextPct <= 0) {
    throw new SelectionError(
      `ctxdiff check: --max-context-pct must be greater than 0 (got ${maxContextPct.toFixed(1)})`,
    );
  }
  if (maxGrowth !== null && maxGrowth < 0) {
    throw new SelectionError(`ctxdiff check: --max-growth cannot be negative (got ${maxGrowth})`);
  }
  if (maxGrowthPct !== null && maxGrowthPct < 0) {
    throw new SelectionError(
      `ctxdiff check: --max-growth-pct cannot be negative (got ${maxGrowthPct.toFixed(1)})`,
    );
  }
  return thresholds;
}

/**
 * `ctxdiff check [assertions] [--project P] [--session S] [--agent A]`: assert a
 * context budget, and exit non-zero when it is blown so CI fails the build.
 *
 * Exit codes, matching the convention every other subcommand follows — **0**
 * every requested assertion passed; **1** at least one was violated (or the
 * trace could not be read, or the session held nothing to check); **2** a usage
 * error (an impossible flag combination, an unresolvable `--session`/`--agent`).
 *
 * Order matters: the flags are validated BEFORE any store is opened, so a
 * malformed check fails the same way whether or not a trace happens to exist.
 *
 * An EMPTY session (or an `--agent` slice with no calls) exits 1 rather than
 * reporting a table of vacuous passes: a check that examined zero turns has
 * proved nothing, and a CI gate that greens on "there was no trace" is worse
 * than no gate at all. Mirrors Python `_cmd_check`.
 */
async function cmdCheck(rest: string[]): Promise<number> {
  // No `--turn`: a budget is a property of a whole run, and a check scoped to a
  // single turn could not answer "did ANY turn blow the budget". Python's
  // subparser registers none either, so passing one is exit 2 in both.
  const args = parseCommon(rest, {
    only: ["agent", "session"],
    // Python's `check` subparser takes no positional, so a stray token here is
    // a usage error in both CLIs rather than a silently adopted project path.
    positional: false,
    extra: {
      "max-context": { type: "string" },
      "context-window": { type: "string" },
      "max-context-pct": { type: "string" },
      "require-stable-prefix": { type: "boolean" },
      "no-dead-schemas": { type: "boolean" },
      "no-tagged-eviction": { type: "boolean" },
      "max-growth": { type: "string" },
      "max-growth-pct": { type: "string" },
    },
    numeric: [
      "max-context",
      "context-window",
      "max-context-pct",
      "max-growth",
      "max-growth-pct",
    ],
  });

  let thresholds: Thresholds;
  try {
    thresholds = checkThresholds(args.values);
  } catch (err) {
    if (err instanceof SelectionError) return reportSelectionError(err);
    throw err; // a UsageError: `main` turns it into exit 2 with its own message
  }

  const session = args.sessions.length ? args.sessions[args.sessions.length - 1] : null;
  const agent = args.agents.length ? args.agents[args.agents.length - 1] : null;
  let opened;
  try {
    opened = await openSession(args.project, session);
  } catch (err) {
    return reportCommandFailure(err);
  }
  const { reader: ct, sessionId, handle } = opened;
  try {
    try {
      checkAgentFilter(ct, sessionId, agent, handle.discovered);
    } catch (err) {
      return reportCommandFailure(err);
    }
    const report = analyzeCheck(ct, thresholds, agent);
    if (report.turnsAnalyzed === 0) {
      const where = agent !== null ? `agent '${agent}'` : "this session";
      process.stderr.write(
        `ctxdiff check: no calls in ${where} — nothing to check (did the run capture?)\n`,
      );
      return 1;
    }
    // The header names the session (and, when the project was DISCOVERED rather
    // than named, the file it came from) with the same `sessionWhere` spelling
    // every selector error uses — so a CI log records which trace produced the
    // verdict, and the newest-.ctrace default can never green a build on a file
    // nobody meant to check.
    process.stdout.write(
      renderCheckReport(report, sessionWhere(sessionId, handle.discovered)) + "\n",
    );
    return checkPassed(report) ? 0 : 1;
  } finally {
    ct.close();
    await handle.close();
  }
}

// --- discovery: sessions / agents ------------------------------------------------

const EMPTY_CWD = "no .ctrace files in the current directory";
const EMPTY_PROJECT = "no sessions in this project";
const EMPTY_STORE = "no sessions in the configured store";

/** Turn one `.ctrace` file's sessions into `sessions`-table rows. The label is
 * the bare filename when the file holds ONE session — the overwhelmingly common
 * case, and what a user recognizes — and `<filename>#<short id>` when it holds
 * several, so each row names something that can actually be selected
 * (`--project <filename> --session <short id>`). Mirrors Python
 * `_file_session_rows`. */
function fileSessionRows(filename: string, sessions: Session[]): SessionRow[] {
  const multi = sessions.length > 1;
  return sessions.map((s) => ({
    label: multi ? `${filename}#${shortId(s.id)}` : filename,
    started: formatLocal(s.startedAt),
    project: s.project,
    provider: s.provider,
    turns: s.turnCount,
    agents: agentsText(s.agents),
  }));
}

/** Every session in `backend`, with the handle always released. An empty store is
 * reported as an empty list rather than an error — a database that exists but
 * holds nothing yet is a perfectly normal state, and the caller prints an empty
 * listing for it exactly as it would for an empty directory. */
async function readSessions(backend: StoreBackend): Promise<Session[]> {
  let reader: Store;
  try {
    reader = await backend.openReader();
  } catch (err) {
    if (err instanceof EmptyStoreError) return [];
    throw err;
  }
  try {
    return await reader.listSessions();
  } finally {
    await reader.close();
  }
}

/**
 * `ctxdiff sessions` (and its hidden `runs` alias): the answer to "what can I
 * pass to --session?".
 *
 * Where it looks mirrors every other command with ONE deliberate exception: with
 * no `--project` and nothing configured, it lists every `*.ctrace` in the working
 * directory rather than narrowing to the newest. Discovery is the one job where
 * narrowing would defeat the purpose — you cannot pick a project you were never
 * shown. A file that fails to open is skipped rather than aborting the listing.
 */
async function cmdSessions(rest: string[]): Promise<number> {
  // Discovery takes `--project`/`--run` and nothing else: this command answers
  // "what can I pass to --session?", so accepting `--session` would be circular
  // and accepting `--agent` would promise a filter it does not apply.
  const args = parseCommon(rest, { only: [] });
  let backend: StoreBackend | null;
  try {
    backend = discoveryBackend(args.project);
  } catch (err) {
    // A bad DSN is reported, not crashed.
    return reportOpenFailure(err);
  }

  if (backend === null) {
    const rows: SessionRow[] = [];
    for (const f of listFileSessions(process.cwd())) {
      rows.push(...fileSessionRows(f.filename, f.sessions));
    }
    process.stdout.write(renderSessionsList(rows, EMPTY_CWD) + "\n");
    return 0;
  }

  const empty = args.project ? EMPTY_PROJECT : EMPTY_STORE;
  let sessions: Session[];
  try {
    sessions = await readSessions(backend);
  } catch (err) {
    return reportOpenFailure(err);
  }

  const path = isFileBackend(backend) ? (backend as SQLiteStore).path : null;
  const rows: SessionRow[] = path
    ? fileSessionRows(basename(path), sessions)
    : // A networked store has no filenames at all, so the short session id is
      // the only label there is — and it is exactly what `--session` takes.
      sessions.map((s) => ({
        label: shortId(s.id),
        started: formatLocal(s.startedAt),
        project: s.project,
        provider: s.provider,
        turns: s.turnCount,
        agents: agentsText(s.agents),
      }));
  process.stdout.write(renderSessionsList(rows, empty) + "\n");
  return 0;
}

/**
 * Aggregate per-session call lists into the `agents` table's rows: for each
 * agent, how many SESSIONS it appears in, how many calls it made in total, and
 * its provider-reported token spend.
 *
 * How: one pass over every session's calls, bucketing by agent name (calls with
 * no label share the same `(unlabeled)` bucket the token report uses) in
 * first-appearance order, counting a session once per agent it contains. The
 * token column is delegated to `usageTotals`, the same rollup `ctxdiff tokens`
 * prints, so the two commands can never report different numbers for the same
 * calls — and it renders '-' rather than 0 when NO call of that agent reported
 * usage, because 0 would read as "this agent was free". Mirrors Python
 * `_agent_rows`.
 */
function agentRows(sessionsCalls: Call[][]): AgentRow[] {
  const order: string[] = [];
  const callsByAgent = new Map<string, Call[]>();
  const sessionCounts = new Map<string, number>();
  for (const calls of sessionsCalls) {
    const seenHere = new Set<string>();
    for (const c of calls) {
      const name = c.agent ? c.agent : UNLABELED;
      if (!callsByAgent.has(name)) {
        order.push(name);
        callsByAgent.set(name, []);
      }
      callsByAgent.get(name)!.push(c);
      if (!seenHere.has(name)) {
        seenHere.add(name);
        sessionCounts.set(name, (sessionCounts.get(name) ?? 0) + 1);
      }
    }
  }
  return order.map((name) => {
    const calls = callsByAgent.get(name)!;
    const totals = usageTotals(calls);
    return {
      name,
      sessions: sessionCounts.get(name)!,
      calls: calls.length,
      tokens: totals.callsWithUsage
        ? comma(totals.inputTokens + totals.outputTokens)
        : "-",
    };
  });
}

/**
 * `ctxdiff agents`: every agent in the project with its footprint aggregated
 * ACROSS all sessions.
 *
 * Why aggregated: "how much does the researcher cost" is a question about the
 * project, not about whichever run happens to be newest — a per-session answer
 * would change every time the user re-ran their agent. Project resolution is the
 * same as `sessions`', including the cwd-wide scan when nothing is configured, so
 * the two discovery commands always describe the same set of traces.
 */
async function cmdAgents(rest: string[]): Promise<number> {
  // `--project`/`--run` only, exactly as Python's subparser registers it: the
  // rollup is over EVERY session and EVERY agent by definition, so a selector
  // here could only ever be ignored.
  const args = parseCommon(rest, { only: [] });
  let backend: StoreBackend | null;
  try {
    backend = discoveryBackend(args.project);
  } catch (err) {
    return reportOpenFailure(err);
  }

  let sessionsCalls: Call[][] = [];
  try {
    if (backend === null) {
      sessionsCalls = listFileCalls(process.cwd());
    } else {
      let reader: Store | null;
      try {
        reader = await backend.openReader();
      } catch (err) {
        if (err instanceof EmptyStoreError) reader = null;
        else throw err;
      }
      if (reader !== null) {
        try {
          for (const s of await reader.listSessions()) {
            sessionsCalls.push(await reader.getCalls(s.id));
          }
        } finally {
          await reader.close();
        }
      }
    }
  } catch (err) {
    return reportOpenFailure(err);
  }

  process.stdout.write(renderAgentsList(agentRows(sessionsCalls)) + "\n");
  return 0;
}

// --- viewer commands -------------------------------------------------------------

/** Open `filePath` in the default browser via the OS opener (`open` on macOS,
 * `start` on Windows, `xdg-open` on Linux). Best-effort and fully guarded: a
 * failing/absent browser NEVER crashes the command — the path is already
 * printed, so the user can open it themselves. Mirrors Python's webbrowser
 * wrapping. */
function openInBrowser(filePath: string): void {
  try {
    const url = "file://" + filePath;
    let cmd: string;
    let args: string[];
    if (platform === "darwin") {
      cmd = "open";
      args = [url];
    } else if (platform === "win32") {
      cmd = "cmd";
      args = ["/c", "start", "", url];
    } else {
      cmd = "xdg-open";
      args = [url];
    }
    const child = spawn(cmd, args, { stdio: "ignore", detached: true });
    child.on("error", () => {
      /* no browser available — already printed the path */
    });
    child.unref();
  } catch {
    /* never let a browser launch crash the command */
  }
}

/** Parse the viewer commands' flags via parseArgs, converting a parse failure
 * into a UsageError (→ exit 2). A leading positional is the `.ctrace` path. */
function parseViewerArgs(
  rest: string[],
  options: Record<string, { type: "string" | "boolean" }>,
): { values: Record<string, unknown>; positionals: string[] } {
  try {
    // `--context-window` is the one numeric flag the viewer commands take, and a
    // negative value must reach the resolver's "must be greater than 0" message
    // rather than die in the parser as "expected one argument" — Python's
    // argparse accepts `-5` as a VALUE via its negative-number heuristic, so
    // without this the two CLIs answer the same typo differently. Same helper,
    // same reason, as the analysis commands (see `joinNegativeNumberValues`).
    const args = joinNegativeNumberValues(rest, ["--context-window"]);
    const { values, positionals } = parseArgs({ args, options, allowPositionals: true });
    return { values: values as Record<string, unknown>, positionals: positionals as string[] };
  } catch (err) {
    throw usageErrorFromParse(err);
  }
}

/** The `--project` / `--run` / positional trio every viewer command accepts,
 * resolved in the same explicit-beats-ambient order the analysis commands use. */
function viewerProject(values: Record<string, unknown>, positionals: string[]): string | undefined {
  return (
    (values.project as string | undefined) ??
    (values.run as string | undefined) ??
    (positionals[0] as string | undefined)
  );
}

/** `ctxdiff export [--project P] [--session S] [--agent A] [--out FILE.html]`:
 * write the project's self-contained three-level HTML dashboard beside the trace
 * (or to `--out`) and print the path. */
async function cmdExport(rest: string[]): Promise<number> {
  const { values, positionals } = parseViewerArgs(rest, {
    project: { type: "string" },
    run: { type: "string" },
    session: { type: "string" },
    agent: { type: "string" },
    "context-window": { type: "string" },
    out: { type: "string" },
  });
  try {
    const out = await writeDashboard(
      viewerProject(values, positionals),
      (values.session as string | undefined) ?? null,
      (values.agent as string | undefined) ?? null,
      values.out as string | undefined,
      resolveWindow(values),
    );
    process.stdout.write(out + "\n");
    return 0;
  } catch (err) {
    return reportCommandFailure(err);
  }
}

/**
 * Render the three-level dashboard for whichever project `openDashboard`
 * resolves — focused on one session, optionally scoped to one agent — and return
 * the path written. `out` wins; otherwise the trace's own `<stem>.html` is used,
 * and a store with no file (Postgres/MySQL) says so rather than inventing a
 * filename. Shared by `export` and `view` so both reach a database identically.
 */
async function writeDashboard(
  project: string | undefined,
  session: string | null,
  agent: string | null,
  out: string | undefined,
  contextWindow: number | null = null,
): Promise<string> {
  const { reader, htmlDefault, sessionSelected, handle } = await openDashboard(
    project,
    session,
    agent,
  );
  try {
    const target = out ?? htmlDefault;
    if (target === null) {
      throw new Error(
        "ctxdiff: the configured store has no file to name the dashboard after " +
          "— pass --out FILE.html",
      );
    }
    return exportStore(reader, target, { agent, sessionSelected, contextWindow });
  } finally {
    reader.close();
    await handle.close();
  }
}

/** `ctxdiff view [--project P] [--session S] [--agent A] [--no-open]`: export
 * the dashboard to a temp file, print its path, and open it in the browser
 * unless `--no-open`. */
async function cmdView(rest: string[]): Promise<number> {
  const { values, positionals } = parseViewerArgs(rest, {
    project: { type: "string" },
    run: { type: "string" },
    session: { type: "string" },
    agent: { type: "string" },
    "context-window": { type: "string" },
    "no-open": { type: "boolean" },
  });
  const tmp = join(tmpdir(), `ctxdiff-${randomUUID()}.html`);
  let out: string;
  try {
    out = await writeDashboard(
      viewerProject(values, positionals),
      (values.session as string | undefined) ?? null,
      (values.agent as string | undefined) ?? null,
      tmp,
      resolveWindow(values),
    );
  } catch (err) {
    return reportCommandFailure(err);
  }
  process.stdout.write(out + "\n");
  if (!values["no-open"]) openInBrowser(out);
  return 0;
}

/** `ctxdiff demo [--out FILE] [--no-open] [--keep]`: build a sample multi-agent
 * trace (no API keys, no network) and open its dashboard. Placement: `--out`
 * writes there (+ its `.html` sibling); `--keep` writes `./ctxdiff-demo.{ctrace,
 * html}`; otherwise tempfiles. Mirrors Python `_cmd_demo`. */
function cmdDemo(rest: string[]): number {
  const { values } = parseViewerArgs(rest, {
    out: { type: "string" },
    "no-open": { type: "boolean" },
    keep: { type: "boolean" },
  });
  let ctracePath: string;
  let htmlPath: string;
  if (values.out) {
    ctracePath = values.out as string;
    htmlPath = ctracePath.replace(/\.[^.]*$/, "") + ".html";
  } else if (values.keep) {
    ctracePath = join(process.cwd(), "ctxdiff-demo.ctrace");
    htmlPath = join(process.cwd(), "ctxdiff-demo.html");
  } else {
    const id = randomUUID();
    ctracePath = join(tmpdir(), `ctxdiff-${id}.ctrace`);
    htmlPath = join(tmpdir(), `ctxdiff-${id}.html`);
  }

  let out: string;
  try {
    buildDemoTrace(ctracePath);
    out = exportHtml(ctracePath, htmlPath);
  } catch (err) {
    process.stderr.write(`ctxdiff: ${(err as Error).message}\n`);
    return 1;
  }

  process.stdout.write(`sample trace  -> ${ctracePath}\n`);
  process.stdout.write(`dashboard     -> ${out}\n`);
  process.stdout.write(
    "This is a sample multi-agent research-pipeline run (no API keys, " +
      "no network) — it shows turn-by-turn diffs, token/schema-bloat " +
      "detection, a cache-prefix break, and two agents on one timeline.\n",
  );
  if (!values["no-open"]) openInBrowser(out);
  process.stdout.write(
    'Trace your own agent next: const tracer = trace.init("my-agent"); ' +
      "const client = tracer.wrap(new OpenAI());\n",
  );
  return 0;
}

// `runs` is deliberately absent: it still dispatches (see `main`), it just isn't
// offered to anyone reading the usage for the first time.
const USAGE =
  "usage: ctxdiff <command> [options]\n" +
  "\n" +
  "commands:\n" +
  "  diff --turn N --turn M   git-style block diff between two turns\n" +
  "  tokens [--turn N]        token heatmap + schema-bloat report\n" +
  "  cache                    prefix-stability report + wasted-spend estimate\n" +
  "  check [assertions]       assert context budgets and fail the build (CI gate)\n" +
  "  sessions                 list the sessions ctxdiff can see\n" +
  "  agents                   list every agent in the project, across all sessions\n" +
  "  view                     open a self-contained HTML dashboard in your browser\n" +
  "  export [--out FILE]      write a self-contained HTML dashboard for the project\n" +
  "  demo                     build a sample dashboard — no API keys, no setup\n" +
  "\n" +
  "common options: [--project PATH|DSN] [--session ID] [--agent NAME]\n";

/** Dispatch on the first argument (the command); return an exit code. With no
 * command, print usage and return 2 (matching the Python CLI's convention). */
export async function main(argv: string[]): Promise<number> {
  const command = argv[0];
  const rest = argv.slice(1);
  try {
    switch (command) {
      case "diff":
        return await cmdDiff(rest);
      case "tokens":
        return await cmdTokens(rest);
      case "cache":
        return await cmdCache(rest);
      case "check":
        return await cmdCheck(rest);
      case "sessions":
      // `runs` is the HIDDEN alias of `sessions`. The command was renamed once a
      // `.ctrace` became a project holding many runs, but every existing script,
      // README and muscle memory says `runs` — so it keeps working, forever.
      case "runs":
        return await cmdSessions(rest);
      case "agents":
        return await cmdAgents(rest);
      case "export":
        return await cmdExport(rest);
      case "view":
        return await cmdView(rest);
      case "demo":
        return cmdDemo(rest);
      default:
        process.stdout.write(USAGE);
        return 2;
    }
  } catch (err) {
    // A UsageError (bad flags/values) → Python's argparse exit code 2 + a
    // one-line stderr message. Any other unexpected throw is reported the same
    // way rather than leaking a stack trace to the user.
    if (err instanceof UsageError) {
      process.stderr.write(err.message + "\n");
      return 2;
    }
    process.stderr.write(`ctxdiff: error: ${(err as Error).message}\n`);
    return 2;
  }
}

// Direct execution: run and exit with the command's code. (Importing this module
// — e.g. from tests — does not trigger this.)
if (
  process.argv[1] &&
  (process.argv[1].endsWith("cli.js") || process.argv[1].endsWith("cli.cjs"))
) {
  // `main` is async now (a configured database is read over the network), so the
  // exit is deferred to its resolution. The catch is the last fail-safe: an
  // unexpected rejection prints one line and exits 2 rather than surfacing an
  // unhandled-rejection stack trace to the user.
  main(process.argv.slice(2)).then(
    (code) => process.exit(code),
    (err: unknown) => {
      process.stderr.write(`ctxdiff: error: ${(err as Error).message}\n`);
      process.exit(2);
    },
  );
}
