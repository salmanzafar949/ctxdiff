#!/usr/bin/env node
/**
 * ctxdiff's command-line entry point (`npx ctxdiff <cmd>`). Matches the Python
 * CLI's command surface — `diff`, `tokens`, `cache`, `runs` (read-only
 * analysis) plus `view`, `export`, `demo` (the HTML dashboard) — with the same
 * flags (`--turn`, `--agent`, `--run`, `--out`, `--no-open`, `--keep`), output,
 * and exit codes. Uses Node's built-in `util.parseArgs`; no CLI framework
 * dependency. As an ergonomic extra over the Python CLI, a positional `.ctrace`
 * path is accepted as an alias for `--run`.
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
  type ReadableStore,
  type Session,
  type StoreBackend,
} from "./store/base.js";
import { resolve as resolveConfigured } from "./store/config.js";
import { SQLiteStore } from "./store/sqlite.js";
import { snapshotStore } from "./store/snapshot.js";
import { diffTurns } from "./analyze/diff.js";
import { analyzeRun, registeredToolNames } from "./analyze/tokens.js";
import { analyzeCache } from "./analyze/cache.js";
import { listRuns } from "./analyze/runs.js";
import { exportHtml, exportStore } from "./viewer/export.js";
import { buildDemoTrace } from "./demo.js";
import {
  renderTurnDiff,
  renderRunTokens,
  renderCacheReport,
  renderRunsList,
  renderUsageSummary,
  renderAgentSummary,
} from "./render.js";

/** Most recently modified `*.ctrace` in `dir`, or null. Backs the `--run`
 * default so the common case (one run in the cwd) needs no flag. Mirrors Python
 * `_find_default_run`. */
function findDefaultRun(dir: string): string | null {
  let entries: string[];
  try {
    entries = readdirSync(dir);
  } catch {
    return null;
  }
  const candidates = entries.filter((f) => f.endsWith(".ctrace")).map((f) => join(dir, f));
  if (!candidates.length) return null;
  return candidates.reduce((best, p) =>
    statSync(p).mtimeMs > statSync(best).mtimeMs ? p : best,
  );
}

/**
 * A trace opened for READING, whichever backend it came from: a `CTrace` on a
 * local file, or an in-memory snapshot of one session of a Postgres/MySQL store.
 * Both are synchronous and both close the same way, so every command below reads
 * one of these without knowing which it got.
 *
 * Deliberately just ONE session's worth of reading plus `close`. `ctxdiff runs`
 * is the only command that wants a store's whole session list, and it takes its
 * own path to it (`runsFromStore`) rather than snapshotting a run it does not
 * need — see `store/snapshot.ts` for why that listing is not free.
 */
interface Reader extends ReadableStore {
  close(): void;
}

/** No trace to read: no `--run`, no configured backend, and no `*.ctrace` in the
 * working directory. Its own type so callers can print the friendly "did the run
 * capture?" line for this case while still reporting a genuine open FAILURE
 * (corrupt file, unreachable database) as an error. Mirrors Python
 * `_NoTraceFound`. */
class NoTraceFound extends Error {}

/**
 * Open the store the read commands should analyze, and report the default HTML
 * output path that goes with it (null when there is no file to derive one from —
 * a networked store).
 *
 * Resolution mirrors the write side's explicit-beats-ambient rule:
 * 1. `--run PATH` (or the positional) — always that file. A path names a file,
 *    so it wins even when a database is configured.
 * 2. A configured NETWORKED backend (`configure()` / `CTXDIFF_STORE`) — read its
 *    newest session, materialized into a snapshot so the synchronous analyzers
 *    can consume it (see `store/snapshot.ts`). Detected by the ABSENCE of
 *    `pathFor`, the same file-backend capability check `Tracer` uses, rather than
 *    an instanceof chain against every backend class.
 * 3. A configured SQLite backend pointing at a concrete file — that file, opened
 *    THROUGH the backend so the read side goes via the same protocol the write
 *    side does.
 * 4. Nothing configured — the most recently modified `*.ctrace` in the cwd,
 *    exactly as before.
 *
 * Throws `NoTraceFound` when nothing resolves, and lets a real open error
 * (corrupt file, dead database, missing driver) propagate with its own message.
 */
async function openSource(
  explicit: string | undefined,
): Promise<{ reader: Reader; htmlDefault: string | null }> {
  if (explicit) return { reader: CTrace.open(explicit), htmlDefault: defaultHtmlFor(explicit) };

  const backend = resolveConfigured();
  if (backend !== null && !isFileBackend(backend)) {
    return { reader: await snapshotStore(await backend.openReader()), htmlDefault: null };
  }
  if (backend !== null) {
    const configuredPath = (backend as SQLiteStore).path;
    if (configuredPath && !isDirectory(configuredPath)) {
      return {
        reader: backend.openReader() as CTrace,
        htmlDefault: defaultHtmlFor(configuredPath),
      };
    }
  }

  const path = findDefaultRun(process.cwd());
  if (path === null) throw new NoTraceFound();
  return { reader: CTrace.open(path), htmlDefault: defaultHtmlFor(path) };
}

/** Whether `path` is an existing directory — used to tell a configured
 * `SQLiteStore` that names ONE file from one that names a directory of them. */
function isDirectory(path: string): boolean {
  const st = statSync(path, { throwIfNoEntry: false });
  return st !== undefined && st.isDirectory();
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
 * Print the right message for a failed `openSource` and return the exit code, so
 * all five read commands report identically: the friendly "did the run capture?"
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

interface ParsedArgs {
  turns: number[];
  agent: string | null;
  run: string | undefined;
}

/** A usage error (bad flags/values). Carries the one-line `ctxdiff: error: …`
 * message to print to stderr; `main` turns it into exit code 2, matching the
 * Python CLI's argparse convention. */
class UsageError extends Error {}

/** Format an int array like Python's list repr: `[1, 2, 3]` (space after each
 * comma), unlike `JSON.stringify` which omits the spaces. */
function pyIntList(arr: number[]): string {
  return "[" + arr.join(", ") + "]";
}

/** Python `int(str)` acceptance: optional surrounding whitespace, optional
 * sign, then decimal digits. Rejects "abc", "1.5", "", "0x10" — matching
 * argparse's `type=int`. */
const INT_RE = /^\s*[+-]?\d+\s*$/;

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

/** Parse the flags common to the analysis commands. `--turn` is `multiple` so
 * `diff` can pass it twice; the callers apply their own count rules. A leading
 * positional is treated as the `.ctrace` path (alias for `--run`). Throws
 * `UsageError` (→ exit 2) on an unknown flag, a missing option value, or a
 * non-integer `--turn`, matching the Python CLI's exit code and message text. */
function parseCommon(rest: string[]): ParsedArgs {
  let parsed;
  try {
    parsed = parseArgs({
      args: rest,
      options: {
        turn: { type: "string", multiple: true },
        agent: { type: "string" },
        run: { type: "string" },
      },
      allowPositionals: true,
    });
  } catch (err) {
    throw usageErrorFromParse(err);
  }
  const rawTurns = (parsed.values.turn as string[] | undefined) ?? [];
  const turns: number[] = [];
  for (const raw of rawTurns) {
    if (!INT_RE.test(raw)) {
      throw new UsageError(`ctxdiff: error: argument --turn: invalid int value: '${raw}'`);
    }
    turns.push(Number.parseInt(raw.trim(), 10));
  }
  return {
    turns,
    agent: (parsed.values.agent as string | undefined) ?? null,
    run: (parsed.values.run as string | undefined) ?? (parsed.positionals[0] as string | undefined),
  };
}

/** `ctxdiff diff --turn N --turn M [--agent A] [--run PATH]`. */
async function cmdDiff(rest: string[]): Promise<number> {
  const args = parseCommon(rest);
  if (args.turns.length !== 2) {
    process.stderr.write(
      "ctxdiff diff requires exactly two --turn flags, e.g. " +
        "'ctxdiff diff --turn 7 --turn 8'\n",
    );
    return 2;
  }
  let ct: Reader;
  try {
    ({ reader: ct } = await openSource(args.run));
  } catch (err) {
    return reportOpenFailure(err);
  }
  try {
    const [turnOld, turnNew] = args.turns;
    if (args.agent !== null) {
      const agentSeqs = ct.getCalls().filter((c) => c.agent === args.agent).map((c) => c.seq);
      const bad = [turnOld, turnNew].filter((t) => !agentSeqs.includes(t));
      if (bad.length) {
        process.stderr.write(
          `ctxdiff: turn(s) ${pyIntList(bad)} are not calls of agent ` +
            `'${args.agent}' (that agent's turns: ${pyIntList(agentSeqs)})\n`,
        );
        return 1;
      }
    }
    let diff;
    try {
      diff = diffTurns(ct, turnOld, turnNew);
    } catch (err) {
      process.stderr.write(`ctxdiff: ${(err as Error).message}\n`);
      return 1;
    }
    process.stdout.write(renderTurnDiff(diff) + "\n");
    return 0;
  } finally {
    ct.close();
  }
}

/** `ctxdiff tokens [--turn N] [--agent A] [--run PATH]`. */
async function cmdTokens(rest: string[]): Promise<number> {
  const args = parseCommon(rest);
  let ct: Reader;
  try {
    ({ reader: ct } = await openSource(args.run));
  } catch (err) {
    return reportOpenFailure(err);
  }
  try {
    // tokens' --turn is single: last value wins (matching argparse without append).
    const turn = args.turns.length ? args.turns[args.turns.length - 1] : null;
    const runTokens = analyzeRun(ct, args.agent);
    const bySeq = new Map(runTokens.calls.map((c) => [c.seq, c]));

    let selected;
    if (turn !== null) {
      if (!bySeq.has(turn)) {
        const where = args.agent !== null ? `agent '${args.agent}'` : "this run";
        const available = [...bySeq.keys()].sort((a, b) => a - b);
        process.stderr.write(
          `ctxdiff: turn ${turn} not found in ${where} ` +
            `(available turns: ${pyIntList(available)})\n`,
        );
        return 1;
      }
      selected = [bySeq.get(turn)!];
    } else {
      selected = runTokens.calls;
    }

    let totalTools: number | null = null;
    if (runTokens.bloat !== null && runTokens.bloat.unusedTools.length) {
      const allBlocks = ct.getCalls().map((c) => ct.getCallBlocks(c.id));
      totalTools = registeredToolNames(allBlocks).size;
    }

    const usageSummary = renderUsageSummary(runTokens.usage, args.agent);
    const agentSummary = renderAgentSummary(runTokens);
    process.stdout.write(
      renderRunTokens(selected, runTokens.bloat, totalTools, agentSummary, usageSummary) + "\n",
    );
    return 0;
  } finally {
    ct.close();
  }
}

/** `ctxdiff cache [--agent A] [--run PATH]`. */
async function cmdCache(rest: string[]): Promise<number> {
  const args = parseCommon(rest);
  let ct: Reader;
  try {
    ({ reader: ct } = await openSource(args.run));
  } catch (err) {
    return reportOpenFailure(err);
  }
  try {
    process.stdout.write(renderCacheReport(analyzeCache(ct, args.agent)) + "\n");
    return 0;
  } finally {
    ct.close();
  }
}

/**
 * `ctxdiff runs`: list the sessions ctxdiff can see. No flags (matches Python).
 *
 * Where it looks mirrors `openSource` (and therefore every other read command):
 * a configured NETWORKED backend, or a configured SQLite backend naming ONE
 * concrete file, is listed session-by-session through the store protocol; with
 * nothing configured it falls back to listing every `*.ctrace` in the cwd. Why
 * that matters: with `CTXDIFF_STORE` set, a `runs` that globbed the working
 * directory would answer "no .ctrace files" about a machine whose traces all
 * live in Postgres.
 */
async function cmdRuns(): Promise<number> {
  try {
    const backend = resolveConfigured();
    if (backend !== null) {
      const configuredPath = isFileBackend(backend) ? (backend as SQLiteStore).path : null;
      if (!isFileBackend(backend) || (configuredPath && !isDirectory(configuredPath))) {
        return await runsFromStore(backend);
      }
    }
  } catch (err) {
    // A bad DSN or a dead database is reported, not crashed.
    return reportOpenFailure(err);
  }
  process.stdout.write(renderRunsList(listRuns(process.cwd())) + "\n");
  return 0;
}

/**
 * List `ctxdiff runs` out of a CONFIGURED store rather than the cwd: one row per
 * SESSION (oldest first) keyed by a short session id instead of a filename,
 * since a store holds many sessions and no filenames at all. Turn counts and
 * agent lists come straight off the `Session` summary, so this is one query set
 * rather than an open-per-file. An EMPTY store is not an error — it prints an
 * empty listing and exits 0, exactly as an empty directory does.
 */
async function runsFromStore(backend: StoreBackend): Promise<number> {
  const empty = "no sessions in the configured store";
  let sessions: Session[];
  try {
    // Straight through the store protocol rather than via `openSource`: a
    // listing needs the session SUMMARIES only, and snapshotting would drag the
    // newest session's every call and block across the network to print one line
    // per session.
    const reader = await backend.openReader();
    try {
      sessions = await reader.listSessions();
    } finally {
      await reader.close();
    }
  } catch (err) {
    if (err instanceof EmptyStoreError) {
      process.stdout.write(renderRunsList([], empty) + "\n");
      return 0;
    }
    throw err;
  }
  const rows = sessions.map((s) => ({
    filename: s.id.slice(0, 12),
    project: s.project,
    provider: s.provider,
    turns: s.turnCount,
    agents: s.agents.length ? s.agents.join(", ") : "-",
  }));
  process.stdout.write(renderRunsList(rows, empty) + "\n");
  return 0;
}

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
    const { values, positionals } = parseArgs({ args: rest, options, allowPositionals: true });
    return { values: values as Record<string, unknown>, positionals: positionals as string[] };
  } catch (err) {
    throw usageErrorFromParse(err);
  }
}

/** `ctxdiff export [--run PATH] [--out FILE.html]`: write a self-contained HTML
 * dashboard beside the trace (or to `--out`) and print the path. */
async function cmdExport(rest: string[]): Promise<number> {
  const { values, positionals } = parseViewerArgs(rest, {
    run: { type: "string" },
    out: { type: "string" },
  });
  const explicit = (values.run as string | undefined) ?? (positionals[0] as string | undefined);
  try {
    const out = await writeDashboard(explicit, values.out as string | undefined);
    process.stdout.write(out + "\n");
    return 0;
  } catch (err) {
    return reportOpenFailure(err);
  }
}

/**
 * Render a dashboard for whichever trace `openSource` resolves and return the
 * path written. `out` wins; otherwise the trace's own `<stem>.html` is used, and
 * a store with no file (Postgres/MySQL) says so rather than inventing a
 * filename. Shared by `export` and `view` so both reach a database identically.
 */
async function writeDashboard(
  explicit: string | undefined,
  out: string | undefined,
): Promise<string> {
  const { reader, htmlDefault } = await openSource(explicit);
  try {
    const target = out ?? htmlDefault;
    if (target === null) {
      throw new Error(
        "ctxdiff: the configured store has no file to name the dashboard after " +
          "— pass --out FILE.html",
      );
    }
    return exportStore(reader, target);
  } finally {
    reader.close();
  }
}

/** `ctxdiff view [--run PATH] [--no-open]`: export the dashboard to a temp file,
 * print its path, and open it in the browser unless `--no-open`. */
async function cmdView(rest: string[]): Promise<number> {
  const { values, positionals } = parseViewerArgs(rest, {
    run: { type: "string" },
    "no-open": { type: "boolean" },
  });
  const explicit = (values.run as string | undefined) ?? (positionals[0] as string | undefined);
  const tmp = join(tmpdir(), `ctxdiff-${randomUUID()}.html`);
  let out: string;
  try {
    out = await writeDashboard(explicit, tmp);
  } catch (err) {
    return reportOpenFailure(err);
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

const USAGE =
  "usage: ctxdiff <command> [options]\n" +
  "\n" +
  "commands:\n" +
  "  diff --turn N --turn M   git-style block diff between two turns\n" +
  "  tokens [--turn N]        token heatmap + schema-bloat report\n" +
  "  cache                    prefix-stability report + wasted-spend estimate\n" +
  "  runs                     list .ctrace files in the working directory\n" +
  "  view                     open a self-contained HTML dashboard in your browser\n" +
  "  export [--out FILE]      write a self-contained HTML dashboard for a run\n" +
  "  demo                     build a sample dashboard — no API keys, no setup\n" +
  "\n" +
  "common options: [--agent A] [--run PATH | positional PATH]\n";

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
      case "runs":
        return await cmdRuns();
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
