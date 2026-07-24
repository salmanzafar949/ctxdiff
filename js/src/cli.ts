#!/usr/bin/env node
/**
 * ctxdiff's command-line entry point (`npx ctxdiff <cmd>`). Read-only analysis
 * commands only — `diff`, `tokens`, `cache`, `runs` — matching the Python CLI's
 * names, flags (`--turn`, `--agent`, `--run`), output, and exit codes. (view /
 * export / demo are a later phase.) Uses Node's built-in `util.parseArgs`; no
 * CLI framework dependency. As an ergonomic extra over the Python CLI, a
 * positional `.ctrace` path is accepted as an alias for `--run`.
 */
import { parseArgs } from "node:util";
import { readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { CTrace } from "./store/ctrace.js";
import { diffTurns } from "./analyze/diff.js";
import { analyzeRun, registeredToolNames } from "./analyze/tokens.js";
import { analyzeCache } from "./analyze/cache.js";
import { listRuns } from "./analyze/runs.js";
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

/** Resolve the `.ctrace` path to open: explicit value if given (positional or
 * `--run`), else the most recently modified `*.ctrace` in the cwd. Mirrors
 * Python `_resolve_run_path`. */
function resolveRunPath(explicit: string | undefined): string | null {
  if (explicit) return explicit;
  return findDefaultRun(process.cwd());
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

/** Open a `.ctrace`, printing the Python-style "no .ctrace here" / open-error
 * messages to stderr and returning null on failure. */
function openOrReport(explicit: string | undefined): CTrace | null {
  const path = resolveRunPath(explicit);
  if (path === null) {
    process.stderr.write("no .ctrace here — did the run capture?\n");
    return null;
  }
  try {
    return CTrace.open(path);
  } catch (err) {
    process.stderr.write(`ctxdiff: ${(err as Error).message}\n`);
    return null;
  }
}

/** `ctxdiff diff --turn N --turn M [--agent A] [--run PATH]`. */
function cmdDiff(rest: string[]): number {
  const args = parseCommon(rest);
  if (args.turns.length !== 2) {
    process.stderr.write(
      "ctxdiff diff requires exactly two --turn flags, e.g. " +
        "'ctxdiff diff --turn 7 --turn 8'\n",
    );
    return 2;
  }
  const ct = openOrReport(args.run);
  if (ct === null) return 1;
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
function cmdTokens(rest: string[]): number {
  const args = parseCommon(rest);
  const ct = openOrReport(args.run);
  if (ct === null) return 1;
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
function cmdCache(rest: string[]): number {
  const args = parseCommon(rest);
  const ct = openOrReport(args.run);
  if (ct === null) return 1;
  try {
    process.stdout.write(renderCacheReport(analyzeCache(ct, args.agent)) + "\n");
    return 0;
  } finally {
    ct.close();
  }
}

/** `ctxdiff runs`: list every `*.ctrace` in the cwd. No flags (matches Python). */
function cmdRuns(): number {
  process.stdout.write(renderRunsList(listRuns(process.cwd())) + "\n");
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
  "\n" +
  "common options: [--agent A] [--run PATH | positional PATH]\n";

/** Dispatch on the first argument (the command); return an exit code. With no
 * command, print usage and return 2 (matching the Python CLI's convention). */
export function main(argv: string[]): number {
  const command = argv[0];
  const rest = argv.slice(1);
  try {
    switch (command) {
      case "diff":
        return cmdDiff(rest);
      case "tokens":
        return cmdTokens(rest);
      case "cache":
        return cmdCache(rest);
      case "runs":
        return cmdRuns();
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
  process.exit(main(process.argv.slice(2)));
}
