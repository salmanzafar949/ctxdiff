/**
 * HEADLINE READ-SIDE TEST: cross-language analyzer conformance. For each
 * command and fixture, the JS CLI's output must be BYTE-IDENTICAL to the
 * existing Python CLI's output on the same `.ctrace` — same diff classification,
 * same token numbers/percentages, same cache attribution, same runs listing.
 * The JS side runs the real `cli.main` from source (stdout captured); the Python
 * side is spawned against the installed Python package (`../venv`). NO_COLOR is
 * forced on both so the comparison is plain text.
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, renameSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { main } from "../src/cli.js";
import { CTrace } from "../src/store/ctrace.js";
import { makeFixtures, writeMultiagent, writeMultiturn, writeProject } from "./helpers/fixtures.js";

const repoRoot = resolve(process.cwd(), "..");
const venvPython = join(repoRoot, "venv", "bin", "python");
const pySrc = join(repoRoot, "src");
const hasVenv = existsSync(venvPython);

let dir: string;
let fx: ReturnType<typeof makeFixtures>;
/** The two session ids in the shared `project.ctrace` fixture, oldest first —
 * the "good run" and the "bad run" of the cross-session regression case. Both
 * CLIs read the SAME file, so these ids are literally the same on both sides. */
let good: string;
let bad: string;

// A FIXED timezone for the whole file. Local-time rendering is now part of the
// compared output (`sessions` prints it), so leaving the zone to the machine
// would make this suite pass or fail depending on where it runs. Pinning it here
// covers the JS side (in-process) and the Python side alike, since `runPy`
// forwards `process.env` to the child.
const CONFORMANCE_TZ = "Asia/Dubai";
const originalTz = process.env.TZ;

beforeAll(() => {
  dir = mkdtempSync(join(tmpdir(), "ctxdiff-conf-analyze-"));
  fx = makeFixtures(dir);
  const ct = CTrace.open(fx.project);
  try {
    [good, bad] = ct.listSessions().map((s) => s.id);
  } finally {
    ct.close();
  }
  process.env.TZ = CONFORMANCE_TZ;
});
afterAll(() => {
  rmSync(dir, { recursive: true, force: true });
  if (originalTz === undefined) delete process.env.TZ;
  else process.env.TZ = originalTz;
});

/** Run the JS CLI in-process, capturing everything it writes to stdout AND
 * stderr. `main` is async — a configured database backend is read over the
 * network — so it is awaited here; the analyzers themselves are still
 * synchronous, so both buffers are complete when it resolves. */
async function runJs(
  argv: string[],
  cwd?: string,
): Promise<{ code: number; out: string; err: string }> {
  const outChunks: string[] = [];
  const errChunks: string[] = [];
  const origOut = process.stdout.write.bind(process.stdout);
  const origErr = process.stderr.write.bind(process.stderr);
  const origCwd = process.cwd();
  // @ts-expect-error narrow override of the write signature for capture
  process.stdout.write = (s: string) => (outChunks.push(typeof s === "string" ? s : String(s)), true);
  // @ts-expect-error narrow override of the write signature for capture
  process.stderr.write = (s: string) => (errChunks.push(typeof s === "string" ? s : String(s)), true);
  if (cwd) process.chdir(cwd);
  let code: number;
  try {
    code = await main(argv);
  } finally {
    process.stdout.write = origOut;
    process.stderr.write = origErr;
    if (cwd) process.chdir(origCwd);
  }
  return { code, out: outChunks.join(""), err: errChunks.join("") };
}

/** Spawn the Python CLI (installed package) with NO_COLOR, returning stdout,
 * stderr and exit code. */
function runPy(argv: string[], cwd?: string): { code: number; out: string; err: string } {
  const proc = spawnSync(
    venvPython,
    ["-c", "from ctxdiff.cli import main; import sys; sys.exit(main(sys.argv[1:]))", ...argv],
    { encoding: "utf8", cwd, env: { ...process.env, PYTHONPATH: pySrc, NO_COLOR: "1" } },
  );
  return { code: proc.status ?? -1, out: proc.stdout, err: proc.stderr };
}

// Force plain-text output on the JS side to match the spawned Python (which is
// non-TTY). Set once for the whole file.
beforeAll(() => {
  process.env.NO_COLOR = "1";
});

/** One success-path case. `code` is the exit status BOTH CLIs must return —
 * still 0 for every read command, but `ctxdiff check` exits 1 on a violated
 * budget while printing its report to stdout, so the expected status has to be
 * stated rather than assumed. A case that exits differently in the two SDKs is
 * a failure even when the text matches: CI reads the exit code, not the text. */
const CASES: { name: string; argv: string[]; code?: number }[] = [];
beforeAll(() => {
  CASES.push(
    { name: "diff turn 1→2 (append growth)", argv: ["diff", "--turn", "1", "--turn", "2", "--run", fx.multiturn] },
    { name: "diff turn 2→3 (modify)", argv: ["diff", "--turn", "2", "--turn", "3", "--run", fx.multiturn] },
    { name: "tokens (bloat + reconciliation)", argv: ["tokens", "--run", fx.multiturn] },
    { name: "tokens --turn 2", argv: ["tokens", "--turn", "2", "--run", fx.multiturn] },
    { name: "tokens multiagent (per-agent)", argv: ["tokens", "--run", fx.multiagent] },
    { name: "tokens --agent researcher", argv: ["tokens", "--agent", "researcher", "--run", fx.multiagent] },
    { name: "cache multiturn (break attribution)", argv: ["cache", "--run", fx.multiturn] },
    { name: "cache multiagent (no cross-agent break)", argv: ["cache", "--run", fx.multiagent] },
    { name: "cache dynamic (fix hint)", argv: ["cache", "--run", fx.dynamic] },
    { name: "cache --agent researcher", argv: ["cache", "--agent", "researcher", "--run", fx.multiagent] },
    // A diff whose added block's snippet contains zero-width + bidi format
    // characters, rendered through pyRepr — exercises non-printable escaping.
    { name: "diff bidi/ZWSP snippet (repr escaping)", argv: ["diff", "--turn", "1", "--turn", "2", "--run", fx.bidi] },

    // --- the session/agent surface -------------------------------------------
    // Discovery: the session table (short ids + LOCAL start times + agents) and
    // the project-wide agent rollup.
    { name: "sessions --project (two sessions, local times)", argv: ["sessions", "--project", fx.project] },
    { name: "sessions --project (single-session file)", argv: ["sessions", "--project", fx.multiagent] },
    { name: "runs alias === sessions", argv: ["runs", "--project", fx.project] },
    { name: "agents --project (aggregated across sessions)", argv: ["agents", "--project", fx.project] },
    { name: "agents --project (single session, no usage)", argv: ["agents", "--project", fx.multiturn] },
    // Selector resolution: explicit id, short-id prefix, and the --run alias.
    { name: "tokens --session <id>", argv: ["tokens", "--project", fx.project, "--session", bad] },
    { name: "tokens --session <prefix>", argv: ["tokens", "--project", fx.project, "--session", bad.slice(0, 12)] },
    { name: "tokens --session --agent", argv: ["tokens", "--project", fx.project, "--session", good, "--agent", "writer"] },
    { name: "tokens --run alias for --project", argv: ["tokens", "--run", fx.project, "--session", good] },
    { name: "cache --session --agent", argv: ["cache", "--project", fx.project, "--session", good, "--agent", "researcher"] },
    { name: "diff within a chosen session", argv: ["diff", "--project", fx.project, "--session", good, "--turn", "1", "--turn", "3"] },
    // The headline cases: cross-session (regression) and cross-agent diffs,
    // including the scope header only they print.
    { name: "diff cross-session (ID:TURN twice)", argv: ["diff", "--project", fx.project, "--session", `${good}:3`, "--session", `${bad}:3`, "--agent", "researcher"] },
    { name: "diff cross-session (one --turn, both sides)", argv: ["diff", "--project", fx.project, "--session", good, "--session", bad, "--turn", "3", "--agent", "researcher"] },
    { name: "diff cross-session (two --turn values)", argv: ["diff", "--project", fx.project, "--session", good, "--session", bad, "--turn", "1", "--turn", "3", "--agent", "researcher"] },
    { name: "diff cross-agent within one session", argv: ["diff", "--project", fx.project, "--session", good, "--agent", "researcher:1", "--agent", "writer:2"] },
    // Awkward stored timestamps: the year 9999/0001 boundaries (echoed raw by
    // both, since only one runtime can even represent the shifted value), an
    // 1800 LMT instant with SECONDS in the offset, the two ISO spellings only
    // `fromisoformat` used to accept, and an unparseable row.
    { name: "sessions with edge-case started_at values", argv: ["sessions", "--project", fx.edge] },
    // `--turn` spellings that normalize: whitespace, a leading +, leading zeros.
    { name: "tokens --turn ' 2 '", argv: ["tokens", "--turn", " 2 ", "--run", fx.multiturn] },
    { name: "tokens --turn +2", argv: ["tokens", "--turn", "+2", "--run", fx.multiturn] },
    { name: "tokens --turn 002", argv: ["tokens", "--turn", "002", "--run", fx.multiturn] },

    // --- `ctxdiff check`: the CI gate ----------------------------------------
    // Both the REPORT and the EXIT CODE are compared. A workflow's build turns
    // red on the exit code and its reviewer reads the report, so the two SDKs
    // disagreeing about either would mean the same trace passing in one
    // language's CI and failing in the other's.
    { name: "check all-pass (three assertions)", argv: ["check", "--run", fx.multiagent, "--max-context", "1000000", "--require-stable-prefix", "--no-dead-schemas"] },
    { name: "check max-context violation", code: 1, argv: ["check", "--run", fx.multiturn, "--max-context", "1"] },
    { name: "check max-context-pct violation", code: 1, argv: ["check", "--run", fx.multiturn, "--context-window", "100", "--max-context-pct", "12.5"] },
    { name: "check max-context-pct pass", argv: ["check", "--run", fx.multiturn, "--context-window", "1000000", "--max-context-pct", "12.5"] },
    { name: "check require-stable-prefix violation (grouped culprit)", code: 1, argv: ["check", "--run", fx.dynamic, "--require-stable-prefix"] },
    { name: "check require-stable-prefix pass", argv: ["check", "--run", fx.multiagent, "--require-stable-prefix"] },
    { name: "check no-dead-schemas violation", code: 1, argv: ["check", "--run", fx.multiturn, "--no-dead-schemas"] },
    { name: "check no-dead-schemas pass (no schemas at all)", argv: ["check", "--run", fx.multiagent, "--no-dead-schemas"] },
    { name: "check max-growth violation", code: 1, argv: ["check", "--run", fx.multiturn, "--max-growth", "0"] },
    { name: "check max-growth-pct violation", code: 1, argv: ["check", "--run", fx.multiturn, "--max-growth-pct", "0"] },
    // Per-agent growth pairing: the interleaved timeline must pair within each
    // agent, never across the hand-off, in both SDKs.
    { name: "check max-growth multiagent (per-agent pairing)", code: 1, argv: ["check", "--run", fx.multiagent, "--max-growth", "0"] },
    { name: "check --agent scoping (fails)", code: 1, argv: ["check", "--run", fx.multiagent, "--agent", "researcher", "--max-context", "1"] },
    { name: "check --agent scoping (passes)", argv: ["check", "--run", fx.multiagent, "--agent", "writer", "--max-context", "1000000"] },
    { name: "check --session --agent", code: 1, argv: ["check", "--project", fx.project, "--session", good, "--agent", "researcher", "--max-context", "1", "--max-growth", "0"] },
    // Every assertion at once, so the fixed report ORDER is compared too.
    { name: "check every assertion at once", code: 1, argv: ["check", "--run", fx.multiturn, "--max-context", "1", "--context-window", "100", "--max-context-pct", "10", "--require-stable-prefix", "--no-dead-schemas", "--max-growth", "0", "--max-growth-pct", "0"] },
    // A one-turn run: no pairs to check, no growth to measure — both SDKs must
    // say so rather than claiming a stability they never measured.
    { name: "check single-turn run (nothing to pair)", argv: ["check", "--project", fx.project, "--session", good, "--agent", "writer", "--require-stable-prefix", "--max-growth", "0"] },
    // UNMEASURED turns: every turn holds a remote-URL image whose cost cannot
    // be known, so the stored totals (4 and 8 tok against a provider-reported
    // 800 and 1,600) are floors. Both SDKs must REFUSE the comparison — exit 1
    // with the same violation lines — rather than passing a 1,600-token turn
    // under a 500-token budget.
    { name: "check unmeasured turns refuse a max-context budget", code: 1, argv: ["check", "--run", fx.unmeasured, "--max-context", "500"] },
    { name: "check unmeasured turns refuse a percentage budget", code: 1, argv: ["check", "--run", fx.unmeasured, "--context-window", "10000", "--max-context-pct", "50"] },
    { name: "check unmeasured turns refuse a growth budget", code: 1, argv: ["check", "--run", fx.unmeasured, "--max-growth", "1000", "--max-growth-pct", "500"] },
    // A four-agent fan-out: nothing can be paired, and both SDKs must say WHY
    // in the same words — "fewer than 2 turns" over four turns is the wording
    // that hides a no-op assertion.
    { name: "check fan-out (4 agents, 1 turn each) — no pairs", argv: ["check", "--run", fx.fanout, "--require-stable-prefix", "--max-growth", "0", "--max-growth-pct", "0"] },

    // --- share of the context window -----------------------------------------
    // The denominator is the user's to supply, so the flag is the only thing
    // that turns a bare token count into `X / W tok · P%`. Both sides of the
    // alarm threshold are compared, because a marker that appeared in one SDK
    // and not the other would be a difference nobody could explain.
    { name: "tokens --context-window (share + percentage)", argv: ["tokens", "--run", fx.multiturn, "--context-window", "1000"] },
    { name: "tokens --context-window below the alarm", argv: ["tokens", "--run", fx.multiturn, "--context-window", "120"] },
    { name: "tokens --context-window past the alarm", argv: ["tokens", "--run", fx.multiturn, "--context-window", "60"] },
    { name: "tokens --context-window multiagent", argv: ["tokens", "--run", fx.multiagent, "--context-window", "100"] },
    // A run whose totals are FLOORS: `(~approx)` must survive the header being
    // rewritten into a share, and must sit after the percentage in both.
    { name: "tokens --context-window with unmeasured turns", argv: ["tokens", "--run", fx.unmeasured, "--context-window", "1000"] },
    { name: "tokens --context-window --turn", argv: ["tokens", "--run", fx.multiturn, "--turn", "2", "--context-window", "1000"] },

    // --- tagged evictions ------------------------------------------------------
    // The interleaved fixture is the point: the hand-off at turn 2 must produce
    // NO line in either SDK, and the researcher's real loss at turn 5 must
    // produce the same one, with the same agent chip and the same snippet
    // quoting.
    { name: "tokens tagged evictions (hand-off is not a loss)", argv: ["tokens", "--run", fx.tagged] },
    { name: "tokens tagged evictions --agent researcher", argv: ["tokens", "--run", fx.tagged, "--agent", "researcher"] },
    { name: "tokens tagged evictions --agent writer (nothing tagged)", argv: ["tokens", "--run", fx.tagged, "--agent", "writer"] },
    // `--turn N` selects one turn, and the stanza obeys the same selector in
    // both SDKs: silent under the turn that still HAD the block, printed under
    // the turn that lost it.
    { name: "tokens tagged evictions --turn 1 (before the loss)", argv: ["tokens", "--run", fx.tagged, "--turn", "1"] },
    { name: "tokens tagged evictions --turn 5 (the losing turn)", argv: ["tokens", "--run", fx.tagged, "--turn", "5"] },
    { name: "check --no-tagged-eviction violation", code: 1, argv: ["check", "--run", fx.tagged, "--no-tagged-eviction"] },
    { name: "check --no-tagged-eviction scoped to the losing agent", code: 1, argv: ["check", "--run", fx.tagged, "--agent", "researcher", "--no-tagged-eviction"] },
    // Three different PASSes that must not read alike: nothing tagged, nothing
    // paired, and a genuine all-clear.
    { name: "check --no-tagged-eviction (nothing tagged)", argv: ["check", "--run", fx.multiturn, "--no-tagged-eviction"] },
    { name: "check --no-tagged-eviction (nothing to pair)", argv: ["check", "--run", fx.fanout, "--no-tagged-eviction"] },
    { name: "check --no-tagged-eviction --agent writer (vacuous pass)", argv: ["check", "--run", fx.tagged, "--agent", "writer", "--no-tagged-eviction"] },
    // Every assertion at once, so the fixed report ORDER with the new row in it
    // is compared rather than assumed.
    { name: "check every assertion including tagged eviction", code: 1, argv: ["check", "--run", fx.tagged, "--max-context", "1", "--context-window", "100", "--max-context-pct", "10", "--require-stable-prefix", "--no-dead-schemas", "--no-tagged-eviction", "--max-growth", "0", "--max-growth-pct", "0"] },
  );
});

// Error-path cases: exit codes must match Python exactly, and the shared error
// message must appear in both stderrs. `exactStderr` cases additionally assert
// byte-identical stderr (the operational-error path, where JS reproduces
// Python's line verbatim; the argparse paths differ only in the prog prefix).
interface ErrorCase {
  name: string;
  argv: string[];
  core: string;
  exactStderr?: boolean;
}
const ERROR_CASES: ErrorCase[] = [];
beforeAll(() => {
  ERROR_CASES.push(
    {
      name: "unknown flag → exit 2",
      argv: ["diff", "--turn", "1", "--turn", "2", "--bogus", "--run", fx.multiturn],
      core: "unrecognized arguments: --bogus",
    },
    {
      name: "missing --turn value → exit 2",
      argv: ["diff", "--turn"],
      core: "argument --turn: expected one argument",
    },
    {
      name: "non-int --turn → exit 2",
      argv: ["tokens", "--turn", "abc", "--run", fx.multiturn],
      core: "argument --turn: invalid int value: 'abc'",
    },
    {
      name: "out-of-range turn list → exit 1, [1, 2, 3] formatting",
      argv: ["diff", "--turn", "1", "--turn", "99", "--run", fx.multiturn],
      core: "turn(s) [99] not found in this run (available turns: [1, 2, 3])",
      exactStderr: true,
    },
    // --- selector errors: exit 2, and the LISTING must match byte for byte ----
    {
      name: "ambiguous session → exit 2 + session listing",
      argv: ["tokens", "--project", fx.project],
      core: "this project holds 2 sessions — pass --session to pick one:",
      exactStderr: true,
    },
    {
      name: "unknown session → exit 2 + session listing",
      argv: ["cache", "--project", fx.project, "--session", "zzzz"],
      core: "no session 'zzzz' in this project — available sessions:",
      exactStderr: true,
    },
    {
      name: "unknown agent → exit 2 + agent listing",
      argv: ["tokens", "--project", fx.project, "--session", good, "--agent", "nope"],
      core: "available agents:",
      exactStderr: true,
    },
    {
      name: "cross-session without --agent → exit 2 + agent listing",
      argv: ["diff", "--project", fx.project, "--session", `${good}:3`, "--session", `${bad}:3`],
      core: "these sessions hold 2 agents — pass --agent to pick one:",
      exactStderr: true,
    },
    {
      name: "cross-session with no turn on either side → exit 2",
      argv: ["diff", "--project", fx.project, "--session", good, "--session", bad, "--agent", "researcher"],
      core: "each side of a cross-session diff needs a turn",
      exactStderr: true,
    },
    {
      name: "cross-session turn not owned by the agent → exit 1, session-qualified",
      argv: ["diff", "--project", fx.project, "--session", `${good}:2`, "--session", `${bad}:3`, "--agent", "researcher"],
      core: "turn 2 is not a call of agent 'researcher' (that agent's turns: [1, 3])",
      exactStderr: true,
    },
    {
      name: "cross-session turn out of range → exit 1, session-qualified",
      argv: ["diff", "--project", fx.project, "--session", `${good}:9`, "--session", `${bad}:3`, "--agent", "researcher"],
      core: "turn 9 is not a call of agent 'researcher' (that agent's turns: [1, 3])",
      exactStderr: true,
    },
    {
      name: "both cross axes at once → exit 2",
      argv: ["diff", "--project", fx.project, "--session", good, "--session", bad, "--agent", "researcher:1", "--agent", "writer:2"],
      core: "compares along ONE axis",
      exactStderr: true,
    },
    {
      name: "three --session values → exit 2",
      argv: ["diff", "--project", fx.project, "--session", good, "--session", bad, "--session", good, "--turn", "1"],
      core: "at most twice",
      exactStderr: true,
    },
    {
      name: ":TURN suffix on a single-session diff → exit 2",
      argv: ["diff", "--project", fx.project, "--session", `${good}:1`, "--turn", "1", "--turn", "3"],
      core: "only means something on a cross-session or cross-agent diff",
      exactStderr: true,
    },
    {
      name: "unknown agent on a cross-agent side → exit 2 + listing",
      argv: ["diff", "--project", fx.project, "--session", good, "--agent", "researcher:1", "--agent", "ghost:2"],
      core: "no agent 'ghost' in session",
      exactStderr: true,
    },
    // --- a selector flag the command does not take -> exit 2, never ignored ---
    // Accepting and ignoring these printed a FULL listing and exited 0, which
    // reads as "the filter matched everything" — silently wrong, forever.
    // (argparse reports them from the top-level parser, so only the exit code
    // and the "unrecognized arguments: <flag>" fragment are compared; the usage
    // block above it is argparse chrome, not output.)
    {
      name: "agents --agent → exit 2",
      argv: ["agents", "--project", fx.project, "--agent", "researcher"],
      core: "unrecognized arguments: --agent",
    },
    {
      name: "agents --session → exit 2",
      argv: ["agents", "--project", fx.project, "--session", "abc"],
      core: "unrecognized arguments: --session",
    },
    {
      name: "sessions --session → exit 2",
      argv: ["sessions", "--project", fx.project, "--session", "abc"],
      core: "unrecognized arguments: --session",
    },
    {
      name: "runs (alias) --turn → exit 2",
      argv: ["runs", "--project", fx.project, "--turn", "1"],
      core: "unrecognized arguments: --turn",
    },
    {
      name: "cache --turn → exit 2",
      argv: ["cache", "--project", fx.project, "--turn", "1"],
      core: "unrecognized arguments: --turn",
    },
    // --- `--turn` value grammar ----------------------------------------------
    {
      name: "a turn beyond a double keeps every digit → exit 1",
      argv: ["tokens", "--project", fx.project, "--session", good, "--turn", "1000000000000000000000"],
      core: "turn 1000000000000000000000 not found in this run",
      exactStderr: true,
    },
    {
      name: "a NEGATIVE turn is a value, not an option → exit 1",
      argv: ["tokens", "--project", fx.project, "--session", good, "--turn", "-1"],
      core: "turn -1 not found in this run",
      exactStderr: true,
    },
    {
      name: "a negative non-integer turn → exit 2, invalid int value",
      argv: ["tokens", "--project", fx.project, "--session", good, "--turn", "-1.5"],
      core: "argument --turn: invalid int value: '-1.5'",
    },
    {
      name: "non-ASCII digits in --turn → exit 2 on both sides",
      argv: ["tokens", "--project", fx.project, "--session", good, "--turn", "٢"],
      core: "argument --turn: invalid int value: '٢'",
    },
    {
      name: "a huge :TURN suffix on a cross-session side → exit 1, all digits",
      argv: ["diff", "--project", fx.project, "--session", `${good}:1000000000000000000000`, "--session", `${bad}:3`, "--agent", "researcher"],
      core: "turn 1000000000000000000000 is not a call of agent 'researcher'",
      exactStderr: true,
    },
    // --- `--project` naming a DIRECTORY: Python repr quoting ------------------
    {
      name: "--project <directory> → exit 1, repr-quoted path",
      argv: ["tokens", "--project", dir],
      core: "SQLiteStore has no single file to read",
      exactStderr: true,
    },
    // --- `ctxdiff check`'s usage errors --------------------------------------
    // Every one of these is a message ctxdiff composes itself (not argparse
    // chrome), so stderr is compared byte for byte: a CI gate that refuses to
    // run must explain itself identically in both SDKs, or a workflow debugged
    // against one is undebuggable against the other.
    {
      name: "check with no assertions → exit 2, lists every flag",
      argv: ["check", "--run", fx.multiturn],
      core: "nothing to assert — pass at least one of",
      exactStderr: true,
    },
    {
      name: "check --max-context-pct with no window → exit 2",
      argv: ["check", "--run", fx.multiturn, "--max-context-pct", "80"],
      core: "--max-context-pct needs a denominator",
      exactStderr: true,
    },
    {
      name: "check --context-window with no percentage → exit 2",
      argv: ["check", "--run", fx.multiturn, "--max-context", "10", "--context-window", "80"],
      core: "--context-window is only used by --max-context-pct",
      exactStderr: true,
    },
    {
      name: "check --max-context 0 → exit 2",
      argv: ["check", "--run", fx.multiturn, "--max-context", "0"],
      core: "--max-context must be greater than 0 (got 0)",
      exactStderr: true,
    },
    // A window of zero is a division by zero and a negative one is not a window,
    // whichever command was asked. The rule lives in the ONE resolver so all
    // four inherit it — and it used to be enforced ONLY by `check`, which is why
    // `tokens --context-window 0` was a ZeroDivisionError traceback in Python and
    // an `⚠ Infinity%` report (exit 0!) in JS. `check` keeps its own `ctxdiff
    // check:` prefix, so both spellings are pinned here.
    {
      name: "tokens --context-window 0 → exit 2, not a division by zero",
      argv: ["tokens", "--run", fx.multiturn, "--context-window", "0"],
      core: "ctxdiff: --context-window must be greater than 0 (got 0)",
      exactStderr: true,
    },
    {
      name: "tokens --context-window -5 → exit 2, not a -260.0% header",
      argv: ["tokens", "--run", fx.multiturn, "--context-window", "-5"],
      core: "ctxdiff: --context-window must be greater than 0 (got -5)",
      exactStderr: true,
    },
    {
      name: "export --context-window 0 → exit 2, not a dashboard of Infinity%",
      argv: ["export", "--run", fx.multiturn, "--context-window", "0", "--out", join(dir, "zero-window.html")],
      core: "ctxdiff: --context-window must be greater than 0 (got 0)",
      exactStderr: true,
    },
    {
      name: "export --context-window -5 → exit 2 (a value, not an option)",
      argv: ["export", "--run", fx.multiturn, "--context-window", "-5", "--out", join(dir, "neg-window.html")],
      core: "ctxdiff: --context-window must be greater than 0 (got -5)",
      exactStderr: true,
    },
    {
      name: "check --context-window 0 → exit 2, in check's own words",
      argv: ["check", "--run", fx.multiturn, "--max-context-pct", "50", "--context-window", "0"],
      core: "ctxdiff check: --context-window must be greater than 0 (got 0)",
      exactStderr: true,
    },
    {
      name: "check --max-context-pct 0 → exit 2, one-decimal echo",
      argv: ["check", "--run", fx.multiturn, "--max-context-pct", "0", "--context-window", "10"],
      core: "--max-context-pct must be greater than 0 (got 0.0)",
      exactStderr: true,
    },
    {
      name: "check a NEGATIVE --max-growth is a value, not an option → exit 2",
      argv: ["check", "--run", fx.multiturn, "--max-growth", "-5"],
      core: "--max-growth cannot be negative (got -5)",
      exactStderr: true,
    },
    {
      name: "check --max-growth-pct -2.5 → exit 2, one-decimal echo",
      argv: ["check", "--run", fx.multiturn, "--max-growth-pct", "-2.5"],
      core: "--max-growth-pct cannot be negative (got -2.5)",
      exactStderr: true,
    },
    {
      name: "check non-int threshold → exit 2",
      argv: ["check", "--run", fx.multiturn, "--max-context", "abc"],
      core: "argument --max-context: invalid int value: 'abc'",
    },
    {
      name: "check exponent-notation percentage → exit 2",
      argv: ["check", "--run", fx.multiturn, "--max-context-pct", "1e5", "--context-window", "10"],
      core: "argument --max-context-pct: invalid float value: '1e5'",
    },
    {
      name: "check --turn → exit 2 (a budget is a whole-run property)",
      argv: ["check", "--run", fx.multiturn, "--turn", "1", "--max-context", "10"],
      core: "unrecognized arguments: --turn",
    },
    {
      name: "check unknown agent → exit 2 + agent listing",
      argv: ["check", "--project", fx.project, "--session", good, "--agent", "nope", "--max-context", "10"],
      core: "available agents:",
      exactStderr: true,
    },
    {
      name: "check ambiguous session → exit 2 + session listing",
      argv: ["check", "--project", fx.project, "--max-context", "10"],
      core: "this project holds 2 sessions — pass --session to pick one:",
      exactStderr: true,
    },
    // A stray positional. Python's `check` subparser registers none, so this is
    // argparse's "unrecognized arguments" — and the JS CLI must not quietly
    // adopt it as `--project`, which turned a mistyped boolean into a full
    // report over a different (or nonexistent) trace.
    {
      name: "check stray positional (a value for a store_true flag) → exit 2",
      argv: ["check", "--run", fx.multiturn, "--require-stable-prefix", "false"],
      core: "unrecognized arguments: false",
    },
    {
      name: "check two stray positionals → exit 2, both named",
      argv: ["check", "--run", fx.multiturn, "--no-dead-schemas", "yes", "please"],
      core: "unrecognized arguments: yes please",
    },
  );
});

describe.skipIf(!hasVenv)("cross-language analyzer conformance (JS output === Python output)", () => {
  it("diff / tokens / cache / check produce byte-identical output to the Python CLI", async () => {
    for (const c of CASES) {
      const expected = c.code ?? 0;
      const js = await runJs(c.argv);
      const py = runPy(c.argv);
      expect(py.code, `python exited ${py.code} for ${c.name}\n${py.err}`).toBe(expected);
      expect(js.code, `js exited ${js.code} for ${c.name}\n${js.err}`).toBe(expected);
      expect(js.out, `mismatch for: ${c.name}`).toBe(py.out);
    }
  });

  it("resolves CTXDIFF_CONTEXT_WINDOW identically in both SDKs", async () => {
    // The environment path cannot be expressed as argv, so it gets its own case.
    // `runPy` forwards `process.env` to the child and `runJs` runs in-process,
    // so setting the variable here exercises BOTH resolvers with one value.
    //
    // Four things are compared: that the variable alone produces percentages;
    // that a flag beats it; that `check`'s `--max-context-pct` accepts it as its
    // denominator with no flag typed; and that an unusable value is the same
    // exit code and the same message on both sides — a CI job that inherits a
    // typo'd variable must fail the same way whichever SDK the workflow picked.
    const saved = process.env.CTXDIFF_CONTEXT_WINDOW;
    try {
      const compare = async (argv: string[], expected: number) => {
        const js = await runJs(argv);
        const py = runPy(argv);
        expect(py.code, `python exited ${py.code}\n${py.err}`).toBe(expected);
        expect(js.code).toBe(py.code);
        expect(js.out).toBe(py.out);
        expect(js.err).toBe(py.err);
      };

      process.env.CTXDIFF_CONTEXT_WINDOW = "1000";
      await compare(["tokens", "--run", fx.multiturn], 0);
      await compare(["tokens", "--run", fx.multiturn, "--context-window", "80"], 0);
      await compare(["check", "--run", fx.multiturn, "--max-context-pct", "1"], 1);
      // An ambient window must never trip check's "you typed a flag nothing
      // reads" rule — only a typed flag can.
      await compare(["check", "--run", fx.multiturn, "--max-context", "1000000"], 0);

      process.env.CTXDIFF_CONTEXT_WINDOW = "not-a-number";
      await compare(["tokens", "--run", fx.multiturn], 2);

      process.env.CTXDIFF_CONTEXT_WINDOW = "   ";
      await compare(["tokens", "--run", fx.multiturn], 0);
    } finally {
      if (saved === undefined) delete process.env.CTXDIFF_CONTEXT_WINDOW;
      else process.env.CTXDIFF_CONTEXT_WINDOW = saved;
    }
  });

  it("sessions listing is byte-identical when scanning a whole directory", async () => {
    for (const command of ["sessions", "runs", "agents"]) {
      const js = await runJs([command], dir);
      const py = runPy([command], dir);
      expect(py.code, `python failed for ${command}`).toBe(0);
      expect(js.code).toBe(py.code);
      expect(js.out, `mismatch for ${command} (cwd scan)`).toBe(py.out);
      expect(js.err).toBe(py.err);
    }
  });

  /**
   * Local-time rendering is the one piece of output that depends on ambient
   * state, so it gets its own matrix rather than riding on the fixed
   * `CONFORMANCE_TZ`: a whole-hour zone, UTC, a half-hour zone, a DST-observing
   * one, and a quarter-hour zone (`Pacific/Chatham`, +12:45) — the shape most
   * likely to break a hand-rolled offset formatter on either side. (That the
   * offset is taken at the timestamp's own INSTANT rather than today's is pinned
   * separately, per-SDK, with summer and winter values.)
   *
   * Node re-reads `process.env.TZ` at runtime and Python's `astimezone()` reads
   * the same `TZ` in the spawned child (which inherits `process.env`), so one
   * assignment pins both sides to the same zone.
   */
  it("local timestamps render identically under every timezone", async () => {
    const zones = ["Asia/Dubai", "UTC", "Asia/Kolkata", "America/Los_Angeles", "Pacific/Chatham"];
    for (const tz of zones) {
      process.env.TZ = tz;
      const js = await runJs(["sessions", "--project", fx.project]);
      const py = runPy(["sessions", "--project", fx.project]);
      expect(py.code, `python failed under TZ=${tz}`).toBe(0);
      expect(js.code).toBe(0);
      expect(js.out, `local-time mismatch under TZ=${tz}`).toBe(py.out);
      // Sanity: the row really does carry an offset, so a silently-UTC
      // implementation on both sides could not pass this test unnoticed.
      expect(js.out).toMatch(/ \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [+-]\d{2}:\d{2} /);
    }
    process.env.TZ = CONFORMANCE_TZ;
  });

  /**
   * The same matrix over the AWKWARD stored timestamps (see `writeEdgeTimestamps`),
   * which is where the two runtimes stop agreeing for free: `datetime` cannot
   * represent the year the boundary rows shift into and `Date` can, `Date`
   * truncates a sub-minute offset to whole minutes and `//` floors, and each
   * side's ISO parser accepts a slightly different set of spellings. Every zone
   * here is chosen to make one of those bite: the two boundary rows overflow in
   * OPPOSITE directions east and west of UTC, and the 1800 row carries an LMT
   * offset with seconds in it in three of the four zones.
   */
  it("awkward stored timestamps render identically under every timezone", async () => {
    const zones = ["Asia/Dubai", "UTC", "America/New_York", "America/Los_Angeles"];
    for (const tz of zones) {
      process.env.TZ = tz;
      const js = await runJs(["sessions", "--project", fx.edge]);
      const py = runPy(["sessions", "--project", fx.edge]);
      expect(py.code, `python failed under TZ=${tz}`).toBe(0);
      expect(js.code, `js failed under TZ=${tz}`).toBe(0);
      expect(js.err, `js wrote to stderr under TZ=${tz}`).toBe("");
      expect(js.out, `edge-timestamp mismatch under TZ=${tz}`).toBe(py.out);
      // Sanity: the unparseable row is echoed, so neither side quietly dropped
      // the file's odd rows to reach agreement.
      expect(js.out).toContain("not a timestamp");
    }
    process.env.TZ = CONFORMANCE_TZ;
  });

  /**
   * Directory discovery: which files a bare `ctxdiff sessions` SEES and in what
   * order, and which one a bare `ctxdiff tokens` then analyzes. Python scans with
   * `sorted(glob("*.ctrace"))`, so a dot-prefixed trace is invisible to it and
   * filenames order by code point; a JS `readdirSync().sort()` sees the hidden
   * file and orders by UTF-16 code unit. Both differences change the ANSWER, not
   * just the listing — a hidden newer trace winning the newest-file default means
   * `tokens` reports on a run the other CLI never opened.
   */
  it("directory discovery sees and orders the same files as Python", async () => {
    const scan = mkdtempSync(join(tmpdir(), "ctxdiff-conf-scan-"));
    try {
      writeMultiturn(scan);
      renameSync(join(scan, "multiturn.ctrace"), join(scan, "visible.ctrace"));
      // U+F900 sorts BEFORE U+1D400 by code point and AFTER it by UTF-16 code
      // unit (the astral character's high surrogate D835 compares first).
      writeMultiturn(scan);
      renameSync(join(scan, "multiturn.ctrace"), join(scan, "\u{F900}.ctrace"));
      writeMultiturn(scan);
      renameSync(join(scan, "multiturn.ctrace"), join(scan, "\u{1D400}.ctrace"));
      // A HIDDEN trace, written last so it is also the newest by mtime.
      writeMultiagent(scan);
      renameSync(join(scan, "multiagent.ctrace"), join(scan, ".hidden.ctrace"));

      for (const argv of [["sessions"], ["runs"], ["agents"], ["tokens"], ["cache"]]) {
        const js = await runJs(argv, scan);
        const py = runPy(argv, scan);
        expect(js.code, `exit code mismatch for ${argv[0]}`).toBe(py.code);
        expect(js.out, `mismatch for ${argv[0]} in a directory with a hidden trace`).toBe(py.out);
        expect(js.err).toBe(py.err);
      }
    } finally {
      rmSync(scan, { recursive: true, force: true });
    }
  });

  /**
   * A project the CLI DISCOVERED rather than was given is named in the selector
   * error, on both sides. `ctxdiff agents` lists agents from every trace in the
   * directory, so the obvious follow-up names an agent that lives in a different
   * file than the newest-file default picked — and the error has to say which.
   */
  it("names a discovered project in selector errors, identically", async () => {
    const scan = mkdtempSync(join(tmpdir(), "ctxdiff-conf-discovered-"));
    try {
      writeProject(scan);
      renameSync(join(scan, "project.ctrace"), join(scan, "one.ctrace"));
      writeMultiturn(scan);
      renameSync(join(scan, "multiturn.ctrace"), join(scan, "two.ctrace"));

      const js = await runJs(["tokens", "--agent", "researcher"], scan);
      const py = runPy(["tokens", "--agent", "researcher"], scan);
      expect(js.code).toBe(2);
      expect(py.code).toBe(2);
      expect(js.err).toBe(py.err);
      expect(js.err).toContain("no agent 'researcher' in two.ctrace (session ");
    } finally {
      rmSync(scan, { recursive: true, force: true });
    }
  });

  it("error paths: exit codes and error messages match the Python CLI", async () => {
    for (const c of ERROR_CASES) {
      const js = await runJs(c.argv);
      const py = runPy(c.argv);
      // Exit code must match Python exactly.
      expect(js.code, `exit code mismatch for ${c.name}`).toBe(py.code);
      // The shared error message appears in both stderrs.
      expect(js.err, `js stderr missing core for ${c.name}`).toContain(c.core);
      expect(py.err, `py stderr missing core for ${c.name}`).toContain(c.core);
      // Operational-error path reproduces Python's line byte-for-byte.
      if (c.exactStderr) {
        expect(js.err, `stderr not byte-identical for ${c.name}`).toBe(py.err);
      }
    }
  });
});
