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

const CASES: { name: string; argv: string[] }[] = [];
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
  );
});

describe.skipIf(!hasVenv)("cross-language analyzer conformance (JS output === Python output)", () => {
  it("diff / tokens / cache produce byte-identical output to the Python CLI", async () => {
    for (const c of CASES) {
      const js = await runJs(c.argv);
      const py = runPy(c.argv);
      expect(py.code, `python failed for ${c.name}`).toBe(0);
      expect(js.code, `js failed for ${c.name}`).toBe(0);
      expect(js.out, `mismatch for: ${c.name}`).toBe(py.out);
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
