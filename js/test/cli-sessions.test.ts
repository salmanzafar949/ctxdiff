/**
 * CLI-level tests for the SESSION/AGENT command surface, run in-process against
 * `src/cli.ts` with stdout/stderr captured.
 *
 * What is pinned here: the two discovery commands (`sessions` with its hidden
 * `runs` alias, and `agents`), the ambiguity contract (several sessions and no
 * `--session` => exit 2 with the pickable listing), selector resolution by
 * prefix, and the two cross diffs — cross-SESSION (the regression case: same
 * agent, two runs) and cross-AGENT (two agents, one run) — including the scope
 * header that keeps `turn 3 → turn 3` from being ambiguous.
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { mkdtempSync, renameSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { main } from "../src/cli.js";
import { CTrace } from "../src/store/ctrace.js";
import { writeMultiagent, writeProject, writeMultiturn } from "./helpers/fixtures.js";

let dir: string;
let projectPath: string;
/** A one-session trace, built ONCE: `writeMultiturn` appends a session to an
 * existing file, so re-running it would silently turn the fixture ambiguous. */
let singlePath: string;
/** The two session ids in `project.ctrace`, oldest first ("good", then "bad"). */
let good: string;
let bad: string;

beforeAll(() => {
  process.env.NO_COLOR = "1";
  dir = mkdtempSync(join(tmpdir(), "ctxdiff-cli-sessions-"));
  projectPath = writeProject(dir);
  singlePath = writeMultiturn(dir);
  const ct = CTrace.open(projectPath);
  try {
    [good, bad] = ct.listSessions().map((s) => s.id);
  } finally {
    ct.close();
  }
});
afterAll(() => rmSync(dir, { recursive: true, force: true }));

/** Run the CLI in-process, capturing stdout AND stderr and the exit code. */
async function run(argv: string[], cwd?: string) {
  const outChunks: string[] = [];
  const errChunks: string[] = [];
  const origOut = process.stdout.write.bind(process.stdout);
  const origErr = process.stderr.write.bind(process.stderr);
  const origCwd = process.cwd();
  // @ts-expect-error narrow override of the write signature for capture
  process.stdout.write = (s: string) => (outChunks.push(String(s)), true);
  // @ts-expect-error narrow override of the write signature for capture
  process.stderr.write = (s: string) => (errChunks.push(String(s)), true);
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

describe("ctxdiff sessions", () => {
  it("lists every session of a project with its local start time and agents", async () => {
    const r = await run(["sessions", "--project", projectPath]);
    expect(r.code).toBe(0);
    const lines = r.out.trimEnd().split("\n");
    expect(lines).toHaveLength(2);
    // Two sessions in one file => each row is labeled <file>#<short id>.
    expect(lines[0]).toContain(`project.ctrace#${good.slice(0, 12)}`);
    expect(lines[1]).toContain(`project.ctrace#${bad.slice(0, 12)}`);
    for (const line of lines) {
      expect(line).toMatch(/ \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [+-]\d{2}:\d{2} /);
      expect(line).toContain("project=pipeline");
      expect(line).toContain("turns=3");
      expect(line).toContain("agents=researcher, writer");
    }
  });

  it("labels a single-session file by its bare filename", async () => {
    const r = await run(["sessions", "--project", singlePath]);
    expect(r.code).toBe(0);
    expect(r.out.trimEnd().split("\n")[0].startsWith("multiturn.ctrace ")).toBe(true);
    expect(r.out).toContain("agents=-");
  });

  it("scans the whole working directory when no project is named", async () => {
    const r = await run(["sessions"], dir);
    expect(r.code).toBe(0);
    expect(r.out).toContain("multiturn.ctrace");
    expect(r.out).toContain("project.ctrace#");
  });

  it("still answers to the hidden `runs` alias", async () => {
    const viaSessions = await run(["sessions", "--project", projectPath]);
    const viaRuns = await run(["runs", "--project", projectPath]);
    expect(viaRuns).toEqual(viaSessions);
  });

  it("says so, and exits 0, for a directory with no traces", async () => {
    const empty = mkdtempSync(join(tmpdir(), "ctxdiff-empty-"));
    try {
      const r = await run(["sessions"], empty);
      expect(r.code).toBe(0);
      expect(r.out).toBe("no .ctrace files in the current directory\n");
    } finally {
      rmSync(empty, { recursive: true, force: true });
    }
  });
});

describe("ctxdiff agents", () => {
  it("aggregates each agent ACROSS every session in the project", async () => {
    const r = await run(["agents", "--project", projectPath]);
    expect(r.code).toBe(0);
    const lines = r.out.trimEnd().split("\n");
    // researcher: 2 sessions x 2 calls; only turn 1 of each reported usage
    // (100+20 and 101+20 = 241). writer: 2 sessions x 1 call (40+8 twice = 96).
    expect(lines[0]).toBe("researcher  sessions=2  calls=4  tokens=241");
    expect(lines[1]).toBe("writer  sessions=2  calls=2  tokens=96");
  });

  it("reports '-' rather than 0 when no call of an agent carried usage", async () => {
    const path = join(dir, "nousage.ctrace");
    const ct = CTrace.openOrCreateSession(path, "p", "openai", "", "2026-07-01T00:00:00Z");
    ct.recordCall({
      seq: 1, params: {}, usage: null, latencyMs: null, error: null,
      callBlocks: [], agent: "solo",
    });
    ct.close();
    const r = await run(["agents", "--project", path]);
    expect(r.code).toBe(0);
    expect(r.out).toBe("solo  sessions=1  calls=1  tokens=-\n");
  });
});

describe("session ambiguity", () => {
  it("refuses to guess between two sessions, exit 2, listing both", async () => {
    for (const argv of [
      ["tokens", "--project", projectPath],
      ["cache", "--project", projectPath],
      ["diff", "--project", projectPath, "--turn", "1", "--turn", "3"],
      ["export", "--project", projectPath],
    ]) {
      const r = await run(argv);
      expect(r.code, `exit code for ${argv[0]}`).toBe(2);
      expect(r.out).toBe("");
      expect(r.err.split("\n")[0]).toBe(
        "ctxdiff: this project holds 2 sessions — pass --session to pick one:",
      );
      expect(r.err).toContain(good.slice(0, 12));
      expect(r.err).toContain(bad.slice(0, 12));
    }
  });

  it("needs no --session when the project holds exactly one", async () => {
    const r = await run(["tokens", "--project", singlePath]);
    expect(r.code).toBe(0);
  });

  it("accepts a short-id prefix and reports an unknown one with the listing", async () => {
    const ok = await run(["tokens", "--project", projectPath, "--session", bad.slice(0, 12)]);
    expect(ok.code).toBe(0);
    const bogus = await run(["tokens", "--project", projectPath, "--session", "zzzz"]);
    expect(bogus.code).toBe(2);
    expect(bogus.err).toContain("no session 'zzzz' in this project — available sessions:");
  });

  it("rejects an --agent that names nobody, listing the real agents", async () => {
    const r = await run([
      "tokens", "--project", projectPath, "--session", good, "--agent", "nope",
    ]);
    expect(r.code).toBe(2);
    expect(r.err).toBe(
      `ctxdiff: no agent 'nope' in session ${good.slice(0, 12)} — available agents:\n` +
        "  researcher\n  writer\n",
    );
  });
});

describe("cross-session diff", () => {
  it("compares the same agent's same turn across two runs", async () => {
    const r = await run([
      "diff", "--project", projectPath,
      "--session", `${good}:3`, "--session", `${bad}:3`,
      "--agent", "researcher",
    ]);
    expect(r.code).toBe(0);
    const lines = r.out.trimEnd().split("\n");
    expect(lines[0]).toBe(
      `── ${good.slice(0, 12)} · researcher · turn 3  →  ` +
        `${bad.slice(0, 12)} · researcher · turn 3 ──`,
    );
    expect(lines[1]).toContain("turn 3 → turn 3");
    expect(lines[1]).toContain("1 blocks changed");
    // The single divergent block, rendered as an inline char diff.
    // Char-level inline diff: the shared trailing "d" of good/bad stays equal.
    expect(r.out).toContain("[-goo-]");
    expect(r.out).toContain("{+ba+}");
    expect(r.out).toContain("= 3 unchanged blocks");
  });

  it("accepts one --turn applied to both sides", async () => {
    const viaSuffix = await run([
      "diff", "--project", projectPath, "--session", `${good}:3`,
      "--session", `${bad}:3`, "--agent", "researcher",
    ]);
    const viaTurn = await run([
      "diff", "--project", projectPath, "--session", good, "--session", bad,
      "--turn", "3", "--agent", "researcher",
    ]);
    expect(viaTurn).toEqual(viaSuffix);
  });

  it("requires --agent when the two runs hold several agents", async () => {
    const r = await run([
      "diff", "--project", projectPath, "--session", `${good}:3`, "--session", `${bad}:3`,
    ]);
    expect(r.code).toBe(2);
    expect(r.err).toBe(
      "ctxdiff: these sessions hold 2 agents — pass --agent to pick one:\n" +
        "  researcher\n  writer\n",
    );
  });

  it("reports a turn that is not that agent's, naming the session", async () => {
    const r = await run([
      "diff", "--project", projectPath, "--session", `${good}:2`,
      "--session", `${bad}:3`, "--agent", "researcher",
    ]);
    expect(r.code).toBe(1);
    expect(r.err).toBe(
      `ctxdiff: session ${good.slice(0, 12)}: turn 2 is not a call of agent ` +
        "'researcher' (that agent's turns: [1, 3])\n",
    );
  });

  it("is a usage error when neither side names a turn", async () => {
    const r = await run([
      "diff", "--project", projectPath, "--session", good, "--session", bad,
      "--agent", "researcher",
    ]);
    expect(r.code).toBe(2);
    expect(r.err).toBe(
      "ctxdiff: each side of a cross-session diff needs a turn — pass " +
        "--session VALUE:TURN twice, or --turn N --turn M\n",
    );
  });
});

describe("cross-agent diff", () => {
  it("compares two agents within one session", async () => {
    const r = await run([
      "diff", "--project", projectPath, "--session", good,
      "--agent", "researcher:1", "--agent", "writer:2",
    ]);
    expect(r.code).toBe(0);
    const lines = r.out.trimEnd().split("\n");
    // The identical session is dropped from the scope header — only the axis
    // that actually differs is shown.
    expect(lines[0]).toBe("── researcher · turn 1  →  writer · turn 2 ──");
    expect(lines[1]).toContain("turn 1 → turn 2");
  });

  it("rejects an unknown agent on either side", async () => {
    const r = await run([
      "diff", "--project", projectPath, "--session", good,
      "--agent", "researcher:1", "--agent", "ghost:2",
    ]);
    expect(r.code).toBe(2);
    expect(r.err).toContain("no agent 'ghost' in session");
  });

  it("refuses to mix the two cross axes", async () => {
    const r = await run([
      "diff", "--project", projectPath, "--session", good, "--session", bad,
      "--agent", "researcher:1", "--agent", "writer:2",
    ]);
    expect(r.code).toBe(2);
    expect(r.err).toContain("compares along ONE axis");
  });

  it("rejects more than two sides", async () => {
    const r = await run([
      "diff", "--project", projectPath,
      "--session", good, "--session", bad, "--session", good, "--turn", "1",
    ]);
    expect(r.code).toBe(2);
    expect(r.err).toContain("at most twice");
  });

  it("rejects a :TURN suffix on an ordinary single-session diff", async () => {
    const r = await run([
      "diff", "--project", projectPath, "--session", `${good}:1`,
      "--turn", "1", "--turn", "3",
    ]);
    expect(r.code).toBe(2);
    expect(r.err).toContain("only means something on a cross-session or cross-agent diff");
  });
});

describe("--run stays an alias of --project", () => {
  it("resolves the same project, byte for byte", async () => {
    const viaProject = await run(["sessions", "--project", projectPath]);
    const viaRun = await run(["sessions", "--run", projectPath]);
    expect(viaRun).toEqual(viaProject);
  });
});

describe("flag surface: only the selectors a command really takes", () => {
  /**
   * Why this is worth its own block: `ctxdiff agents --agent researcher` reads
   * as "list only that agent". Registering `--agent` for every command and then
   * ignoring it on the ones that cannot honor it printed EVERY agent and exited
   * 0 — indistinguishable from "the filter matched everything", so a script
   * grepping that output was wrong forever and never learned. The Python CLI
   * registers only the flags each subparser acts on and exits 2; so does this.
   */
  it.each([
    ["cache", ["cache", "--turn", "1"]],
    ["sessions --session", ["sessions", "--session", "abc"]],
    ["sessions --agent", ["sessions", "--agent", "researcher"]],
    ["sessions --turn", ["sessions", "--turn", "1"]],
    ["runs --agent", ["runs", "--agent", "researcher"]],
    ["agents --agent", ["agents", "--agent", "researcher"]],
    ["agents --session", ["agents", "--session", "abc"]],
    ["agents --turn", ["agents", "--turn", "1"]],
  ])("%s rejects a selector it does not take", async (_name, argv) => {
    const r = await run([...argv, "--project", projectPath]);
    expect(r.code).toBe(2);
    expect(r.out).toBe("");
    expect(r.err).toContain("unrecognized arguments: ");
    expect(r.err).toContain(argv[1]);
  });

  it("still accepts every selector the command DOES take", async () => {
    expect((await run(["cache", "--project", projectPath, "--session", good,
      "--agent", "researcher"])).code).toBe(0);
    expect((await run(["tokens", "--project", projectPath, "--session", good,
      "--agent", "researcher", "--turn", "1"])).code).toBe(0);
    expect((await run(["sessions", "--project", projectPath])).code).toBe(0);
    expect((await run(["agents", "--run", projectPath])).code).toBe(0);
  });
});

describe("--turn value grammar", () => {
  it("echoes a turn too large for a double with every digit typed", async () => {
    // `parseInt` produces a double, so the raw 22 digits used to come back as
    // `1e+21` — a number the user never typed, in the one message whose job is
    // to quote what they did. Python's ints are exact and say all 22.
    const r = await run([
      "tokens", "--project", projectPath, "--session", good,
      "--turn", "1000000000000000000000",
    ]);
    expect(r.code).toBe(1);
    expect(r.err).toBe(
      "ctxdiff: turn 1000000000000000000000 not found in this run " +
        "(available turns: [1, 2, 3])\n",
    );
  });

  it("accepts a NEGATIVE turn as a value, not as an option", async () => {
    // argparse takes `-1` as `--turn`'s value (its negative-number heuristic);
    // `parseArgs` would take it as an option and fail with "expected one
    // argument", a different exit code AND a different message.
    const r = await run([
      "tokens", "--project", projectPath, "--session", good, "--turn", "-1",
    ]);
    expect(r.code).toBe(1);
    expect(r.err).toBe(
      "ctxdiff: turn -1 not found in this run (available turns: [1, 2, 3])\n",
    );
  });

  it("reports a negative NON-integer as a bad int, like argparse does", async () => {
    const r = await run([
      "tokens", "--project", projectPath, "--session", good, "--turn", "-1.5",
    ]);
    expect(r.code).toBe(2);
    expect(r.err).toContain("argument --turn: invalid int value: '-1.5'");
  });

  it("normalizes the int spellings Python's int() normalizes", async () => {
    for (const spelling of [" 3 ", "+3", "003"]) {
      const r = await run([
        "tokens", "--project", projectPath, "--session", good, "--turn", spelling,
      ]);
      expect(r.code, `for ${JSON.stringify(spelling)}`).toBe(0);
      expect(r.out).toContain("turn 3 ·");
    }
  });

  it("rejects non-ASCII digits", async () => {
    const r = await run([
      "tokens", "--project", projectPath, "--session", good, "--turn", "٢",
    ]);
    expect(r.code).toBe(2);
    expect(r.err).toContain("argument --turn: invalid int value: '٢'");
  });

  it("echoes a huge :TURN suffix in full on a cross-session diff", async () => {
    const r = await run([
      "diff", "--project", projectPath, "--session", `${good}:1000000000000000000000`,
      "--session", `${bad}:3`, "--agent", "researcher",
    ]);
    expect(r.code).toBe(1);
    expect(r.err).toContain("turn 1000000000000000000000 is not a call of agent");
  });
});

describe("a DISCOVERED project is named in selector errors", () => {
  let scanDir: string;
  beforeAll(() => {
    // `one.ctrace` holds researcher + writer; `two.ctrace` is written second and
    // is therefore the newest — the file a no-flag command defaults to.
    scanDir = mkdtempSync(join(tmpdir(), "ctxdiff-discovered-"));
    writeProject(scanDir);
    renameSync(join(scanDir, "project.ctrace"), join(scanDir, "one.ctrace"));
    writeMultiturn(scanDir);
    renameSync(join(scanDir, "multiturn.ctrace"), join(scanDir, "two.ctrace"));
  });
  afterAll(() => rmSync(scanDir, { recursive: true, force: true }));

  it("names the file the scan chose, not only a session short id", async () => {
    // `ctxdiff agents` lists 'researcher' (it scans every file), so this is the
    // obvious next command — and it lands in the OTHER file. Naming that file is
    // the only hint the user gets that `--project` is the fix.
    const listed = await run(["agents"], scanDir);
    expect(listed.out).toContain("researcher");

    const r = await run(["tokens", "--agent", "researcher"], scanDir);
    expect(r.code).toBe(2);
    expect(r.err.startsWith("ctxdiff: no agent 'researcher' in two.ctrace (session ")).toBe(true);
    expect(r.err).toContain("— available agents:");
  });

  it("leaves a project the user NAMED unlabeled", async () => {
    const r = await run(["tokens", "--project", projectPath, "--session", good,
      "--agent", "nope"]);
    expect(r.code).toBe(2);
    expect(r.err.startsWith(
      `ctxdiff: no agent 'nope' in session ${good.slice(0, 12)} —`)).toBe(true);
  });
});

describe("directory scanning matches Python's glob", () => {
  let scanDir: string;
  beforeAll(() => {
    scanDir = mkdtempSync(join(tmpdir(), "ctxdiff-scan-"));
    writeMultiturn(scanDir);
    renameSync(join(scanDir, "multiturn.ctrace"), join(scanDir, "visible.ctrace"));
    // U+F900 sorts BEFORE U+1D400 by code point (Python's `sorted`) and AFTER
    // it by UTF-16 code unit (JS's default comparator, which sees the high
    // surrogate D835 first) — the one pair that tells the two orders apart.
    writeMultiturn(scanDir);
    renameSync(join(scanDir, "multiturn.ctrace"), join(scanDir, "\u{F900}.ctrace"));
    writeMultiturn(scanDir);
    renameSync(join(scanDir, "multiturn.ctrace"), join(scanDir, "\u{1D400}.ctrace"));
    // A HIDDEN trace, written LAST so it is also the newest by mtime.
    writeMultiagent(scanDir);
    renameSync(join(scanDir, "multiagent.ctrace"), join(scanDir, ".hidden.ctrace"));
  });
  afterAll(() => rmSync(scanDir, { recursive: true, force: true }));

  it("skips dot-prefixed traces, which `glob('*.ctrace')` never matches", async () => {
    const r = await run(["sessions"], scanDir);
    expect(r.code).toBe(0);
    expect(r.out).not.toContain(".hidden.ctrace");
    expect(r.out).toContain("visible.ctrace");
  });

  it("orders filenames by CODE POINT, as Python's sorted() does", async () => {
    const r = await run(["sessions"], scanDir);
    const labels = r.out.trimEnd().split("\n").map((l) => l.split("  ")[0]);
    expect(labels).toEqual(["visible.ctrace", "\u{F900}.ctrace", "\u{1D400}.ctrace"]);
  });

  it("never lets a hidden trace win the newest-file default", async () => {
    // `.hidden.ctrace` is the newest file in the directory and holds a
    // two-agent run; the default must still be a VISIBLE trace, or `tokens`
    // reports numbers for a project Python would never have opened.
    const r = await run(["tokens"], scanDir);
    expect(r.code).toBe(0);
    expect(r.out).not.toContain("researcher");
  });
});

describe("--project naming a DIRECTORY", () => {
  it("quotes the path the way Python's repr does", async () => {
    // The message is `SQLiteStore has no single file to read (path=...)`, built
    // on both sides from the store's own path. Python interpolates `!r`, which
    // prefers single quotes and only switches to double quotes when the string
    // itself contains one — `JSON.stringify` always emits double, so the two
    // CLIs printed different bytes for the same directory.
    const plain = await run(["tokens", "--project", dir]);
    expect(plain.code).toBe(1);
    expect(plain.err).toContain(`(path='${dir}')`);

    const quoted = mkdtempSync(join(tmpdir(), "ctxdiff-quote-'-"));
    try {
      const r = await run(["tokens", "--project", quoted]);
      expect(r.code).toBe(1);
      // A path containing a single quote flips Python's repr to double quotes.
      expect(r.err).toContain(`(path="${quoted}")`);
    } finally {
      rmSync(quoted, { recursive: true, force: true });
    }
  });
});
