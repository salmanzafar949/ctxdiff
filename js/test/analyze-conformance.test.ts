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
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { main } from "../src/cli.js";
import { makeFixtures } from "./helpers/fixtures.js";

const repoRoot = resolve(process.cwd(), "..");
const venvPython = join(repoRoot, "venv", "bin", "python");
const pySrc = join(repoRoot, "src");
const hasVenv = existsSync(venvPython);

let dir: string;
let fx: { multiturn: string; multiagent: string; dynamic: string; bidi: string };

beforeAll(() => {
  dir = mkdtempSync(join(tmpdir(), "ctxdiff-conf-analyze-"));
  fx = makeFixtures(dir);
});
afterAll(() => rmSync(dir, { recursive: true, force: true }));

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

  it("runs listing is byte-identical (same cwd)", async () => {
    const js = await runJs(["runs"], dir);
    const py = runPy(["runs"], dir);
    expect(py.code).toBe(0);
    expect(js.out).toBe(py.out);
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
