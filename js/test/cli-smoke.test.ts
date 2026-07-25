/**
 * CLI smoke tests: spawn the BUILT `dist/cli.js` (the actual `npx ctxdiff`
 * binary) as a child process and assert exit codes + expected output
 * substrings. Complements the in-process conformance test by exercising the
 * real shebang'd bundle end to end. Skipped with a clear message when dist isn't
 * built (CI builds before testing; locally run `npm run build` first).
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { makeFixtures } from "./helpers/fixtures.js";

const cliPath = resolve(process.cwd(), "dist", "cli.js");
const hasBuild = existsSync(cliPath);

let dir: string;
let fx: ReturnType<typeof makeFixtures>;

beforeAll(() => {
  dir = mkdtempSync(join(tmpdir(), "ctxdiff-cli-smoke-"));
  fx = makeFixtures(dir);
});
afterAll(() => rmSync(dir, { recursive: true, force: true }));

function run(argv: string[], cwd?: string) {
  const proc = spawnSync(process.execPath, [cliPath, ...argv], {
    encoding: "utf8",
    cwd,
    env: { ...process.env, NO_COLOR: "1" },
  });
  return { code: proc.status ?? -1, out: proc.stdout, err: proc.stderr };
}

describe.skipIf(!hasBuild)("ctxdiff CLI (dist/cli.js) smoke", () => {
  it("diff --turn N --turn M exits 0 with a diff header", () => {
    const r = run(["diff", "--turn", "2", "--turn", "3", fx.multiturn]);
    expect(r.code).toBe(0);
    expect(r.out).toContain("turn 2 → turn 3");
    expect(r.out).toContain("blocks changed");
  });

  it("tokens exits 0 with the bloat warning", () => {
    const r = run(["tokens", fx.multiturn]);
    expect(r.code).toBe(0);
    expect(r.out).toContain("schema bloat");
    expect(r.out).toContain("% of avg context");
  });

  it("cache exits 0 and reports the prefix break + hint", () => {
    const r = run(["cache", fx.dynamic]);
    expect(r.code).toBe(0);
    expect(r.out).toContain("warning:");
    expect(r.out).toContain("hint:");
  });

  it("runs lists fixtures in the cwd", () => {
    const r = run(["runs"], dir);
    expect(r.code).toBe(0);
    expect(r.out).toContain("multiturn.ctrace");
    expect(r.out).toContain("provider=openai");
  });

  it("sessions lists each session with a local timestamp column", () => {
    const r = run(["sessions"], dir);
    expect(r.code).toBe(0);
    expect(r.out).toContain("multiturn.ctrace");
    // Two sessions in the project fixture => each row is <file>#<short id>.
    expect(r.out).toContain("project.ctrace#");
    expect(r.out).toMatch(/ \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [+-]\d{2}:\d{2} /);
  });

  it("agents aggregates the project's agents across sessions", () => {
    const r = run(["agents", "--project", fx.project]);
    expect(r.code).toBe(0);
    expect(r.out).toContain("researcher  sessions=2  calls=4");
    expect(r.out).toContain("writer  sessions=2  calls=2");
  });

  it("an ambiguous project exits 2 and lists the sessions to pick from", () => {
    const r = run(["tokens", "--project", fx.project]);
    expect(r.code).toBe(2);
    expect(r.out).toBe("");
    expect(r.err).toContain("pass --session to pick one");
  });

  it("diff with the wrong --turn count is a usage error (exit 2)", () => {
    const r = run(["diff", "--turn", "1", fx.multiturn]);
    expect(r.code).toBe(2);
    expect(r.err).toContain("exactly two --turn flags");
  });

  it("a missing run path exits 1 with a clear message", () => {
    const r = run(["tokens", "--run", join(dir, "does-not-exist.ctrace")]);
    expect(r.code).toBe(1);
    expect(r.err).toContain("ctxdiff:");
  });

  it("no command prints usage and exits 2", () => {
    const r = run([]);
    expect(r.code).toBe(2);
    expect(r.out).toContain("usage: ctxdiff");
  });

  // --- CLI error-handling parity fixes (guarded without the Python venv) ----

  it("an unknown flag is a usage error, not a stack trace (exit 2)", () => {
    const r = run(["diff", "--turn", "1", "--turn", "2", "--bogus", fx.multiturn]);
    expect(r.code).toBe(2);
    expect(r.err).toContain("unrecognized arguments: --bogus");
    expect(r.err).not.toMatch(/at \w+.*\(.*:\d+:\d+\)/); // no stack frames
  });

  it("a non-integer --turn is a usage error (exit 2)", () => {
    const r = run(["tokens", "--turn", "abc", fx.multiturn]);
    expect(r.code).toBe(2);
    expect(r.err).toContain("argument --turn: invalid int value: 'abc'");
  });

  it("an out-of-range turn list uses Python list formatting [1, 2, 3]", () => {
    const r = run(["diff", "--turn", "1", "--turn", "99", fx.multiturn]);
    expect(r.code).toBe(1);
    expect(r.err).toContain("turn(s) [99] not found in this run (available turns: [1, 2, 3])");
  });

  it("renders a bidi/zero-width snippet with Python-style escapes", () => {
    const r = run(["diff", "--turn", "1", "--turn", "2", fx.bidi]);
    expect(r.code).toBe(0);
    // ZWSP U+200B and bidi LRM/RLM appear escaped, not raw.
    expect(r.out).toContain("\\u200b");
    expect(r.out).toContain("\\u200e");
    expect(r.out).toContain("\\u200f");
  });

  // --- viewer / demo commands -----------------------------------------------

  it("export writes a self-contained .html and prints its path (exit 0)", () => {
    const outHtml = join(dir, "smoke-export.html");
    const r = run(["export", "--run", fx.multiturn, "--out", outHtml]);
    expect(r.code).toBe(0);
    expect(r.out.trim()).toBe(outHtml);
    expect(existsSync(outHtml)).toBe(true);
    const html = readFileSync(outHtml, "utf-8");
    expect(html).toContain("<!DOCTYPE html>");
    expect(html).not.toMatch(/https?:\/\//); // self-contained
  });

  it("view --no-open writes a dashboard and prints the path (exit 0, no browser)", () => {
    const r = run(["view", "--no-open", "--run", fx.multiturn]);
    expect(r.code).toBe(0);
    const p = r.out.trim();
    expect(p.endsWith(".html")).toBe(true);
    expect(existsSync(p)).toBe(true);
    rmSync(p, { force: true });
  });

  it("export --agent preselects the level; an unknown one exits 2 with the list", () => {
    // `--agent` on the dashboard commands PRESELECTS which of the three levels
    // the page opens on rather than filtering what the file contains — the HTML
    // still covers the whole project either way.
    const outHtml = join(dir, "smoke-agent.html");
    const ok = run(["export", "--run", fx.multiagent, "--agent", "researcher", "--out", outHtml]);
    expect(ok.code).toBe(0);
    const html = readFileSync(outHtml, "utf-8");
    expect(html).toContain('"start": {"level": 3, "agent": "researcher"');
    expect(html).toContain('"name": "writer"'); // the other agent is still there

    const bad = run(["view", "--no-open", "--run", fx.multiagent, "--agent", "nobody"]);
    expect(bad.code).toBe(2);
    expect(bad.err).toContain("no agent 'nobody' in this project");
    expect(bad.err).toContain("researcher");
  });

  it("demo's dashboard lands on the agent listing (two agents in the sample)", () => {
    const demoCtrace = join(dir, "smoke-demo-levels.ctrace");
    const r = run(["demo", "--no-open", "--out", demoCtrace]);
    expect(r.code).toBe(0);
    const htmlPath = r.out
      .split("\n")
      .find((l) => l.startsWith("dashboard"))!
      .split("->")[1]
      .trim();
    const html = readFileSync(htmlPath, "utf-8");
    expect(html).toContain('"start": {"level": 1, "agent": null, "session": null}');
    expect(html).toContain('"name": "researcher"');
    expect(html).toContain('"name": "writer"');
    expect(html).not.toMatch(/https?:\/\//);
  });

  it("demo --no-open --out writes a sample trace + dashboard (exit 0)", () => {
    const demoCtrace = join(dir, "smoke-demo.ctrace");
    const r = run(["demo", "--no-open", "--out", demoCtrace]);
    expect(r.code).toBe(0);
    expect(r.out).toContain("sample trace  ->");
    expect(r.out).toContain("dashboard     ->");
    expect(existsSync(demoCtrace)).toBe(true);
    expect(existsSync(join(dir, "smoke-demo.html"))).toBe(true);
  });
});
