/**
 * `ctxdiff check` — the CI gate (JS side).
 *
 * Two properties are worth more than any individual assertion here, and most of
 * this file exists to defend them:
 *
 * 1. **A check must never pass by not looking.** No assertions, an empty
 *    session, an `--agent` matching nobody — every one is a non-zero exit,
 *    never a green tick over an unexamined trace. A CI gate whose failure mode
 *    is "silently verified nothing" is worse than no gate.
 * 2. **`check` and the analysis commands can never disagree.** The thresholds
 *    are compared against the very numbers `ctxdiff tokens` and `ctxdiff cache`
 *    print, so a red build and a hand-run report always tell one story.
 *
 * Byte-for-byte agreement with the PYTHON `ctxdiff check` is asserted in
 * `analyze-conformance.test.ts`, which runs both CLIs over the same fixtures;
 * this file covers the behavior that is the same on both sides.
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { main } from "../src/cli.js";
import { CTrace } from "../src/store/ctrace.js";
import { analyzeCheck, checkPassed, type Thresholds } from "../src/analyze/check.js";
import { configure } from "../src/store/config.js";
import type {
  Call,
  CallBlock,
  ReadableStore,
  Run,
  Session,
  Store,
  StoreBackend,
} from "../src/store/base.js";
import { makeFixtures } from "./helpers/fixtures.js";

let dir: string;
let fx: ReturnType<typeof makeFixtures>;
const savedNoColor = process.env.NO_COLOR;

beforeAll(() => {
  // The report is compared as plain text and pasted into CI logs; ANSI escapes
  // would be literal garbage in both.
  process.env.NO_COLOR = "1";
  dir = mkdtempSync(join(tmpdir(), "ctxdiff-check-"));
  fx = makeFixtures(dir);
});
afterAll(() => {
  rmSync(dir, { recursive: true, force: true });
  configure({ store: null });
  if (savedNoColor === undefined) delete process.env.NO_COLOR;
  else process.env.NO_COLOR = savedNoColor;
});

/** A Thresholds with everything off — the base every case turns exactly one
 * thing on from, so a verdict can only be about that one thing. */
function none(): Thresholds {
  return {
    maxContext: null,
    contextWindow: null,
    maxContextPct: null,
    requireStablePrefix: false,
    noDeadSchemas: false,
    maxGrowth: null,
    maxGrowthPct: null,
  };
}

/** Run the CLI in-process with both streams captured. */
async function runCli(argv: string[]): Promise<{ code: number; out: string; err: string }> {
  const outChunks: string[] = [];
  const errChunks: string[] = [];
  const origOut = process.stdout.write.bind(process.stdout);
  const origErr = process.stderr.write.bind(process.stderr);
  // @ts-expect-error narrow override of the write signature for capture
  process.stdout.write = (s: string) => (outChunks.push(String(s)), true);
  // @ts-expect-error narrow override of the write signature for capture
  process.stderr.write = (s: string) => (errChunks.push(String(s)), true);
  let code: number;
  try {
    code = await main(argv);
  } finally {
    process.stdout.write = origOut;
    process.stderr.write = origErr;
  }
  return { code, out: outChunks.join(""), err: errChunks.join("") };
}

/** Run one analyzer pass over a fixture and hand back the report. */
function check(path: string, t: Partial<Thresholds>, agent: string | null = null) {
  const ct = CTrace.open(path);
  try {
    return analyzeCheck(ct, { ...none(), ...t }, agent);
  } finally {
    ct.close();
  }
}

/** Every turn's total, straight from the analyzer the check reads — used to
 * derive expectations instead of hard-coding tokenizer output, so a re-pin
 * moves the fixture's numbers without silently invalidating these tests. */
function totals(path: string): number[] {
  const report = check(path, { maxContext: 1 });
  return report.assertions[0].details.map((d) =>
    Number(/· ([\d,]+) tok/.exec(d)![1].replace(/,/g, "")),
  );
}

// --- the analyzer: each assertion, passing and failing ---------------------------

describe("max-context", () => {
  it("passes under budget and still reports the run's high-water mark", () => {
    const report = check(fx.multiturn, { maxContext: 1_000_000 });
    expect(checkPassed(report)).toBe(true);
    const [a] = report.assertions;
    expect(a.name).toBe("max-context");
    // A PASS that only says "PASS" tells nobody how close the run came, which
    // is the number worth watching across successive pull requests.
    expect(a.summary).toMatch(/^peak [\d,]+ tok at turn \d+ · limit 1,000,000$/);
    expect(a.details).toEqual([]);
  });

  it("fails naming every offending turn and by how much it went over", () => {
    const report = check(fx.multiturn, { maxContext: 1 });
    expect(checkPassed(report)).toBe(false);
    const [a] = report.assertions;
    expect(a.summary).toMatch(/^3 turns over limit · peak [\d,]+ tok at turn \d+ · limit 1$/);
    expect(a.details).toHaveLength(3);
    for (const [i, d] of a.details.entries()) {
      expect(d).toMatch(new RegExp(`^turn ${i + 1} · [\\d,]+ tok · [\\d,]+ over limit$`));
    }
  });
});

describe("max-context-pct", () => {
  it("measures against the window the USER supplies and states the token budget", () => {
    // ctxdiff ships no model→window table by design, so the denominator is an
    // input — and the report says what percentage works out to in tokens, so
    // no reader has to recompute it.
    const peak = Math.max(...totals(fx.multiturn));
    const report = check(fx.multiturn, { contextWindow: 1000, maxContextPct: 50 });
    const [a] = report.assertions;
    expect(a.name).toBe("max-context-pct");
    expect(checkPassed(report)).toBe(peak <= 500);
    expect(a.summary).toContain("of 1,000 tok window at turn");
    expect(a.summary).toContain("limit 50.0% (500 tok)");
  });

  it("fails when a turn exceeds the percentage of the window", () => {
    const peak = Math.max(...totals(fx.multiturn));
    // A window that makes the peak turn exactly over a 50% limit.
    const report = check(fx.multiturn, { contextWindow: peak, maxContextPct: 50 });
    expect(checkPassed(report)).toBe(false);
    expect(report.assertions[0].details.at(-1)).toContain("100.0% of");
  });
});

describe("require-stable-prefix", () => {
  it("passes on an append-only run and reports the guaranteed prefix", () => {
    const report = check(fx.multiagent, { requireStablePrefix: true });
    expect(checkPassed(report)).toBe(true);
    expect(report.assertions[0].summary).toMatch(
      /^prefix stable across all 2 turn pairs · min stable prefix [\d,]+ tok$/,
    );
  });

  it("fails naming the turn pair, the culprit block and the analyzer's own reason", () => {
    // Every word of the explanation is the cache profiler's; nothing is
    // re-derived here, so `check` and `cache` cannot tell different stories.
    const report = check(fx.dynamic, { requireStablePrefix: true });
    expect(checkPassed(report)).toBe(false);
    const [a] = report.assertions;
    expect(a.summary).toMatch(/^2 breaks across 2 turn pairs · [\d,]+ tok re-billed$/);
    expect(a.details).toHaveLength(1); // one culprit, collapsed across both pairs
    expect(a.details[0]).toContain("turn 1 → turn 2 [system·modified] breaks 2/2 pairs — ");
    expect(a.details[0]).toContain("first difference at char");
  });
});

describe("no-dead-schemas", () => {
  it("fails naming each registered-but-never-invoked tool", () => {
    const report = check(fx.multiturn, { noDeadSchemas: true });
    expect(checkPassed(report)).toBe(false);
    const [a] = report.assertions;
    expect(a.summary).toMatch(/^2 of 2 registered tools never used · [\d,]+ tok\/call \(/);
    expect(a.details).toEqual([
      "tool schema 'web_search' registered but never invoked",
      "tool schema 'calculator' registered but never invoked",
    ]);
  });

  it("passes on a run with no tool schemas and says there are none", () => {
    // Reporting `0 of 0` would read like a measurement that was never taken.
    const report = check(fx.multiagent, { noDeadSchemas: true });
    expect(checkPassed(report)).toBe(true);
    expect(report.assertions[0].summary).toBe("no tool schemas registered");
  });
});

describe("growth", () => {
  it("reports the largest single-turn jump with both totals", () => {
    // "grew by 30" is only interpretable next to what it grew from.
    const report = check(fx.multiturn, { maxGrowth: 0 });
    expect(checkPassed(report)).toBe(false);
    for (const d of report.assertions[0].details) {
      expect(d).toMatch(/^turn \d+ → turn \d+ · \+[\d,]+ tok \([\d,]+ → [\d,]+\) · limit 0$/);
    }
  });

  it("passes with a generous limit and still names the peak", () => {
    const report = check(fx.multiturn, { maxGrowth: 1_000_000 });
    expect(checkPassed(report)).toBe(true);
    expect(report.assertions[0].summary).toMatch(
      /^peak growth [\d,-]+ tok at turn \d+ · limit 1,000,000$/,
    );
  });

  it("measures growth WITHIN an agent, never across a hand-off", () => {
    // THE multi-agent correctness property. The adjacent pairs on this timeline
    // are researcher→writer and writer→researcher; measuring across those would
    // report jumps that describe nothing but the hand-offs. Pairing within each
    // agent — the cache profiler's rule — reports the two REAL growths.
    const report = check(fx.multiagent, { maxGrowth: 0 });
    const details = report.assertions[0].details;
    expect(details).toHaveLength(2);
    expect(details[0]).toContain("turn 1 → turn 3 [agent:researcher]");
    expect(details[1]).toContain("turn 2 → turn 4 [agent:writer]");
  });

  it("expresses percentage growth against the previous turn", () => {
    const report = check(fx.multiturn, { maxGrowthPct: 0 });
    expect(checkPassed(report)).toBe(false);
    for (const d of report.assertions[0].details) {
      expect(d).toMatch(
        /^turn \d+ → turn \d+ · \+\d+\.\d% \([\d,]+ → [\d,]+ tok\) · limit 0\.0%$/,
      );
    }
  });

  it("says WHY it passed when a run has no pair to measure at all", () => {
    // A one-turn run has nothing to grow from and no prefix to be stable
    // against. Both assertions pass — there is nothing to violate — but each
    // states that rather than claiming a stability/flatness it never measured.
    const source = CTrace.open(fx.multiturn);
    let firstTurnBlocks;
    try {
      firstTurnBlocks = source.getCallBlocks(source.getCalls()[0].id);
    } finally {
      source.close();
    }
    const path = join(dir, "single.ctrace");
    const ct = CTrace.create(path, "single", "openai", "", "2026-07-09T00:00:00Z");
    ct.recordCall({
      seq: 1,
      params: { model: "gpt-4o" },
      usage: null,
      latencyMs: 10,
      error: null,
      callBlocks: firstTurnBlocks,
      provider: "openai",
    });
    ct.close();

    const report = check(path, { maxGrowth: 0, maxGrowthPct: 0, requireStablePrefix: true });
    expect(checkPassed(report)).toBe(true);
    expect(report.assertions.map((a) => a.summary)).toEqual([
      "fewer than 2 turns — no pairs to check",
      "fewer than 2 turns — no growth to measure",
      "fewer than 2 turns — no growth to measure",
    ]);
  });
});

describe("scope and ordering", () => {
  it("scopes every assertion to one agent", () => {
    const report = check(fx.multiagent, { maxContext: 1 }, "writer");
    expect(report.agent).toBe("writer");
    expect(report.turnsAnalyzed).toBe(2);
    for (const d of report.assertions[0].details) expect(d).toContain("[agent:writer]");
  });

  it("reports assertions in a fixed order regardless of how they were requested", () => {
    // So two workflows asking for the same assertions get the same output, and
    // a diff of two CI logs is meaningful.
    const report = check(fx.multiturn, {
      maxGrowthPct: 1e6,
      noDeadSchemas: true,
      maxContext: 1e6,
      requireStablePrefix: true,
      maxGrowth: 1e6,
      contextWindow: 1e6,
      maxContextPct: 99,
    });
    expect(report.assertions.map((a) => a.name)).toEqual([
      "max-context",
      "max-context-pct",
      "require-stable-prefix",
      "no-dead-schemas",
      "max-growth",
      "max-growth-pct",
    ]);
  });

  it("includes ONLY the assertions that were requested", () => {
    // An assertion nobody asked for is absent, not a vacuous PASS: a report
    // listing unrequested checks would let a reader believe a budget is being
    // enforced when nothing is enforcing it.
    const report = check(fx.multiturn, { noDeadSchemas: true });
    expect(report.assertions.map((a) => a.name)).toEqual(["no-dead-schemas"]);
  });
});

// --- unmeasured turns: a floor is never certified ---------------------------------

describe("turns whose total is a floor", () => {
  it("refuses to certify a max-context budget instead of passing on the floor", () => {
    // THE false-pass regression. Every turn of this fixture holds a remote-URL
    // image whose cost ctxdiff cannot know, so the stored totals (4 and 8) are
    // lower bounds on the 800 and 1,600 tokens the provider actually billed.
    // Comparing that to a 500-token budget can only ever produce a PASS, and it
    // would be silent and permanent.
    const report = check(fx.unmeasured, { maxContext: 500 });
    expect(checkPassed(report)).toBe(false);
    const [a] = report.assertions;
    expect(a.summary).toBe("2 turns unmeasured · peak 2 tok (~approx) at turn 2 · limit 500");
    expect(a.details).toEqual([
      "turn 1 · 1 tok (~approx) · 1 block of unknown token cost — a floor, not a measurement · limit 500",
      "turn 2 · 2 tok (~approx) · 2 blocks of unknown token cost — a floor, not a measurement · limit 500",
    ]);
  });

  it("reports a turn that is BOTH over the limit and unmeasured exactly once", () => {
    // The overage is already proved and the floor cannot un-prove it, so it is
    // an over-limit violation and not also an unmeasured one.
    const report = check(fx.unmeasured, { maxContext: 1 });
    const [a] = report.assertions;
    expect(a.summary).toBe(
      "1 turn over limit · 1 turn unmeasured · peak 2 tok (~approx) at turn 2 · limit 1",
    );
    expect(a.details).toEqual([
      "turn 2 · 2 tok (~approx) · 1 over limit",
      "turn 1 · 1 tok (~approx) · 1 block of unknown token cost — a floor, not a measurement · limit 1",
    ]);
  });

  it("refuses a percentage budget too, quoting the percentage as the bound it is", () => {
    const report = check(fx.unmeasured, { contextWindow: 1000, maxContextPct: 50 });
    expect(checkPassed(report)).toBe(false);
    const [a] = report.assertions;
    expect(a.summary).toBe(
      "2 turns unmeasured · peak 0.2% (~approx) of 1,000 tok window at turn 2 · " +
        "limit 50.0% (500 tok)",
    );
    expect(a.details[0]).toBe(
      "turn 1 · 1 tok (~approx) · 0.1% of 1,000 tok window · 1 block of unknown " +
        "token cost — a floor, not a measurement · limit 50.0% (500 tok)",
    );
  });

  it("refuses a growth budget when either turn of the pair is unmeasured", () => {
    // The error runs both ways — an unmeasured EARLIER turn overstates the
    // growth, an unmeasured LATER one understates it — so neither a pass nor a
    // numeric violation would be defensible.
    const report = check(fx.unmeasured, { maxGrowth: 1000 });
    expect(checkPassed(report)).toBe(false);
    const [a] = report.assertions;
    expect(a.summary).toBe(
      "1 turn unmeasured · peak growth 1 tok (~approx) at turn 2 · limit 1,000",
    );
    expect(a.details).toEqual([
      "turn 1 → turn 2 · +1 tok (~approx) (1 → 2) · 3 blocks of unknown token cost — " +
        "a floor, not a measurement · limit 1,000",
    ]);
  });

  it("reports an unmeasured growth PAIR without a percentage", () => {
    // The percentage would be derived from a floor, and quoting it beside a
    // limit is exactly the confusion the refusal exists to avoid.
    const report = check(fx.unmeasured, { maxGrowthPct: 500 });
    expect(checkPassed(report)).toBe(false);
    expect(report.assertions[0].details).toEqual([
      "turn 1 → turn 2 · 1 → 2 tok (~approx) · 3 blocks of unknown token cost — " +
        "a floor, not a measurement · limit 500.0%",
    ]);
  });

  it("does not call an ordinary estimated turn unmeasured", () => {
    // The distinction that keeps this from crying wolf: an estimate is a
    // number, and a run of them still passes — marked `(~approx)`, never
    // refused.
    const report = check(fx.multiagent, { maxContext: 1_000_000 });
    expect(checkPassed(report)).toBe(true);
    expect(report.assertions[0].summary).not.toContain("unmeasured");
  });
});

// --- an assertion with nothing to measure says WHICH nothing ----------------------

describe("nothing to pair", () => {
  it("distinguishes a fan-out from a single-turn run", () => {
    // Four turns, four agents, one turn each. Pairing is per-agent by design,
    // so every adjacent pair is a hand-off and none is analyzable — but
    // "fewer than 2 turns" over a four-turn run reads as a typo and hides the
    // fact that three assertions measured nothing at all.
    const report = check(fx.fanout, {
      requireStablePrefix: true,
      maxGrowth: 0,
      maxGrowthPct: 0,
    });
    expect(checkPassed(report)).toBe(true);
    expect(report.assertions.map((a) => a.summary)).toEqual([
      "no consecutive same-agent pairs (4 agents, 1 turn each) — no pairs to check",
      "no consecutive same-agent pairs (4 agents, 1 turn each) — no growth to measure",
      "no consecutive same-agent pairs (4 agents, 1 turn each) — no growth to measure",
    ]);
  });

  it("still says 'fewer than 2 turns' when there really are fewer than 2", () => {
    const report = check(fx.project, {
      requireStablePrefix: true,
      maxGrowth: 0,
      maxGrowthPct: 0,
    }, "writer");
    expect(report.assertions.map((a) => a.summary)).toEqual([
      "fewer than 2 turns — no pairs to check",
      "fewer than 2 turns — no growth to measure",
      "fewer than 2 turns — no growth to measure",
    ]);
  });
});

// --- the CLI: exit codes and messages ----------------------------------------------

describe("ctxdiff check CLI", () => {
  it("exits 0 and prints a pass verdict", async () => {
    const { code, out } = await runCli([
      "check", "--project", fx.multiagent,
      "--max-context", "1000000", "--require-stable-prefix", "--no-dead-schemas",
    ]);
    expect(code).toBe(0);
    expect(out.startsWith("ctxdiff check · 4 turns · session ")).toBe(true);
    expect(out).toContain("check passed · 3 assertions");
    expect(out).not.toContain("FAIL");
  });

  it("exits 1 on a violation so CI fails the build", async () => {
    // The headline contract. Without a non-zero exit the whole feature is a
    // pretty report nobody's CI reacts to.
    const { code, out } = await runCli([
      "check", "--project", fx.multiturn, "--max-context", "1000000", "--no-dead-schemas",
    ]);
    expect(code).toBe(1);
    expect(out).toContain("check FAILED · 1 of 2 assertions failed");
    expect(out).toContain("FAIL  no-dead-schemas");
    expect(out).toContain("PASS  max-context");
  });

  it("exits 2 with no assertions rather than passing vacuously", async () => {
    const { code, err } = await runCli(["check", "--project", fx.multiturn]);
    expect(code).toBe(2);
    expect(err).toContain("nothing to assert");
    for (const flag of [
      "--max-context", "--max-context-pct", "--require-stable-prefix",
      "--no-dead-schemas", "--max-growth", "--max-growth-pct",
    ]) {
      expect(err).toContain(flag);
    }
  });

  it("exits 2 for a percentage with no window, and for a window with no percentage", async () => {
    const a = await runCli(["check", "--project", fx.multiturn, "--max-context-pct", "80"]);
    expect(a.code).toBe(2);
    expect(a.err).toContain("--max-context-pct needs a denominator");

    const b = await runCli([
      "check", "--project", fx.multiturn, "--max-context", "8000", "--context-window", "8000",
    ]);
    expect(b.code).toBe(2);
    expect(b.err).toContain("--context-window is only used by --max-context-pct");
  });

  it("exits 2 for out-of-range limits, but allows a growth budget of exactly zero", async () => {
    // A zero window is a division by zero and a negative growth budget is not a
    // thing to assert — but "this context must not grow at all" is.
    const zero = await runCli(["check", "--project", fx.multiturn, "--max-context", "0"]);
    expect(zero.code).toBe(2);
    expect(zero.err).toContain("--max-context must be greater than 0 (got 0)");

    const neg = await runCli(["check", "--project", fx.multiturn, "--max-growth", "-5"]);
    expect(neg.code).toBe(2);
    expect(neg.err).toContain("--max-growth cannot be negative (got -5)");

    const ok = await runCli(["check", "--project", fx.multiturn, "--max-growth", "0"]);
    expect(ok.code).toBe(1); // a real verdict, not a usage error
  });

  it("exits 2 for a non-numeric threshold", async () => {
    const i = await runCli(["check", "--project", fx.multiturn, "--max-context", "abc"]);
    expect(i.code).toBe(2);
    expect(i.err).toContain("argument --max-context: invalid int value: 'abc'");

    const f = await runCli([
      "check", "--project", fx.multiturn, "--max-context-pct", "1e5", "--context-window", "10",
    ]);
    expect(f.code).toBe(2);
    expect(f.err).toContain("argument --max-context-pct: invalid float value: '1e5'");
  });

  it("exits 1 on the unmeasured repro rather than passing a 1,600-token turn", async () => {
    const { code, out } = await runCli([
      "check", "--project", fx.unmeasured, "--max-context", "500",
    ]);
    expect(code).toBe(1);
    expect(out).toContain("FAIL  max-context");
    expect(out).not.toContain("PASS");
    expect(out).toContain("a floor, not a measurement");
  });

  it("names the trace the verdict was computed from", async () => {
    // With no `--project` the CLI reads the most recently modified `*.ctrace`
    // in the working directory — the GitHub Action's default — so an unrelated
    // newer trace can be checked, pass, and leave a report indistinguishable
    // from one over the intended run.
    const cwd = process.cwd();
    try {
      process.chdir(dir);
      const { out } = await runCli(["check", "--max-context", "1000000"]);
      expect(out.split("\n")[0]).toMatch(
        /^ctxdiff check · \d+ turns? · [\w.-]+\.ctrace \(session [0-9a-f]{12}\)$/,
      );
    } finally {
      process.chdir(cwd);
    }
  });

  it("reports a NAMED project as its session id alone", async () => {
    // The filename adds nothing when it is already in the command just run.
    const { out } = await runCli([
      "check", "--project", fx.multiturn, "--max-context", "1000000",
    ]);
    expect(out.split("\n")[0]).toMatch(/^ctxdiff check · 3 turns · session [0-9a-f]{12}$/);
  });

  it("rejects a stray positional instead of adopting it as --project", async () => {
    // `--require-stable-prefix` is a boolean flag, so `--require-stable-prefix
    // false` leaves `false` as a positional nothing claims. Python's `check`
    // subparser registers none, so it is exit 2 there — adopting it here as a
    // project path swallowed the mistake and reported on a different trace.
    const { code, err } = await runCli([
      "check", "--project", fx.multiturn, "--require-stable-prefix", "false",
    ]);
    expect(code).toBe(2);
    expect(err).toContain("unrecognized arguments: false");
  });

  it("rejects --turn, which a whole-run budget cannot honor", async () => {
    const { code, err } = await runCli([
      "check", "--project", fx.multiturn, "--turn", "2", "--max-context", "10",
    ]);
    expect(code).toBe(2);
    expect(err).toContain("unrecognized arguments: --turn");
  });

  it("exits 2 and lists the real agents for a typo'd --agent", async () => {
    // A typo must not filter every call away and report a table of passes —
    // that is a check that would stay green forever.
    const { code, err } = await runCli([
      "check", "--project", fx.multiagent, "--agent", "resercher", "--max-context", "1",
    ]);
    expect(code).toBe(2);
    expect(err).toContain("researcher");
  });

  it("scopes the check to one agent and says so in the header", async () => {
    const bad = await runCli([
      "check", "--project", fx.multiagent, "--agent", "researcher", "--max-context", "1",
    ]);
    expect(bad.code).toBe(1);
    expect(bad.out.startsWith("ctxdiff check · 2 turns · agent researcher · session ")).toBe(
      true,
    );

    const good = await runCli([
      "check", "--project", fx.multiagent, "--agent", "researcher",
      "--max-context", "1000000",
    ]);
    expect(good.code).toBe(0);
  });

  it("emits no ANSI escapes under NO_COLOR", async () => {
    const { out } = await runCli([
      "check", "--project", fx.multiturn, "--no-dead-schemas",
    ]);
    expect(out).not.toContain("\x1b[");
  });

  it("exits 1 — not 0 — when there is no trace to check", async () => {
    // No trace is a FAILURE. The day capture silently breaks, a check that
    // greened on an absent trace would keep the build green forever.
    const empty = mkdtempSync(join(tmpdir(), "ctxdiff-empty-"));
    const cwd = process.cwd();
    try {
      process.chdir(empty);
      const { code, err } = await runCli(["check", "--max-context", "100"]);
      expect(code).toBe(1);
      expect(err).toContain("no .ctrace here");
    } finally {
      process.chdir(cwd);
      rmSync(empty, { recursive: true, force: true });
    }
  });

  it("lists `check` in the usage text", async () => {
    const { code, out } = await runCli([]);
    expect(code).toBe(2);
    expect(out).toContain("check [assertions]");
  });
});

// --- the same check against a NETWORKED store ----------------------------------------

/**
 * A minimal asynchronous `Store` + `StoreBackend` pair standing in for
 * Postgres/MySQL, built by reading a real `.ctrace` and answering every read
 * through a promise.
 *
 * Why a fake rather than the real adapters: what `check` has to prove here is
 * that it reaches a NETWORKED store at all — which in this SDK means going
 * through `snapshotStore`, since every analyzer is synchronous and a database
 * is not. That path is identical for Postgres, MySQL and this, and the adapters'
 * own SQL is covered end-to-end in `store-backends.test.ts`. Nothing is
 * mocked away that `check` itself depends on.
 */
function networkBackend(path: string): StoreBackend {
  const source = CTrace.open(path);
  const sessions = source.listSessions();
  const run = source.getRun();
  const calls = new Map<string, Call[]>(sessions.map((s) => [s.id, source.getCalls(s.id)]));
  const blocks = new Map<string, CallBlock[]>();
  for (const list of calls.values()) {
    for (const c of list) blocks.set(c.id, source.getCallBlocks(c.id));
  }
  source.close();

  const store: Store = {
    recordCall: async () => "unused",
    noteModel: async () => {},
    listSessions: async (): Promise<Session[]> => sessions,
    getRun: async (): Promise<Run> => run,
    getCalls: async (sessionId?: string): Promise<Call[]> =>
      calls.get(sessionId ?? run.id) ?? [],
    getCallBlocks: async (callId: string): Promise<CallBlock[]> => blocks.get(callId) ?? [],
    close: async () => {},
  };
  // No `pathFor`: that ABSENCE is the capability check the CLI uses to tell a
  // networked backend from a file one, so omitting it is what makes this fake
  // take the database code path.
  return { openReader: async () => store } as unknown as StoreBackend;
}

describe("ctxdiff check against a configured database", () => {
  it("gates a networked store with the same verdicts as a .ctrace", async () => {
    // The team most likely to WANT a CI gate is the one already pointing
    // several containers at a shared store; a check that only worked on a file
    // would be unavailable to exactly them.
    configure({ store: networkBackend(fx.multiturn) });
    try {
      const fail = await runCli(["check", "--max-context", "1000000", "--no-dead-schemas"]);
      expect(fail.code).toBe(1);
      expect(fail.out).toContain("ctxdiff check · 3 turns");
      expect(fail.out).toContain("FAIL  no-dead-schemas");
      expect(fail.out).toContain("tool schema 'web_search' registered but never invoked");

      const pass = await runCli(["check", "--max-context", "1000000"]);
      expect(pass.code).toBe(0);
    } finally {
      configure({ store: null });
    }
  });

  it("produces the same report for a database and for the file behind it", async () => {
    // Storage is not allowed to change a verdict. Same trace, two backends, one
    // byte-identical report.
    const fromFile = await runCli([
      "check", "--project", fx.multiturn, "--max-context", "1000000",
      "--no-dead-schemas", "--max-growth", "0",
    ]);
    configure({ store: networkBackend(fx.multiturn) });
    let fromDb;
    try {
      fromDb = await runCli([
        "check", "--max-context", "1000000", "--no-dead-schemas", "--max-growth", "0",
      ]);
    } finally {
      configure({ store: null });
    }
    expect(fromDb.out).toBe(fromFile.out);
    expect(fromDb.code).toBe(fromFile.code);
  });
});
