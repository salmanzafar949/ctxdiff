/**
 * The context window: resolving it, and rendering a turn as a share of it. The
 * JS twin of `tests/test_window.py`, assertion for assertion.
 *
 * The property defended here is that ctxdiff never invents a denominator. There
 * is no model→context-window table in the package (the same decision that keeps
 * a price table out), so a percentage appears only when the user has stated the
 * window — as a flag or as `CTXDIFF_CONTEXT_WINDOW` — and with no window every
 * command prints exactly the bytes it printed before percentages existed.
 *
 * The second property is that the flag and the environment variable are resolved
 * in ONE place, so `tokens`, `check` and the exported dashboard can never be
 * scored against two different windows on the same machine.
 */
import { describe, it, expect, beforeAll, afterAll, beforeEach } from "vitest";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { main } from "../src/cli.js";
import { bootPage } from "./helpers/page.js";
import { CTrace } from "../src/store/ctrace.js";
import type { CallBlock } from "../src/models.js";
import {
  CONTEXT_WINDOW_ALARM_PCT,
  CONTEXT_WINDOW_ENV,
  ContextWindowError,
  formatWindowShare,
  isAlarming,
  parseContextWindow,
  resolveContextWindow,
  windowPct,
} from "../src/analyze/window.js";

let dir: string;
let tracePath: string;
const originalEnv = process.env[CONTEXT_WINDOW_ENV];

/** A CallBlock with an explicit token count, so a turn's total is a fixture
 * constant rather than something the tokenizer decides. */
function cb(
  text: string,
  position: number,
  role = "user",
  label = "user",
  tokens = 10,
): CallBlock {
  return {
    block: {
      contentHash: `h:${text}`,
      role,
      kind: "message",
      text,
      tokenCount: tokens,
      tokenMethod: "tiktoken",
    },
    position,
    label,
    labelSource: "heuristic",
  };
}

beforeAll(() => {
  dir = mkdtempSync(join(tmpdir(), "ctxdiff-window-"));
  // Turn totals of exactly 40 and 90: against a 100-token window one turn sits
  // comfortably under the alarm threshold and the other is well past it.
  tracePath = join(dir, "w.ctrace");
  const ct = CTrace.create(tracePath, "demo", "openai", "gpt-4o", "2026-07-09T00:00:00Z");
  const turn1 = [cb("system", 0, "system", "system", 25), cb("hello", 1, "user", "user", 15)];
  const turn2 = [
    ...turn1,
    cb("reply", 2, "assistant", "history", 30),
    cb("again", 3, "user", "user", 20),
  ];
  ct.recordCall({
    seq: 1, params: { model: "gpt-4o" }, usage: null, latencyMs: 1, error: null,
    callBlocks: turn1, provider: "openai",
  });
  ct.recordCall({
    seq: 2, params: { model: "gpt-4o" }, usage: null, latencyMs: 1, error: null,
    callBlocks: turn2, provider: "openai",
  });
  ct.close();
  process.env.NO_COLOR = "1";
});
afterAll(() => {
  rmSync(dir, { recursive: true, force: true });
  if (originalEnv === undefined) delete process.env[CONTEXT_WINDOW_ENV];
  else process.env[CONTEXT_WINDOW_ENV] = originalEnv;
});
beforeEach(() => {
  delete process.env[CONTEXT_WINDOW_ENV];
});

/** Run the CLI in-process, capturing stdout and stderr. */
async function run(argv: string[]): Promise<{ code: number; out: string; err: string }> {
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

describe("context window resolution", () => {
  it("lets the flag win over the environment", () => {
    // A flag typed on this invocation is the most specific thing anyone said.
    process.env[CONTEXT_WINDOW_ENV] = "200000";
    expect(resolveContextWindow(8000)).toBe(8000);
  });

  it("reads the window from the environment when no flag does", () => {
    // The whole point of the variable: every command agrees without retyping.
    process.env[CONTEXT_WINDOW_ENV] = "200000";
    expect(resolveContextWindow(null)).toBe(200000);
  });

  it("returns null — never a guessed default — when neither is set", () => {
    expect(resolveContextWindow(null)).toBeNull();
  });

  it("treats an empty variable as unset rather than as an error", () => {
    // `CTXDIFF_CONTEXT_WINDOW= ctxdiff tokens` is how a shell unsets a variable
    // for one command; failing there would make that idiom unusable.
    process.env[CONTEXT_WINDOW_ENV] = "   ";
    expect(resolveContextWindow(null)).toBeNull();
  });

  it.each(["abc", "200k", "1e5", "12.5", "٢"])(
    "reports rather than ignores an unusable variable (%s)",
    (value) => {
      // A percentage that quietly stops rendering looks exactly like one that is
      // fine. The grammar is the same ASCII narrowing `--turn` uses, so a
      // non-ASCII digit Python's `int()` would accept is refused on both sides.
      process.env[CONTEXT_WINDOW_ENV] = value;
      expect(() => resolveContextWindow(null)).toThrow(ContextWindowError);
    },
  );

  it.each(["0", "-5"])("refuses a non-positive window (%s)", (value) => {
    expect(() => parseContextWindow(value)).toThrow(ContextWindowError);
  });

  it.each([0, -5])("refuses a non-positive FLAG too (%s)", (value) => {
    // The rule belongs to the WINDOW, not to the place it was typed. The
    // resolver used to hand the flag back unchecked, so the environment path
    // rejected a zero while the flag path rendered `⚠ Infinity%`, and
    // `--context-window -5` rendered `-260.0%`.
    delete process.env[CONTEXT_WINDOW_ENV];
    expect(() => resolveContextWindow(value)).toThrow(
      `ctxdiff: --context-window must be greater than 0 (got ${value})`,
    );
  });
});

describe("share-of-window rendering", () => {
  it("names both numbers and the percentage", () => {
    expect(formatWindowShare(18400, 200000)).toBe("18,400 / 200,000 tok · 9.2%");
  });

  it("trips the alarm marker at the documented threshold", () => {
    // Compared against the DISPLAYED percentage, so a turn shown as 80.0% is
    // marked and one shown as 79.9% is not.
    expect(isAlarming(CONTEXT_WINDOW_ALARM_PCT)).toBe(true);
    expect(isAlarming(CONTEXT_WINDOW_ALARM_PCT - 0.1)).toBe(false);
    expect(formatWindowShare(80, 100)).toContain("⚠");
    expect(formatWindowShare(79, 100)).not.toContain("⚠");
  });

  it("rounds to one decimal the way CPython does", () => {
    expect(windowPct(1, 3)).toBe(33.3);
    expect(formatWindowShare(1, 3)).toBe("1 / 3 tok · 33.3%");
  });
});

describe("the CLI", () => {
  it("leaves `tokens` unchanged when no window is stated", async () => {
    const { out } = await run(["tokens", "--project", tracePath]);
    expect(out).toContain("turn 1 · 40 tokens");
  });

  it("renders the share — quiet and alarming — from one flag", async () => {
    const { out } = await run(["tokens", "--project", tracePath, "--context-window", "100"]);
    expect(out).toContain("turn 1 · 40 / 100 tok · 40.0%");
    expect(out).toContain("turn 2 · 90 / 100 tok · ⚠ 90.0%");
  });

  it("reads the window from the environment with no flag typed", async () => {
    process.env[CONTEXT_WINDOW_ENV] = "100";
    const { out } = await run(["tokens", "--project", tracePath]);
    expect(out).toContain("turn 1 · 40 / 100 tok · 40.0%");
  });

  it("reports a bad environment window as a usage error, before opening a store", async () => {
    process.env[CONTEXT_WINDOW_ENV] = "lots";
    const { code, err } = await run(["tokens", "--project", tracePath]);
    expect(code).toBe(2);
    expect(err).toContain(CONTEXT_WINDOW_ENV);
  });

  it.each([
    ["tokens", "0"], ["tokens", "-5"],
    ["view", "0"], ["view", "-5"],
    ["export", "0"], ["export", "-5"],
  ])("refuses a non-positive --context-window on `%s` (%s)", async (command, value) => {
    // All four commands inherit the rule from the ONE resolver, so none of them
    // can render a percentage against a window that is not one. Exit 2 — a usage
    // error, the same code the environment path has always returned — instead of
    // an `⚠ Infinity%` dashboard or a `-260.0%` turn header.
    delete process.env[CONTEXT_WINDOW_ENV];
    const argv = [command, "--project", tracePath, "--context-window", value];
    if (command === "export") argv.push("--out", join(dir, "nonpositive.html"));
    const { code, err } = await run(argv);
    expect(code).toBe(2);
    expect(err).toContain(`ctxdiff: --context-window must be greater than 0 (got ${value})`);
  });

  it("lets `check` keep naming itself when it refuses a zero window", async () => {
    // Every usage error `check` emits is prefixed `ctxdiff check:` so a CI log
    // says which step spoke; the shared resolver must not take that away.
    delete process.env[CONTEXT_WINDOW_ENV];
    const { code, err } = await run([
      "check", "--project", tracePath, "--max-context-pct", "50", "--context-window", "0",
    ]);
    expect(code).toBe(2);
    expect(err).toContain("ctxdiff check: --context-window must be greater than 0 (got 0)");
  });

  it("scores `check` and `tokens` against the same window", async () => {
    // A gate scored against a window a human's report never saw would be a gate
    // nobody could audit.
    process.env[CONTEXT_WINDOW_ENV] = "100";
    const tokens = await run(["tokens", "--project", tracePath]);
    const check = await run(["check", "--project", tracePath, "--max-context-pct", "50"]);
    expect(tokens.out).toContain("90.0%");
    expect(check.code).toBe(1);
    expect(check.out).toContain("90.0% of 100 tok window");
  });

  it("never lets an ambient window trigger the unused-flag error", async () => {
    // Rule 3 of check's validation is about what the user TYPED.
    process.env[CONTEXT_WINDOW_ENV] = "100";
    const { code } = await run(["check", "--project", tracePath, "--max-context", "1000"]);
    expect(code).toBe(0);
  });

  it("still refuses a typed --context-window that nothing consumes", async () => {
    const { code } = await run([
      "check", "--project", tracePath, "--max-context", "1000", "--context-window", "100",
    ]);
    expect(code).toBe(2);
  });

  it("embeds the window and every turn's precomputed percentage in the dashboard", async () => {
    // The browser never rounds: both SDKs write the same digits into the island.
    process.env[CONTEXT_WINDOW_ENV] = "100";
    const out = join(dir, "d.html");
    const { code } = await run(["export", "--project", tracePath, "--out", out]);
    expect(code).toBe(0);
    const html = readFileSync(out, "utf-8");
    const island = /<script id="ctxdiff-data" type="application\/json">([\s\S]*?)<\/script>/.exec(
      html,
    )![1];
    const payload = JSON.parse(island.replace(/<\\\//g, "</"));
    expect(payload.tokens.context_window).toBe(100);
    expect(payload.tokens.window_alarm_pct).toBe(CONTEXT_WINDOW_ALARM_PCT);
    expect(payload.tokens.calls.map((c: { pct_of_window: number }) => c.pct_of_window)).toEqual([
      40.0, 90.0,
    ]);
  });

  it("shows the share in the rendered page, not just in the payload", async () => {
    // The payload says what the dashboard COULD show; only executing the page
    // says what it does. The turn panel's heading and the header's peak line are
    // the two places a window actually surfaces, and the alarming turn's bar
    // must also carry the percentage in its ACCESSIBLE label — a hot bar that
    // differs only by hue tells a screen-reader user nothing.
    process.env[CONTEXT_WINDOW_ENV] = "100";
    const out = join(dir, "d-page.html");
    expect((await run(["export", "--project", tracePath, "--out", out])).code).toBe(0);
    const page = bootPage(readFileSync(out, "utf-8"));
    expect(page.visibleLevel()).toBe(3); // single agent, single session
    expect(page.byId.get("alloc")!.text()).toContain("40 / 100 tok · 40.0%");
    expect(page.byId.get("h-meta")!.text()).toContain("peak 90 / 100 tok · ⚠ 90.0%");
    const bars = page.bars();
    expect(bars[1].attrs["aria-label"]).toContain("90 / 100 tok · ⚠ 90.0%");
    expect(bars[1].className).toContain("hot");
    expect(bars[0].className).not.toContain("hot");
  });

  it("renders the plain token count in the page when no window is known", async () => {
    const out = join(dir, "d-page-nowindow.html");
    expect((await run(["export", "--project", tracePath, "--out", out])).code).toBe(0);
    const page = bootPage(readFileSync(out, "utf-8"));
    expect(page.byId.get("alloc")!.text()).toContain("40 tokens");
    expect(page.byId.get("alloc")!.text()).not.toContain("/ 100");
    expect(page.bars().every((b) => !b.className.includes("hot"))).toBe(true);
  });

  it("embeds nulls — never an invented denominator — without a window", async () => {
    const out = join(dir, "d-nowindow.html");
    const { code } = await run(["export", "--project", tracePath, "--out", out]);
    expect(code).toBe(0);
    const html = readFileSync(out, "utf-8");
    const island = /<script id="ctxdiff-data" type="application\/json">([\s\S]*?)<\/script>/.exec(
      html,
    )![1];
    const payload = JSON.parse(island.replace(/<\\\//g, "</"));
    expect(payload.tokens.context_window).toBeNull();
    expect(
      payload.tokens.calls.every((c: { pct_of_window: number | null }) => c.pct_of_window === null),
    ).toBe(true);
  });
});
