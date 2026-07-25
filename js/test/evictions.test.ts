/**
 * The tagged-eviction detector. The JS twin of `tests/test_evictions.py`.
 *
 * Almost every test here is a COUNTER-example, and deliberately so. "The block
 * you tagged 'rag' at turn 3 was evicted at turn 6" is a strong claim — it tells
 * a developer their agent lost something it was told to keep — and a report that
 * cries wolf once is a report nobody reads again. So the detector is narrowed
 * three ways, and each narrowing gets a test that would fail loudly if it were
 * dropped: only TAGGED blocks; only within one agent's timeline; only when the
 * block never comes back.
 *
 * The fourth property is not a narrowing but a consequence of REUSING the
 * differ: a tagged block whose text was edited in place is `modified`, not
 * `evicted`, and a naive hash-presence scan would get that wrong.
 */
import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { main } from "../src/cli.js";
import { bootPage } from "./helpers/page.js";
import { CTrace } from "../src/store/ctrace.js";
import type { CallBlock } from "../src/models.js";
import { analyzeEvictions } from "../src/analyze/evictions.js";

let dir: string;

/** A CallBlock whose label SOURCE is the thing under test. `tagged` stands in
 * for what `tracer.tag()` produces during capture. */
function cb(
  text: string,
  position: number,
  opts: { role?: string; label?: string; tagged?: boolean; tokens?: number } = {},
): CallBlock {
  const role = opts.role ?? "user";
  const tagged = opts.tagged === true;
  return {
    block: {
      contentHash: `h:${role}:${text}`,
      role,
      kind: "message",
      text,
      tokenCount: opts.tokens ?? 10,
      tokenMethod: "tiktoken",
    },
    position,
    label: opts.label ?? (tagged ? "rag" : role),
    labelSource: tagged ? "tagged" : "heuristic",
  };
}

/** Write a trace from a list of [agent, blocks] turns, numbered from 1. */
function write(name: string, turns: [string | null, CallBlock[]][]): string {
  const path = join(dir, name);
  const ct = CTrace.create(path, "demo", "openai", "gpt-4o", "2026-07-10T00:00:00Z");
  turns.forEach(([agent, callBlocks], i) => {
    ct.recordCall({
      seq: i + 1, params: { model: "gpt-4o" }, usage: null, latencyMs: 1,
      error: null, callBlocks, agent, provider: "openai",
    });
  });
  ct.close();
  return path;
}

function report(path: string, agent: string | null = null) {
  const ct = CTrace.open(path);
  try {
    return analyzeEvictions(ct, agent);
  } finally {
    ct.close();
  }
}

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

beforeAll(() => {
  dir = mkdtempSync(join(tmpdir(), "ctxdiff-evict-"));
  process.env.NO_COLOR = "1";
});
afterAll(() => rmSync(dir, { recursive: true, force: true }));

const SYS = () => cb("system", 0, { role: "system", label: "system" });

/** The canonical positive case, reused by the `check` tests below. */
function evictingTrace(name: string): string {
  const rag = cb("Enterprise pricing FAQ", 1, { tagged: true, tokens: 42 });
  return write(name, [
    [null, [SYS(), rag]],
    [null, [SYS(), rag, cb("reply", 2, { role: "assistant" })]],
    [null, [SYS(), cb("reply", 1, { role: "assistant" })]],
  ]);
}

describe("the positive case", () => {
  it("names both the turn a tagged block entered and the turn it left", () => {
    const r = report(evictingTrace("positive.ctrace"));
    expect(r.evictions).toHaveLength(1);
    const e = r.evictions[0];
    expect([e.label, e.enteredSeq, e.lastSeenSeq, e.evictedSeq]).toEqual(["rag", 1, 2, 3]);
    expect(e.tokens).toBe(42);
    expect(r.taggedBlocks).toBe(1);
    expect(r.pairsAnalyzed).toBe(2);
  });
});

describe("taggedness is a property of the content, not of one call", () => {
  it("keeps a block tagged on ONE call tagged for the whole run", () => {
    // The defect this module shipped with, and the exact shape of its own
    // headline example. `tracer.tag()` is next-call-only BY DESIGN — it buffers
    // for the NEXT recorded call and clears itself — so a block tagged once is
    // `labelSource === "tagged"` on exactly ONE call and `heuristic` on every
    // later one carrying the same text. An eviction is detected on the pair
    // (turn 4, turn 5), and asking "was it tagged on turn 4?" answers no for a
    // block tagged on turn 1, so the feature was silent on the one story it
    // advertises: tag at turn 1, evicted at turn 5.
    const taggedRag = cb("Enterprise pricing FAQ", 1, { tagged: true, tokens: 42 });
    // The same text one turn later with only the role heuristic — what the
    // recorder writes once tag()'s one-call buffer is spent.
    const plainRag = cb("Enterprise pricing FAQ", 1, { tokens: 42 });
    expect(plainRag.block.contentHash).toBe(taggedRag.block.contentHash);
    const r = report(
      write("tag-once.ctrace", [
        [null, [SYS(), taggedRag]],
        [null, [SYS(), plainRag, cb("a", 2, { role: "assistant" })]],
        [null, [SYS(), plainRag, cb("a", 2, { role: "assistant" })]],
        [null, [SYS(), cb("a", 1, { role: "assistant" })]],
      ]),
    );
    expect(r.evictions).toHaveLength(1);
    const e = r.evictions[0];
    expect([e.label, e.taggedSeq, e.enteredSeq, e.lastSeenSeq, e.evictedSeq]).toEqual([
      "rag", 1, 1, 3, 4,
    ]);
  });

  it("names the turn the tag was actually applied", () => {
    // The turn the block ENTERED and the turn it was TAGGED are two different
    // facts: a block present from turn 1 but tagged from turn 2 was not tagged
    // at turn 1, and saying so describes a tag that never existed.
    const r = report(
      write("tag-later.ctrace", [
        [null, [SYS(), cb("Enterprise pricing FAQ", 1), cb("ask", 2)]],
        [null, [SYS(), cb("Enterprise pricing FAQ", 1, { tagged: true }), cb("ask", 2)]],
        [null, [SYS(), cb("ask", 1)]],
      ]),
    );
    expect(r.evictions[0].enteredSeq).toBe(1); // the content was there from the start
    expect(r.evictions[0].taggedSeq).toBe(2); // ...but nobody vouched for it until 2
  });

  it("quotes the label from the first turn that tagged the content", () => {
    // Re-tagging the same text under a second name must not make the report
    // quote the later name against the earlier turn.
    const r = report(
      write("retag.ctrace", [
        [null, [SYS(), cb("Enterprise pricing FAQ", 1, { tagged: true, label: "rag" }), cb("ask", 2)]],
        [null, [SYS(), cb("Enterprise pricing FAQ", 1, { tagged: true, label: "notes" }), cb("ask", 2)]],
        [null, [SYS(), cb("ask", 1)]],
      ]),
    );
    expect([r.evictions[0].label, r.evictions[0].taggedSeq]).toEqual(["rag", 1]);
  });

  it("counts one lost block however many memberships it had", () => {
    // A call may legitimately carry the same text twice. The differ reports two
    // `evicted` entries for it — two memberships really did disappear — but the
    // developer lost ONE block, and counting the memberships is what made
    // `check` say "2 tagged blocks evicted of 1" and `tokens` print the same
    // stanza twice.
    const r = report(
      write("twice.ctrace", [
        [
          null,
          [
            SYS(),
            cb("Enterprise pricing FAQ", 1, { tagged: true, tokens: 42 }),
            cb("Enterprise pricing FAQ", 2, { tagged: true, tokens: 42 }),
            cb("ask", 3),
          ],
        ],
        [null, [SYS(), cb("ask", 1)]],
      ]),
    );
    expect(r.evictions).toHaveLength(1);
    expect(r.taggedBlocks).toBe(1);
  });
});

describe("the three narrowings", () => {
  it("never reports a heuristically labeled block", () => {
    // Every multi-turn agent evicts ordinary history — that is what a context
    // window IS. Reporting it would bury the one line worth reading.
    const r = report(
      write("heuristic.ctrace", [
        [null, [SYS(), cb("ancient history", 1, { role: "assistant" })]],
        [null, [SYS()]],
      ]),
    );
    expect(r.evictions).toEqual([]);
    expect(r.taggedBlocks).toBe(0);
  });

  it("does not mistake an agent hand-off for an eviction", () => {
    const rSys = cb("researcher", 0, { role: "system", label: "system" });
    const wSys = cb("writer", 0, { role: "system", label: "system" });
    const rag = cb("Retrieved passage", 1, { tagged: true });
    const r = report(
      write("handoff.ctrace", [
        ["researcher", [rSys, rag]],
        ["writer", [wSys, cb("compose", 1)]],
        ["researcher", [rSys, rag, cb("more", 2)]],
      ]),
    );
    expect(r.evictions).toEqual([]);
    expect(r.agentsAnalyzed).toBe(2);
  });

  it("still finds an eviction on one agent's own timeline", () => {
    // The per-agent scoping must not become a way to miss a real loss.
    const rSys = cb("researcher", 0, { role: "system", label: "system" });
    const wSys = cb("writer", 0, { role: "system", label: "system" });
    const rag = cb("Retrieved passage", 1, { tagged: true });
    const r = report(
      write("handoff-real.ctrace", [
        ["researcher", [rSys, rag]],
        ["writer", [wSys, cb("compose", 1)]],
        ["researcher", [rSys, cb("later", 1, { role: "assistant" })]],
      ]),
    );
    expect(r.evictions).toHaveLength(1);
    const e = r.evictions[0];
    expect([e.agent, e.enteredSeq, e.evictedSeq]).toEqual(["researcher", 1, 3]);
  });

  it("stays silent about a block that comes back", () => {
    // Absent for one turn and back the next is a re-rank, not a loss.
    const rag = cb("Enterprise pricing FAQ", 1, { tagged: true });
    const r = report(
      write("returns.ctrace", [
        [null, [SYS(), rag]],
        [null, [SYS(), cb("filler", 1)]],
        [null, [SYS(), rag, cb("filler", 2)]],
      ]),
    );
    expect(r.evictions).toEqual([]);
  });

  it("reports a block that leaves twice once, for the departure it did not return from", () => {
    const rag = cb("Enterprise pricing FAQ", 1, { tagged: true });
    const r = report(
      write("twice.ctrace", [
        [null, [SYS(), rag]],
        [null, [SYS(), cb("filler", 1)]],
        [null, [SYS(), rag, cb("filler", 2)]],
        [null, [SYS(), cb("filler", 1), cb("more", 2)]],
      ]),
    );
    expect(r.evictions).toHaveLength(1);
    const e = r.evictions[0];
    expect([e.enteredSeq, e.lastSeenSeq, e.evictedSeq]).toEqual([1, 3, 4]);
  });
});

describe("the differ-reuse consequence", () => {
  it("treats an edited tagged block as modified, not evicted", () => {
    // The reason this module calls `diffCalls` instead of comparing hash sets.
    // It is also the detector's one documented BLIND SPOT: a tagged block
    // replaced in the same slot by different text of the same role and kind
    // reads to the differ as an edit. Whatever the differ calls it, this report
    // calls it — a second opinion here would make `diff` and `tokens` disagree.
    const r = report(
      write("edited.ctrace", [
        [null, [SYS(), cb("Brief revision A", 1, { tagged: true, label: "brief" })]],
        [null, [SYS(), cb("Brief revision B", 1, { tagged: true, label: "brief" })]],
      ]),
    );
    expect(r.evictions).toEqual([]);
    expect(r.taggedBlocks).toBe(2); // two distinct texts were tagged
  });
});

describe("reporting", () => {
  it("distinguishes 'nothing was compared' from 'nothing was lost'", () => {
    const r = report(
      write("onepair.ctrace", [[null, [cb("Enterprise pricing FAQ", 0, { tagged: true })]]]),
    );
    expect(r.pairsAnalyzed).toBe(0);
    expect(r.taggedBlocks).toBe(1);
  });

  it("prints the eviction in the words the bug is described in", async () => {
    const path = evictingTrace("cli.ctrace");
    const { out } = await run(["tokens", "--project", path]);
    expect(out).toContain("⚠ the block you tagged 'rag' at turn 1 was evicted at turn 3");
    expect(out).toContain("'Enterprise pricing FAQ'");
    expect(out).toContain(
      "[rag·user] 42 tok · entered at turn 1 · last present at turn 2 · never returned",
    );
  });

  it("reports only the selected turn's eviction under --turn", async () => {
    // `--turn 1` selects ONE turn, and the eviction stanza has to obey the same
    // selector: printing "…was evicted at turn 3" under a turn-1 report
    // describes something the reader did not ask about and cannot see.
    const path = evictingTrace("cli-turn.ctrace");
    expect((await run(["tokens", "--project", path, "--turn", "1"])).out).not.toContain(
      "evicted",
    );
    expect((await run(["tokens", "--project", path, "--turn", "3"])).out).toContain(
      "was evicted at turn 3",
    );
  });

  it("says nothing when there is nothing to say", async () => {
    // No reassuring line: a section that prints on every clean run trains people
    // to stop reading it.
    const path = write("clean.ctrace", [
      [null, [SYS()]],
      [null, [SYS(), cb("hi", 1)]],
    ]);
    const { out } = await run(["tokens", "--project", path]);
    expect(out).not.toContain("evicted");
  });
});

describe("the dashboard", () => {
  it("renders the eviction stanza in the exported page", async () => {
    // The payload says what the dashboard COULD show; only executing the page
    // says what it does. The panel carries the same three lines the CLI prints,
    // so a screenshot of one and a paste of the other read as the same tool.
    const path = evictingTrace("page.ctrace");
    const out = join(dir, "page.html");
    expect((await run(["export", "--project", path, "--out", out])).code).toBe(0);
    const page = bootPage(readFileSync(out, "utf-8"));
    const alloc = page.byId.get("alloc")!.text();
    expect(alloc).toContain("⚠ the block you tagged 'rag' at turn 1 was evicted at turn 3");
    expect(alloc).toContain("Enterprise pricing FAQ");
    expect(alloc).toContain(
      "42 tok · entered at turn 1 · last present at turn 2 · never returned",
    );
  });

  it("renders no eviction stanza when nothing was lost", async () => {
    const path = write("page-clean.ctrace", [
      [null, [SYS()]],
      [null, [SYS(), cb("hi", 1)]],
    ]);
    const out = join(dir, "page-clean.html");
    expect((await run(["export", "--project", path, "--out", out])).code).toBe(0);
    const page = bootPage(readFileSync(out, "utf-8"));
    expect(page.byId.get("alloc")!.text()).not.toContain("was evicted at turn");
  });
});

describe("the check assertion", () => {
  it("fails the build on a tagged eviction", async () => {
    const path = evictingTrace("check-fail.ctrace");
    const { code, out } = await run(["check", "--project", path, "--no-tagged-eviction"]);
    expect(code).toBe(1);
    expect(out).toContain("FAIL  no-tagged-eviction");
    expect(out).toContain("the block you tagged 'rag' at turn 1 was evicted at turn 3 · 42 tok");
  });

  it("never reports more blocks lost than it counted", async () => {
    // The summary's numerator and its denominator have to be the same KIND of
    // thing. Counting eviction EVENTS against distinct tagged BLOCKS produced
    // the self-contradiction "2 tagged blocks evicted of 1" the moment one call
    // carried the same tagged text twice.
    const path = write("check-twice.ctrace", [
      [
        null,
        [
          SYS(),
          cb("Enterprise pricing FAQ", 1, { tagged: true, tokens: 42 }),
          cb("Enterprise pricing FAQ", 2, { tagged: true, tokens: 42 }),
          cb("ask", 3),
        ],
      ],
      [null, [SYS(), cb("ask", 1)]],
    ]);
    const { code, out } = await run(["check", "--project", path, "--no-tagged-eviction"]);
    expect(code).toBe(1);
    expect(out).toContain("1 tagged block evicted of 1 across 1 turn pair");
    expect(out.split("was evicted at turn 2").length - 1).toBe(1);
  });

  it("distinguishes an untagged run from a clean one", async () => {
    // A tick that measured nothing is the failure mode `check` exists to avoid.
    const path = write("untagged.ctrace", [
      [null, [SYS()]],
      [null, [SYS(), cb("hi", 1)]],
    ]);
    const { code, out } = await run(["check", "--project", path, "--no-tagged-eviction"]);
    expect(code).toBe(0);
    expect(out).toContain("no tagged blocks in this run");
  });

  it("reports a genuine all-clear with both denominators", async () => {
    const rag = cb("Enterprise pricing FAQ", 1, { tagged: true });
    const path = write("allclear.ctrace", [
      [null, [SYS(), rag]],
      [null, [SYS(), rag, cb("hi", 2)]],
    ]);
    const { code, out } = await run(["check", "--project", path, "--no-tagged-eviction"]);
    expect(code).toBe(0);
    expect(out).toContain("all 1 tagged block survived 1 turn pair");
  });

  it("offers the new assertion when nothing was asked for", async () => {
    const path = evictingTrace("menu.ctrace");
    const { code, err } = await run(["check", "--project", path]);
    expect(code).toBe(2);
    expect(err).toContain("--no-tagged-eviction");
  });
});
