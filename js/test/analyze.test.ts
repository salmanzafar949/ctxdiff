import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { CTrace } from "../src/store/ctrace.js";
import { diffTurns } from "../src/analyze/diff.js";
import { analyzeRun, detectBloat, extractToolName } from "../src/analyze/tokens.js";
import { analyzeCache } from "../src/analyze/cache.js";
import { listFileSessions } from "../src/analyze/sessions.js";
import { makeFixtures } from "./helpers/fixtures.js";

let dir: string;
let fx: { multiturn: string; multiagent: string; dynamic: string };

beforeAll(() => {
  dir = mkdtempSync(join(tmpdir(), "ctxdiff-analyze-"));
  fx = makeFixtures(dir);
});
afterAll(() => rmSync(dir, { recursive: true, force: true }));

describe("diff analyzer", () => {
  it("classifies a system + user modification between turns 2 and 3", () => {
    const ct = CTrace.open(fx.multiturn);
    try {
      const d = diffTurns(ct, 2, 3);
      const kinds = d.entries.map((e) => e.kind);
      // two 'modified' (system prompt, last user), rest unchanged; no add/evict.
      expect(kinds.filter((k) => k === "modified").length).toBe(2);
      expect(kinds.filter((k) => k === "added").length).toBe(0);
      expect(kinds.filter((k) => k === "evicted").length).toBe(0);
      const modified = d.entries.filter((e) => e.kind === "modified");
      expect(modified[0].inlineDiff).not.toBeNull();
      // modified counts as evict-old + add-new for budgeting → both deltas > 0
      expect(d.tokensAdded).toBeGreaterThan(0);
      expect(d.tokensEvicted).toBeGreaterThan(0);
    } finally {
      ct.close();
    }
  });

  it("sees pure history growth from turn 1 to 2 as added-only (stable prefix)", () => {
    const ct = CTrace.open(fx.multiturn);
    try {
      const d = diffTurns(ct, 1, 2);
      expect(d.entries.some((e) => e.kind === "added")).toBe(true);
      expect(d.entries.some((e) => e.kind === "evicted")).toBe(false);
      expect(d.entries.some((e) => e.kind === "modified")).toBe(false);
    } finally {
      ct.close();
    }
  });

  it("throws a clear error for a missing turn", () => {
    const ct = CTrace.open(fx.multiturn);
    try {
      expect(() => diffTurns(ct, 1, 99)).toThrow(/turn\(s\) \[99\] not found/);
    } finally {
      ct.close();
    }
  });
});

describe("tokens analyzer", () => {
  it("attributes labels, reconciles usage, and flags unused tool schemas", () => {
    const ct = CTrace.open(fx.multiturn);
    try {
      const rt = analyzeRun(ct);
      expect(rt.calls).toHaveLength(3);
      // call 1: tool_schema is the biggest slice (two unused schemas)
      const c1 = rt.calls[0];
      expect(c1.slices[0].label).toBe("tool_schema");
      // slice pcts sum to ~100
      const sum = c1.slices.reduce((a, s) => a + s.pct, 0);
      expect(Math.abs(sum - 100)).toBeLessThan(0.5);
      // reconciliation: provider 40 prompt tokens vs our block total
      expect(c1.reconciliationDelta).toBe(40 - c1.totalTokens);
      // bloat: both tools unused
      expect(rt.bloat).not.toBeNull();
      expect(rt.bloat!.unusedTools.sort()).toEqual(["calculator", "web_search"]);
      // usage rollup
      expect(rt.usage.inputTokens).toBe(40 + 70 + 95);
      expect(rt.usage.callsWithUsage).toBe(3);
    } finally {
      ct.close();
    }
  });

  it("produces a per-agent breakdown for a multi-agent run", () => {
    const ct = CTrace.open(fx.multiagent);
    try {
      const rt = analyzeRun(ct);
      expect(rt.byAgent).not.toBeNull();
      expect([...rt.byAgent!.keys()]).toEqual(["researcher", "writer"]);
      expect(rt.usage.byAgent).not.toBeNull();
    } finally {
      ct.close();
    }
  });

  it("extractToolName handles OpenAI-nested and bare shapes; detectBloat returns null with no schemas", () => {
    expect(extractToolName(JSON.stringify({ type: "function", function: { name: "f" } }))).toBe("f");
    expect(extractToolName(JSON.stringify({ name: "bare" }))).toBe("bare");
    expect(extractToolName("not json")).toBe("<unparsed>");
    expect(detectBloat([[]])).toBeNull();
  });
});

describe("cache analyzer", () => {
  it("attributes a prefix break to the modified system block (multiturn)", () => {
    const ct = CTrace.open(fx.multiturn);
    try {
      const r = analyzeCache(ct);
      expect(r.pairsAnalyzed).toBe(2);
      expect(r.breaks).toHaveLength(1);
      expect(r.breaks[0].culpritKind).toBe("modified");
      expect(r.breaks[0].culpritLabel).toBe("system");
      expect(r.breaks[0].divergentPosition).toBe(0);
    } finally {
      ct.close();
    }
  });

  it("never counts a cross-agent hand-off as a break (multiagent)", () => {
    const ct = CTrace.open(fx.multiagent);
    try {
      const r = analyzeCache(ct);
      // 2 agents × 1 within-agent pair each = 2 pairs, NO breaks
      expect(r.agentsAnalyzed).toBe(2);
      expect(r.pairsAnalyzed).toBe(2);
      expect(r.breaks).toHaveLength(0);
      expect(r.pairsByAgent).not.toBeNull();
    } finally {
      ct.close();
    }
  });

  it("emits the dynamic-field fix hint when an early system block breaks every turn", () => {
    const ct = CTrace.open(fx.dynamic);
    try {
      const r = analyzeCache(ct);
      expect(r.breaks.length).toBe(2);
      expect(r.breaks.every((b) => b.culpritKind === "modified")).toBe(true);
      expect(r.fixHint).toMatch(/dynamic value inside an early system block/);
    } finally {
      ct.close();
    }
  });

  it("returns an explanatory note for a single-call run (no pairs)", () => {
    const p = join(dir, "single.ctrace");
    const ct = CTrace.create(p, "one", "openai", "", "2026-07-04T00:00:00Z");
    ct.recordCall({ seq: 1, params: { model: "gpt-4o" }, usage: null, latencyMs: 1, error: null, callBlocks: [] });
    ct.close();
    const r = analyzeCache(CTrace.open(p));
    expect(r.pairsAnalyzed).toBe(0);
    expect(r.estimatedWasteNote).toMatch(/fewer than 2 calls/);
  });
});

describe("session scanner", () => {
  it("lists every .ctrace in a directory with its sessions, sorted by filename", () => {
    const files = listFileSessions(dir);
    const names = files.map((f) => f.filename);
    expect(names).toContain("multiturn.ctrace");
    expect(names).toContain("multiagent.ctrace");
    expect(names).toEqual([...names].sort());
    const ma = files.find((f) => f.filename === "multiagent.ctrace")!;
    expect(ma.sessions).toHaveLength(1);
    expect(ma.sessions[0].project).toBe("pipeline");
    expect(ma.sessions[0].provider).toBe("openai");
    expect(ma.sessions[0].turnCount).toBe(4);
    expect(ma.sessions[0].agents).toEqual(["researcher", "writer"]);
    const mt = files.find((f) => f.filename === "multiturn.ctrace")!;
    expect(mt.sessions[0].agents).toEqual([]); // no named agents
  });
});
