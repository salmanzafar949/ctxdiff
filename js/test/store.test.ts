import { describe, it, expect, afterEach } from "vitest";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { rmSync } from "node:fs";
import { CTrace } from "../src/store/ctrace.js";
import { SCHEMA_VERSION } from "../src/store/schema.js";
import { contentHash, type Block, type CallBlock } from "../src/models.js";

const created: string[] = [];
function tmpTrace(): string {
  const p = join(tmpdir(), `ctxdiff-${randomUUID()}.ctrace`);
  created.push(p);
  return p;
}
afterEach(() => {
  for (const p of created.splice(0)) {
    try {
      rmSync(p, { force: true });
    } catch {
      /* ignore */
    }
  }
});

function mkBlock(role: string, kind: string, text: string): Block {
  return {
    contentHash: contentHash(role, kind, text),
    role,
    kind,
    text,
    tokenCount: 1,
    tokenMethod: "tiktoken",
  };
}
function mkCallBlock(b: Block, position: number): CallBlock {
  return { block: b, position, label: b.role, labelSource: "heuristic" };
}

describe("CTrace store", () => {
  it("create writes a run row with the v2 schema version", () => {
    const path = tmpTrace();
    const ct = CTrace.create(path, "proj", "openai", "", "2026-07-24T00:00:00Z");
    const run = ct.getRun();
    expect(run.project).toBe("proj");
    expect(run.provider).toBe("openai");
    expect(run.models).toEqual([]);
    expect(run.startedAt).toBe("2026-07-24T00:00:00Z");
    ct.close();
    // reopen validates schema_version == 2 accepted
    const ct2 = CTrace.open(path);
    expect(ct2.getRun().project).toBe("proj");
    ct2.close();
    expect(SCHEMA_VERSION).toBe(2);
  });

  it("roundtrips a call with ordered blocks + usage + attribution", () => {
    const path = tmpTrace();
    const ct = CTrace.create(path, "proj", "openai", "");
    const blocks = [
      mkCallBlock(mkBlock("system", "message", "you are helpful"), 0),
      mkCallBlock(mkBlock("user", "message", "hi"), 1),
    ];
    const callId = ct.recordCall({
      seq: 1,
      params: { model: "gpt-4o", temperature: 0.5 },
      usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
      latencyMs: 42,
      error: null,
      callBlocks: blocks,
      agent: "planner",
      step: "answer",
      provider: "openai",
    });
    ct.close();

    const r = CTrace.open(path);
    const calls = r.getCalls();
    expect(calls).toHaveLength(1);
    const c = calls[0];
    expect(c.id).toBe(callId);
    expect(c.seq).toBe(1);
    expect(c.params).toEqual({ model: "gpt-4o", temperature: 0.5 });
    expect(c.usage).toEqual({
      prompt_tokens: 5,
      completion_tokens: 2,
      total_tokens: 7,
    });
    expect(c.latencyMs).toBe(42);
    expect(c.agent).toBe("planner");
    expect(c.step).toBe("answer");
    expect(c.provider).toBe("openai");

    const cbs = r.getCallBlocks(c.id);
    expect(cbs.map((cb) => cb.position)).toEqual([0, 1]);
    expect(cbs.map((cb) => cb.block.text)).toEqual(["you are helpful", "hi"]);

    // model rolled up onto the run
    expect(r.getRun().models).toEqual(["gpt-4o"]);
    r.close();
  });

  it("dedups a block shared across two calls (stored once)", () => {
    const path = tmpTrace();
    const ct = CTrace.create(path, "proj", "openai", "");
    const shared = mkBlock("system", "message", "shared system prompt");
    ct.recordCall({
      seq: 1,
      params: { model: "gpt-4o" },
      usage: null,
      latencyMs: null,
      error: null,
      callBlocks: [mkCallBlock(shared, 0)],
    });
    ct.recordCall({
      seq: 2,
      params: { model: "gpt-4o" },
      usage: null,
      latencyMs: null,
      error: null,
      callBlocks: [mkCallBlock(shared, 0)],
    });
    ct.close();

    const r = CTrace.open(path);
    expect(r.getCalls()).toHaveLength(2);
    // Both calls reference the SAME block hash.
    const cbs1 = r.getCallBlocks(r.getCalls()[0].id);
    const cbs2 = r.getCallBlocks(r.getCalls()[1].id);
    expect(cbs1[0].block.contentHash).toBe(cbs2[0].block.contentHash);
    r.close();

    // Direct count: exactly one row in `block` despite two references.
    const probe = CTrace.open(path);
    // @ts-expect-error reach into the private db purely to assert dedup at the row level
    const n = probe["db"]
      .prepare("SELECT COUNT(*) AS n FROM block")
      .get() as { n: number };
    expect(n.n).toBe(1);
    // @ts-expect-error same, for call_block
    const m = probe["db"]
      .prepare("SELECT COUNT(*) AS n FROM call_block")
      .get() as { n: number };
    expect(m.n).toBe(2);
    probe.close();
  });

  it("rolls up multiple distinct models in first-seen order, deduped", () => {
    const path = tmpTrace();
    const ct = CTrace.create(path, "proj", "openai", "");
    for (const model of ["gpt-4o", "gpt-4o-mini", "gpt-4o"]) {
      ct.recordCall({
        seq: ct.getCalls().length + 1,
        params: { model },
        usage: null,
        latencyMs: null,
        error: null,
        callBlocks: [],
      });
    }
    expect(ct.getRun().models).toEqual(["gpt-4o", "gpt-4o-mini"]);
    ct.close();
  });

  it("open rejects a schema version newer than supported", () => {
    const path = tmpTrace();
    const ct = CTrace.create(path, "proj", "openai", "");
    ct.close();
    const bumped = CTrace.open(path);
    // @ts-expect-error tamper with the stored version to simulate a future file
    bumped["db"].prepare("UPDATE run SET schema_version = 99").run();
    bumped.close();
    expect(() => CTrace.open(path)).toThrow(/newer than supported/);
  });
});
