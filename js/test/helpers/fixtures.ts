/**
 * Shared `.ctrace` fixture builders for the analyzer parity tests. Each fixture
 * is written straight through the CTrace store (deterministic, no network) and
 * exercises a specific analyzer path. Blocks are hashed/counted/labeled exactly
 * like the recorder, so fixtures look like real captures — and the SAME files
 * feed the cross-language comparison against the Python CLI.
 */
import { CTrace } from "../../src/store/ctrace.js";
import { contentHash, basicLabel, type CallBlock } from "../../src/models.js";
import { countTokens } from "../../src/tokenize.js";
import { join } from "node:path";

type Row = [string, string, string]; // [role, kind, text]

/** Build a CallBlock list from [role, kind, text] tuples, mirroring the recorder. */
function blocks(rows: Row[], provider = "openai"): CallBlock[] {
  return rows.map(([role, kind, text], position) => {
    const [tokenCount, tokenMethod] = countTokens(text, provider);
    const [label, labelSource] = basicLabel(role, kind, text, []);
    return {
      block: { contentHash: contentHash(role, kind, text), role, kind, text, tokenCount, tokenMethod },
      position,
      label,
      labelSource,
    };
  });
}

/** A multi-turn trace: growing history (cache-friendly), a turn that modifies
 * the system block + a user block (diff), unused tool schemas (bloat), usage. */
export function writeMultiturn(dir: string): string {
  const path = join(dir, "multiturn.ctrace");
  const ct = CTrace.create(path, "research", "openai", "", "2026-07-01T00:00:00Z");
  const sys: Row = ["system", "message", "You are a helpful research assistant. Be concise and cite sources."];
  const toolA: Row = ["system", "tool_schema", JSON.stringify({ type: "function", function: { name: "web_search" } })];
  const toolB: Row = ["system", "tool_schema", JSON.stringify({ type: "function", function: { name: "calculator" } })];

  ct.recordCall({
    seq: 1, params: { model: "gpt-4o", temperature: 0.2 },
    usage: { prompt_tokens: 40, completion_tokens: 12, total_tokens: 52 },
    latencyMs: 120, error: null,
    callBlocks: blocks([sys, toolA, toolB, ["user", "message", "What is the capital of France?"]]),
    provider: "openai",
  });
  ct.recordCall({
    seq: 2, params: { model: "gpt-4o", temperature: 0.2 },
    usage: { prompt_tokens: 70, completion_tokens: 20, total_tokens: 90 },
    latencyMs: 140, error: null,
    callBlocks: blocks([
      sys, toolA, toolB,
      ["user", "message", "What is the capital of France?"],
      ["assistant", "message", "The capital of France is Paris."],
      ["user", "message", "And its population?"],
    ]),
    provider: "openai",
  });
  ct.recordCall({
    seq: 3, params: { model: "gpt-4o", temperature: 0.2 },
    usage: { prompt_tokens: 95, completion_tokens: 25, total_tokens: 120 },
    latencyMs: 150, error: null,
    callBlocks: blocks([
      ["system", "message", "You are a helpful research assistant. Be concise and cite sources. Always answer in English."],
      toolA, toolB,
      ["user", "message", "What is the capital of France?"],
      ["assistant", "message", "The capital of France is Paris."],
      ["user", "message", "And its population, roughly?"],
    ]),
    provider: "openai",
  });
  ct.close();
  return path;
}

/** A two-agent trace: researcher + writer interleaved, so cache grouping must
 * not count the cross-agent hand-off as a break. */
export function writeMultiagent(dir: string): string {
  const path = join(dir, "multiagent.ctrace");
  const ct = CTrace.create(path, "pipeline", "openai", "", "2026-07-02T00:00:00Z");
  const rSys: Row = ["system", "message", "You are the RESEARCHER. Gather facts."];
  const wSys: Row = ["system", "message", "You are the WRITER. Compose prose."];
  ct.recordCall({ seq: 1, params: { model: "gpt-4o" }, usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }, latencyMs: 100, error: null, callBlocks: blocks([rSys, ["user", "message", "Find facts about Mars."]]), agent: "researcher", step: "gather", provider: "openai" });
  ct.recordCall({ seq: 2, params: { model: "gpt-4o" }, usage: { prompt_tokens: 12, completion_tokens: 6, total_tokens: 18 }, latencyMs: 110, error: null, callBlocks: blocks([wSys, ["user", "message", "Write an intro about Mars."]]), agent: "writer", step: "compose", provider: "openai" });
  ct.recordCall({ seq: 3, params: { model: "gpt-4o" }, usage: { prompt_tokens: 20, completion_tokens: 8, total_tokens: 28 }, latencyMs: 120, error: null, callBlocks: blocks([rSys, ["user", "message", "Find facts about Mars."], ["assistant", "message", "Mars is the fourth planet."], ["user", "message", "More detail?"]]), agent: "researcher", step: "gather", provider: "openai" });
  ct.recordCall({ seq: 4, params: { model: "gpt-4o" }, usage: { prompt_tokens: 22, completion_tokens: 9, total_tokens: 31 }, latencyMs: 130, error: null, callBlocks: blocks([wSys, ["user", "message", "Write an intro about Mars."], ["assistant", "message", "Mars, the red planet, ..."], ["user", "message", "Expand it."]]), agent: "writer", step: "compose", provider: "openai" });
  ct.close();
  return path;
}

/** A dynamic-timestamp trace: an early system block carries a changing timestamp,
 * breaking the prefix identically every turn (triggers the fix hint). */
export function writeDynamic(dir: string): string {
  const path = join(dir, "dynamic.ctrace");
  const ct = CTrace.create(path, "dynamic", "openai", "", "2026-07-03T00:00:00Z");
  for (let t = 1; t <= 3; t++) {
    ct.recordCall({
      seq: t, params: { model: "gpt-4o" },
      usage: { prompt_tokens: 30 + t, completion_tokens: 5, total_tokens: 35 + t },
      latencyMs: 100, error: null,
      callBlocks: blocks([
        ["system", "message", `Current time is 2026-07-03T10:0${t}:00Z. You are an assistant.`],
        ["user", "message", `Question number ${t}?`],
      ]),
      provider: "openai",
    });
  }
  ct.close();
  return path;
}

/** A trace whose second turn ADDS a user block carrying zero-width and bidi
 * format characters (ZWSP, LRM/RLM), so the diff renders that block's snippet
 * through `pyRepr` — exercising non-printable escaping against Python. Built
 * from code points (never literal control chars). */
export function writeBidi(dir: string): string {
  const cp = String.fromCodePoint;
  const bidiText = `Reply${cp(0x200b)}now ${cp(0x200e)}left${cp(0x200f)}right ok`;
  const path = join(dir, "bidi.ctrace");
  const ct = CTrace.create(path, "bidi", "openai", "", "2026-07-05T00:00:00Z");
  const sys: Row = ["system", "message", "You are an assistant."];
  const user1: Row = ["user", "message", "First question?"];
  ct.recordCall({ seq: 1, params: { model: "gpt-4o" }, usage: { prompt_tokens: 8, completion_tokens: 2, total_tokens: 10 }, latencyMs: 100, error: null, callBlocks: blocks([sys, user1]), provider: "openai" });
  ct.recordCall({ seq: 2, params: { model: "gpt-4o" }, usage: { prompt_tokens: 14, completion_tokens: 3, total_tokens: 17 }, latencyMs: 110, error: null, callBlocks: blocks([sys, user1, ["assistant", "message", "First answer."], ["user", "message", bidiText]]), provider: "openai" });
  ct.close();
  return path;
}

export function makeFixtures(dir: string): {
  multiturn: string;
  multiagent: string;
  dynamic: string;
  bidi: string;
} {
  return {
    multiturn: writeMultiturn(dir),
    multiagent: writeMultiagent(dir),
    dynamic: writeDynamic(dir),
    bidi: writeBidi(dir),
  };
}
