import { describe, it, expect } from "vitest";
import { AnthropicAdapter } from "../src/capture/anthropic.js";
import { contentHash } from "../src/models.js";

const a = new AnthropicAdapter();

describe("AnthropicAdapter block extraction", () => {
  it("orders system -> tools -> messages", () => {
    const blocks = a.extractBlocks({
      model: "claude-3-5-sonnet",
      max_tokens: 100,
      system: "be terse",
      tools: [{ name: "get_weather", input_schema: { type: "object" } }],
      messages: [{ role: "user", content: "hi" }],
    });
    expect(blocks.map((b) => [b.role, b.kind])).toEqual([
      ["system", "message"],
      ["system", "tool_schema"],
      ["user", "message"],
    ]);
    expect(blocks[0].text).toBe("be terse");
  });

  it("handles a list-form system (one block per text part)", () => {
    const blocks = a.extractBlocks({
      system: [
        { type: "text", text: "part one" },
        { type: "text", text: "part two" },
      ],
      messages: [],
    });
    expect(blocks.map((b) => b.text)).toEqual(["part one", "part two"]);
    expect(blocks.every((b) => b.role === "system" && b.kind === "message")).toBe(true);
  });

  it("splits array message content into content_part blocks", () => {
    const blocks = a.extractBlocks({
      messages: [
        {
          role: "assistant",
          content: [
            { type: "text", text: "hello" },
            { type: "tool_use", id: "t1", name: "f", input: { a: 1 } },
          ],
        },
      ],
    });
    expect(blocks.map((b) => b.kind)).toEqual(["content_part", "content_part"]);
    expect(blocks.every((b) => b.role === "assistant")).toBe(true);
  });

  it("serializes tool schemas with sorted keys (stable JSON)", () => {
    const blocks = a.extractBlocks({
      tools: [{ name: "f", input_schema: { type: "object" } }],
      messages: [],
    });
    expect(blocks[0].text).toBe('{"input_schema": {"type": "object"}, "name": "f"}');
  });
});

describe("AnthropicAdapter extractParams", () => {
  it("drops content keys, keeps model + sampling", () => {
    const p = a.extractParams({
      model: "claude-3-5-sonnet",
      max_tokens: 100,
      temperature: 0.4,
      system: "x",
      messages: [{ role: "user", content: "hi" }],
      tools: [{ name: "f" }],
    });
    expect(p).toEqual({ model: "claude-3-5-sonnet", max_tokens: 100, temperature: 0.4 });
  });
});

describe("AnthropicAdapter extractUsage", () => {
  it("maps input_tokens/output_tokens", () => {
    expect(a.extractUsage({ usage: { input_tokens: 10, output_tokens: 5 } })).toEqual({
      input_tokens: 10,
      output_tokens: 5,
    });
  });
  it("returns null when absent", () => {
    expect(a.extractUsage({})).toBeNull();
    expect(a.extractUsage(null)).toBeNull();
  });
});

describe("AnthropicAdapter accumulateStreamUsage (split across two events)", () => {
  it("takes input_tokens from message_start and output_tokens from message_delta", () => {
    const state: Record<string, unknown> = {};
    a.accumulateStreamUsage(
      { type: "message_start", message: { usage: { input_tokens: 10, output_tokens: 1 } } },
      state,
    );
    expect(state).toEqual({ input_tokens: 10 });
    a.accumulateStreamUsage({ type: "content_block_delta", delta: { text: "hi" } }, state);
    expect(state).toEqual({ input_tokens: 10 });
    a.accumulateStreamUsage({ type: "message_delta", usage: { output_tokens: 5 } }, state);
    expect(state).toEqual({ input_tokens: 10, output_tokens: 5 });
  });

  it("never throws on a malformed event", () => {
    const state: Record<string, unknown> = {};
    expect(() => a.accumulateStreamUsage(null, state)).not.toThrow();
    expect(() => a.accumulateStreamUsage({ type: "message_start" }, state)).not.toThrow();
    expect(state).toEqual({});
  });
});

describe("AnthropicAdapter cross-language hash parity (golden)", () => {
  // These digests are produced identically by the Python AnthropicAdapter for
  // the SAME logical request (verified against ../venv). They lock parity.
  it("matches the Python adapter's block hashes", () => {
    const blocks = a.extractBlocks({
      model: "claude-3-5-sonnet",
      max_tokens: 100,
      system: "be terse",
      tools: [{ name: "get_weather", input_schema: { type: "object", properties: { city: { type: "string" } } } }],
      messages: [
        { role: "user", content: "hi" },
        {
          role: "assistant",
          content: [
            { type: "text", text: "hello" },
            { type: "tool_use", id: "t1", name: "get_weather", input: { city: "NYC" } },
          ],
        },
      ],
    });
    const hashes = blocks.map((b) => contentHash(b.role, b.kind, b.text));
    expect(hashes).toEqual([
      "82beea5d13ca177090c64aa3cf03441047e9eb229ce7d36be563b3fbf85b7c0e",
      "5333077cda23c8a16d4dc790049c4ff3a243d439e211012b94eff7a810235988",
      "4e6c4093072114cd3ec3641653e12f750391cded3515bf460ccd07162c647685",
      "0828d25725f64ce5cea73a52da744a812f241c72b8c8f0c096a343ed42103c45",
      "2b87757d53478306b9ee6bcd80eb9958b9aef65e4c2ac7f56a1a00903e1fb539",
    ]);
  });
});
