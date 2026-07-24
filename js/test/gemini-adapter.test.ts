import { describe, it, expect } from "vitest";
import { GeminiAdapter } from "../src/capture/gemini.js";
import { contentHash } from "../src/models.js";

const g = new GeminiAdapter();

describe("GeminiAdapter block extraction", () => {
  it("pulls systemInstruction (string) from config first, then tools, then contents", () => {
    const blocks = g.extractBlocks({
      model: "gemini-2.0-flash",
      contents: "hello",
      config: {
        systemInstruction: "be terse",
        tools: [{ functionDeclarations: [{ name: "calc" }] }],
      },
    });
    expect(blocks.map((b) => [b.role, b.kind])).toEqual([
      ["system", "message"],
      ["system", "tool_schema"],
      ["user", "message"],
    ]);
    expect(blocks[0].text).toBe("be terse");
    expect(blocks[2].text).toBe("hello");
  });

  it("reads systemInstruction from a config object (typed-config parity case)", () => {
    // In JS the typed GenerateContentConfig is an object; property access works
    // uniformly. A class-instance config resolves the same way.
    class FakeConfig {
      systemInstruction = "from typed config";
      temperature = 0.2;
    }
    const blocks = g.extractBlocks({
      contents: "hi",
      config: new FakeConfig() as unknown as Record<string, unknown>,
    });
    expect(blocks[0]).toEqual({ role: "system", kind: "message", text: "from typed config" });
  });

  it("maps role 'model' -> 'assistant' and splits parts into content_part blocks", () => {
    const blocks = g.extractBlocks({
      contents: [
        { role: "user", parts: [{ text: "q" }] },
        { role: "model", parts: [{ text: "a" }] },
      ],
    });
    expect(blocks.map((b) => [b.role, b.kind, b.text])).toEqual([
      ["user", "content_part", "q"],
      ["assistant", "content_part", "a"],
    ]);
  });

  it("treats a bare string in a contents list as a user message", () => {
    const blocks = g.extractBlocks({ contents: ["just text"] });
    expect(blocks).toEqual([{ role: "user", kind: "message", text: "just text" }]);
  });

  it("degrades an explicit empty-string role to 'user' (Python parity)", () => {
    // Python's `role or "user"` treats "" as falsy -> "user"; JS must too, and
    // the resulting block must hash identically to Python's (c695a947…,
    // verified against ../venv).
    const blocks = g.extractBlocks({
      contents: [{ role: "", parts: [{ text: "hi" }] }],
    });
    expect(blocks).toHaveLength(1);
    expect(blocks[0].role).toBe("user");
    expect(contentHash(blocks[0].role, blocks[0].kind, blocks[0].text)).toBe(
      "c695a947c2f1fed28746c62e4a02dcdb262c44d7a75adaafe06cd5d7f07cae08",
    );
  });

  it("coerces a degenerate text:null part to an empty string (documented divergence)", () => {
    // No real SDK emits `text: null`. Python keeps None (which then fails the
    // NOT NULL insert and drops the call fail-open); JS deliberately coerces to
    // "" so the call still records and `block.text` stays a real string. See the
    // text:null note in spec/ctrace-schema.md.
    const blocks = g.extractBlocks({
      contents: [{ role: "user", parts: [{ type: "text", text: null }] }],
    });
    expect(blocks).toHaveLength(1);
    expect(blocks[0].text).toBe("");
    expect(typeof blocks[0].text).toBe("string");
  });

  it("ignores a non-string/non-array systemInstruction (matches Python)", () => {
    const blocks = g.extractBlocks({
      contents: "hi",
      config: { systemInstruction: { parts: [{ text: "typed content" }] } },
    });
    // Python's adapter only handles str/list system instructions; a Content
    // object is dropped. JS mirrors that — only the user message survives.
    expect(blocks).toEqual([{ role: "user", kind: "message", text: "hi" }]);
  });
});

describe("GeminiAdapter extractParams", () => {
  it("drops contents/config but lifts sampling fields off config", () => {
    const p = g.extractParams({
      model: "gemini-2.0-flash",
      contents: "hi",
      config: { systemInstruction: "x", temperature: 0.5, maxOutputTokens: 256, topP: 0.9 },
    });
    expect(p).toEqual({
      model: "gemini-2.0-flash",
      temperature: 0.5,
      maxOutputTokens: 256,
      topP: 0.9,
    });
  });
});

describe("GeminiAdapter extractUsage", () => {
  it("maps usageMetadata camelCase into Python snake_case keys", () => {
    expect(
      g.extractUsage({
        usageMetadata: { promptTokenCount: 12, candidatesTokenCount: 4, totalTokenCount: 16 },
      }),
    ).toEqual({
      prompt_token_count: 12,
      candidates_token_count: 4,
      total_token_count: 16,
    });
  });
  it("returns null when absent", () => {
    expect(g.extractUsage({})).toBeNull();
    expect(g.extractUsage(null)).toBeNull();
  });
});

describe("GeminiAdapter accumulateStreamUsage (cumulative overwrite)", () => {
  it("overwrites state with the latest chunk's cumulative counts", () => {
    const state: Record<string, unknown> = {};
    g.accumulateStreamUsage(
      { usageMetadata: { promptTokenCount: 12, candidatesTokenCount: 1, totalTokenCount: 13 } },
      state,
    );
    g.accumulateStreamUsage(
      { usageMetadata: { promptTokenCount: 12, candidatesTokenCount: 4, totalTokenCount: 16 } },
      state,
    );
    // last chunk wins (cumulative, not summed)
    expect(state).toEqual({
      promptTokenCount: 12,
      candidatesTokenCount: 4,
      totalTokenCount: 16,
    });
  });

  it("leaves state unchanged for a chunk with no usageMetadata; never throws", () => {
    const state: Record<string, unknown> = { promptTokenCount: 5 };
    expect(() => g.accumulateStreamUsage({ candidates: [] }, state)).not.toThrow();
    expect(() => g.accumulateStreamUsage(null, state)).not.toThrow();
    expect(state).toEqual({ promptTokenCount: 5 });
  });
});

describe("GeminiAdapter cross-language hash parity (golden)", () => {
  // Digests produced identically by the Python GeminiAdapter for the SAME
  // logical request (verified against ../venv).
  it("matches the Python adapter's block hashes", () => {
    const blocks = g.extractBlocks({
      model: "gemini-2.0-flash",
      contents: [
        { role: "user", parts: [{ text: "what is 2+2" }] },
        { role: "model", parts: [{ text: "4" }] },
      ],
      config: {
        systemInstruction: "be terse",
        temperature: 0.5,
        tools: [{ functionDeclarations: [{ name: "calc" }] }],
      },
    });
    const hashes = blocks.map((b) => contentHash(b.role, b.kind, b.text));
    expect(hashes).toEqual([
      "82beea5d13ca177090c64aa3cf03441047e9eb229ce7d36be563b3fbf85b7c0e",
      "588ea1ab834a0757a960f3b893473501a24d6ae49f570dc216ad47f076f14c50",
      "ab1e226c6e6c4f9a19edebd4920c64faca26af6dd08ee219b0bcc2fae5781d4a",
      "b3382c4eb794d997f40c0d894a8ad08bfa0d75bcb8eb4d2df44e1711beec0d63",
    ]);
  });
});
