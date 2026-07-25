import { describe, it, expect } from "vitest";
import { countTokens } from "../src/tokenize.js";

describe("countTokens", () => {
  it("counts OpenAI exactly via o200k_base and marks 'tiktoken'", () => {
    const [count, method] = countTokens("Hello, world!", "openai");
    // Matches Python tiktoken o200k_base: [13225, 11, 2375, 0] -> 4 tokens.
    expect(count).toBe(4);
    expect(method).toBe("tiktoken");
  });

  it("returns a positive exact count for longer OpenAI text", () => {
    const [count, method] = countTokens(
      "The quick brown fox jumps over the lazy dog.",
      "openai",
    );
    expect(count).toBeGreaterThan(0);
    expect(method).toBe("tiktoken");
  });

  it("estimates for non-openai providers and marks 'estimate'", () => {
    const [count, method] = countTokens("a".repeat(400), "anthropic");
    expect(count).toBe(100); // ceil(400/4)
    expect(method).toBe("estimate");
  });

  it("estimate is at least 1 token for any non-empty text", () => {
    const [count, method] = countTokens("x", "anthropic");
    expect(count).toBe(1);
    expect(method).toBe("estimate");
  });

  it("empty string is 0 tokens, keeping the provider's method label", () => {
    expect(countTokens("", "openai")).toEqual([0, "tiktoken"]);
    expect(countTokens("", "anthropic")).toEqual([0, "estimate"]);
  });

  it("a literal special-token spelling encodes as text rather than throwing", () => {
    // gpt-tokenizer refuses by default to encode text that spells a control
    // token. ctxdiff is measuring a payload that was already sent, and the
    // OpenAI API escapes those spellings, so the model saw plain characters
    // too — counting them as plain characters is the truthful number, and it
    // is the one Python's `disallowed_special=()` produces: 9, not the
    // character estimate's 5.
    expect(countTokens("a <|endoftext|> b", "openai")).toEqual([9, "tiktoken"]);
  });

  it("a special-token block does not degrade the blocks around it", () => {
    // The JS mirror of the Python regression test. gpt-tokenizer's guard is
    // WIDER than tiktoken's (it rejects the whole `<|...|>` family, while
    // o200k_base only reserves `<|endoftext|>` and `<|endofprompt|>`), so
    // before the fix `<|im_start|>` was an exact count in Python and an
    // estimate here — a cross-SDK disagreement on ordinary text. Both are now
    // plain text in both SDKs, and neighbours are untouched either way.
    expect(countTokens("hello world", "openai")).toEqual([2, "tiktoken"]);
    expect(countTokens("<|im_start|>", "openai")).toEqual([6, "tiktoken"]);
    expect(countTokens("<|endofprompt|>", "openai")).toEqual([7, "tiktoken"]);
    expect(countTokens("hello world", "openai")).toEqual([2, "tiktoken"]);
  });
});
