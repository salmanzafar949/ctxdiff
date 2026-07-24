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
});
