/**
 * Token counting. Exact for OpenAI via a pure-JS `o200k_base` tokenizer
 * (gpt-tokenizer — verified to match Python's tiktoken byte-for-byte, e.g.
 * "Hello, world!" -> [13225, 11, 2375, 0] in both); a documented heuristic
 * estimate for every other provider. The returned `method` string lets every
 * view mark estimates as approximate. Mirrors Python `count_tokens`.
 */
import { encode as encodeO200k } from "gpt-tokenizer/encoding/o200k_base";

/**
 * Exact OpenAI token count over the `o200k_base` encoding (GPT-4o family). The
 * gpt-tokenizer encoder is loaded once at module init and reused; there is no
 * network access (the merges table ships inside the package), so unlike the
 * Python path there is no download-on-first-use failure mode to guard.
 */
function tiktokenCount(text: string): number {
  return encodeO200k(text).length;
}

/**
 * Estimate tokens when no exact tokenizer exists. Uses the well-known
 * ~4-characters-per-token rule of thumb, rounded up so any non-empty text is at
 * least 1 token. Empty text is zero. Mirrors Python `_estimate_count`.
 */
function estimateCount(text: string): number {
  if (!text) return 0;
  return Math.max(1, Math.ceil(text.length / 4));
}

/**
 * Count tokens for `text` under `provider`, returning [count, method].
 * OpenAI -> exact ('tiktoken'); anything else -> heuristic ('estimate'). Empty
 * text is always zero tokens (but keeps the provider's method label, matching
 * the Python SDK: an empty string for openai is (0, 'tiktoken')). If the exact
 * tokenizer ever throws, fall back to the estimate and mark it 'estimate' — a
 * debugging tool must never crash the host over a token count.
 */
export function countTokens(text: string, provider: string): [number, string] {
  if (!text) return [0, provider === "openai" ? "tiktoken" : "estimate"];
  if (provider === "openai") {
    try {
      return [tiktokenCount(text), "tiktoken"];
    } catch {
      return [estimateCount(text), "estimate"];
    }
  }
  return [estimateCount(text), "estimate"];
}
