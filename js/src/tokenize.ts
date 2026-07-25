/**
 * Token counting. Exact for OpenAI via a pure-JS `o200k_base` tokenizer
 * (gpt-tokenizer — verified to match Python's tiktoken byte-for-byte, e.g.
 * "Hello, world!" -> [13225, 11, 2375, 0] in both); a documented heuristic
 * estimate for every other provider. The returned `method` string lets every
 * view mark estimates as approximate. Mirrors Python `count_tokens`.
 */
import { encode as encodeO200k } from "gpt-tokenizer/encoding/o200k_base";

/**
 * An empty disallowed-special set, built once and reused.
 *
 * By default gpt-tokenizer (like tiktoken) THROWS on any text that literally
 * spells a special token — `<|endoftext|>`, `<|fim_prefix|>`, … — so a caller
 * cannot smuggle a control token into a prompt by accident. ctxdiff is not
 * building a prompt; it is measuring one that was already sent, and the OpenAI
 * API escapes those spellings rather than honouring them, so to the model they
 * were ordinary text too. Disabling the guard makes the literal encode as the
 * plain characters it is, which is both the truthful count and byte-for-byte
 * what Python's `encode(text, disallowed_special=())` returns — nine tokens
 * for `a <|endoftext|> b` in both SDKs.
 */
const NOTHING_DISALLOWED: Set<string> = new Set();

/**
 * Exact OpenAI token count over the `o200k_base` encoding (GPT-4o family). The
 * gpt-tokenizer encoder is loaded once at module init and reused; there is no
 * network access (the merges table ships inside the package), so unlike the
 * Python path there is no download-on-first-use failure mode to guard.
 */
function tiktokenCount(text: string): number {
  return encodeO200k(text, { disallowedSpecial: NOTHING_DISALLOWED }).length;
}

/**
 * Estimate tokens when no exact tokenizer exists. Uses the well-known
 * ~4-characters-per-token rule of thumb, rounded up so any non-empty text is at
 * least 1 token. Empty text is zero. Mirrors Python `_estimate_count`.
 *
 * A CHARACTER HERE MEANS A UNICODE CODE POINT, which is why this spreads the
 * string rather than reading `.length`. JS strings are UTF-16, so `.length`
 * counts CODE UNITS and every astral-plane character — emoji, flag sequences,
 * skin-tone modifiers, math alphanumerics, CJK ext-B — counts twice; Python's
 * `len()` counts code points and counts it once. Since every provider EXCEPT
 * openai is counted by this function, `.length` made bedrock, anthropic and
 * gemini traces render DIFFERENT token numbers in the two SDKs for byte-identical
 * content (a system block of `Répondez en français 🇫🇷` was 7 tokens here and 6
 * in Python) — exactly the cross-SDK divergence the pinned tokenizers and the
 * golden corpus exist to prevent. Hashes were never affected (counts are not
 * hashed), but `ctxdiff tokens`, the cache profiler's re-billed totals and the
 * dashboard's percentages all were. Iterating the string yields code points, so
 * the two SDKs now agree by construction. Grapheme CLUSTERS are deliberately not
 * the unit: Python does not use them either, and the rule of thumb is about
 * character volume, not user-perceived glyphs.
 */
function estimateCount(text: string): number {
  if (!text) return 0;
  return Math.max(1, Math.ceil([...text].length / 4));
}

/**
 * Count tokens for `text` under `provider`, returning [count, method].
 * OpenAI -> exact ('tiktoken'); anything else -> heuristic ('estimate'). Empty
 * text is always zero tokens (but keeps the provider's method label, matching
 * the Python SDK: an empty string for openai is (0, 'tiktoken')). If the exact
 * tokenizer ever throws, fall back to the estimate and mark it 'estimate' — a
 * debugging tool must never crash the host over a token count.
 *
 * The fallback is scoped to the offending text and nothing else: the encoder
 * stays live, so the very next block still gets an exact count. Python matches
 * this exactly — its `_ENCODER_UNAVAILABLE` latch now fires only when the
 * encoder cannot be CONSTRUCTED (a failure mode this SDK does not have, since
 * the merges table is bundled), never for a single text that would not encode.
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
