/**
 * Anthropic Messages adapter (@anthropic-ai/sdk). Reads the kwargs passed to
 * `client.messages.create(...)` — plain objects/arrays, no dependency on the
 * SDK itself. Differs from OpenAI in two ways this adapter absorbs: `system` is
 * a top-level field (string or list of text blocks), and usage is reported as
 * input_tokens/output_tokens.
 *
 * Mirrors the Python `ctxdiff.capture.anthropic` adapter byte-for-byte: same
 * block ordering (system → tools → messages), same roles/kinds, same
 * stable-JSON serialization, so the SAME logical request hashes identically
 * across the two SDKs.
 */
import type { Adapter } from "./base.js";
import type { RawBlock } from "../models.js";
import { stableStringify } from "../models.js";

// Request keys that carry block *content* rather than sampling params.
const CONTENT_KEYS = new Set(["system", "messages", "tools"]);

/** Stable JSON identical to Python's `json.dumps(x, sort_keys=True,
 * ensure_ascii=False)` — the exact serialization the Python adapter uses for
 * tool schemas and non-string content parts, so hashes match cross-language. */
function sjson(x: unknown): string {
  return stableStringify(x);
}

function isRecord(x: unknown): x is Record<string, unknown> {
  return typeof x === "object" && x !== null && !Array.isArray(x);
}

export class AnthropicAdapter implements Adapter {
  readonly provider = "anthropic";
  // `messages.stream(...)` (the `client.messages.stream(...)` convenience
  // helper, used as an async-iterable MessageStream) is a SECOND completion
  // method sharing the EXACT same request/response shape as `messages.create`.
  // Confirmed empirically (Phase JS2 probe against @anthropic-ai/sdk 0.114.0)
  // that the events it yields (message_start/message_delta/...) are the SAME
  // raw event objects a `create({stream:true})` call yields, so
  // `accumulateStreamUsage` below covers both paths with no changes.
  readonly createPaths: string[][] = [
    ["messages", "create"],
    ["messages", "stream"],
  ];

  /**
   * Flatten a request into ordered RawBlocks: the top-level `system` first (one
   * block for a string, or one per entry for a list of text blocks), then tool
   * schemas, then messages. Order mirrors how the tokens actually sit in the
   * sent context. A message with array content becomes one 'content_part' block
   * per part; otherwise one 'message' block. Mirrors Python `extract_blocks`.
   */
  extractBlocks(kwargs: Record<string, unknown>): RawBlock[] {
    const blocks: RawBlock[] = [];

    const system = kwargs["system"];
    if (typeof system === "string" && system) {
      blocks.push({ role: "system", kind: "message", text: system });
    } else if (Array.isArray(system)) {
      for (const part of system) {
        const text =
          typeof part === "string"
            ? part
            : isRecord(part)
              ? ((part["text"] as string) ?? "")
              : "";
        blocks.push({ role: "system", kind: "message", text });
      }
    }

    const tools = kwargs["tools"];
    if (Array.isArray(tools)) {
      for (const tool of tools) {
        blocks.push({ role: "system", kind: "tool_schema", text: sjson(tool) });
      }
    }

    const messages = kwargs["messages"];
    if (Array.isArray(messages)) {
      for (const msg of messages) {
        if (!isRecord(msg)) continue;
        const role = (msg["role"] as string) ?? "user";
        const content = msg["content"];
        if (Array.isArray(content)) {
          for (const part of content) {
            blocks.push({
              role,
              kind: "content_part",
              text: typeof part === "string" ? part : sjson(part),
            });
          }
        } else {
          blocks.push({
            role,
            kind: "message",
            text: typeof content === "string" ? content : content ? sjson(content) : "",
          });
        }
      }
    }
    return blocks;
  }

  /** Return request kwargs minus the content-bearing keys. Mirrors Python
   * `extract_params`. */
  extractParams(kwargs: Record<string, unknown>): Record<string, unknown> {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(kwargs)) {
      if (!CONTENT_KEYS.has(k)) out[k] = v;
    }
    return out;
  }

  /**
   * Map Anthropic's `response.usage.input_tokens/output_tokens` into a plain
   * object, or null when absent. A `.parse()` raw-response wrapper fallback is
   * included for transparency-hop callers (never throws). Mirrors Python
   * `extract_usage`.
   */
  extractUsage(response: unknown): Record<string, unknown> | null {
    let usage = this.readUsage(response);
    if (usage == null) {
      const parse = (response as { parse?: unknown } | null)?.parse;
      if (typeof parse === "function") {
        try {
          usage = this.readUsage(parse.call(response));
        } catch {
          usage = null;
        }
      }
    }
    if (usage == null) return null;
    const u = usage as Record<string, unknown>;
    return {
      input_tokens: u["input_tokens"] ?? null,
      output_tokens: u["output_tokens"] ?? null,
    };
  }

  private readUsage(response: unknown): unknown {
    if (response == null || typeof response !== "object") return null;
    return (response as { usage?: unknown }).usage ?? null;
  }

  /**
   * Fold usage from ONE streamed Anthropic event into `state`. Unlike OpenAI,
   * Anthropic splits input and output counts across TWO event types rather than
   * one final chunk: `message_start` carries the (already-known) input count at
   * `chunk.message.usage.input_tokens`, and `message_delta` (emitted once near
   * the end, alongside the stop reason) carries the output count at
   * `chunk.usage.output_tokens`. Both are written under the SAME key names
   * `extractUsage` reads, so the eventual synthetic response reads back
   * identically. Duck-typed and wrapped in try/catch so a malformed event never
   * interrupts the caller's own iteration. Mirrors Python
   * `accumulate_stream_usage`.
   */
  accumulateStreamUsage(chunk: unknown, state: Record<string, unknown>): void {
    try {
      if (chunk == null || typeof chunk !== "object") return;
      const c = chunk as Record<string, unknown>;
      const eventType = c["type"];
      if (eventType === "message_start") {
        const message = c["message"];
        const usage =
          message && typeof message === "object"
            ? (message as Record<string, unknown>)["usage"]
            : null;
        const inputTokens =
          usage && typeof usage === "object"
            ? (usage as Record<string, unknown>)["input_tokens"]
            : undefined;
        if (inputTokens !== undefined && inputTokens !== null) {
          state["input_tokens"] = inputTokens;
        }
      } else if (eventType === "message_delta") {
        const usage = c["usage"];
        const outputTokens =
          usage && typeof usage === "object"
            ? (usage as Record<string, unknown>)["output_tokens"]
            : undefined;
        if (outputTokens !== undefined && outputTokens !== null) {
          state["output_tokens"] = outputTokens;
        }
      }
    } catch {
      // never break the caller's iteration
    }
  }
}
