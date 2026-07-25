/**
 * OpenAI adapter, covering BOTH completion methods the SDK exposes: the
 * established Chat Completions API (`client.chat.completions.create(...)`) and
 * its successor, the Responses API (`client.responses.create(...)`). It works
 * on plain request objects/arrays with no dependency on the openai SDK itself.
 * The two request shapes are disjoint (`messages` vs `input`/`instructions`) so
 * `extractBlocks`/`extractParams` dispatch on which keys are present;
 * `extractUsage` duck-types both response usage shapes.
 *
 * Mirrors the Python `ctxdiff.capture.openai` adapter's block model
 * byte-for-byte: same block ordering, same roles/kinds, same stable-JSON
 * serialization of tool schemas and multi-part content.
 */
import type { Adapter } from "./base.js";
import type { RawBlock } from "../models.js";
import { stableStringify } from "../models.js";
import { imageRawBlock } from "../images.js";

// Request keys that carry block *content* rather than sampling params.
const CHAT_CONTENT_KEYS = new Set(["messages", "tools"]);
// Responses: `input`/`instructions` carry conversation content, `tools` the
// schemas. `previous_response_id` is deliberately NOT here — it's chain
// linkage, not content, so it stays in params.
const RESPONSES_CONTENT_KEYS = new Set(["input", "instructions", "tools"]);
// Part types within a Responses `input` content list whose "text" field is the
// human-readable payload; anything else falls back to stable JSON of the part.
const RESPONSES_TEXT_PART_TYPES = new Set(["input_text", "output_text"]);

/** Stable JSON identical to Python's `json.dumps(x, sort_keys=True,
 * ensure_ascii=False)` — the exact serialization the Python adapter uses for
 * tool schemas and non-string content parts, so hashes match cross-language. */
function sjson(x: unknown): string {
  return stableStringify(x);
}

function isRecord(x: unknown): x is Record<string, unknown> {
  return typeof x === "object" && x !== null && !Array.isArray(x);
}

/**
 * Distinguish the Responses request shape from Chat Completions: the two are
 * disjoint on the wire — Chat Completions always sends `messages`; Responses
 * never does, using `input`/`instructions` instead. Kwargs with neither key
 * default to the chat shape. Mirrors Python `_is_responses_shape`.
 */
function isResponsesShape(kwargs: Record<string, unknown>): boolean {
  return (
    !("messages" in kwargs) && ("input" in kwargs || "instructions" in kwargs)
  );
}

export class OpenAIAdapter implements Adapter {
  readonly provider = "openai";
  readonly createPaths: string[][] = [
    ["chat", "completions", "create"],
    ["responses", "create"],
    ["chat", "completions", "stream"],
    ["responses", "stream"],
  ];

  extractBlocks(kwargs: Record<string, unknown>): RawBlock[] {
    if (isResponsesShape(kwargs)) return this.extractResponsesBlocks(kwargs);
    return this.extractChatBlocks(kwargs);
  }

  /**
   * Flatten a Responses-API request into ordered RawBlocks: `instructions`
   * FIRST (a system message at the front of context), then `tools` (Responses
   * tool schemas are already flat — NOT nested under a "function" key the way
   * Chat Completions' are — serialized to stable JSON), then `input` in wire
   * order. Every item is handled defensively so a malformed shape never throws.
   * Mirrors Python `_extract_responses_blocks`.
   */
  private extractResponsesBlocks(
    kwargs: Record<string, unknown>,
  ): RawBlock[] {
    const blocks: RawBlock[] = [];
    const instructions = kwargs["instructions"];
    if (instructions) {
      blocks.push({
        role: "system",
        kind: "message",
        text: typeof instructions === "string" ? instructions : sjson(instructions),
      });
    }
    const tools = (kwargs["tools"] as unknown[]) || [];
    if (Array.isArray(tools)) {
      for (const tool of tools) {
        blocks.push({ role: "system", kind: "tool_schema", text: sjson(tool) });
      }
    }
    const input = kwargs["input"];
    if (typeof input === "string") {
      blocks.push({ role: "user", kind: "message", text: input });
    } else if (Array.isArray(input)) {
      for (const item of input) {
        blocks.push(...this.extractResponsesInputItem(item));
      }
    }
    return blocks;
  }

  /** Extract the block(s) for one entry of a Responses `input` list. Mirrors
   * Python `_extract_responses_input_item`. */
  private extractResponsesInputItem(item: unknown): RawBlock[] {
    if (isRecord(item) && "role" in item) {
      const role = (item["role"] as string) ?? "user";
      const content = item["content"];
      if (Array.isArray(content)) {
        return content.map((part) => {
          // An `input_image` part becomes an 'image' block whose text is a
          // short descriptor and whose identity is the image bytes — never the
          // base64 payload. See src/images.ts.
          const image = imageRawBlock(role, part, this.provider);
          if (image !== null) return image;
          let text: string;
          if (
            isRecord(part) &&
            RESPONSES_TEXT_PART_TYPES.has(part["type"] as string)
          ) {
            text = (part["text"] as string) ?? "";
          } else {
            text = sjson(part);
          }
          return { role, kind: "content_part", text };
        });
      }
      return [
        {
          role,
          kind: "message",
          text: typeof content === "string" ? content : content ? sjson(content) : "",
        },
      ];
    }
    if (isRecord(item) && item["type"] === "function_call") {
      return [{ role: "assistant", kind: "content_part", text: sjson(item) }];
    }
    if (isRecord(item) && item["type"] === "function_call_output") {
      return [{ role: "tool", kind: "content_part", text: sjson(item) }];
    }
    // Defensive fallback: any other shape is still captured, not dropped.
    return [{ role: "user", kind: "content_part", text: sjson(item) }];
  }

  /**
   * Flatten a Chat Completions request into ordered RawBlocks: tool schemas
   * first (they occupy the front of the context window), then each message. A
   * message with array content becomes one 'content_part' block per part;
   * otherwise one 'message' block. Then one 'content_part' per `tool_calls`
   * entry (and the legacy single `function_call`). When content is empty but
   * tool_calls/function_call is present, the empty 'message' block is skipped —
   * the tool_call parts ARE the message. Mirrors Python `_extract_chat_blocks`.
   */
  private extractChatBlocks(kwargs: Record<string, unknown>): RawBlock[] {
    const blocks: RawBlock[] = [];
    const tools = (kwargs["tools"] as unknown[]) || [];
    if (Array.isArray(tools)) {
      for (const tool of tools) {
        blocks.push({ role: "system", kind: "tool_schema", text: sjson(tool) });
      }
    }
    const messages = (kwargs["messages"] as unknown[]) || [];
    if (!Array.isArray(messages)) return blocks;
    for (const msg of messages) {
      if (!isRecord(msg)) continue;
      const role = (msg["role"] as string) ?? "user";
      const content = msg["content"];
      const toolCalls = (msg["tool_calls"] as unknown[]) || [];
      const functionCall = msg["function_call"];
      const hasCalls =
        (Array.isArray(toolCalls) && toolCalls.length > 0) || !!functionCall;

      if (Array.isArray(content)) {
        for (const part of content) {
          // An `image_url` part becomes an 'image' block whose text is a short
          // descriptor and whose identity is the image bytes — never the base64
          // data URI. See src/images.ts.
          const image = imageRawBlock(role, part, this.provider);
          if (image !== null) {
            blocks.push(image);
            continue;
          }
          blocks.push({
            role,
            kind: "content_part",
            text: typeof part === "string" ? part : sjson(part),
          });
        }
      } else if (content || !hasCalls) {
        blocks.push({
          role,
          kind: "message",
          text: typeof content === "string" ? content : content ? sjson(content) : "",
        });
      }

      if (Array.isArray(toolCalls)) {
        for (const call of toolCalls) {
          blocks.push({
            role,
            kind: "content_part",
            text: typeof call === "string" ? call : sjson(call),
          });
        }
      }
      if (functionCall) {
        blocks.push({
          role,
          kind: "content_part",
          text: typeof functionCall === "string" ? functionCall : sjson(functionCall),
        });
      }
    }
    return blocks;
  }

  /**
   * Return every request kwarg except the content-bearing keys, so stored
   * params capture model/sampling settings without duplicating blocks. Which
   * keys count as content depends on the shape. Mirrors Python `extract_params`.
   */
  extractParams(kwargs: Record<string, unknown>): Record<string, unknown> {
    const contentKeys = isResponsesShape(kwargs)
      ? RESPONSES_CONTENT_KEYS
      : CHAT_CONTENT_KEYS;
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(kwargs)) {
      if (!contentKeys.has(k)) out[k] = v;
    }
    return out;
  }

  /**
   * Pull provider-reported usage off `response.usage` (duck-typed) into a plain
   * object, or null when absent. Two naming families: Chat Completions reports
   * prompt_tokens/completion_tokens/total_tokens; Responses reports
   * input_tokens/output_tokens/total_tokens. Both are duck-typed so ONE method
   * serves both shapes; if both somehow appeared, the prompt_tokens family wins.
   * Mirrors Python `extract_usage`. A `.parse()` raw-response wrapper fallback
   * is included for transparency-hop callers (never throws).
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
    const promptTokens = u["prompt_tokens"] ?? null;
    const completionTokens = u["completion_tokens"] ?? null;
    if (promptTokens !== null || completionTokens !== null) {
      return {
        prompt_tokens: promptTokens,
        completion_tokens: completionTokens,
        total_tokens: u["total_tokens"] ?? null,
      };
    }
    const inputTokens = u["input_tokens"] ?? null;
    const outputTokens = u["output_tokens"] ?? null;
    if (inputTokens !== null || outputTokens !== null) {
      return {
        input_tokens: inputTokens,
        output_tokens: outputTokens,
        total_tokens: u["total_tokens"] ?? null,
      };
    }
    // Neither family present: preserve the chat default shape.
    return {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: u["total_tokens"] ?? null,
    };
  }

  private readUsage(response: unknown): unknown {
    if (response == null || typeof response !== "object") return null;
    const u = (response as { usage?: unknown }).usage;
    return u ?? null;
  }

  /**
   * Fold usage from ONE streamed chunk into `state`, covering the two shapes an
   * OpenAI stream can carry (confirmed empirically against openai 6.49.0, JS
   * SDK, Phase JS1 Step 0):
   *
   * - Chat Completions (`create({stream:true})` AND the `.stream()` helper):
   *   both iterate raw `ChatCompletionChunk`s in JS (the JS `.stream()` helper
   *   yields raw chunks, NOT the typed events the Python SDK's helper emits).
   *   Every chunk has `.usage`, non-null only on the LAST chunk and ONLY when
   *   the caller opted in via `stream_options:{include_usage:true}` — ctxdiff
   *   never injects that itself. Its shape matches a non-streaming
   *   `ChatCompletion.usage`, so the same key names are written into `state`.
   * - Responses (`create({stream:true})` AND `responses.stream()`): the
   *   terminal `response.completed` event carries the completed Response at
   *   `.response`, whose `.usage` is input/output/total_tokens — emitted
   *   unconditionally, no opt-in needed.
   *
   * A defensive `chunk.chunk.usage` branch mirrors the Python `.stream()`
   * typed-event shape; it is inert on the JS SDK (which yields raw chunks) but
   * costs nothing and future-proofs against a helper that re-wraps chunks.
   *
   * Duck-typed throughout and wrapped in try/catch: a malformed chunk must
   * never interrupt the caller's own iteration.
   */
  accumulateStreamUsage(chunk: unknown, state: Record<string, unknown>): void {
    try {
      if (chunk == null || typeof chunk !== "object") return;
      const c = chunk as Record<string, unknown>;

      const usage = c["usage"];
      if (usage && typeof usage === "object") {
        if (this.foldChatUsage(usage as Record<string, unknown>, state)) return;
      }

      if (c["type"] === "response.completed") {
        const resp = c["response"];
        const respUsage =
          resp && typeof resp === "object"
            ? (resp as Record<string, unknown>)["usage"]
            : null;
        if (respUsage && typeof respUsage === "object") {
          const ru = respUsage as Record<string, unknown>;
          state["input_tokens"] = ru["input_tokens"] ?? null;
          state["output_tokens"] = ru["output_tokens"] ?? null;
          state["total_tokens"] = ru["total_tokens"] ?? null;
          return;
        }
      }

      if (c["type"] === "chunk") {
        const inner = c["chunk"];
        const innerUsage =
          inner && typeof inner === "object"
            ? (inner as Record<string, unknown>)["usage"]
            : null;
        if (innerUsage && typeof innerUsage === "object") {
          this.foldChatUsage(innerUsage as Record<string, unknown>, state);
        }
      }
    } catch {
      // never break the caller's iteration
    }
  }

  /** Write chat-family usage into state if present; return whether it did. */
  private foldChatUsage(
    usage: Record<string, unknown>,
    state: Record<string, unknown>,
  ): boolean {
    const promptTokens = usage["prompt_tokens"] ?? null;
    const completionTokens = usage["completion_tokens"] ?? null;
    if (promptTokens !== null || completionTokens !== null) {
      state["prompt_tokens"] = promptTokens;
      state["completion_tokens"] = completionTokens;
      state["total_tokens"] = usage["total_tokens"] ?? null;
      return true;
    }
    return false;
  }
}
