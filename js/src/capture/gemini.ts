/**
 * Google Gemini adapter (@google/genai SDK). Reads the request kwargs a caller
 * passes to `client.models.generateContent(...)` — plain objects/strings, or
 * the SDK's own config object duck-typed via property access — and the response
 * object, with no dependency on the @google/genai SDK itself.
 *
 * Two shapes this adapter absorbs that OpenAI/Anthropic don't have: `contents`
 * (no top-level `messages` key — a string, or a list of strings/objects-with-
 * parts) and `config` (an object carrying BOTH the system instruction/tools AND
 * sampling params like temperature, in one bag).
 *
 * Mirrors the Python `ctxdiff.capture.gemini` adapter's normalization. Field
 * names follow the JS SDK's camelCase (`systemInstruction`, `usageMetadata`,
 * `maxOutputTokens`), while the emitted usage dict keys use Python's snake_case
 * (`prompt_token_count`, ...) so a Gemini `.ctrace` renders identically in the
 * Python viewer.
 */
import type { Adapter } from "./base.js";
import type { RawBlock } from "../models.js";
import { stableStringify } from "../models.js";
import { imageRawBlock } from "../images.js";

// Request keys that carry block *content* rather than sampling params. `config`
// also carries sampling fields (temperature etc.) — those are pulled back out
// in extractParams.
const CONTENT_KEYS = new Set(["contents", "config"]);

// Non-content sampling fields that may live on `config`, in the JS SDK's
// camelCase spelling.
const CONFIG_PARAM_FIELDS = [
  "temperature",
  "maxOutputTokens",
  "topP",
  "topK",
] as const;

const ROLE_MAP: Record<string, string> = { model: "assistant" };

function sjson(x: unknown): string {
  return stableStringify(x);
}

function isRecord(x: unknown): x is Record<string, unknown> {
  return typeof x === "object" && x !== null && !Array.isArray(x);
}

/**
 * Read `name` off `config`, which is a plain object (object literal) or an SDK
 * config instance — property access works for both in JS (there is no dict vs
 * typed-object distinction the way Python's adapter must handle). Returns
 * undefined when `config` is absent or the field is missing.
 */
function cfgGet(config: unknown, name: string): unknown {
  if (config == null || typeof config !== "object") return undefined;
  return (config as Record<string, unknown>)[name];
}

export class GeminiAdapter implements Adapter {
  readonly provider = "gemini";
  // `generateContentStream` is a SEPARATELY-NAMED method (not a `stream:true`
  // kwarg on `generateContent`, unlike OpenAI/Anthropic) that returns a
  // Promise resolving to a DIRECT async-iterable of response chunks — confirmed
  // empirically (Phase JS2 probe against @google/genai 2.13.0). trace.ts routes
  // it through the raw-stream path (its last path segment ends with "stream"
  // but isn't exactly "stream", which instead means the `.stream()` manager
  // helpers), same as `create({stream:true})` on the other providers.
  readonly createPaths: string[][] = [
    ["models", "generateContent"],
    ["models", "generateContentStream"],
  ];

  /**
   * Flatten a request into ordered RawBlocks: `config.systemInstruction` first
   * (one 'system' 'message' block per instruction string — it sits at the very
   * front of the sent context), then tool schemas from `config.tools` (one
   * 'tool_schema' block each, JSON-serialized for stable diffing), then
   * `contents`. A plain string `contents` becomes one 'user' 'message' block. A
   * list `contents` yields, per entry: a string → one 'user' 'message' block;
   * an object with `parts` → one 'content_part' block per part (part text if the
   * part is a string or an object carrying a 'text' key, else stable JSON), with
   * role taken from the entry's 'role' ('model' → 'assistant', anything else
   * passed through, defaulting to 'user'). Mirrors Python `extract_blocks`.
   *
   * Parity note: a `systemInstruction` that is neither a string nor an array
   * (e.g. a typed `Content` object) is ignored — matching the Python adapter,
   * which only handles the string and list shapes.
   */
  extractBlocks(kwargs: Record<string, unknown>): RawBlock[] {
    const blocks: RawBlock[] = [];
    const config = kwargs["config"];

    const systemInstruction = cfgGet(config, "systemInstruction");
    if (typeof systemInstruction === "string" && systemInstruction) {
      blocks.push({ role: "system", kind: "message", text: systemInstruction });
    } else if (Array.isArray(systemInstruction)) {
      for (const part of systemInstruction) {
        const text =
          typeof part === "string"
            ? part
            : isRecord(part)
              ? ((part["text"] as string) ?? "")
              : "";
        blocks.push({ role: "system", kind: "message", text });
      }
    }

    const tools = cfgGet(config, "tools");
    if (Array.isArray(tools)) {
      for (const tool of tools) {
        blocks.push({ role: "system", kind: "tool_schema", text: sjson(tool) });
      }
    }

    const contents = kwargs["contents"];
    if (typeof contents === "string") {
      if (contents) blocks.push({ role: "user", kind: "message", text: contents });
    } else if (Array.isArray(contents)) {
      for (const entry of contents) {
        if (typeof entry === "string") {
          blocks.push({ role: "user", kind: "message", text: entry });
          continue;
        }
        const rawRole = isRecord(entry) ? (entry["role"] as string | undefined) : undefined;
        // `role || "user"` (not `?? "user"`) so an explicit empty-string role
        // degrades to "user", matching Python's `role or "user"` — an entry with
        // `role: ""` must not survive as "".
        const role = ROLE_MAP[rawRole ?? ""] ?? (rawRole || "user");
        const parts = isRecord(entry) ? entry["parts"] : undefined;
        if (Array.isArray(parts)) {
          for (const part of parts) {
            // An `inlineData`/`fileData` part carrying an image MIME type
            // becomes an 'image' block whose text is a short descriptor and
            // whose identity is the image bytes — never the base64 payload.
            // Non-image inline data (audio, video, PDF) is untouched and still
            // serializes as before; see src/images.ts.
            const image = imageRawBlock(role, part, this.provider);
            if (image !== null) {
              blocks.push(image);
              continue;
            }
            let text: string;
            if (typeof part === "string") {
              text = part;
            } else if (isRecord(part) && "text" in part) {
              // `?? ""` keeps `text` a real string when the key is present but
              // null — see the `text: null` parity note in spec/ctrace-schema.md.
              text = (part["text"] as string) ?? "";
            } else {
              text = sjson(part);
            }
            blocks.push({ role, kind: "content_part", text });
          }
        }
      }
    }
    return blocks;
  }

  /**
   * Return every request kwarg except the content-bearing 'contents'/'config'
   * keys, plus any non-content sampling fields (temperature, maxOutputTokens,
   * topP, topK) duck-typed off `config` when present — so 'model' and sampling
   * settings are captured without duplicating block content. Mirrors Python
   * `extract_params`.
   */
  extractParams(kwargs: Record<string, unknown>): Record<string, unknown> {
    const params: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(kwargs)) {
      if (!CONTENT_KEYS.has(k)) params[k] = v;
    }
    const config = kwargs["config"];
    for (const field of CONFIG_PARAM_FIELDS) {
      const value = cfgGet(config, field);
      if (value !== undefined && value !== null) params[field] = value;
    }
    return params;
  }

  /**
   * Pull provider-reported usage off `response.usageMetadata` (duck-typed) into
   * a plain object with Python's snake_case keys, or null when absent. Unlike
   * OpenAI/Anthropic, Gemini has no raw-response wrapper hop to fall back
   * through, so a plain property read suffices. Mirrors Python `extract_usage`.
   */
  extractUsage(response: unknown): Record<string, unknown> | null {
    if (response == null || typeof response !== "object") return null;
    const usage = (response as { usageMetadata?: unknown }).usageMetadata;
    if (usage == null || typeof usage !== "object") return null;
    const u = usage as Record<string, unknown>;
    return {
      prompt_token_count: u["promptTokenCount"] ?? null,
      candidates_token_count: u["candidatesTokenCount"] ?? null,
      total_token_count: u["totalTokenCount"] ?? null,
    };
  }

  /**
   * Fold ONE streamed chunk's `usageMetadata` into `state`. Gemini's
   * `usageMetadata` is CUMULATIVE — each successive chunk already carries the
   * running totals for the whole response so far (confirmed Phase JS2 probe),
   * unlike OpenAI (a single final delta) or Anthropic (split across two events).
   * So accumulation here is a plain OVERWRITE with the latest chunk's counts
   * (last chunk wins), never a sum; a chunk with no `usageMetadata` leaves
   * `state` unchanged. State keys are the SDK's camelCase names so
   * `extractUsage` reads them off the synthetic stream response identically to a
   * real one (see trace.ts's synthetic response exposing state under BOTH
   * `usage` and `usageMetadata`). Wrapped in try/catch — a malformed chunk must
   * never interrupt the caller's iteration. Mirrors Python
   * `accumulate_stream_usage`.
   */
  accumulateStreamUsage(chunk: unknown, state: Record<string, unknown>): void {
    try {
      if (chunk == null || typeof chunk !== "object") return;
      const usage = (chunk as { usageMetadata?: unknown }).usageMetadata;
      if (usage != null && typeof usage === "object") {
        const u = usage as Record<string, unknown>;
        state["promptTokenCount"] = u["promptTokenCount"] ?? null;
        state["candidatesTokenCount"] = u["candidatesTokenCount"] ?? null;
        state["totalTokenCount"] = u["totalTokenCount"] ?? null;
      }
    } catch {
      // never break the caller's iteration
    }
  }
}
