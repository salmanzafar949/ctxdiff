/**
 * AWS Bedrock adapter (Converse API, via `@aws-sdk/client-bedrock-runtime`).
 *
 * The block/param/usage extractors below are a faithful port of Python's
 * `ctxdiff.capture.bedrock`, and they have to be: the Converse WIRE SHAPE is
 * identical in the two SDKs (`messages: [{role, content: [{text}|{image}|
 * {toolUse}|{toolResult}]}]`, `system: [{text}]`, `toolConfig.tools[].toolSpec`,
 * `inferenceConfig`), so the same logical request captured by the Python SDK
 * and by this one must produce the same blocks and therefore the same hashes.
 * `test/conformance.test.ts` pins that against the real Python extractor.
 *
 * WHAT IS NOT SHARED WITH PYTHON IS HOW THE CALL IS INTERCEPTED. boto3 exposes
 * one METHOD PER OPERATION — `client.converse(...)`, `client.converse_stream(
 * ...)` — so Python's adapter names them as attribute paths and trace.py wraps
 * them. The AWS SDK v3 has no such methods. It has exactly ONE:
 *
 *     await client.send(new ConverseCommand({ modelId, messages, ... }))
 *     await client.send(new ConverseStreamCommand({ ... }))   // -> { stream }
 *
 * The operation and the payload both live on the COMMAND OBJECT, so `send` is
 * the only interceptable path and the real dispatch happens on the argument —
 * `interpretCall` below, called by trace.ts's interceptor. That keeps trace.ts
 * provider-agnostic: it asks the adapter "is this call recordable, what is its
 * request payload, and does it stream?" and the Bedrock-specific knowledge of
 * command classes stays here.
 *
 * ANY OTHER COMMAND PASSES THROUGH UNRECORDED AND SILENT. `InvokeModelCommand`
 * carries a raw provider-specific `body` (an Anthropic Messages payload, a
 * Titan payload, a Cohere payload — a JSON *string*, whose schema depends on
 * `modelId`), not the Converse shape; guessing at it would store blocks that
 * misrepresent the request. Embeddings, `ApplyGuardrail`, `ListInvocations`
 * and the async-invoke family are not LLM context at all. So they are handed
 * straight back to the host with nothing recorded, and WITHOUT a warning: one
 * `BedrockRuntimeClient` commonly serves both Converse and embeddings, and a
 * line of log noise per unrelated `send()` would train users to ignore ctxdiff's
 * warnings — which is worse than the silence, because the warnings that DO fire
 * (a degraded store, a dead recorder) are the ones that matter.
 */
import type { Adapter, CallShape } from "./base.js";
import type { RawBlock } from "../models.js";
import { stableStringify } from "../models.js";
import { imageRawBlock } from "../images.js";

/**
 * Request keys excluded from params verbatim: `system`/`messages`/`toolConfig`
 * carry block CONTENT (kept out so content is never stored twice);
 * `inferenceConfig` is dropped in its raw object form too because its scalar
 * fields are flattened into params individually below instead.
 */
const CONTENT_KEYS = new Set(["system", "messages", "toolConfig", "inferenceConfig"]);

/** `inferenceConfig` fields that are sampling params, not content; flattened
 * directly into params under their own (Converse-native) names. */
const INFERENCE_CONFIG_FIELDS = ["maxTokens", "temperature", "topP", "stopSequences"] as const;

/**
 * The two Converse operations, and whether each one streams. Keyed by the
 * SMITHY OPERATION NAME (`Converse`, `ConverseStream`) rather than the command
 * class name, because that is what `operationOf` normalizes to. Every other
 * operation is absent on purpose — see the module docstring.
 */
const OPERATIONS: Record<string, { streaming: boolean }> = {
  Converse: { streaming: false },
  ConverseStream: { streaming: true },
};

function isRecord(x: unknown): x is Record<string, unknown> {
  return typeof x === "object" && x !== null && !Array.isArray(x);
}

function sjson(x: unknown): string {
  return stableStringify(x);
}

/**
 * The Smithy operation name behind one AWS SDK v3 command object, or null when
 * the argument is not a command at all.
 *
 * Two signals, in order, because neither alone is safe:
 *
 *  1. THE CLASS NAME — `ConverseCommand` -> `Converse`. Correct for every
 *     command the SDK exports, and readable, but a bundler that mangles class
 *     names (an aggressive browser/edge build) turns it into `t`.
 *  2. THE COMMAND'S SMITHY SCHEMA — `command.schema` is the operation's wire
 *     descriptor, `[traitBits, "com.amazonaws.bedrockruntime", "Converse",
 *     {http: [...]}, ...]`, with the operation name at index 2 (confirmed
 *     against @aws-sdk/client-bedrock-runtime 3.9xx). It is data, not an
 *     identifier, so minification cannot touch it.
 *
 * Reading a stray `.schema` off some unrelated object cannot cause a false
 * capture: whatever comes back still has to be one of the two names in
 * `OPERATIONS`, and anything else passes through unrecorded.
 */
function operationOf(command: unknown): string | null {
  if (command == null || typeof command !== "object") return null;
  const name = (command as { constructor?: { name?: unknown } }).constructor?.name;
  if (typeof name === "string" && name.length > "Command".length && name.endsWith("Command")) {
    return name.slice(0, -"Command".length);
  }
  const schema = (command as { schema?: unknown }).schema;
  if (Array.isArray(schema) && typeof schema[2] === "string") return schema[2];
  return null;
}

export class BedrockAdapter implements Adapter {
  readonly provider = "bedrock";

  /**
   * ONE path, because the AWS SDK v3 has one method: `client.send(command)`.
   * Which operation is being performed — and whether it streams — is decided
   * per call by `interpretCall`, not by the path.
   */
  readonly createPaths: string[][] = [["send"]];

  /**
   * WHERE the iterator lives in what a `ConverseStreamCommand` resolves to.
   * Unlike every other provider's streaming call — which hands back the stream
   * itself — this one returns an ENVELOPE that merely CONTAINS it:
   * `{ $metadata: {...}, stream: AsyncIterable<ConverseStreamOutput> }`
   * (confirmed against the real SDK, exactly as botocore's
   * `{"ResponseMetadata": ..., "stream": EventStream}` on the Python side).
   *
   * Declaring the member by name lets trace.ts proxy ONLY that member and hand
   * the host its envelope otherwise untouched, rather than wrapping the object
   * itself — which would give the caller a proxy whose next line,
   * `response.$metadata`, reads through a stream proxy. Mirrors Python's
   * `BedrockAdapter.stream_envelope_key`.
   */
  readonly streamEnvelopeKey = "stream";

  /**
   * Decide what ONE intercepted `client.send(...)` call is, from the command
   * object the host passed.
   *
   * Returns the request payload (`command.input` — already the Converse wire
   * shape, which is why every extractor below is shared verbatim with the
   * boto3 path) plus whether the result will be a stream, or NULL for "not a
   * call this adapter records", which trace.ts turns into an untouched
   * pass-through.
   *
   * THE LEGACY CALLBACK FORM IS NOT RECORDED. `send(command, cb)` is a real,
   * still-supported typed overload of the AWS SDK v3 client, and on it `send`
   * returns `undefined` instead of a Promise — the result arrives at the host's
   * callback. The interceptor cannot see that result, so it would take its
   * synchronous non-streaming branch and record the call with `usage: null` and
   * `latencyMs: 0`: numbers that were never observed, sitting in a trace whose
   * whole value is being evidence of what actually happened. A MISSING call is
   * an obvious gap; a call with fabricated zeros is a wrong answer. So the
   * callback form returns null here and passes through untouched — the host's
   * callback still fires with its real response, exactly as if ctxdiff were not
   * installed. (The promise form is the SDK's documented default and what every
   * `await client.send(...)` uses, so this costs capture only for a legacy style.)
   *
   * `args[1]` when it is NOT a function (the middleware/http options bag) is
   * deliberately ignored rather than rejected: it changes how the request is
   * dispatched, never what is in it, and the interceptor forwards every
   * argument to the real `send` unchanged.
   */
  interpretCall(args: unknown[]): CallShape | null {
    if (typeof args[1] === "function") return null;
    const command = args[0];
    const operation = operationOf(command);
    if (operation === null) return null;
    const op = OPERATIONS[operation];
    if (op === undefined) return null;
    const input = (command as { input?: unknown }).input;
    // A command with no `input` is still that operation, just empty; record it
    // as an empty request rather than dropping the turn.
    return { kwargs: isRecord(input) ? input : {}, streaming: op.streaming };
  }

  /**
   * Flatten a Converse request into ordered RawBlocks: `system` first (one
   * 'system'-role 'message' block per entry — text from the entry's `text` key
   * when present, else stable JSON for any other Converse system-block shape
   * such as `cachePoint`), then tool schemas from `toolConfig.tools` (one
   * 'tool_schema' block per tool, JSON of the `toolSpec`), then `messages` (one
   * 'content_part' block per content entry).
   *
   * Order mirrors SEND order — system → tool schemas → messages — the way the
   * tokens actually sit in the sent context. Roles pass through as-is: Converse
   * has no 'tool' role, a `toolResult` lives inside a user-role message's
   * `content` list.
   *
   * An `{image: {format, source: {bytes}}}` entry becomes an `[image W×H · ~N
   * tok]` block whose identity is the image BYTES, via the shared
   * `imageRawBlock` — never the base64/serialized payload, which would both
   * mis-measure the cost and make the same picture a different block each turn.
   * Mirrors Python `extract_blocks`.
   */
  extractBlocks(kwargs: Record<string, unknown>): RawBlock[] {
    const blocks: RawBlock[] = [];

    const system = kwargs["system"];
    if (Array.isArray(system)) {
      for (const entry of system) {
        const text =
          isRecord(entry) && "text" in entry ? ((entry["text"] as string) ?? "") : sjson(entry);
        blocks.push({ role: "system", kind: "message", text });
      }
    }

    const toolConfig = kwargs["toolConfig"];
    const tools = isRecord(toolConfig) ? toolConfig["tools"] : undefined;
    if (Array.isArray(tools)) {
      for (const tool of tools) {
        const spec = isRecord(tool) && "toolSpec" in tool ? tool["toolSpec"] : tool;
        blocks.push({ role: "system", kind: "tool_schema", text: sjson(spec) });
      }
    }

    const messages = kwargs["messages"];
    if (Array.isArray(messages)) {
      for (const msg of messages) {
        // `role || "user"` (not `??`) so an explicit empty-string role degrades
        // to "user", matching Python's `msg.get("role", "user")` on real
        // payloads where the key is always present and non-empty.
        const rawRole = isRecord(msg) ? (msg["role"] as string | undefined) : undefined;
        const role = rawRole || "user";
        const content = isRecord(msg) ? msg["content"] : undefined;
        if (!Array.isArray(content)) continue;
        for (const part of content) {
          const image = imageRawBlock(role, part, this.provider);
          if (image !== null) {
            blocks.push(image);
            continue;
          }
          const text =
            isRecord(part) && "text" in part ? ((part["text"] as string) ?? "") : sjson(part);
          blocks.push({ role, kind: "content_part", text });
        }
      }
    }
    return blocks;
  }

  /**
   * Return every request key except the content-bearing `system`/`messages`/
   * `toolConfig` ones (so `modelId` survives untouched), plus any
   * `inferenceConfig` scalars (maxTokens, temperature, topP, stopSequences)
   * flattened in when present — a missing field never appears in params as a
   * spurious null. Mirrors Python `extract_params`.
   */
  extractParams(kwargs: Record<string, unknown>): Record<string, unknown> {
    const params: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(kwargs)) {
      if (!CONTENT_KEYS.has(k)) params[k] = v;
    }
    const inferenceConfig = kwargs["inferenceConfig"];
    if (isRecord(inferenceConfig)) {
      for (const field of INFERENCE_CONFIG_FIELDS) {
        const value = inferenceConfig[field];
        if (value !== undefined && value !== null) params[field] = value;
      }
    }
    return params;
  }

  /**
   * Pull provider-reported usage off `response.usage` into a plain object, or
   * null when the response carries none. A `ConverseCommandOutput` reports
   * `{inputTokens, outputTokens, totalTokens}` — the SAME key names the Python
   * SDK stores, since Converse's wire JSON is camelCase in both languages, so
   * a Bedrock `.ctrace` written here renders identically in the Python viewer
   * with no translation (contrast Gemini, whose two SDKs disagree on case).
   * Mirrors Python `extract_usage`.
   */
  extractUsage(response: unknown): Record<string, unknown> | null {
    if (response == null || typeof response !== "object") return null;
    const usage = (response as { usage?: unknown }).usage;
    if (usage == null || typeof usage !== "object") return null;
    const u = usage as Record<string, unknown>;
    return {
      inputTokens: u["inputTokens"] ?? null,
      outputTokens: u["outputTokens"] ?? null,
      totalTokens: u["totalTokens"] ?? null,
    };
  }

  /**
   * Fold usage from ONE `ConverseStream` event into `state`.
   *
   * A Converse stream emits `messageStart` → `contentBlockDelta`* →
   * `contentBlockStop` → `messageStop` → and, LAST, a single `metadata` event
   * carrying the whole exchange's counts at `chunk.metadata.usage`: Bedrock
   * reports input and output together, once, at the end — unlike Anthropic's
   * split across two events, and with no caller opt-in to arrange, unlike
   * OpenAI chat's `stream_options`. So this is a plain overwrite from that one
   * event and a no-op for every other event type.
   *
   * State keys are the SDK's own names, so `extractUsage` reads them off the
   * synthetic stream response exactly as it would off a real one. Wrapped in a
   * catch-all like every other adapter's: a malformed or unexpected event must
   * never interrupt the caller's own iteration. Mirrors Python
   * `accumulate_stream_usage`.
   */
  accumulateStreamUsage(chunk: unknown, state: Record<string, unknown>): void {
    try {
      if (chunk == null || typeof chunk !== "object") return;
      const metadata = (chunk as { metadata?: unknown }).metadata;
      if (!isRecord(metadata)) return;
      const usage = metadata["usage"];
      if (!isRecord(usage)) return;
      state["inputTokens"] = usage["inputTokens"] ?? null;
      state["outputTokens"] = usage["outputTokens"] ?? null;
      state["totalTokens"] = usage["totalTokens"] ?? null;
    } catch {
      // never break the caller's iteration
    }
  }
}
