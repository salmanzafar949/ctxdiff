/**
 * The LangChain callback handler — capture through LangChain's OWN extension
 * point instead of through its internals. The JS twin of Python's
 * `ctxdiff/capture/langchain.py`, and deliberately a line-for-line analogue:
 * the two must derive the same provider, rebuild the same wire shapes and map
 * usage onto the same key names, or a `.ctrace` written by one SDK would stop
 * deduping against one written by the other.
 *
 * WHY A CALLBACK. `tracer.wrap()` needs a provider SDK client; a LangChain app
 * hands you a `ChatOpenAI`, not an `OpenAI`. Reaching inside LangChain for the
 * client it happens to hold is a bet on someone else's private structure — it
 * breaks silently on any refactor. A callback is the supported way in: it
 * fires for EVERY integration, streaming or not, and LangGraph propagates it
 * through a whole graph, so one handler covers a whole agent.
 *
 * THE HARD REQUIREMENT: HASH IDENTITY. A trace captured through LangChain and
 * one captured by wrapping the SDK directly must produce the SAME BLOCKS for
 * the same logical request — same role, kind, text, hash — or the two would
 * never dedup and a team using both would see phantom "everything changed"
 * diffs. The mechanism is NOT a second block extractor:
 *
 *     LangChain messages  ──►  the provider's OWN WIRE SHAPE  ──►  the SAME
 *                              (plain objects)                     adapter
 *
 * `toWire()` rebuilds the request object LangChain itself is about to send
 * (verified against real request bodies — see `test/langchain.test.ts`), and
 * that object goes to the very same `OpenAIAdapter`/`AnthropicAdapter`/
 * `GeminiAdapter`/`BedrockAdapter` the direct path uses. That includes
 * multimodal content: a
 * message's parts are rebuilt ONE PER ENTRY (see `wireParts`), so an image
 * reaches the adapter as an image and is hashed over its bytes exactly as a
 * direct capture's would be.
 *
 * The claim is "hash-identical to a direct capture IN THE SAME SDK", and it
 * holds without exception. Across SDKs it holds for everything except a tool
 * call, whose arguments LangChain re-serializes with the host language's own
 * JSON serializer — see `toolCallsOf` for why that is inherent.
 *
 * NO IMPORT OF LANGCHAIN. Everything here is duck-typed, and the handler is
 * returned as a plain object implementing LangChain's `CallbackHandlerMethods`
 * — which `callbacks: [...]` accepts directly in JS (unlike Python, whose
 * pydantic-validated field demands a real `BaseCallbackHandler` subclass). So
 * ctxdiff keeps zero LangChain dependencies, optional or otherwise.
 */

/** LangChain's own provider id (`metadata.ls_provider`, set by @langchain/core
 * for every chat model) mapped to ctxdiff's adapter name. Tried FIRST because
 * the framework sets it, so it survives an integration renaming its class. */
const LS_PROVIDERS: Record<string, string> = {
  openai: "openai",
  azure_openai: "openai",
  azure: "openai",
  anthropic: "anthropic",
  google_vertexai: "gemini",
  google_genai: "gemini",
  google_anthropic_vertex: "anthropic",
  amazon_bedrock: "bedrock",
  bedrock: "bedrock",
  bedrock_converse: "bedrock",
  ollama: "openai", // OpenAI-compatible wire shape
  together: "openai",
  fireworks: "openai",
  groq: "openai",
  deepseek: "openai",
  xai: "openai",
};

/** Fallback signal: the model class name, from `serialized.id[-1]`. Used when
 * `ls_provider` is absent — an older @langchain/core, or a hand-rolled chat
 * model that sets no metadata. */
const CLASS_PROVIDERS: Record<string, string> = {
  ChatOpenAI: "openai",
  AzureChatOpenAI: "openai",
  ChatAnthropic: "anthropic",
  ChatVertexAI: "gemini",
  ChatGoogleGenerativeAI: "gemini",
  ChatBedrock: "bedrock",
  ChatBedrockConverse: "bedrock",
  BedrockChat: "bedrock",
};

/** What an unrecognized integration is treated as. OpenAI's chat shape is the
 * de-facto lingua franca, and the choice is visible rather than silent: the
 * provider is stored on the call, so a trace says which adapter read it. */
export const DEFAULT_PROVIDER = "openai";

/** LangChain message type -> the role provider wire formats use. */
const ROLES: Record<string, string> = {
  human: "user",
  ai: "assistant",
  system: "system",
  tool: "tool",
  function: "function",
  developer: "developer",
};

/** Per provider, the key names that provider's adapter READS off a response —
 * so LangChain's normalized counts are handed to `extractUsage` in exactly the
 * shape a real response would have carried them, and the usage finally STORED
 * is whatever `extractUsage` returns, byte-identical to a direct capture's. A
 * null third entry means that provider reports no total (Anthropic doesn't).
 *
 * Gemini is the one place these are NOT the stored names: the @google/genai JS
 * SDK spells its counts `promptTokenCount`/`candidatesTokenCount`/
 * `totalTokenCount` (camelCase), and `GeminiAdapter.extractUsage` converts
 * them to the snake_case names a `.ctrace` stores — the same conversion, and
 * the same camelCase state keys, the streaming path's `accumulateStreamUsage`
 * already writes. Writing the STORED names here instead meant `extractUsage`
 * looked for `promptTokenCount`, found nothing, and recorded a LangChain
 * Gemini call with `{prompt_token_count: null, …}` — counts LangChain had
 * reported correctly and this map then threw away. (Python has no such split:
 * google-genai's Python SDK uses the snake_case names natively.) */
const USAGE_KEYS: Record<string, [string, string, string | null]> = {
  openai: ["prompt_tokens", "completion_tokens", "total_tokens"],
  anthropic: ["input_tokens", "output_tokens", null],
  gemini: ["promptTokenCount", "candidatesTokenCount", "totalTokenCount"],
  bedrock: ["inputTokens", "outputTokens", "totalTokens"],
};

/** Where each provider's request object carries the model id. */
const MODEL_KEYS: Record<string, string> = {
  openai: "model",
  anthropic: "model",
  gemini: "model",
  bedrock: "modelId",
};

type Dict = Record<string, unknown>;

function asDict(value: unknown): Dict | null {
  return value != null && typeof value === "object" && !Array.isArray(value)
    ? (value as Dict)
    : null;
}

/**
 * Decide which ctxdiff adapter reads this LangChain request.
 *
 * The handler never sees a provider SDK client — only provider-AGNOSTIC
 * messages — so the provider comes from the model description LangChain passes
 * alongside them: `metadata.ls_provider` first, then the class name from
 * `serialized.id`. Anything unrecognized falls back to `DEFAULT_PROVIDER`
 * rather than throwing: a debugger that refuses to record an unfamiliar
 * integration is worse than one that records it in the most widely-compatible
 * shape and says which shape it used. Mirrors Python `provider_for`.
 */
export function providerFor(serialized: unknown, metadata: unknown): string {
  const meta = asDict(metadata);
  const ls = meta?.["ls_provider"];
  if (typeof ls === "string") {
    const mapped = LS_PROVIDERS[ls.toLowerCase()];
    if (mapped) return mapped;
  }
  const ser = asDict(serialized);
  let name: unknown = undefined;
  const id = ser?.["id"];
  if (Array.isArray(id) && id.length > 0) name = id[id.length - 1];
  if (typeof name !== "string") name = ser?.["name"];
  if (typeof name === "string" && CLASS_PROVIDERS[name]) return CLASS_PROVIDERS[name];
  return DEFAULT_PROVIDER;
}

/** The wire role for one LangChain message. Duck-typed so this module never
 * imports langchain types: an explicit `.role` (a `ChatMessage`'s free-form
 * role) wins, then the message type via `getType()`/`_getType()`/`.type`,
 * then "user" — a message is always attributed to somebody, never dropped. */
function roleOf(message: unknown): string {
  const m = message as {
    role?: unknown;
    type?: unknown;
    getType?: () => string;
    _getType?: () => string;
  };
  if (typeof m?.role === "string" && m.role) return m.role;
  let kind: unknown = undefined;
  if (typeof m?.getType === "function") kind = m.getType();
  else if (typeof m?._getType === "function") kind = m._getType();
  else kind = m?.type;
  return (typeof kind === "string" && ROLES[kind]) || "user";
}

/**
 * The provider-shaped tool calls on an assistant message, in OpenAI's wire
 * form. The NORMALIZED `.tool_calls` comes first, rebuilt with
 * `JSON.stringify(args)` — because that is exactly what LangChain's own
 * converter does when it sends the message back, so the `arguments` string
 * matches the wire character for character (verified against captured request
 * bodies in `test/langchain.test.ts`). Using `additional_kwargs.tool_calls`
 * verbatim looks more faithful and is not: it holds the PROVIDER's original
 * JSON text, whose whitespace LangChain does not preserve when it
 * re-serializes. It stays as the fallback for an integration that keeps only
 * the raw form. Mirrors Python `_tool_calls_of`.
 *
 * THE ONE PLACE CROSS-SDK IDENTITY STOPS. `JSON.stringify` writes
 * `{"city":"Dubai"}` and Python's `json.dumps` writes `{"city": "Dubai"}`, so
 * this block — and only this block — hashes differently in the two SDKs. That
 * is inherent, not a bug to normalize away: each handler reproduces its own
 * framework's real request, and two DIRECT captures of those same requests
 * diverge by exactly the same bytes with no ctxdiff involved. Emitting a
 * common form here would trade a guarantee verified against the wire for one
 * that is not. The divergence is pinned by both suites and stated in both
 * READMEs.
 */
function toolCallsOf(message: unknown): Dict[] {
  const m = message as { additional_kwargs?: unknown; tool_calls?: unknown };
  const calls = m?.tool_calls;
  if (Array.isArray(calls) && calls.length > 0) {
    const rebuilt = calls
      .filter((c): c is Dict => asDict(c) !== null)
      .map((call) => ({
        id: call["id"],
        type: "function",
        function: {
          name: call["name"],
          arguments: JSON.stringify(call["args"] ?? {}),
        },
      }));
    if (rebuilt.length > 0) return rebuilt;
  }
  const raw = asDict(m?.additional_kwargs)?.["tool_calls"];
  if (Array.isArray(raw) && raw.length > 0) return raw as Dict[];
  return [];
}

/** Flatten a message's content down to plain text, for the places a wire
 * format takes ONE string: a system prompt, a tool result's text. NOT for a
 * message's content on the typed-part providers — see `wireParts`. Mirrors
 * Python `_text_of`. */
function textOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") return part;
        const text = asDict(part)?.["text"];
        return typeof text === "string" ? text : "";
      })
      .join("");
  }
  return content == null ? "" : String(content);
}

/**
 * A message's content as ONE PART PER ENTRY, for the wire formats whose
 * content is a list of typed parts (Gemini's `parts`).
 *
 * Text entries become that format's text part, `{text}` — the shape
 * `GeminiAdapter` reads, and the shape @langchain/google-genai really puts on
 * the wire. EVERY OTHER ENTRY IS PASSED THROUGH UNCHANGED so the adapter's own
 * per-part handling sees it, above all `imageRawBlock`, which turns an image
 * part into a block whose identity is the picture's BYTES. That is what makes
 * the same screenshot ONE block whether it arrived as LangChain's `image_url`
 * data URI here or as `inlineData` bytes in a direct capture.
 *
 * The alternative — flattening the list to one string through `textOf` — is
 * what this branch used to do, and it lost data twice over: every non-text
 * part was dropped outright (a vision turn recorded 1 block instead of 2, with
 * the image's whole token cost gone from `ctxdiff tokens`), and several text
 * parts collapsed into one block where the real wire keeps them separate.
 *
 * Empty text contributes NO part rather than an empty one: LangChain does not
 * send an empty text part, so emitting one would itself be a divergence.
 * Mirrors Python `_wire_parts`.
 */
function wireParts(content: unknown): unknown[] {
  if (content == null) return [];
  if (typeof content === "string") return content ? [{ text: content }] : [];
  if (!Array.isArray(content)) {
    const text = String(content);
    return text ? [{ text }] : [];
  }
  const parts: unknown[] = [];
  for (const entry of content) {
    if (typeof entry === "string") {
      if (entry) parts.push({ text: entry });
      continue;
    }
    const text = asDict(entry)?.["text"];
    if (typeof text === "string") {
      if (text) parts.push({ text });
      continue;
    }
    parts.push(entry);
  }
  return parts;
}

/**
 * LangChain messages -> OpenAI Chat Completions `messages`. Verified against
 * the real request bodies @langchain/openai sends: a plain message is
 * `{role, content}`; an assistant message carrying tool calls sends
 * `content: null` alongside `tool_calls`; a tool message becomes
 * `{role: "tool", content, tool_call_id}`. Array content (multimodal parts)
 * passes through untouched — the adapter turns each part into its own block.
 */
function openaiMessages(messages: unknown[]): Dict[] {
  return messages.map((message) => {
    const m = message as { content?: unknown; tool_call_id?: unknown };
    const role = roleOf(message);
    const entry: Dict = { role, content: m?.content ?? null };
    if (role === "assistant") {
      const toolCalls = toolCallsOf(message);
      if (toolCalls.length > 0) {
        entry["tool_calls"] = toolCalls;
        if (!m?.content) entry["content"] = null;
      }
    }
    if (role === "tool") entry["tool_call_id"] = m?.tool_call_id ?? null;
    return entry;
  });
}

/**
 * LangChain messages -> Anthropic Messages `system` + `messages`. Anthropic
 * takes the system prompt OUT of the message list, as a top-level field, which
 * is what `AnthropicAdapter` expects — a single string system message becomes
 * the bare string form (identical blocks to a direct `messages.create({system})`
 * call), several become the list-of-text-blocks form. Tool calls and results
 * become Anthropic's own `tool_use`/`tool_result` content blocks when the
 * message doesn't already carry them verbatim.
 */
function anthropicWire(messages: unknown[]): Dict {
  const systems: string[] = [];
  const out: Dict[] = [];
  for (const message of messages) {
    const m = message as { content?: unknown; tool_calls?: unknown; tool_call_id?: unknown };
    const role = roleOf(message);
    if (role === "system") {
      systems.push(textOf(m?.content));
      continue;
    }
    if (role === "tool") {
      out.push({
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: m?.tool_call_id ?? null, content: m?.content },
        ],
      });
      continue;
    }
    const calls = role === "assistant" && Array.isArray(m?.tool_calls) ? m.tool_calls : null;
    if (calls && calls.length > 0 && !Array.isArray(m?.content)) {
      const blocks: Dict[] = [];
      if (m?.content) blocks.push({ type: "text", text: textOf(m.content) });
      for (const call of calls as Dict[]) {
        blocks.push({
          type: "tool_use",
          id: call["id"],
          name: call["name"],
          input: call["args"] ?? {},
        });
      }
      out.push({ role, content: blocks });
      continue;
    }
    out.push({ role, content: m?.content ?? null });
  }
  const wire: Dict = { messages: out };
  if (systems.length === 1) wire["system"] = systems[0];
  else if (systems.length > 1) {
    wire["system"] = systems.map((text) => ({ type: "text", text }));
  }
  return wire;
}

/**
 * LangChain messages -> @google/genai `contents` + `config.systemInstruction`.
 * Gemini names the assistant role "model" and wraps every message in a `parts`
 * array, which is what `GeminiAdapter` reads; a tool call becomes a
 * `functionCall` part and a tool result a `functionResponse` part.
 *
 * Message content becomes one part per content entry (`wireParts`) — the real
 * wire carries `[{text}, {text}, {inlineData}]` for a two-text-plus-image
 * turn, so the recorded blocks must too.
 */
function geminiWire(messages: unknown[]): Dict {
  const system: string[] = [];
  const contents: Dict[] = [];
  for (const message of messages) {
    const m = message as { content?: unknown; tool_calls?: unknown; name?: unknown };
    const role = roleOf(message);
    if (role === "system") {
      system.push(textOf(m?.content));
      continue;
    }
    if (role === "tool") {
      contents.push({
        role: "user",
        parts: [
          {
            functionResponse: {
              name: m?.name ?? null,
              response: { result: textOf(m?.content) },
            },
          },
        ],
      });
      continue;
    }
    const parts: unknown[] = wireParts(m?.content);
    if (role === "assistant" && Array.isArray(m?.tool_calls)) {
      for (const call of m.tool_calls as Dict[]) {
        parts.push({ functionCall: { name: call["name"], args: call["args"] ?? {} } });
      }
    }
    contents.push({ role: role === "assistant" ? "model" : "user", parts });
  }
  const wire: Dict = { contents };
  if (system.length > 0) wire["config"] = { systemInstruction: system.join("\n") };
  return wire;
}

/**
 * LangChain messages -> Bedrock Converse `system` + `messages`.
 *
 * Converse takes system prompts as a list of `{text}` blocks and every
 * message's content as a list of typed blocks, with tool calls as `toolUse`
 * and results as `toolResult` inside a USER-role message (the Converse API has
 * no tool role) — the shapes `BedrockAdapter` reads.
 *
 * Message content becomes one block per content entry (`wireParts`) — the real
 * wire carries `[{text: "a"}, {text: "b"}, {image: {...}}]` for a
 * two-text-plus-image turn, so the recorded blocks must too. Mirrors Python
 * `_bedrock_wire`.
 */
function bedrockWire(messages: unknown[]): Dict {
  const system: Dict[] = [];
  const out: Dict[] = [];
  for (const message of messages) {
    const m = message as { content?: unknown; tool_calls?: unknown; tool_call_id?: unknown };
    const role = roleOf(message);
    if (role === "system") {
      system.push({ text: textOf(m?.content) });
      continue;
    }
    if (role === "tool") {
      out.push({
        role: "user",
        content: [
          {
            toolResult: {
              toolUseId: m?.tool_call_id ?? null,
              content: [{ text: textOf(m?.content) }],
            },
          },
        ],
      });
      continue;
    }
    const blocks: unknown[] = wireParts(m?.content);
    if (role === "assistant" && Array.isArray(m?.tool_calls)) {
      for (const call of m.tool_calls as Dict[]) {
        blocks.push({
          toolUse: {
            toolUseId: call["id"],
            name: call["name"],
            input: call["args"] ?? {},
          },
        });
      }
    }
    out.push({ role, content: blocks });
  }
  const wire: Dict = { messages: out };
  if (system.length > 0) wire["system"] = system;
  return wire;
}

/**
 * Rebuild the request object LangChain is about to send, in `provider`'s own
 * wire shape — the single function the whole hash-identity promise rests on.
 *
 * Content comes from the messages, normalized per provider. TOOL SCHEMAS and
 * sampling params come from `invocationParams`, which LangChain has ALREADY
 * converted into the target provider's format (ChatOpenAI hands over
 * OpenAI-shaped tool objects, ChatAnthropic Anthropic-shaped ones), so they are
 * carried across verbatim into whichever key that provider's request uses and
 * are therefore byte-identical to what a direct call would have recorded.
 * Everything else in `invocationParams` is merged in as ordinary request
 * kwargs — exactly where each adapter's `extractParams` picks params up — and
 * the model id is written under the key that provider uses so the session's
 * model roll-up still works. Mirrors Python `to_wire`.
 */
export function toWire(
  provider: string,
  messages: unknown[],
  invocationParams: unknown,
  modelName?: unknown,
): Dict {
  const params: Dict = { ...(asDict(invocationParams) ?? {}) };
  const tools = params["tools"];
  delete params["tools"];
  delete params["functions"]; // legacy alias; schemas travel via `tools`

  let wire: Dict;
  if (provider === "anthropic") {
    wire = anthropicWire(messages);
    if (tools) wire["tools"] = tools;
  } else if (provider === "gemini") {
    wire = geminiWire(messages);
    if (tools) {
      const config = asDict(wire["config"]) ?? {};
      config["tools"] = tools;
      wire["config"] = config;
    }
  } else if (provider === "bedrock") {
    wire = bedrockWire(messages);
    if (tools) wire["toolConfig"] = { tools };
  } else {
    wire = { messages: openaiMessages(messages) };
    if (tools) wire["tools"] = tools;
  }

  const model =
    params["model_id"] ?? params["modelId"] ?? params["model"] ?? params["model_name"] ?? modelName;
  for (const key of ["model_id", "modelId", "model", "model_name"]) delete params[key];
  for (const [key, value] of Object.entries(params)) {
    // Never let a param overwrite content the messages already produced.
    if (!(key in wire)) wire[key] = value;
  }
  if (model !== undefined && model !== null) wire[MODEL_KEYS[provider] ?? "model"] = model;
  return wire;
}

/**
 * Pull `{input_tokens, output_tokens, total_tokens}` out of an `LLMResult`,
 * whichever place this integration put them. `usage_metadata` on the generated
 * message is @langchain/core's PROVIDER-INDEPENDENT shape and is populated for
 * streaming runs too (where `llmOutput` is empty), so it is tried first;
 * `llmOutput.tokenUsage`/`usage` is the fallback. Defensive throughout — an
 * unanticipated result shape must yield no usage, never an exception, since
 * this runs inside a host's live agent. Mirrors Python `_usage_metadata`.
 */
function usageMetadata(result: unknown): Dict | null {
  try {
    const generations = (result as { generations?: unknown })?.generations;
    if (Array.isArray(generations)) {
      for (const batch of generations) {
        for (const generation of Array.isArray(batch) ? batch : []) {
          const usage = asDict((generation as { message?: { usage_metadata?: unknown } })?.message
            ?.usage_metadata);
          if (usage && Object.keys(usage).length > 0) return usage;
        }
      }
    }
    const output = asDict((result as { llmOutput?: unknown })?.llmOutput);
    for (const key of ["tokenUsage", "token_usage", "usage"]) {
      const usage = asDict(output?.[key]);
      if (usage && Object.keys(usage).length > 0) {
        return {
          input_tokens: usage["promptTokens"] ?? usage["prompt_tokens"] ?? usage["input_tokens"],
          output_tokens:
            usage["completionTokens"] ?? usage["completion_tokens"] ?? usage["output_tokens"],
          total_tokens: usage["totalTokens"] ?? usage["total_tokens"],
        };
      }
    }
  } catch {
    // usage is best-effort, never a failure
  }
  return null;
}

/**
 * Map LangChain's normalized token counts onto the key names `provider`'s own
 * `extractUsage` returns, so a LangChain-captured call stores the same usage
 * object a directly-captured one would. An empty result means no counts were
 * reported at all, and the call is recorded honestly with `usage: null` rather
 * than fabricated zeros. Mirrors Python `usage_state`.
 */
export function usageState(provider: string, result: unknown): Dict {
  const counts = usageMetadata(result);
  if (!counts) return {};
  const [inputKey, outputKey, totalKey] = USAGE_KEYS[provider] ?? USAGE_KEYS["openai"]!;
  const state: Dict = {
    [inputKey]: counts["input_tokens"],
    [outputKey]: counts["output_tokens"],
  };
  if (totalKey !== null) state[totalKey] = counts["total_tokens"];
  return state;
}

/** One in-flight LangChain run: everything the end callback needs that only
 * the start callback knew. Kept per run id because LangChain interleaves runs
 * freely (a LangGraph fan-out, a `RunnableParallel`), so "the last request
 * seen" is never a safe assumption. */
interface Pending {
  start: number;
  kwargs: Dict;
  provider: string;
  recorder: unknown;
  tags: [string, string][];
  step: string | null;
}

/** How many concurrently-open runs are tracked before the oldest are dropped.
 * A run whose end callback never fires (a cancelled task, a handler detached
 * mid-flight) would otherwise pin its request payload forever; a debugging tool
 * must not be able to leak a host's memory. Far above any real fan-out. */
const MAX_PENDING = 2048;

/** The minimum of `Tracer` this handler uses — declared structurally so this
 * module doesn't import trace.ts (which imports this one). */
interface HandlerTracer {
  recorderFor(provider: string): unknown;
  consumeContext(): { tags: [string, string][]; step: string | null };
  onCreate(args: {
    kwargs: Dict;
    response: unknown;
    latencyMs: number | null;
    error: string | null;
    recorder: never;
    agent: string | null;
    tags: [string, string][];
    step: string | null;
  }): void;
}

/** The handler object `tracer.langchainHandler()` returns: LangChain's
 * `CallbackHandlerMethods` shape, which `callbacks: [...]` accepts directly. */
export interface CtxdiffCallbackHandler {
  name: string;
  handleChatModelStart(
    llm: unknown,
    messages: unknown[][],
    runId: string,
    parentRunId?: string,
    extraParams?: Record<string, unknown>,
    tags?: string[],
    metadata?: Record<string, unknown>,
    runName?: string,
  ): void;
  handleLLMStart(
    llm: unknown,
    prompts: string[],
    runId: string,
    parentRunId?: string,
    extraParams?: Record<string, unknown>,
    tags?: string[],
    metadata?: Record<string, unknown>,
    runName?: string,
  ): void;
  handleLLMEnd(output: unknown, runId: string): void;
  handleLLMError(err: unknown, runId: string): void;
}

/** Best-effort error type name, mirroring Python's `type(exc).__name__`. */
function errName(exc: unknown): string {
  if (exc instanceof Error) return exc.constructor?.name || exc.name || "Error";
  if (exc && typeof exc === "object") {
    return (exc as { constructor?: { name?: string } }).constructor?.name || "Error";
  }
  return "Error";
}

/**
 * Build the callback handler. Used by `Tracer.langchainHandler()` — see that
 * method for the user-facing docs.
 *
 * EVERY callback is fail-open. LangChain does swallow handler errors by
 * default, but that is not a guarantee this code is entitled to lean on
 * (`raiseError` exists, and other frameworks re-dispatch these callbacks), so
 * each method catches its own: a broken tracer must never break the agent it
 * is watching.
 */
export function buildHandler(
  tracer: HandlerTracer,
  agent: string | null,
): CtxdiffCallbackHandler {
  const pending = new Map<string, Pending>();
  let warnedNoAdapter = false;

  /** Shared body of the two start callbacks: resolve provider -> recorder ->
   * wire request, and park it. The latency clock starts here, so a recorded
   * LangChain call measures the same span the direct path does. Attribution
   * (`tag()`/`step()`) is snapshotted HERE too, on the tick that made the
   * call — the end callback may run in a different async context, where
   * re-reading it could pick up a sibling branch's label. */
  const start = (
    serialized: unknown,
    messages: unknown[],
    runId: string,
    extraParams: unknown,
    metadata: unknown,
  ): void => {
    const provider = providerFor(serialized, metadata);
    const recorder = tracer.recorderFor(provider);
    if (recorder == null) {
      // A provider this SDK has no adapter for. All four `LS_PROVIDERS` map to
      // one now (openai / anthropic / gemini / bedrock), so in practice this is
      // reached only when the run's STORE could not be opened. Recording
      // through some other provider's adapter would store blocks that never
      // went on the wire, so the call is skipped — once, loudly, then silently.
      if (!warnedNoAdapter) {
        warnedNoAdapter = true;
        console.warn(
          `ctxdiff: no adapter for provider '${provider}'; LangChain calls to it ` +
            "will not be recorded by the JS SDK",
        );
      }
      return;
    }
    const params = asDict(extraParams)?.["invocation_params"];
    const modelName = asDict(metadata)?.["ls_model_name"];
    const kwargs = toWire(provider, messages, params, modelName);
    const { tags, step } = tracer.consumeContext();
    if (pending.size >= MAX_PENDING) {
      // Map iteration order is insertion order, so the first key is the oldest.
      const oldest = pending.keys().next();
      if (!oldest.done) pending.delete(oldest.value);
    }
    pending.set(runId, {
      start: performance.now(),
      kwargs,
      provider,
      recorder,
      tags,
      step,
    });
  };

  /** Claim an in-flight run, exactly once — a duplicate end callback (or an
   * error following an end) then finds nothing and records nothing, the same
   * "record exactly once" guarantee the stream proxy gets from `finalized`. */
  const take = (runId: string): Pending | undefined => {
    const found = pending.get(runId);
    if (found !== undefined) pending.delete(runId);
    return found;
  };

  /** Hand one finished call to the tracer. Nothing from here on is
   * LangChain-specific: the call is indistinguishable from one captured by
   * wrapping the SDK directly. */
  const record = (run: Pending, response: unknown, error: string | null): void => {
    tracer.onCreate({
      kwargs: run.kwargs,
      response,
      latencyMs: Math.round(performance.now() - run.start),
      error,
      recorder: run.recorder as never,
      agent,
      tags: run.tags,
      step: run.step,
    });
  };

  return {
    name: "ctxdiff",

    /** A chat model is about to be called. `messages` is an array of message
     * ARRAYS (one per prompt in a batch); chat models are effectively always
     * called with one, and only the first is recorded — a genuinely batched
     * generate reports ONE run id for the whole batch, so recording several
     * calls against it would invent turns that never happened. */
    handleChatModelStart(llm, messages, runId, _parentRunId, extraParams, _tags, metadata) {
      try {
        start(llm, messages?.[0] ?? [], runId, extraParams, metadata);
      } catch (err) {
        console.warn("ctxdiff: langchain handleChatModelStart failed", err);
      }
    },

    /** The non-chat (completion-style) entry point. Each prompt string is
     * treated as a user message so the same normalization, adapter and block
     * shapes apply. */
    handleLLMStart(llm, prompts, runId, _parentRunId, extraParams, _tags, metadata) {
      try {
        const prompt = prompts?.[0] ?? "";
        start(llm, [{ type: "human", content: prompt }], runId, extraParams, metadata);
      } catch (err) {
        console.warn("ctxdiff: langchain handleLLMStart failed", err);
      }
    },

    /** The call finished: pair the result with the parked request, map its
     * token counts into the provider's own usage shape, and record — through
     * the tracer's ordinary fail-open `onCreate`, so a LangChain-captured call
     * gets the same seq assignment, snapshotting and writer path as a directly
     * captured one. */
    handleLLMEnd(output, runId) {
      try {
        const run = take(runId);
        if (run === undefined) return;
        const state = usageState(run.provider, output);
        const hasUsage = Object.keys(state).length > 0;
        // The SAME synthetic-response shape the streaming path builds, so both
        // routes reach `extractUsage` through identical objects: `usage` for
        // OpenAI/Anthropic, `usageMetadata` for Gemini, one object behind both.
        record(run, hasUsage ? { usage: state, usageMetadata: state } : null, null);
      } catch (err) {
        console.warn("ctxdiff: langchain handleLLMEnd failed", err);
      }
    },

    /** The call failed: record it as a FAILED call carrying the error's type
     * name, exactly as the direct path records a provider error, so a failed
     * turn stays visible instead of vanishing from the trace. */
    handleLLMError(err, runId) {
      try {
        const run = take(runId);
        if (run === undefined) return;
        record(run, null, errName(err));
      } catch (caught) {
        console.warn("ctxdiff: langchain handleLLMError failed", caught);
      }
    },
  };
}
