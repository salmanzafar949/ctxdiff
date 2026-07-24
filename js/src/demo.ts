/**
 * Builds a sample, self-describing multi-agent `.ctrace` — the payload behind
 * `ctxdiff demo`. Zero-friction: no API keys, no network, no provider SDK
 * installed, and byte-for-byte the same demo content on every run. A faithful
 * port of Python `ctxdiff.demo`: same scenario, same fixed strings, same call
 * structure, so the JS demo trace matches the Python one block-for-block.
 *
 * It drives the REAL public capture API (`init`/`wrap`/`tag`) through two tiny
 * fake clients whose shape satisfies `wrap`'s provider detection and the
 * OpenAI/Anthropic adapters' usage extraction — so the demo exercises the same
 * recorder/adapter/store code a real integration would, not a shortcut. No I/O
 * is possible: each fake `create` just returns the next canned usage object.
 *
 * The scenario (a two-agent research pipeline on prompt-cache pricing) is built
 * to light up every dashboard panel: block dedup (a stable researcher system
 * prompt), a dynamic timestamp block that breaks the researcher's cache prefix
 * every turn (fix hint), an unused `delete_index` tool schema (bloat), a
 * `tag("rag", …)` on the writer's synthesis, and growing per-agent history.
 */
import { init } from "./trace.js";
import { stableStringify } from "./models.js";

// --- fake provider clients ---------------------------------------------------
// No provider SDK is imported: `wrap` detects a provider by duck-typing the
// client's resource shape, and the adapters' usage extraction only reads a
// response's `.usage`. Each fake below is the minimum shape that satisfies both.

type Usage = [number, number];

/** An OpenAI-shaped fake: `chat.completions.create` returns the next scripted
 * usage; a truthy `responses` makes provider detection resolve it to `openai`. */
function fakeOpenAIClient(usages: Usage[]): object {
  let i = 0;
  return {
    chat: {
      completions: {
        create: (_kwargs: unknown) => {
          const [promptTokens, completionTokens] = usages[i++];
          return {
            usage: {
              prompt_tokens: promptTokens,
              completion_tokens: completionTokens,
              total_tokens: promptTokens + completionTokens,
            },
          };
        },
      },
    },
    responses: {},
  };
}

/** An Anthropic-shaped fake: `messages.create` returns the next scripted usage;
 * the absence of `chat`/`responses` resolves detection to `anthropic`. */
function fakeAnthropicClient(usages: Usage[]): object {
  let i = 0;
  return {
    messages: {
      create: (_kwargs: unknown) => {
        const [inputTokens, outputTokens] = usages[i++];
        return { usage: { input_tokens: inputTokens, output_tokens: outputTokens } };
      },
    },
  };
}

// --- scenario content ---------------------------------------------------------
// Every string below is a fixed literal (no clock, no randomness), copied
// verbatim from the Python demo so both SDKs produce identical demo content.

const TS1 = "2026-07-24T09:58:03Z";
const TS2 = "2026-07-24T09:59:47Z";
const TS3 = "2026-07-24T10:02:15Z";

const RESEARCHER_SYSTEM =
  "You are a research analyst agent in a two-agent pipeline. Your job is " +
  "to find and verify evidence from public vendor documentation, then hand " +
  "off precise, sourced findings to a writer agent. Always name your " +
  "source (vendor + doc section). Do not speculate beyond what a source " +
  "states.";

const WRITER_SYSTEM =
  "You are a technical writer agent. Turn the research findings you're " +
  "given into a clear, well-cited summary for engineering leadership. Keep " +
  "it tight and never introduce a claim the research didn't supply.";

const RESEARCH_QUESTION =
  "Summarize the most recent evidence on prompt-cache pricing across " +
  "OpenAI, Anthropic, and Gemini — specifically how cache writes vs cache " +
  "reads are billed, and whether the discount changes with prefix length.";

const SEARCH_WEB_SCHEMA = {
  type: "function",
  function: {
    name: "search_web",
    description: "Search public documentation for a query and return top results.",
    parameters: {
      type: "object",
      properties: { query: { type: "string" } },
      required: ["query"],
    },
  },
};

const DELETE_INDEX_SCHEMA = {
  type: "function",
  function: {
    name: "delete_index",
    description: "Permanently delete a named search index. Destructive; requires confirmation.",
    parameters: {
      type: "object",
      properties: { index_name: { type: "string" } },
      required: ["index_name"],
    },
  },
};

const TOOLS = [SEARCH_WEB_SCHEMA, DELETE_INDEX_SCHEMA];

// The model's own tool invocation, echoed back on every later turn. `arguments`
// is stable-JSON (sorted keys, Python-style separators) to match the Python
// demo's `json.dumps(..., sort_keys=True)` byte-for-byte, so the block hash is
// identical across SDKs.
const TOOL_CALL_SEARCH = {
  id: "call_search_1",
  type: "function",
  function: {
    name: "search_web",
    arguments: stableStringify({
      query: "prompt cache pricing OpenAI Anthropic Gemini cache write vs read 2026",
    }),
  },
};

const SEARCH_RESULTS =
  "Search results:\n" +
  "1. OpenAI docs, 'Prompt Caching' (2026-06): cached input tokens billed " +
  "at 50% of standard input price; a cache hit requires an identical " +
  "prefix of at least 1024 tokens.\n" +
  "2. Anthropic docs, 'Prompt caching' (2026-05): cache writes cost ~25% " +
  "MORE than a normal input token (a one-time write premium); cache reads " +
  "cost roughly 10% of a normal input token; entries expire after 5 " +
  "minutes of inactivity.\n" +
  "3. Google Gemini docs, 'Context caching' (2026-06): a flat per-hour " +
  "storage fee plus a reduced per-token rate on cache hits; the minimum " +
  "cacheable content size varies by model.";

const ANALYSIS_1 =
  "Findings so far: all three vendors discount cache HITS relative to a " +
  "fresh input token, but the shapes differ — OpenAI's discount is a flat " +
  "~50% with a 1024-token minimum prefix; Anthropic charges a write " +
  "premium (~+25%) but drops reads to ~10% of normal price; Gemini bills a " +
  "separate hourly storage fee on top of a reduced hit rate. Sourced from " +
  "each vendor's own caching docs. Handing off to the writer agent for a " +
  "leadership-ready summary.";

const FOLLOWUP_QUESTION =
  "Can you confirm Anthropic's cache write premium applies per write, not " +
  "per token stored — and cite the doc section?";

const RAG_FINDINGS =
  "OpenAI: ~50% discount on cache hits, 1024-token minimum prefix, no " +
  "write premium. Anthropic: ~25% write premium, cache reads ~10% of " +
  "normal input price, 5-minute TTL. Gemini: flat hourly storage fee plus " +
  "a reduced per-token hit rate.";

const SYNTHESIS_PROMPT =
  "Turn the research findings below into a concise, engineering-" +
  "leadership-ready summary (3-4 sentences, no jargon) comparing how " +
  "OpenAI, Anthropic, and Gemini price prompt-cache hits and writes.\n\n" +
  `Findings:\n${RAG_FINDINGS}`;

const DRAFT_V1 =
  "Prompt caching now meaningfully changes unit economics for high-" +
  "repetition agent workloads. OpenAI gives a flat ~50% discount on cache " +
  "hits once a request's shared prefix passes 1024 tokens, with no extra " +
  "charge to populate the cache. Anthropic instead charges a one-time " +
  "~25% premium to write into cache but drops the price of a cache hit to " +
  "roughly a tenth of normal input, and evicts unused entries after five " +
  "minutes. Gemini's model is the most different: a flat hourly storage " +
  "fee on top of a reduced per-token rate on hits, which favors long-" +
  "lived, high-traffic caches over bursty ones.";

const NEW_FINDING_FOR_WRITER =
  "Confirmed with the research agent: Anthropic's ~25% cache-write " +
  "premium is a one-time, per-request charge (not per-token storage) — " +
  "see 'Prompt caching' docs, Pricing section.";

const REVISION_REQUEST =
  "Please tighten the summary to under 120 words and add one caveat " +
  "sentence noting that Anthropic's write premium is a one-time, per-" +
  "request charge, not ongoing storage — per the research agent's " +
  "confirmation above.";

const DRAFT_V2 =
  "Prompt caching now materially changes the economics of high-repetition " +
  "agent workloads. OpenAI discounts cache hits ~50% once a shared prefix " +
  "exceeds 1024 tokens, at no extra write cost. Anthropic charges a one-" +
  "time ~25% premium to populate the cache but discounts hits to roughly " +
  "a tenth of normal input price, with a five-minute idle TTL. Gemini " +
  "instead bills a flat hourly storage fee plus a reduced per-token hit " +
  "rate, favoring long-lived caches. Caveat: Anthropic's write premium is " +
  "a one-time, per-request charge, not ongoing storage.";

const FINALIZE_REQUEST =
  "This reads well — finalize it as the closing paragraph of the report; " +
  "flag anything that still needs a citation.";

// Usage figures, in call order per agent (fixed constants).
const RESEARCHER_USAGES: Usage[] = [
  [850, 60],
  [1400, 180],
  [1900, 150],
];
const WRITER_USAGES: Usage[] = [
  [520, 210],
  [760, 240],
  [980, 190],
];

/**
 * Build a realistic, deterministic multi-agent `.ctrace` at `path` via the real
 * public capture API and return `path`. No network, no provider SDK. Six calls
 * interleaved across two agents (two hand-offs), each request built from the
 * fixed strings above. Mirrors Python `build_demo_trace`.
 */
export function buildDemoTrace(path: string): string {
  const tracer = init("research-pipeline-demo", { path });
  const researcher = tracer.wrap(fakeOpenAIClient(RESEARCHER_USAGES), { agent: "researcher" }) as {
    chat: { completions: { create: (k: unknown) => unknown } };
  };
  const writer = tracer.wrap(fakeAnthropicClient(WRITER_USAGES), { agent: "writer" }) as {
    messages: { create: (k: unknown) => unknown };
  };

  // turn 1 — researcher opens the investigation and calls search_web.
  researcher.chat.completions.create({
    model: "gpt-4o",
    tools: TOOLS,
    messages: [
      { role: "system", content: `Current session time: ${TS1}` },
      { role: "system", content: RESEARCHER_SYSTEM },
      { role: "user", content: RESEARCH_QUESTION },
    ],
  });

  // turn 2 — the tool result comes back; TS1 → TS2 is the first cache break.
  researcher.chat.completions.create({
    model: "gpt-4o",
    tools: TOOLS,
    messages: [
      { role: "system", content: `Current session time: ${TS2}` },
      { role: "system", content: RESEARCHER_SYSTEM },
      { role: "user", content: RESEARCH_QUESTION },
      { role: "assistant", content: null, tool_calls: [TOOL_CALL_SEARCH] },
      { role: "tool", tool_call_id: "call_search_1", content: SEARCH_RESULTS },
    ],
  });

  // turn 3 — hand-off to the writer: a rag-tagged synthesis request.
  tracer.tag("rag", [RAG_FINDINGS]);
  writer.messages.create({
    model: "claude-3-5-sonnet-20241022",
    system: WRITER_SYSTEM,
    messages: [{ role: "user", content: SYNTHESIS_PROMPT }],
  });

  // turn 4 — hand-off back to the researcher. TS2 → TS3: second cache break.
  researcher.chat.completions.create({
    model: "gpt-4o",
    tools: TOOLS,
    messages: [
      { role: "system", content: `Current session time: ${TS3}` },
      { role: "system", content: RESEARCHER_SYSTEM },
      { role: "user", content: RESEARCH_QUESTION },
      { role: "assistant", content: null, tool_calls: [TOOL_CALL_SEARCH] },
      { role: "tool", tool_call_id: "call_search_1", content: SEARCH_RESULTS },
      { role: "assistant", content: ANALYSIS_1 },
      { role: "user", content: FOLLOWUP_QUESTION },
    ],
  });

  // turn 5 — writer revises (pure append after its stable prefix; no break).
  writer.messages.create({
    model: "claude-3-5-sonnet-20241022",
    system: WRITER_SYSTEM,
    messages: [
      { role: "user", content: SYNTHESIS_PROMPT },
      { role: "assistant", content: DRAFT_V1 },
      { role: "user", content: `${NEW_FINDING_FOR_WRITER}\n\n${REVISION_REQUEST}` },
    ],
  });

  // turn 6 — writer finalizes (same agent, pure append again; still stable).
  writer.messages.create({
    model: "claude-3-5-sonnet-20241022",
    system: WRITER_SYSTEM,
    messages: [
      { role: "user", content: SYNTHESIS_PROMPT },
      { role: "assistant", content: DRAFT_V1 },
      { role: "user", content: `${NEW_FINDING_FOR_WRITER}\n\n${REVISION_REQUEST}` },
      { role: "assistant", content: DRAFT_V2 },
      { role: "user", content: FINALIZE_REQUEST },
    ],
  });

  tracer.close();
  return path;
}
