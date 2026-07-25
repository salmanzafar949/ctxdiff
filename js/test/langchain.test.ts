/**
 * The LangChain CALLBACK HANDLER (JS), exercised against real @langchain/core,
 * @langchain/openai and @langchain/langgraph with HTTP stubbed by a custom
 * `fetch` — the idiomatic capture path for a LangChain/LangGraph app, and the
 * twin of Python's `tests/eval/test_langchain_handler.py` /
 * `tests/eval/test_langgraph.py`.
 *
 * The load-bearing assertions are the identity ones:
 *
 *  - HASH IDENTITY WITH DIRECT CAPTURE: the blocks the handler records must
 *    equal, hash for hash, what wrapping the provider SDK directly records for
 *    the same request — checked end to end, and against the exact JSON body
 *    LangChain put on the wire.
 *  - CROSS-SDK IDENTITY: the same prompt through the PYTHON handler must
 *    produce the same hashes, which is pinned here as literals that the Python
 *    suite asserts too (`test_cross_sdk_hashes_are_pinned` there) — with the
 *    single documented exception of a TOOL CALL, whose arguments LangChain
 *    re-serializes with each language's own JSON serializer. That divergence
 *    is pinned too, in both directions, so it cannot drift unnoticed.
 */
import { describe, it, expect, afterEach } from "vitest";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { rmSync } from "node:fs";
import OpenAI from "openai";
import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { AIMessage, HumanMessage, SystemMessage, ToolMessage } from "@langchain/core/messages";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatBedrockConverse } from "@langchain/aws";
import { BedrockRuntimeClient } from "@aws-sdk/client-bedrock-runtime";
import { Readable } from "node:stream";
import { END, START, StateGraph } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { z } from "zod";
import { init } from "../src/trace.js";
import { CTrace } from "../src/store/ctrace.js";
import { buildBlock } from "../src/capture/recorder.js";
import { OpenAIAdapter } from "../src/capture/openai.js";
import { GeminiAdapter } from "../src/capture/gemini.js";
import { BedrockAdapter } from "../src/capture/bedrock.js";
import { providerFor, toWire, usageState } from "../src/capture/langchain.js";

/** A 4×4 PNG, base64 — byte-identical to the one the Python suite's
 * multimodal tests use, so both SDKs are measuring the same picture. Embedded
 * as a literal rather than rebuilt: the point here is the part-per-entry
 * normalization, not PNG construction (which `images.test.ts` covers). */
const PNG_4x4_B64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAC0lEQVR4nGNgYAAAAAMAAbitOmMAAAAASUVORK5CYII=";
const PNG_4x4_URI = `data:image/png;base64,${PNG_4x4_B64}`;

/** The `arguments` string each SDK's LangChain puts on the wire when it
 * re-serializes the SAME logical tool call, and the content hash each SDK
 * therefore stores for the assistant tool-call block. `JSON.stringify` emits
 * no separators, Python's `json.dumps` emits `", "`/`": "` — so the two
 * diverge by design, each faithful to its own framework's real request. Both
 * literals are asserted by BOTH suites (see the Python
 * `test_cross_sdk_tool_call_hashes_are_pinned_as_known_divergent`). */
const JS_TOOL_ARGUMENTS = '{"city":"Dubai"}';
const JS_TOOL_CALL_HASH = "65b07374b968257020127bfbf113d7fad62b16abad9fc957d195a87674b4f8c1";
const PY_TOOL_CALL_HASH = "aed9a5ef806c38eb945801ff65af115866c127e282ce357e253ef0942f7a9cbe";

const created: string[] = [];
function tmpTrace(): string {
  const p = join(tmpdir(), `ctxdiff-lc-${randomUUID()}.ctrace`);
  created.push(p);
  return p;
}
afterEach(() => {
  for (const p of created.splice(0)) {
    try {
      rmSync(p, { force: true });
    } catch {
      /* best effort */
    }
  }
});

const CHAT_RESPONSE = {
  id: "chatcmpl-1",
  object: "chat.completion",
  created: 1,
  model: "gpt-4o",
  choices: [
    { index: 0, message: { role: "assistant", content: "Hello there!" }, finish_reason: "stop" },
  ],
  usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
};

const TOOL_CALL_RESPONSE = {
  id: "chatcmpl-1",
  object: "chat.completion",
  created: 1,
  model: "gpt-4o",
  choices: [
    {
      index: 0,
      finish_reason: "tool_calls",
      message: {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "call_1",
            type: "function",
            function: { name: "get_weather", arguments: '{"city": "Dubai"}' },
          },
        ],
      },
    },
  ],
  usage: { prompt_tokens: 20, completion_tokens: 8, total_tokens: 28 },
};

const FINAL_RESPONSE = {
  id: "chatcmpl-2",
  object: "chat.completion",
  created: 1,
  model: "gpt-4o",
  choices: [
    {
      index: 0,
      finish_reason: "stop",
      message: { role: "assistant", content: "It is sunny in Dubai." },
    },
  ],
  usage: { prompt_tokens: 30, completion_tokens: 5, total_tokens: 35 },
};

/** A `fetch` stand-in that records every request body and replays canned
 * responses in order — the JS equivalent of the Python suite's respx stub, and
 * the thing that lets these tests check the handler against what LangChain
 * ACTUALLY sent rather than against another copy of ctxdiff's own opinion. */
function stubFetch(responses: unknown[], sent: unknown[]) {
  let i = 0;
  return async (_url: unknown, init: { body?: string }): Promise<Response> => {
    sent.push(JSON.parse(init?.body ?? "{}"));
    const body = responses[Math.min(i++, responses.length - 1)];
    return new Response(JSON.stringify(body), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  };
}

/** The content hashes a DIRECT capture of `wire` would store: the real OpenAI
 * adapter's blocks through the real `buildBlock`. */
function openaiHashes(wire: Record<string, unknown>): string[] {
  return new OpenAIAdapter()
    .extractBlocks(wire)
    .map((rb) => buildBlock(rb, "openai").contentHash);
}

/** The hashes a direct capture of the Gemini REST body would store. One
 * translation is needed and only one: `generateContent` takes `contents` plus
 * a `config` bag, while the REST body puts the system instruction at the top
 * level as `systemInstruction.parts[].text`. The CONTENT itself — the part
 * list these tests exist to check — is passed through verbatim. */
function geminiHashes(body: Record<string, unknown>): string[] {
  const system = (body["systemInstruction"] ?? {}) as { parts?: { text?: string }[] };
  const texts = (system.parts ?? []).map((p) => p.text ?? "");
  const kwargs: Record<string, unknown> = { contents: body["contents"] };
  if (texts.length > 0) kwargs["config"] = { systemInstruction: texts.join("\n") };
  return new GeminiAdapter()
    .extractBlocks(kwargs)
    .map((rb) => buildBlock(rb, "gemini").contentHash);
}

/** The hashes a direct capture of a Bedrock Converse request body would store.
 * The wire body IS the adapter's input for Converse (unlike Gemini, which needs
 * a REST-body translation), so no massaging is involved: what @langchain/aws
 * sent is handed straight to the same adapter `tracer.wrap(client)` uses. */
function bedrockHashes(body: Record<string, unknown>): string[] {
  return new BedrockAdapter()
    .extractBlocks(body)
    .map((rb) => buildBlock(rb, "bedrock").contentHash);
}

/**
 * A real `BedrockRuntimeClient` whose transport is stubbed, for injecting into
 * `ChatBedrockConverse({ client })` — the integration's own supported seam
 * (@langchain/aws takes no custom fetch, and the AWS SDK does not use fetch on
 * Node anyway). Every request body it sees is pushed into `sent`.
 */
function bedrockStubClient(sent: Record<string, unknown>[]): BedrockRuntimeClient {
  return new BedrockRuntimeClient({
    region: "us-east-1",
    credentials: { accessKeyId: "x", secretAccessKey: "y" },
    requestHandler: {
      handle: async (request: { body?: unknown }) => {
        sent.push(JSON.parse(String(request.body ?? "{}")));
        return {
          response: {
            statusCode: 200,
            reason: "OK",
            headers: { "content-type": "application/json" },
            body: Readable.from([
              Buffer.from(
                JSON.stringify({
                  output: { message: { role: "assistant", content: [{ text: "42°C" }] } },
                  stopReason: "end_turn",
                  usage: { inputTokens: 12, outputTokens: 6, totalTokens: 18 },
                }),
              ),
            ]),
          },
        };
      },
      updateHttpClientConfig(): void {},
      httpHandlerConfigs: () => ({}),
    } as never,
  });
}

/** A canned Gemini `generateContent` response body, the same shape the Python
 * eval suite's `canned_gemini_response` builds. */
const GEMINI_RESPONSE = {
  candidates: [{ content: { parts: [{ text: "Hello there!" }], role: "model" }, finishReason: "STOP" }],
  usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5, totalTokenCount: 15 },
  modelVersion: "gemini-2.0-flash",
};

describe("langchain callback handler", () => {
  it("captures a chat call with usage, params and ordered blocks", async () => {
    const path = tmpTrace();
    const tracer = init("p", { path });
    const llm = new ChatOpenAI({
      model: "gpt-4o",
      apiKey: "x",
      callbacks: [tracer.langchainHandler()],
      configuration: { fetch: stubFetch([CHAT_RESPONSE], []) },
    });

    const out = await llm.invoke([
      ["system", "You are helpful."],
      ["human", "hi"],
    ]);
    expect(out.content).toBe("Hello there!");
    await tracer.close();

    const ct = CTrace.open(path);
    expect(ct.getRun().provider).toBe("openai");
    const calls = ct.getCalls();
    expect(calls).toHaveLength(1);
    expect(calls[0]!.provider).toBe("openai");
    expect(calls[0]!.usage).toEqual({
      prompt_tokens: 10,
      completion_tokens: 5,
      total_tokens: 15,
    });
    expect(calls[0]!.params["model"]).toBe("gpt-4o");
    expect(calls[0]!.error).toBeNull();
    const blocks = ct.getCallBlocks(calls[0]!.id);
    expect(blocks.map((b) => [b.block.role, b.block.kind, b.block.text])).toEqual([
      ["system", "message", "You are helpful."],
      ["user", "message", "hi"],
    ]);
    ct.close();
  });

  it("records blocks hash-identical to directly wrapping the OpenAI SDK", async () => {
    // THE headline guarantee: the same prompt through the handler and through
    // `tracer.wrap(new OpenAI())` must carry the same block hashes, or a team
    // running both paths would see one prompt as two unrelated contexts.
    const path = tmpTrace();
    const tracer = init("p", { path });
    const fetchStub = stubFetch([CHAT_RESPONSE], []);

    const llm = new ChatOpenAI({
      model: "gpt-4o",
      apiKey: "x",
      callbacks: [tracer.langchainHandler()],
      configuration: { fetch: fetchStub },
    });
    await llm.invoke([
      ["system", "You are helpful."],
      ["human", "hi"],
    ]);

    const client = tracer.wrap(
      new OpenAI({ apiKey: "x", fetch: fetchStub as never }),
    ) as OpenAI;
    await client.chat.completions.create({
      model: "gpt-4o",
      messages: [
        { role: "system", content: "You are helpful." },
        { role: "user", content: "hi" },
      ],
    });
    await tracer.close();

    const ct = CTrace.open(path);
    const [viaHandler, viaWrap] = ct.getCalls();
    expect(ct.getCallBlocks(viaHandler!.id).map((b) => b.block.contentHash)).toEqual(
      ct.getCallBlocks(viaWrap!.id).map((b) => b.block.contentHash),
    );
    expect(viaHandler!.usage).toEqual(viaWrap!.usage);
    ct.close();
  });

  it("records blocks matching the wire body LangChain actually sent", async () => {
    const path = tmpTrace();
    const sent: unknown[] = [];
    const tracer = init("p", { path });
    const llm = new ChatOpenAI({
      model: "gpt-4o",
      apiKey: "x",
      callbacks: [tracer.langchainHandler()],
      configuration: { fetch: stubFetch([CHAT_RESPONSE], sent) },
    });
    await llm.invoke([
      ["system", "Be terse."],
      ["human", "explain hashing"],
    ]);
    await tracer.close();

    const ct = CTrace.open(path);
    const calls = ct.getCalls();
    expect(ct.getCallBlocks(calls[0]!.id).map((b) => b.block.contentHash)).toEqual(
      openaiHashes(sent[0] as Record<string, unknown>),
    );
    ct.close();
  });

  it("pins the cross-SDK hashes the Python handler produces for the same prompt", async () => {
    // These literals are asserted by the Python suite too. They are what makes
    // "a trace written by either SDK dedups against the other" a test rather
    // than a claim: both handlers must normalize `SystemMessage("You are
    // helpful.") + HumanMessage("hi")` to the same two blocks.
    const path = tmpTrace();
    const tracer = init("p", { path });
    const llm = new ChatOpenAI({
      model: "gpt-4o",
      apiKey: "x",
      callbacks: [tracer.langchainHandler()],
      configuration: { fetch: stubFetch([CHAT_RESPONSE], []) },
    });
    await llm.invoke([
      ["system", "You are helpful."],
      ["human", "hi"],
    ]);
    await tracer.close();

    const ct = CTrace.open(path);
    const calls = ct.getCalls();
    expect(ct.getCallBlocks(calls[0]!.id).map((b) => b.block.contentHash)).toEqual([
      "5a3c8882fb013e887693ebf8ce6c593b4f4f4131edbaffff2b5c163b412aca1e",
      "4e6c4093072114cd3ec3641653e12f750391cded3515bf460ccd07162c647685",
    ]);
    ct.close();
  });

  it("pins the tool-call hash the two SDKs are KNOWN to disagree on", async () => {
    // The limit of cross-SDK hash identity, pinned so it cannot drift
    // unnoticed. LangChain re-serializes a normalized tool call with the host
    // language's own JSON serializer, and `JSON.stringify` and `json.dumps`
    // disagree about separators — so the same logical message hashes
    // differently in the two SDKs. Each handler is faithful to its OWN
    // framework's real wire (that is what the wire-body tests assert), which
    // is the stronger guarantee: normalizing to a common form would make the
    // recorded block stop matching the body LangChain actually sent, and a
    // direct capture of the two frameworks' requests would still differ by
    // exactly these bytes anyway. The Python suite asserts the mirror of
    // this, so a change on either side fails on both.
    const path = tmpTrace();
    const tracer = init("p", { path });
    const handler = tracer.langchainHandler();
    handler.handleChatModelStart(
      { id: ["langchain", "chat_models", "openai", "ChatOpenAI"] },
      [
        [
          new HumanMessage("weather in Dubai?"),
          new AIMessage({
            content: "",
            tool_calls: [{ name: "get_weather", args: { city: "Dubai" }, id: "call_1" }],
          }),
        ],
      ],
      "run-1",
      undefined,
      { invocation_params: { model: "gpt-4o" } },
    );
    handler.handleLLMEnd({ generations: [] }, "run-1");
    await tracer.close();

    const ct = CTrace.open(path);
    const blocks = ct.getCallBlocks(ct.getCalls()[0]!.id);
    const callBlock = blocks[1]!.block;
    expect(JSON.parse(callBlock.text)["function"]["arguments"]).toBe(JS_TOOL_ARGUMENTS);
    expect(callBlock.contentHash).toBe(JS_TOOL_CALL_HASH);
    expect(JS_TOOL_CALL_HASH).not.toBe(PY_TOOL_CALL_HASH);
    ct.close();
  });

  it("records a Gemini vision turn matching the wire body LangChain sent", async () => {
    // The Gemini branch, checked against the body @langchain/google-genai
    // really sent — a vision turn with two text parts and an image, which is
    // where the branch was wrong. The real wire carries three parts, so the
    // trace must carry three blocks. @langchain/google-genai takes no custom
    // fetch, so the global one is swapped for the duration of the call.
    const path = tmpTrace();
    const sent: Record<string, unknown>[] = [];
    const tracer = init("p", { path });
    const llm = new ChatGoogleGenerativeAI({
      model: "gemini-2.0-flash",
      apiKey: "x",
      callbacks: [tracer.langchainHandler()],
    });

    const realFetch = globalThis.fetch;
    globalThis.fetch = (async (_url: unknown, req: { body?: string }) => {
      sent.push(JSON.parse(req?.body ?? "{}"));
      return new Response(JSON.stringify(GEMINI_RESPONSE), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }) as typeof globalThis.fetch;
    try {
      await llm.invoke([
        new SystemMessage("Be terse."),
        new HumanMessage({
          content: [
            { type: "text", text: "look at this:" },
            { type: "text", text: " carefully" },
            { type: "image_url", image_url: { url: PNG_4x4_URI } },
          ],
        }),
      ]);
    } finally {
      globalThis.fetch = realFetch;
    }
    await tracer.close();

    const ct = CTrace.open(path);
    const calls = ct.getCalls();
    expect(calls).toHaveLength(1);
    expect(calls[0]!.provider).toBe("gemini");
    expect(calls[0]!.usage).toEqual({
      prompt_token_count: 10,
      candidates_token_count: 5,
      total_token_count: 15,
    });

    const blocks = ct.getCallBlocks(calls[0]!.id);
    expect(blocks.map((b) => [b.block.role, b.block.kind, b.block.text])).toEqual([
      ["system", "message", "Be terse."],
      ["user", "content_part", "look at this:"],
      ["user", "content_part", " carefully"],
      ["user", "image", "[image 4×4 · ~258 tok]"],
    ]);
    expect(blocks.map((b) => b.block.contentHash)).toEqual(
      geminiHashes(sent[0] as Record<string, unknown>),
    );
    ct.close();
  });

  it("records a ChatBedrockConverse turn matching the wire body LangChain sent", async () => {
    // The Bedrock branch, end to end against real @langchain/aws with the AWS
    // SDK's transport stubbed — a tool-using exchange, because that is where
    // the Converse shape is least like everyone else's: system prompts are a
    // list of `{text}` blocks, tool schemas ride under `toolConfig`, and a tool
    // RESULT lives inside a user-role message (Converse has no tool role).
    const path = tmpTrace();
    const sent: Record<string, unknown>[] = [];
    const tracer = init("p", { path });
    const getWeather = tool(async () => "42°C", {
      name: "get_weather",
      description: "Look up the weather for a city.",
      schema: z.object({ city: z.string() }),
    });
    const llm = new ChatBedrockConverse({
      model: "anthropic.claude-3-haiku-20240307-v1:0",
      region: "us-east-1",
      credentials: { accessKeyId: "x", secretAccessKey: "y" },
      temperature: 0.2,
      maxTokens: 256,
      callbacks: [tracer.langchainHandler()],
      client: bedrockStubClient(sent),
    }).bindTools([getWeather]);

    await llm.invoke([
      new SystemMessage("Be terse."),
      new HumanMessage("weather in Dubai?"),
      new AIMessage({
        content: "",
        tool_calls: [{ id: "tooluse_1", name: "get_weather", args: { city: "Dubai" } }],
      }),
      new ToolMessage({ tool_call_id: "tooluse_1", content: "42°C" }),
    ]);
    await tracer.close();

    const ct = CTrace.open(path);
    const calls = ct.getCalls();
    expect(calls).toHaveLength(1);
    expect(calls[0]!.provider).toBe("bedrock");
    // Usage mapped back onto Bedrock's own key names, and the sampling params
    // lifted out of `inferenceConfig` exactly as a direct capture's would be.
    expect(calls[0]!.usage).toEqual({ inputTokens: 12, outputTokens: 6, totalTokens: 18 });
    expect(calls[0]!.params).toEqual({
      modelId: "anthropic.claude-3-haiku-20240307-v1:0",
      maxTokens: 256,
      temperature: 0.2,
    });

    const blocks = ct.getCallBlocks(calls[0]!.id);
    expect(blocks.map((b) => [b.block.role, b.block.kind])).toEqual([
      ["system", "message"],
      ["system", "tool_schema"],
      ["user", "content_part"],
      ["assistant", "content_part"],
      ["user", "content_part"],
    ]);
    expect(blocks[0]!.block.text).toBe("Be terse.");
    expect(blocks[2]!.block.text).toBe("weather in Dubai?");
    // THE LOAD-BEARING ASSERTION: hash for hash with a direct capture of the
    // JSON @langchain/aws actually put on the wire.
    expect(blocks.map((b) => b.block.contentHash)).toEqual(bedrockHashes(sent[0]!));
    ct.close();
  });

  it("captures a streamed call once, with usage", async () => {
    // LangChain fires `handleLLMEnd` once the stream is consumed, with the
    // aggregated result — so streaming needs no special handling on this path.
    const sse =
      'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-4o",' +
      '"choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n' +
      'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-4o",' +
      '"choices":[{"index":0,"delta":{"content":" there!"},"finish_reason":"stop"}]}\n\n' +
      'data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-4o",' +
      '"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n' +
      "data: [DONE]\n\n";
    const path = tmpTrace();
    const tracer = init("p", { path });
    const llm = new ChatOpenAI({
      model: "gpt-4o",
      apiKey: "x",
      streamUsage: true,
      callbacks: [tracer.langchainHandler()],
      configuration: {
        fetch: async () =>
          new Response(sse, { status: 200, headers: { "content-type": "text/event-stream" } }),
      },
    });

    let text = "";
    for await (const chunk of await llm.stream("hi")) text += chunk.content;
    expect(text).toBe("Hello there!");
    await tracer.close();

    const ct = CTrace.open(path);
    const calls = ct.getCalls();
    expect(calls).toHaveLength(1);
    expect(calls[0]!.usage).toEqual({
      prompt_tokens: 10,
      completion_tokens: 5,
      total_tokens: 15,
    });
    expect(ct.getCallBlocks(calls[0]!.id).map((b) => b.block.text)).toEqual(["hi"]);
    ct.close();
  });

  it("records a failed call with its error, and re-throws to the host", async () => {
    const path = tmpTrace();
    const tracer = init("p", { path });
    const llm = new ChatOpenAI({
      model: "gpt-4o",
      apiKey: "x",
      maxRetries: 0,
      callbacks: [tracer.langchainHandler()],
      configuration: {
        fetch: async () =>
          new Response(JSON.stringify({ error: { message: "boom" } }), {
            status: 500,
            headers: { "content-type": "application/json" },
          }),
      },
    });

    await expect(llm.invoke("hi")).rejects.toBeTruthy();
    await tracer.close();

    const ct = CTrace.open(path);
    const calls = ct.getCalls();
    expect(calls).toHaveLength(1);
    expect(calls[0]!.error).not.toBeNull();
    expect(calls[0]!.usage).toBeNull();
    expect(ct.getCallBlocks(calls[0]!.id).map((b) => b.block.text)).toEqual(["hi"]);
    ct.close();
  });

  it("stamps the agent name on every call it records", async () => {
    const path = tmpTrace();
    const tracer = init("p", { path });
    const llm = new ChatOpenAI({
      model: "gpt-4o",
      apiKey: "x",
      callbacks: [tracer.langchainHandler({ agent: "researcher" })],
      configuration: { fetch: stubFetch([CHAT_RESPONSE], []) },
    });
    await llm.invoke("hi");
    await tracer.close();

    const ct = CTrace.open(path);
    expect(ct.getCalls()[0]!.agent).toBe("researcher");
    ct.close();
  });

  it("records each run exactly once and ignores an unmatched end", async () => {
    // Driven directly (no HTTP): an end for a run that never started records
    // nothing, and a repeated end for a run that did records only once.
    const path = tmpTrace();
    const tracer = init("p", { path });
    const handler = tracer.langchainHandler();

    handler.handleLLMEnd({ generations: [] }, "never-started");
    handler.handleChatModelStart(
      { id: ["langchain", "chat_models", "openai", "ChatOpenAI"] },
      [[new HumanMessage("hi")]],
      "run-1",
      undefined,
      { invocation_params: { model: "gpt-4o" } },
    );
    handler.handleLLMEnd({ generations: [] }, "run-1");
    handler.handleLLMEnd({ generations: [] }, "run-1");
    await tracer.close();

    const ct = CTrace.open(path);
    const calls = ct.getCalls();
    expect(calls).toHaveLength(1);
    expect(ct.getCallBlocks(calls[0]!.id).map((b) => b.block.text)).toEqual(["hi"]);
    ct.close();
  });

  it("never breaks the host when recording is broken (fail-open)", async () => {
    const path = tmpTrace();
    const tracer = init("p", { path });
    const handler = tracer.langchainHandler();
    // Break the tracer's own record path: the LangChain call must still work.
    (tracer as unknown as { onCreate: () => void }).onCreate = () => {
      throw new Error("recording is broken");
    };
    const llm = new ChatOpenAI({
      model: "gpt-4o",
      apiKey: "x",
      callbacks: [handler],
      configuration: { fetch: stubFetch([CHAT_RESPONSE], []) },
    });
    const out = await llm.invoke("hi");
    expect(out.content).toBe("Hello there!");
    await tracer.close();
  });
});

describe("langchain normalization", () => {
  // Unit coverage for the branches the end-to-end tests can't reach without
  // every provider integration installed. The message stand-ins carry exactly
  // the attributes the normalizer duck-types.
  const msg = (
    type: string,
    content: unknown,
    extra: Record<string, unknown> = {},
  ): Record<string, unknown> => ({ type, content, ...extra });

  it("derives the provider from LangChain's own ls_provider first", () => {
    expect(providerFor({ id: ["x", "SomethingNew"] }, { ls_provider: "anthropic" })).toBe(
      "anthropic",
    );
    expect(providerFor(null, { ls_provider: "google_vertexai" })).toBe("gemini");
    expect(providerFor(null, { ls_provider: "amazon_bedrock" })).toBe("bedrock");
    expect(providerFor(null, { ls_provider: "azure_openai" })).toBe("openai");
  });

  it("falls back to the model class name, then to the OpenAI wire shape", () => {
    expect(providerFor({ id: ["langchain", "chat_models", "x", "ChatAnthropic"] }, null)).toBe(
      "anthropic",
    );
    expect(providerFor({ id: ["x", "ChatVertexAI"] }, null)).toBe("gemini");
    expect(providerFor({ id: ["x", "ChatBrandNew"] }, {})).toBe("openai");
    expect(providerFor(null, null)).toBe("openai");
  });

  it("rebuilds the OpenAI wire shape, tools and params", () => {
    const tools = [{ type: "function", function: { name: "f" } }];
    const wire = toWire("openai", [msg("system", "sys"), msg("human", "hi")], {
      model: "gpt-4o",
      temperature: 0.2,
      tools,
    });
    expect(wire["messages"]).toEqual([
      { role: "system", content: "sys" },
      { role: "user", content: "hi" },
    ]);
    expect(wire["tools"]).toEqual(tools);
    expect(wire["model"]).toBe("gpt-4o");
    expect(wire["temperature"]).toBe(0.2);
  });

  it("rebuilds tool calls the way LangChain re-serializes them, raw form as fallback", () => {
    // Rebuilt from the normalized `.tool_calls` with the same JSON.stringify
    // LangChain itself uses — matching the wire, whitespace included.
    const rebuilt = toWire(
      "openai",
      [msg("ai", "", { tool_calls: [{ name: "f", args: { city: "Dubai" }, id: "call_1" }] })],
      {},
    );
    expect((rebuilt["messages"] as Record<string, unknown>[])[0]!["tool_calls"]).toEqual([
      { id: "call_1", type: "function", function: { name: "f", arguments: '{"city":"Dubai"}' } },
    ]);
    expect((rebuilt["messages"] as Record<string, unknown>[])[0]!["content"]).toBeNull();

    // Only when there is no normalized form does the provider's raw payload
    // get used, verbatim.
    const raw = [
      { id: "call_1", type: "function", function: { name: "f", arguments: '{"x": 1}' } },
    ];
    const kept = toWire("openai", [msg("ai", "", { additional_kwargs: { tool_calls: raw } })], {});
    expect((kept["messages"] as Record<string, unknown>[])[0]!["tool_calls"]).toEqual(raw);
  });

  it("lifts the Anthropic system prompt out of the message list", () => {
    const wire = toWire("anthropic", [msg("system", "sys"), msg("human", "hi")], {
      model: "claude",
      max_tokens: 100,
    });
    expect(wire["system"]).toBe("sys");
    expect(wire["messages"]).toEqual([{ role: "user", content: "hi" }]);
    expect(wire["max_tokens"]).toBe(100);

    const two = toWire("anthropic", [msg("system", "a"), msg("system", "b")], {});
    expect(two["system"]).toEqual([
      { type: "text", text: "a" },
      { type: "text", text: "b" },
    ]);
  });

  it("uses Gemini's contents/parts shape and the model role", () => {
    const wire = toWire(
      "gemini",
      [msg("system", "sys"), msg("human", "hi"), msg("ai", "there")],
      { model: "gemini-2.0-flash" },
    );
    expect((wire["config"] as Record<string, unknown>)["systemInstruction"]).toBe("sys");
    expect(wire["contents"]).toEqual([
      { role: "user", parts: [{ text: "hi" }] },
      { role: "model", parts: [{ text: "there" }] },
    ]);
    expect(wire["model"]).toBe("gemini-2.0-flash");
  });

  it("emits one Gemini part per content entry, images included", () => {
    // A vision turn is THREE parts, not one. Flattening the content list to a
    // single string dropped the image entirely (its whole token cost vanished
    // from the trace) and merged the two text entries into one block, so the
    // same turn recorded fewer blocks through LangChain than through a direct
    // capture and stopped deduping against it. Mirrors the Python unit test
    // `test_gemini_wire_emits_one_part_per_content_entry`.
    const imagePart = { type: "image_url", image_url: { url: PNG_4x4_URI } };
    const wire = toWire(
      "gemini",
      [
        msg("human", [
          { type: "text", text: "a" },
          { type: "text", text: "b" },
          imagePart,
        ]),
      ],
      {},
    );
    expect(wire["contents"]).toEqual([
      { role: "user", parts: [{ text: "a" }, { text: "b" }, imagePart] },
    ]);

    const blocks = new GeminiAdapter().extractBlocks(wire);
    expect(blocks.map((b) => [b.role, b.kind, b.text])).toEqual([
      ["user", "content_part", "a"],
      ["user", "content_part", "b"],
      // ...and the image is an IMAGE block: a descriptor, a real vision-token
      // estimate, and identity taken from the picture's bytes.
      ["user", "image", "[image 4×4 · ~258 tok]"],
    ]);
  });

  it("hashes a Gemini vision turn exactly like a direct capture", () => {
    // The same picture as an OpenAI-style data URI (what LangChain hands the
    // handler) and as Gemini `inlineData` bytes (what goes on the wire) is ONE
    // block — image identity is the pixels, not the wrapper.
    const viaHandler = toWire(
      "gemini",
      [msg("human", [{ type: "text", text: "what is this?" }, { type: "image_url", image_url: { url: PNG_4x4_URI } }])],
      {},
    );
    const direct = {
      contents: [
        {
          role: "user",
          parts: [
            { text: "what is this?" },
            { inlineData: { mimeType: "image/png", data: PNG_4x4_B64 } },
          ],
        },
      ],
    };
    const adapter = new GeminiAdapter();
    expect(adapter.extractBlocks(viaHandler).map((b) => b.hashInput)).toEqual(
      adapter.extractBlocks(direct).map((b) => b.hashInput),
    );
  });

  it("uses Bedrock's Converse shape: system blocks, typed content, toolUse/toolResult", () => {
    // Converse is the odd one out three times over: `system` is a LIST of
    // {text} blocks (never a bare string, unlike Anthropic), content is a list
    // of typed blocks, and there is no tool role — a tool result rides inside a
    // USER message. Mirrors Python `test_bedrock_wire_*`.
    const wire = toWire(
      "bedrock",
      [
        msg("system", "sys"),
        msg("human", "weather?"),
        msg("ai", "", { tool_calls: [{ id: "t1", name: "get_weather", args: { city: "Dubai" } }] }),
        msg("tool", "42°C", { tool_call_id: "t1" }),
      ],
      { toolConfig: { tools: [{ toolSpec: { name: "get_weather" } }] } },
      "anthropic.claude-3-haiku",
    );
    expect(wire["system"]).toEqual([{ text: "sys" }]);
    expect(wire["messages"]).toEqual([
      { role: "user", content: [{ text: "weather?" }] },
      {
        role: "assistant",
        content: [{ toolUse: { toolUseId: "t1", name: "get_weather", input: { city: "Dubai" } } }],
      },
      {
        role: "user",
        content: [{ toolResult: { toolUseId: "t1", content: [{ text: "42°C" }] } }],
      },
    ]);
    // The model id goes under Bedrock's own key, so the session's model roll-up
    // still works.
    expect(wire["modelId"]).toBe("anthropic.claude-3-haiku");
    // Tool schemas ride under `toolConfig`, whichever way the integration
    // reported them — verbatim, so they are byte-identical to a direct capture.
    expect(wire["toolConfig"]).toEqual({ tools: [{ toolSpec: { name: "get_weather" } }] });
    const viaTools = toWire("bedrock", [], { tools: [{ toolSpec: { name: "a" } }] });
    expect(viaTools["toolConfig"]).toEqual({ tools: [{ toolSpec: { name: "a" } }] });
  });

  it("emits one Bedrock content block per content entry, images included", () => {
    // The same part-per-entry rule as Gemini, for the same reason: flattening a
    // vision turn to one string would drop the image and its whole token cost.
    const imagePart = { type: "image_url", image_url: { url: PNG_4x4_URI } };
    const wire = toWire(
      "bedrock",
      [msg("human", [{ type: "text", text: "a" }, { type: "text", text: "b" }, imagePart])],
      {},
    );
    expect(wire["messages"]).toEqual([
      { role: "user", content: [{ text: "a" }, { text: "b" }, imagePart] },
    ]);
    expect(new BedrockAdapter().extractBlocks(wire).map((b) => [b.role, b.kind, b.text])).toEqual([
      ["user", "content_part", "a"],
      ["user", "content_part", "b"],
      // Bedrock shares Anthropic's w×h/750 vision formula, so the SAME picture
      // costs 1 token here and 258 on Gemini — each provider's own published
      // formula, never a shared guess.
      ["user", "image", "[image 4×4 · ~1 tok]"],
    ]);
  });

  it("hashes a Bedrock vision turn exactly like a direct capture", () => {
    // The picture as an OpenAI-style data URI (what LangChain hands the
    // handler) and as Converse `image.source.bytes` (what goes on the wire) is
    // ONE block — image identity is the pixels, not the wrapper.
    const viaHandler = toWire(
      "bedrock",
      [
        msg("human", [
          { type: "text", text: "what is this?" },
          { type: "image_url", image_url: { url: PNG_4x4_URI } },
        ]),
      ],
      {},
    );
    const direct = {
      messages: [
        {
          role: "user",
          content: [
            { text: "what is this?" },
            { image: { format: "png", source: { bytes: PNG_4x4_B64 } } },
          ],
        },
      ],
    };
    const adapter = new BedrockAdapter();
    expect(adapter.extractBlocks(viaHandler).map((b) => b.hashInput)).toEqual(
      adapter.extractBlocks(direct).map((b) => b.hashInput),
    );
  });

  it("keeps plain string content as a single Gemini text part", () => {
    // The common case is unchanged, and an empty message contributes no part
    // rather than a phantom empty block.
    const wire = toWire("gemini", [msg("human", "hi"), msg("ai", "")], {});
    expect(wire["contents"]).toEqual([
      { role: "user", parts: [{ text: "hi" }] },
      { role: "model", parts: [] },
    ]);
  });

  it("never lets invocation params overwrite rebuilt content", () => {
    const wire = toWire("openai", [msg("human", "hi")], {
      messages: [{ role: "user", content: "WRONG" }],
    });
    expect(wire["messages"]).toEqual([{ role: "user", content: "hi" }]);
  });

  it("maps usage onto each provider's own key names", () => {
    const result = {
      generations: [
        [{ message: { usage_metadata: { input_tokens: 10, output_tokens: 5, total_tokens: 15 } } }],
      ],
    };
    expect(usageState("openai", result)).toEqual({
      prompt_tokens: 10,
      completion_tokens: 5,
      total_tokens: 15,
    });
    expect(usageState("anthropic", result)).toEqual({ input_tokens: 10, output_tokens: 5 });
    // Gemini's names are the @google/genai SDK's camelCase ones, because what
    // this builds is a stand-in RESPONSE that the adapter still has to read.
    // The STORED shape is what `extractUsage` makes of it — the snake_case
    // names a `.ctrace` carries, identical to Python's. Naming the stored keys
    // here instead left `extractUsage` finding nothing and storing all nulls.
    expect(usageState("gemini", result)).toEqual({
      promptTokenCount: 10,
      candidatesTokenCount: 5,
      totalTokenCount: 15,
    });
    expect(new GeminiAdapter().extractUsage({ usageMetadata: usageState("gemini", result) })).toEqual(
      { prompt_token_count: 10, candidates_token_count: 5, total_token_count: 15 },
    );
    expect(usageState("bedrock", result)).toEqual({
      inputTokens: 10,
      outputTokens: 5,
      totalTokens: 15,
    });
  });

  it("falls back to llmOutput, and reports nothing rather than zeros", () => {
    expect(
      usageState("openai", {
        generations: [],
        llmOutput: { tokenUsage: { promptTokens: 3, completionTokens: 1, totalTokens: 4 } },
      }),
    ).toEqual({ prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 });
    expect(usageState("openai", { generations: [] })).toEqual({});
    expect(usageState("openai", {})).toEqual({});
  });
});

describe("langgraph", () => {
  /** A minimal but genuine ReAct graph: a model node, a tool node, and a
   * conditional edge that loops back once the model asks for a tool. Built by
   * hand so the test depends only on LangGraph's stable core — and so the
   * callback propagation being exercised is visibly the GRAPH's. */
  function buildGraph(llm: ChatOpenAI) {
    const getWeather = tool(async () => "sunny", {
      name: "get_weather",
      description: "Get the weather for a city.",
      schema: z.object({ city: z.string() }),
    });
    const bound = llm.bindTools([getWeather]);
    const graph = new StateGraph<{ messages: unknown[] }>({
      channels: { messages: { reducer: (a: unknown[], b: unknown[]) => [...a, ...b], default: () => [] } },
    } as never)
      .addNode("model", async (state: { messages: unknown[] }) => ({
        messages: [await bound.invoke(state.messages as never)],
      }))
      .addNode("tools", new ToolNode([getWeather]) as never)
      .addEdge(START, "model" as never)
      .addConditionalEdges(
        "model" as never,
        (state: { messages: unknown[] }) => {
          const last = state.messages[state.messages.length - 1] as { tool_calls?: unknown[] };
          return last?.tool_calls?.length ? "tools" : END;
        },
        { tools: "tools", [END]: END } as never,
      )
      .addEdge("tools" as never, "model" as never);
    return graph.compile();
  }

  it("captures every turn of a LangGraph run, with the handler attached only at invoke", async () => {
    // The whole point: LangGraph PROPAGATES the callbacks given at invoke time
    // down through every node, so the handler is never attached to the model.
    const path = tmpTrace();
    const sent: unknown[] = [];
    const tracer = init("p", { path });
    const llm = new ChatOpenAI({
      model: "gpt-4o",
      apiKey: "x",
      configuration: { fetch: stubFetch([TOOL_CALL_RESPONSE, FINAL_RESPONSE], sent) },
    });

    const result = await buildGraph(llm).invoke(
      { messages: [new HumanMessage("weather in Dubai?")] },
      { callbacks: [tracer.langchainHandler({ agent: "weatherbot" })] },
    );
    const messages = result.messages as { content: unknown }[];
    expect(messages[messages.length - 1]!.content).toBe("It is sunny in Dubai.");
    await tracer.close();

    const ct = CTrace.open(path);
    const calls = ct.getCalls();
    expect(calls).toHaveLength(2); // callbacks reached both model turns
    expect(calls.map((c) => c.seq)).toEqual([1, 2]);
    expect(calls.every((c) => c.agent === "weatherbot")).toBe(true);
    expect(calls[0]!.usage).toEqual({
      prompt_tokens: 20,
      completion_tokens: 8,
      total_tokens: 28,
    });

    const first = ct.getCallBlocks(calls[0]!.id);
    expect(first.map((b) => [b.block.role, b.block.kind])).toEqual([
      ["system", "tool_schema"],
      ["user", "message"],
    ]);
    const second = ct.getCallBlocks(calls[1]!.id);
    // Turn 1's blocks are the SAME blocks (same hashes) at the front of turn 2
    // — a stable cache prefix, not a rewritten context — followed by the tool
    // call and its result.
    expect(second.slice(0, 2).map((b) => b.block.contentHash)).toEqual(
      first.map((b) => b.block.contentHash),
    );
    expect(second.slice(2).map((b) => [b.block.role, b.block.kind])).toEqual([
      ["assistant", "content_part"],
      ["tool", "message"],
    ]);
    expect(second[3]!.block.text).toBe("sunny");

    // ...and every turn matches the body LangChain actually put on the wire.
    calls.forEach((call, i) => {
      expect(ct.getCallBlocks(call.id).map((b) => b.block.contentHash)).toEqual(
        openaiHashes(sent[i] as Record<string, unknown>),
      );
    });
    ct.close();
  });
});
