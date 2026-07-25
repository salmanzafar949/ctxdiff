/**
 * The Bedrock adapter in isolation — the JS twin of Python's
 * `tests/test_bedrock_adapter.py`, plus the tests that only exist here: the
 * AWS SDK v3 has no method per operation, so `interpretCall` (which command is
 * this, is it recordable, where is the payload) is JS-only machinery and is
 * pinned accordingly.
 *
 * The extractor assertions deliberately mirror the Python file case for case.
 * The Converse wire shape is identical in the two SDKs, so any divergence in
 * these is a cross-SDK hash divergence — `conformance.test.ts` proves the
 * agreement end to end, and these say WHERE it broke when it does.
 */
import { describe, it, expect } from "vitest";
import { ConverseCommand, ConverseStreamCommand, InvokeModelCommand } from "@aws-sdk/client-bedrock-runtime";
import { BedrockAdapter } from "../src/capture/bedrock.js";

describe("BedrockAdapter.extractBlocks", () => {
  it("orders system, then tool schemas, then message content parts", () => {
    const blocks = new BedrockAdapter().extractBlocks({
      modelId: "anthropic.claude-3-haiku",
      system: [{ text: "Be helpful." }],
      toolConfig: { tools: [{ toolSpec: { name: "lookup", inputSchema: {} } }] },
      messages: [{ role: "user", content: [{ text: "hi" }] }],
    });
    expect(blocks.map((b) => b.kind)).toEqual(["message", "tool_schema", "content_part"]);
    expect(blocks.map((b) => b.role)).toEqual(["system", "system", "user"]);
    expect(blocks[0].text).toBe("Be helpful.");
    expect(blocks[1].text).toContain("lookup");
    expect(blocks[2].text).toBe("hi");
  });

  it("serializes a non-text system block as stable JSON", () => {
    // A `cachePoint` (or any other Converse system-block shape) has no `text`
    // key; it must stay diffable rather than vanish.
    const blocks = new BedrockAdapter().extractBlocks({
      system: [{ cachePoint: { type: "default" } }],
      messages: [],
    });
    expect(blocks[0].kind).toBe("message");
    expect(blocks[0].role).toBe("system");
    expect(blocks[0].text).toContain("cachePoint");
  });

  it("emits one tool_schema block per tool, from the toolSpec", () => {
    const blocks = new BedrockAdapter().extractBlocks({
      messages: [],
      toolConfig: {
        tools: [
          { toolSpec: { name: "a", inputSchema: {} } },
          { toolSpec: { name: "b", inputSchema: {} } },
        ],
      },
    });
    expect(blocks.map((b) => b.kind)).toEqual(["tool_schema", "tool_schema"]);
    expect(blocks[0].text).toContain('"a"');
    expect(blocks[1].text).toContain('"b"');
  });

  it("keeps text parts verbatim and stable-JSONs everything else, role passed through", () => {
    // Converse has no 'tool' role: a toolResult rides inside a USER message.
    const blocks = new BedrockAdapter().extractBlocks({
      messages: [
        {
          role: "user",
          content: [
            { text: "look this up" },
            { toolResult: { toolUseId: "1", content: [{ text: "42" }] } },
          ],
        },
        { role: "assistant", content: [{ text: "the answer is 42" }] },
      ],
    });
    expect(blocks.map((b) => b.kind)).toEqual(["content_part", "content_part", "content_part"]);
    expect(blocks.map((b) => b.role)).toEqual(["user", "user", "assistant"]);
    expect(blocks[0].text).toBe("look this up");
    expect(blocks[1].text).toContain("toolResult");
    expect(blocks[1].text).toContain("toolUseId");
    expect(blocks[2].text).toBe("the answer is 42");
  });

  it("turns an image part into an image block hashed over the BYTES", () => {
    // 1×1 PNG. The block's text is a descriptor and its identity is the
    // picture, never the base64 — the whole point of images.ts.
    const png = Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAC0lEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC",
      "base64",
    );
    const blocks = new BedrockAdapter().extractBlocks({
      messages: [
        {
          role: "user",
          content: [{ text: "what is this" }, { image: { format: "png", source: { bytes: png } } }],
        },
      ],
    });
    expect(blocks.map((b) => b.kind)).toEqual(["content_part", "image"]);
    expect(blocks[1].text).toMatch(/^\[image 1×1 · ~\d+ tok\]$/);
    // Identity is the bytes, not the descriptor, and the cost is an estimate.
    expect(blocks[1].hashInput).not.toBe(blocks[1].text);
    expect(blocks[1].tokenMethod).toBe("estimate");
    // The base64 payload is nowhere in the stored text.
    expect(blocks[1].text).not.toContain("iVBOR");
  });

  it("does not raise when system/toolConfig/messages are absent", () => {
    const adapter = new BedrockAdapter();
    expect(adapter.extractBlocks({ messages: [{ role: "user", content: [{ text: "hi" }] }] })).toEqual([
      { role: "user", kind: "content_part", text: "hi" },
    ]);
    expect(adapter.extractBlocks({})).toEqual([]);
  });
});

describe("BedrockAdapter.extractParams", () => {
  it("drops content keys, keeps modelId, flattens inferenceConfig scalars", () => {
    const params = new BedrockAdapter().extractParams({
      modelId: "anthropic.claude-3-haiku",
      system: [{ text: "x" }],
      messages: [{ role: "user", content: [{ text: "hi" }] }],
      toolConfig: { tools: [] },
      inferenceConfig: { maxTokens: 256, temperature: 0.2, topP: 0.9, stopSequences: ["END"] },
    });
    expect(params).toEqual({
      modelId: "anthropic.claude-3-haiku",
      maxTokens: 256,
      temperature: 0.2,
      topP: 0.9,
      stopSequences: ["END"],
    });
  });

  it("works with no inferenceConfig, and flattens only the fields present", () => {
    const adapter = new BedrockAdapter();
    expect(adapter.extractParams({ modelId: "m", messages: [] })).toEqual({ modelId: "m" });
    expect(
      adapter.extractParams({ modelId: "m", messages: [], inferenceConfig: { maxTokens: 100 } }),
    ).toEqual({ modelId: "m", maxTokens: 100 });
  });
});

describe("BedrockAdapter.extractUsage", () => {
  it("reads Converse's three counts off the response", () => {
    expect(
      new BedrockAdapter().extractUsage({
        output: { message: { role: "assistant", content: [{ text: "hi" }] } },
        usage: { inputTokens: 12, outputTokens: 6, totalTokens: 18 },
      }),
    ).toEqual({ inputTokens: 12, outputTokens: 6, totalTokens: 18 });
  });

  it("returns null when the response carries no usage at all", () => {
    const adapter = new BedrockAdapter();
    expect(adapter.extractUsage({ output: {} })).toBeNull();
    expect(adapter.extractUsage(null)).toBeNull();
    expect(adapter.extractUsage("nope")).toBeNull();
  });
});

describe("BedrockAdapter.accumulateStreamUsage", () => {
  it("takes the counts off the trailing metadata event only", () => {
    const state: Record<string, unknown> = {};
    const adapter = new BedrockAdapter();
    adapter.accumulateStreamUsage({ messageStart: { role: "assistant" } }, state);
    expect(state).toEqual({});
    adapter.accumulateStreamUsage(
      {
        metadata: {
          usage: { inputTokens: 12, outputTokens: 6, totalTokens: 18 },
          metrics: { latencyMs: 100 },
        },
      },
      state,
    );
    // The SAME key names extractUsage returns, so a streamed call's stored
    // usage is indistinguishable from a non-streamed one's.
    expect(state).toEqual({ inputTokens: 12, outputTokens: 6, totalTokens: 18 });
  });

  it("ignores malformed events without raising (fail-open)", () => {
    const state: Record<string, unknown> = {};
    const adapter = new BedrockAdapter();
    for (const chunk of [null, undefined, 42, { metadata: null }, { metadata: { usage: "nope" } }]) {
      adapter.accumulateStreamUsage(chunk, state);
    }
    expect(state).toEqual({});
  });
});

describe("BedrockAdapter.interpretCall (the AWS SDK v3 dispatch)", () => {
  const adapter = new BedrockAdapter();

  it("intercepts the single `send` path, and declares the stream envelope", () => {
    // ONE path, because the SDK has one method. Which operation it is gets
    // decided per call, from the command.
    expect(adapter.createPaths).toEqual([["send"]]);
    expect(adapter.streamEnvelopeKey).toBe("stream");
  });

  it("reads a ConverseCommand as a non-streaming call over command.input", () => {
    const input = { modelId: "m", messages: [{ role: "user", content: [{ text: "hi" }] }] };
    const shape = adapter.interpretCall([new ConverseCommand(input)]);
    expect(shape).toEqual({ kwargs: input, streaming: false });
  });

  it("reads a ConverseStreamCommand as a STREAMING call over command.input", () => {
    const input = { modelId: "m", messages: [{ role: "user", content: [{ text: "hi" }] }] };
    const shape = adapter.interpretCall([new ConverseStreamCommand(input)]);
    expect(shape?.streaming).toBe(true);
    expect(shape?.kwargs).toEqual(input);
  });

  it("ignores the second argument (options bag / callback)", () => {
    const input = { modelId: "m", messages: [] };
    const shape = adapter.interpretCall([new ConverseCommand(input), { abortSignal: undefined }]);
    expect(shape).toEqual({ kwargs: input, streaming: false });
  });

  it("returns null for every non-Converse command — nothing is guessed at", () => {
    // InvokeModel carries a raw provider-specific `body` STRING whose schema
    // depends on modelId. Reading it as Converse would store blocks that never
    // existed, so the call passes through unrecorded.
    expect(adapter.interpretCall([new InvokeModelCommand({ modelId: "m", body: "{}" })])).toBeNull();
    expect(adapter.interpretCall([{ constructor: { name: "ListAsyncInvokesCommand" }, input: {} }])).toBeNull();
    expect(adapter.interpretCall([undefined])).toBeNull();
    expect(adapter.interpretCall([])).toBeNull();
    expect(adapter.interpretCall(["not a command"])).toBeNull();
  });

  it("falls back to the Smithy schema when the class name has been minified", () => {
    // A bundler that mangles class names turns `ConverseCommand` into `t`; the
    // operation name still sits in the command's own wire descriptor.
    const real = new ConverseCommand({ modelId: "m", messages: [] });
    const minified = {
      constructor: { name: "t" },
      schema: real.schema,
      input: { modelId: "m", messages: [] },
    };
    expect(adapter.interpretCall([minified])?.streaming).toBe(false);
    expect(adapter.interpretCall([minified])?.kwargs).toEqual({ modelId: "m", messages: [] });
  });

  it("records a command with no input as an empty request rather than dropping it", () => {
    expect(adapter.interpretCall([new ConverseCommand({} as never)])).toEqual({
      kwargs: {},
      streaming: false,
    });
  });
});
