/**
 * Wrapping a REAL `@aws-sdk/client-bedrock-runtime` client end to end — the JS
 * twin of Python's `tests/eval/test_wrap_bedrock.py` +
 * `test_wrap_bedrock_stream.py`, and the place the JS-only interception design
 * is proved.
 *
 * WHY THE INTERCEPTION IS DIFFERENT HERE. boto3 exposes one method per
 * operation (`client.converse(...)`, `client.converse_stream(...)`), so Python
 * intercepts by ATTRIBUTE PATH. The AWS SDK v3 has exactly one method —
 * `client.send(command)` — with the operation and the payload both living on
 * the COMMAND. So the proxy hooks `send` and the adapter dispatches on the
 * argument (`BedrockAdapter.interpretCall`). Everything below exercises that
 * through the real client: the real middleware stack, the real serializer, the
 * real event-stream deserializer.
 *
 * WHY THE HTTP IS STUBBED THE WAY IT IS. The AWS SDK v3 takes a
 * `requestHandler` in its config — the supported seam, one layer below the
 * middleware stack — so the canned response travels through every real layer
 * above it. For `ConverseStream` that response body is genuine binary
 * `vnd.amazon.eventstream` FRAMES (`frame()` below), hand-encoded because the
 * SDK ships a decoder and no encoder — exactly the choice the Python suite
 * made, and for the same reason: the two facts this feature depends on (the
 * result is an ENVELOPE containing the stream, and the stream is a real async
 * iterable produced by the SDK's own deserializer) are properties of that real
 * code, not of a fixture.
 */
import { describe, it, expect, afterEach, vi } from "vitest";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { rmSync } from "node:fs";
import { Readable } from "node:stream";
import { crc32 } from "node:zlib";
import {
  BedrockRuntimeClient,
  ConverseCommand,
  ConverseStreamCommand,
  InvokeModelCommand,
} from "@aws-sdk/client-bedrock-runtime";
import { init } from "../src/trace.js";
import { CTrace } from "../src/store/ctrace.js";

const created: string[] = [];
function tmpTrace(): string {
  const p = join(tmpdir(), `ctxdiff-br-${randomUUID()}.ctrace`);
  created.push(p);
  return p;
}
afterEach(() => {
  vi.restoreAllMocks();
  for (const p of created.splice(0)) {
    try {
      rmSync(p, { force: true });
    } catch {
      /* best effort */
    }
  }
});

/** Every request body the stubbed handler saw, newest last — so a test can
 * assert what the SDK really put on the wire, not just what we handed it. */
let sentBodies: unknown[] = [];

/**
 * A `requestHandler` that answers every request from `body` without touching
 * the network. `contentType` selects the deserializer branch the SDK takes:
 * `application/json` for Converse, `application/vnd.amazon.eventstream` for
 * ConverseStream.
 */
function stubHandler(body: Buffer, contentType: string): object {
  return {
    handle: async (request: { body?: unknown }) => {
      sentBodies.push(typeof request.body === "string" ? JSON.parse(request.body) : request.body);
      return {
        response: {
          statusCode: 200,
          reason: "OK",
          headers: { "content-type": contentType },
          body: Readable.from([body]),
        },
      };
    },
    updateHttpClientConfig(): void {},
    httpHandlerConfigs(): object {
      return {};
    },
  };
}

/** A real BedrockRuntimeClient wired to a stubbed transport. Credentials are
 * static so nothing reaches for the instance metadata service. */
function stubClient(body: Buffer, contentType = "application/json"): BedrockRuntimeClient {
  sentBodies = [];
  return new BedrockRuntimeClient({
    region: "us-east-1",
    credentials: { accessKeyId: "x", secretAccessKey: "y" },
    requestHandler: stubHandler(body, contentType) as never,
  });
}

function jsonBody(obj: unknown): Buffer {
  return Buffer.from(JSON.stringify(obj));
}

const CONVERSE_RESPONSE = {
  output: { message: { role: "assistant", content: [{ text: "4" }] } },
  stopReason: "end_turn",
  usage: { inputTokens: 12, outputTokens: 6, totalTokens: 18 },
  metrics: { latencyMs: 100 },
};

// --- vnd.amazon.eventstream encoding ---------------------------------------

/** One event-stream header: 1-byte name length, the name, the value type
 * (7 = UTF-8 string, the only type these headers use), a 2-byte big-endian
 * value length, then the value. */
function header(name: string, value: string): Buffer {
  const n = Buffer.from(name);
  const v = Buffer.from(value);
  const len = Buffer.alloc(2);
  len.writeUInt16BE(v.length);
  return Buffer.concat([Buffer.from([n.length]), n, Buffer.from([7]), len, v]);
}

/**
 * ONE `vnd.amazon.eventstream` message the way Bedrock puts it on the wire: a
 * 12-byte prelude (total length, headers length, CRC32 of those eight bytes),
 * the three headers the SDK's parser dispatches on (`:event-type` — which
 * member of `ConverseStreamOutput` this is — `:message-type`, `:content-type`),
 * the JSON payload, and a trailing CRC32 over everything before it.
 */
function frame(eventType: string, payload: unknown): Buffer {
  const headers = Buffer.concat([
    header(":event-type", eventType),
    header(":message-type", "event"),
    header(":content-type", "application/json"),
  ]);
  const body = Buffer.from(JSON.stringify(payload));
  const prelude = Buffer.alloc(8);
  prelude.writeUInt32BE(16 + headers.length + body.length, 0);
  prelude.writeUInt32BE(headers.length, 4);
  const preludeCrc = Buffer.alloc(4);
  preludeCrc.writeUInt32BE(crc32(prelude) >>> 0);
  const message = Buffer.concat([prelude, preludeCrc, headers, body]);
  const messageCrc = Buffer.alloc(4);
  messageCrc.writeUInt32BE(crc32(message) >>> 0);
  return Buffer.concat([message, messageCrc]);
}

/**
 * The full event sequence a real Converse stream emits, in order:
 * `messageStart`, two `contentBlockDelta`s, `contentBlockStop`, `messageStop`,
 * and LAST the `metadata` event carrying usage — the only event with token
 * counts on it.
 */
function streamBody(text = "Hello there!", inputTokens = 12, outputTokens = 6): Buffer {
  return Buffer.concat([
    frame("messageStart", { role: "assistant" }),
    frame("contentBlockDelta", { delta: { text: text.slice(0, 5) }, contentBlockIndex: 0 }),
    frame("contentBlockDelta", { delta: { text: text.slice(5) }, contentBlockIndex: 0 }),
    frame("contentBlockStop", { contentBlockIndex: 0 }),
    frame("messageStop", { stopReason: "end_turn" }),
    frame("metadata", {
      usage: { inputTokens, outputTokens, totalTokens: inputTokens + outputTokens },
      metrics: { latencyMs: 100 },
    }),
  ]);
}

// --- detection --------------------------------------------------------------

describe("wrap Bedrock: detection", () => {
  it("recognizes a BedrockRuntimeClient and returns a working proxy", async () => {
    const path = tmpTrace();
    const client = stubClient(jsonBody(CONVERSE_RESPONSE));
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(client) as BedrockRuntimeClient;
    // A proxy, not the original object — but every member still resolves.
    expect(wrapped).not.toBe(client);
    expect(typeof wrapped.send).toBe("function");
    expect(wrapped.config.region).toBe(client.config.region);
    await wrapped.send(
      new ConverseCommand({ modelId: "m", messages: [{ role: "user", content: [{ text: "hi" }] }] }),
    );
    await tracer.close();
    const r = CTrace.open(path);
    expect(r.getRun().provider).toBe("bedrock");
    r.close();
  });

  it("recognizes a Bedrock client whose class name was minified, via serviceId", () => {
    const path = tmpTrace();
    const tracer = init("proj", { path });
    // Same duck shape, anonymous class: the resolved config's serviceId is the
    // signal that survives a bundler.
    const disguised = { send: () => undefined, config: { serviceId: "Bedrock Runtime" } };
    expect(tracer.wrap(disguised)).not.toBe(disguised);
  });

  it("leaves OTHER AWS SDK v3 clients completely alone", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const path = tmpTrace();
    const tracer = init("proj", { path });
    // Every v3 client is {send, config, middlewareStack}. Claiming them on that
    // shape alone would proxy S3/DynamoDB/SQS traffic that can never produce a
    // context block, so detection is narrowed to the Bedrock serviceId.
    for (const serviceId of ["S3", "DynamoDB", "SQS", "Bedrock"]) {
      const other = { send: () => undefined, config: { serviceId }, middlewareStack: {} };
      expect(tracer.wrap(other)).toBe(other);
    }
    expect(warn).toHaveBeenCalled();
  });
});

// --- non-streaming ----------------------------------------------------------

describe("wrap Bedrock: ConverseCommand (non-streaming)", () => {
  it("records blocks/usage/params/agent, returns the response unchanged, never mutates the request", async () => {
    const path = tmpTrace();
    const client = stubClient(jsonBody(CONVERSE_RESPONSE));
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(client, { agent: "solver" }) as BedrockRuntimeClient;

    const request = {
      modelId: "anthropic.claude-3-haiku-20240307-v1:0",
      system: [{ text: "Be terse." }],
      toolConfig: {
        tools: [
          {
            toolSpec: {
              name: "lookup",
              description: "look something up",
              inputSchema: { json: { type: "object", properties: { q: { type: "string" } } } },
            },
          },
        ],
      },
      messages: [{ role: "user" as const, content: [{ text: "what is 2+2" }] }],
      inferenceConfig: { maxTokens: 256, temperature: 0.2 },
    };
    const snapshot = structuredClone(request);
    const out = await wrapped.send(new ConverseCommand(request));

    // The host's own result, untouched, including the SDK's $metadata.
    expect(out.output?.message?.content?.[0]?.text).toBe("4");
    expect(out.$metadata.httpStatusCode).toBe(200);
    // ctxdiff never mutates the caller's request.
    expect(request).toEqual(snapshot);
    await tracer.close();

    const r = CTrace.open(path);
    const calls = r.getCalls();
    expect(calls).toHaveLength(1);
    expect(calls[0].provider).toBe("bedrock");
    expect(calls[0].agent).toBe("solver");
    expect(calls[0].usage).toEqual({ inputTokens: 12, outputTokens: 6, totalTokens: 18 });
    // Content keys dropped; modelId kept; inferenceConfig scalars flattened.
    expect(calls[0].params).toEqual({
      modelId: "anthropic.claude-3-haiku-20240307-v1:0",
      maxTokens: 256,
      temperature: 0.2,
    });
    const blocks = r.getCallBlocks(calls[0].id).map((cb) => [cb.block.role, cb.block.kind]);
    expect(blocks).toEqual([
      ["system", "message"],
      ["system", "tool_schema"],
      ["user", "content_part"],
    ]);
    expect(r.getRun().models).toEqual(["anthropic.claude-3-haiku-20240307-v1:0"]);
    r.close();
  });

  it("records an image turn as an image block, hashed over the bytes", async () => {
    const path = tmpTrace();
    const client = stubClient(jsonBody(CONVERSE_RESPONSE));
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(client) as BedrockRuntimeClient;

    // A 4×4 PNG as the AWS SDK really carries it: raw bytes, not base64.
    const png = Buffer.from(
      "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAC0lEQVR4nGNgYAAAAAMAAbitOmMAAAAASUVORK5CYII=",
      "base64",
    );
    await wrapped.send(
      new ConverseCommand({
        modelId: "m",
        messages: [
          {
            role: "user",
            content: [
              { text: "what is in this picture" },
              { image: { format: "png", source: { bytes: new Uint8Array(png) } } },
            ],
          },
        ],
      }),
    );
    await tracer.close();

    const r = CTrace.open(path);
    const blocks = r.getCallBlocks(r.getCalls()[0].id).map((cb) => cb.block);
    expect(blocks.map((b) => b.kind)).toEqual(["content_part", "image"]);
    // Bedrock shares Anthropic's published w×h/750 vision formula (see
    // src/images.ts), so a 4×4 picture rounds to a single token.
    expect(blocks[1].text).toBe("[image 4×4 · ~1 tok]");
    expect(blocks[1].tokenMethod).toBe("estimate");
    // The base64/byte payload is never stored as text.
    expect(blocks[1].text).not.toContain("iVBOR");
    r.close();
  });

  it("records a failed call and re-raises the host's own error", async () => {
    const path = tmpTrace();
    const client = new BedrockRuntimeClient({
      region: "us-east-1",
      credentials: { accessKeyId: "x", secretAccessKey: "y" },
      maxAttempts: 1,
      requestHandler: {
        handle: async () => {
          throw new TypeError("connection reset");
        },
        updateHttpClientConfig(): void {},
        httpHandlerConfigs: () => ({}),
      } as never,
    });
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(client) as BedrockRuntimeClient;
    await expect(
      wrapped.send(new ConverseCommand({ modelId: "m", messages: [] })),
    ).rejects.toThrow("connection reset");
    await tracer.close();

    const r = CTrace.open(path);
    const calls = r.getCalls();
    expect(calls).toHaveLength(1);
    expect(calls[0].error).toBe("TypeError");
    r.close();
  });
});

// --- non-Converse commands --------------------------------------------------

describe("wrap Bedrock: non-Converse commands", () => {
  it("passes an InvokeModelCommand straight through, records nothing, warns nothing", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const path = tmpTrace();
    // InvokeModel's response body is a raw provider-specific blob.
    const client = stubClient(jsonBody({ completion: "hi", stop_reason: "end_turn" }));
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(client) as BedrockRuntimeClient;

    const out = await wrapped.send(
      new InvokeModelCommand({
        modelId: "anthropic.claude-v2",
        body: JSON.stringify({ prompt: "\n\nHuman: hi\n\nAssistant:", max_tokens_to_sample: 10 }),
      }),
    );
    // The host's own result, unchanged.
    expect(JSON.parse(Buffer.from(out.body as Uint8Array).toString())).toEqual({
      completion: "hi",
      stop_reason: "end_turn",
    });
    await tracer.close();

    const r = CTrace.open(path);
    // The turn is NOT recorded: InvokeModel's body is a provider-specific
    // payload, not the Converse shape, and ctxdiff does not guess.
    expect(r.getCalls()).toHaveLength(0);
    r.close();
    // And it is SILENT: one BedrockRuntimeClient commonly serves Converse and
    // embeddings alike, so a warning per unrelated send() would be pure noise.
    expect(warn).not.toHaveBeenCalled();
  });

  it("still records the Converse calls made through the same client", async () => {
    const path = tmpTrace();
    const client = stubClient(jsonBody(CONVERSE_RESPONSE));
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(client) as BedrockRuntimeClient;
    await wrapped.send(new InvokeModelCommand({ modelId: "m", body: "{}" }));
    await wrapped.send(
      new ConverseCommand({ modelId: "m", messages: [{ role: "user", content: [{ text: "hi" }] }] }),
    );
    await wrapped.send(new InvokeModelCommand({ modelId: "m", body: "{}" }));
    await tracer.close();

    const r = CTrace.open(path);
    const calls = r.getCalls();
    expect(calls).toHaveLength(1);
    expect(r.getCallBlocks(calls[0].id).map((cb) => cb.block.text)).toEqual(["hi"]);
    r.close();
  });
});

// --- the legacy callback overload -------------------------------------------

describe("wrap Bedrock: send(command, callback)", () => {
  it("passes the legacy callback form through UNRECORDED rather than inventing a response", async () => {
    const path = tmpTrace();
    // `send(command, cb)` is a supported typed overload and it returns
    // `undefined` — the response reaches the host's CALLBACK, never the
    // interceptor. Recording it anyway would store `usage: null, latencyMs: 0`:
    // numbers no one ever observed.
    const response = { output: { message: { role: "assistant", content: [{ text: "4" }] } } };
    let sawCommand: unknown = null;
    const fake = {
      config: { serviceId: "Bedrock Runtime" },
      send(command: unknown, cb: (err: unknown, data: unknown) => void): undefined {
        sawCommand = command;
        cb(null, response);
        return undefined;
      },
    };
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(fake) as {
      send: (c: unknown, cb: (err: unknown, data: unknown) => void) => undefined;
    };

    const command = new ConverseCommand({
      modelId: "m",
      messages: [{ role: "user", content: [{ text: "what is 2+2" }] }],
    });
    let delivered: unknown = null;
    const returned = wrapped.send(command, (_err, data) => {
      delivered = data;
    });

    // Fail-open still holds: the real `send` ran with the host's own arguments,
    // and the host's callback fired with the real response.
    expect(returned).toBeUndefined();
    expect(sawCommand).toBe(command);
    expect(delivered).toBe(response);
    await tracer.close();

    const r = CTrace.open(path);
    // Nothing recorded — a missing turn is an honest gap; a turn carrying a
    // zero latency and a null usage that were never measured is a wrong answer.
    expect(r.getCalls()).toHaveLength(0);
    r.close();
  });

  it("keeps recording the promise form through the same client", async () => {
    const path = tmpTrace();
    const client = stubClient(jsonBody(CONVERSE_RESPONSE));
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(client) as BedrockRuntimeClient;
    // The options-bag overload — `args[1]` present but NOT a function — is
    // still recorded: it changes how the request is dispatched, never what
    // is in it.
    await wrapped.send(
      new ConverseCommand({ modelId: "m", messages: [{ role: "user", content: [{ text: "hi" }] }] }),
      { requestTimeout: 1000 } as never,
    );
    await tracer.close();

    const r = CTrace.open(path);
    const calls = r.getCalls();
    expect(calls).toHaveLength(1);
    expect(calls[0].usage).toEqual({ inputTokens: 12, outputTokens: 6, totalTokens: 18 });
    r.close();
  });
});

// --- streaming --------------------------------------------------------------

describe("wrap Bedrock: ConverseStreamCommand (streaming)", () => {
  it("proxies ONLY the stream member, passes every event through, and folds the trailing metadata usage", async () => {
    const path = tmpTrace();
    const client = stubClient(streamBody(), "application/vnd.amazon.eventstream");
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(client, { agent: "writer" }) as BedrockRuntimeClient;

    const out = await wrapped.send(
      new ConverseStreamCommand({
        modelId: "anthropic.claude-3-haiku-20240307-v1:0",
        system: [{ text: "Be terse." }],
        messages: [{ role: "user", content: [{ text: "say hello" }] }],
      }),
    );
    // The ENVELOPE reaches the host intact: $metadata is a real object, not a
    // stream proxy. (Wrapping the envelope itself is the bug this shape exists
    // to avoid — the host's very next line reads $metadata.)
    expect(out.$metadata.httpStatusCode).toBe(200);
    expect(out.stream).toBeDefined();

    const events: unknown[] = [];
    for await (const event of out.stream!) events.push(event);
    // Every event, in order, unchanged.
    expect(events).toEqual([
      { messageStart: { role: "assistant" } },
      { contentBlockDelta: { delta: { text: "Hello" }, contentBlockIndex: 0 } },
      { contentBlockDelta: { delta: { text: " there!" }, contentBlockIndex: 0 } },
      { contentBlockStop: { contentBlockIndex: 0 } },
      { messageStop: { stopReason: "end_turn" } },
      {
        metadata: {
          usage: { inputTokens: 12, outputTokens: 6, totalTokens: 18 },
          metrics: { latencyMs: 100 },
        },
      },
    ]);
    await tracer.close();

    const r = CTrace.open(path);
    const calls = r.getCalls();
    expect(calls).toHaveLength(1);
    expect(calls[0].agent).toBe("writer");
    // Usage came off the single trailing metadata event — the same key names a
    // non-streamed call stores.
    expect(calls[0].usage).toEqual({ inputTokens: 12, outputTokens: 6, totalTokens: 18 });
    expect(calls[0].error).toBeNull();
    expect(r.getCallBlocks(calls[0].id).map((cb) => cb.block.text)).toEqual([
      "Be terse.",
      "say hello",
    ]);
    r.close();
  });

  it("records once when the caller breaks out early, with the usage seen so far", async () => {
    const path = tmpTrace();
    const client = stubClient(streamBody(), "application/vnd.amazon.eventstream");
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(client) as BedrockRuntimeClient;

    const out = await wrapped.send(
      new ConverseStreamCommand({
        modelId: "m",
        messages: [{ role: "user", content: [{ text: "say hello" }] }],
      }),
    );
    let seen = 0;
    for await (const _event of out.stream!) {
      seen += 1;
      if (seen === 2) break; // before the metadata event ever arrives
    }
    await tracer.close();

    const r = CTrace.open(path);
    const calls = r.getCalls();
    expect(calls).toHaveLength(1);
    // Honest: no metadata event was consumed, so no usage is claimed.
    expect(calls[0].usage).toBeNull();
    expect(r.getCallBlocks(calls[0].id)).toHaveLength(1);
    r.close();
  });

  it("records a mid-stream failure as a FAILED call and re-raises it unchanged", async () => {
    const path = tmpTrace();
    // A truncated final frame: the SDK's own decoder raises part-way through.
    const truncated = streamBody().subarray(0, streamBody().length - 40);
    const client = stubClient(truncated, "application/vnd.amazon.eventstream");
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(client) as BedrockRuntimeClient;

    const out = await wrapped.send(
      new ConverseStreamCommand({
        modelId: "m",
        messages: [{ role: "user", content: [{ text: "say hello" }] }],
      }),
    );
    let raised: unknown = null;
    const events: unknown[] = [];
    try {
      for await (const event of out.stream!) events.push(event);
    } catch (err) {
      raised = err;
    }
    // The stream's own error reached the host.
    expect(raised).toBeInstanceOf(Error);
    // ...after the events that DID arrive were delivered, untouched.
    expect(events.length).toBeGreaterThan(0);
    await tracer.close();

    const r = CTrace.open(path);
    const calls = r.getCalls();
    expect(calls).toHaveLength(1);
    // Recorded as failed, not as a silently usage-less success.
    expect(calls[0].error).not.toBeNull();
    expect(r.getCallBlocks(calls[0].id)).toHaveLength(1);
    r.close();
  });

  it("hands back the host's own result when the envelope carries no stream (fail-open)", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const path = tmpTrace();
    // An error-shaped / renamed response: the declared `stream` member is
    // absent. Proxying the envelope anyway would hand the caller a stream proxy
    // where its next line reads `response.$metadata` — capture is lost here,
    // the host's call is not.
    const envelope = { $metadata: { httpStatusCode: 200 }, somethingElse: 1 };
    const fake = {
      config: { serviceId: "Bedrock Runtime" },
      send: async () => envelope,
    };
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(fake) as { send: (c: unknown) => Promise<unknown> };
    const out = await wrapped.send(new ConverseStreamCommand({ modelId: "m", messages: [] }));
    // The caller's REAL object, identically — not a copy, not a proxy.
    expect(out).toBe(envelope);
    expect((out as { $metadata: { httpStatusCode: number } }).$metadata.httpStatusCode).toBe(200);
    await tracer.close();

    const r = CTrace.open(path);
    expect(r.getCalls()).toHaveLength(0);
    r.close();
    expect(warn).not.toHaveBeenCalled();
  });
});

// --- fail-open --------------------------------------------------------------

describe("wrap Bedrock: fail-open", () => {
  it("a broken recorder never breaks the host's call, streaming or not", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const path = tmpTrace();
    const tracer = init("proj", { path });
    const broken = tracer as unknown as { ct: { recordCall: () => never } };

    const client = tracer.wrap(stubClient(jsonBody(CONVERSE_RESPONSE))) as BedrockRuntimeClient;
    broken.ct.recordCall = () => {
      throw new Error("boom");
    };
    const out = await client.send(
      new ConverseCommand({ modelId: "m", messages: [{ role: "user", content: [{ text: "hi" }] }] }),
    );
    expect(out.output?.message?.content?.[0]?.text).toBe("4");

    const streamer = tracer.wrap(
      stubClient(streamBody(), "application/vnd.amazon.eventstream"),
    ) as BedrockRuntimeClient;
    const sout = await streamer.send(
      new ConverseStreamCommand({
        modelId: "m",
        messages: [{ role: "user", content: [{ text: "hi" }] }],
      }),
    );
    const events: unknown[] = [];
    for await (const event of sout.stream!) events.push(event);
    expect(events).toHaveLength(6);
    await tracer.close();
  });

  it("an adapter whose interpretCall throws degrades to an untouched pass-through", async () => {
    const path = tmpTrace();
    const tracer = init("proj", { path });
    const sentinel = { $metadata: {}, output: "real result" };
    const client = tracer.wrap({
      config: { serviceId: "Bedrock Runtime" },
      send: async () => sentinel,
    }) as { send: (c: unknown) => Promise<unknown> };
    // A command object whose very inspection explodes.
    const hostile = new Proxy(
      {},
      {
        get(): never {
          throw new Error("nope");
        },
      },
    );
    expect(await client.send(hostile)).toBe(sentinel);
    await tracer.close();
    const r = CTrace.open(path);
    expect(r.getCalls()).toHaveLength(0);
    r.close();
  });
});
