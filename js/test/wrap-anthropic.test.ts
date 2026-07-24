import { describe, it, expect, afterEach, vi } from "vitest";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { rmSync } from "node:fs";
import Anthropic from "@anthropic-ai/sdk";
import { init } from "../src/trace.js";
import { CTrace } from "../src/store/ctrace.js";

const created: string[] = [];
function tmpTrace(): string {
  const p = join(tmpdir(), `ctxdiff-anth-${randomUUID()}.ctrace`);
  created.push(p);
  return p;
}
afterEach(() => {
  vi.restoreAllMocks();
  for (const p of created.splice(0)) {
    try {
      rmSync(p, { force: true });
    } catch {
      /* ignore */
    }
  }
});

const MESSAGE_EVENTS = [
  {
    type: "message_start",
    message: {
      id: "m",
      type: "message",
      role: "assistant",
      content: [],
      model: "claude",
      stop_reason: null,
      usage: { input_tokens: 10, output_tokens: 1 },
    },
  },
  { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
  { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "he" } },
  { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "llo" } },
  { type: "content_block_stop", index: 0 },
  { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 5 } },
  { type: "message_stop" },
];

/** Real Anthropic client with a canned HTTP stub. `capture` (if given) records
 * the exact request body the SDK sent, so tests can assert the request was
 * never mutated / no params injected. */
function stubClient(handler: (body: Record<string, unknown>) => Response, capture?: (b: unknown) => void): Anthropic {
  const fetchFn = async (_url: unknown, init?: { body?: string }) => {
    const body = init?.body ? JSON.parse(init.body) : {};
    capture?.(body);
    return handler(body);
  };
  return new Anthropic({ apiKey: "test", fetch: fetchFn as unknown as typeof fetch });
}
function jsonResponse(obj: unknown): Response {
  return new Response(JSON.stringify(obj), { headers: { "content-type": "application/json" } });
}
function sseResponse(events: { type: string }[]): Response {
  const body = events.map((e) => `event: ${e.type}\ndata: ${JSON.stringify(e)}\n\n`).join("");
  return new Response(body, { headers: { "content-type": "text/event-stream" } });
}

describe("wrap Anthropic: non-streaming messages.create", () => {
  it("records blocks/usage/params/agent and returns the response unchanged; never mutates the request", async () => {
    const path = tmpTrace();
    let sent: unknown;
    const client = stubClient(
      () =>
        jsonResponse({
          id: "msg",
          type: "message",
          role: "assistant",
          model: "claude-3-5-sonnet-20241022",
          content: [{ type: "text", text: "hi there" }],
          stop_reason: "end_turn",
          usage: { input_tokens: 10, output_tokens: 5 },
        }),
      (b) => (sent = b),
    );
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(client, { agent: "assistant-a" }) as Anthropic;

    const req = {
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 100,
      system: "be terse",
      messages: [{ role: "user" as const, content: "hi" }],
    };
    const res = await wrapped.messages.create(req);
    expect((res.content[0] as { text: string }).text).toBe("hi there");
    tracer.close();

    // Request the SDK actually sent carries ONLY the caller's fields — ctxdiff
    // injected nothing (no stream_options, no stream flag, etc.).
    expect(sent).toEqual({
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 100,
      system: "be terse",
      messages: [{ role: "user", content: "hi" }],
    });

    const r = CTrace.open(path);
    const calls = r.getCalls();
    expect(calls).toHaveLength(1);
    expect(calls[0].provider).toBe("anthropic");
    expect(calls[0].agent).toBe("assistant-a");
    expect(calls[0].usage).toEqual({ input_tokens: 10, output_tokens: 5 });
    expect(calls[0].params).toEqual({ model: "claude-3-5-sonnet-20241022", max_tokens: 100 });
    const cbs = r.getCallBlocks(calls[0].id);
    expect(cbs.map((cb) => cb.block.text)).toEqual(["be terse", "hi"]);
    expect(cbs[0].block.tokenMethod).toBe("estimate"); // non-openai -> estimate
    expect(r.getRun().models).toEqual(["claude-3-5-sonnet-20241022"]);
    r.close();
  });
});

describe("wrap Anthropic: streaming via create({stream:true})", () => {
  it("passes events through unchanged and folds usage from message_start + message_delta", async () => {
    const path = tmpTrace();
    const client = stubClient(() => sseResponse(MESSAGE_EVENTS));
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(client) as Anthropic;

    const stream = await wrapped.messages.create({
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 100,
      messages: [{ role: "user", content: "hi" }],
      stream: true,
    });
    const texts: string[] = [];
    for await (const ev of stream) {
      if (ev.type === "content_block_delta" && ev.delta.type === "text_delta") {
        texts.push(ev.delta.text);
      }
    }
    expect(texts).toEqual(["he", "llo"]);
    tracer.close();

    const r = CTrace.open(path);
    const calls = r.getCalls();
    expect(calls).toHaveLength(1);
    // input from message_start, output from message_delta
    expect(calls[0].usage).toEqual({ input_tokens: 10, output_tokens: 5 });
    r.close();
  });
});

describe("wrap Anthropic: .stream() convenience helper", () => {
  it("records the call with folded usage and delivers every event", async () => {
    const path = tmpTrace();
    const client = stubClient(() => sseResponse(MESSAGE_EVENTS));
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(client) as Anthropic;

    const stream = wrapped.messages.stream({
      model: "claude-3-5-sonnet-20241022",
      max_tokens: 100,
      messages: [{ role: "user", content: "hi" }],
    });
    let events = 0;
    for await (const _ev of stream) events++;
    expect(events).toBe(MESSAGE_EVENTS.length);
    tracer.close();

    const r = CTrace.open(path);
    const calls = r.getCalls();
    expect(calls).toHaveLength(1);
    expect(calls[0].usage).toEqual({ input_tokens: 10, output_tokens: 5 });
    r.close();
  });
});

describe("wrap Anthropic: fail-open", () => {
  it("host error is re-raised unchanged and recorded as a failed call", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const path = tmpTrace();
    const client = stubClient(
      () => new Response(JSON.stringify({ type: "error", error: { type: "overloaded_error", message: "busy" } }), { status: 529, headers: { "content-type": "application/json" } }),
    );
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(client) as Anthropic;

    await expect(
      wrapped.messages.create({
        model: "claude-3-5-sonnet-20241022",
        max_tokens: 100,
        messages: [{ role: "user", content: "hi" }],
      }),
    ).rejects.toBeInstanceOf(Error);
    tracer.close();

    const r = CTrace.open(path);
    const calls = r.getCalls();
    expect(calls).toHaveLength(1);
    expect(calls[0].error).toBeTruthy(); // recorded as a failed call
    expect(calls[0].usage).toBeNull();
    r.close();
  });

  it("host call still returns when recording throws", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const path = tmpTrace();
    const client = stubClient(() =>
      jsonResponse({
        id: "msg",
        type: "message",
        role: "assistant",
        model: "claude",
        content: [{ type: "text", text: "ok" }],
        stop_reason: "end_turn",
        usage: { input_tokens: 1, output_tokens: 1 },
      }),
    );
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(client) as Anthropic;
    const ct = (tracer as unknown as { ct: CTrace }).ct;
    (ct as unknown as { recordCall: () => never }).recordCall = () => {
      throw new Error("boom");
    };
    const res = await wrapped.messages.create({
      model: "claude",
      max_tokens: 100,
      messages: [{ role: "user", content: "hi" }],
    });
    expect((res.content[0] as { text: string }).text).toBe("ok");
    tracer.close();
  });
});
