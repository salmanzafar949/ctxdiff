import { describe, it, expect, afterEach, vi } from "vitest";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { rmSync } from "node:fs";
import OpenAI from "openai";
import { init } from "../src/trace.js";
import { CTrace } from "../src/store/ctrace.js";
import { OpenAIAdapter } from "../src/capture/openai.js";

const created: string[] = [];
function tmpTrace(): string {
  const p = join(tmpdir(), `ctxdiff-wrap-${randomUUID()}.ctrace`);
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

type FetchLike = (url: string, body: Record<string, unknown>) => Response;

/** Build a real OpenAI client whose HTTP layer is a canned stub (no network). */
function stubClient(handler: FetchLike): OpenAI {
  const fetchFn = async (url: unknown, init?: { body?: string }) => {
    const body = init?.body ? JSON.parse(init.body) : {};
    return handler(String(url), body);
  };
  return new OpenAI({ apiKey: "test", fetch: fetchFn as unknown as typeof fetch });
}

function jsonResponse(obj: unknown): Response {
  return new Response(JSON.stringify(obj), {
    headers: { "content-type": "application/json" },
  });
}
function sseResponse(events: unknown[]): Response {
  const body =
    events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join("") +
    "data: [DONE]\n\n";
  return new Response(body, { headers: { "content-type": "text/event-stream" } });
}

describe("wrap: non-streaming chat completion", () => {
  it("records blocks, usage, params and agent; returns the response unchanged", async () => {
    const path = tmpTrace();
    const client = stubClient(() =>
      jsonResponse({
        id: "cmpl-1",
        object: "chat.completion",
        model: "gpt-4o",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "hi there" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
      }),
    );
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(client, { agent: "planner" }) as OpenAI;

    const res = await wrapped.chat.completions.create({
      model: "gpt-4o",
      temperature: 0.3,
      messages: [
        { role: "system", content: "be terse" },
        { role: "user", content: "hi" },
      ],
    });
    // Host response is unchanged and fully intact.
    expect(res.choices[0].message.content).toBe("hi there");
    tracer.close();

    const r = CTrace.open(path);
    const calls = r.getCalls();
    expect(calls).toHaveLength(1);
    expect(calls[0].agent).toBe("planner");
    expect(calls[0].provider).toBe("openai");
    expect(calls[0].params).toEqual({ model: "gpt-4o", temperature: 0.3 });
    expect(calls[0].usage).toEqual({
      prompt_tokens: 5,
      completion_tokens: 2,
      total_tokens: 7,
    });
    const cbs = r.getCallBlocks(calls[0].id);
    expect(cbs.map((cb) => cb.block.text)).toEqual(["be terse", "hi"]);
    expect(cbs[0].block.tokenMethod).toBe("tiktoken");
    expect(r.getRun().models).toEqual(["gpt-4o"]);
    r.close();
  });
});

describe("wrap: streaming via create({stream:true})", () => {
  it("passes chunks through unchanged and folds usage from the final chunk", async () => {
    const path = tmpTrace();
    const client = stubClient(() =>
      sseResponse([
        {
          id: "1",
          object: "chat.completion.chunk",
          choices: [{ index: 0, delta: { content: "he" }, finish_reason: null }],
        },
        {
          id: "1",
          object: "chat.completion.chunk",
          choices: [
            { index: 0, delta: { content: "llo" }, finish_reason: "stop" },
          ],
        },
        {
          id: "1",
          object: "chat.completion.chunk",
          choices: [],
          usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
        },
      ]),
    );
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(client) as OpenAI;

    const stream = await wrapped.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "hi" }],
      stream: true,
      stream_options: { include_usage: true },
    });

    const texts: string[] = [];
    for await (const chunk of stream) {
      const delta = chunk.choices?.[0]?.delta?.content;
      if (delta) texts.push(delta);
    }
    // Chunks delivered unchanged, in order.
    expect(texts).toEqual(["he", "llo"]);
    tracer.close();

    const r = CTrace.open(path);
    const calls = r.getCalls();
    expect(calls).toHaveLength(1);
    // usage folded from the stream, recorded like a non-streaming call
    expect(calls[0].usage).toEqual({
      prompt_tokens: 5,
      completion_tokens: 2,
      total_tokens: 7,
    });
    const cbs = r.getCallBlocks(calls[0].id);
    expect(cbs.map((cb) => cb.block.text)).toEqual(["hi"]);
    r.close();
  });
});

describe("wrap: streaming snapshots the request at call time", () => {
  // Regression: the deferred streaming record used to read the host's LIVE
  // request object at finalize (after the stream drained), so a host that
  // mutated its own messages array mid-stream (rewrite the turn, append an
  // assistant placeholder to fill while streaming, reuse it next turn) got
  // blocks recorded that were NEVER sent in this call. The interceptor now
  // deep-clones the request synchronously at call time, so finalize records
  // exactly what went out. FAILS pre-fix (records MUTATED/INJECTED), passes now.
  it("records what was SENT even if the host mutates the array before consuming the stream", async () => {
    const path = tmpTrace();
    const client = stubClient(() =>
      sseResponse([
        {
          id: "1",
          object: "chat.completion.chunk",
          choices: [{ index: 0, delta: { content: "ok" }, finish_reason: "stop" }],
        },
        {
          id: "1",
          object: "chat.completion.chunk",
          choices: [],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        },
      ]),
    );
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(client) as OpenAI;

    // The host's LIVE array — the object it will mutate mid-stream.
    const messages: { role: "user" | "assistant"; content: string }[] = [
      { role: "user", content: "ORIGINAL" },
    ];
    // A tag whose needle points at the ORIGINAL text: proves needle-matching
    // still resolves against the snapshot (not the mutated live ref).
    tracer.tag("the-question", ["ORIGINAL"]);

    const stream = await wrapped.chat.completions.create({
      model: "gpt-4o",
      messages,
      stream: true,
      stream_options: { include_usage: true },
    });

    // Host mutates its own request BEFORE consuming the stream.
    messages[0].content = "MUTATED";
    messages.push({ role: "assistant", content: "INJECTED" });

    for await (const _chunk of stream) {
      void _chunk;
    }
    tracer.close();

    const r = CTrace.open(path);
    const calls = r.getCalls();
    expect(calls).toHaveLength(1);
    const cbs = r.getCallBlocks(calls[0].id);
    // Exactly what was sent — no MUTATED text, no INJECTED assistant block.
    expect(cbs.map((cb) => cb.block.text)).toEqual(["ORIGINAL"]);
    // tag() needle matched the snapshot's text.
    expect(cbs[0].label).toBe("the-question");
    expect(cbs[0].labelSource).toBe("tagged");
    r.close();
  });
});

describe("wrap: NON-streaming snapshots the request at call time too", () => {
  // Regression, and the twin of the streaming case above. The snapshot used to
  // be taken only `if (streaming)`, on the reasoning that a non-streaming call
  // "records synchronously in `.then`, before the host resumes". That is not
  // what `.then` does: it runs on a later microtask, AFTER an await boundary, so
  // every other branch of the host's fan-out has already had its turn. A
  // concurrent branch that appends to the SAME messages array while the request
  // is in flight was therefore recorded into that call — content the model never
  // saw, in a file whose only job is to be evidence of what was sent.
  //
  // The snapshot is now unconditional. This is provider-agnostic plumbing in
  // trace.ts, so it is pinned here on OpenAI as well as on Bedrock (where the
  // defect was found) — see wrap-bedrock.test.ts.
  it("a concurrent mutation during the in-flight await is NOT recorded", async () => {
    const path = tmpTrace();
    // The host's shared, live array — the object a concurrent branch appends to.
    const messages: { role: "user" | "assistant"; content: string }[] = [
      { role: "user", content: "ORIGINAL" },
    ];
    // The stub resolves only after the concurrent branch has run: it yields to
    // the event loop, which is exactly the window a real network call opens.
    const client = stubClient(() =>
      jsonResponse({
        id: "cmpl-1",
        object: "chat.completion",
        model: "gpt-4o",
        choices: [
          { index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
      }),
    );
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(client) as OpenAI;

    const inFlight = wrapped.chat.completions.create({ model: "gpt-4o", messages });
    // A second async branch mutating the shared array WHILE the call is in
    // flight — the normal shape of an agent fan-out sharing one history.
    const mutator = (async () => {
      await Promise.resolve();
      messages.push({ role: "assistant", content: "INJECTED-DURING-FLIGHT" });
      messages[0].content = "MUTATED-DURING-FLIGHT";
    })();
    await Promise.all([inFlight, mutator]);
    tracer.close();

    const r = CTrace.open(path);
    const calls = r.getCalls();
    expect(calls).toHaveLength(1);
    const cbs = r.getCallBlocks(calls[0].id);
    // Exactly the one block that went on the wire.
    expect(cbs.map((cb) => cb.block.text)).toEqual(["ORIGINAL"]);
    r.close();
  });

  it("falls back to the live request when it is not structured-cloneable", async () => {
    const path = tmpTrace();
    const client = stubClient(() =>
      jsonResponse({
        id: "cmpl-1",
        object: "chat.completion",
        model: "gpt-4o",
        choices: [
          { index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
      }),
    );
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(client) as OpenAI;

    // A function-valued member makes structuredClone throw. Fail-open: capture
    // degrades to the live reference rather than the call failing.
    const params = {
      model: "gpt-4o",
      messages: [{ role: "user" as const, content: "hi" }],
      onUncloneable: () => undefined,
    };
    const res = await wrapped.chat.completions.create(params as never);
    expect(res.choices[0].message.content).toBe("ok");
    tracer.close();

    const r = CTrace.open(path);
    const calls = r.getCalls();
    expect(calls).toHaveLength(1);
    expect(r.getCallBlocks(calls[0].id).map((cb) => cb.block.text)).toEqual(["hi"]);
    r.close();
  });
});

describe("wrap: .stream() convenience helper", () => {
  it("records the call with folded usage and delivers every chunk", async () => {
    const path = tmpTrace();
    const client = stubClient(() =>
      sseResponse([
        {
          id: "1",
          object: "chat.completion.chunk",
          choices: [
            { index: 0, delta: { role: "assistant", content: "hi" }, finish_reason: "stop" },
          ],
        },
        {
          id: "1",
          object: "chat.completion.chunk",
          choices: [],
          usage: { prompt_tokens: 9, completion_tokens: 1, total_tokens: 10 },
        },
      ]),
    );
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(client) as OpenAI;

    const stream = wrapped.chat.completions.stream({
      model: "gpt-4o",
      messages: [{ role: "user", content: "hi" }],
      stream_options: { include_usage: true },
    });

    let chunkCount = 0;
    for await (const _chunk of stream) chunkCount++;
    expect(chunkCount).toBe(2);
    tracer.close();

    const r = CTrace.open(path);
    const calls = r.getCalls();
    expect(calls).toHaveLength(1);
    expect(calls[0].usage).toEqual({
      prompt_tokens: 9,
      completion_tokens: 1,
      total_tokens: 10,
    });
    r.close();
  });
});

describe("wrap: responses API", () => {
  it("records instructions/input blocks and input_tokens usage", async () => {
    const path = tmpTrace();
    const client = stubClient(() =>
      jsonResponse({
        id: "resp-1",
        object: "response",
        status: "completed",
        model: "gpt-4o",
        output: [],
        usage: { input_tokens: 8, output_tokens: 3, total_tokens: 11 },
      }),
    );
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(client) as OpenAI;

    await wrapped.responses.create({
      model: "gpt-4o",
      instructions: "you are a bot",
      input: "hello",
    });
    tracer.close();

    const r = CTrace.open(path);
    const calls = r.getCalls();
    expect(calls).toHaveLength(1);
    expect(calls[0].usage).toEqual({
      input_tokens: 8,
      output_tokens: 3,
      total_tokens: 11,
    });
    const cbs = r.getCallBlocks(calls[0].id);
    expect(cbs.map((cb) => [cb.block.role, cb.block.text])).toEqual([
      ["system", "you are a bot"],
      ["user", "hello"],
    ]);
    r.close();
  });
});

describe("wrap: fail-open guarantees", () => {
  it("host call still returns when recording throws", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const path = tmpTrace();
    const client = stubClient(() =>
      jsonResponse({
        id: "cmpl",
        object: "chat.completion",
        choices: [
          { index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      }),
    );
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(client) as OpenAI;

    // Sabotage the store so recordCall throws deep inside the recorder.
    const ct = (tracer as unknown as { ct: CTrace }).ct;
    (ct as unknown as { recordCall: () => never }).recordCall = () => {
      throw new Error("boom");
    };

    const res = await wrapped.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "hi" }],
    });
    // Host is untouched despite the recorder blowing up.
    expect(res.choices[0].message.content).toBe("ok");
    tracer.close();
  });

  it("every chunk is still delivered when usage accumulation throws", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    // Make the adapter's per-chunk accumulator throw for this test only.
    const spy = vi
      .spyOn(OpenAIAdapter.prototype, "accumulateStreamUsage")
      .mockImplementation(() => {
        throw new Error("accumulate boom");
      });
    const path = tmpTrace();
    const client = stubClient(() =>
      sseResponse([
        { id: "1", object: "chat.completion.chunk", choices: [{ index: 0, delta: { content: "a" }, finish_reason: null }] },
        { id: "1", object: "chat.completion.chunk", choices: [{ index: 0, delta: { content: "b" }, finish_reason: "stop" }] },
      ]),
    );
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(client) as OpenAI;
    const stream = await wrapped.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "hi" }],
      stream: true,
    });
    const texts: string[] = [];
    for await (const chunk of stream) {
      const d = chunk.choices?.[0]?.delta?.content;
      if (d) texts.push(d);
    }
    // Despite the accumulator throwing on every chunk, all chunks reached us.
    expect(texts).toEqual(["a", "b"]);
    expect(spy).toHaveBeenCalled();
    tracer.close();

    // Call still recorded (with no usage, since accumulation failed).
    const r = CTrace.open(path);
    expect(r.getCalls()).toHaveLength(1);
    expect(r.getCalls()[0].usage).toBeNull();
    r.close();
  });

  it("an unrecognized client is returned unwrapped (no throw)", () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const tracer = init("proj", { path: tmpTrace() });
    const notAClient = { foo: "bar" };
    const out = tracer.wrap(notAClient);
    expect(out).toBe(notAClient);
    tracer.close();
  });
});
