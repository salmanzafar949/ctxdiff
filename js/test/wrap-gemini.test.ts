import { describe, it, expect, afterEach, vi } from "vitest";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { rmSync } from "node:fs";
import { GoogleGenAI } from "@google/genai";
import { init } from "../src/trace.js";
import { CTrace } from "../src/store/ctrace.js";

const created: string[] = [];
function tmpTrace(): string {
  const p = join(tmpdir(), `ctxdiff-gem-${randomUUID()}.ctrace`);
  created.push(p);
  return p;
}

const realFetch = globalThis.fetch;
let sentBodies: unknown[] = [];

/** Install a stub for the global fetch @google/genai uses. `handler` returns the
 * canned Response; every request body is captured into `sentBodies`. */
function setFetch(handler: (body: Record<string, unknown>) => Response): void {
  sentBodies = [];
  globalThis.fetch = (async (_url: unknown, init?: { body?: string }) => {
    const body = init?.body ? JSON.parse(init.body) : {};
    sentBodies.push(body);
    return handler(body);
  }) as unknown as typeof fetch;
}

afterEach(() => {
  globalThis.fetch = realFetch;
  vi.restoreAllMocks();
  for (const p of created.splice(0)) {
    try {
      rmSync(p, { force: true });
    } catch {
      /* ignore */
    }
  }
});

function jsonResponse(obj: unknown): Response {
  return new Response(JSON.stringify(obj), { status: 200, headers: { "content-type": "application/json" } });
}
function sseResponse(chunks: unknown[]): Response {
  const body = chunks.map((o) => `data: ${JSON.stringify(o)}\r\n\r\n`).join("");
  return new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } });
}

describe("wrap Gemini: non-streaming generateContent", () => {
  it("records blocks/usage/params/agent, returns response unchanged, does not mutate the request", async () => {
    const path = tmpTrace();
    setFetch(() =>
      jsonResponse({
        candidates: [{ content: { role: "model", parts: [{ text: "4" }] }, finishReason: "STOP" }],
        usageMetadata: { promptTokenCount: 12, candidatesTokenCount: 4, totalTokenCount: 16 },
      }),
    );
    const client = new GoogleGenAI({ apiKey: "test" });
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(client, { agent: "solver" }) as GoogleGenAI;

    const req = {
      model: "gemini-2.0-flash",
      contents: "what is 2+2",
      config: { systemInstruction: "be terse", temperature: 0.5 },
    };
    const snapshot = JSON.parse(JSON.stringify(req));
    const res = await wrapped.models.generateContent(req);
    expect(res.candidates?.[0]?.content?.parts?.[0]?.text).toBe("4");
    // ctxdiff never mutates the caller's request object.
    expect(req).toEqual(snapshot);
    tracer.close();

    const r = CTrace.open(path);
    const calls = r.getCalls();
    expect(calls).toHaveLength(1);
    expect(calls[0].provider).toBe("gemini");
    expect(calls[0].agent).toBe("solver");
    // camelCase usageMetadata mapped to Python snake_case keys
    expect(calls[0].usage).toEqual({
      prompt_token_count: 12,
      candidates_token_count: 4,
      total_token_count: 16,
    });
    // sampling lifted off config; content keys dropped
    expect(calls[0].params).toEqual({ model: "gemini-2.0-flash", temperature: 0.5 });
    const cbs = r.getCallBlocks(calls[0].id);
    expect(cbs.map((cb) => [cb.block.role, cb.block.text])).toEqual([
      ["system", "be terse"],
      ["user", "what is 2+2"],
    ]);
    expect(r.getRun().models).toEqual(["gemini-2.0-flash"]);
    r.close();
  });
});

describe("wrap Gemini: streaming generateContentStream", () => {
  it("passes chunks through unchanged and folds cumulative usage (last chunk wins)", async () => {
    const path = tmpTrace();
    setFetch(() =>
      sseResponse([
        {
          candidates: [{ content: { role: "model", parts: [{ text: "he" }] } }],
          usageMetadata: { promptTokenCount: 12, candidatesTokenCount: 1, totalTokenCount: 13 },
        },
        {
          candidates: [{ content: { role: "model", parts: [{ text: "llo" }] } }],
          usageMetadata: { promptTokenCount: 12, candidatesTokenCount: 4, totalTokenCount: 16 },
        },
      ]),
    );
    const client = new GoogleGenAI({ apiKey: "test" });
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(client) as GoogleGenAI;

    const stream = await wrapped.models.generateContentStream({
      model: "gemini-2.0-flash",
      contents: "hi",
    });
    const texts: string[] = [];
    for await (const chunk of stream) {
      const t = chunk.candidates?.[0]?.content?.parts?.[0]?.text;
      if (t) texts.push(t);
    }
    expect(texts).toEqual(["he", "llo"]);
    tracer.close();

    const r = CTrace.open(path);
    const calls = r.getCalls();
    expect(calls).toHaveLength(1);
    // cumulative -> last chunk's totals, mapped to snake_case
    expect(calls[0].usage).toEqual({
      prompt_token_count: 12,
      candidates_token_count: 4,
      total_token_count: 16,
    });
    r.close();
  });
});

describe("wrap Gemini: fail-open", () => {
  it("host call still returns when recording throws", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const path = tmpTrace();
    setFetch(() =>
      jsonResponse({
        candidates: [{ content: { role: "model", parts: [{ text: "ok" }] }, finishReason: "STOP" }],
        usageMetadata: { promptTokenCount: 3, candidatesTokenCount: 1, totalTokenCount: 4 },
      }),
    );
    const client = new GoogleGenAI({ apiKey: "test" });
    const tracer = init("proj", { path });
    const wrapped = tracer.wrap(client) as GoogleGenAI;
    const ct = (tracer as unknown as { ct: CTrace }).ct;
    (ct as unknown as { recordCall: () => never }).recordCall = () => {
      throw new Error("boom");
    };
    const res = await wrapped.models.generateContent({ model: "gemini-2.0-flash", contents: "hi" });
    expect(res.candidates?.[0]?.content?.parts?.[0]?.text).toBe("ok");
    tracer.close();
  });
});

describe("wrap Gemini: Vertex AI mode", () => {
  it("captures a Vertex-mode client through the same adapter, with the same block hashes", async () => {
    // Vertex is the SAME client class in a different mode — only the endpoint
    // differs, and detection keys off the client, never the URL. So this must
    // be captured identically to AI Studio mode, hash for hash: a team moving
    // a project between the two must not see its whole context light up as new.
    const path = tmpTrace();
    const seenUrls: string[] = [];
    globalThis.fetch = (async (url: unknown, init?: { body?: string }) => {
      seenUrls.push(String(url));
      sentBodies.push(init?.body ? JSON.parse(init.body) : {});
      return jsonResponse({
        candidates: [{ content: { role: "model", parts: [{ text: "4" }] }, finishReason: "STOP" }],
        usageMetadata: { promptTokenCount: 12, candidatesTokenCount: 4, totalTokenCount: 16 },
      });
    }) as unknown as typeof fetch;
    sentBodies = [];

    const request = {
      model: "gemini-2.0-flash",
      contents: "2+2?",
      config: { systemInstruction: "be terse" },
    };
    const tracer = init("proj", { path });
    const vertex = tracer.wrap(
      new GoogleGenAI({
        vertexai: true,
        project: "my-project",
        location: "us-central1",
        // A stand-in auth client, so the SDK never reaches for Application
        // Default Credentials (which would look for a metadata server / gcloud
        // config and make this test depend on the machine it runs on).
        googleAuthOptions: {
          authClient: { getRequestHeaders: async () => new Headers() } as never,
        },
      }),
    ) as GoogleGenAI;
    const studio = tracer.wrap(new GoogleGenAI({ apiKey: "test" })) as GoogleGenAI;

    const res = await vertex.models.generateContent(request);
    expect(res.candidates?.[0]?.content?.parts?.[0]?.text).toBe("4");
    await studio.models.generateContent(request);
    tracer.close();

    // The Vertex call really did go to the Vertex endpoint, not the AI Studio one.
    expect(seenUrls[0]).toContain("aiplatform.googleapis.com");

    const ct = CTrace.open(path);
    expect(ct.getRun().provider).toBe("gemini");
    const calls = ct.getCalls();
    expect(calls).toHaveLength(2);
    expect(calls[0]!.usage).toEqual({
      prompt_token_count: 12,
      candidates_token_count: 4,
      total_token_count: 16,
    });
    expect(ct.getCallBlocks(calls[0]!.id).map((b) => b.block.contentHash)).toEqual(
      ct.getCallBlocks(calls[1]!.id).map((b) => b.block.contentHash),
    );
    ct.close();
  });
});
