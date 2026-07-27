/**
 * The 2026-07-27 dogfood fixes, JS side — mirrors Python's
 * `test_provider_label.py` plus the tokens/store additions:
 *
 * 1. OpenAI-compatible endpoint ATTRIBUTION: an OpenAI-SDK client pointed at
 *    Gemini's/Anthropic's compat endpoint records that vendor's name on the
 *    run and each call (unknown hosts record "openai-compatible"); the
 *    adapter — and therefore capture mechanics — stays openai.
 * 2. `usageTotals` counts OpenAI-chat streams sent without
 *    `stream_options.include_usage` so the renderer can NAME the remedy for
 *    missing usage instead of dead-ending.
 * 3. `recordCall` passively checkpoints the WAL, so copying the bare
 *    `.ctrace` from a still-open store (a server that never close()s) ships
 *    a complete trace.
 */
import { describe, it, expect, afterEach } from "vitest";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { copyFileSync, rmSync } from "node:fs";
import OpenAI from "openai";
import { init } from "../src/trace.js";
import { CTrace } from "../src/store/ctrace.js";
import { usageTotals } from "../src/analyze/tokens.js";
import { renderUsageSummary } from "../src/render.js";
import type { Block, Call, CallBlock } from "../src/models.js";

const created: string[] = [];
function tmpPath(name: string): string {
  const p = join(tmpdir(), `ctxdiff-dogfood-${randomUUID()}-${name}`);
  created.push(p);
  return p;
}
afterEach(() => {
  for (const p of created.splice(0)) {
    for (const suffix of ["", "-wal", "-shm"]) {
      try {
        rmSync(p + suffix, { force: true });
      } catch {
        /* ignore */
      }
    }
  }
});

/** A real OpenAI client with a canned HTTP layer, pointable at any baseURL. */
function stubClient(baseURL?: string): OpenAI {
  const fetchFn = async () =>
    new Response(
      JSON.stringify({
        id: "cmpl-1",
        object: "chat.completion",
        model: "gemini-2.5-flash",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "hi there" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
      }),
      { headers: { "content-type": "application/json" } },
    );
  return new OpenAI({
    apiKey: "test",
    ...(baseURL ? { baseURL } : {}),
    fetch: fetchFn as unknown as typeof fetch,
  });
}

async function traceOneCall(baseURL?: string): Promise<{ run: string; call: string | null }> {
  const path = tmpPath("label.ctrace");
  const tracer = init("proj", { path });
  const wrapped = tracer.wrap(stubClient(baseURL)) as OpenAI;
  await wrapped.chat.completions.create({
    model: "gemini-2.5-flash",
    messages: [{ role: "user", content: "hi" }],
  });
  tracer.close();
  const ct = CTrace.open(path);
  try {
    return { run: ct.getRun().provider, call: ct.getCalls()[0].provider };
  } finally {
    ct.close();
  }
}

describe("openai-compatible endpoint attribution", () => {
  it("records gemini on run and call for Gemini's compat endpoint", async () => {
    const { run, call } = await traceOneCall(
      "https://generativelanguage.googleapis.com/v1beta/openai/",
    );
    expect(run).toBe("gemini");
    expect(call).toBe("gemini");
  });

  it("records anthropic for Anthropic's compat endpoint", async () => {
    const { run, call } = await traceOneCall("https://api.anthropic.com/v1/");
    expect(run).toBe("anthropic");
    expect(call).toBe("anthropic");
  });

  it("records openai-compatible for an unrecognized OSS host", async () => {
    const { run, call } = await traceOneCall("http://localhost:11434/v1");
    expect(run).toBe("openai-compatible");
    expect(call).toBe("openai-compatible");
  });

  it("keeps the historical openai label for the default client", async () => {
    const { run, call } = await traceOneCall();
    expect(run).toBe("openai");
    expect(call).toBe("openai");
  });
});

function usageCall(seq: number, params: Record<string, unknown>, usage: Record<string, unknown> | null = null): Call {
  return {
    id: `c-${seq}`,
    runId: "run",
    seq,
    params,
    usage,
    latencyMs: 10,
    error: null,
    agent: null,
    step: null,
    provider: null,
  };
}

describe("streamed-without-include_usage diagnosis", () => {
  it("counts OpenAI-chat streams missing include_usage and names the remedy", () => {
    const totals = usageTotals([
      usageCall(1, {
        model: "gpt-4o",
        stream: true,
        messages: [{ role: "user", content: "hi" }],
      }),
    ]);
    expect(totals.callsWithUsage).toBe(0);
    expect(totals.streamedWithoutUsage).toBe(1);
    const summary = renderUsageSummary(totals);
    expect(summary).toContain("no provider usage reported");
    expect(summary).toContain("1 streamed call recorded no usage");
    expect(summary).toContain('stream_options={"include_usage": true}');
  });

  it("does not count calls that reported usage or opted in", () => {
    const totals = usageTotals([
      usageCall(
        1,
        {
          model: "gpt-4o",
          stream: true,
          stream_options: { include_usage: true },
          messages: [{ role: "user", content: "hi" }],
        },
        { prompt_tokens: 5, completion_tokens: 2 },
      ),
    ]);
    expect(totals.streamedWithoutUsage).toBe(0);
    expect(renderUsageSummary(totals)).not.toContain("streamed call");
  });

  it("ignores non-streamed and non-chat-shaped calls", () => {
    const totals = usageTotals([
      usageCall(1, { model: "gpt-4o", messages: [{ role: "user", content: "hi" }] }),
      usageCall(2, { model: "gemini-2.5-flash", stream: true, contents: [{ parts: [{ text: "hi" }] }] }),
    ]);
    expect(totals.streamedWithoutUsage).toBe(0);
    expect(renderUsageSummary(totals)).not.toContain("streamed call");
  });
});

function callBlock(text: string, position: number): CallBlock {
  const block: Block = {
    contentHash: `h-${text}`,
    role: "user",
    kind: "message",
    text,
    tokenCount: text.length,
    tokenMethod: "tiktoken",
  };
  return { block, position, label: "user", labelSource: "heuristic" };
}

describe("live-store WAL freshness", () => {
  it("a copy of the bare .ctrace taken while the store is open contains the calls", () => {
    const path = tmpPath("live.ctrace");
    const ct = CTrace.create(path, "agent", "openai", "gpt-4o");
    ct.recordCall({
      seq: 1,
      params: { model: "gpt-4o" },
      usage: null,
      latencyMs: 10,
      error: null,
      callBlocks: [callBlock("hello world", 0)],
    });
    ct.recordCall({
      seq: 2,
      params: { model: "gpt-4o" },
      usage: null,
      latencyMs: 10,
      error: null,
      callBlocks: [callBlock("hello world", 0)],
    });

    // Copy ONLY the main file — deliberately NOT close()ing first and NOT
    // copying the -wal/-shm sidecars: exactly what a user sharing a trace
    // from a still-running server does (dogfood finding 2026-07-27).
    const copy = tmpPath("shared.ctrace");
    copyFileSync(path, copy);
    ct.close();

    const reader = CTrace.open(copy);
    try {
      // The checkpoint is throttled (≥1 per second), so the copy may lag the
      // WAL by up to that interval — the guarantee under test is that the
      // bare file is never the pre-fix EMPTY SHELL: the first call always
      // checkpoints, so at least one call must be visible.
      expect(reader.getCalls().length).toBeGreaterThanOrEqual(1);
    } finally {
      reader.close();
    }
  });
});
