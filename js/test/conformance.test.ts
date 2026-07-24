/**
 * THE HEADLINE TEST: cross-language conformance. A `.ctrace` written by the JS
 * SDK must open in the EXISTING Python SDK's reader and yield the same run,
 * calls, blocks, hashes and labels the JS reader sees. This is the whole point
 * of the byte-for-byte parity work — a trace captured in a JS agent opens in
 * `ctxdiff view` (Python) unmodified.
 */
import { describe, it, expect } from "vitest";
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { rmSync } from "node:fs";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { randomUUID } from "node:crypto";
import OpenAI from "openai";
import { init } from "../src/trace.js";
import { CTrace } from "../src/store/ctrace.js";
import { Recorder } from "../src/capture/recorder.js";
import { AnthropicAdapter } from "../src/capture/anthropic.js";
import { GeminiAdapter } from "../src/capture/gemini.js";
import type { Adapter } from "../src/capture/base.js";

// vitest runs with cwd = the js/ package dir; the repo root is one level up.
const repoRoot = resolve(process.cwd(), "..");
const venvPython = join(repoRoot, "venv", "bin", "python");
const pySrc = join(repoRoot, "src");

function stubClient(response: unknown): OpenAI {
  const fetchFn = async () =>
    new Response(JSON.stringify(response), {
      headers: { "content-type": "application/json" },
    });
  return new OpenAI({ apiKey: "test", fetch: fetchFn as unknown as typeof fetch });
}

describe("cross-language conformance (JS writes, Python reads)", () => {
  const hasVenv = existsSync(venvPython);

  it.skipIf(!hasVenv)(
    "a JS-written .ctrace opens in the Python reader with identical calls/blocks/hashes",
    async () => {
      const path = join(tmpdir(), `ctxdiff-conf-${randomUUID()}.ctrace`);
      try {
        // 1. Write a trace via the JS SDK: two calls, shared system block +
        //    tool schema + user message, so dedup and multiple calls are
        //    exercised across the boundary.
        const client = stubClient({
          id: "cmpl",
          object: "chat.completion",
          model: "gpt-4o",
          choices: [
            { index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" },
          ],
          usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
        });
        const tracer = init("conformance", { path });
        const wrapped = tracer.wrap(client, { agent: "planner" }) as OpenAI;
        tracer.mark("answer");
        for (const q of ["what is 2+2", "and 3+3"]) {
          await wrapped.chat.completions.create({
            model: "gpt-4o",
            tools: [{ type: "function", function: { name: "calc" } }],
            messages: [
              { role: "system", content: "be terse" },
              { role: "user", content: q },
            ],
          });
        }
        tracer.close();

        // 2. What the JS reader sees.
        const rjs = CTrace.open(path);
        const jsCalls = rjs.getCalls();
        const jsFirstBlocks = rjs
          .getCallBlocks(jsCalls[0].id)
          .map((cb) => `${cb.block.contentHash} ${cb.label}`);
        rjs.close();
        expect(jsCalls).toHaveLength(2);

        // 3. What the Python reader sees — spawned against the existing Python
        //    SDK, exactly as the spec prescribes.
        const pyScript = `
from ctxdiff.store.ctrace import CTrace
ct = CTrace.open(${JSON.stringify(path)})
print(len(ct.get_calls()), ct.get_run().provider)
for cb in ct.get_call_blocks(ct.get_calls()[0].id):
    print(cb.block.content_hash, cb.label)
`;
        const proc = spawnSync(venvPython, ["-c", pyScript], {
          encoding: "utf8",
          env: { ...process.env, PYTHONPATH: pySrc },
        });

        // Python must exit cleanly (exit 0). stderr is surfaced in the failure
        // message for diagnosis but is not itself the gate — a transient
        // interpreter warning must not fail a conformance check whose real
        // signal is "the reader opened the file and ran to completion".
        expect(
          proc.status,
          `python reader failed (status ${proc.status}):\n${proc.stderr}`,
        ).toBe(0);

        const lines = proc.stdout.trim().split("\n");
        // First line: "<n_calls> <provider>"
        expect(lines[0]).toBe("2 openai");
        // Remaining lines: the same "<hash> <label>" the JS reader produced,
        // in the same order.
        expect(lines.slice(1)).toEqual(jsFirstBlocks);
      } finally {
        rmSync(path, { force: true });
      }
    },
  );

  /**
   * Record one call for `adapter` through the real Recorder (tokens → hash →
   * label → store) into a fresh `.ctrace`, then open it in the Python venv and
   * assert Python reports the same provider and the SAME block hashes/labels the
   * JS reader produced. Proves an Anthropic/Gemini trace written by JS opens
   * byte-compatibly in the Python SDK — the whole point of the parity work.
   */
  function assertProviderConformance(
    provider: string,
    adapter: Adapter,
    kwargs: Record<string, unknown>,
    response: unknown,
  ): void {
    const path = join(tmpdir(), `ctxdiff-conf-${provider}-${randomUUID()}.ctrace`);
    try {
      const ct = CTrace.create(path, `conf-${provider}`, provider, "", new Date().toISOString());
      const rec = new Recorder(ct, adapter, null);
      rec.record({
        seq: 1,
        kwargs,
        response,
        latencyMs: 3,
        error: null,
        tagged: [],
        agent: "a1",
      });
      ct.close();

      const rjs = CTrace.open(path);
      const jsCalls = rjs.getCalls();
      const jsBlocks = rjs
        .getCallBlocks(jsCalls[0].id)
        .map((cb) => `${cb.block.contentHash} ${cb.label}`);
      rjs.close();

      const pyScript = `
from ctxdiff.store.ctrace import CTrace
ct = CTrace.open(${JSON.stringify(path)})
print(len(ct.get_calls()), ct.get_run().provider)
for cb in ct.get_call_blocks(ct.get_calls()[0].id):
    print(cb.block.content_hash, cb.label)
`;
      const proc = spawnSync(venvPython, ["-c", pyScript], {
        encoding: "utf8",
        env: { ...process.env, PYTHONPATH: pySrc },
      });
      expect(
        proc.status,
        `python reader failed (status ${proc.status}):\n${proc.stderr}`,
      ).toBe(0);
      const lines = proc.stdout.trim().split("\n");
      expect(lines[0]).toBe(`1 ${provider}`);
      expect(lines.slice(1)).toEqual(jsBlocks);
    } finally {
      rmSync(path, { force: true });
    }
  }

  it.skipIf(!hasVenv)(
    "an Anthropic .ctrace written by JS opens in the Python reader with identical hashes",
    () => {
      assertProviderConformance(
        "anthropic",
        new AnthropicAdapter(),
        {
          model: "claude-3-5-sonnet-20241022",
          max_tokens: 100,
          system: "be terse",
          tools: [{ name: "get_weather", input_schema: { type: "object" } }],
          messages: [{ role: "user", content: "hi" }],
        },
        { usage: { input_tokens: 10, output_tokens: 5 } },
      );
    },
  );

  it.skipIf(!hasVenv)(
    "a Gemini .ctrace written by JS opens in the Python reader with identical hashes",
    () => {
      assertProviderConformance(
        "gemini",
        new GeminiAdapter(),
        {
          model: "gemini-2.0-flash",
          contents: [
            { role: "user", parts: [{ text: "what is 2+2" }] },
            { role: "model", parts: [{ text: "4" }] },
          ],
          config: { systemInstruction: "be terse", temperature: 0.5 },
        },
        { usageMetadata: { promptTokenCount: 12, candidatesTokenCount: 4, totalTokenCount: 16 } },
      );
    },
  );

  // No separate "venv unavailable" case: when the venv is absent the real
  // conformance `it`s above are reported by vitest as explicit SKIPs (via
  // `it.skipIf`), which already surfaces the state honestly — a vacuous passing
  // assertion here would only disguise a skipped cross-language run as green.
});
