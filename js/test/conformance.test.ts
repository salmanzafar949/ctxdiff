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

  // No separate "venv unavailable" case: when the venv is absent the real
  // conformance `it` above is reported by vitest as an explicit SKIP (via
  // `it.skipIf`), which already surfaces the state honestly — a vacuous passing
  // assertion here would only disguise a skipped cross-language run as green.
});
