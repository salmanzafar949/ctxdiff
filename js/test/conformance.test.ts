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
import { countTokens } from "../src/tokenize.js";
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
    "a JS-written MULTI-SESSION project .ctrace lists identical sessions/calls in the Python reader",
    async () => {
      const path = join(tmpdir(), `ctxdiff-conf-multi-${randomUUID()}.ctrace`);
      try {
        // Two separate init()s APPEND two sessions to ONE project file — the
        // project-scoped model. Session A: one call under agent "planner";
        // session B: two calls under agent "worker".
        const mkStub = () =>
          stubClient({
            id: "cmpl",
            object: "chat.completion",
            model: "gpt-4o",
            choices: [
              { index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" },
            ],
            usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
          });

        const tA = init("proj", { path });
        const wA = tA.wrap(mkStub(), { agent: "planner" }) as OpenAI;
        await wA.chat.completions.create({
          model: "gpt-4o",
          messages: [{ role: "user", content: "session A q1" }],
        });
        tA.close();

        const tB = init("proj", { path });
        const wB = tB.wrap(mkStub(), { agent: "worker" }) as OpenAI;
        for (const q of ["session B q1", "session B q2"]) {
          await wB.chat.completions.create({
            model: "gpt-4o",
            messages: [{ role: "user", content: q }],
          });
        }
        tB.close();

        // What the JS reader sees: two sessions, oldest-first, with turn counts
        // and agents. Emit a stable, comparable projection.
        const rjs = CTrace.open(path);
        const jsSessions = rjs
          .listSessions()
          .map((s) => `${s.turnCount} ${s.provider} ${s.agents.join(",")}`);
        rjs.close();
        expect(jsSessions).toEqual(["1 openai planner", "2 openai worker"]);

        // The Python reader must list the SAME sessions (oldest-first) with the
        // same per-session call counts and agent sets — cross-language proof the
        // multi-session project file is byte-compatible.
        const pyScript = `
from ctxdiff.store.ctrace import CTrace
ct = CTrace.open(${JSON.stringify(path)})
for s in ct.list_sessions():
    print(s.turn_count, s.provider, ",".join(s.agents))
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
        expect(lines).toEqual(jsSessions);
      } finally {
        for (const suffix of ["", "-wal", "-shm"]) {
          rmSync(path + suffix, { force: true });
        }
      }
    },
  );

  it.skipIf(!hasVenv)(
    "a Python-written MULTI-SESSION project .ctrace lists identical sessions in the JS reader (vice-versa)",
    () => {
      const path = join(tmpdir(), `ctxdiff-conf-pymulti-${randomUUID()}.ctrace`);
      try {
        // Python appends two sessions to one project file via its own project-
        // scoped write path (open_or_create_session), mirroring the JS direction.
        const pyScript = `
import datetime
from ctxdiff.store.ctrace import CTrace
def sess(agent, n):
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    ct = CTrace.open_or_create_session(${JSON.stringify(path)}, project="proj", provider="openai", model="", started_at=now)
    for s in range(1, n + 1):
        ct.record_call(seq=s, params={"model": "gpt-4o"}, usage=None, latency_ms=1, error=None, call_blocks=[], agent=agent)
    ct.close()
sess("planner", 1)
sess("worker", 2)
`;
        const proc = spawnSync(venvPython, ["-c", pyScript], {
          encoding: "utf8",
          env: { ...process.env, PYTHONPATH: pySrc },
        });
        expect(
          proc.status,
          `python writer failed (status ${proc.status}):\n${proc.stderr}`,
        ).toBe(0);

        // The JS reader lists the same two sessions, oldest-first, with matching
        // per-session turn counts and agents.
        const ct = CTrace.open(path);
        const sessions = ct.listSessions();
        expect(
          sessions.map((s) => `${s.turnCount} ${s.provider} ${s.agents.join(",")}`),
        ).toEqual(["1 openai planner", "2 openai worker"]);
        // Session-less JS read defaults to the newest (Python's second session).
        expect(ct.getCalls()).toHaveLength(2);
        for (const c of ct.getCalls()) expect(c.agent).toBe("worker");
        ct.close();
      } finally {
        for (const suffix of ["", "-wal", "-shm"]) {
          rmSync(path + suffix, { force: true });
        }
      }
    },
  );

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

/**
 * Cross-language TOKENIZER conformance, asserted on the pair `(count, method)`
 * rather than on the count alone.
 *
 * The method is half the promise. A block counted at 9 by one SDK and estimated
 * at 5 by the other is an obvious divergence; a block estimated at 9 by one and
 * counted at 9 by the other is a silent one — the numbers agree today and drift
 * apart the moment the text changes, and only one of the two reports is honest
 * about being approximate. Both are compared here.
 *
 * The battery is built from literal special-token spellings on purpose. That is
 * where the two SDKs used to disagree in two independent ways: tiktoken's
 * `o200k_base` reserves only `<|endoftext|>` and `<|endofprompt|>`, while
 * gpt-tokenizer's guard rejects the whole `<|...|>` family, so `<|im_start|>`
 * was exact in Python and an estimate in JS; and Python additionally LATCHED
 * the refusal process-wide, so every later count in a Python capture degraded
 * to an estimate while JS kept counting exactly. Both SDKs now switch the guard
 * off (`disallowed_special=()` / `disallowedSpecial: new Set()`) and count the
 * literals as the plain text the API actually delivers to the model.
 *
 * The trailing ordinary strings are not padding: they are the neighbours. They
 * are compared AFTER the special-token entries in the same interpreter and the
 * same Node process, so a per-text refusal that leaked back into either
 * encoder's cached state would show up here as an estimate.
 */
describe("cross-language conformance (token counts AND methods agree)", () => {
  const hasVenv = existsSync(venvPython);

  it.skipIf(!hasVenv)(
    "Python and JS return identical (count, method) for special-token literals",
    () => {
      const probes = [
        "hello world",
        "a <|endoftext|> b",
        "<|endoftext|>",
        "<|endofprompt|>",
        "<|im_start|>system<|im_end|>",
        "<|fim_prefix|>def f():<|fim_suffix|>",
        "<|not_a_real_token|>",
        "<|endoftext| with no closing pipe",
        'prompt = "summarize" + "<|endoftext|>" + user_input',
        "an ordinary sentence counted after every literal above",
        "🚀 ship it — unicode after a special token still counts exactly",
      ];

      // The JS half, in one pass over the battery.
      const js = probes.map((t) => countTokens(t, "openai"));

      // The Python half, spawned against the venv SDK. The probes are handed
      // over as a JSON argument rather than interpolated into source, so a
      // quote or backslash in a probe cannot change the program being run.
      const pyScript = `
import json, sys
from ctxdiff.tokenize.counter import count_tokens
for text in json.loads(sys.argv[1]):
    count, method = count_tokens(text, "openai")
    print(count, method)
`;
      const proc = spawnSync(venvPython, ["-c", pyScript, JSON.stringify(probes)], {
        encoding: "utf8",
        env: { ...process.env, PYTHONPATH: pySrc },
      });
      expect(
        proc.status,
        `python tokenizer failed (status ${proc.status}):\n${proc.stderr}`,
      ).toBe(0);

      const py = proc.stdout
        .trim()
        .split("\n")
        .map((line) => {
          const [count, method] = line.split(" ");
          return [Number(count), method] as [number, string];
        });

      expect(py).toHaveLength(probes.length);
      // Compared as one array so a failure names every disagreeing probe at
      // once instead of stopping at the first.
      expect(js).toEqual(py);

      // ...and every one of them is EXACT. Asserted separately because
      // `js.toEqual(py)` would also be satisfied by both SDKs estimating
      // everything — the vacuous-agreement failure mode.
      expect(js.map(([, method]) => method)).toEqual(probes.map(() => "tiktoken"));
    },
  );
});
