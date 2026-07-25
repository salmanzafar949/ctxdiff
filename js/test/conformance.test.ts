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
import { Readable } from "node:stream";
import OpenAI from "openai";
import { BedrockRuntimeClient, ConverseCommand } from "@aws-sdk/client-bedrock-runtime";
import { init } from "../src/trace.js";
import { CTrace } from "../src/store/ctrace.js";
import { Recorder } from "../src/capture/recorder.js";
import { AnthropicAdapter } from "../src/capture/anthropic.js";
import { GeminiAdapter } from "../src/capture/gemini.js";
import { countTokens } from "../src/tokenize.js";
import { imageRawBlock } from "../src/images.js";
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

/**
 * Cross-language conformance for the ESTIMATE path — the OTHER tokenizer, and
 * the one that covers MORE of ctxdiff: openai is the only provider counted
 * exactly, so bedrock, anthropic and gemini traces are all rendered from
 * `_estimate_count`/`estimateCount`. Until this test existed, nothing compared
 * the two implementations of it across the language boundary.
 *
 * The battery is astral-plane text on purpose, because that is the ONE input on
 * which the two used to disagree. Both divide a character count by four, but JS
 * strings are UTF-16 and `text.length` counts CODE UNITS, so every character
 * outside the BMP — emoji, ZWJ sequences, regional-indicator flags, skin-tone
 * modifiers, math alphanumerics, CJK ext-B — counted DOUBLE against Python's
 * `len()`, which counts code points. A Converse system block of
 * `Répondez en français 🇫🇷` was 7 tokens in JS and 6 in Python: same trace, same
 * hash, different rendered number depending on which SDK wrote it. Hashes were
 * never at risk (token counts are not part of the hashed tuple), but `ctxdiff
 * tokens`, the cache profiler's re-billed totals and the dashboard's percentages
 * all were — which is precisely the class of divergence the pinned tokenizers
 * and the golden corpus exist to catch, and this one slipped past both because
 * no golden fixture had astral text on an estimate provider.
 *
 * Run across all three estimate providers, not just one, so a future
 * per-provider branch in either counter cannot diverge unnoticed.
 */
describe("cross-language conformance (estimate counts agree on astral text)", () => {
  const hasVenv = existsSync(venvPython);

  it.skipIf(!hasVenv)("Python and JS return identical (count, method) for astral text", () => {
    const probes = [
      "Répondez en français 🇫🇷", // the reviewer's repro: a Converse system block
      "🚀", // a single astral character: 1 code point, 2 UTF-16 units
      "👨‍👩‍👧‍👦", // ZWJ family — four astral emoji joined by three ZWJs
      "🏳️‍🌈 🏴󠁧󠁢󠁳󠁣󠁴󠁿", // flag sequences, incl. a tag-sequence flag (all astral tags)
      "🤦🏽‍♀️ 👍🏿", // skin-tone modifiers (themselves astral)
      "𝕌𝕟𝕚𝕔𝕠𝕕𝕖 𝓶𝓪𝓽𝓱 𝔞𝔩𝔭𝔥𝔞", // math alphanumerics
      "𠜎 𤭢 𰻞", // CJK extension B/G
      "🚀".repeat(64), // long enough that the /4 divide, not the max(1,…), decides
      "plain ascii with no astral characters at all",
      "combining é vs precomposed é — BMP only, must be unchanged by the fix",
      "日本語のトークン化テスト", // BMP CJK: a control that must not move
      "", // empty stays zero under either counter
    ];
    const providers = ["bedrock", "anthropic", "gemini"];

    const js = providers.flatMap((p) => probes.map((t) => countTokens(t, p)));

    // Probes AND providers are handed over as JSON arguments rather than
    // interpolated into source, so nothing in a probe can change the program.
    const pyScript = `
import json, sys
from ctxdiff.tokenize.counter import count_tokens
probes = json.loads(sys.argv[1])
for provider in json.loads(sys.argv[2]):
    for text in probes:
        count, method = count_tokens(text, provider)
        print(count, method)
`;
    const proc = spawnSync(
      venvPython,
      ["-c", pyScript, JSON.stringify(probes), JSON.stringify(providers)],
      { encoding: "utf8", env: { ...process.env, PYTHONPATH: pySrc } },
    );
    expect(proc.status, `python tokenizer failed (status ${proc.status}):\n${proc.stderr}`).toBe(0);

    const py = proc.stdout
      .trim()
      .split("\n")
      .map((line) => {
        const [count, method] = line.split(" ");
        return [Number(count), method] as [number, string];
      });

    expect(py).toHaveLength(js.length);
    expect(js).toEqual(py);

    // ...and every one of them really went through the ESTIMATE path, so the
    // agreement above cannot be the vacuous kind (both SDKs exact-counting).
    expect(js.map(([, method]) => method)).toEqual(js.map(() => "estimate"));

    // The specific number from the bug report, pinned: `Répondez en français 🇫🇷`
    // is 23 code points (the flag is two regional indicators) -> ceil(23/4) = 6.
    // Under `.length` it was 25 UTF-16 units -> ceil(25/4) = 7.
    expect(countTokens("Répondez en français 🇫🇷", "bedrock")).toEqual([6, "estimate"]);
  });
});


/**
 * Cross-language IMAGE conformance — the fourth and newest byte-for-byte
 * promise, and the one with the most moving parts behind it.
 *
 * An image block is not simply hashed and counted like a text block. Four
 * derived values have to match across the two SDKs, and each is computed by
 * independent code in each language:
 *
 *   1. the DESCRIPTOR text (`[image 1024×768 · ~765 tok]`), including its
 *      thousands rounding, which uses `floor(n/100 + 0.5)` on both sides
 *      precisely because Python rounds half-to-even and JS rounds half-up;
 *   2. the HASH INPUT — a sha256 over the decoded image bytes, which means the
 *      two base64 decoders must agree on padding, whitespace and the URL-safe
 *      alphabet before either hash is taken;
 *   3. the TOKEN ESTIMATE, from three different published provider formulas
 *      re-implemented in each language with integer-only arithmetic;
 *   4. the TOKEN METHOD, which must be `estimate` on both sides — the honesty
 *      claim, and the one a vacuous agreement could hide.
 *
 * The battery deliberately spans every provider shape, all four sniffable
 * formats, both degradations (unknown format, remote URL) and every `detail`
 * value, so a divergence in any single branch of any of the four surfaces
 * fails here rather than in a user's trace.
 */
describe("cross-language conformance (image blocks agree byte for byte)", () => {
  const hasVenv = existsSync(venvPython);

  it.skipIf(!hasVenv)(
    "Python and JS produce identical descriptor, hash input, estimate and method",
    () => {
      // The bytes are built ONCE, here, and shipped to Python as base64 inside
      // the probe payload. Rebuilding them on each side would test two image
      // generators rather than two image readers.
      const b64 = (bytes: number[]) => Buffer.from(bytes).toString("base64");
      const pngBytes = (w: number, h: number) => {
        // Signature + IHDR only: the sniffer reads nothing past offset 24, and a
        // header is all that is needed to exercise it.
        const be32 = (n: number) => [(n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255];
        return [
          0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
          0, 0, 0, 13, 0x49, 0x48, 0x44, 0x52,
          ...be32(w), ...be32(h), 8, 2, 0, 0, 0,
        ];
      };
      const gifBytes = (w: number, h: number) => [
        0x47, 0x49, 0x46, 0x38, 0x39, 0x61,
        w & 255, (w >>> 8) & 255, h & 255, (h >>> 8) & 255,
        0xf0, 0, 0, 0, 0, 0, 0, 0, 0, 0x3b,
      ];
      const png = b64(pngBytes(1024, 768));
      const wide = b64(pngBytes(2000, 1200));
      const gif = b64(gifBytes(640, 480));
      const bmp = b64([0x42, 0x4d, ...new Array(60).fill(0)]);
      // A structurally valid IHDR declaring 18 exapixels — the truncated /
      // fuzzed / hostile header both SDKs must refuse to believe, in step.
      const monster = b64(pngBytes(0xffffffff, 0xffffffff));

      // [provider, part] pairs. Every provider shape, every degradation.
      const probes: [string, unknown][] = [
        ["openai", { type: "image_url", image_url: { url: `data:image/png;base64,${png}`, detail: "high" } }],
        ["openai", { type: "image_url", image_url: { url: `data:image/png;base64,${png}`, detail: "low" } }],
        ["openai", { type: "image_url", image_url: { url: `data:image/png;base64,${png}` } }],
        ["openai", { type: "image_url", image_url: { url: `data:image/png;base64,${wide}`, detail: "auto" } }],
        ["openai", { type: "image_url", image_url: `data:image/png;base64,${png}` }],
        ["openai", { type: "image_url", image_url: { url: "https://cdn.example.com/a.png" } }],
        ["openai", { type: "image_url", image_url: { url: "https://cdn.example.com/a.png", detail: "low" } }],
        ["openai", { type: "input_image", image_url: `data:image/png;base64,${png}`, detail: "high" }],
        ["openai", { type: "input_image", file_id: "file-3d9a17c04be84e2fb0c5" }],
        ["openai", { type: "image_url", image_url: { url: `data:image/bmp;base64,${bmp}` } }],
        ["anthropic", { type: "image", source: { type: "base64", media_type: "image/png", data: png } }],
        ["anthropic", { type: "image", source: { type: "base64", media_type: "image/png", data: png.replace(/=+$/, "") } }],
        ["anthropic", { type: "image", source: { type: "url", url: "https://cdn.example.com/a.png" } }],
        ["anthropic", { type: "image", source: { type: "file", file_id: "file_011CQrsTuVwXyZ" } }],
        ["gemini", { inline_data: { mime_type: "image/gif", data: gif } }],
        ["gemini", { inlineData: { mimeType: "image/gif", data: gif } }],
        ["gemini", { inline_data: { mime_type: "audio/wav", data: gif } }],
        ["gemini", { file_data: { mime_type: "image/jpeg", file_uri: "https://gen.googleapis.com/v1/files/7k2m" } }],
        ["bedrock", { image: { format: "png", source: { bytes: png } } }],
        ["bedrock", { image: { format: "png", source: { s3Location: { uri: "s3://shots/frame-1.png" } } } }],
        ["some-oss-gateway", { type: "image_url", image_url: { url: `data:image/png;base64,${png}` } }],
        // The cost-affecting envelope: a cache breakpoint riding as a sibling of
        // the payload, and an unrecognized sibling key. Both must fold into the
        // hash input identically in the two languages — the stable-JSON of the
        // part's remainder is the only place the two serializers could disagree.
        ["anthropic", { type: "image", source: { type: "base64", media_type: "image/png", data: png }, cache_control: { type: "ephemeral" } }],
        ["openai", { type: "image_url", image_url: { url: `data:image/png;base64,${png}` }, x_provider_hint: "grounding" }],
        // An implausible header: both SDKs must clamp it to "unknown" rather than
        // one of them reporting 8,068,951,256,159,688 tokens.
        ["gemini", { type: "image_url", image_url: { url: `data:image/png;base64,${monster}` } }],
      ];

      // The JS half. `null` marks a part this SDK does NOT treat as an image —
      // that decision is part of the contract too (the audio probe above must be
      // null on both sides, or one SDK would silently rewrite an audio blob).
      const js = probes.map(([provider, part]) => {
        const rb = imageRawBlock("user", part, provider);
        return rb === null ? null : [rb.text, rb.hashInput, rb.tokenCount, rb.tokenMethod];
      });

      // The Python half, spawned against the venv SDK. The battery is handed
      // over as a JSON argument rather than interpolated into source, so nothing
      // in a probe can change the program being run.
      const pyScript = `
import json, sys
from ctxdiff.images import image_raw_block
out = []
for provider, part in json.loads(sys.argv[1]):
    rb = image_raw_block("user", part, provider)
    out.append(None if rb is None else [rb.text, rb.hash_input, rb.token_count, rb.token_method])
print(json.dumps(out))
`;
      const proc = spawnSync(venvPython, ["-c", pyScript, JSON.stringify(probes)], {
        encoding: "utf8",
        env: { ...process.env, PYTHONPATH: pySrc },
      });
      expect(
        proc.status,
        `python image extraction failed (status ${proc.status}):\n${proc.stderr}`,
      ).toBe(0);
      const py = JSON.parse(proc.stdout) as (unknown[] | null)[];

      // Compared as one array so a failure names every disagreeing probe at once
      // instead of stopping at the first.
      expect(js).toEqual(py);

      // ...and the agreement is not vacuous. Every image probe must actually
      // have produced an image block marked as an estimate, and the ones with
      // sniffable bytes must carry a real size and a non-zero cost — otherwise
      // "both SDKs returned null for everything" would pass the check above.
      const images = js.filter((r): r is unknown[] => r !== null);
      expect(images).toHaveLength(probes.length - 1); // all but the audio probe
      expect(images.every((r) => r[3] === "estimate")).toBe(true);
      expect(images.filter((r) => String(r[0]).includes("×")).length).toBeGreaterThanOrEqual(10);
      expect(js[0]).toEqual(["[image 1024×768 · ~765 tok]", js[0]![1], 765, "estimate"]);

      // The envelope probes are not vacuous either: the SAME bytes at two detail
      // levels, with a cache breakpoint, and with an unknown sibling must be
      // FOUR distinct hash inputs — agreeing across the languages that they
      // differ is the whole point of shipping them through this battery.
      const hashOf = (i: number) => String(js[i]![1]);
      const last = probes.length - 1; // the three envelope probes are last, in order
      expect(new Set([hashOf(0), hashOf(1), hashOf(last - 2), hashOf(last - 1)]).size).toBe(4);
      // …while the clamped monster header degrades to a bare, costless image.
      expect(js[last]).toEqual(["[image]", js[last]![1], 0, "estimate"]);
    },
  );
});


/**
 * Cross-SDK BEDROCK conformance — the acceptance criterion for the JS Bedrock
 * adapter, and the reason it is a PORT of the Python one rather than a
 * re-derivation.
 *
 * The two SDKs get to Bedrock by completely different routes: boto3 exposes
 * `client.converse(...)` as a method and hands the adapter a plain dict of
 * kwargs; the AWS SDK v3 exposes one `client.send(command)` and hides the same
 * payload on `command.input`. The Converse WIRE SHAPE underneath is identical,
 * so the blocks — and therefore the hashes — must be too. If they are not, a
 * team running a Python service and a Node service against the same model sees
 * every shared system prompt as two different blocks, and `ctxdiff diff` stops
 * meaning anything across the pair.
 *
 * So this drives the REAL `@aws-sdk/client-bedrock-runtime` client through the
 * real interceptor into a real `.ctrace`, then asks the REAL Python adapter for
 * the same request's blocks and compares the stored hashes. It covers the three
 * shapes most likely to drift: a system block, a tool schema (stable JSON, two
 * different serializers), and an IMAGE (hashed over decoded bytes that arrive
 * as a `Uint8Array` here and as `bytes` there).
 */
describe("cross-language conformance (a Bedrock Converse request hashes identically)", () => {
  const hasVenv = existsSync(venvPython);

  it.skipIf(!hasVenv)(
    "the JS SDK's captured blocks equal the Python adapter's, hash for hash",
    async () => {
      // A 4×4 PNG, base64 — the same picture both sides will hash, shipped as
      // base64 so it survives the JSON hop and is turned back into each
      // language's own byte type before either adapter sees it.
      const PNG_4x4_B64 =
        "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAC0lEQVR4nGNgYAAAAAMAAbitOmMAAAAASUVORK5CYII=";

      // ONE logical Converse request, in the shape BOTH SDKs put on the wire.
      const request = {
        modelId: "anthropic.claude-3-5-sonnet-20240620-v1:0",
        system: [{ text: "You are a terse assistant." }, { cachePoint: { type: "default" } }],
        toolConfig: {
          tools: [
            {
              toolSpec: {
                name: "get_weather",
                description: "Look up the weather for a city.",
                inputSchema: {
                  json: {
                    type: "object",
                    properties: { city: { type: "string" }, unit: { type: "string" } },
                    required: ["city"],
                  },
                },
              },
            },
          ],
        },
        messages: [
          {
            role: "user",
            content: [
              { text: "What is in this picture, and what is the weather there?" },
              { image: { format: "png", source: { bytes: PNG_4x4_B64 } } },
            ],
          },
          {
            role: "assistant",
            content: [
              { toolUse: { toolUseId: "tooluse_1", name: "get_weather", input: { city: "Dubai" } } },
            ],
          },
          {
            role: "user",
            content: [
              { toolResult: { toolUseId: "tooluse_1", content: [{ text: "42°C" }] } },
            ],
          },
        ],
        inferenceConfig: { maxTokens: 512, temperature: 0.2 },
      };

      const path = join(tmpdir(), `ctxdiff-bedrock-${randomUUID()}.ctrace`);
      try {
        // --- the JS half: the REAL AWS SDK client, the REAL interceptor ------
        // The image arrives as the Uint8Array the AWS SDK actually carries.
        const jsRequest = structuredClone(request);
        jsRequest.messages[0]!.content[1]!.image!.source.bytes = new Uint8Array(
          Buffer.from(PNG_4x4_B64, "base64"),
        ) as never;

        const client = new BedrockRuntimeClient({
          region: "us-east-1",
          credentials: { accessKeyId: "x", secretAccessKey: "y" },
          requestHandler: {
            handle: async () => ({
              response: {
                statusCode: 200,
                reason: "OK",
                headers: { "content-type": "application/json" },
                body: Readable.from([
                  Buffer.from(
                    JSON.stringify({
                      output: { message: { role: "assistant", content: [{ text: "hot" }] } },
                      stopReason: "end_turn",
                      usage: { inputTokens: 40, outputTokens: 3, totalTokens: 43 },
                    }),
                  ),
                ]),
              },
            }),
            updateHttpClientConfig(): void {},
            httpHandlerConfigs: () => ({}),
          } as never,
        });
        const tracer = init("bedrock-conformance", { path });
        const wrapped = tracer.wrap(client) as BedrockRuntimeClient;
        await wrapped.send(new ConverseCommand(jsRequest as never));
        await tracer.close();

        const ct = CTrace.open(path);
        const calls = ct.getCalls();
        expect(calls).toHaveLength(1);
        const js = ct
          .getCallBlocks(calls[0]!.id)
          .map((cb) => [cb.block.role, cb.block.kind, cb.block.text, cb.block.contentHash]);
        const jsParams = calls[0]!.params;
        const jsUsage = calls[0]!.usage;
        ct.close();

        // --- the Python half: the REAL BedrockAdapter -------------------------
        // The payload travels as a JSON argument (never interpolated into
        // source, so nothing in it can change the program being run) and the
        // image is turned back into the `bytes` boto3 would have handed over.
        const pyScript = `
import base64, json, sys
from ctxdiff.capture.bedrock import BedrockAdapter
from ctxdiff.models import content_hash

kwargs = json.loads(sys.argv[1])
for message in kwargs.get("messages", []):
    for part in message.get("content", []):
        source = (part.get("image") or {}).get("source") if isinstance(part, dict) else None
        if isinstance(source, dict) and isinstance(source.get("bytes"), str):
            source["bytes"] = base64.b64decode(source["bytes"])

adapter = BedrockAdapter()
blocks = [
    [b.role, b.kind, b.text,
     content_hash(b.role, b.kind, b.hash_input if b.hash_input is not None else b.text)]
    for b in adapter.extract_blocks(kwargs)
]
response = {"usage": {"inputTokens": 40, "outputTokens": 3, "totalTokens": 43}}
print(json.dumps({"blocks": blocks,
                  "params": adapter.extract_params(kwargs),
                  "usage": adapter.extract_usage(response)}))
`;
        const proc = spawnSync(venvPython, ["-c", pyScript, JSON.stringify(request)], {
          encoding: "utf8",
          env: { ...process.env, PYTHONPATH: pySrc },
        });
        expect(
          proc.status,
          `python bedrock extraction failed (status ${proc.status}):\n${proc.stderr}`,
        ).toBe(0);
        const py = JSON.parse(proc.stdout) as {
          blocks: unknown[][];
          params: Record<string, unknown>;
          usage: Record<string, unknown>;
        };

        // THE ASSERTION. Same roles, same kinds, same stored text, same hashes —
        // compared as one array so a failure names every divergence at once.
        expect(js).toEqual(py.blocks);
        // ...and the params/usage a `.ctrace` carries alongside them.
        expect(jsParams).toEqual(py.params);
        expect(jsUsage).toEqual(py.usage);

        // The agreement is not vacuous: all five block shapes really are there,
        // in send order, and the image really was hashed over its BYTES rather
        // than serialized (its stored text is a descriptor, and no two blocks
        // collide).
        expect(js.map((b) => `${b[0]}/${b[1]}`)).toEqual([
          "system/message",
          "system/message",
          "system/tool_schema",
          "user/content_part",
          "user/image",
          "assistant/content_part",
          "user/content_part",
        ]);
        expect(js[4]![2]).toBe("[image 4×4 · ~1 tok]");
        expect(String(js[4]![3])).not.toBe(String(js[3]![3]));
        expect(new Set(js.map((b) => b[3])).size).toBe(js.length);
      } finally {
        rmSync(path, { force: true });
      }
    },
  );
});
