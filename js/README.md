# ctxdiff (JavaScript / TypeScript)

[![npm](https://img.shields.io/npm/v/ctxdiff.svg?logo=npm&logoColor=white)](https://www.npmjs.com/package/ctxdiff)
[![CI](https://github.com/salmanzafar949/ctxdiff/actions/workflows/js.yml/badge.svg)](https://github.com/salmanzafar949/ctxdiff/actions/workflows/js.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/salmanzafar949/ctxdiff/blob/main/LICENSE)
[![Node 22+](https://img.shields.io/badge/node-22%2B-blue.svg)](https://nodejs.org/)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/salmanzafar949/ctxdiff/blob/main/CONTRIBUTING.md)

**git diff for your agent's context window.** See exactly what your LLM saw — turn by turn, block by block.

`ctxdiff` is a local-first debugger for the context window of LLM agents. Wrap your OpenAI, Anthropic, or Gemini client in one line, run your agent, and every call's context is recorded — as content-hashed, deduplicated *blocks* — into a single-file SQLite trace you can inspect, diff, and share. Nothing leaves your machine.

This is the **JavaScript/TypeScript** SDK. It writes the exact same `.ctrace` format as the [Python SDK](https://pypi.org/project/ctxdiff/) — traces are cross-compatible in both directions, and the CLI/dashboard output is byte-identical across the two.

> Prompt wording is ~10% of the battle. The other 90% is **context engineering** — what the model sees, in what order, at what cost. When an agent misbehaves at turn 8, `ctxdiff` answers the three questions a raw JSON log can't: *what exactly did the model see, what changed since turn 7, and what did it cost?*

```ts
import { trace } from "ctxdiff";
import OpenAI from "openai";

const tracer = trace.init("customer-support-agent");
const client = tracer.wrap(new OpenAI());   // ← the only line you add

await client.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "What's your refund window?" }],
});

tracer.close();                             // writes ./customer-support-agent-<id>.ctrace
```

Then `npx ctxdiff view` opens the self-contained dashboard — a turn scrubber, agent chips and filtering, git-style turn diffs, a token heatmap, cache-break attribution, and light/dark themes.

> **See it in 30 seconds — no API key, no setup:**
> ```bash
> npm i ctxdiff
> npx ctxdiff demo      # builds a sample multi-agent trace and opens this dashboard
> ```
> A realistic research-pipeline run (two agents, real SDK shapes, zero network) already showing turn diffs, token/schema-bloat, a cache-prefix break, and an agent hand-off.

**Jump to:** [How it's different](#how-its-different) · [Features](#features) · [Install](#install) · [Quickstart](#quickstart) · [The CLI](#the-cli) · [HTML dashboard](#html-dashboard) · [Providers](#providers) · [How it works](#how-it-works) · [Provider recipes](#provider-recipes) · [The `.ctrace` format](#the-ctrace-format) · [Design principles](#design-principles) · [Contributing](#contributing)

---

## How it's different

`ctxdiff` isn't a tracing platform. Tools like LangSmith, Langfuse, and Phoenix show you a **list of calls** — a snapshot of each request, aggregated across your fleet, for monitoring cost and health over time. That's a different job.

`ctxdiff` shows you the **diff between calls** — what got *added, evicted, or modified* in the context window, turn by turn, down to the character that broke your prompt-cache prefix. It's the debugger you reach for when one specific run went wrong at turn 8, not the dashboard you watch in production.

|  | Observability platforms | ctxdiff |
|---|---|---|
| **Question** | Is my fleet healthy & cheap over time? | Why did *this* run break at turn 8? |
| **Shows you** | A list of calls (snapshot) | The **diff** between calls (delta) |
| **Scope** | Many runs, aggregated | One run, turn by turn |
| **Runs** | Hosted service / self-hosted server | Local — one `npm i`, one file |
| **Your data** | Sent to a platform | Never leaves your machine |

It's built to sit **alongside** your observability stack, not replace it.

---

## Features

**What ctxdiff does:**

- 🔌 **One-line capture** — `tracer.wrap(client)` records every LLM call's full context, verbatim, into a single-file SQLite `.ctrace`. [Fail-open by design](#fail-open-guarantee): a ctxdiff error can never break your app.
- 🧬 **Content-hashed block storage** — every message, content part, and tool schema is a deduplicated block; a stable system prompt across 40 turns is stored once.
- 🌐 **Three provider surfaces** — OpenAI (chat + Responses), Anthropic, and Google Gemini, including streaming and the `.stream()` convenience helpers.
- 🟩🟥🟨 **Git-style turn diffing** — `npx ctxdiff diff --turn 7 --turn 8`: exactly which blocks were added, evicted, or modified (with char-level inline diffs) between any two turns.
- 📊 **Token attribution** — `npx ctxdiff tokens`: where the budget goes per turn (system / rag / history / schemas…), reconciled against provider-reported usage, plus **schema-bloat detection** — tools you registered but never call, taxing every request.
- 💸 **Prompt-cache profiling** — `npx ctxdiff cache`: finds exactly what breaks your cache prefix (down to the changed characters), counts re-billed tokens, and suggests the fix.
- 🖥️ **Self-contained HTML dashboard** — `npx ctxdiff view` / `export`: a one-file, zero-external-request dashboard with a turn scrubber, diff panel, token heatmap, cache findings, and block inspector — safe to attach to a bug ticket.
- 🏷️ **Semantic tagging** — `tracer.tag("rag", chunks)` for exact provenance labels; a cheap heuristic covers the rest.
- 🤝 **Multi-agent runs** — `tracer.wrap(client, { agent: "researcher" })` and `tracer.mark("step")` attribute every call to the agent (and step) that made it; `--agent` filters `diff`/`tokens`/`cache`, and the dashboard colors each agent's turns. Cross-agent hand-offs are never miscounted as cache breaks.
- 🔒 **Privacy first** — local-first (no network, no telemetry), a redaction hook that runs before anything touches disk, and HTML exports that strip request params down to the model name.
- ✅ **Honest numbers** — exact `o200k_base` token counts for OpenAI (matching Python's `tiktoken`); estimates are always *marked* as estimates.

**What it doesn't do (yet) — JS SDK specifics:**

- ⏳ **AWS Bedrock** — supported in the Python SDK, not yet ported to JS.
- ⏳ **Abandoned streams** — a stream you obtain but *never iterate at all* isn't recorded. JS has no deterministic finalizer, and GC-timed `FinalizationRegistry` recording was deliberately avoided; streams that are consumed, broken out of early, errored, or exhausted all record. (In practice you always iterate a stream you asked for.)
- ⏳ **Live tail** — the dashboard is post-run; it doesn't update while the agent runs.
- ⏳ **Background recording** — capture is synchronous on the call path (fast, but not zero-cost).
- ℹ️ **Cross-language diff edge** — an integer-valued float inside a JSON-Schema numeric keyword (e.g. `default: 3.0`) normalizes as `3` in JS vs `3.0` in Python, so the *same tool schema authored in both SDKs* would hash differently. This affects only a cross-language diff; JS→Python reads are unaffected (readers never re-hash) and single-language dedup is fully consistent. See [`spec/ctrace-schema.md`](../spec/ctrace-schema.md).

---

## Install

Requires **Node ≥ 22** (uses the built-in `node:sqlite`).

```bash
npm i ctxdiff
```

The only runtime dependency is a pure-JS tokenizer (`gpt-tokenizer`) for exact OpenAI token counts. The provider SDKs (`openai`, `@anthropic-ai/sdk`, `@google/genai`) are **optional peer dependencies** — `ctxdiff` wraps whatever client you already use, and never imports them itself.

---

## Quickstart

Wrap a client, use it exactly as you normally would, then read the trace back.

```ts
import { trace, CTrace } from "ctxdiff";
import OpenAI from "openai";

// 1. Start a trace and wrap your client
const tracer = trace.init("support-agent");
const client = tracer.wrap(new OpenAI());

// 2. Use the client normally — every call is recorded
await client.chat.completions.create({
  model: "gpt-4o",
  messages: [
    { role: "system", content: "You are a terse support agent." },
    { role: "user", content: "How do I reset my password?" },
  ],
});
tracer.close();

// 3. Read the trace back (or use the CLI / dashboard)
const ct = CTrace.open(tracer.path);
for (const call of ct.getCalls()) {
  console.log(`turn ${call.seq}:`, ct.getCallBlocks(call.id).length, "blocks");
}
ct.close();
```

---

## The CLI

`npx ctxdiff <command>` — read-only analysis over any `.ctrace`, including traces written by the **Python** SDK. Output is byte-identical to the Python CLI.

| Command | What it does |
|---|---|
| `diff --turn N --turn M` | git-style block diff between two turns (char-level inline diffs) |
| `tokens [--turn N]` | per-label token heatmap, provider reconciliation, schema-bloat report |
| `cache` | prompt-cache prefix-break profiler + price-free wasted-spend estimate |
| `runs` | list `.ctrace` files in the working directory |
| `view [--no-open]` | open a self-contained HTML dashboard in your browser |
| `export [--out FILE.html]` | write a self-contained HTML dashboard for a run |
| `demo [--out FILE] [--keep] [--no-open]` | build a sample multi-agent dashboard — no API keys, no setup |

Common options: `--agent A` scopes to one agent; `--run PATH` picks a trace (default: most recently modified `*.ctrace` in the cwd). A positional path also works, e.g. `npx ctxdiff tokens my-run.ctrace`.

---

## HTML dashboard

`npx ctxdiff view` (or `export`) produces **one HTML file** with everything inline — no CDN, no fonts, no external request of any kind — so it's safe to attach to a bug ticket or open offline. All trace text is HTML-escaped and rendered via `textContent`, never `innerHTML`, so untrusted trace data can never execute. It renders byte-identically to the Python viewer.

Panels: a turn scrubber, agent chips + filtering, the block diff for the selected turn, a token-allocation heatmap, cache-break findings, a full block inspector, and a context-growth chart — light and dark themes.

---

## Providers

`tracer.wrap(client)` auto-detects the provider from the client you pass:

| Provider | Client | Methods captured |
| --- | --- | --- |
| OpenAI | `openai` | `chat.completions.create` / `.stream()`, `responses.create` / `.stream()`, `stream: true` |
| Anthropic | `@anthropic-ai/sdk` | `messages.create` (+ `stream: true`), `messages.stream()` |
| Gemini | `@google/genai` | `models.generateContent`, `models.generateContentStream` |

Sync and async clients are both handled. Streaming usage is folded from each provider's own events (OpenAI final-chunk `usage`; Anthropic `message_start` + `message_delta`; Gemini cumulative `usageMetadata`) and recorded once the stream completes. An unrecognized client is returned unwrapped (with a warning), never throwing.

---

## How it works

1. **Capture.** `tracer.wrap(client)` returns a transparent `Proxy` over your client. It forwards everything untouched except the completion methods, which it intercepts to record the request's context — verbatim, wire-level — after the real call runs.
2. **Block model.** Each request is flattened into ordered **blocks** (a message, a content part, a tool schema). A block's identity is `sha256(role · kind · normalized-text)`, so identical content is stored once and referenced many times. Diffing is an ordered hash-list comparison.
3. **Analysis.** `diff`, `tokens`, and `cache` are pure functions over the stored blocks — re-runnable, and the single source of truth the dashboard embeds.

### The block model

`role` ∈ `system · user · assistant · tool` · `kind` ∈ `message · content_part · tool_schema`. Labels (`system · user · history · rag · tool_schema · tool_output`) come from a cheap heuristic unless you override them with `tracer.tag(...)`.

---

## Usage guide

### Semantic tagging

```ts
tracer.mark("retrieve");              // sticky step label until changed
tracer.tag("rag", [retrievedDoc]);    // labels the NEXT call's matching blocks as "rag"
```

`mark` is sticky across calls; `tag` applies to the next call only. Each is optional — a role-based heuristic labels everything else.

### Multi-agent runs

```ts
const researcher = tracer.wrap(new OpenAI(), { agent: "researcher" });
const writer = tracer.wrap(new Anthropic(), { agent: "writer" });
```

Every call is attributed to its agent. `--agent` filters the CLI; the dashboard colors each agent's turns; and the cache profiler analyzes each agent's timeline separately so a hand-off is never mistaken for a cache break.

### Redaction

```ts
const tracer = trace.init("support-agent", {
  redact: (block) => ({ ...block, text: scrub(block.text) }),
});
```

The block is hashed and token-counted **before** redaction, so identity/dedup stays stable; only the stored text changes. A throwing redactor is caught and the text replaced with a sentinel — capture never breaks.

> **Scope:** redaction applies to block text. Request `params`, provider `usage`, and error names are stored as-is.

### Fail-open guarantee

Wrapping is **absolute** fail-open: it never throws into your app, never breaks stream iteration, never drops/reorders/delays a chunk, and never alters your request. If tracing can't run, your app runs unchanged.

### Token counting

Exact for OpenAI via the pure-JS `o200k_base` tokenizer (verified to match Python's `tiktoken` byte-for-byte); a documented `~4-chars/token` estimate for other providers. Estimated counts are always marked `estimate`, never passed off as exact.

---

## Provider recipes

### OpenAI

```ts
import OpenAI from "openai";
const client = tracer.wrap(new OpenAI());
await client.chat.completions.create({ model: "gpt-4o", messages });
await client.responses.create({ model: "gpt-4o", input: "hello" });
// streaming: create({ stream: true, stream_options: { include_usage: true } }) or .stream()
```

### Anthropic / Claude

```ts
import Anthropic from "@anthropic-ai/sdk";
const client = tracer.wrap(new Anthropic(), { agent: "writer" });
await client.messages.create({ model: "claude-3-5-sonnet-20241022", max_tokens: 1024, messages });
// streaming: create({ stream: true }) or the messages.stream() helper
```

### Google Gemini

```ts
import { GoogleGenAI } from "@google/genai";
const client = tracer.wrap(new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY }));
await client.models.generateContent({ model: "gemini-2.0-flash", contents: "hello",
  config: { systemInstruction: "be terse" } });
// streaming: models.generateContentStream({ ... })
```

---

## The `.ctrace` format

One run = one SQLite file, versioned by `schema_version` so an old or foreign file is rejected with a clear error rather than misread. The format is a shared cross-SDK contract documented in [`spec/ctrace-schema.md`](../spec/ctrace-schema.md) — a trace written by the JS SDK opens in the Python `ctxdiff` CLI and viewer, and vice-versa.

| Table | Row | Key columns |
|---|---|---|
| `run` | one per file | `project`, `provider`, `started_at`, `ctxdiff_version`, `schema_version` |
| `call` | one per LLM request | `seq`, `params` (JSON), `usage` (JSON), `latency_ms`, `error`, `agent`, `step` |
| `block` | one per **distinct** context unit | `content_hash` (PK), `role`, `kind`, `text`, `token_count`, `token_method` |
| `call_block` | membership of a block in a call | `call_id`, `block_id`, `position`, `label`, `label_source` |

---

## Design principles

- **Local-first.** No network calls, no telemetry, no external services.
- **Fail-open, always.** Capture can never break your application.
- **Wire-level truth.** The proxy records what was actually sent, verbatim; interpretation is a separate, re-runnable layer.
- **Cross-SDK parity.** The JS and Python SDKs produce byte-identical hashes, CLI output, and dashboards on the same trace.
- **One file per run.** The `.ctrace` *is* the shareable artifact — no bundle, no server.

---

## Contributing

Issues and PRs welcome — see [CONTRIBUTING.md](https://github.com/salmanzafar949/ctxdiff/blob/main/CONTRIBUTING.md).

```bash
cd js
npm install
npm run build
npm test          # unit + cross-language conformance against the Python SDK
```

---

## License

[Apache-2.0](https://github.com/salmanzafar949/ctxdiff/blob/main/LICENSE).
