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

**Jump to:** [How it's different](#how-its-different) · [Features](#features) · [Install](#install) · [Quickstart](#quickstart) · [The CLI](#the-cli) · [HTML dashboard](#html-dashboard) · [Storage backends](#storage-backends) · [Providers](#providers) · [How it works](#how-it-works) · [Provider recipes](#provider-recipes) · [The `.ctrace` format](#the-ctrace-format) · [Design principles](#design-principles) · [Contributing](#contributing)

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
- 🗄️ **[Pluggable storage](#storage-backends)** — local-first `.ctrace` by default; `configure({ store: new PostgresStore({ dsn }) })` (or `CTXDIFF_STORE=…`) once, and every run lands in your **PostgreSQL/MySQL** instead. Tables auto-create, the drivers are optional peers, and a dead database degrades capture without ever touching your agent.
- 🔒 **Privacy first** — local-first (no network, no telemetry), a redaction hook that runs before anything touches disk, and HTML exports that strip request params down to the model name.
- ✅ **Honest numbers** — exact `o200k_base` token counts for OpenAI (matching Python's `tiktoken`); estimates are always *marked* as estimates.

**What it doesn't do (yet) — JS SDK specifics:**

- ⏳ **AWS Bedrock** — supported in the Python SDK, not yet ported to JS.
- ⏳ **Abandoned streams** — a stream you obtain but *never iterate at all* isn't recorded. JS has no deterministic finalizer, and GC-timed `FinalizationRegistry` recording was deliberately avoided; streams that are consumed, broken out of early, errored, or exhausted all record. (In practice you always iterate a stream you asked for.)
- ⏳ **Live tail** — the dashboard is post-run; it doesn't update while the agent runs.
- ⏳ **Background recording (local file only)** — writing to the local `.ctrace` is synchronous on the call path (fast, but not zero-cost); a [database backend](#storage-backends) already writes off it, via a serial background writer.
- ℹ️ **Cross-language diff edge** — an integer-valued float inside a JSON-Schema numeric keyword (e.g. `default: 3.0`) normalizes as `3` in JS vs `3.0` in Python, so the *same tool schema authored in both SDKs* would hash differently. This affects only a cross-language diff; JS→Python reads are unaffected (readers never re-hash) and single-language dedup is fully consistent. See [`spec/ctrace-schema.md`](../spec/ctrace-schema.md).

---

## Install

Requires **Node ≥ 22** (uses the built-in `node:sqlite`).

```bash
npm i ctxdiff
```

The only runtime dependency is a pure-JS tokenizer (`gpt-tokenizer`) for exact OpenAI token counts. The provider SDKs (`openai`, `@anthropic-ai/sdk`, `@google/genai`) are **optional peer dependencies** — `ctxdiff` wraps whatever client you already use, and never imports them itself.

Storage is a local `.ctrace` file by default, with nothing extra to install. To keep traces in a database you already run, add the matching driver — see [Storage backends](#storage-backends):

```bash
npm i pg        # PostgreSQL
npm i mysql2    # MySQL / MariaDB
```

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
| `runs` | list `.ctrace` files in the working directory (or, with a [database configured](#storage-backends), that store's sessions) |
| `view [--no-open]` | open a self-contained HTML dashboard in your browser |
| `export [--out FILE.html]` | write a self-contained HTML dashboard for a run |
| `demo [--out FILE] [--keep] [--no-open]` | build a sample multi-agent dashboard — no API keys, no setup |

Common options: `--agent A` scopes to one agent; `--run PATH` picks a trace (default: most recently modified `*.ctrace` in the cwd). A positional path also works, e.g. `npx ctxdiff tokens my-run.ctrace`.

---

## HTML dashboard

`npx ctxdiff view` (or `export`) produces **one HTML file** with everything inline — no CDN, no fonts, no external request of any kind — so it's safe to attach to a bug ticket or open offline. All trace text is HTML-escaped and rendered via `textContent`, never `innerHTML`, so untrusted trace data can never execute. It renders byte-identically to the Python viewer.

Panels: a turn scrubber, agent chips + filtering, the block diff for the selected turn, a token-allocation heatmap, cache-break findings, a full block inspector, and a context-growth chart — light and dark themes.

---

## Storage backends

By default `ctxdiff` is **local-first and zero-config**: `trace.init("my-agent")` writes `./my-agent.ctrace`, a plain SQLite file you can open, query, email or attach to a ticket. Nothing to install, nothing to run, no server. That default never changes on its own — everything below is opt-in.

When you'd rather keep traces in a database you already run (a shared team dashboard, an agent fleet across many containers, a place where `.ctrace` files can't live), point `ctxdiff` at it **once** and every later `trace.init()` follows:

```ts
import { configure, trace, PostgresStore } from "ctxdiff";

configure({ store: new PostgresStore({ dsn: "postgresql://user:pw@db.internal/agents" }) });

// ...from here on, unchanged:
const tracer = trace.init("support-agent");
const client = tracer.wrap(new OpenAI());
```

Or set an environment variable and change **no code at all**:

```bash
export CTXDIFF_STORE=postgresql://user:pw@db.internal/agents   # PostgreSQL
export CTXDIFF_STORE=mysql://user:pw@db.internal/agents        # MySQL / MariaDB
export CTXDIFF_STORE=sqlite:///var/lib/ctxdiff/agents.ctrace   # one SQLite file
export CTXDIFF_STORE=~/traces                                  # a directory: ~/traces/<project>.ctrace
```

The value is a **location**, not a backend name: `CTXDIFF_STORE=postgres` is rejected with a message showing the URL form, rather than quietly creating a local SQLite file called `postgres`.

### The three backends

| Backend | Install | Configure with |
|---|---|---|
| **SQLite** (default) | — built in (`node:sqlite`) | nothing, or `new SQLiteStore({ path })` |
| **PostgreSQL** | `npm i pg` | `new PostgresStore({ dsn: "postgresql://..." })` |
| **MySQL / MariaDB** | `npm i mysql2` | `new MySQLStore({ dsn: "mysql://..." })` |

The drivers ([`pg`](https://node-postgres.com/) and [`mysql2`](https://sidorares.github.io/node-mysql2/)) are **optional peer dependencies, imported lazily at connect time**. `ctxdiff`'s core still has exactly one runtime dependency (`gpt-tokenizer`), `import "ctxdiff"` never loads a database driver, and a store configured for a backend whose driver isn't installed tells you what to install instead of crashing.

### Tables are created for you

On first connect the adapter runs `CREATE TABLE IF NOT EXISTS` for its four tables — `ctxdiff_run`, `ctxdiff_call`, `ctxdiff_block`, `ctxdiff_call_block` — in whatever database the DSN points at. **There is no migration step.** The tables are prefixed because they live in a database you share with your own application, and connecting again is a harmless no-op.

It's the same logical model as a `.ctrace` file, in every backend: sessions, calls, content-hashed blocks stored **once** and referenced by position, and call→block membership. Analyzers can't tell the difference — the same conformance suite runs against all three, against real PostgreSQL and MySQL servers as well as SQLite. The tables and columns match the **Python** SDK's exactly, so a database written by either SDK reads in the other.

Same *semantics*, too, not just the same shape:

- **Sessions are ordered by write order**, never by `startedAt`. Several containers writing into one database will disagree about the clock; ordering by an insert-order column (SQLite's `rowid`, `BIGSERIAL`, `AUTO_INCREMENT`) means "the newest session" is the one written last on every backend, and stays stable between reads.
- **Keys compare byte-exactly.** MySQL's default collation is case-insensitive, which would make two content hashes differing only in case the same block; ctxdiff pins `ascii_bin` on its id/hash columns so content-addressed dedup means what it says.
- **Free-text columns are unbounded** on every backend, so a 400-character provider error or tag label stores rather than rejecting the call.

### Reading from a database

The CLI and the dashboard read from the configured store too. With `CTXDIFF_STORE` set (or after `configure()`), the read commands analyze the **newest session** in the database rather than looking for a `.ctrace` in the working directory:

```bash
npx ctxdiff tokens                   # newest session in the configured store
npx ctxdiff diff --turn 7 --turn 8
npx ctxdiff runs                     # every session in the store, oldest first
npx ctxdiff export --out run.html    # same self-contained dashboard
```

`--run PATH` always wins: a path names a file, so it reads that `.ctrace` even when a database is configured. The same rule applies on the write side — `trace.init(project, { path })` is always a local file. `export`/`view` need an explicit `--out` against a database, since there is no trace filename to derive one from.

### A database is never allowed to break your agent

The [fail-open guarantee](#fail-open-guarantee) covers a networked store completely, and matters more there than for a local file — Node runs your agent on ONE thread, so anything that waits is something your whole process waits for:

- **No database I/O ever happens on your call path.** Not the writes, and not the connect either: `tracer.wrap()` builds a queue and returns, and the run's single writer opens the session in the background. A database that is slow, wedged or absent costs your agent nothing, and the event loop keeps turning throughout.
- **Every wait is bounded.** Connects (5s) and statements (10s, both configurable) are bounded server-side *and* client-side, because a server-side `statement_timeout` can never fire when the packets carrying its verdict are the ones being dropped. TCP keepalives catch a peer that dies while the connection is idle, and `tracer.close()` is bounded by the store's own statement timeout — a wedged database can slow your shutdown, never prevent it.
- **A dead database degrades capture, never the run.** If the store can't be reached or created, `ctxdiff` logs **one** warning, records nothing, and your calls run exactly as if `ctxdiff` weren't there. It never silently falls back to writing a local file you didn't ask for.
- **A connection killed mid-run is reopened.** A pooler recycle, a failover or a database restart costs the write that was in flight, not the rest of the run.
- **A typo'd DSN behaves the same way** — a misconfigured trace destination is a tracing problem, and tracing problems must not take down the program being traced.

`await tracer.close()` is the flush point for a database-backed run: it drains the queue and closes the connection. (For the local `.ctrace` the write is synchronous, so `tracer.close()` behaves exactly as it always has.)

If you *forget* `close()`, your program still **exits normally** — the tracing connection is detached from Node's event loop, so it can never hold a finished process open. The cost is the writes still queued at exit, which is the right trade: a debugging tool that stops your script from ending would be a much worse bug than a lost final turn.

### Writing your own backend

`Store` and `StoreBackend` (exported from `ctxdiff`) are small structural interfaces — seven methods and two, respectively. Anything satisfying them can be passed to `configure({ store })`; no base class, no registration. Every method may return a value *or* a promise, which is how one protocol serves both the synchronous `node:sqlite` store and the promise-based network ones.

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

**Per-async-context semantics.** `tag()` and `mark()` are scoped to the **current async context**, held in an [`AsyncLocalStorage`](https://nodejs.org/api/async_context.html) — not global tracer state. The interceptor snapshots the current tag/step **synchronously at call time** (before any `await`), so a plain fan-out where each branch labels then calls records correctly with no bleed:

```ts
await Promise.all(
  docs.map(async (doc, i) => {
    tracer.mark(`worker-${i}`);
    tracer.tag("rag", [doc]);
    await client.chat.completions.create({ model, messages: build(doc) }); // records worker-i / rag
  }),
);
```

The one hazard: if a branch `await`s **between** labeling and the call, concurrent siblings share the root store and a sibling's `mark()` can land in the gap and relabel it. For that — and any concurrent phase-labeling — use the scoped `step()` form, which gives each branch its **own** isolated context.

**Scoped `step()` — the recommended concurrency-safe form.** Mirrors the Python `Tracer.step()` context manager. The callback form runs your block inside a fresh async context (via `AsyncLocalStorage.run`), so every call inside records `step=label` and nothing leaks into a sibling branch; a branch that opens no `step()` records `step=null`, never a sibling's leftover label:

```ts
// callback form — fully concurrency-safe, isolates each branch
await Promise.all(
  phases.map((p) => tracer.step(p.label, async () => {
    await client.chat.completions.create({ model, messages: p.messages });
  })),
);

// Disposable form (TS 5.2 `using`) — for sequential / single-context phases
{
  using _phase = tracer.step("answer");
  await client.chat.completions.create({ model, messages });
} // previous step restored on scope exit
```

Prefer the **callback form** under `Promise.all`; the `using` form uses `enterWith` and is leak-proof for sequential nesting but — like a bare `mark()` — can share state across branches started in the same tick. `step()` is fully reentrant: nested scopes restore the exact enclosing label on exit.

> **Node note:** unlike the Python SDK, ctxdiff-js does no background writer thread — Node is single-threaded and `node:sqlite` writes stay synchronous on the main thread, so there's no cross-thread DB race to guard against. The concurrency fix here is purely the per-async-context tag/step isolation.

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

Wrapping is **absolute** fail-open: it never throws into your app, never breaks stream iteration, never drops/reorders/delays a chunk, and never alters your request. If tracing can't run, your app runs unchanged. With a [database backend](#storage-backends) it goes further: no store I/O of any kind — not even the connect — sits on your call path, and every wait ctxdiff makes is bounded, so a wedged database cannot stall your event loop or hang your shutdown.

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
