# ctxdiff (JavaScript / TypeScript)

[![npm](https://img.shields.io/npm/v/ctxdiff.svg?logo=npm&logoColor=white)](https://www.npmjs.com/package/ctxdiff)
[![CI](https://github.com/salmanzafar949/ctxdiff/actions/workflows/js.yml/badge.svg)](https://github.com/salmanzafar949/ctxdiff/actions/workflows/js.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/salmanzafar949/ctxdiff/blob/main/LICENSE)
[![Node 22+](https://img.shields.io/badge/node-22%2B-blue.svg)](https://nodejs.org/)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/salmanzafar949/ctxdiff/blob/main/CONTRIBUTING.md)

**Find the character that's breaking your agent's prompt cache on every turn** — and the tool schemas you pay for on every call but never invoke.

*`git diff` for your agent's context window:* see exactly what your LLM saw, turn by turn, block by block.

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
- 🌐 **Five provider surfaces** — OpenAI (chat + Responses), Anthropic, Google Gemini (AI Studio **and Vertex AI**), AWS Bedrock (Converse, streaming included) — including streaming and the `.stream()` convenience helpers — plus **[LangChain/LangGraph via a callback handler](#langchain--langgraph)**. Same coverage as the Python SDK, and the same request hashes identically in either.
- 🟩🟥🟨 **Git-style turn diffing** — `npx ctxdiff diff --turn 7 --turn 8`: exactly which blocks were added, evicted, or modified (with char-level inline diffs) between any two turns.
- 📊 **Token attribution** — `npx ctxdiff tokens`: where the budget goes per turn (system / rag / history / schemas…), reconciled against provider-reported usage, plus **schema-bloat detection** — tools you registered but never call, taxing every request.
- 💸 **Prompt-cache profiling** — `npx ctxdiff cache`: finds exactly what breaks your cache prefix (down to the changed characters), counts re-billed tokens, and suggests the fix.
- 📐 **Percent of the context window** — `npx ctxdiff tokens --context-window 200000`: `18,400 / 200,000 tok · 9.2%`, with a `⚠` past 80%. Proximity to the limit is what causes the silent truncation you are debugging; the window is yours to state, because ctxdiff ships no model→window table it could get wrong.

- 🧠 **Evicted tagged blocks** — *"the block you tagged `rag` at turn 3 was evicted at turn 6"*. The single most common root cause of "the agent forgot the thing I told it", named outright.

- 🚦 **Context budgets in CI** — `npx ctxdiff check --max-context 8000 --require-stable-prefix --no-dead-schemas`: assert the budget, exit non-zero when it regresses. Ships as a [GitHub Action](#github-action) that posts the PASS/FAIL table to the job summary — so context size becomes a tracked metric on every pull request, not something you remember to look at.
- 🖥️ **Self-contained HTML dashboard** — `npx ctxdiff view` / `export`: a one-file, zero-external-request, **three-level** dashboard — every agent in the project, then that agent's sessions (in your local timezone), then its turn-by-turn scrubber, diff panel, token heatmap, cache findings and block inspector — safe to attach to a bug ticket.
- 🦜 **[LangChain & LangGraph, natively](#langchain--langgraph)** — `callbacks: [tracer.langchainHandler()]`, and every chat-model call in a graph is captured, whatever the provider. The blocks it records are **hash-identical** to wrapping that provider's SDK directly, so traces dedup instead of looking like unrelated contexts. Across SDKs too — except a tool call, whose JSON LangChain itself serializes differently in each language ([note](#what-it-doesnt-do-yet--js-sdk-specifics)).
- 🏷️ **Semantic tagging** — `tracer.tag("rag", chunks)` for exact provenance labels; a cheap heuristic covers the rest.
- 🤝 **Multi-agent runs** — `tracer.wrap(client, { agent: "researcher" })` and `tracer.mark("step")` attribute every call to the agent (and step) that made it; `--agent` filters `diff`/`tokens`/`cache`/`check` (and names a side of a cross-agent diff), `npx ctxdiff agents` rolls every agent up across a project's sessions, and the dashboard colors each agent's turns. Cross-agent hand-offs are never miscounted as cache breaks.
- 🗄️ **[Pluggable storage](#storage-backends)** — local-first `.ctrace` by default; `configure({ store: new PostgresStore({ dsn }) })` (or `CTXDIFF_STORE=…`) once, and every run lands in your **PostgreSQL/MySQL** instead. Tables auto-create, the drivers are optional peers, and a dead database degrades capture without ever touching your agent.
- 🔒 **Privacy first** — local-first (no network, no telemetry), a redaction hook that runs before anything touches disk, and HTML exports that strip request params down to the model name.
- ✅ **Honest numbers** — exact `o200k_base` token counts for OpenAI (matching Python's `tiktoken`); estimates are always *marked* as estimates.

**What it doesn't do (yet) — JS SDK specifics:**

- ⏳ **Abandoned streams** — a stream you obtain but *never iterate at all* isn't recorded. JS has no deterministic finalizer, and GC-timed `FinalizationRegistry` recording was deliberately avoided; streams that are consumed, broken out of early, errored, or exhausted all record. (In practice you always iterate a stream you asked for.)
- ⏳ **Live tail** — the dashboard is post-run; it doesn't update while the agent runs.
- ⏳ **Background recording (local file only)** — writing to the local `.ctrace` is synchronous on the call path (fast, but not zero-cost); a [database backend](#storage-backends) already writes off it, via a serial background writer.
- ℹ️ **Cross-language LangChain edge: tool calls** — LangChain re-serializes a tool call's arguments with the host language's own JSON serializer, and `JSON.stringify` emits `{"city":"Dubai"}` where Python's `json.dumps` emits `{"city": "Dubai"}`. The handler reproduces its own framework's real request byte for byte (that is the stronger guarantee, and what makes a LangChain trace dedup against a *direct* trace in the same SDK), so a **tool-call block hashes differently in the two SDKs** — as it already would between two direct captures of those same two requests, with no ctxdiff involved. Both hashes are pinned by both suites so the divergence cannot drift. Everything else — messages, system prompts, tool *schemas*, images — is cross-SDK identical.
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

> **What "byte-identical" covers:** every command's **output** — stdout, the operational errors on stderr, the selector errors (`no session …`, `no agent …`, the ambiguity listings) and the **exit code** — for the same trace and the same `TZ`. It does **not** cover argparse's help/usage chrome: `--help` text and the `usage: …` block Python prints above a bad-flag message have no equivalent here, so those errors match on exit code 2 and on the `error: …` line's substance rather than byte for byte. This CLI also accepts a leading positional `.ctrace` path as an alias for `--project`, which the Python CLI does not — an addition, never a difference in what either prints.

| Command | What it does |
|---|---|
| `diff --turn N --turn M` | git-style block diff between two turns (char-level inline diffs) |
| `tokens [--turn N] [--context-window N]` | per-label token heatmap, share of the context window, tagged-eviction warnings, provider reconciliation, schema-bloat report |
| `cache` | prompt-cache prefix-break profiler + price-free wasted-spend estimate |
| `check [assertions]` | assert context budgets and **exit non-zero** when they regress (the CI gate) |
| `sessions` | list every session ctxdiff can see — id, **local** start time, turns, agents |
| `agents` | every agent in the project, aggregated across **all** its sessions |
| `view [--no-open]` | open a self-contained HTML dashboard in your browser |
| `export [--out FILE.html]` | write a self-contained HTML dashboard for the project |
| `demo [--out FILE] [--keep] [--no-open]` | build a sample multi-agent dashboard — no API keys, no setup |

(`runs` is kept as a hidden alias of `sessions`, so existing scripts keep working.)

### Selectors

Every analysis command takes the same four, resolved identically:

| Selector | Means | Default |
| --- | --- | --- |
| `--project PATH\|DSN` | which project DB — a `.ctrace` path or a [database DSN](#storage-backends) | the configured store, else the most recently modified `*.ctrace` in the cwd |
| `--session ID` | which session in it (an id, or any unambiguous prefix) | the only session — **required when the project holds several** (except `export`/`view`) |
| `--agent NAME` | scope to one agent | all agents |
| `--turn N` | a specific turn | every turn |

`--run` remains an alias for `--project`, and a positional path still works (`npx ctxdiff tokens my-run.ctrace`).

**Ambiguity is never guessed at.** One session and no flag is needed; several, and the *analysis* command stops with a usage error (exit 2) that *lists the sessions* — quietly analyzing "the newest" would answer confidently about a run you weren't asking about. An `--agent` matching nobody is likewise a bad flag, not an empty report.

`export` and `view` are the one exception: the [dashboard](#html-dashboard) covers the whole project, so they never require `--session` and use the selectors to pick which of its three levels to open on.

Two extra diff shapes fall out of the selectors, both reusing the same differ:

```bash
# cross-session — the regression case: same agent, same turn, two runs
npx ctxdiff diff --session 4f3a2b1c9d8e:8 --session 9e8d7c6b5a4f:8 --agent researcher

# cross-agent — two agents inside one session
npx ctxdiff diff --session 4f3a2b1c9d8e --agent researcher:1 --agent writer:2
```

Both print a scope header naming the two sides (an ordinary same-session diff is unchanged). Timestamps are stored UTC and displayed in **your local timezone** with the offset shown — identical to the Python CLI for the same stored value.

### Percent of the context window

`18,400 tok` means nothing on its own. `18,400 / 200,000 — 9%` means you have room; `156,000 / 200,000 — 78%` means the next tool result is going to push something out of the window. State the window and every turn header becomes a share of it:

```
$ npx ctxdiff tokens --context-window 200000
turn 12 · 18,400 / 200,000 tok · 9.2%
  ...
turn 31 · 164,000 / 200,000 tok · ⚠ 82.0%
```

The `⚠` appears at **80%** and above — not because 80% is a failure (the provider would error if you exceeded the window, and you would know) but because that is where the *silent* failures start: sliding-window trimmers, a `max_tokens` reservation for the response, one large tool result arriving next turn. Past 80% the headroom is smaller than a typical tool output. The marker is compared against the percentage as printed, so `80.0%` is always marked and `79.9%` never is.

**Why you state the window.** ctxdiff ships **no model→context-window table**, deliberately — the same reason it ships no price table. Those numbers differ per model, per provider and per deployment and change under you; a stale one does not degrade, it lies. So it comes from you:

```bash
npx ctxdiff tokens --context-window 200000     # this invocation
export CTXDIFF_CONTEXT_WINDOW=200000           # every invocation, including the dashboard
```

The flag wins when both are set. `tokens`, `check`, `view` and `export` resolve the window through the same path, so a CI gate and the report you read beside it can never be scored against two different windows — and all four refuse a window of zero or less (`ctxdiff: --context-window must be greater than 0`, exit 2) rather than render a percentage of something that is not a window. With neither set, nothing changes — every command prints exactly the bytes it printed before percentages existed. `(~approx)` still applies: a share computed from a partly-estimated total is exactly as approximate as the total.

### Evicted tagged blocks

"The agent forgot the thing I told it" almost always has one mechanical cause: a block that *was* in the context is not in it any more. ctxdiff already classifies that as an eviction, and [`tracer.tag()`](#semantic-tagging) already records which blocks you considered load-bearing — so `npx ctxdiff tokens` names the join outright:

```
⚠ the block you tagged 'rag' at turn 3 was evicted at turn 6
  'Context: Refund policy: 30 days from delivery, unworn items only. Also…'
  [rag·user] 1,240 tok · entered at turn 3 · last present at turn 5 · never returned
```

**You tag it once.** [`tracer.tag()`](#semantic-tagging) applies to the *next* recorded call only, so a block tagged on turn 3 is stored `tagged` on turn 3 and heuristically labeled on every later turn carrying the same text. The report follows the **content**, not the call: if you ever vouched for that text anywhere in an agent's timeline, it stays vouched-for until it leaves. The headline quotes the turn the *tag* was applied; the facts line quotes the turn the *content* entered.

Four things it will **not** say, each on purpose:

| Not reported | Why |
|---|---|
| heuristically-labeled blocks | Only `tracer.tag()`ed blocks count. Every multi-turn agent evicts ordinary history — that *is* what a context window is — so including it would bury the one line worth reading. |
| a hand-off between agents | Pairing is per agent, the same rule the cache profiler uses. The researcher's block is "missing" from the writer's next call because it was never in the writer's context. |
| a block that comes back | Absent for one turn and back the next was crowded out, not forgotten. A block that leaves twice is reported once, for the departure it did not return from. |
| a block whose text was *edited* | A same-slot content change is `modified`, not `evicted` — whatever `ctxdiff diff` calls it, this report calls it. |

Also available as a CI assertion (`--no-tagged-eviction`, below) and as a dashboard panel.

### `ctxdiff check` — the CI gate

Everything above is something you *run and read*. `check` is the same analysis with a **threshold** attached and a **non-zero exit** when it's crossed, so a context regression fails the build instead of waiting to be noticed.

```bash
npx ctxdiff check --max-context 8000 --require-stable-prefix --no-dead-schemas
```

```
ctxdiff check · 6 turns · my-agent.ctrace (session 4f3a2b1c9d8e)
PASS  max-context            peak 7,214 tok at turn 4 · limit 8,000
FAIL  require-stable-prefix  2 breaks across 5 turn pairs · 1,482 tok re-billed
  turn 1 → turn 2 [agent:researcher] [system·modified] breaks 2/2 pairs — modified system block — first difference at char 39: '31' → '48'
FAIL  no-dead-schemas        1 of 2 registered tools never used · 220 tok/call (35.2% of avg context)
  tool schema 'deploy_to_production' registered but never invoked

check FAILED · 2 of 3 assertions failed
```

Every assertion is **opt-in**, and `check` with none is a usage error (exit 2) rather than a pass — a gate that verified nothing must never be the thing keeping a build green.

| Assertion | Fails when |
| --- | --- |
| `--max-context N` | any turn's context exceeds **N tokens** |
| `--max-context-pct P` + `--context-window N` | any turn exceeds **P%** of a context window **you state** |
| `--require-stable-prefix` | the prompt-cache prefix breaks anywhere in the run |
| `--no-dead-schemas` | a tool schema is registered but never invoked |
| `--no-tagged-eviction` | a block you [tagged](#semantic-tagging) entered an agent's context and later left it for good |
| `--max-growth N` / `--max-growth-pct P` | the context grows more than that between two consecutive turns of the same agent |

Plus the same `--project` / `--session` / `--agent` selectors, so one workflow can hold each agent to its own budget. **Exit codes**: `0` all-pass · `1` violated (or no trace to read) · `2` usage error.

ctxdiff ships **no model→window table** by design — windows differ per model and per provider and change under you — so `--max-context-pct` needs a window from you, as `--context-window N` or as `CTXDIFF_CONTEXT_WINDOW` in the job's environment ([the same resolution path](#percent-of-the-context-window) `tokens` and the dashboard use), and reports both the percentage and the token budget it works out to. A `--context-window` you *typed* and nothing reads is still a usage error; an inherited environment variable is not.

`--no-tagged-eviction` is the [eviction report](#evicted-tagged-blocks) with a threshold of zero — same analyzer, same sentence, same per-agent scoping. It reports three different PASSes, because "PASS" here has three meanings and only one is reassuring: `no tagged blocks in this run — nothing to lose`, `fewer than 2 turns — no pairs to check`, and `all 3 tagged blocks survived 4 turn pairs`.

`check` is a threshold layer over the analyzers you already read (it calls the same `tokens` and `cache` code paths), so a red build and a hand-run report can never tell two different stories.

**A floor is never certified.** Some blocks cost real tokens ctxdiff cannot know — an image passed as a **remote URL** (never fetched) or a `file_id`, or in a format the sniffer doesn't recognize. Those are stored as **zero** tokens rather than a fabricated guess, which makes that turn's total a lower bound, and comparing a lower bound to a budget has exactly one possible wrong answer: a silent PASS. So `check` refuses the comparison and reports the turn as `unmeasured` (`8 tok (~approx) · 2 blocks of unknown token cost — a floor, not a measurement`) instead. The header also names the trace and session the verdict came from, since with no `--project` the newest `*.ctrace` in the directory is what gets read.

### GitHub Action

Your agent's tests already run on every pull request. Point ctxdiff at that run and the context window becomes something CI *watches*:

```yaml
# .github/workflows/context-budget.yml
name: context budget
on: [pull_request]

jobs:
  budget:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: "22" }

      # 1. Run your agent's tests — `tracer.wrap(client)` is already in the code
      #    path, so the run leaves a .ctrace behind.
      - run: npm ci && npm test

      # 2. Assert the budget. Non-zero exit fails the job.
      - uses: salmanzafar949/ctxdiff@v1
        with:
          runtime: node          # `npx ctxdiff` — no pip, no setup-python
          project: my-agent.ctrace
          max-context: 8000
          require-stable-prefix: true
          no-dead-schemas: true
          no-tagged-eviction: true
```

The action installs ctxdiff, runs `check`, writes the PASS/FAIL table to the **job summary** (`$GITHUB_STEP_SUMMARY` — free, tokenless, and works on pull requests from forks, where a PR comment's `pull-requests: write` is read-only no matter what the repo settings say), and exits with the check's own status. Full input reference and the PR-comment escape hatch: [the Python README's GitHub Action section](../README.md#github-action).

---

## HTML dashboard

`npx ctxdiff view` (or `export`) produces **one HTML file** with everything inline — no CDN, no fonts, no external request of any kind — so it's safe to attach to a bug ticket or open offline. It renders byte-identically to the Python viewer.

### Three levels: agents → sessions → turns

The dashboard is **agent-first**, because "which agent" is the question you actually have when you open a project that holds many runs.

| Level | What it lists | Click a row to |
| --- | --- | --- |
| **1 — Agents** | every agent in the project, aggregated across **all** its sessions: sessions, calls, provider-reported spend, first and last seen | drill to that agent's sessions |
| **2 — Sessions** | every session that agent appeared in, newest first, with the session's **local** start time, that agent's turns in it (`2 / 3`), its spend, and the model | drill to that session's turns |
| **3 — Turns** | the turn scrubber, block diff, token heatmap, cache findings, block inspector and growth chart — scoped to the chosen agent within the chosen session | inspect a turn |

A breadcrumb (`all agents › researcher › 2026-07-21 22:42:30 +04:00`) walks back up, and the agent chips at level 3 change or clear the scope. **A single-agent, single-session project opens straight on level 3**, so the common case never clicks twice.

`view`/`export` cover the whole project, so neither needs `--session` even when the project holds many runs; the selectors instead preselect the opening level:

```bash
npx ctxdiff view                                   # all agents (level 1)
npx ctxdiff view --agent researcher                # that agent's sessions (level 2)
npx ctxdiff view --session 4f3a2b1c9d8e            # that session's turns (level 3)
npx ctxdiff view --agent researcher --session 4f3a2b1c
```

### Timestamps are converted in *your* browser

Every timestamp is stored UTC and rendered in the **viewer's** local timezone with the offset shown, at render time — never baked in at export. Two people in two timezones read the same bytes as their own wall clock, and it matches `npx ctxdiff sessions`' local-time column exactly.

### What gets embedded (and the cap)

Levels 1 and 2 are **never capped** — they are aggregates over every session's call rows (no block reads), so no session or agent is ever hidden however large the database. Level 3 detail — the part with real size — is embedded for the **25 most recent sessions**, plus whichever session `--session` names; older sessions still appear at level 2 with their totals, marked *detail not embedded*, and the page names the cap.

### Security

All trace-derived text — block text, **agent names**, session labels, provider and model strings — is rendered via `textContent`, never `innerHTML`, so untrusted trace data can never execute. Each call's stored `params` is reduced to `{ model }` on export, so sampling settings and API keys never reach the shareable file.

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

The CLI and the dashboard read from the configured store too. With `CTXDIFF_STORE` set (or after `configure()`), the read commands analyze a session in the database rather than looking for a `.ctrace` in the working directory:

```bash
npx ctxdiff sessions                       # every session in the store, oldest first
npx ctxdiff agents                         # every agent, aggregated across all of them
npx ctxdiff tokens --session 4f3a2b1c9d8e  # ...one of them (no flag needed if there's one)
npx ctxdiff diff --session 4f3a2b1c9d8e --turn 7 --turn 8
npx ctxdiff export --session 4f3a2b1c9d8e --out run.html
```

A shared database fills up with sessions fast, so the same [ambiguity rule](#selectors) applies to the analysis commands: with more than one session `--session` is required, and the error lists them. `export`/`view` need no session at all — the [dashboard](#html-dashboard) lists every agent and session in the store for you.

`--project PATH` (or its `--run` alias) always wins: a path names a file, so it reads that `.ctrace` even when a database is configured. The same rule applies on the write side — `trace.init(project, { path })` is always a local file. `export`/`view` need an explicit `--out` against a database, since there is no trace filename to derive one from.

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
| Gemini | `@google/genai` | `models.generateContent`, `models.generateContentStream` (AI Studio and Vertex AI — same client class, same adapter) |
| AWS Bedrock | `@aws-sdk/client-bedrock-runtime` | `send(new ConverseCommand(...))` and `send(new ConverseStreamCommand(...))` — see [AWS Bedrock](#aws-bedrock) |
| LangChain / LangGraph | *any* chat model | via `tracer.langchainHandler()` — see [LangChain & LangGraph](#langchain--langgraph) |

Sync and async clients are both handled. Streaming usage is folded from each provider's own events (OpenAI final-chunk `usage`; Anthropic `message_start` + `message_delta`; Gemini cumulative `usageMetadata`; Bedrock's single trailing `metadata` event) and recorded once the stream completes. An unrecognized client is returned unwrapped (with a warning), never throwing.

None of these SDKs is a dependency of `ctxdiff` — they are **optional peers**, detected by duck-typing whatever you pass. `npm i ctxdiff` still installs exactly one runtime dependency (`gpt-tokenizer`).

---

## How it works

1. **Capture.** `tracer.wrap(client)` returns a transparent `Proxy` over your client. It forwards everything untouched except the completion methods, which it intercepts to record the request's context — verbatim, wire-level — after the real call runs.
2. **Block model.** Each request is flattened into ordered **blocks** (a message, a content part, a tool schema, an image). A block's identity is `sha256(role · kind · normalized-text)`, so identical content is stored once and referenced many times. Diffing is an ordered hash-list comparison.
3. **Analysis.** `diff`, `tokens`, and `cache` are pure functions over the stored blocks — re-runnable, and the single source of truth the dashboard embeds.

### The block model

`role` ∈ `system · user · assistant · tool` · `kind` ∈ `message · content_part · tool_schema · image`. Labels (`system · user · history · rag · tool_schema · tool_output`) come from a cheap heuristic unless you override them with `tracer.tag(...)`.

### Images and multimodal content

An image never reaches the store as its bytes. A vision content part — an OpenAI `image_url` or `input_image`, an Anthropic `{ type: "image", source: … }`, a Gemini `inlineData`/`fileData` — becomes a block with `kind: "image"` that looks like this:

```
[image 1024×768 · ~765 tok]
```

- **The base64 is not stored and not tokenized.** A `data:image/png;base64,…` part used to be JSON-serialized into the block text, so a 100 KB screenshot bloated the `.ctrace` *and* was counted by the tokenizer as prose — tens of thousands of phantom tokens for an image that really costs a few hundred.
- **Identity is the pixels, plus what changes their cost.** The hash is taken over a sha256 of the image bytes, so the same screenshot re-sent every turn is **one** block referenced many times — `diff` reports it unchanged and the cache profiler sees a stable prefix — while two *different* images of the same size stay two blocks. The bytes are not the whole request, so the hash also covers OpenAI's `detail` (the same screenshot at `low` and at `high` costs 85 vs 765 tokens, and is two blocks) and anything else the content part carried — notably Anthropic's `cache_control`, so moving a cache breakpoint onto an image is a change the cache profiler reports rather than a silent one.
- **The count is the provider's own published vision formula** applied to dimensions read from the image header (PNG/JPEG/GIF/WebP, header-only, no image library added as a dependency). It is always `tokenMethod: "estimate"`, so a turn containing an image is reported as approximate — a vision estimate is never presented as an exact tokenizer count.
- **Nothing is ever fetched.** A remote `https://` URL or a provider-side file id is recorded as a reference and degrades to `[image]` with no estimate. ctxdiff stays local-first, and a trace's numbers never depend on whether the host was online.

Non-image multimodal parts (audio, video, PDFs, opaque file ids) are untouched. Both SDKs produce byte-identical descriptors, hashes and estimates; the full contract is in [`spec/ctrace-schema.md`](../spec/ctrace-schema.md#image-blocks).

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

**`gpt-tokenizer` is pinned to an exact version** (no caret), and so is Python's `tiktoken`. They are two independent reimplementations of the same `o200k_base` table on independent release cadences. A `.ctrace` is safe either way — a block's hash is `sha256(role ‖ kind ‖ text)` and token counts are *not* hashed — but every rendered number is downstream of a token count, so `npm install` picking up a newer minor could change published numbers with no ctxdiff commit behind it. A committed [golden corpus](https://github.com/salmanzafar949/ctxdiff/tree/main/spec/golden) makes **both** SDKs reproduce a frozen set of CLI outputs and dashboard hashes on every CI run. Re-pinning is deliberate: bump both, run `npm run golden:regen`, review the diff.

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

**Vertex AI** is the same client class in Vertex mode, captured by the same adapter with nothing extra to configure — only the endpoint differs, and ctxdiff detects on the client, not the URL:

```ts
const client = tracer.wrap(new GoogleGenAI({ vertexai: true, project: "my-project", location: "us-central1" }));
```

### AWS Bedrock

```ts
import { BedrockRuntimeClient, ConverseCommand } from "@aws-sdk/client-bedrock-runtime";

const client = tracer.wrap(new BedrockRuntimeClient({ region: "us-east-1" }));
await client.send(new ConverseCommand({
  modelId: "anthropic.claude-3-haiku-20240307-v1:0",
  system: [{ text: "You are a support agent." }],
  messages: [{ role: "user", content: [{ text: "What's your refund window?" }] }],
  inferenceConfig: { maxTokens: 256 },
}));
```

`ConverseStreamCommand` takes the identical request shape and is captured the same way — including usage, which Bedrock reports once, on the single trailing `metadata` event:

```ts
const { stream } = await client.send(new ConverseStreamCommand({ modelId, messages }));
for await (const event of stream) { /* every event, unchanged, in order */ }
```

Three things worth knowing:

- **The AWS SDK v3 has one method, not one per operation.** Everything goes through `client.send(command)`, so ctxdiff hooks `send` and dispatches on the *command*: `ConverseCommand` and `ConverseStreamCommand` are recorded from `command.input`, and **every other command passes straight through, unrecorded and silently** — `InvokeModelCommand` carries a raw provider-specific `body` string rather than the Converse shape, and embeddings/guardrail calls aren't context at all. One client can serve Converse and embeddings side by side without a line of log noise.
- **The streaming call returns an envelope**, `{ $metadata, stream }`, not the stream itself. ctxdiff proxies only the `stream` member and hands the rest of the object back untouched, so `response.$metadata` still works exactly as it would unwrapped. If the envelope has no `stream` at all, you get your own object back, unwrapped — capture is lost there, your call is not.
- **The same request hashes identically in Python.** The Converse wire shape is the same in both SDKs, so a `.ctrace` written here dedups against one written by `boto3` — pinned by a conformance test that runs the *real* Python adapter over the same payload (system blocks, tool schemas and an image included) and compares hashes.

Detection keys off the client (its class name, or the resolved config's `serviceId`), never the URL — and deliberately narrowly: other AWS SDK v3 clients share the same `{send, config}` shape and are left completely alone.

### LangChain & LangGraph

LangChain hands you a `ChatOpenAI`, not an `OpenAI`, so there is nothing for `wrap()` to take. Use the **callback handler** instead — LangChain's own extension point:

```ts
import { ChatOpenAI } from "@langchain/openai";

const handler = tracer.langchainHandler();
const llm = new ChatOpenAI({ model: "gpt-4o", callbacks: [handler] });
await llm.invoke("What's your refund window?");     // captured, with usage
```

For **LangGraph** — which propagates callbacks through the whole graph — attach it once, at invoke time, and every model call in every node is captured:

```ts
await graph.invoke(state, { callbacks: [tracer.langchainHandler({ agent: "researcher" })] });
```

- **Every provider** — `ChatOpenAI`, `ChatAnthropic`, `ChatVertexAI`, `ChatBedrockConverse` and anything else following the interface, each normalized by its own provider's adapter; one handler can serve several models in one run. All four branches are live and each is verified against the real integration's request body, so the JS handler now covers exactly what the Python one does.
- **Streaming, tool calls, errors** — LangChain reports the finished result either way, so a streamed call records once *with* usage; tool schemas, the assistant's tool call and the tool result all become blocks in the turn that sent them; a failed call is recorded as a failed call.
- **Multimodal turns keep every part.** A message carrying two text parts and an image is three blocks, with the image hashed over its *bytes* — so the same screenshot is one block whether it arrived through LangChain or through a direct capture, and its vision-token cost shows up in `ctxdiff tokens`. Verified against the real request bodies `@langchain/openai`, `@langchain/google-genai` and `@langchain/aws` put on the wire.
- **Hash identity is the point.** The handler rebuilds the request in the provider's own wire shape and hands it to the *same* adapter the direct path uses, so its blocks are hash-identical to what `tracer.wrap()` records for the same request — checked end to end and against the actual JSON body LangChain sent. Across SDKs, the same prompt through the [Python handler](../README.md#langchain--langgraph) produces the same hashes too, pinned as shared literals by both suites — with one documented exception: a **tool call** (see the cross-language note below).

No LangChain dependency is added: the handler is returned as a plain `CallbackHandlerMethods` object, which `callbacks: [...]` accepts directly.

---

## The `.ctrace` format

One run = one SQLite file, versioned by `schema_version` so an old or foreign file is rejected with a clear error rather than misread. The format is a shared cross-SDK contract documented in [`spec/ctrace-schema.md`](../spec/ctrace-schema.md) — a trace written by the JS SDK opens in the Python `ctxdiff` CLI and viewer, and vice-versa.

| Table | Row | Key columns |
|---|---|---|
| `run` | one per file | `project`, `provider`, `started_at`, `ctxdiff_version`, `schema_version` |
| `call` | one per LLM request | `seq`, `params` (JSON), `usage` (JSON), `latency_ms`, `error`, `agent`, `step` |
| `block` | one per **distinct** context unit | `content_hash` (PK), `role`, `kind`, `text`, `token_count`, `token_method` |
| `call_block` | membership of a block in a call | `call_id`, `block_id`, `position`, `label`, `label_source` |

**Block kinds:** `message` · `content_part` · `tool_schema` · `image` · **Token method:** `tiktoken` (exact) · `estimate` (heuristic, or a vision estimate for an `image` block)

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
npm test              # unit + cross-language conformance against the Python SDK
npm run test:golden   # the cross-SDK golden corpus, on its own
npm run golden:regen  # rewrite the goldens (needs the repo's Python venv)
```

`npm test` includes `test/golden.test.ts`, which rebuilds every fixture in [`spec/golden/corpus/`](https://github.com/salmanzafar949/ctxdiff/tree/main/spec/golden) with this SDK's tokenizer and compares against committed CLI output and dashboard hashes — **the same files the Python suite compares against.** Nothing in it skips: a missing expectation or an unpinned tokenizer is a failure. Regenerate whenever a change is *supposed* to move a number, and put the diff in the PR. Full rationale and the re-pinning procedure: **[spec/golden/README.md](https://github.com/salmanzafar949/ctxdiff/blob/main/spec/golden/README.md)**.

---

## License

`ctxdiff` is free and open source software, released under the [MIT License](https://github.com/salmanzafar949/ctxdiff/blob/main/LICENSE).
