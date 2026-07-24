# ctxdiff

[![PyPI](https://img.shields.io/pypi/v/ctxdiff.svg?logo=pypi&logoColor=white)](https://pypi.org/project/ctxdiff/)
[![npm](https://img.shields.io/npm/v/ctxdiff.svg?logo=npm&logoColor=white)](https://www.npmjs.com/package/ctxdiff)
[![CI](https://github.com/salmanzafar949/ctxdiff/actions/workflows/publish.yml/badge.svg)](https://github.com/salmanzafar949/ctxdiff/actions/workflows/publish.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/salmanzafar949/ctxdiff/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Node 22+](https://img.shields.io/badge/node-%E2%89%A522-brightgreen.svg?logo=node.js&logoColor=white)](https://nodejs.org)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/salmanzafar949/ctxdiff/blob/main/CONTRIBUTING.md)

**git diff for your agent's context window.** See exactly what your LLM saw — turn by turn, block by block.

**Pick your language:** [🐍 Python](#install) · [🟨 JavaScript / TypeScript](js/README.md) — same `.ctrace` format, same CLI, cross-compatible.

`ctxdiff` is a local-first debugger for the context window of LLM agents. Wrap your OpenAI, Anthropic, Gemini, or Bedrock client in one line, run your agent, and every call's context is recorded — as content-hashed, deduplicated *blocks* — into a single-file SQLite trace you can inspect, diff, and share. Nothing leaves your machine.

> Prompt wording is ~10% of the battle. The other 90% is **context engineering** — what the model sees, in what order, at what cost. When an agent misbehaves at turn 8, `ctxdiff` answers the three questions a raw JSON log can't: *what exactly did the model see, what changed since turn 7, and what did it cost?*

> 📦 **Two SDKs, one format.** ctxdiff ships for **Python** (`pip install ctxdiff`) and **JavaScript/TypeScript** (`npm i ctxdiff`). Both write the same `.ctrace` file and share the same CLI — a trace captured in one language opens in the other's viewer. Code samples below are **tabbed by language** (click to switch).

<details open>
<summary>🐍 <b>Python</b></summary>

```python
from ctxdiff import trace
from openai import OpenAI

tracer = trace.init("customer-support-agent")
client = tracer.wrap(OpenAI())          # ← the only line you add

client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's your refund window?"}],
)

tracer.close()                          # writes ./customer-support-agent-<id>.ctrace
```
</details>

<details open>
<summary>🟨 <b>JavaScript / TypeScript</b></summary>

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
</details>

Then `ctxdiff view` opens the self-contained dashboard — here debugging a **multi-agent** run (researcher + writer), with agent filtering, turn-by-turn diffs, an agent handoff, and light/dark themes:

![ctxdiff dashboard — a multi-agent run: agent chips and filtering, git-style turn diffs, token allocation, cache-break attribution, light and dark themes](https://raw.githubusercontent.com/salmanzafar949/ctxdiff/main/assets/ctxdiff-dashboard-agents.gif)

> **See it in 30 seconds — no API key, no setup:**
> ```bash
> pip install ctxdiff && ctxdiff demo        # Python
> npm  i       ctxdiff && npx ctxdiff demo    # JavaScript / TypeScript
> ```
> `demo` builds a sample multi-agent trace and opens this dashboard — a realistic research-pipeline run (two agents, real SDK shapes, zero network) already showing turn diffs, token/schema-bloat, a cache-prefix break, and an agent hand-off. Both SDKs produce a byte-identical dashboard.

**Jump to:** [How it's different](#how-its-different) · [Features](#features) · [Install](#install) · [Quickstart](#quickstart) · [The CLI](#the-cli) · [HTML dashboard](#html-dashboard) · [Storage backends](#storage-backends) · [Supported providers](#supported-providers) · [How it works](#how-it-works) · [Usage guide](#usage-guide) · [Provider recipes](#provider-recipes) · [The `.ctrace` format](#the-ctrace-format) · [Design principles](#design-principles) · [Roadmap](#roadmap) · [Development](#development)

---

## How it's different

`ctxdiff` isn't a tracing platform. Tools like LangSmith, Langfuse, and Phoenix show you a **list of calls** — a snapshot of each request, aggregated across your fleet, for monitoring cost and health over time. That's a different job.

`ctxdiff` shows you the **diff between calls** — what got *added, evicted, or modified* in the context window, turn by turn, down to the character that broke your prompt-cache prefix. It's the debugger you reach for when one specific run went wrong at turn 8, not the dashboard you watch in production.

|  | Observability platforms | ctxdiff |
|---|---|---|
| **Question** | Is my fleet healthy & cheap over time? | Why did *this* run break at turn 8? |
| **Shows you** | A list of calls (snapshot) | The **diff** between calls (delta) |
| **Scope** | Many runs, aggregated | One run, turn by turn |
| **Runs** | Hosted service / self-hosted server | Local — one `pip install`, one file |
| **Your data** | Sent to a platform | Never leaves your machine |

It's built to sit **alongside** your observability stack, not replace it. Use them for "is production okay"; use `ctxdiff` for "what exactly did the model see, and what changed."

---

## Features

**What ctxdiff does** *(each link jumps to the details)*:

- 🔌 **[One-line capture](#quickstart)** — `tracer.wrap(client)` records every LLM call's full context, verbatim, into a single-file SQLite `.ctrace`. [Fail-open by design](#fail-open-guarantee): a ctxdiff error can never break your app.
- 🧬 **[Content-hashed block storage](#the-block-model)** — every message, content part, and tool schema is a deduplicated block; a stable system prompt across 40 turns is stored once.
- 🌐 **[Seven provider surfaces](#supported-providers)** — OpenAI, Azure OpenAI, Anthropic, Google Gemini, AWS Bedrock (Converse), any OpenAI-compatible OSS endpoint (Ollama/vLLM/…), and LangChain via client injection.
- 🟩🟥🟨 **[Git-style turn diffing](#ctxdiff-diff---turn-n---turn-m)** — `ctxdiff diff --turn 7 --turn 8`: exactly which blocks were added, evicted, or modified (with char-level inline diffs) between any two turns.
- 📊 **[Token attribution](#ctxdiff-tokens---turn-n)** — `ctxdiff tokens`: where the budget goes per turn (system / rag / history / schemas…), reconciled against provider-reported usage, plus **schema-bloat detection** — tools you registered but never call, taxing every request.
- 💸 **[Prompt-cache profiling](#ctxdiff-cache)** — `ctxdiff cache`: finds exactly what breaks your cache prefix (down to the changed characters), counts re-billed tokens, and suggests the fix.
- 🖥️ **[Self-contained HTML dashboard](#html-dashboard)** — `ctxdiff view` / `ctxdiff export`: a one-file, zero-external-request dashboard with a turn scrubber, diff panel, token heatmap, cache findings, and block inspector — safe to attach to a bug ticket.
- 🏷️ **[Semantic tagging](#semantic-tagging)** — `tracer.tag("rag", chunks)` for exact provenance labels; a cheap heuristic covers the rest.
- 🤝 **[Multi-agent runs](#multi-agent-runs)** — `tracer.wrap(client, agent="researcher")` and `tracer.mark("step")` attribute every call to the agent (and step) that made it; `--agent` filters on `diff`/`tokens`/`cache`, and the dashboard colors each agent's turns. Cross-agent hand-offs are never miscounted as cache breaks.
- 🗄️ **[Pluggable storage](#storage-backends)** — local-first `.ctrace` by default; `ctxdiff.configure(store=PostgresStore(dsn=...))` (or `CTXDIFF_STORE=…`) once, and every run lands in your **PostgreSQL/MySQL** instead. Tables auto-create, drivers are optional extras, and a dead database degrades capture without ever touching your agent.
- 🔒 **[Privacy first](#redaction)** — local-first (no network, no telemetry), a redaction hook that runs before anything touches disk, and HTML exports that strip request params down to the model name.
- ✅ **[Honest numbers](#token-counting)** — exact `tiktoken` counts for OpenAI; estimates are always *marked* as estimates, never passed off as precise.

**What it doesn't do (yet):**

- ⏳ **Bedrock streaming usage** — Bedrock's `converse_stream` is a separate, not-yet-wrapped method; use the non-streaming `converse` if you need usage from Bedrock today. (This is the **only** streaming gap — OpenAI chat+Responses, Anthropic, and Gemini streaming usage are all captured, including the `.stream()` convenience-manager helpers; see [Streaming usage](#streaming-usage).)
- ⏳ **Live tail** — the dashboard is post-run; it doesn't update while the agent is still running.
- ⏳ **Background recording** — capture is synchronous on the call path (fast, but not zero-cost; threaded agents aren't recorded).
- ⏳ **Native LangChain/LangGraph callbacks** — [LangChain works via client injection](#langchain) today; a first-class callback handler is planned.
- ⏳ **VS Code extension** — the dashboard will be embeddable in an editor panel.

See [The CLI](#the-cli) below for every subcommand, with real sample output.

---

## Install

<details open>
<summary>🐍 <b>Python</b> (≥ 3.10)</summary>

```bash
pip install ctxdiff
```

The only runtime dependency is [`tiktoken`](https://github.com/openai/tiktoken) (for exact OpenAI token counts). The provider SDKs (`openai`, `anthropic`, …) are **not** dependencies — `ctxdiff` wraps whatever client you already use.

Install from source, or with the real-SDK eval extra:

```bash
git clone https://github.com/salmanzafar949/ctxdiff && cd ctxdiff && pip install -e .
pip install -e ".[eval]"   # openai, anthropic, google-genai, boto3, langchain, respx — for tests only
```

Storage is a local `.ctrace` file by default, with nothing extra to install. To keep traces in a database you already run, add the matching extra — see [Storage backends](#storage-backends):

```bash
pip install 'ctxdiff[postgres]'   # psycopg 3
pip install 'ctxdiff[mysql]'      # PyMySQL
```
</details>

<details>
<summary>🟨 <b>JavaScript / TypeScript</b> (Node ≥ 22)</summary>

```bash
npm i ctxdiff
```

Requires **Node ≥ 22** (uses the built-in `node:sqlite`). The only runtime dependency is a pure-JS tokenizer (`gpt-tokenizer`) for exact OpenAI token counts. The provider SDKs (`openai`, `@anthropic-ai/sdk`, `@google/genai`) are **optional peer dependencies** — `ctxdiff` wraps whatever client you already use and never imports them itself. The `.ctrace` it writes opens in the Python `ctxdiff view`, and vice-versa.
</details>

---

## Quickstart

Wrap a client, use it exactly as you normally would, then read the trace back.

<details open>
<summary>🐍 <b>Python</b></summary>

```python
from ctxdiff import trace
from ctxdiff.store.ctrace import CTrace
from openai import OpenAI

# 1. Start a trace and wrap your client
tracer = trace.init("support-agent")
client = tracer.wrap(OpenAI())

# 2. Use the client normally — every call is recorded
client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a support agent. Be precise."},
        {"role": "user", "content": "What's your refund window?"},
    ],
)
tracer.close()

# 3. Read the trace back
ct = CTrace.open(tracer.path)
for call in ct.get_calls():
    print(f"turn {call.seq}  usage={call.usage}")
    for cb in ct.get_call_blocks(call.id):
        b = cb.block
        print(f"  [{cb.label:<11}] {b.role:<9} {b.token_count:>4} tok  {b.text[:60]!r}")
ct.close()
```
</details>

<details>
<summary>🟨 <b>JavaScript / TypeScript</b></summary>

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
    { role: "system", content: "You are a support agent. Be precise." },
    { role: "user", content: "What's your refund window?" },
  ],
});
tracer.close();

// 3. Read the trace back
const ct = CTrace.open(tracer.path);
for (const call of ct.getCalls()) {
  console.log(`turn ${call.seq}  usage=`, call.usage);
  for (const cb of ct.getCallBlocks(call.id)) {
    const b = cb.block;
    console.log(`  [${cb.label}] ${b.role} ${b.tokenCount} tok  ${JSON.stringify(b.text.slice(0, 60))}`);
  }
}
ct.close();
```
</details>

Either way, turn 1 records the same blocks (`.ctrace` files are cross-compatible):

```
turn 1  usage={'prompt_tokens': 24, 'completion_tokens': 8, 'total_tokens': 32}
  [system     ] system       9 tok  'You are a support agent. Be precise.'
  [user       ] user         5 tok  "What's your refund window?"
```

A `.ctrace` is just a SQLite file — you can also inspect it with any SQLite tool:

```bash
sqlite3 support-agent-*.ctrace "SELECT seq, usage FROM call ORDER BY seq;"
```

---

## The CLI

> **🐍 Python:** `ctxdiff <command>` &nbsp;·&nbsp; **🟨 JavaScript:** `npx ctxdiff <command>` — same commands, same flags, **byte-identical output**. The examples below use the Python form; prefix with `npx` for the JS SDK. Either CLI reads any `.ctrace`, no matter which SDK wrote it.

Every subcommand reads a `.ctrace`; `--run PATH` picks which one, and defaults to the most recently modified `*.ctrace` in the current directory when omitted — the common case (one run in the working dir) needs no flag at all. Color is automatic (git-style ANSI) and turns off whenever stdout isn't a real terminal, or when [`NO_COLOR`](https://no-color.org) is set — the output below has `NO_COLOR=1` so it pastes cleanly.

### `ctxdiff diff --turn N --turn M`

Git-style block diff between two turns: added (`+`, green), evicted (`−`, red), and modified (`~`, yellow, with an inline char-level diff) blocks, with unchanged blocks folded into one summary line.

```
$ ctxdiff diff --turn 1 --turn 2
── turn 1 → turn 2 · 3 blocks changed · +56 −26 tokens ──
~ [system·system] You are a support agent. Be precise. Current time: 2026-07-24T10:00:0[-0-]{+4+}Z
+ [history·assistant] 'Checking that for you.'  +5 tok
+ [rag·user] 'Context: Refund policy: 30 days from delivery, unworn items only. Also…'  +25 tok
= 3 unchanged blocks · 138 tok
```

### `ctxdiff tokens [--turn N]`

Token allocation per turn as a proportional bar chart, one label slice per row, biggest spender first; reconciles against provider-reported usage when available (`Δ` line); appends a schema-bloat warning when a registered tool schema is never invoked anywhere in the run.

```
$ ctxdiff tokens
turn 1 · 164 tokens
  ████████████████████████       tool_schema       133 tok   81.1%
  █████                          system             26 tok   15.9%
  █                              user                5 tok    3.0%
  provider reports 55 prompt tokens · Δ -109

turn 3 · 255 tokens
  ████████████████               tool_schema       133 tok   52.2%
  ██████                         history            50 tok   19.6%
  ████                           user               32 tok   12.5%
  ███                            system             26 tok   10.2%
  ██                             tool_output        14 tok    5.5%
  provider reports 85 prompt tokens · Δ -170

⚠ schema bloat: issue_refund — 1 of 2 registered tools never used this run — 77 tok (37.7% of avg context) spent on dead schemas every call
```

A call whose total mixes any `estimate`-method blocks in with exact ones is marked `(~approx)` next to its token total — never presented as exact when it isn't.

### `ctxdiff cache`

Prefix-stability report across every consecutive turn pair: finds where the provider's byte-for-byte cache prefix breaks, attributes it to the responsible block, and estimates the wasted re-billed spend — price-free, since per-token discounts vary by provider and change over time.

```
$ ctxdiff cache
⚠ warning: [system·modified] breaks the prefix on every turn (2/2 pairs)
  'You are a support agent. Be precise. Current time: 2026-07-24T10:00:04Z'
  modified system block — first difference at char 69: '0' → '4'

stable prefix (min): 133 tokens
re-billed: 183 tokens
183 tokens re-billed across 2 turns that a stable prefix would have served from cache (cached input is typically billed at a fraction of the full input price — check your provider's current rates)
hint: a dynamic value inside an early system block breaks the prefix every turn — move volatile content below the stable blocks
```

A run with a stable prefix throughout prints a single green `✓ prefix stable across all N turn pairs` line instead.

### `ctxdiff runs`

Lists every `*.ctrace` in the working directory with its project, provider, and turn count — a quick "what runs do I have here" before picking one with `--run`. With a [database configured](#storage-backends) it lists that store's sessions instead, like every other read command.

### `ctxdiff export [--out FILE.html]` / `ctxdiff view [--no-open]`

Write (`export`) or write-and-open (`view`) the self-contained HTML dashboard — see [HTML dashboard](#html-dashboard) below.

---

## HTML dashboard

`ctxdiff view` opens a local time-travel dashboard for a run in your browser; `ctxdiff export --out run.html` writes the same dashboard to a path you choose, without opening anything — the one you attach to a bug ticket. Both call the same exporter, so they're always in sync.

The output is **one self-contained `.html` file**: the page, styles, script, and the entire run's data are embedded in a single JSON island — no CDN, no font, no image, no external request of any kind (asserted in tests: the file contains no `http://`/`https://` substring anywhere). It opens from a `file://` URL, works offline, and is safe to email or attach to an issue tracker.

Seven panels, all reading from the same precomputed analyzer output the CLI uses (one source of truth — the dashboard never re-implements diff/token/cache logic in JavaScript):

- **Scrubber** — a turn-by-turn strip across the top; click a bar or use ← → to jump between turns.
- **Turn diff** — the selected turn's added/evicted/modified blocks vs. the previous turn.
- **Token allocation** — the selected turn's label breakdown, same data as `ctxdiff tokens`.
- **Cache alignment** — every prefix break found across the run, same data as `ctxdiff cache`.
- **Blocks** — the full block list for the selected turn (role, kind, label, token count, an 8-char content-hash prefix).
- **Growth** — context size across turns, so a run that balloons is visible at a glance.
- **Header stats** — project, provider, run start time, distinct-vs-total block counts (the dedup story).

Block text is written into the page as a JSON island and rendered with `textContent` at view time — never `innerHTML` — so a captured block containing `</script>` or literal HTML markup can never execute or break out of the tag, even though it's shown verbatim. The one deliberate redaction on export: each call's stored `params` is reduced to `{"model": ...}` — sampling settings, API keys, or anything else that might have ridden along in `params` never makes it into the shareable file (block text redaction is still governed by your own `redact()` hook, applied earlier at capture time).

---

## Storage backends

By default `ctxdiff` is **local-first and zero-config**: `trace.init("my-agent")` writes `./my-agent.ctrace`, a plain SQLite file you can open, query, email or attach to a ticket. Nothing to install, nothing to run, no server. That default never changes on its own — everything below is opt-in.

When you'd rather keep traces in a database you already run (a shared team dashboard, an agent fleet across many containers, a place where `.ctrace` files can't live), point `ctxdiff` at it **once** and every later `trace.init()` follows:

```python
import ctxdiff
from ctxdiff import PostgresStore

ctxdiff.configure(store=PostgresStore(dsn="postgresql://user:pw@db.internal/agents"))

# ...from here on, unchanged:
tracer = ctxdiff.trace.init("support-agent")
client = tracer.wrap(OpenAI())
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
| **SQLite** (default) | — built in | nothing, or `SQLiteStore(path=...)` |
| **PostgreSQL** | `pip install 'ctxdiff[postgres]'` | `PostgresStore(dsn="postgresql://...")` |
| **MySQL / MariaDB** | `pip install 'ctxdiff[mysql]'` | `MySQLStore(dsn="mysql://...")` |

The drivers ([`psycopg`](https://www.psycopg.org/psycopg3/) 3 and [PyMySQL](https://github.com/PyMySQL/PyMySQL)) are **optional extras, imported lazily at connect time**. `ctxdiff`'s core still has exactly one runtime dependency (`tiktoken`), importing `ctxdiff` never imports a database driver, and a store configured for a backend whose extra isn't installed tells you what to install instead of crashing.

### Tables are created for you

On first connect the adapter runs `CREATE TABLE IF NOT EXISTS` for its four tables — `ctxdiff_run`, `ctxdiff_call`, `ctxdiff_block`, `ctxdiff_call_block` — in whatever database the DSN points at. **There is no migration step.** The tables are prefixed because they live in a database you share with your own application, and connecting again is a harmless no-op.

It's the same logical model as a `.ctrace` file, in every backend: sessions, calls, content-hashed blocks stored **once** and referenced by position, and call→block membership. Analyzers can't tell the difference — the same conformance suite runs against all three, against real PostgreSQL and MySQL servers as well as SQLite.

Same *semantics*, too, not just the same shape:

- **Sessions are ordered by write order**, never by `started_at`. Several containers writing into one database will disagree about the clock; ordering by an insert-order column (SQLite's `rowid`, `BIGSERIAL`, `AUTO_INCREMENT`) means "the newest session" is the one written last on every backend, and stays stable between reads.
- **Keys compare byte-exactly.** MySQL's default collation is case-insensitive, which would make two content hashes differing only in case the same block; ctxdiff pins `ascii_bin` on its id/hash columns so content-addressed dedup means what it says.
- **Free-text columns are unbounded** on every backend, so a 400-character provider error or tag label stores rather than rejecting the call.

### Reading from a database

The CLI and the dashboard read from the configured store too. With `CTXDIFF_STORE` set (or after `configure()`), the read commands analyze the **newest session** in the database rather than looking for a `.ctrace` in the working directory:

```bash
ctxdiff tokens                   # newest session in the configured store
ctxdiff diff --turn 7 --turn 8
ctxdiff runs                     # every session in the store, oldest first
ctxdiff export --out run.html    # same self-contained dashboard
```

`--run PATH` always wins: a path names a file, so it reads that `.ctrace` even when a database is configured. The same rule applies on the write side — `trace.init(project, path=...)` is always a local file.

### A database is never allowed to break your agent

The [fail-open guarantee](#fail-open-guarantee) covers a networked store completely, and matters more there than for a local file:

- **No database I/O ever happens on your call path.** Not the writes, and not the connect either: the session is opened by the run's single background writer thread, so `tracer.wrap()` returns immediately even against a database that is slow, wedged, or not there at all.
- **Connects and statements are bounded** (5s connect, 10s statement, by default — both configurable), with TCP keepalives and `tcp_user_timeout` so a server that completes its handshake and then stops answering can't wedge the writer either. `tracer.close()` is bounded by the same statement timeout.
- **A dead database degrades capture, never the run.** If the store can't be reached or created, `ctxdiff` logs **one** warning, records nothing, and your calls run exactly as if `ctxdiff` weren't there. It never silently falls back to writing a local file you didn't ask for.
- **A connection killed mid-run is reopened.** A pooler recycle, a failover or a database restart costs the write that was in flight, not the rest of the run.
- **A typo'd DSN behaves the same way** — a misconfigured trace destination is a tracing problem, and tracing problems must not take down the program being traced.

### Writing your own backend

`Store` and `StoreBackend` (`ctxdiff.store.base`) are small runtime-checkable protocols — seven methods and two, respectively. Anything satisfying them can be passed to `configure(store=...)`; no base class, no registration.

---

## Supported providers

`ctxdiff` detects the provider from the client you pass to `wrap()` and applies the matching adapter. Detection keys off the client's module, so anything built on the OpenAI or Anthropic SDK works — including Azure and OpenAI-compatible OSS endpoints.

| Provider | Client | Notes |
|----------|--------|-------|
| **OpenAI** | `openai.OpenAI(...)` | Chat Completions **and Responses API** |
| **Azure OpenAI** | `openai.AzureOpenAI(...)` | Same adapter, zero config |
| **Anthropic / Claude** | `anthropic.Anthropic(...)` | Messages API |
| **Google Gemini** | `google.genai.Client(...)` | Generate Content API — `models.generate_content` **and** `models.generate_content_stream` (streaming usage captured) |
| **AWS Bedrock** | `boto3.client("bedrock-runtime")` | Converse API (`client.converse(...)`) |
| **Open-source models** | `openai.OpenAI(base_url="http://localhost:11434/v1", ...)` | Any OpenAI-compatible endpoint — Ollama, vLLM, LM Studio, Together, Groq, … |
| **LangChain** | `langchain_openai.ChatOpenAI(...)` | Via client injection — see [LangChain](#langchain) |

Passing an unrecognized client raises immediately, so misconfiguration fails loudly at setup rather than silently at record time:

```python
tracer.wrap(some_unknown_client)
# ValueError: ctxdiff: unrecognized client module '...'; supported providers: ['anthropic', 'bedrock', 'gemini', 'openai']
```

---

## How it works

```
[ your agent ]
     │  client = tracer.wrap(OpenAI())
     ▼
[ CAPTURE ]   a transparent proxy intercepts the completion call
     │        · calls the real method first (host is never delayed or altered)
     │        · fail-open: a ctxdiff error can never break your app
     ▼
[ STORE ]     one SQLite .ctrace file per run
     │        · every message / content part / tool schema is a content-hashed "block"
     │        · identical blocks are stored once, referenced per call (dedup)
     ▼
[ READ ]      CTrace.open(path) → runs, calls, blocks
     ▼
[ ANALYZE ]   diff_turns / analyze_run / analyze_cache — pure functions the
     │        CLI and HTML viewer both call, so every number agrees
     ▼
[ RENDER ]    the CLI (colored text) or `ctxdiff view`/`export` (HTML)
```

**Capture is deliberately dumb; interpretation lives downstream.** The proxy records what was actually sent on the wire and nothing more. Whether a block is "a RAG chunk" or "history" is decided by labels, not baked into capture — so the recorder has no opinions to get wrong, and re-analysis of an old trace never needs a re-run.

### The block model

The smallest independently-diffable unit of context is a **block**: one message, one content part, or one tool schema. Each block's identity is `sha256(role + kind + text)`, so a stable system prompt reused across 40 turns is stored **once** and referenced 40 times. Diffing two turns then reduces to comparing two ordered lists of hashes.

---

## Usage guide

> The snippets in this section and in [Provider recipes](#provider-recipes) are shown in **Python**. The **JavaScript/TypeScript** API mirrors them one-to-one (`trace.init` → `tracer.wrap(new Client())` → `tracer.close()`, `CTrace.open(...)` to read back) — see the **[JS SDK README](js/README.md)** for JS-native recipes (async, streaming, `.stream()` helpers, tagging, multi-agent).

### Wrapping a client

`trace.init()` starts a run; `wrap()` returns a transparent proxy. The proxy behaves *exactly* like the original client — same attributes, same return values — it just records the completion call as a side effect.

```python
tracer = trace.init(
    "my-agent",                     # project name (labels the run)
    path="runs/session-42.ctrace",  # optional; defaults to ./my-agent-<id>.ctrace
)
client = tracer.wrap(OpenAI())
# ... use client ...
tracer.close()
```

`tracer.path` tells you where the trace was written. Call `tracer.close()` when the run is done to close the store cleanly.

### Async clients

`wrap()` transparently intercepts async clients too — `AsyncOpenAI`, `AsyncAnthropic`, and `genai.Client(...).aio` — via call-time awaitable detection, so `await`ed calls are captured exactly like sync ones, no extra config needed:

```python
client = tracer.wrap(AsyncOpenAI())
resp = await client.chat.completions.create(model="gpt-4o", messages=[...])
```

(Bedrock stays sync-only — boto3 has no first-party async client.)

### Streaming usage

`stream=True` calls (sync or async) are captured too, including token `usage` — the interceptor wraps the returned stream so every chunk still reaches your code unchanged and immediately, and records the call once the stream completes (exhausted, closed, or its `with`/`async with` block exited):

```python
stream = client.chat.completions.create(
    model="gpt-4o", messages=[...],
    stream=True, stream_options={"include_usage": True},  # OpenAI chat: opt in for usage
)
for chunk in stream:
    ...  # your code sees every chunk, unmodified
```

Whether usage is actually captured depends on what the provider puts on the wire:

- **Anthropic** and **OpenAI Responses** streams report usage unconditionally — no caller action needed.
- **OpenAI Chat Completions** streams only report usage on a final chunk when the *caller* passes `stream_options={"include_usage": True}`. ctxdiff never injects this for you (it would alter your own request) — without it, the call is still captured but `usage` is honestly `None`.
- **Bedrock** (`converse_stream`) is a separate, not-yet-wrapped method — use the non-streaming `converse` if you need usage from Bedrock today.

A stream you never fully consume, close, or use as a context manager still gets recorded, best-effort, on garbage collection — with whatever usage was accumulated before you moved on (possibly none).

**Gemini** has no `stream=True` kwarg — streaming is its own method, `generate_content_stream` (sync) / `client.aio.models.generate_content_stream` (async), returning a direct iterator rather than a kwarg-toggled stream or a `.stream()` manager. It's captured the same way, with usage reported unconditionally on every chunk (cumulative — ctxdiff keeps the latest chunk's totals, not a running sum):

```python
for chunk in client.models.generate_content_stream(model="gemini-2.0-flash", contents="..."):
    ...  # your code sees every chunk, unmodified
```

The `.stream()` convenience-manager helpers — the style each provider's own docs actually recommend — are captured the same way, sync and async:

```python
with client.messages.stream(model="claude-opus-4-8", max_tokens=1024, messages=[...]) as stream:
    for event in stream:
        ...  # your code sees every event, unmodified

with client.chat.completions.stream(model="gpt-4o", messages=[...]) as stream:
    ...

with client.responses.stream(model="gpt-4o", input="...") as stream:
    ...
```

`Anthropic.messages.stream`, OpenAI's `chat.completions.stream`, and OpenAI's `responses.stream` all work this way — `with`/`async with client.messages.stream(...) as stream:` — nothing is recorded until you actually enter the block (that's when the provider request fires), and the same usage rules above apply (Anthropic/Responses unconditional, Chat Completions needs your own `stream_options={"include_usage": True}`). Gemini has no equivalent `.stream()` manager — see `generate_content_stream` above. Bedrock's `converse_stream` is still not wrapped, same as the raw `stream=True` path above.

### Semantic tagging

Blocks are auto-labeled by a cheap heuristic (`system`, `user`, `history`, `tool_schema`, `tool_output`). For **exact** provenance — especially distinguishing retrieved RAG chunks from ordinary user text — tag the content *before* the call it belongs to:

```python
chunks = retriever.search("refund policy")          # your RAG retrieval
tracer.tag("rag", [c.text for c in chunks])         # applies to the NEXT call

client.chat.completions.create(
    model="gpt-4o",
    messages=[
        system_prompt,
        {"role": "user", "content": f"Context:\n{joined_chunks}\n\nAnswer: ..."},
    ],
)
```

Any block whose text contains a tagged string is stored with `label="rag"` and `label_source="tagged"`. Untagged apps lose nothing but label precision — capture, dedup, and token counting all work regardless. `tag()` accepts a list of strings or dicts (it reads a `text`/`content` field from dicts).

### Multi-agent runs

A single codebase often drives several agents — a researcher, a writer, a critic — sometimes across different providers, all within one run. Name each client's agent at wrap time, and optionally `mark()` the current step; every call is then attributed to the agent (and step) that made it, on one shared, monotonic global timeline.

```python
tracer = trace.init("research-pipeline")
researcher = tracer.wrap(OpenAI(), agent="researcher")     # per-agent adapter + recorder
writer     = tracer.wrap(Anthropic(), agent="writer")      # a DIFFERENT provider, same run

tracer.mark("gather")                                       # sticky: labels every later call…
researcher.chat.completions.create(model="gpt-4o", messages=[...])
tracer.mark("draft")                                        # …until you change or clear it
writer.messages.create(model="claude-sonnet-4-5", messages=[...])
tracer.close()
```

- **`wrap(client, agent=...)`** — each wrap builds its own provider adapter and recorder, so two agents on two providers each record correctly (no cross-contamination). `agent` is optional; unlabeled calls are grouped as `(unlabeled)`.
- **`mark(step)`** — sets a **sticky** step label applied to every subsequent call in the **current execution context** until the next `mark()`; `mark(None)` clears it. (Contrast `tag()`, which is next-call-only.) Stickiness is per context: `asyncio.gather`/`to_thread` copy the context per task, so a task's `mark()` never relabels a sibling's calls. **Caveat — raw thread pools:** a `ThreadPoolExecutor` reuses workers without resetting their context, so a `mark()` lingers on that worker — a later task on the same worker that does *not* call `mark()` inherits the previous step. Under a raw pool, call `mark()` at the start of every task, or use the scoped form below.
- **`with tracer.step("phase"): ...`** *(recommended under concurrency)* — scopes the step label to the block and **resets it on exit**, so it can't leak across logical tasks even in a reused thread pool (and stays correct under asyncio). A task that opens no `step()` block records `step=None`, never a sibling's leftover label.
- **`--agent NAME`** — filters `ctxdiff diff`, `tokens`, and `cache` to one agent's calls. Turn numbers stay global `seq` values everywhere; `diff --agent` validates that both `--turn` values belong to that agent.
- **Agent-aware analysis** — cache-prefix stability is computed **within each agent's own timeline**, so an adjacent cross-agent hand-off is never mistaken for a cache break. `ctxdiff tokens` prints a per-agent token summary, `ctxdiff runs` lists each trace's agents, and the [HTML dashboard](#html-dashboard) shows a colored chip per agent, an agent-colored underline on each turn bar, and an "agent hand-off" marker (diffing against that agent's *own* previous turn).

`ctxdiff tokens` also opens with a run-level rollup of **provider-reported** usage (input/output tokens, normalized across all four provider key shapes), with an honest coverage fraction and a per-agent breakdown:

```
run total · in 18,400 tok · out 640 tok (5/6 calls reported usage)
  researcher · in 12,900 · out 410
  writer     · in  5,500 · out 230
```

### Reading a trace back

Traces are read through the `CTrace` API. A run has calls (one per LLM request, in `seq` order); each call has ordered blocks. The same calls work against any backend — `CTrace` is just the SQLite implementation of the `Store` protocol, so a `PostgresStore(...).open_reader()` handle reads identically (see [Storage backends](#storage-backends)).

```python
from ctxdiff.store.ctrace import CTrace

ct = CTrace.open("runs/session-42.ctrace")

run = ct.get_run()
# Run(id, project, started_at, provider, models, ctxdiff_version)

for call in ct.get_calls():
    # Call(id, run_id, seq, params, usage, latency_ms, error, agent, step, provider)
    print(call.seq, call.params.get("model"), call.usage, call.latency_ms, call.agent)

    for cb in ct.get_call_blocks(call.id):
        # CallBlock(block, position, label, label_source)
        b = cb.block
        # Block(content_hash, role, kind, text, token_count, token_method)
        print(cb.position, cb.label, b.role, b.token_count, b.token_method)

ct.close()
```

Because a `.ctrace` is plain SQLite, ad-hoc queries work too:

```bash
# how many DISTINCT blocks vs total references? (shows dedup at work)
sqlite3 session-42.ctrace "SELECT COUNT(*) FROM block;"
sqlite3 session-42.ctrace "SELECT COUNT(*) FROM call_block;"
```

### Redaction

Context payloads are the most sensitive data in an AI stack — system prompts, retrieved customer documents, tool arguments — and `.ctrace` files get attached to bug tickets. A `redact` hook runs on every block **before it is written to disk**:

```python
import dataclasses
from ctxdiff.models import Block

def scrub(block: Block) -> Block:
    """Return a redacted copy of the block (Block is a frozen dataclass)."""
    return dataclasses.replace(block, text=my_pii_scrubber(block.text))

tracer = trace.init("my-agent", redact=scrub)
```

The block's `content_hash` is computed from the **original** text (so dedup stays stable), and only the stored text is replaced. A redactor that raises does not break the run — the block's text falls back to `"[redaction-error]"` and capture continues.

> **Scope:** redaction applies to block text. Request `params`, provider `usage`, and error names are stored as-is — keep that in mind if you pass sensitive values as sampling params.

### Fail-open guarantee

**A debugging tool must never break the program it debugs.** Every capture path is wrapped so that *any* error inside `ctxdiff` — an adapter bug, a full disk, a tokenizer failure — is caught, logged once, and swallowed; your original call proceeds and returns its real result. The **only** exception that propagates is your own LLM call's error, re-raised unchanged:

```python
try:
    client.chat.completions.create(...)   # raises RateLimitError from the provider
except RateLimitError:
    ...   # you still get YOUR error — and the failed call is recorded with error set
```

### Token counting

Every block records a `token_count` and an honest `token_method`:

- **OpenAI-family** → exact counts via `tiktoken` (`token_method="tiktoken"`).
- **Anthropic, Gemini, Bedrock** → a documented estimate, since none publishes a local tokenizer (`token_method="estimate"`).

Estimates are always labeled as such — never presented as exact. If `tiktoken` is unavailable for any reason, counting degrades to an estimate rather than dropping the capture, and never reaches the network at record time.

---

## Provider recipes

> **Python** below. For **JavaScript/TypeScript** — OpenAI (chat + Responses), Anthropic, and Gemini, including streaming and `.stream()` helpers — see the **[JS SDK README → Provider recipes](js/README.md#provider-recipes)**. (Azure, Bedrock, OSS-endpoint, and LangChain recipes are Python-only for now.)

### OpenAI

```python
from openai import OpenAI
client = tracer.wrap(OpenAI())
client.chat.completions.create(model="gpt-4o", messages=[...])
```

### OpenAI Responses API

The same `wrap()` also captures `client.responses.create(...)` — the Responses API the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) builds on — off the *same* wrapped client, no separate call. `instructions` is captured as the leading system block, flat `tools` schemas next, then `input` (string or a list of message/tool items); usage is read from `input_tokens`/`output_tokens` instead of `prompt_tokens`/`completion_tokens`, and `previous_response_id` is kept in params since it's chain linkage, not content.

```python
from openai import OpenAI
client = tracer.wrap(OpenAI())
client.responses.create(
    model="gpt-4o",
    instructions="You are a support agent.",
    input="What's your refund window?",
)
```

Works async too, exactly like the chat path:

```python
resp = await tracer.wrap(AsyncOpenAI()).responses.create(model="gpt-4o", input="hi")
```

### Azure OpenAI

No special configuration — Azure clients live in the `openai` package, so the OpenAI adapter applies automatically.

```python
from openai import AzureOpenAI
client = tracer.wrap(AzureOpenAI(
    azure_endpoint="https://<resource>.openai.azure.com",
    api_version="2024-02-01",
))
client.chat.completions.create(model="<deployment-name>", messages=[...])
```

### Anthropic / Claude

The Anthropic adapter handles the top-level `system` field and `input_tokens`/`output_tokens` usage shape.

```python
from anthropic import Anthropic
client = tracer.wrap(Anthropic())
client.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    system="You are a support agent.",
    messages=[{"role": "user", "content": "What's your refund window?"}],
)
```

### Google Gemini

The Gemini adapter handles `google-genai`'s shape: `config.system_instruction`/`config.tools` (a single bag that also carries sampling params like `temperature`) and `contents` (a string, or a list of role/parts entries).

```python
from google import genai
client = tracer.wrap(genai.Client(api_key="..."))
client.models.generate_content(
    model="gemini-2.0-flash",
    contents="What's your refund window?",
    config={"system_instruction": "You are a support agent."},
)
```

### AWS Bedrock

The Bedrock adapter handles boto3's `bedrock-runtime` Converse API: `system` (a list of `{"text": ...}` blocks, never a bare string), `messages`, `toolConfig.tools`, and `inferenceConfig`'s sampling fields — detection keys off the client's *class name* (`BedrockRuntime`), since every boto3 service client shares the same `botocore.client` module.

```python
import boto3
client = tracer.wrap(boto3.client("bedrock-runtime", region_name="us-east-1"))
client.converse(
    modelId="anthropic.claude-3-haiku-20240307-v1:0",
    system=[{"text": "You are a support agent."}],
    messages=[{"role": "user", "content": [{"text": "What's your refund window?"}]}],
    inferenceConfig={"maxTokens": 256},
)
```

### Open-source models

Any model served behind an OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, Together, Groq, …) is just an `openai.OpenAI` client with a custom `base_url` — it captures identically:

```python
from openai import OpenAI
client = tracer.wrap(OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"))
client.chat.completions.create(model="llama3", messages=[...])
```

The same holds for **aggregators and proxies that speak the OpenAI API** — no extra code:

- **[OpenRouter](https://openrouter.ai)** — one endpoint in front of hundreds of models: `OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)`, then `wrap()` it. Every model you route through it is captured.
- **[LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy)** — a self-hosted OpenAI-compatible gateway over 100+ providers: point `OpenAI(base_url="http://localhost:4000")` at it and `wrap()`. Captures whatever the proxy fronts.

Because detection keys off the `openai` client module, both are captured by the OpenAI adapter with zero ctxdiff-specific configuration.

### LangChain

LangChain holds its own SDK client internally, so wrap the underlying OpenAI client and inject the proxy into `ChatOpenAI`:

```python
from openai import OpenAI
from langchain_openai import ChatOpenAI

oa = OpenAI()
wrapped = tracer.wrap(oa)

llm = ChatOpenAI(
    client=wrapped.chat.completions,   # inject the wrapped resource
    root_client=wrapped,
    model="gpt-4o",
)

llm.invoke("What's your refund window?")   # captured, with usage
```

> Wrapping the `ChatOpenAI` object directly raises (it isn't an SDK client) — inject as above.
> **Streaming caveat:** with `streaming=True`, the call is captured and the stream proxy records once it completes — but LangChain's default `ChatOpenAI` doesn't pass `stream_options={"include_usage": True}` itself, so `usage` still comes back `None` unless you configure LangChain to send it (see [Streaming usage](#streaming-usage) above).

---

## The `.ctrace` format

One run = one SQLite file. The schema is small and stable, versioned by `schema_version` so an old or foreign file is rejected with a clear error rather than misread.

| Table | Row | Key columns |
|-------|-----|-------------|
| `run` | one per file | `project`, `provider`, `started_at`, `ctxdiff_version`, `schema_version` |
| `call` | one per LLM request | `seq`, `params` (JSON), `usage` (JSON), `latency_ms`, `error` |
| `block` | one per **distinct** context unit | `content_hash` (PK), `role`, `kind`, `text`, `token_count`, `token_method` |
| `call_block` | membership of a block in a call | `call_id`, `block_id`, `position`, `label`, `label_source` |

**Block labels:** `system` · `user` · `history` · `rag` · `tool_schema` · `tool_output`
**Label source:** `heuristic` (auto) · `tagged` (via `tracer.tag`)
**Token method:** `tiktoken` (exact) · `estimate`

Because blocks are content-addressed and stored once, a long run with a stable prefix stays compact, and future diffing is a hash comparison.

---

## Design principles

- **Local-first.** No network calls, no telemetry, no external services. Your context never leaves your machine.
- **Fail-open, always.** Capture can never break your application.
- **Wire-level truth.** The proxy records what was actually sent, verbatim; interpretation is a separate, re-runnable layer.
- **Honest numbers.** Estimated token counts are always labeled as estimates.
- **One file per run.** The `.ctrace` *is* the shareable artifact — no bundle, no server.

---

## Roadmap

Everything under [What it doesn't do (yet)](#features) is the roadmap, in rough priority order: Bedrock streaming usage (`converse_stream`), live tail, background recording, a native LangChain callback handler, and the VS Code extension — plus smaller items tracked in the issues (e.g. rolling per-call model ids up onto `run.models`).

---

## Development

```bash
pip install -e ".[dev]"     # ctxdiff + pytest
pytest                      # unit suite

pip install -e ".[eval]"    # + real provider SDKs and respx
pytest tests/eval           # real-SDK integration tests (HTTP stubbed, no network, no keys)
```

The eval suite drives the real `openai`, `anthropic`, `google-genai`, `boto3`, and `langchain` SDKs with their HTTP transport stubbed (`respx` for httpx-based SDKs, `botocore.stub.Stubber` for boto3), so it needs no API keys and makes no network calls. It skips cleanly if the `eval` extra isn't installed.

---

## License

[Apache-2.0](https://github.com/salmanzafar949/ctxdiff/blob/main/LICENSE).
