# ctxdiff

[![PyPI](https://img.shields.io/pypi/v/ctxdiff.svg)](https://pypi.org/project/ctxdiff/)
[![CI](https://github.com/salmanzafar949/ctxdiff/actions/workflows/publish.yml/badge.svg)](https://github.com/salmanzafar949/ctxdiff/actions/workflows/publish.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/salmanzafar949/ctxdiff/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/salmanzafar949/ctxdiff/blob/main/CONTRIBUTING.md)

**git diff for your agent's context window.** See exactly what your LLM saw — turn by turn, block by block.

`ctxdiff` is a local-first debugger for the context window of LLM agents. Wrap your OpenAI, Anthropic, Gemini, or Bedrock client in one line, run your agent, and every call's context is recorded — as content-hashed, deduplicated *blocks* — into a single-file SQLite trace you can inspect, diff, and share. Nothing leaves your machine.

> Prompt wording is ~10% of the battle. The other 90% is **context engineering** — what the model sees, in what order, at what cost. When an agent misbehaves at turn 8, `ctxdiff` answers the three questions a raw JSON log can't: *what exactly did the model see, what changed since turn 7, and what did it cost?*

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

Then `ctxdiff view` opens the self-contained dashboard — here debugging a **multi-agent** run (researcher + writer), with agent filtering, turn-by-turn diffs, an agent handoff, and light/dark themes:

![ctxdiff dashboard — a multi-agent run: agent chips and filtering, git-style turn diffs, token allocation, cache-break attribution, light and dark themes](https://raw.githubusercontent.com/salmanzafar949/ctxdiff/main/assets/ctxdiff-dashboard-agents.gif)

> **See it in 30 seconds — no API key, no setup:**
> ```bash
> pip install ctxdiff
> ctxdiff demo          # builds a sample multi-agent trace and opens this dashboard
> ```
> A realistic research-pipeline run (two agents, real SDK shapes, zero network) already showing turn diffs, token/schema-bloat, a cache-prefix break, and an agent hand-off.

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
- 🔒 **[Privacy first](#redaction)** — local-first (no network, no telemetry), a redaction hook that runs before anything touches disk, and HTML exports that strip request params down to the model name.
- ✅ **[Honest numbers](#token-counting)** — exact `tiktoken` counts for OpenAI; estimates are always *marked* as estimates, never passed off as precise.

**What it doesn't do (yet):**

- ⏳ **Streaming usage** — streamed calls are captured, but token `usage` isn't (the response is a stream object at record time).
- ⏳ **Live tail** — the dashboard is post-run; it doesn't update while the agent is still running.
- ⏳ **Background recording** — capture is synchronous on the call path (fast, but not zero-cost; threaded agents aren't recorded).
- ⏳ **Native LangChain/LangGraph callbacks** — [LangChain works via client injection](#langchain) today; a first-class callback handler is planned.
- ⏳ **VS Code extension** — the dashboard will be embeddable in an editor panel.

See [The CLI](#the-cli) below for every subcommand, with real sample output.

---

## Install

`ctxdiff` targets **Python ≥ 3.10**.

```bash
pip install ctxdiff
```

<details>
<summary>…or install from source</summary>

```bash
git clone https://github.com/salmanzafar949/ctxdiff
cd ctxdiff
pip install -e .
```
</details>

The only runtime dependency is [`tiktoken`](https://github.com/openai/tiktoken) (for exact OpenAI token counts). The provider SDKs (`openai`, `anthropic`, …) are **not** dependencies — `ctxdiff` wraps whatever client you already use.

To run the real-SDK evaluation suite, install the optional extra:

```bash
pip install -e ".[eval]"   # openai, anthropic, google-genai, boto3, langchain, respx — for tests only
```

---

## Quickstart

Wrap a client, use it exactly as you normally would, then read the trace back.

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

Lists every `*.ctrace` in the working directory with its project, provider, and turn count — a quick "what runs do I have here" before picking one with `--run`.

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

## Supported providers

`ctxdiff` detects the provider from the client you pass to `wrap()` and applies the matching adapter. Detection keys off the client's module, so anything built on the OpenAI or Anthropic SDK works — including Azure and OpenAI-compatible OSS endpoints.

| Provider | Client | Notes |
|----------|--------|-------|
| **OpenAI** | `openai.OpenAI(...)` | Chat Completions **and Responses API** |
| **Azure OpenAI** | `openai.AzureOpenAI(...)` | Same adapter, zero config |
| **Anthropic / Claude** | `anthropic.Anthropic(...)` | Messages API |
| **Google Gemini** | `google.genai.Client(...)` | Generate Content API (`models.generate_content`) |
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
- **`mark(step)`** — sets a **sticky** step label applied to every subsequent call across all agents until the next `mark()`; `mark(None)` clears it. (Contrast `tag()`, which is next-call-only.)
- **`--agent NAME`** — filters `ctxdiff diff`, `tokens`, and `cache` to one agent's calls. Turn numbers stay global `seq` values everywhere; `diff --agent` validates that both `--turn` values belong to that agent.
- **Agent-aware analysis** — cache-prefix stability is computed **within each agent's own timeline**, so an adjacent cross-agent hand-off is never mistaken for a cache break. `ctxdiff tokens` prints a per-agent token summary, `ctxdiff runs` lists each trace's agents, and the [HTML dashboard](#html-dashboard) shows a colored chip per agent, an agent-colored underline on each turn bar, and an "agent hand-off" marker (diffing against that agent's *own* previous turn).

`ctxdiff tokens` also opens with a run-level rollup of **provider-reported** usage (input/output tokens, normalized across all four provider key shapes), with an honest coverage fraction and a per-agent breakdown:

```
run total · in 18,400 tok · out 640 tok (5/6 calls reported usage)
  researcher · in 12,900 · out 410
  writer     · in  5,500 · out 230
```

### Reading a trace back

Traces are read through the `CTrace` API. A run has calls (one per LLM request, in `seq` order); each call has ordered blocks.

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
> **Streaming caveat:** with `streaming=True`, the call is captured but token `usage` is not (the interceptor sees the stream before it's consumed). Streaming usage capture is tracked for a later milestone.

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

Everything under [What it doesn't do (yet)](#features) is the roadmap, in rough priority order: streaming usage capture, live tail, background recording, a native LangChain callback handler, and the VS Code extension — plus smaller items tracked in the issues (e.g. rolling per-call model ids up onto `run.models`).

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
