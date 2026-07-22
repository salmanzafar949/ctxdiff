# ctxdiff

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Status: M1 · capture + store](https://img.shields.io/badge/status-M1%20%C2%B7%20capture%20%2B%20store-orange.svg)](#status)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**git diff for your agent's context window.** See exactly what your LLM saw — turn by turn, block by block.

`ctxdiff` is a local-first debugger for the context window of LLM agents. Wrap your OpenAI or Anthropic client in one line, run your agent, and every call's context is recorded — as content-hashed, deduplicated *blocks* — into a single-file SQLite trace you can inspect, diff, and share. Nothing leaves your machine.

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

---

## Status

`ctxdiff` is built in milestones. **Available today (M1): capture + store** — the foundation everything else reads from.

| Milestone | What it adds | State |
|-----------|--------------|-------|
| **M1 — Capture + Store** | One-line SDK wrapping, content-hashed `.ctrace` traces, fail-open capture, redaction, token counting | ✅ **available now** |
| M2 — CLI diff | `ctxdiff diff --turn 7 --turn 8` — git-style added/evicted/modified context | 🔜 planned |
| M3 — Token heatmap | Token allocation per turn, "schema bloat" detection | 🔜 planned |
| M4 — Cache profiler | Prompt-cache prefix-break detection + wasted-spend estimate | 🔜 planned |
| M5 — Web viewer | Local viewer with a time-travel scrubber + self-contained HTML export | 🔜 planned |

This README documents **M1** — the capture and storage layer, and the Python API for reading traces back. The `ctxdiff diff` / `tokens` / `cache` / `view` commands land in later milestones.

---

## Install

`ctxdiff` targets **Python ≥ 3.10**.

```bash
# from source (PyPI package coming soon)
git clone https://github.com/salmanzafar949/ctxdiff
cd ctxdiff
pip install -e .
```

The only runtime dependency is [`tiktoken`](https://github.com/openai/tiktoken) (for exact OpenAI token counts). The provider SDKs (`openai`, `anthropic`, …) are **not** dependencies — `ctxdiff` wraps whatever client you already use.

To run the real-SDK evaluation suite, install the optional extra:

```bash
pip install -e ".[eval]"   # openai, anthropic, langchain, respx — for tests only
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

## Supported providers

`ctxdiff` detects the provider from the client you pass to `wrap()` and applies the matching adapter. Detection keys off the client's module, so anything built on the OpenAI or Anthropic SDK works — including Azure and OpenAI-compatible OSS endpoints.

| Provider | Client | Notes |
|----------|--------|-------|
| **OpenAI** | `openai.OpenAI(...)` | Chat Completions |
| **Azure OpenAI** | `openai.AzureOpenAI(...)` | Same adapter, zero config |
| **Anthropic / Claude** | `anthropic.Anthropic(...)` | Messages API |
| **Open-source models** | `openai.OpenAI(base_url="http://localhost:11434/v1", ...)` | Any OpenAI-compatible endpoint — Ollama, vLLM, LM Studio, Together, Groq, … |
| **LangChain** | `langchain_openai.ChatOpenAI(...)` | Via client injection — see [LangChain](#langchain) |

Passing an unrecognized client raises immediately, so misconfiguration fails loudly at setup rather than silently at record time:

```python
tracer.wrap(some_unknown_client)
# ValueError: ctxdiff: unrecognized client module '...'; supported providers: ['anthropic', 'openai']
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
[ READ ]      CTrace.open(path) → runs, calls, blocks   (analyzers land in M2–M5)
```

**Capture is deliberately dumb; interpretation lives downstream.** The proxy records what was actually sent on the wire and nothing more. Whether a block is "a RAG chunk" or "history" is decided by labels, not baked into capture — so the recorder has no opinions to get wrong, and re-analysis of an old trace never needs a re-run.

### The block model

The smallest independently-diffable unit of context is a **block**: one message, one content part, or one tool schema. Each block's identity is `sha256(role + kind + text)`, so a stable system prompt reused across 40 turns is stored **once** and referenced 40 times. Diffing two turns (M2) then reduces to comparing two ordered lists of hashes.

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

### Reading a trace back

Traces are read through the `CTrace` API. A run has calls (one per LLM request, in `seq` order); each call has ordered blocks.

```python
from ctxdiff.store.ctrace import CTrace

ct = CTrace.open("runs/session-42.ctrace")

run = ct.get_run()
# Run(id, project, started_at, provider, models, ctxdiff_version)

for call in ct.get_calls():
    # Call(id, run_id, seq, params, usage, latency_ms, error)
    print(call.seq, call.params.get("model"), call.usage, call.latency_ms)

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
- **Anthropic** → a documented estimate, since there's no public local tokenizer (`token_method="estimate"`).

Estimates are always labeled as such — never presented as exact. If `tiktoken` is unavailable for any reason, counting degrades to an estimate rather than dropping the capture, and never reaches the network at record time.

---

## Provider recipes

### OpenAI

```python
from openai import OpenAI
client = tracer.wrap(OpenAI())
client.chat.completions.create(model="gpt-4o", messages=[...])
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

M1 (this release) is the capture + store foundation. Next:

- **M2 — CLI diff:** `ctxdiff diff --turn N --turn M` renders added / evicted / modified context blocks in a git-style view.
- **M3 — Token heatmap:** `ctxdiff tokens` shows where the budget went and flags unused tool-schema "bloat."
- **M4 — Cache profiler:** `ctxdiff cache` finds what breaks the provider prompt-cache prefix and estimates the wasted spend.
- **M5 — Web viewer:** `ctxdiff view` opens a local time-travel scrubber; `ctxdiff export` emits a self-contained HTML snapshot for bug tickets.

Also tracked: streaming-usage capture, async clients, a native LangChain callback integration, and populating `run.models`.

---

## Development

```bash
pip install -e ".[dev]"     # ctxdiff + pytest
pytest                      # unit suite

pip install -e ".[eval]"    # + real provider SDKs and respx
pytest tests/eval           # real-SDK integration tests (HTTP stubbed, no network, no keys)
```

The eval suite drives the real `openai`, `anthropic`, and `langchain` SDKs with their HTTP transport stubbed, so it needs no API keys and makes no network calls. It skips cleanly if the `eval` extra isn't installed.

---

## License

[Apache-2.0](LICENSE).
