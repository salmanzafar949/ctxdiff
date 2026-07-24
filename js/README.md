# ctxdiff (JavaScript/TypeScript SDK)

Local-first **context-window debugger for LLM agents** — "git diff for your
agent's context window." This is the JS/TS capture SDK: wrap your OpenAI client
and every call's context is recorded, content-hashed, into a local `.ctrace`
SQLite file.

The `.ctrace` it writes is **byte-compatible with the Python SDK**, so a trace
captured from a JS agent opens in the existing Python viewer:

```bash
ctxdiff view my-project-1a2b3c4d.ctrace
```

Requires **Node 22+** (uses the built-in `node:sqlite`). Zero storage
dependencies; one small dependency (a pure-JS tokenizer for exact OpenAI token
counts).

## Install

```bash
npm i ctxdiff
```

`openai` is a peer dependency — you bring your own client.

## Use

```ts
import OpenAI from "openai";
import { trace } from "ctxdiff";

const tracer = trace.init("my-project");
const client = tracer.wrap(new OpenAI());

// Use `client` exactly like a normal OpenAI client — chat.completions.create,
// responses.create, streaming (stream: true), and the .stream() helper are all
// captured transparently.
const res = await client.chat.completions.create({
  model: "gpt-4o",
  messages: [{ role: "user", content: "hello" }],
});

tracer.close();
```

Wrapping is **fail-open**: it never throws into your app, never breaks stream
iteration, never drops/reorders a chunk, and never alters your request. If
tracing can't run, your app runs unchanged.

> **Streaming note:** a streamed call is recorded once the stream completes —
> consumed to the end, broken out of early, or errored. A stream you obtain but
> **never iterate at all** is not recorded: JS has no deterministic finalizer,
> and we deliberately avoid GC-timed `FinalizationRegistry` recording. In
> practice you always iterate a stream you asked for, so this is a non-issue.

### Optional labeling

```ts
tracer.mark("retrieve");                 // sticky step label until changed
tracer.tag("rag", [retrievedDoc]);       // label the NEXT call's matching blocks
```

### Redaction

```ts
const tracer = trace.init("my-project", {
  redact: (block) => ({ ...block, text: scrub(block.text) }),
});
```

The block is hashed and token-counted **before** redaction, so identity/dedup
stays stable; only the stored text changes.

## Providers

`tracer.wrap(client)` auto-detects the provider from the client you pass:

| Provider  | Client                          | Methods captured                                                              |
| --------- | ------------------------------- | ----------------------------------------------------------------------------- |
| OpenAI    | `openai`                        | `chat.completions.create` / `.stream()`, `responses.create` / `.stream()`, `stream:true` |
| Anthropic | `@anthropic-ai/sdk`             | `messages.create` (+ `stream:true`), `messages.stream()`                       |
| Gemini    | `@google/genai`                 | `models.generateContent`, `models.generateContentStream`                       |

All are peer dependencies — you bring your own client(s). Streaming usage is
folded from the provider's own events (OpenAI final-chunk `usage`; Anthropic
`message_start` + `message_delta`; Gemini cumulative `usageMetadata`) and
recorded once the stream completes. Traces from any provider open in the same
Python `ctxdiff view`. (AWS Bedrock is not yet in the JS SDK.)

## CLI (`npx ctxdiff`)

Read-only analysis over any `.ctrace` — including ones written by the **Python**
SDK. Output is byte-identical to the Python `ctxdiff` CLI.

```bash
npx ctxdiff diff --turn 7 --turn 8      # git-style block diff between two turns
npx ctxdiff tokens                      # token heatmap + schema-bloat report
npx ctxdiff cache                       # prompt-cache prefix-break profiler
npx ctxdiff runs                        # list .ctrace files in the cwd
```

Flags mirror the Python CLI: `--turn` (twice for `diff`, once optional for
`tokens`), `--agent` to scope to one agent, `--run PATH` to pick a trace
(default: most recently modified `*.ctrace` in the cwd). A positional path also
works, e.g. `npx ctxdiff tokens my-run.ctrace`. (`view`/`export`/`demo` are not
in the JS CLI yet — use the Python CLI for those.)

## Format

The `.ctrace` format is documented as a shared cross-SDK contract in
[`spec/ctrace-schema.md`](../spec/ctrace-schema.md).
