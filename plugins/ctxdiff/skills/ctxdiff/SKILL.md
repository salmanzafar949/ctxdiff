---
name: ctxdiff
description: Use when debugging LLM agent behavior or costs — "agent misbehaves/forgets at turn N", context window inspection, token attribution, prompt-cache breaks, dead tool schemas, or comparing what a model saw between turns or runs. Captures and diffs the full context of OpenAI/Anthropic/Gemini/Bedrock/LangChain calls (Python + JS/TS).
---

# ctxdiff — git diff for an LLM agent's context window

Local-first debugger for what the model **actually saw**, turn by turn. One wrap line records every call's full context into a single-file SQLite `.ctrace` (content-hashed, deduplicated blocks). Nothing leaves the machine.

Reach for it when the question is any of: *what exactly did the model see at turn N? what changed since the previous turn/run? where do the tokens go? what broke the prompt cache? which registered tools are never used?* Raw JSON logs can't answer these; ctxdiff's analyzers can.

## Capture (add one line, run the agent)

**Python** (`pip install ctxdiff`):
```python
from ctxdiff import trace
tracer = trace.init("my-project")            # appends a session to ./my-project.ctrace
client = tracer.wrap(OpenAI())               # also: Anthropic(), genai, boto3 bedrock clients
# multi-agent: tracer.wrap(client, agent="researcher"); steps: tracer.mark("plan")
tracer.close()                               # flushes; safe to omit in servers (see gotchas)
```

**JS/TS** (`npm i ctxdiff`):
```js
import { trace } from "ctxdiff";
const tracer = trace.init("my-project");
const client = tracer.wrap(new OpenAI());    // { agent: "researcher" } as 2nd arg for teams
```

**LangChain/LangGraph**: pass ctxdiff's callback handler instead of wrapping (see README section "LangChain / LangGraph").

Both SDKs write the same format — a trace captured in one opens in the other's CLI/viewer. Capture is fail-open: a ctxdiff error can never break the app.

## Analyze (run in the directory holding the .ctrace)

Python CLI: `ctxdiff` / `uvx ctxdiff` · JS: `npx ctxdiff` — identical output.

| Question | Command |
|---|---|
| What sessions/agents exist? | `ctxdiff sessions` · `ctxdiff agents` |
| What changed between turns? | `ctxdiff diff --turn 7 --turn 8` (`--session`, `--agent` to scope; two `--run trace:N` args for cross-run regression) |
| Where do tokens go? | `ctxdiff tokens` — per-turn bars by kind (system/user/history/tool_schema/tool_output) |
| What breaks the prompt cache? | `ctxdiff cache` — finds the exact changed characters in the prefix |
| CI budget gate | `ctxdiff check --max-context N --require-stable-prefix --no-dead-schemas` (non-zero exit; GitHub Action available) |
| Human-viewable dashboard | `ctxdiff view` / `ctxdiff export` (self-contained HTML) |

Reading the output:
- `⚠ schema bloat: X never used` → that tool's JSON schema taxes **every** call; register tools conditionally.
- `no provider usage reported` → all calls streamed without usage (see gotcha 1); numbers shown are honest estimates (`~approx`).
- In `diff`, `[history·assistant]`/`[tool_output]` blocks appearing/vanishing shows how the app rebuilds history — verify it matches the intended design.
- Images appear as `[image WxH · ~N tok]` blocks (hashed, deduped — not base64-counted).

## MCP (for clients without a shell)

`pip install 'ctxdiff[mcp]'` then register `ctxdiff mcp --runs-dir /path/to/traces` (stdio). Tools include `ctxdiff_explain(run, turn)` — one call answering "why did it break at turn N". `--redact` withholds raw text when the client's model is remote. Note: `--runs-dir` is **not recursive** — point it at the exact directory containing `.ctrace` files. With a shell available, prefer the CLI above.

## Gotchas (from real dogfooding)

1. **Streaming + OpenAI chat API records no usage unless the caller passes `stream_options: {"include_usage": true}`** — ctxdiff never injects it (wire-truth). Add it to get exact token counts on streamed calls.
2. **OpenAI-compatible endpoints** (Gemini/Anthropic/Ollama via `base_url`) are recorded as `provider=openai`. Tag reality yourself: `tracer.wrap(client, agent="gemini")`.
3. **Long-lived servers that never `close()`**: data sits in the SQLite `-wal` sidecar; the bare `.ctrace` file may be a near-empty shell until any ctxdiff CLI read checkpoints it. Run any read command (e.g. `ctxdiff sessions`) before copying/sharing a trace from a still-running process.
4. Only request-side context is stored (that's the point); response text lives in your app's own logs.
