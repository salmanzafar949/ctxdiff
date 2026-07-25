"""ctxdiff as an MCP server — the agent debugging an agent.

`ctxdiff mcp` speaks the Model Context Protocol over **stdio**, so the coding
agent already sitting in Claude Code / Cursor can answer "why did my agent break
at turn 8" by calling ctxdiff itself instead of asking its human to paste
terminal output back in. Agents re-read their tool list every session, which is
the one distribution channel that cannot be forgotten.

Three rules this package exists to hold, all of them consequences of what
ctxdiff is:

1. **Fourth renderer, never a fourth implementation.** Every number here comes
   from the SAME pure analyzers the CLI, the HTML dashboard and `ctxdiff check`
   read — `analyze_run`, `analyze_cache`, `diff_turns`, `analyze_evictions`,
   `Store.list_sessions`. A debugger that gives two answers to one question is
   worse than no debugger, so this package computes nothing itself: it opens a
   store (through the CLI's own resolver) and shapes what the analyzers return.
2. **Token discipline is the product.** We are a context-efficiency tool; an
   MCP server that dumps a 50 KB RAG chunk into the debugging agent's context
   window would refute its own thesis. Results are compact JSON — hashes,
   labels, token counts, and only the CHANGED hunk — under a hard byte cap.
   See `payload.py`.
3. **Text that leaves the machine is a consent boundary, and text that comes
   back in is an injection vector.** `--redact` turns off raw captured text
   everywhere (including `ctxdiff_block`), and everything that IS returned is
   wrapped in a `<captured-untrusted-input>` fence, because a `.ctrace` is full
   of attacker-influenced strings: user messages, retrieved documents, tool
   output. See `payload.py` and the server instructions in `server.py`.

The `mcp` SDK is an OPTIONAL extra (`pip install 'ctxdiff[mcp]'`). Nothing in
this module imports it — `ctxdiff mcp` checks for it and prints
`MISSING_EXTRA_HINT` rather than letting an ImportError traceback be the user's
first experience of the feature. `payload.py` and `tools.py` are likewise
SDK-free, so the whole result-shaping layer is importable, and testable, with
the extra absent; only `server.py` touches FastMCP."""
from __future__ import annotations

# The one-line install hint `ctxdiff mcp` prints when the extra is missing. Kept
# here — in the package's SDK-free entry module — so the CLI can show it without
# importing anything that would itself fail.
MISSING_EXTRA_HINT = (
    "ctxdiff: the MCP server needs the official `mcp` Python SDK, which ships "
    "as an optional extra — install it with `pip install 'ctxdiff[mcp]'` "
    "(everything else in ctxdiff works without it)"
)

__all__ = ["MISSING_EXTRA_HINT"]
