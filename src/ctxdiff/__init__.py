"""ctxdiff — a context-window debugger for LLM agents.

Public surface is deliberately tiny: `from ctxdiff import trace`, then
`trace.init(...)` and `tracer.wrap(client)`.
"""
__version__ = "0.1.0.dev0"

# __version__ must exist before this import: trace.py transitively imports
# ctxdiff.store.ctrace, which does `from ctxdiff import __version__` at load time.
from ctxdiff import trace  # noqa: E402,F401

__all__ = ["trace"]
