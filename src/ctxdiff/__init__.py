"""ctxdiff — a context-window debugger for LLM agents.

Public surface is deliberately tiny: `from ctxdiff import trace`, then
`trace.init(...)` and `tracer.wrap(client)`.

Storage is local-first by default — each `trace.init(project)` writes
`./<project>.ctrace`, a plain SQLite file, with no configuration at all. To
point ctxdiff at a database you already run, configure a backend ONCE at
startup and every later `trace.init()` follows:

    import ctxdiff
    from ctxdiff import PostgresStore
    ctxdiff.configure(store=PostgresStore(dsn="postgresql://user@host/agents"))

...or set `CTXDIFF_STORE` and change no code at all. Tables are created if they
don't exist; there is no migration step.
"""
__version__ = "0.5.0"

# __version__ must exist before this import: trace.py transitively imports
# ctxdiff.store.ctrace, which does `from ctxdiff import __version__` at load time.
from ctxdiff import trace  # noqa: E402,F401
from ctxdiff.store import (  # noqa: E402
    MySQLStore,
    PostgresStore,
    SQLiteStore,
    configure,
)

__all__ = ["trace", "configure", "SQLiteStore", "PostgresStore", "MySQLStore"]
