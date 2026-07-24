"""The storage contract every ctxdiff backend implements — the seam that lets a
user point ctxdiff at their own database instead of a local `.ctrace` file.

Two protocols, deliberately small, both derived from what the existing SQLite
store (`CTrace`) already does and what the rest of the codebase already calls:

- `Store` — a handle bound to ONE session (one `run` row). The recorder writes
  through it; the analyzers/CLI/viewer read through it. Nothing else.
- `StoreBackend` — the configurable, connection-less DESCRIPTION of where data
  lives (`SQLiteStore(path=...)`, `PostgresStore(dsn=...)`, `MySQLStore(...)`).
  `trace.init()` asks it to open a new session; the CLI asks it for a reader.
  A backend holds NO connection until one of those is called, which is what
  makes `ctxdiff.configure(store=PostgresStore(dsn=...))` cheap, side-effect
  free, and safe to evaluate at import time in a user's module.

The row dataclasses (`Run`/`Session`/`Call`) live here rather than in the SQLite
module because they are the backend-independent shape every store returns; they
are re-exported from `ctxdiff.store.ctrace` so existing imports keep working.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from ctxdiff.models import CallBlock


def parse_started_at(value: str) -> datetime:
    """Parse a session's stored `started_at` into a tz-AWARE UTC datetime,
    tolerant of BOTH formats a store may carry: the canonical UTC-with-offset
    string new sessions write (`...+00:00`, or a trailing `Z`) AND a legacy
    naive/UTC-without-offset string older rows used. A naive value is assumed
    to be UTC and coerced to aware, so downstream local-timezone rendering is
    always unambiguous regardless of which format produced the row. Any value
    ISO-parsing can't make sense of raises ValueError to the caller."""
    # `fromisoformat` accepts a trailing 'Z' only on 3.11+; normalize it first
    # so both spellings of UTC round-trip identically.
    text = value.strip()
    if text.endswith("Z") or text.endswith("z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        # Legacy naive row: it was written as UTC, so stamp UTC onto it rather
        # than letting it be interpreted in the reader's local zone.
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class EmptyStoreError(ValueError):
    """A store that exists and is readable but holds NO sessions yet.

    A `ValueError` subclass so every existing `except ValueError` / `raises(
    ValueError)` caller keeps working unchanged; its own type so the one caller
    that must tell "nothing recorded yet" apart from "this store is broken" —
    `ctxdiff runs`, which prints an empty listing for the former and an error
    for the latter — can do so without string-matching the message."""


@dataclass(frozen=True)
class Run:
    """One agent execution (one session), as stored in the `run` table."""
    id: str
    project: str
    started_at: str
    provider: str
    models: list[str]
    ctxdiff_version: str


@dataclass(frozen=True)
class Session:
    """A one-line summary of one session in a project store, as returned by
    `Store.list_sessions()` — the shape a session picker lists from. `agents` is
    the set of distinct agent labels seen on this session's calls, in
    first-appearance order ([] for a single-agent/pre-v2 session); `turn_count`
    is how many calls it holds. `started_at` is the raw stored string — use
    `parse_started_at()` for a tz-aware datetime to render in local time."""
    id: str
    project: str
    started_at: str
    provider: str
    models: list[str]
    agents: list[str]
    turn_count: int


@dataclass(frozen=True)
class Call:
    """One LLM request/response ('turn'), as stored in the `call` table.
    `agent`/`step`/`provider` are the v2 attribution fields (all nullable): the
    agent that made the call, the sticky step label active at the time, and the
    provider it went through. They surface as None for a v1 file read under v2
    code, and default to None so pre-v2 callers/tests can construct a Call
    without them."""
    id: str
    run_id: str
    seq: int
    params: dict
    usage: dict | None
    latency_ms: int | None
    error: str | None
    agent: str | None = None
    step: str | None = None
    provider: str | None = None


@runtime_checkable
class Store(Protocol):
    """A live handle to ONE session in a ctxdiff store — the only surface the
    recorder and the readers ever touch. Seven methods, each earning its place
    because something in the codebase already calls it:

    Write side (the capture path, `capture/recorder.py`):
    - `record_call` — persist one turn plus its ordered blocks atomically, with
      content-hash dedup on the blocks and call->block membership carrying
      position/label. Returns the new call id.
    - `note_model` — roll a per-call model id up onto the session's `models`
      list (first-seen order, deduped). Called by `record_call` and directly by
      tests; part of the write contract because the run's model is only ever
      really known per call.

    Read side (`analyze/*`, `cli/main.py`, `viewer/export.py`):
    - `list_sessions` — every session in the store, oldest first, with agents
      and turn counts (the session-picker query).
    - `get_run` — one session's metadata; `session_id=None` means the session
      this handle is bound to (the newest, for a reader).
    - `get_calls` — one session's calls in turn order.
    - `get_call_blocks` — one call's blocks in position order, rehydrated.

    Lifecycle:
    - `close` — release the connection; must never raise.

    Anything a specific backend needs beyond this (SQLite's schema-version gate,
    Postgres' statement timeout) stays private to that backend."""

    def record_call(self, seq: int, params: dict, usage: dict | None,
                    latency_ms: int | None, error: str | None,
                    call_blocks: list[CallBlock],
                    agent: str | None = None, step: str | None = None,
                    provider: str | None = None) -> str:
        """Persist one call and its ordered blocks in a single transaction,
        deduping blocks by content hash. Returns the new call id."""
        ...

    def note_model(self, model: str | None) -> None:
        """Append `model` to this session's `models` list the first time it is
        seen; ignore None/empty and repeats."""
        ...

    def list_sessions(self) -> list[Session]:
        """Every session in this store, oldest first."""
        ...

    def get_run(self, session_id: str | None = None) -> Run:
        """One session's run row (default: this handle's bound session)."""
        ...

    def get_calls(self, session_id: str | None = None) -> list[Call]:
        """One session's calls ordered by turn sequence."""
        ...

    def get_call_blocks(self, call_id: str) -> list[CallBlock]:
        """One call's blocks in position order."""
        ...

    def close(self) -> None:
        """Release the underlying connection. Best-effort; never raises."""
        ...


@runtime_checkable
class StoreBackend(Protocol):
    """A connection-less description of WHERE traces live, and the factory for
    `Store` handles onto it. This is what `ctxdiff.configure(store=...)` takes
    and what `CTXDIFF_STORE` resolves to.

    Constructing a backend must not connect, create tables, or touch the
    network — all of that happens in `open_session()`/`open_reader()`, so a
    misconfigured or unreachable database surfaces inside `Tracer.wrap()`'s
    fail-open guard (degrade capture, warn once) rather than exploding at
    import time in the user's module."""

    def open_session(self, project: str, provider: str, model: str = "",
                     started_at: str = "") -> Store:
        """Create the schema if missing, start a NEW session in this store, and
        return a `Store` bound to it. The write entry point `trace.init()` uses
        on the first `wrap()`."""
        ...

    def open_reader(self) -> Store:
        """Open a read handle bound to the NEWEST session in this store — the
        entry point the CLI/viewer use when no explicit session is named."""
        ...
