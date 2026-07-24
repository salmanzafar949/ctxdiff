"""Read/write access to a `.ctrace` file — a plain SQLite database holding one
run, its calls, and content-addressed blocks. Writers dedup blocks by hash;
readers reconstruct ordered CallBlocks. No analysis lives here."""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass

from ctxdiff import __version__
from ctxdiff.models import Block, CallBlock
from ctxdiff.store.schema import DDL, SCHEMA_VERSION

# started_at is passed in by the caller (the tracer) rather than read from the
# clock here, so the store stays a pure I/O layer and is trivially testable.


@dataclass(frozen=True)
class Run:
    """One agent execution, as stored in the `run` table."""
    id: str
    project: str
    started_at: str
    provider: str
    models: list[str]
    ctxdiff_version: str


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


class CTrace:
    """A handle to one `.ctrace` file. Construct via `create()` (new run) or
    `open()` (existing run); never call the initializer directly."""

    def __init__(self, conn: sqlite3.Connection, run_id: str):
        """Wrap an already-open, already-initialized connection for one run.
        How: stores the live connection and the id of the single run it
        operates against, then snapshots the `call` table's column set (via
        PRAGMA table_info) ONCE so reads/writes can adapt to a v1 file (which
        lacks the v2 agent/step/provider columns) without a PRAGMA per query.
        Trusts `create()`/`open()` to have done all setup/validation already,
        per the class docstring's "never call directly" contract."""
        self._conn = conn
        self._run_id = run_id
        self._call_cols = {row[1] for row in conn.execute("PRAGMA table_info(call)")}
        self._has_v2_cols = {"agent", "step", "provider"} <= self._call_cols
        # In-memory mirror of the run's `models` column, seeded once here so
        # `note_model` can gate on a cheap set-membership check instead of a
        # SELECT on every call. Order matches first-seen order in the DB.
        row = conn.execute(
            "SELECT models FROM run WHERE id = ?", (run_id,)).fetchone()
        self._models: list[str] = json.loads(row[0]) if row else []
        self._models_seen = set(self._models)

    # --- construction ------------------------------------------------------

    @classmethod
    def create(cls, path: str, project: str, provider: str, model: str,
               started_at: str = "") -> "CTrace":
        """Create a fresh `.ctrace` at `path`, apply the schema, and write the
        single run row. Foreign keys are enabled so referential integrity holds.
        `started_at` defaults to empty (the tracer supplies a real timestamp).

        `models` starts as `[model]` when a real model id is passed, but as an
        EMPTY list when `model` is falsy (None/""): the run's model is really a
        per-CALL fact (`wrap()` doesn't know it yet at run-creation time — see
        trace.py), so seeding a placeholder here would just store a bogus
        blank string forever if no call ever arrives. `note_model()` is what
        actually populates `models` from real call params as they come in.

        `check_same_thread=False` + WAL: a live capture run drives all its
        writes through ONE dedicated writer thread (see `trace.py`'s `_Writer`),
        NOT the thread that opened the connection here. sqlite3 otherwise
        forbids using a connection from a thread other than its creator, so the
        check is disabled — safe because ctxdiff still serializes every write
        onto that single writer thread (never concurrent access), it just isn't
        the creating thread. WAL (write-ahead logging) lets those writes commit
        without blocking readers and is enabled once at creation (it's a
        persistent property of the file). Both are set BEFORE the DDL so the
        whole run is created under them."""
        conn = sqlite3.connect(path, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(DDL)
            run_id = uuid.uuid4().hex
            models = [model] if model else []
            conn.execute(
                "INSERT INTO run VALUES (?,?,?,?,?,?,?)",
                (run_id, project, started_at, provider,
                 json.dumps(models), __version__, SCHEMA_VERSION),
            )
            conn.commit()
        except Exception:
            # DDL/insert failed after the connection was opened; close it so
            # we don't leak a file handle/lock on the way out.
            conn.close()
            raise
        return cls(conn, run_id)

    @classmethod
    def open(cls, path: str) -> "CTrace":
        """Open an existing `.ctrace` read/write, accepting ANY schema version
        this build understands (v1 and v2). A v1 file is read as-is and NEVER
        migrated on open — a debugger must not rewrite the evidence it inspects;
        its missing agent/step/provider columns simply surface as None. Only a
        file whose version is NEWER than supported is rejected, with a clear
        ValueError telling the reader to upgrade rather than letting a
        mismatched read fail obscurely later."""
        conn = sqlite3.connect(path)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            row = conn.execute(
                "SELECT id, schema_version FROM run LIMIT 1").fetchone()
            if row is None:
                raise ValueError(f"{path}: not a ctrace file (no run row)")
            run_id, version = row
            if version > SCHEMA_VERSION:
                raise ValueError(
                    f"{path}: schema version {version} is newer than supported "
                    f"{SCHEMA_VERSION} — upgrade ctxdiff to read this file")
            if version < 1:
                raise ValueError(
                    f"{path}: schema version {version} is not a recognized ctrace version")
        except Exception:
            # Reject-and-abort paths (bad file, schema mismatch) and any read
            # error alike must not leak the connection on the way out.
            conn.close()
            raise
        return cls(conn, run_id)

    # --- writing -----------------------------------------------------------

    def record_call(self, seq: int, params: dict, usage: dict | None,
                    latency_ms: int | None, error: str | None,
                    call_blocks: list[CallBlock],
                    agent: str | None = None, step: str | None = None,
                    provider: str | None = None) -> str:
        """Persist one call and its ordered blocks in a single transaction.
        Each block is upserted by content_hash (stored once, ignored if already
        present — the dedup mechanism); each membership is written to call_block
        with its position and label. `agent`/`step`/`provider` are the v2
        attribution fields (nullable). Returns the new call id.

        create() always writes the v2 schema, so the v2 columns are present
        whenever record_call runs (the tracer only ever writes to a trace it
        created); the write still adapts to `_has_v2_cols` defensively so a
        hypothetical write against a v1 handle degrades to the v1 shape rather
        than raising on a missing column."""
        call_id = uuid.uuid4().hex
        with self._conn:  # transaction: all-or-nothing
            if self._has_v2_cols:
                self._conn.execute(
                    "INSERT INTO call VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (call_id, self._run_id, seq, json.dumps(params),
                     json.dumps(usage) if usage is not None else None,
                     latency_ms, error, agent, step, provider),
                )
            else:
                self._conn.execute(
                    "INSERT INTO call VALUES (?,?,?,?,?,?,?)",
                    (call_id, self._run_id, seq, json.dumps(params),
                     json.dumps(usage) if usage is not None else None,
                     latency_ms, error),
                )
            for cb in call_blocks:
                b = cb.block
                # INSERT OR IGNORE: first writer of a hash wins; repeats are
                # no-ops, which is exactly content-addressed dedup.
                self._conn.execute(
                    "INSERT OR IGNORE INTO block VALUES (?,?,?,?,?,?)",
                    (b.content_hash, b.role, b.kind, b.text,
                     b.token_count, b.token_method),
                )
                self._conn.execute(
                    "INSERT INTO call_block VALUES (?,?,?,?,?)",
                    (call_id, b.content_hash, cb.position, cb.label, cb.label_source),
                )
        # Roll this call's model up onto the run — the fix for the
        # long-standing `run.models == ['']` gap: the run's model is only
        # ever really known per-call, so every write here is a chance to
        # backfill it. `model`/`modelId` covers openai/anthropic/gemini
        # (which all name the param "model") and bedrock (which names it
        # "modelId" instead, per its Converse API shape) without needing a
        # per-provider special case here.
        self.note_model(params.get("model") or params.get("modelId"))
        return call_id

    def note_model(self, model: str | None) -> None:
        """Append `model` to the run's `models` list the first time it's seen,
        preserving first-seen order and deduping repeats; ignores None/empty
        so a call with no model param never pollutes the list with a blank
        entry. Cheap: the common case (a model already known) costs only an
        in-memory set lookup — the `models` JSON is re-serialized and the run
        row UPDATEd only on an actual new model, not on every call."""
        if not model or model in self._models_seen:
            return
        self._models_seen.add(model)
        self._models.append(model)
        with self._conn:  # single-statement transaction
            self._conn.execute(
                "UPDATE run SET models = ? WHERE id = ?",
                (json.dumps(self._models), self._run_id),
            )

    # --- reading -----------------------------------------------------------

    def get_run(self) -> Run:
        """Return the run row as a Run, decoding the models JSON array."""
        r = self._conn.execute(
            "SELECT id, project, started_at, provider, models, ctxdiff_version "
            "FROM run WHERE id = ?", (self._run_id,)).fetchone()
        return Run(id=r[0], project=r[1], started_at=r[2], provider=r[3],
                   models=json.loads(r[4]), ctxdiff_version=r[5])

    def get_calls(self) -> list[Call]:
        """Return all calls for this run ordered by turn sequence. Selects the
        v2 agent/step/provider columns only when they exist (a v1 file lacks
        them); for a v1 file those three fields surface as None on every Call,
        so downstream code needs no special-casing."""
        base = "id, run_id, seq, params, usage, latency_ms, error"
        if self._has_v2_cols:
            rows = self._conn.execute(
                f"SELECT {base}, agent, step, provider "
                "FROM call WHERE run_id = ? ORDER BY seq", (self._run_id,)).fetchall()
            return [
                Call(id=r[0], run_id=r[1], seq=r[2], params=json.loads(r[3]),
                     usage=json.loads(r[4]) if r[4] is not None else None,
                     latency_ms=r[5], error=r[6],
                     agent=r[7], step=r[8], provider=r[9])
                for r in rows
            ]
        rows = self._conn.execute(
            f"SELECT {base} FROM call WHERE run_id = ? ORDER BY seq",
            (self._run_id,)).fetchall()
        return [
            Call(id=r[0], run_id=r[1], seq=r[2], params=json.loads(r[3]),
                 usage=json.loads(r[4]) if r[4] is not None else None,
                 latency_ms=r[5], error=r[6])
            for r in rows
        ]

    def get_call_blocks(self, call_id: str) -> list[CallBlock]:
        """Reconstruct one call's blocks in position order by joining call_block
        to block. Rebuilds full CallBlock/Block objects for downstream analysis."""
        rows = self._conn.execute(
            "SELECT cb.position, cb.label, cb.label_source, "
            "       b.content_hash, b.role, b.kind, b.text, b.token_count, b.token_method "
            "FROM call_block cb JOIN block b ON b.content_hash = cb.block_id "
            "WHERE cb.call_id = ? ORDER BY cb.position", (call_id,)).fetchall()
        result = []
        for r in rows:
            block = Block(content_hash=r[3], role=r[4], kind=r[5], text=r[6],
                          token_count=r[7], token_method=r[8])
            result.append(CallBlock(block=block, position=r[0],
                                    label=r[1], label_source=r[2]))
        return result

    def close(self) -> None:
        """Checkpoint the WAL back into the main database file, then close the
        connection. The explicit `wal_checkpoint(TRUNCATE)` guarantees a fresh
        reader that opens the file on ANOTHER connection right after close sees
        every committed write immediately (and shrinks the -wal sidecar),
        rather than relying on SQLite's automatic on-close checkpoint. The
        checkpoint is best-effort — a failure here (e.g. WAL not in use for a
        v1 file opened read-only) must never stop the connection from closing,
        so it is swallowed."""
        try:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:  # noqa: BLE001 — checkpoint is best-effort; always close
            pass
        self._conn.close()
