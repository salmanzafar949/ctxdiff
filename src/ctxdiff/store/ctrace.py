"""Read/write access to a `.ctrace` file — a plain SQLite database that is a
project DB holding one OR MANY sessions (each session is one `run` row), their
calls, and content-addressed blocks. Writers dedup blocks by hash; readers
reconstruct ordered CallBlocks and can list/select sessions. No analysis lives
here.

Project-scoped model: one file per PROJECT, appended to over time. Each
`trace.init()` opens the project DB (creating it if absent) and inserts a NEW
`run` row — one session. A single-session file (one `run` row) is simply the
degenerate case and reads identically to before.

`CTrace` is the DEFAULT implementation of the backend-independent `Store`
protocol (`ctxdiff.store.base`) — local-first, zero-config, and what every
existing user keeps getting unless they explicitly configure another backend.
It satisfies the protocol structurally (no base class, no behavior change); the
row dataclasses it returns now live in `base.py` and are re-exported here so
`from ctxdiff.store.ctrace import Call, CTrace, Run, Session` still works."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid

from ctxdiff import __version__
from ctxdiff.models import Block, CallBlock
from ctxdiff.store.base import (  # noqa: F401 — re-exported for import compat
    Call,
    EmptyStoreError,
    Run,
    Session,
    parse_started_at,
)
from ctxdiff.store.schema import DDL, SCHEMA_VERSION

# started_at is passed in by the caller (the tracer) rather than read from the
# clock here, so the store stays a pure I/O layer and is trivially testable.

# Multi-writer knobs. Multiple Tracers in one process AND multiple processes may
# write the same project DB concurrently. WAL (set at create/open time) lets
# readers run while one writer commits; `busy_timeout` makes SQLite BLOCK an
# incoming writer for up to this long waiting for the lock instead of failing
# instantly; the bounded retry loop below is the last line of defence on top of
# that, so a brief contention degrades fail-open (drop + warn upstream) rather
# than raising into the host or hanging.
_BUSY_TIMEOUT_MS = 5000
# Minimum seconds between the passive WAL checkpoints record_call runs (see
# there). Real agents call an LLM at most a few times a second, so a 1s
# throttle costs nothing in practice — but synthetic writers (test fixtures,
# bulk imports) hammer thousands of back-to-back calls, and checkpointing
# every one of them measurably slows exactly those (a CI golden-fixture hook
# blew its timeout the day this shipped unthrottled). The bare file may lag
# the WAL by up to this interval; close() still guarantees a full TRUNCATE.
_CHECKPOINT_INTERVAL_S = 1.0
_WRITE_MAX_ATTEMPTS = 6      # total tries before giving up (fail-open upstream)
_WRITE_BACKOFF_START = 0.05  # seconds; doubles each retry, capped by _WRITE_BACKOFF_MAX
_WRITE_BACKOFF_MAX = 0.5


class CTrace:
    """A handle to one session in one `.ctrace` file — ctxdiff's DEFAULT `Store`
    implementation (see `ctxdiff.store.base.Store`), which it satisfies
    structurally rather than by inheritance so nothing about the local-first
    path changed when the protocol was introduced. Construct via `create()` (new
    file, one run) or `open_or_create_session()` (append a session) or `open()`
    (read); never call the initializer directly."""

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
        # Monotonic timestamp of the last passive checkpoint; zero means
        # "never", so the FIRST record_call always checkpoints (which is what
        # keeps a freshly-created file immediately copyable).
        self._last_checkpoint = 0.0

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
            # busy_timeout FIRST — before WAL/foreign_keys/DDL, which all take a
            # write lock and would otherwise fail INSTANTLY with "database is
            # locked" under concurrent creation on the same file. Set ahead of
            # everything else, it makes those very statements block-and-wait for
            # the lock instead (see open_or_create_session for the full story).
            conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            run_id = uuid.uuid4().hex
            models = [model] if model else []

            def _setup():
                # The whole setup (WAL enable + DDL + run INSERT) can still
                # surface a transient "database is locked" under heavy
                # multi-writer contention even with busy_timeout, so it runs
                # through the same bounded locked-retry as record_call. Made
                # safe to re-run: a leading rollback drops any partial state
                # from a prior locked attempt, WAL is idempotent, the DDL is
                # IF NOT EXISTS, and the INSERT is wrapped in its own
                # transaction so a failed attempt leaves no half-written row.
                conn.rollback()
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA foreign_keys = ON")
                conn.executescript(DDL)
                with conn:
                    conn.execute(
                        "INSERT INTO run VALUES (?,?,?,?,?,?,?)",
                        (run_id, project, started_at, provider,
                         json.dumps(models), __version__, SCHEMA_VERSION),
                    )
            cls._with_write_retry(_setup)
        except Exception:
            # DDL/insert failed after the connection was opened; close it so
            # we don't leak a file handle/lock on the way out.
            conn.close()
            raise
        return cls(conn, run_id)

    @classmethod
    def open_or_create_session(cls, path: str, project: str, provider: str,
                               model: str, started_at: str = "") -> "CTrace":
        """Start a NEW session in the project DB at `path`, creating the file if
        it doesn't exist yet — the project-scoped write entry point the tracer
        uses on its first `wrap()`. What: opens (or creates) the connection with
        the full write configuration, ensures the schema is present, then inserts
        ONE fresh `run` row (a new session with its own uuid id and `started_at`)
        and returns a CTrace bound to it. Every call this Tracer records lands
        against that session's run id, so many sessions coexist in one file.

        How append works with zero special-casing: the DDL is all
        `CREATE TABLE IF NOT EXISTS`, so running it against an already-populated
        project DB is a harmless no-op — the single code path both creates a
        brand-new file and appends to an existing one. The only guard is a schema
        gate: if the file already holds rows from a NEWER schema than this build
        understands, we refuse to append (mirroring `open()`'s rejection) rather
        than write a mixed-version file.

        Connection config mirrors `create()` — `check_same_thread=False` (the
        writer thread, not this creating thread, owns the connection), WAL, and
        `busy_timeout` — plus the same config is what makes concurrent writers
        (other Tracers here, other processes elsewhere) block-and-wait on the
        lock instead of failing instantly; the writer's `record_call` adds a
        bounded retry on top (see `_with_write_retry`)."""
        conn = sqlite3.connect(path, check_same_thread=False)
        try:
            # busy_timeout FIRST — before ANY lock-taking statement. This is the
            # crux of the concurrent-creation fix: `PRAGMA journal_mode = WAL`
            # and the `IF NOT EXISTS` DDL both acquire a write lock, and when a
            # dozen processes race to create the SAME project file at once (all
            # from a barrier) whichever ones lose the lock would otherwise raise
            # "database is locked" INSTANTLY — straight out into the host via
            # `Tracer.wrap`. Setting busy_timeout ahead of them makes those
            # statements block-and-wait for the lock instead of failing.
            conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            run_id = uuid.uuid4().hex
            models = [model] if model else []

            def _setup():
                # The ENTIRE setup — WAL enable, DDL, schema-version gate, and
                # the session INSERT — runs under the same bounded locked-retry
                # as record_call, because busy_timeout alone doesn't cover every
                # transient (a WAL write-write conflict can still surface
                # "database is locked" under heavy contention). Made safe to
                # re-run: a leading rollback clears any partial state from a
                # prior locked attempt, WAL is idempotent, the DDL is
                # IF NOT EXISTS, the version gate is a pure read, and the INSERT
                # is wrapped in its own transaction so a failed attempt leaves
                # no half-written run row. A non-lock error (e.g. the newer-
                # schema ValueError below) propagates immediately, unretried.
                conn.rollback()
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA foreign_keys = ON")
                conn.executescript(DDL)  # IF NOT EXISTS — no-op on an existing DB
                # ...and because it IS a no-op, an existing PHYSICALLY-v1 `call`
                # table survives the DDL unchanged. Widen it before writing, or
                # this session's agent/step/provider would be silently dropped
                # under a run row stamped schema_version=2.
                cls._upgrade_call_table(conn)
                # Refuse to append into a file written by a newer ctxdiff.
                row = conn.execute(
                    "SELECT MAX(schema_version) FROM run").fetchone()
                existing_version = row[0] if row else None
                if existing_version is not None and existing_version > SCHEMA_VERSION:
                    raise ValueError(
                        f"{path}: schema version {existing_version} is newer than "
                        f"supported {SCHEMA_VERSION} — upgrade ctxdiff to write this file")
                with conn:
                    conn.execute(
                        "INSERT INTO run VALUES (?,?,?,?,?,?,?)",
                        (run_id, project, started_at, provider,
                         json.dumps(models), __version__, SCHEMA_VERSION),
                    )
            cls._with_write_retry(_setup)
        except Exception:
            conn.close()
            raise
        return cls(conn, run_id)

    @staticmethod
    def _upgrade_call_table(conn: sqlite3.Connection) -> None:
        """Bring an existing `call` table up to the v2 layout by ADDing whichever
        of `agent`/`step`/`provider` are missing — on the WRITE path only.

        Why it's needed: `CREATE TABLE IF NOT EXISTS` no-ops against a table that
        already exists, so appending a session into a file some older ctxdiff
        wrote physically as v1 left `_has_v2_cols` False — every new call went in
        with the 7-column v1 shape and lost its attribution, even though the new
        run row was stamped `schema_version = 2`. Silent data loss, and newly
        reachable now that the default path is a STABLE `./<project>.ctrace` that
        can land on a pre-existing user file.

        Upgrading in place beats downgrading the version stamp because the
        columns are nullable: pre-existing v1 calls simply read back None,
        nothing is rewritten or reinterpreted, and the file stays readable by
        both SDKs. `ADD COLUMN` of a nullable TEXT is an O(1) header-only change
        in SQLite — existing rows are not rewritten.

        Read-only `open()` deliberately does NOT call this: a debugger must not
        rewrite the evidence it inspects, so only a writer about to append its
        own v2 rows upgrades the layout."""
        cols = {row[1] for row in conn.execute("PRAGMA table_info(call)")}
        for col in ("agent", "step", "provider"):
            if col not in cols:
                conn.execute(f"ALTER TABLE call ADD COLUMN {col} TEXT")

    @staticmethod
    def _with_write_retry(fn):
        """Run a write `fn` (a zero-arg callable wrapping one transaction),
        retrying a bounded number of times when SQLite reports the database is
        locked. Why on top of `busy_timeout`: busy_timeout already blocks the
        writer while another connection holds the lock, but under heavy
        multi-writer contention SQLite can still surface `OperationalError:
        database is locked` (e.g. a WAL write-write conflict); a few short
        exponential-backoff retries clear those transients. Bounded on both axes
        — a fixed attempt cap and a capped per-sleep — so a genuinely stuck lock
        gives up (re-raising for the caller's fail-open guard to drop+warn)
        rather than hanging. Non-lock OperationalErrors and all other exceptions
        propagate immediately, unretried. Because each `fn` is a self-contained
        transaction that rolls back on failure, a retry re-runs it cleanly with
        no partial-write risk."""
        delay = _WRITE_BACKOFF_START
        for attempt in range(_WRITE_MAX_ATTEMPTS):
            try:
                return fn()
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == _WRITE_MAX_ATTEMPTS - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, _WRITE_BACKOFF_MAX)

    @classmethod
    def open(cls, path: str) -> "CTrace":
        """Open an existing project `.ctrace` read/write, accepting ANY schema
        version this build understands (v1 and v2). A v1 file is read as-is and
        NEVER migrated on open — a debugger must not rewrite the evidence it
        inspects; its missing agent/step/provider columns simply surface as
        None. Only a file whose version is NEWER than supported is rejected, with
        a clear ValueError telling the reader to upgrade rather than letting a
        mismatched read fail obscurely later.

        A project DB may hold MANY sessions; the returned handle is bound to the
        NEWEST session (last-inserted `run` row) so a session-less read analyzes
        the run the user JUST made, not the first one ever recorded into this
        accumulating project file. A single-session/v1 file is unaffected —
        newest == the only row. Multi-session consumers select a specific run via
        `list_sessions()` plus the `session_id=` reader arguments rather than
        relying on this default binding. `busy_timeout` is set FIRST (before any
        other statement) so reads don't fail instantly against a file a live
        writer is concurrently committing to."""
        conn = sqlite3.connect(path)
        try:
            conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            conn.execute("PRAGMA foreign_keys = ON")
            row = conn.execute(
                "SELECT id, schema_version FROM run ORDER BY rowid DESC LIMIT 1").fetchone()
            if row is None:
                raise EmptyStoreError(f"{path}: not a ctrace file (no run row)")
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

        def _txn():
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
        # One self-contained transaction, retried on a transient lock under
        # concurrent multi-writer access (see `_with_write_retry`); a rolled-back
        # attempt re-runs cleanly, so no partial write can result.
        self._with_write_retry(_txn)
        # Roll this call's model up onto the run — the fix for the
        # long-standing `run.models == ['']` gap: the run's model is only
        # ever really known per-call, so every write here is a chance to
        # backfill it. `model`/`modelId` covers openai/anthropic/gemini
        # (which all name the param "model") and bedrock (which names it
        # "modelId" instead, per its Converse API shape) without needing a
        # per-provider special case here.
        #
        # Guarded: the call above is ALREADY COMMITTED, so a failure of this
        # best-effort roll-up must not propagate out of record_call — that would
        # make the caller log "failed to record call" for a call that WAS
        # persisted. It is self-healing anyway: note_model only marks a model
        # seen once its UPDATE commits, so the next call carrying the same model
        # retries it.
        try:
            self.note_model(params.get("model") or params.get("modelId"))
        except Exception:  # noqa: BLE001 — roll-up is best-effort; call is saved
            pass
        # PASSIVE checkpoint, throttled to one per _CHECKPOINT_INTERVAL_S:
        # under WAL, committed pages live in the `-wal` sidecar until a
        # checkpoint copies them into the main file — and a long-lived process
        # (a server that never calls close()) can leave the `.ctrace` a
        # near-empty shell indefinitely, so anyone copying/sharing the bare
        # file mid-run ships an empty trace (dogfood finding 2026-07-27).
        # PASSIVE transfers whatever it can WITHOUT ever blocking concurrent
        # readers or writers (unlike close()'s TRUNCATE), which is what makes
        # it safe on the hot path; the time throttle is what makes it safe for
        # synthetic writers that hammer thousands of back-to-back calls (test
        # fixtures, imports) where per-call fsyncs measurably drag. Best-effort
        # like the roll-up above: the call is already committed, so a
        # checkpoint failure must never surface as a failed record.
        now = time.monotonic()
        if now - self._last_checkpoint >= _CHECKPOINT_INTERVAL_S:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                self._last_checkpoint = now
            except Exception:  # noqa: BLE001 — durability is done; checkpoint is a bonus
                pass
        return call_id

    def note_model(self, model: str | None) -> None:
        """Append `model` to the run's `models` list the first time it's seen,
        preserving first-seen order and deduping repeats; ignores None/empty
        so a call with no model param never pollutes the list with a blank
        entry. Cheap: the common case (a model already known) costs only an
        in-memory set lookup — the `models` JSON is re-serialized and the run
        row UPDATEd only on an actual new model, not on every call.

        COMMIT-THEN-MARK ordering matters: the in-memory `_models_seen`/`_models`
        mirror is updated INSIDE the retried transaction, only AFTER the UPDATE
        succeeds. Marking the model seen up front (the obvious ordering) meant a
        write that failed its retry budget left the model permanently "already
        known", so no later call ever retried it and `run.models` stayed [] for
        the life of the run. Deferring the mutation makes the roll-up
        self-healing: a failed attempt changes nothing, and the next call
        carrying that model tries again."""
        if not model or model in self._models_seen:
            return
        # Candidate list built without touching the mirror, so a failed (or
        # retried) attempt can never leave a half-applied state behind.
        nxt = [*self._models, model]

        def _txn():
            with self._conn:  # single-statement transaction
                self._conn.execute(
                    "UPDATE run SET models = ? WHERE id = ?",
                    (json.dumps(nxt), self._run_id),
                )
            self._models = nxt
            self._models_seen.add(model)
        self._with_write_retry(_txn)  # retry on transient lock (multi-writer)

    # --- reading -----------------------------------------------------------

    def list_sessions(self) -> list[Session]:
        """List every session (every `run` row) in this project DB, oldest first,
        each as a `Session` summary: id, project, started_at, provider, models,
        the set of agents seen on its calls (first-appearance order), and its
        turn (call) count. This is the reader half of the project-scoped model —
        a single-session file simply returns a one-element list.

        How: three cheap queries assembled in Python rather than one big join, so
        turn counts and agent sets stay correct and easy to reason about. The
        per-run turn count comes from a `GROUP BY run_id` COUNT; the agent sets
        come from scanning `(run_id, agent)` in seq order and deduping per run to
        preserve first-appearance order — but only when the v2 `agent` column
        exists (a v1 file has no agents, so every session reports `[]`)."""
        runs = self._conn.execute(
            "SELECT id, project, started_at, provider, models "
            "FROM run ORDER BY rowid").fetchall()

        counts: dict[str, int] = {}
        for run_id, n in self._conn.execute(
                "SELECT run_id, COUNT(*) FROM call GROUP BY run_id").fetchall():
            counts[run_id] = n

        agents: dict[str, list[str]] = {}
        if self._has_v2_cols:
            for run_id, agent in self._conn.execute(
                    "SELECT run_id, agent FROM call "
                    "WHERE agent IS NOT NULL ORDER BY run_id, seq").fetchall():
                seen = agents.setdefault(run_id, [])
                if agent not in seen:
                    seen.append(agent)

        return [
            Session(id=r[0], project=r[1], started_at=r[2], provider=r[3],
                    models=json.loads(r[4]), agents=agents.get(r[0], []),
                    turn_count=counts.get(r[0], 0))
            for r in runs
        ]

    def get_run(self, session_id: str | None = None) -> Run:
        """Return one session's `run` row as a Run, decoding the models JSON
        array. `session_id` selects which session in a multi-session project DB;
        it defaults to the handle's bound session (the newest — see `open()`), so
        a single-session file needs no argument and reads exactly as before."""
        run_id = session_id or self._run_id
        r = self._conn.execute(
            "SELECT id, project, started_at, provider, models, ctxdiff_version "
            "FROM run WHERE id = ?", (run_id,)).fetchone()
        return Run(id=r[0], project=r[1], started_at=r[2], provider=r[3],
                   models=json.loads(r[4]), ctxdiff_version=r[5])

    def get_calls(self, session_id: str | None = None) -> list[Call]:
        """Return all calls for ONE session ordered by turn sequence.
        `session_id` selects the session (defaults to the bound one — see
        `get_run()`), so a project DB's sessions read in isolation while a
        single-session file is unchanged. Selects the v2 agent/step/provider
        columns only when they exist (a v1 file lacks them); for a v1 file those
        three fields surface as None on every Call, so downstream code needs no
        special-casing."""
        run_id = session_id or self._run_id
        base = "id, run_id, seq, params, usage, latency_ms, error"
        if self._has_v2_cols:
            rows = self._conn.execute(
                f"SELECT {base}, agent, step, provider "
                "FROM call WHERE run_id = ? ORDER BY seq", (run_id,)).fetchall()
            return [
                Call(id=r[0], run_id=r[1], seq=r[2], params=json.loads(r[3]),
                     usage=json.loads(r[4]) if r[4] is not None else None,
                     latency_ms=r[5], error=r[6],
                     agent=r[7], step=r[8], provider=r[9])
                for r in rows
            ]
        rows = self._conn.execute(
            f"SELECT {base} FROM call WHERE run_id = ? ORDER BY seq",
            (run_id,)).fetchall()
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
