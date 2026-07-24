"""The shared machinery behind the Postgres and MySQL backends: one `Store`
implementation over any DB-API 2.0 connection, plus the backend base class that
connects lazily and creates the schema if it is missing.

Everything that is genuinely dialect-specific — the DDL's column TYPES, the
block-dedup upsert syntax, how a connection is opened and bounded — lives in a
small `Dialect` object supplied by `postgres.py`/`mysql.py`. Everything else
(the logical model, the statement text, the transaction shape, the fail-open
retry) is written ONCE here, so the two adapters cannot drift apart in
semantics: the conformance suite runs the same assertions against both.

Why the SQL here can be shared verbatim: psycopg 3 and PyMySQL both use the
`%s` (format/pyformat) paramstyle, so a single parameterized statement text
works for both. That is not an accident — it is one of the reasons PyMySQL was
chosen over mysqlclient/mysql-connector.

Naming: every table is prefixed `ctxdiff_` because, unlike a private `.ctrace`
file, these tables live in a database the user shares with their own
application — the prefix keeps ctxdiff's four tables obviously ctxdiff's and
avoids colliding with a `run`/`call`/`block` table that is already there. Three
columns are also renamed from their SQLite spellings (`usage` -> `usage_json`,
`position` -> `pos`, `text` -> `body`) because `usage` is a RESERVED word in
MySQL and the other two are keyword-adjacent; picking non-reserved names beats
quoting identifiers in every statement.

Session ordering: SQLite orders sessions by physical `rowid` — their INSERT
order. The network stores reproduce that with an explicit `insert_order`
column, spelled `BIGSERIAL` on Postgres and `BIGINT AUTO_INCREMENT` on MySQL
(where it is legal as a plain KEY column alongside the `id` primary key). It
would have been cheaper to order by `(started_at, id)`, and that is what this
module used to do — but it is wrong twice over, and both ways matter for the
case these backends exist to serve, several containers writing into one
database:

- clocks disagree. A container a few seconds behind writes a session that
  sorts BEFORE one written minutes earlier, so `open_reader()` binds — and
  `ctxdiff diff` analyzes — somebody else's run;
- timestamps tie. Two sessions starting in the same microsecond fell back to
  comparing random uuid4 hex: a coin flip that could reorder them between one
  read and the next, where SQLite was deterministic.

Insert order has neither problem: it is assigned by the database, is total, and
means the same thing on all three backends. `started_at` is still stored and
still required to be non-empty — it is what the UI DISPLAYS — it is just no
longer what anything sorts by."""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone

from ctxdiff import __version__
from ctxdiff.models import Block, CallBlock
from ctxdiff.store.base import Call, EmptyStoreError, Run, Session
from ctxdiff.store.schema import SCHEMA_VERSION

# Bounded write retry (mirrors the SQLite store's philosophy). A network DB can
# transiently refuse a write — a deadlock between two agents' writer threads, a
# serialization failure under a strict isolation level — where an immediate
# retry succeeds. Bounded on both axes so a genuinely broken DB gives up fast
# and lets the caller's fail-open guard drop the call, rather than blocking the
# writer thread (and therefore `close()`) indefinitely.
_WRITE_MAX_ATTEMPTS = 3
_WRITE_BACKOFF_START = 0.05
_WRITE_BACKOFF_MAX = 0.2

# Bounded retry for the auto-create DDL when two backends race to create the
# same table (see `SQLBackend._ensure_schema`). Three attempts because the race
# is between processes starting together and one of them wins each round; the
# delay is a fixed, tiny pause — just long enough for the winner's catalog
# insert to commit, and short enough that a cold start is not noticeably slower.
_SCHEMA_MAX_ATTEMPTS = 3
_SCHEMA_RETRY_DELAY = 0.1

# Default seconds to wait for a TCP connect/handshake before giving up. Small on
# purpose: capture must degrade quickly, never stall an agent's first LLM call
# waiting on a database that is not there.
DEFAULT_CONNECT_TIMEOUT = 5

# Default seconds any single statement may run before the server aborts it. The
# writer thread is shared by the whole run and is joined by `close()`, so an
# unbounded query would turn a slow DB into a hung shutdown.
DEFAULT_STATEMENT_TIMEOUT = 10

RUN_TABLE = "ctxdiff_run"
CALL_TABLE = "ctxdiff_call"
BLOCK_TABLE = "ctxdiff_block"
CALL_BLOCK_TABLE = "ctxdiff_call_block"

# --- statements shared by every dialect (both use the `%s` paramstyle) --------

_INSERT_RUN = (
    f"INSERT INTO {RUN_TABLE} "
    "(id, project, started_at, provider, models, ctxdiff_version, schema_version) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s)")

_SELECT_NEWEST_RUN = (
    f"SELECT id, schema_version FROM {RUN_TABLE} "
    "ORDER BY insert_order DESC LIMIT 1")

_SELECT_MAX_SCHEMA = f"SELECT MAX(schema_version) FROM {RUN_TABLE}"

_SELECT_SESSIONS = (
    f"SELECT id, project, started_at, provider, models FROM {RUN_TABLE} "
    "ORDER BY insert_order")

_SELECT_TURN_COUNTS = f"SELECT run_id, COUNT(*) FROM {CALL_TABLE} GROUP BY run_id"

_SELECT_AGENTS = (
    f"SELECT run_id, agent FROM {CALL_TABLE} "
    "WHERE agent IS NOT NULL ORDER BY run_id, seq")

_SELECT_RUN = (
    "SELECT id, project, started_at, provider, models, ctxdiff_version "
    f"FROM {RUN_TABLE} WHERE id = %s")

_SELECT_CALLS = (
    "SELECT id, run_id, seq, params, usage_json, latency_ms, error, "
    f"agent, step, provider FROM {CALL_TABLE} WHERE run_id = %s ORDER BY seq")

_INSERT_CALL = (
    f"INSERT INTO {CALL_TABLE} "
    "(id, run_id, seq, params, usage_json, latency_ms, error, agent, step, provider) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")

_INSERT_CALL_BLOCK = (
    f"INSERT INTO {CALL_BLOCK_TABLE} "
    "(call_id, block_id, pos, label, label_source) VALUES (%s, %s, %s, %s, %s)")

_SELECT_CALL_BLOCKS = (
    "SELECT cb.pos, cb.label, cb.label_source, b.content_hash, b.role, "
    "       b.kind, b.body, b.token_count, b.token_method "
    f"FROM {CALL_BLOCK_TABLE} cb JOIN {BLOCK_TABLE} b "
    "  ON b.content_hash = cb.block_id "
    "WHERE cb.call_id = %s ORDER BY cb.pos")

_UPDATE_MODELS = f"UPDATE {RUN_TABLE} SET models = %s WHERE id = %s"


class SQLStore:
    """A `Store` handle bound to ONE session in a networked SQL database.

    Owns a single DB-API connection, exactly like `CTrace` owns a single sqlite3
    connection, and is used the same way: the tracer's ONE writer thread does
    every write through it (so the connection is never touched concurrently),
    and a separate read handle is opened by the CLI/viewer.

    Construct via `SQLBackend.open_session()`/`open_reader()`, never directly —
    those apply the schema, run the version gate, and bind the run id."""

    def __init__(self, conn, dialect, run_id: str, models: list[str] | None = None,
                 reconnect=None):
        """Wrap an already-connected, already-migrated connection for one
        session. How: stores the connection, the dialect (for the one piece of
        SQL that differs — the block dedup upsert), and the session id every
        write lands against, then seeds the in-memory mirror of the run's
        `models` list so `note_model` can gate on a set lookup instead of a
        SELECT per call — the same trick, and the same first-seen ordering, as
        the SQLite store.

        `reconnect` is a zero-arg callable returning a FRESH connection to the
        same database — in practice the owning backend's `_connect`. It is what
        makes the retry in `_with_write_retry` mean anything when the failure
        was the connection itself dying: without it, "retry" re-runs the write
        against the same dead socket. Optional so a handle can be built without
        one (a reader, a unit test), in which case a lost connection simply
        fails as it did before."""
        self._conn = conn
        self._dialect = dialect
        self._run_id = run_id
        self._models: list[str] = list(models or [])
        self._models_seen = set(self._models)
        self._reconnect = reconnect

    # --- internals -----------------------------------------------------------

    def _query(self, sql: str, params: tuple = ()) -> list[tuple]:
        """Run one read query and return all rows as plain tuples. A cursor is
        opened and closed per query (cheap on both drivers) so no cursor state
        leaks between reads, and results are materialized immediately — callers
        get ordinary Python tuples, never a driver-specific lazy row object."""
        cur = self._conn.cursor()
        try:
            cur.execute(sql, params)
            return [tuple(row) for row in cur.fetchall()]
        finally:
            cur.close()

    def _with_write_retry(self, fn):
        """Run a write `fn` (a zero-arg callable wrapping ONE transaction),
        retrying a bounded number of times when the failure was transient
        (deadlock/serialization) or the connection itself died. Why: unlike a
        local file, a networked DB legitimately refuses a write that would
        succeed moments later, and losing a call to that would be needless
        capture loss. Bounded on both axes — fixed attempt cap, capped backoff —
        so a genuinely broken DB re-raises fast into the recorder's fail-open
        guard instead of stalling the writer thread. Each attempt rolls back
        before re-running, so a retry can never double-write.

        The connection-loss case is handled by REOPENING first (see
        `_reopen`). This is the difference between a retry that works and one
        that only looks like it does: a killed backend, a recycled pooler
        connection or a database restarted mid-deploy leaves this handle's
        socket permanently dead, so re-running the same statement on it fails
        identically every time — capture for the whole process ends at the
        first blip, with one warning, even though the database is healthy again
        a second later."""
        delay = _WRITE_BACKOFF_START
        for attempt in range(_WRITE_MAX_ATTEMPTS):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 — classified below, then re-raised
                lost = self._connection_lost(exc)
                self._rollback()
                last_attempt = attempt == _WRITE_MAX_ATTEMPTS - 1
                if last_attempt or not (lost or self._dialect.is_transient(exc)):
                    raise
                if lost and not self._reopen():
                    raise    # nothing to retry ON; fail now rather than spin
                time.sleep(delay)
                delay = min(delay * 2, _WRITE_BACKOFF_MAX)

    def _connection_lost(self, exc: Exception) -> bool:
        """True when the failure means THIS CONNECTION is gone, rather than the
        statement having been refused.

        Two signals, because neither alone is enough. The dialect classifies the
        error itself (Postgres reports the kill as SQLSTATE 57P01
        `admin_shutdown` / the 08xxx class; PyMySQL as errno 2006/2013) — but
        only for the FIRST failure after the kill: every later attempt raises a
        driver-level "the connection is closed" carrying no SQLSTATE at all. So
        the connection's own liveness flag is consulted too (`closed` on
        psycopg, `open` on PyMySQL), which stays authoritative afterwards."""
        if self._dialect.is_connection_lost(exc):
            return True
        return _connection_is_dead(self._conn)

    def _reopen(self) -> bool:
        """Replace this handle's dead connection with a fresh one, returning
        whether it worked. How: closes the old connection (best-effort — it is
        already gone; this just releases the local socket) and swaps in whatever
        the backend's `_connect` returns, which re-applies the same
        timeouts/session settings the original was opened with. A handle built
        without a `reconnect` callable, or a reconnect that itself fails (the
        server is still down), returns False so the caller re-raises into the
        fail-open guard instead of retrying against nothing."""
        if self._reconnect is None:
            return False
        _close_quietly(self._conn)
        try:
            self._conn = self._reconnect()
        except Exception:  # noqa: BLE001 — still unreachable; fail open, try next call
            return False
        return True

    def _rollback(self) -> None:
        """Abandon the current transaction, swallowing any failure. Guarded
        because rollback is itself a network round-trip: on a dropped connection
        it raises, and that must not mask the ORIGINAL write error the caller is
        about to see (or fail a retry that would otherwise have worked)."""
        try:
            self._conn.rollback()
        except Exception:  # noqa: BLE001 — best-effort; the real error is the caller's
            pass

    # --- writing -------------------------------------------------------------

    def record_call(self, seq: int, params: dict, usage: dict | None,
                    latency_ms: int | None, error: str | None,
                    call_blocks: list[CallBlock],
                    agent: str | None = None, step: str | None = None,
                    provider: str | None = None) -> str:
        """Persist one call and its ordered blocks in a single transaction —
        the same contract, and the same on-the-wire content, as the SQLite
        store's `record_call`. Each block is upserted by content_hash (stored
        once, ignored if already present: content-addressed dedup, via the
        dialect's no-op-on-duplicate insert); each membership row records the
        block's position and label within THIS call. Returns the new call id.

        Params/usage are stored as JSON TEXT (not a native JSON column) so both
        dialects, and the SQLite store, hold byte-identical values and no driver
        type-mapping can reinterpret them on the way back out."""
        call_id = uuid.uuid4().hex

        def _txn():
            cur = self._conn.cursor()
            try:
                cur.execute(_INSERT_CALL, (
                    call_id, self._run_id, seq, json.dumps(params),
                    json.dumps(usage) if usage is not None else None,
                    latency_ms, error, agent, step, provider))
                for cb in call_blocks:
                    b = cb.block
                    cur.execute(self._dialect.insert_block_dedup(), (
                        b.content_hash, b.role, b.kind, b.text,
                        b.token_count, b.token_method))
                    cur.execute(_INSERT_CALL_BLOCK, (
                        call_id, b.content_hash, cb.position,
                        cb.label, cb.label_source))
                self._conn.commit()
            finally:
                cur.close()

        self._with_write_retry(_txn)
        # Best-effort model roll-up onto the run row, mirroring the SQLite store
        # exactly: the call is ALREADY COMMITTED, so a failure here must not
        # propagate (the caller would log "failed to record" for a call that was
        # saved). It is self-healing — `note_model` only marks a model seen once
        # its UPDATE commits, so the next call carrying it retries.
        try:
            self.note_model(params.get("model") or params.get("modelId"))
        except Exception:  # noqa: BLE001 — roll-up is best-effort; call is saved
            pass
        return call_id

    def note_model(self, model: str | None) -> None:
        """Append `model` to the run's `models` list the first time it is seen,
        preserving first-seen order and deduping repeats; ignores None/empty so
        a call with no model param never stores a blank entry. The common case
        (a model already known) costs one set lookup and no round-trip.

        COMMIT-THEN-MARK ordering, same as the SQLite store: the in-memory
        mirror is updated only AFTER the UPDATE commits, so a write that
        exhausts its retry budget leaves nothing marked and the next call
        carrying that model tries again."""
        if not model or model in self._models_seen:
            return
        nxt = [*self._models, model]

        def _txn():
            cur = self._conn.cursor()
            try:
                cur.execute(_UPDATE_MODELS, (json.dumps(nxt), self._run_id))
                self._conn.commit()
            finally:
                cur.close()
            self._models = nxt
            self._models_seen.add(model)

        self._with_write_retry(_txn)

    # --- reading -------------------------------------------------------------

    def list_sessions(self) -> list[Session]:
        """Every session in this database, OLDEST FIRST, each summarized with
        its models, the distinct agents seen on its calls (first-appearance
        order), and its turn count. Assembled from three cheap queries rather
        than one wide join — identical to the SQLite reader, so the two
        backends' `Session` lists are indistinguishable."""
        runs = self._query(_SELECT_SESSIONS)

        counts: dict[str, int] = {}
        for run_id, n in self._query(_SELECT_TURN_COUNTS):
            counts[run_id] = int(n)

        agents: dict[str, list[str]] = {}
        for run_id, agent in self._query(_SELECT_AGENTS):
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
        """One session's row as a `Run`, decoding the models JSON array.
        `session_id` selects a session in a multi-session database; it defaults
        to the handle's bound session (the newest, for a reader) so the common
        single-session read needs no argument."""
        rows = self._query(_SELECT_RUN, (session_id or self._run_id,))
        r = rows[0]
        return Run(id=r[0], project=r[1], started_at=r[2], provider=r[3],
                   models=json.loads(r[4]), ctxdiff_version=r[5])

    def get_calls(self, session_id: str | None = None) -> list[Call]:
        """All calls for ONE session, ordered by turn sequence. `session_id`
        defaults to the bound session, so sessions read in isolation from each
        other even though they share tables. The attribution columns
        (agent/step/provider) always exist here — unlike SQLite, which may be
        reading a physically-v1 file — so there is no column-presence branch."""
        rows = self._query(_SELECT_CALLS, (session_id or self._run_id,))
        return [
            Call(id=r[0], run_id=r[1], seq=int(r[2]), params=json.loads(r[3]),
                 usage=json.loads(r[4]) if r[4] is not None else None,
                 latency_ms=r[5], error=r[6],
                 agent=r[7], step=r[8], provider=r[9])
            for r in rows
        ]

    def get_call_blocks(self, call_id: str) -> list[CallBlock]:
        """Reconstruct one call's blocks in position order by joining the
        membership table to the deduped block table — rebuilding the same
        `CallBlock`/`Block` objects the analyzers consume from any backend."""
        rows = self._query(_SELECT_CALL_BLOCKS, (call_id,))
        result = []
        for r in rows:
            block = Block(content_hash=r[3], role=r[4], kind=r[5], text=r[6],
                          token_count=int(r[7]), token_method=r[8])
            result.append(CallBlock(block=block, position=int(r[0]),
                                    label=r[1], label_source=r[2]))
        return result

    def close(self) -> None:
        """Close the connection, best-effort and never raising — it runs on the
        writer thread's way out and inside the CLI's `finally`, where an
        exception would be worse than a leaked socket the OS reclaims anyway. A
        rollback first discards any transaction left open by a failed write, so
        a server-side lock is released immediately rather than at timeout."""
        self._rollback()
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001 — close is best-effort on the way out
            pass


class SQLBackend:
    """Base for the networked backends (`PostgresStore`, `MySQLStore`): inert
    until used, then connect-and-create-if-missing.

    A subclass supplies the dialect (DDL types + dedup syntax + transient-error
    classification) and `_connect()`. Everything about the lifecycle — creating
    the tables on first connect, gating on schema version, inserting the session
    row, binding a reader to the newest session — is shared, so the two adapters
    behave identically by construction."""

    dialect = None  # set by the subclass

    def _connect(self):
        """Open one bounded DB-API connection. Implemented per adapter (each
        driver takes its own connect arguments); must import its driver lazily
        and raise a clear install hint if the extra is missing."""
        raise NotImplementedError

    def _ensure_schema(self, conn) -> None:
        """Create ctxdiff's four tables if they are not already there — the
        "no manual migration step" promise. How: runs the dialect's
        `CREATE TABLE IF NOT EXISTS` statements one per `execute()` (neither
        driver runs multi-statement scripts by default) and commits, so running
        this on every connect — which is what makes auto-create work without a
        "have I set up yet?" flag — costs one cheap no-op round trip against an
        existing database.

        Retried on a CATALOG RACE, which is the one case `IF NOT EXISTS` does
        not cover. `IF NOT EXISTS` is a check, not a lock: two Postgres backends
        that pass it in the same instant both insert into `pg_type`/`pg_class`,
        and the loser dies with a duplicate-key error (SQLSTATE 23505) on a
        system index — or, at a slightly different interleaving, 42P07
        "relation already exists". This is not a rare corner: it is the SHAPE OF
        A COLD START. A service scaling up points every replica at the same
        empty database at the same moment, and un-retried, all but one of them
        record nothing for their entire run — the deployment where capture
        matters most is the one that has none. Retrying is trivially safe
        because the winner has now created the table, so the retry's
        `IF NOT EXISTS` statements are pure no-ops. MySQL serializes its DDL and
        never produces this, so there the retry never fires.

        Each attempt rolls back first: the failing statement leaves the
        transaction aborted on Postgres, where every subsequent statement would
        fail with 25P02 until it is."""
        for attempt in range(_SCHEMA_MAX_ATTEMPTS):
            try:
                cur = conn.cursor()
                try:
                    for statement in self.dialect.ddl():
                        cur.execute(statement)
                    conn.commit()
                    return
                finally:
                    cur.close()
            except Exception as exc:  # noqa: BLE001 — classified, then re-raised
                _rollback_quietly(conn)
                if (attempt == _SCHEMA_MAX_ATTEMPTS - 1
                        or not self.dialect.is_schema_race(exc)):
                    raise
                time.sleep(_SCHEMA_RETRY_DELAY)

    def open_session(self, project: str, provider: str, model: str = "",
                     started_at: str = "") -> SQLStore:
        """Connect, create the schema if missing, and INSERT one new session
        row — the write entry point `trace.init()` uses on its first `wrap()`.

        `models` starts empty when `model` is falsy: the run's model is really a
        per-CALL fact that `wrap()` does not know yet, so seeding a placeholder
        would store a permanent blank; `note_model()` backfills it from real
        call params. `started_at` is required in substance — an empty value is
        replaced with 'now' so a session always carries the timestamp the UI
        displays; it is deliberately NOT what sessions are ordered by (see the
        module docstring).

        A database already holding rows from a NEWER ctxdiff schema is refused
        rather than mixed, mirroring the SQLite store's gate. On ANY failure the
        connection is closed before the error escapes, so a rejected open never
        leaks a socket."""
        conn = self._connect()
        try:
            self._ensure_schema(conn)
            existing = self._max_schema_version(conn)
            if existing is not None and existing > SCHEMA_VERSION:
                raise ValueError(
                    f"ctxdiff: store schema version {existing} is newer than "
                    f"supported {SCHEMA_VERSION} — upgrade ctxdiff to write here")
            run_id = uuid.uuid4().hex
            models = [model] if model else []
            cur = conn.cursor()
            try:
                cur.execute(_INSERT_RUN, (
                    run_id, project, started_at or _utc_now(), provider,
                    json.dumps(models), __version__, SCHEMA_VERSION))
                conn.commit()
            finally:
                cur.close()
        except Exception:
            _close_quietly(conn)
            raise
        # `_connect` (not a bound-and-forget lambda over `conn`) is handed over
        # as the reconnect callable, so a connection killed mid-run is replaced
        # with a fresh one carrying the same timeouts — see `SQLStore._reopen`.
        return SQLStore(conn, self.dialect, run_id, models,
                        reconnect=self._connect)

    def open_reader(self) -> SQLStore:
        """Connect and bind a handle to the NEWEST session in this database —
        what the CLI/viewer read when the user names no session. Newest means
        the greatest `insert_order`, i.e. the session written LAST, which is
        the SQLite reader's `ORDER BY rowid DESC` and is immune to a writer
        whose clock is behind. A database with no sessions raises a clear
        `EmptyStoreError` instead of returning an empty handle that would fail
        obscurely later. The schema is ensured first so reading a database
        ctxdiff has never written reports "no sessions", not "no such table"."""
        conn = self._connect()
        try:
            self._ensure_schema(conn)
            cur = conn.cursor()
            try:
                cur.execute(_SELECT_NEWEST_RUN)
                row = cur.fetchone()
            finally:
                cur.close()
            if row is None:
                raise EmptyStoreError(
                    "ctxdiff: no sessions recorded in this store")
            run_id, version = row[0], row[1]
            if version > SCHEMA_VERSION:
                raise ValueError(
                    f"ctxdiff: store schema version {version} is newer than "
                    f"supported {SCHEMA_VERSION} — upgrade ctxdiff to read here")
        except Exception:
            _close_quietly(conn)
            raise
        return SQLStore(conn, self.dialect, run_id)

    @staticmethod
    def _max_schema_version(conn):
        """Highest `schema_version` any session in this database was written
        with, or None when there are none yet. Its own tiny helper so
        `open_session` reads as a sequence of intentions rather than cursor
        bookkeeping."""
        cur = conn.cursor()
        try:
            cur.execute(_SELECT_MAX_SCHEMA)
            row = cur.fetchone()
        finally:
            cur.close()
        return None if row is None else row[0]


def _utc_now() -> str:
    """Canonical UTC-with-offset timestamp (`...+00:00`), the same spelling the
    tracer writes, used as the fallback when a caller supplies no
    `started_at` — see `open_session` for why it may not be blank."""
    return datetime.now(timezone.utc).isoformat()


def _close_quietly(conn) -> None:
    """Close a connection, swallowing failures — used on error paths where the
    real exception is already on its way up and must not be masked."""
    try:
        conn.close()
    except Exception:  # noqa: BLE001
        pass


def _rollback_quietly(conn) -> None:
    """Roll back, swallowing failures — used between retry attempts, where the
    connection may itself be the thing that broke and the caller is about to
    re-raise the REAL error either way."""
    try:
        conn.rollback()
    except Exception:  # noqa: BLE001
        pass


def _connection_is_dead(conn) -> bool:
    """Whether a DB-API connection is known to be unusable, asked in the two
    spellings the supported drivers answer in: psycopg exposes `closed` (truthy
    once the socket is gone), PyMySQL exposes `open` (False once it is). Both
    are consulted by attribute so neither driver has to be imported here, and a
    connection object that reports neither is assumed ALIVE — the caller only
    uses this to decide whether to reopen, and guessing "dead" for an unknown
    driver would throw away a perfectly good connection."""
    closed = getattr(conn, "closed", None)
    if closed:
        return True
    if getattr(conn, "open", True) is False:
        return True
    return False
