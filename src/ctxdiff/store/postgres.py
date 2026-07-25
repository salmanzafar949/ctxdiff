"""The PostgreSQL backend — `ctxdiff[postgres]`, on psycopg 3.

    import ctxdiff
    from ctxdiff import PostgresStore
    ctxdiff.configure(store=PostgresStore(dsn="postgresql://user@host/agents"))

...or `CTXDIFF_STORE=postgresql://user@host/agents` with no code change at all.
Tables are created on first connect if they are not already there; there is no
migration step.

The driver is imported LAZILY, inside `_connect()`, for two reasons: ctxdiff's
core must stay dependency-light (tiktoken only), and a user who has configured
Postgres but not installed the extra must get a clear one-line install hint at
connect time — inside the tracer's fail-open guard — rather than an
`ImportError` crash at `import ctxdiff`."""
from __future__ import annotations

from ctxdiff.store.sql import (
    BLOCK_TABLE,
    CALL_BLOCK_TABLE,
    CALL_TABLE,
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_STATEMENT_TIMEOUT,
    RUN_TABLE,
    SQLBackend,
)

# SQLSTATE classes worth one bounded retry rather than dropping the call:
# serialization failure and deadlock (two agents' writer threads racing on the
# same block rows), lock-not-available, and the whole 08xxx connection-exception
# class (a connection recycled by a pooler/proxy mid-write).
_TRANSIENT_SQLSTATES = {"40001", "40P01", "55P03"}
_TRANSIENT_SQLSTATE_PREFIXES = ("08",)

# SQLSTATEs meaning THIS CONNECTION is gone and the retry must reopen before it
# can mean anything: the 08xxx connection-exception class, plus the 57Pxx
# operator-intervention codes a kill produces — `pg_terminate_backend` reports
# 57P01 (admin_shutdown), a crash-restart 57P02, an idle-session timeout 57P03.
_CONNECTION_LOST_SQLSTATES = {"57P01", "57P02", "57P03"}
_CONNECTION_LOST_PREFIXES = ("08",)

# SQLSTATEs raised when two backends create the same table at the same instant.
# `CREATE TABLE IF NOT EXISTS` checks the catalog without locking it, so both
# proceed and the loser fails on a system index — as a duplicate key (23505,
# e.g. `pg_type_typname_nsp_index` for the table's row type, or
# `pg_class_relname_nsp_index` for a serial column's sequence) or, at a slightly
# different interleaving, as an outright duplicate_table/duplicate_object.
_SCHEMA_RACE_SQLSTATES = {"23505", "42P07", "42710"}

# TCP keepalive probing for a connection sitting IDLE between an agent's calls
# (seconds / seconds / probes). Chosen so a silently-dead peer is detected in
# well under a minute — 10s of idle, then 3 probes 5s apart — rather than the
# OS default of two hours. Deliberately more aggressive than a general-purpose
# pool would be: this connection is a debugging tool's, and noticing it is dead
# is worth far more than the handful of packets.
_KEEPALIVE_IDLE = 10
_KEEPALIVE_INTERVAL = 5
_KEEPALIVE_COUNT = 3


class _PostgresDialect:
    """Everything about Postgres that the shared SQL layer cannot assume.

    Types: Postgres' `TEXT` is unbounded AND indexable, so the DDL is a nearly
    literal transcription of the `.ctrace` schema — no VARCHAR lengths to
    invent, which is the single biggest structural difference from MySQL.
    `latency_ms` is the one deliberate widening: `INTEGER` milliseconds are a
    32-bit value that overflows after ~24 days, and an overflow REJECTS the
    whole call rather than storing an odd number.

    Ordering: `insert_order` is a `BIGSERIAL` — the portable stand-in for
    SQLite's implicit `rowid`, giving sessions a monotonic write order that is
    independent of the `started_at` clock. It is the ordering key for
    `list_sessions()` and for binding a reader to the newest session, so
    containers whose clocks disagree (the reason to share one database in the
    first place) still agree on which run is the latest.

    Dedup: `ON CONFLICT (content_hash) DO NOTHING` — the standard-flavoured
    upsert. It names the conflicting column explicitly, so it can only ever
    no-op on a duplicate CONTENT HASH; any other constraint violation still
    raises, which is what content-addressed dedup should mean."""

    name = "postgres"

    @staticmethod
    def ddl() -> tuple[str, ...]:
        """The `CREATE TABLE IF NOT EXISTS` statements, in dependency order
        (parents before the tables whose foreign keys reference them), one per
        statement because psycopg does not run multi-statement scripts by
        default. Re-running them against an existing database is a no-op, which
        is what makes auto-create safe on every connect."""
        return (
            f"""CREATE TABLE IF NOT EXISTS {RUN_TABLE} (
  id              TEXT PRIMARY KEY,
  insert_order    BIGSERIAL NOT NULL UNIQUE,
  project         TEXT NOT NULL,
  started_at      TEXT NOT NULL,
  provider        TEXT NOT NULL,
  models          TEXT NOT NULL,
  ctxdiff_version TEXT NOT NULL,
  schema_version  INTEGER NOT NULL
)""",
            f"""CREATE TABLE IF NOT EXISTS {CALL_TABLE} (
  id          TEXT PRIMARY KEY,
  run_id      TEXT NOT NULL REFERENCES {RUN_TABLE}(id),
  seq         INTEGER NOT NULL,
  params      TEXT NOT NULL,
  usage_json  TEXT,
  latency_ms  BIGINT,
  error       TEXT,
  agent       TEXT,
  step        TEXT,
  provider    TEXT,
  UNIQUE (run_id, seq)
)""",
            f"""CREATE TABLE IF NOT EXISTS {BLOCK_TABLE} (
  content_hash TEXT PRIMARY KEY,
  role         TEXT NOT NULL,
  kind         TEXT NOT NULL,
  body         TEXT NOT NULL,
  token_count  INTEGER NOT NULL,
  token_method TEXT NOT NULL
)""",
            f"""CREATE TABLE IF NOT EXISTS {CALL_BLOCK_TABLE} (
  call_id      TEXT NOT NULL REFERENCES {CALL_TABLE}(id),
  block_id     TEXT NOT NULL REFERENCES {BLOCK_TABLE}(content_hash),
  pos          INTEGER NOT NULL,
  label        TEXT NOT NULL,
  label_source TEXT NOT NULL,
  PRIMARY KEY (call_id, pos)
)""",
        )

    @staticmethod
    def insert_block_dedup() -> str:
        """Insert a block, or do nothing if this content hash is already
        stored — the content-addressed dedup primitive, in Postgres syntax."""
        return (f"INSERT INTO {BLOCK_TABLE} "
                "(content_hash, role, kind, body, token_count, token_method) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (content_hash) DO NOTHING")

    @staticmethod
    def is_transient(exc: Exception) -> bool:
        """True when a failed write is worth ONE bounded retry. How: reads the
        SQLSTATE psycopg attaches to every server error (`exc.sqlstate`) and
        matches it against the serialization/deadlock/lock codes plus the whole
        `08xxx` connection-exception class. Anything without a SQLSTATE (a bug
        in ctxdiff, a bad value, a closed cursor) is NOT transient and re-raises
        immediately — retrying it would only delay the fail-open drop."""
        state = getattr(exc, "sqlstate", None)
        if not state:
            return False
        return (state in _TRANSIENT_SQLSTATES
                or state.startswith(_TRANSIENT_SQLSTATE_PREFIXES))

    @staticmethod
    def is_connection_lost(exc: Exception) -> bool:
        """True when the error means the CONNECTION died, not just the
        statement — the case a retry can only survive by reopening first. How:
        matches the 08xxx connection-exception class and the 57Pxx
        operator-intervention codes (`pg_terminate_backend`, a crash restart, an
        idle-session timeout). psycopg reports the kill this way exactly ONCE;
        every later statement raises a plain `OperationalError` with no SQLSTATE
        at all, which is why the caller ALSO checks the connection's own
        `closed` flag rather than relying on this alone."""
        state = getattr(exc, "sqlstate", None)
        if not state:
            return False
        return (state in _CONNECTION_LOST_SQLSTATES
                or state.startswith(_CONNECTION_LOST_PREFIXES))

    @staticmethod
    def is_schema_race(exc: Exception) -> bool:
        """True when the auto-create DDL lost a race to another process
        creating the SAME table (see `_SCHEMA_RACE_SQLSTATES`), which the caller
        retries because the winner has by then made the retry a no-op.
        Deliberately narrow: an ordinary 23505 on a ctxdiff table is a real
        constraint violation and must not be retried, which is why this is
        consulted ONLY from `_ensure_schema` and never from the write path."""
        state = getattr(exc, "sqlstate", None)
        return bool(state) and state in _SCHEMA_RACE_SQLSTATES


class PostgresStore(SQLBackend):
    """Points ctxdiff at a PostgreSQL database. Connection-less until used: this
    constructor validates nothing and opens nothing, so
    `configure(store=PostgresStore(dsn=...))` at module import is free and a
    dead database surfaces inside the tracer's fail-open guard."""

    dialect = _PostgresDialect

    def __init__(self, dsn: str, connect_timeout: int = DEFAULT_CONNECT_TIMEOUT,
                 statement_timeout: int = DEFAULT_STATEMENT_TIMEOUT):
        """Record where to connect and the two bounds that keep a slow/dead
        database from ever hanging the host: `connect_timeout` seconds for the
        TCP connect/handshake, and `statement_timeout` seconds for any single
        statement once connected. Both default small (see `store/sql.py`)
        because capture must degrade quickly rather than stall an agent."""
        self.dsn = dsn
        self.connect_timeout = connect_timeout
        self.statement_timeout = statement_timeout

    def _connect(self):
        """Open one bounded psycopg 3 connection.

        How: imports the driver lazily (turning a missing extra into an
        actionable install hint instead of an import-time crash), connects with
        `connect_timeout` so an unreachable host fails in seconds rather than
        the OS default, leaves autocommit OFF so `record_call`'s multi-statement
        write is one real transaction, and then bounds every subsequent
        statement server-side with `statement_timeout` — without which a lock
        held by another session could block the writer thread, and therefore
        `tracer.close()`, indefinitely.

        The socket options are the third bound, and the one the other two cannot
        substitute for. `connect_timeout` covers connect and authentication and
        NOTHING after them; `statement_timeout` is enforced by the server, so it
        can never fire when the problem is that packets are not arriving. A box
        that completes the handshake and is then wedged or partitioned away
        leaves the client blocked in `recv` forever — the exact hang this
        addresses. `tcp_user_timeout` bounds how long the kernel will retransmit
        unacknowledged data before giving up (so a statement in flight fails
        instead of hanging), and the keepalives do the same for a connection
        sitting IDLE between calls."""
        try:
            import psycopg  # noqa: PLC0415 — lazy: keeps core dependency-light
        except ImportError as exc:  # pragma: no cover - exercised via stub in tests
            raise ImportError(
                "ctxdiff: PostgresStore needs psycopg — install the extra with "
                "`pip install 'ctxdiff[postgres]'`") from exc
        conn = self._connect_socket_bounded(psycopg)
        try:
            cur = conn.cursor()
            try:
                # Milliseconds, per Postgres. Applied to the session so it
                # covers every statement this connection will ever run.
                cur.execute(f"SET statement_timeout = {int(self.statement_timeout) * 1000}")
                conn.commit()
            finally:
                cur.close()
        except Exception:
            # A connection we could not configure is a connection we do not
            # want: close it here so the failure cannot leak a socket.
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
            raise
        return conn

    def _connect_socket_bounded(self, psycopg):
        """`psycopg.connect` with the TCP-level bounds applied, falling back
        once if the platform rejects `tcp_user_timeout`.

        Why the fallback: `TCP_USER_TIMEOUT` is a Linux socket option, and libpq
        FAILS THE WHOLE CONNECTION ("setsockopt(TCP_USER_TIMEOUT) not
        supported") when asked for a non-zero value on a platform without it —
        macOS, older BSDs. A knob whose job is to harden the connection must
        never be able to prevent one, so an error that names the option retries
        without it (keeping the keepalives, which are portable). Everything else
        propagates untouched: a wrong password must not be retried into a
        second failed login attempt."""
        keepalive = {
            "keepalives": 1,
            "keepalives_idle": _KEEPALIVE_IDLE,
            "keepalives_interval": _KEEPALIVE_INTERVAL,
            "keepalives_count": _KEEPALIVE_COUNT,
        }
        try:
            return psycopg.connect(
                self.dsn, connect_timeout=self.connect_timeout, autocommit=False,
                # Milliseconds, matching the statement bound: data that cannot
                # be delivered within the time the statement itself is allowed
                # is already lost.
                tcp_user_timeout=int(self.statement_timeout) * 1000,
                **keepalive)
        except Exception as exc:  # noqa: BLE001 — re-raised unless it IS the knob
            if "tcp_user_timeout" not in str(exc).lower():
                raise
            return psycopg.connect(
                self.dsn, connect_timeout=self.connect_timeout,
                autocommit=False, **keepalive)
