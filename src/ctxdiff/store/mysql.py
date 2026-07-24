"""The MySQL/MariaDB backend — `ctxdiff[mysql]`, on PyMySQL.

    import ctxdiff
    from ctxdiff import MySQLStore
    ctxdiff.configure(store=MySQLStore(dsn="mysql://user:pw@host:3306/agents"))

...or `CTXDIFF_STORE=mysql://user:pw@host/agents`. Tables are created on first
connect if missing; there is no migration step.

WHY PyMySQL over mysqlclient: mysqlclient is a C extension that needs a
compiler and the MySQL client headers at install time — a hostile dependency
for a debugging tool whose whole pitch is a one-line add to an existing agent.
PyMySQL is pure Python, installs everywhere with no build step, is DB-API 2.0
compliant, and uses the same `%s` paramstyle as psycopg 3 — which is what lets
`store/sql.py` share one set of parameterized statements between the two
backends instead of maintaining two. mysqlclient's speed advantage is
irrelevant here: ctxdiff writes one small transaction per LLM call, off the
host's call path, on a background thread.

The driver is imported lazily inside `_connect()` so core stays
dependency-light and a missing extra becomes a clear install hint at connect
time (inside the tracer's fail-open guard) rather than an import crash."""
from __future__ import annotations

from urllib.parse import unquote, urlparse

from ctxdiff.store.sql import (
    BLOCK_TABLE,
    CALL_BLOCK_TABLE,
    CALL_TABLE,
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_STATEMENT_TIMEOUT,
    RUN_TABLE,
    SQLBackend,
)

# MySQL server error codes worth one bounded retry: deadlock found, lock wait
# timeout (both routine under concurrent writers touching the same block rows),
# and the two "connection went away" codes a proxy/pooler can produce mid-write.
_TRANSIENT_ERRNOS = {1213, 1205, 2006, 2013}

# Errnos meaning THIS CONNECTION is gone, so the retry must reopen before it can
# succeed: 2006 (server has gone away — a restart, a `KILL`, or wait_timeout),
# 2013 (lost connection during the query), 2003/2002 (cannot connect at all,
# which is what a reconnect attempt itself reports).
_CONNECTION_LOST_ERRNOS = {2002, 2003, 2006, 2013}

# Errnos raised when two clients create the same object at once. MySQL
# serializes DDL, so `CREATE TABLE IF NOT EXISTS` really is safe here and this
# never fires in practice — it exists so the two adapters classify the same
# situations, rather than leaving MySQL a silent exception to the rule.
_SCHEMA_RACE_ERRNOS = {1050, 1061, 1022, 1062}

# Identifier lengths. Ids and content hashes are hex strings; VARCHAR(64) covers
# both with room to spare and — unlike TEXT — can be a PRIMARY KEY / FOREIGN KEY
# in MySQL without a prefix length, which is the crux of the DDL difference from
# Postgres. Free-text columns use LONGTEXT (a prompt can be megabytes).
_ID_LEN = 64
# `started_at` is an ISO-8601 timestamp (~32 chars); the slack is for callers
# that pass their own spelling.
_TIMESTAMP_LEN = 64

# Byte-exact comparison, pinned per column on every id/hash/timestamp.
#
# MySQL's default collation for utf8mb4 (`utf8mb4_0900_ai_ci` on 8.0+) is
# accent- and CASE-INSENSITIVE, which quietly changes what this schema MEANS
# rather than merely how it sorts:
#   - `content_hash` is the primary key of a CONTENT-ADDRESSED table. Under a
#     *_ci collation two hashes differing only in case are one key, so the
#     second block is deduped away and its call reads back the FIRST block's
#     text — corruption with no error anywhere.
#   - `started_at` compares by collation rather than by bytes, so an ISO-8601
#     string stops behaving like the byte-ordered value every other backend
#     treats it as.
# These columns are hex ids and ISO timestamps — ASCII by construction — so
# `ascii_bin` is both the narrowest and the most exact choice, and it makes
# MySQL match SQLite's BINARY and Postgres' C-locale TEXT. It is spelled on
# each COLUMN, not as a table default, so it cannot be changed out from under
# the schema by an `ALTER TABLE ... CONVERT TO CHARACTER SET`.
_BIN = "CHARACTER SET ascii COLLATE ascii_bin"

# Seconds added to the statement timeout to get PyMySQL's socket read/write
# timeout. The socket bound must sit ABOVE the server-side statement bound, not
# below it: with `read_timeout = connect_timeout` (5s) under a 10s
# `max_execution_time`, the server knob was unreachable and every statement that
# legitimately took 5-10s was killed client-side as a "lost connection" — which
# then took the whole connection (and, before the reconnect in store/sql.py, all
# further capture) down with it. The margin covers the round trip in which the
# server reports its OWN timeout as a proper error; the socket timeout stays the
# backstop for a server that stops answering entirely.
_SOCKET_TIMEOUT_MARGIN = 2


def _errno(exc: Exception) -> int | None:
    """The MySQL server error code carried by a driver exception, or None.
    How: PyMySQL raises with `args == (errno, message)`, so the code is the
    first argument — when it is an int. Anything else (a bug in ctxdiff, a bad
    value, a plain `RuntimeError`) has no errno and is classified by every
    caller as "not retryable", which is what makes an unrecognized failure
    re-raise immediately into the fail-open guard instead of being retried."""
    args = getattr(exc, "args", ())
    if not args or not isinstance(args[0], int):
        return None
    return args[0]


class _MySQLDialect:
    """Everything about MySQL that the shared SQL layer cannot assume.

    Types: MySQL cannot index a `TEXT` column without a prefix length, so every
    column that is a primary key, foreign key or unique key must be a bounded
    `VARCHAR(64)` — and ONLY those. Everything else is `LONGTEXT`, because a
    bounded VARCHAR on a free-text column is not a size limit, it is a data
    loss: under STRICT mode (the default since 5.7) a 400-character provider
    error, tag label or agent name raises error 1406 and, since the call and
    its blocks are written in ONE transaction, drops the WHOLE turn — on MySQL
    alone, where SQLite and Postgres store it fine. `latency_ms` is `BIGINT`
    for the same reason: `INT` milliseconds overflow after ~24 days.
    Tables are explicitly `ENGINE=InnoDB` because the foreign keys and
    transactional writes below are meaningless on MyISAM, and
    `CHARSET=utf8mb4` so prompts containing emoji/CJK round-trip byte-exactly —
    with `ascii_bin` pinned on the id/hash/timestamp columns (see `_BIN`).

    Ordering: `insert_order` is an `AUTO_INCREMENT` column — MySQL's portable
    equivalent of SQLite's implicit `rowid`, and the ONLY total order that
    survives clocks disagreeing between the containers sharing this database.
    MySQL requires an AUTO_INCREMENT column to be the first column of a key,
    which a plain `UNIQUE KEY` satisfies without disturbing the `id` primary
    key.

    Dedup: `ON DUPLICATE KEY UPDATE content_hash = content_hash` — a no-op
    assignment that makes a duplicate hash silently succeed. Deliberately NOT
    `INSERT IGNORE`, which downgrades EVERY error (truncation, bad values,
    FK violations) to a warning and would let a malformed block be silently
    half-stored; this form no-ops on exactly the duplicate-key case and lets
    every other error raise."""

    name = "mysql"

    @staticmethod
    def ddl() -> tuple[str, ...]:
        """The `CREATE TABLE IF NOT EXISTS` statements, in dependency order
        (parents first, since the child tables declare foreign keys), one per
        statement because PyMySQL does not run multi-statement scripts by
        default. Re-running them is a no-op, which is what makes auto-create
        safe on every connect."""
        return (
            f"""CREATE TABLE IF NOT EXISTS {RUN_TABLE} (
  id              VARCHAR({_ID_LEN}) {_BIN} NOT NULL,
  insert_order    BIGINT NOT NULL AUTO_INCREMENT,
  project         LONGTEXT NOT NULL,
  started_at      VARCHAR({_TIMESTAMP_LEN}) {_BIN} NOT NULL,
  provider        LONGTEXT NOT NULL,
  models          LONGTEXT NOT NULL,
  ctxdiff_version LONGTEXT NOT NULL,
  schema_version  INT NOT NULL,
  UNIQUE KEY ctxdiff_run_insert_order (insert_order),
  PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
            f"""CREATE TABLE IF NOT EXISTS {CALL_TABLE} (
  id          VARCHAR({_ID_LEN}) {_BIN} NOT NULL,
  run_id      VARCHAR({_ID_LEN}) {_BIN} NOT NULL,
  seq         INT NOT NULL,
  params      LONGTEXT NOT NULL,
  usage_json  LONGTEXT,
  latency_ms  BIGINT,
  error       LONGTEXT,
  agent       LONGTEXT,
  step        LONGTEXT,
  provider    LONGTEXT,
  PRIMARY KEY (id),
  UNIQUE KEY ctxdiff_call_run_seq (run_id, seq),
  FOREIGN KEY (run_id) REFERENCES {RUN_TABLE}(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
            f"""CREATE TABLE IF NOT EXISTS {BLOCK_TABLE} (
  content_hash VARCHAR({_ID_LEN}) {_BIN} NOT NULL,
  role         LONGTEXT NOT NULL,
  kind         LONGTEXT NOT NULL,
  body         LONGTEXT NOT NULL,
  token_count  INT NOT NULL,
  token_method LONGTEXT NOT NULL,
  PRIMARY KEY (content_hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
            f"""CREATE TABLE IF NOT EXISTS {CALL_BLOCK_TABLE} (
  call_id      VARCHAR({_ID_LEN}) {_BIN} NOT NULL,
  block_id     VARCHAR({_ID_LEN}) {_BIN} NOT NULL,
  pos          INT NOT NULL,
  label        LONGTEXT NOT NULL,
  label_source LONGTEXT NOT NULL,
  PRIMARY KEY (call_id, pos),
  FOREIGN KEY (call_id) REFERENCES {CALL_TABLE}(id),
  FOREIGN KEY (block_id) REFERENCES {BLOCK_TABLE}(content_hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
        )

    @staticmethod
    def insert_block_dedup() -> str:
        """Insert a block, or no-op if this content hash is already stored —
        the content-addressed dedup primitive, in MySQL syntax. See the class
        docstring for why this and not `INSERT IGNORE`."""
        return (f"INSERT INTO {BLOCK_TABLE} "
                "(content_hash, role, kind, body, token_count, token_method) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE content_hash = content_hash")

    @staticmethod
    def is_transient(exc: Exception) -> bool:
        """True when a failed write is worth ONE bounded retry. How: PyMySQL
        raises with `args == (errno, message)`, so the first arg is matched
        against the deadlock/lock-wait/connection-lost codes. Anything without a
        recognizable integer errno (a bug in ctxdiff, a bad value) is NOT
        transient and re-raises immediately into the fail-open guard."""
        return _errno(exc) in _TRANSIENT_ERRNOS

    @staticmethod
    def is_connection_lost(exc: Exception) -> bool:
        """True when the error means the CONNECTION died rather than the
        statement being refused — a `KILL`, a server restart, a pooler recycle,
        `wait_timeout` — so the retry reopens before trying again instead of
        re-running against a dead socket."""
        return _errno(exc) in _CONNECTION_LOST_ERRNOS

    @staticmethod
    def is_schema_race(exc: Exception) -> bool:
        """True when the auto-create DDL collided with another client creating
        the same object. See `_SCHEMA_RACE_ERRNOS`: MySQL serializes DDL, so
        this is a parity hook rather than a live code path."""
        return _errno(exc) in _SCHEMA_RACE_ERRNOS


class MySQLStore(SQLBackend):
    """Points ctxdiff at a MySQL/MariaDB database. Connection-less until used,
    so `configure(store=MySQLStore(dsn=...))` at import time is free and an
    unreachable server surfaces inside the tracer's fail-open guard.

    Takes a URL DSN (`mysql://user:pw@host:3306/dbname`), individual connection
    fields, or both — explicit fields win, so a DSN from the environment can be
    overridden in code without string surgery."""

    dialect = _MySQLDialect

    def __init__(self, dsn: str | None = None, host: str | None = None,
                 port: int | None = None, user: str | None = None,
                 password: str | None = None, database: str | None = None,
                 connect_timeout: int = DEFAULT_CONNECT_TIMEOUT,
                 statement_timeout: int = DEFAULT_STATEMENT_TIMEOUT):
        """Record how to connect, plus the two bounds that stop a slow/dead
        server from hanging the host: `connect_timeout` seconds for the TCP
        connect/handshake, and `statement_timeout` seconds for any single
        statement — applied BOTH server-side (`max_execution_time`) and, plus a
        small margin, as PyMySQL's socket read/write timeout, so a server that
        accepts a connection and then goes silent cannot wedge the writer
        thread. See `_SOCKET_TIMEOUT_MARGIN` for why the socket bound is derived
        from the statement timeout rather than the connect one. Nothing is
        connected or validated here."""
        self.dsn = dsn
        self._overrides = {"host": host, "port": port, "user": user,
                           "password": password, "database": database}
        self.connect_timeout = connect_timeout
        self.statement_timeout = statement_timeout

    def connect_kwargs(self) -> dict:
        """Resolve the DSN and the explicit fields into PyMySQL connect
        arguments. How: parses the URL (percent-decoding user/password, which
        routinely contain `@` or `/`), applies sane defaults (localhost:3306),
        then overlays any explicitly-passed field so code beats the DSN. Public
        because it is exactly what a test wants to assert on without opening a
        connection."""
        parsed = urlparse(self.dsn) if self.dsn else None
        kwargs = {
            "host": (parsed.hostname if parsed else None) or "localhost",
            "port": (parsed.port if parsed else None) or 3306,
            "user": unquote(parsed.username) if parsed and parsed.username else None,
            "password": unquote(parsed.password) if parsed and parsed.password else None,
            "database": (parsed.path.lstrip("/") if parsed and parsed.path else None) or None,
        }
        for key, value in self._overrides.items():
            if value is not None:
                kwargs[key] = value
        return kwargs

    def _connect(self):
        """Open one bounded PyMySQL connection.

        How: imports the driver lazily (missing extra -> actionable install
        hint, not an import crash), connects with connect/read/write timeouts so
        neither an unreachable host nor a silent one can block the writer
        thread, keeps autocommit OFF so `record_call` is one real transaction,
        forces utf8mb4 so prompt text round-trips byte-exactly, and then bounds
        statements server-side. The server-side bound is applied best-effort:
        `max_execution_time` exists on MySQL 5.7.8+ but not on MariaDB (which
        spells it `max_statement_time`, in seconds), so an unknown-variable
        error there is swallowed rather than failing an otherwise good
        connection — the client-side socket timeouts are the real guarantee."""
        try:
            import pymysql  # noqa: PLC0415 — lazy: keeps core dependency-light
        except ImportError as exc:  # pragma: no cover - exercised via stub in tests
            raise ImportError(
                "ctxdiff: MySQLStore needs PyMySQL — install the extra with "
                "`pip install 'ctxdiff[mysql]'`") from exc
        socket_timeout = int(self.statement_timeout) + _SOCKET_TIMEOUT_MARGIN
        conn = pymysql.connect(
            **self.connect_kwargs(),
            connect_timeout=self.connect_timeout,
            read_timeout=socket_timeout,
            write_timeout=socket_timeout,
            autocommit=False,
            charset="utf8mb4",
        )
        try:
            cur = conn.cursor()
            try:
                cur.execute("SET SESSION max_execution_time = "
                            f"{int(self.statement_timeout) * 1000}")
            except Exception:  # noqa: BLE001 — MariaDB/old MySQL: no such variable
                conn.rollback()
            finally:
                cur.close()
        except Exception:  # noqa: BLE001 — never fail a good connection over a knob
            pass
        return conn
