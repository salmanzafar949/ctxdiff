"""A stub DB-API driver that lets the Postgres and MySQL adapters be exercised
END-TO-END with no server running.

Why not a mock: a mock that records calls would prove the adapters CALL a
driver, not that the SQL they generate is correct — the auto-create DDL, the
dedup upsert, the joins and orderings would all be unverified, and the adapters
would have no real coverage in CI. So this driver is REAL: it executes every
statement the adapter emits against a private SQLite database (a temp file, so
several "connections" to the same fake server see each other's committed data,
exactly like a real one), translating the handful of constructs SQLite spells
differently. Everything else — the table/column names, the parameter order, the
transaction shape, the reads — is executed verbatim, so a mistake in the
adapters' SQL fails the tests.

What is translated, and why each is honest:
- `%s` -> `?` — a paramstyle difference only; the parameters themselves and
  their ORDER are untouched, so a wrong parameter order still fails.
- `ENGINE=InnoDB DEFAULT CHARSET=utf8mb4` — a storage-engine clause with no
  SQLite equivalent; stripping it leaves the columns and keys intact.
- `UNIQUE KEY <name> (cols)` -> `UNIQUE (cols)` — MySQL's spelling of a named
  unique constraint; the constraint itself is preserved and still enforced.
- `ON DUPLICATE KEY UPDATE col = col` -> `ON CONFLICT DO NOTHING` — MySQL's
  no-op upsert; the SQLite form has identical semantics, so dedup is genuinely
  tested rather than assumed. (Postgres' `ON CONFLICT (col) DO NOTHING` is
  SQLite's native syntax and passes through UNCHANGED.)
- `SET statement_timeout` / `SET SESSION max_execution_time` — server knobs
  with no SQLite equivalent; recorded in the statement log (so tests can assert
  the adapter bounds its statements) and then skipped.
- the `insert_order` column (`BIGSERIAL` / `BIGINT AUTO_INCREMENT`, plus its
  key) is DROPPED, and every `ORDER BY insert_order` is rewritten to
  `ORDER BY rowid`. SQLite cannot auto-increment a second, non-primary-key
  column — but it does not need to: `rowid` IS the monotonic insert-order
  integer the adapters add that column to obtain, so the ordering the tests
  assert on is genuinely enforced here rather than accidentally satisfied by a
  column full of NULLs.
- `CHARACTER SET ascii COLLATE ascii_bin` — MySQL's spelling of the byte-exact
  comparison SQLite gives every TEXT column by default (BINARY); dropping the
  clause leaves SQLite's semantics IDENTICAL to what the clause asks for, so a
  case-sensitivity test still means something here.

`fail_connect=` makes `connect()` raise instead, which is how the unreachable-
database fail-open test gets a dead server without one."""
from __future__ import annotations

import re
import sqlite3
import sys
import types

# Statements that configure a real server and mean nothing to SQLite. They are
# still logged before being skipped, so a test can assert the adapter sets them.
_IGNORED_PREFIXES = ("SET STATEMENT_TIMEOUT", "SET SESSION MAX_EXECUTION_TIME")


def translate(sql: str) -> str | None:
    """Rewrite one adapter-generated statement into its SQLite equivalent, or
    return None for a server-knob statement that should be skipped entirely.
    Deliberately narrow: anything not listed in the module docstring passes
    through untouched, so the adapters' real SQL is what gets executed."""
    if sql.strip().upper().startswith(_IGNORED_PREFIXES):
        return None
    out = sql
    # The database-assigned insert-order column. SQLite already HAS one (the
    # implicit `rowid`) but cannot declare a second AUTOINCREMENT column, so the
    # declaration and its key are dropped and the ordering is expressed against
    # `rowid` — the same monotonic insert counter, so the ordering under test is
    # really enforced. Done BEFORE the generic `UNIQUE KEY` rewrite below, which
    # would otherwise turn the dropped key into a constraint on a missing column.
    out = re.sub(r"^\s*UNIQUE KEY\s+\w+\s*\(insert_order\),?[ \t]*\n", "", out,
                 flags=re.IGNORECASE | re.MULTILINE)
    out = re.sub(r"^\s*insert_order\s+(BIGSERIAL|BIGINT)[^\n]*\n", "", out,
                 flags=re.IGNORECASE | re.MULTILINE)
    out = re.sub(r"ORDER BY insert_order", "ORDER BY rowid", out,
                 flags=re.IGNORECASE)
    # MySQL's per-column byte-exact collation. SQLite compares TEXT byte-exactly
    # already, so removing the clause preserves the semantics it requests.
    out = re.sub(r"\s*CHARACTER SET ascii COLLATE ascii_bin", "", out,
                 flags=re.IGNORECASE)
    # MySQL's no-op upsert -> SQLite's. Matched with the column name so a
    # DIFFERENT (i.e. wrong) upsert wouldn't quietly translate.
    out = re.sub(r"ON DUPLICATE KEY UPDATE\s+(\w+)\s*=\s*\1",
                 "ON CONFLICT DO NOTHING", out, flags=re.IGNORECASE)
    # Named unique constraint -> anonymous one (same constraint).
    out = re.sub(r"UNIQUE KEY\s+\w+\s*\(", "UNIQUE (", out, flags=re.IGNORECASE)
    # Storage engine / charset clause: no SQLite equivalent.
    out = re.sub(r"\s*ENGINE=\w+(\s+DEFAULT\s+CHARSET=\w+)?", "", out,
                 flags=re.IGNORECASE)
    # paramstyle: format -> qmark.
    out = out.replace("%s", "?")
    return out


class FakeCursor:
    """A DB-API cursor that logs the ORIGINAL statement (what the adapter
    actually generated — that is what tests assert on) and executes the
    translated one against SQLite."""

    def __init__(self, conn: "FakeConnection"):
        """Bind to the fake connection and open a real SQLite cursor under it."""
        self._conn = conn
        self._cur = conn.raw.cursor()

    def execute(self, sql, params=()):
        """Record `(sql, params)` on the connection's shared log, then run the
        translated statement — unless translation says to skip it (a server
        knob), in which case the log still shows the adapter emitted it."""
        self._conn.log.append((sql, tuple(params) if params else ()))
        translated = translate(sql)
        if translated is None:
            return
        self._cur.execute(translated, tuple(params) if params else ())

    def fetchall(self):
        """All remaining rows of the last query, as SQLite returns them."""
        return self._cur.fetchall()

    def fetchone(self):
        """The next row of the last query, or None."""
        return self._cur.fetchone()

    def close(self):
        """Close the underlying SQLite cursor."""
        self._cur.close()


class FakeConnection:
    """A DB-API connection backed by a real SQLite database file. Shares the
    driver-wide statement log so a test can inspect every statement any
    connection issued."""

    def __init__(self, path: str, log: list, connect_kwargs: dict):
        """Open the backing SQLite file (thread-check disabled, since the
        tracer's writer thread uses a connection opened on another thread — the
        same reason the real store does it) and remember the kwargs the adapter
        connected with, so tests can assert on timeouts and DSN parsing."""
        self.raw = sqlite3.connect(path, check_same_thread=False)
        self.raw.execute("PRAGMA foreign_keys = ON")
        self.log = log
        self.connect_kwargs = connect_kwargs

    def cursor(self):
        """Open a cursor (DB-API)."""
        return FakeCursor(self)

    def commit(self):
        """Commit the current transaction (DB-API)."""
        self.raw.commit()

    def rollback(self):
        """Roll back the current transaction (DB-API)."""
        self.raw.rollback()

    def close(self):
        """Close the connection (DB-API)."""
        self.raw.close()


class FakeDriver:
    """The stub driver module's state: where the backing SQLite file lives, the
    shared statement log, every connect() kwargs dict seen, and an optional
    canned connect failure."""

    def __init__(self, path: str, fail_connect: Exception | None = None):
        """Point the driver at `path` (its "server"), start an empty statement
        log, and optionally arm it to fail every connection attempt."""
        self.path = path
        self.log: list = []
        self.connections: list[dict] = []
        self.fail_connect = fail_connect

    def connect(self, *args, **kwargs):
        """The driver entry point both adapters call. Records the arguments
        (psycopg passes the DSN positionally, PyMySQL passes fields as kwargs —
        both are captured), raises the armed failure if there is one, and
        otherwise hands back a connection onto the backing SQLite file."""
        seen = dict(kwargs)
        if args:
            seen["dsn"] = args[0]
        self.connections.append(seen)
        if self.fail_connect is not None:
            raise self.fail_connect
        return FakeConnection(self.path, self.log, seen)

    def statements(self) -> list[str]:
        """Every statement issued so far, in order — the assertion surface for
        SQL-generation tests."""
        return [sql for sql, _ in self.log]

    def module(self) -> types.ModuleType:
        """Wrap this driver in a module object suitable for
        `monkeypatch.setitem(sys.modules, "psycopg"/"pymysql", ...)`, so the
        adapters' REAL lazy `import` statement resolves to it — the import path
        under test is the production one, not a patched-out function."""
        mod = types.ModuleType("fake_dbapi")
        mod.connect = self.connect
        mod.Error = Exception
        mod.driver = self
        return mod


def install(monkeypatch, driver_name: str, path: str,
            fail_connect: Exception | None = None) -> FakeDriver:
    """Install a stub driver under `driver_name` ("psycopg" or "pymysql") for
    the duration of a test, backed by the SQLite file at `path`. Returns the
    `FakeDriver` so the test can read its statement log and connect kwargs."""
    driver = FakeDriver(path, fail_connect=fail_connect)
    monkeypatch.setitem(sys.modules, driver_name, driver.module())
    return driver
