"""Adapter-level tests for the Postgres and MySQL backends: the SQL they
generate, how they connect and bound themselves, and — the part that matters
most — that a database which is down, slow or misconfigured degrades capture
without ever touching the host program.

The conformance suite (`test_store_conformance.py`) proves both adapters have
identical SEMANTICS. This file proves the dialect-specific DETAILS are right,
which a semantics test cannot see: `TEXT` vs `VARCHAR(64)` keys, `ON CONFLICT
DO NOTHING` vs `ON DUPLICATE KEY UPDATE`, connect/statement timeouts, and the
install hint a missing extra produces."""
from __future__ import annotations

import logging
import socket
import sys
import threading
import time
import types

import pytest

from ctxdiff import trace
from ctxdiff.models import Block, CallBlock, content_hash
from ctxdiff.store.mysql import MySQLStore, _MySQLDialect
from ctxdiff.store.postgres import PostgresStore, _PostgresDialect
from ctxdiff.store.sql import SQLStore

from tests import fakedb

PG_DSN = "postgresql://u:p@localhost:5432/ctxdiff"
MY_DSN = "mysql://u:p@localhost:3306/ctxdiff"


class _Usage:
    prompt_tokens = 3; completion_tokens = 1; total_tokens = 4
class _Resp:
    usage = _Usage()


class _FakeCompletions:
    def __init__(self): self.calls = []
    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Resp()


class _FakeChat:
    def __init__(self): self.completions = _FakeCompletions()


class _FakeOpenAI:
    __module__ = "openai"
    def __init__(self): self.chat = _FakeChat()


def _cb(text: str) -> CallBlock:
    """One CallBlock with a real content hash, for exercising the write path."""
    block = Block(content_hash=content_hash("user", "message", text),
                  role="user", kind="message", text=text,
                  token_count=1, token_method="estimate")
    return CallBlock(block=block, position=0, label="user",
                     label_source="heuristic")


def _pg(monkeypatch, tmp_path, **kwargs):
    """Install the stub psycopg driver and return (backend, driver)."""
    driver = fakedb.install(monkeypatch, "psycopg", str(tmp_path / "pg.sqlite"),
                            **kwargs)
    return PostgresStore(dsn=PG_DSN), driver


def _my(monkeypatch, tmp_path, **kwargs):
    """Install the stub PyMySQL driver and return (backend, driver)."""
    driver = fakedb.install(monkeypatch, "pymysql", str(tmp_path / "my.sqlite"),
                            **kwargs)
    return MySQLStore(dsn=MY_DSN), driver


# --- auto-create DDL ----------------------------------------------------------


def test_postgres_auto_creates_all_four_tables_if_not_exists(monkeypatch, tmp_path):
    """First connect creates ctxdiff's four tables — with IF NOT EXISTS, so
    connecting again is a harmless no-op and there is no migration step."""
    backend, driver = _pg(monkeypatch, tmp_path)
    backend.open_session(project="p", provider="openai", started_at="t").close()
    ddl = [s for s in driver.statements() if s.startswith("CREATE TABLE")]
    assert len(ddl) == 4
    for table in ("ctxdiff_run", "ctxdiff_call", "ctxdiff_block",
                  "ctxdiff_call_block"):
        assert any(f"CREATE TABLE IF NOT EXISTS {table} " in s for s in ddl)


def test_mysql_auto_creates_all_four_tables_if_not_exists(monkeypatch, tmp_path):
    """Same auto-create contract on MySQL."""
    backend, driver = _my(monkeypatch, tmp_path)
    backend.open_session(project="p", provider="openai", started_at="t").close()
    ddl = [s for s in driver.statements() if s.startswith("CREATE TABLE")]
    assert len(ddl) == 4
    for table in ("ctxdiff_run", "ctxdiff_call", "ctxdiff_block",
                  "ctxdiff_call_block"):
        assert any(f"CREATE TABLE IF NOT EXISTS {table} " in s for s in ddl)


def test_postgres_ddl_uses_unbounded_text_columns():
    """Postgres' TEXT is both unbounded and indexable, so keys are plain TEXT —
    no invented VARCHAR lengths. This is the structural difference from MySQL."""
    ddl = "\n".join(_PostgresDialect.ddl())
    assert "content_hash TEXT PRIMARY KEY" in ddl
    assert "id              TEXT PRIMARY KEY" in ddl
    assert "VARCHAR" not in ddl


def test_mysql_ddl_uses_varchar_keys_longtext_bodies_and_innodb():
    """MySQL cannot index TEXT without a prefix length, so every key column is
    a bounded VARCHAR while free text stays LONGTEXT; InnoDB + utf8mb4 are
    explicit because the foreign keys and emoji-bearing prompts depend on
    them."""
    ddl = "\n".join(_MySQLDialect.ddl())
    assert "content_hash VARCHAR(64)" in ddl
    assert "PRIMARY KEY (content_hash)" in ddl
    assert "body         LONGTEXT NOT NULL" in ddl
    assert ddl.count("ENGINE=InnoDB DEFAULT CHARSET=utf8mb4") == 4
    assert "FOREIGN KEY (run_id) REFERENCES ctxdiff_run(id)" in ddl


def test_mysql_ddl_pins_binary_collation_on_every_key_and_ordering_column():
    """MySQL's DEFAULT collation for utf8mb4 is `utf8mb4_0900_ai_ci` —
    ACCENT- and CASE-INSENSITIVE — which silently changes what the schema
    means:

    - `content_hash` is the PRIMARY KEY of the content-addressed block table,
      so under a *_ci collation two hashes differing only in case are the SAME
      key: the second block is deduped away and reads back the FIRST one's
      text. Silent data corruption.
    - `started_at` sorts by collation rather than by bytes, so an ISO timestamp
      comparison stops being a byte comparison.

    Pinning `ascii_bin` (these columns are hex ids and ISO-8601 timestamps —
    ASCII by construction) makes both comparisons byte-exact, matching SQLite's
    BINARY and Postgres' C-locale TEXT."""
    ddl = "\n".join(_MySQLDialect.ddl())
    binary = "CHARACTER SET ascii COLLATE ascii_bin"
    for line in ddl.splitlines():
        column = line.strip().split(" ")[0]
        if column in ("id", "run_id", "block_id", "call_id", "content_hash",
                      "started_at"):
            assert binary in line, f"{column} is not byte-collated: {line!r}"
    # ...and it is spelled on the columns, never as a table-wide default that a
    # later ALTER could quietly change.
    assert "DEFAULT CHARSET=utf8mb4" in ddl


def test_mysql_ddl_bounds_only_the_columns_that_must_be_indexable():
    """Only the four id/hash columns need a bounded VARCHAR (MySQL cannot key a
    TEXT column without a prefix length). Everything else is free text and must
    NOT be VARCHAR(255): under STRICT mode a 400-character error/label/agent/
    project raises (1406, "Data too long") and — because the write is one
    transaction — drops the WHOLE call, on MySQL only, where SQLite and
    Postgres store it fine."""
    ddl = "\n".join(_MySQLDialect.ddl())
    for free_text in ("error", "label", "label_source", "agent", "step",
                      "provider", "project", "role", "kind", "token_method",
                      "ctxdiff_version"):
        assert "VARCHAR" not in _column_line(ddl, free_text), \
            f"{free_text} is a bounded VARCHAR and can be truncated/rejected"
    # latency_ms is a duration in milliseconds: INT tops out at ~24 days.
    assert "BIGINT" in _column_line(ddl, "latency_ms")
    # The keys stay bounded — that is what makes them indexable at all.
    assert "VARCHAR(64)" in _column_line(ddl, "content_hash")


def _column_line(ddl: str, column: str) -> str:
    """Every DDL line that DEFINES `column` (the line starting with its name),
    joined — so a per-column type assertion can't be satisfied by a mention of
    the same word in another table's key clause."""
    return "\n".join(line for line in ddl.splitlines()
                     if line.strip().split(" ")[0] == column)


def test_mysql_ddl_avoids_reserved_words():
    """The network schema deliberately renames `usage`/`position`/`text`, since
    `usage` is a MySQL RESERVED word and the others are keyword-adjacent —
    cheaper than quoting every identifier in every statement."""
    ddl = "\n".join(_MySQLDialect.ddl())
    assert "usage_json" in ddl and " usage " not in ddl
    assert "  pos " in ddl and "position" not in ddl
    assert "  body " in ddl


# --- dedup dialect ------------------------------------------------------------


def test_postgres_dedup_uses_on_conflict_do_nothing(monkeypatch, tmp_path):
    """Block dedup on Postgres is `ON CONFLICT (content_hash) DO NOTHING` —
    naming the column, so only a duplicate CONTENT HASH can no-op."""
    backend, driver = _pg(monkeypatch, tmp_path)
    store = backend.open_session(project="p", provider="openai", started_at="t")
    store.record_call(seq=1, params={}, usage=None, latency_ms=None,
                      error=None, call_blocks=[_cb("hello")])
    store.close()
    inserts = [s for s in driver.statements() if "INTO ctxdiff_block" in s]
    assert len(inserts) == 1
    assert "ON CONFLICT (content_hash) DO NOTHING" in inserts[0]


def test_mysql_dedup_uses_on_duplicate_key_update_not_insert_ignore(
        monkeypatch, tmp_path):
    """Block dedup on MySQL is a no-op `ON DUPLICATE KEY UPDATE`, deliberately
    NOT `INSERT IGNORE` — which would downgrade every error (truncation, bad
    values, FK violations) to a warning and could half-store a block."""
    backend, driver = _my(monkeypatch, tmp_path)
    store = backend.open_session(project="p", provider="openai", started_at="t")
    store.record_call(seq=1, params={}, usage=None, latency_ms=None,
                      error=None, call_blocks=[_cb("hello")])
    store.close()
    inserts = [s for s in driver.statements() if "INTO ctxdiff_block" in s]
    assert len(inserts) == 1
    assert "ON DUPLICATE KEY UPDATE content_hash = content_hash" in inserts[0]
    assert "INSERT IGNORE" not in inserts[0]


def test_dedup_insert_is_issued_per_block_but_stores_one_row(monkeypatch, tmp_path):
    """The adapter always ISSUES the insert (it is the DB that decides the row
    already exists) — two calls sharing a block emit two inserts and store one
    row. Proves dedup is the database's atomic upsert, not a client-side cache
    that a second process would miss."""
    backend, driver = _pg(monkeypatch, tmp_path)
    store = backend.open_session(project="p", provider="openai", started_at="t")
    store.record_call(seq=1, params={}, usage=None, latency_ms=None,
                      error=None, call_blocks=[_cb("same")])
    store.record_call(seq=2, params={}, usage=None, latency_ms=None,
                      error=None, call_blocks=[_cb("same")])
    try:
        assert len([s for s in driver.statements()
                    if "INTO ctxdiff_block" in s]) == 2
        assert store._query("SELECT COUNT(*) FROM ctxdiff_block")[0][0] == 1
    finally:
        store.close()


# --- connecting: bounded, lazy, and clearly diagnosed -------------------------


def test_postgres_bounds_connect_and_statement_time(monkeypatch, tmp_path):
    """A networked store must not be able to hang the writer thread: the
    connect is bounded client-side and every statement is bounded server-side."""
    driver = fakedb.install(monkeypatch, "psycopg", str(tmp_path / "pg.sqlite"))
    backend = PostgresStore(dsn=PG_DSN, connect_timeout=3, statement_timeout=7)
    backend.open_session(project="p", provider="openai", started_at="t").close()
    assert driver.connections[0]["connect_timeout"] == 3
    assert driver.connections[0]["autocommit"] is False
    assert "SET statement_timeout = 7000" in driver.statements()


def test_mysql_bounds_connect_read_and_write_time(monkeypatch, tmp_path):
    """PyMySQL's socket read/write timeouts matter as much as connect: a server
    that accepts a connection and then goes silent must not wedge the writer.

    The socket bound is derived from the STATEMENT timeout, not the connect
    one: with `read_timeout == connect_timeout` (5s) and `max_execution_time`
    (10s), the server-side knob was unreachable — every statement was killed
    client-side first, so a legitimately slow 5-10s write died as a "lost
    connection" instead of running to completion. A small margin on top lets
    the server's own timeout fire FIRST and report a real error, while the
    socket timeout stays the backstop for a server that says nothing at all."""
    driver = fakedb.install(monkeypatch, "pymysql", str(tmp_path / "my.sqlite"))
    backend = MySQLStore(dsn=MY_DSN, connect_timeout=4, statement_timeout=9)
    backend.open_session(project="p", provider="openai", started_at="t").close()
    conn = driver.connections[0]
    assert conn["connect_timeout"] == 4
    assert conn["read_timeout"] == conn["write_timeout"]
    assert conn["read_timeout"] > 9          # never cuts a legal statement short
    assert conn["read_timeout"] <= 9 + 5     # ...but still bounded, and tightly
    assert conn["autocommit"] is False
    assert conn["charset"] == "utf8mb4"
    assert "SET SESSION max_execution_time = 9000" in driver.statements()


def test_mysql_parses_dsn_into_connect_fields(monkeypatch, tmp_path):
    """A URL DSN is decomposed into PyMySQL's connect fields, percent-decoding
    credentials (passwords routinely contain `@` or `/`)."""
    backend = MySQLStore(dsn="mysql://alice:p%40ss%2Fword@db.internal:3307/traces")
    kwargs = backend.connect_kwargs()
    assert kwargs == {"host": "db.internal", "port": 3307, "user": "alice",
                      "password": "p@ss/word", "database": "traces"}


def test_mysql_explicit_fields_override_the_dsn():
    """Explicit constructor fields beat the DSN, so a DSN from the environment
    can be adjusted in code without string surgery."""
    backend = MySQLStore(dsn=MY_DSN, host="replica", database="other")
    kwargs = backend.connect_kwargs()
    assert kwargs["host"] == "replica"
    assert kwargs["database"] == "other"
    assert kwargs["port"] == 3306          # untouched fields still come from the DSN


def test_backends_do_not_connect_until_used(monkeypatch, tmp_path):
    """Constructing a backend touches nothing — no driver import, no socket —
    so `configure(store=PostgresStore(...))` at module import is free and a
    dead database cannot fail an import."""
    driver = fakedb.install(monkeypatch, "psycopg", str(tmp_path / "pg.sqlite"))
    PostgresStore(dsn=PG_DSN)
    MySQLStore(dsn=MY_DSN)
    assert driver.connections == []


def test_missing_psycopg_gives_an_install_hint(monkeypatch, tmp_path):
    """A configured Postgres store with the extra not installed must produce an
    actionable install hint AT CONNECT TIME — never an import crash when the
    user merely imports ctxdiff."""
    monkeypatch.setitem(sys.modules, "psycopg", None)  # makes `import psycopg` fail
    with pytest.raises(ImportError) as excinfo:
        PostgresStore(dsn=PG_DSN).open_session(project="p", provider="openai")
    assert "pip install 'ctxdiff[postgres]'" in str(excinfo.value)


def test_missing_pymysql_gives_an_install_hint(monkeypatch, tmp_path):
    """Same for MySQL."""
    monkeypatch.setitem(sys.modules, "pymysql", None)
    with pytest.raises(ImportError) as excinfo:
        MySQLStore(dsn=MY_DSN).open_session(project="p", provider="openai")
    assert "pip install 'ctxdiff[mysql]'" in str(excinfo.value)


def test_open_reader_on_an_empty_store_is_a_clear_error(monkeypatch, tmp_path):
    """Reading a database ctxdiff has never written says "no sessions", not
    "no such table" — the schema is ensured before the lookup."""
    backend, _ = _pg(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="no sessions recorded"):
        backend.open_reader()


# --- fail-open: a dead database must never touch the host ---------------------


def test_unreachable_database_never_breaks_the_host_call(monkeypatch, tmp_path,
                                                         caplog):
    """THE guarantee. With a database that refuses every connection, the wrapped
    client still returns the provider's real response, no exception escapes, and
    capture degrades with exactly ONE warning."""
    fakedb.install(monkeypatch, "psycopg", str(tmp_path / "pg.sqlite"),
                   fail_connect=OSError("connection refused"))
    caplog.set_level(logging.WARNING, logger="ctxdiff")

    t = trace.init("p", store=PostgresStore(dsn=PG_DSN))
    client = _FakeOpenAI()
    wrapped = t.wrap(client)
    for i in range(5):
        got = wrapped.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": f"q{i}"}])
        assert isinstance(got, _Resp)      # the host's own response, untouched
    t.close()

    assert len(client.chat.completions.calls) == 5   # every real call went through
    degraded = [r for r in caplog.records if "capture degraded" in r.message]
    assert len(degraded) == 1


def test_unreachable_database_leaves_no_local_file(monkeypatch, tmp_path):
    """A failed database never silently falls back to writing a `.ctrace` — a
    user who configured Postgres must not find a surprise file on disk."""
    fakedb.install(monkeypatch, "pymysql", str(tmp_path / "my.sqlite"),
                   fail_connect=OSError("connection refused"))
    monkeypatch.chdir(tmp_path)
    t = trace.init("p", store=MySQLStore(dsn=MY_DSN))
    wrapped = t.wrap(_FakeOpenAI())
    wrapped.chat.completions.create(model="gpt-4o", messages=[])
    t.close()
    assert list(tmp_path.glob("*.ctrace")) == []
    assert t.path is None      # a networked backend has no path


def test_a_database_that_dies_mid_run_degrades_quietly(monkeypatch, tmp_path,
                                                       caplog):
    """A store that starts healthy and then fails EVERY write warns once and
    keeps the host running — the writer thread absorbs it, the host never sees
    it, and `close()` still returns."""
    backend, _ = _pg(monkeypatch, tmp_path)
    caplog.set_level(logging.WARNING, logger="ctxdiff")
    t = trace.init("p", store=backend)
    client = _FakeOpenAI()
    wrapped = t.wrap(client)

    def _boom(*a, **k):
        raise RuntimeError("server went away")
    t._recorder._ct.record_call = _boom

    for i in range(4):
        assert isinstance(wrapped.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": f"q{i}"}]), _Resp)
    t.close()

    assert len(client.chat.completions.calls) == 4
    failures = [r for r in caplog.records if "failed to persist" in r.message]
    assert len(failures) == 1


# --- bounded retry ------------------------------------------------------------


class _StubDialect:
    """Minimal dialect for exercising the retry loop in isolation: an exception
    whose `args[0]` is 'transient' is retryable, and one whose `args[0]` is
    'lost' means the connection itself died."""
    @staticmethod
    def is_transient(exc):
        return getattr(exc, "args", (None,))[0] == "transient"

    @staticmethod
    def is_connection_lost(exc):
        return getattr(exc, "args", (None,))[0] == "lost"


class _StubConn:
    """A connection that counts rollbacks and closes — the retry loop's only
    interactions with it. `closed` mirrors psycopg's liveness flag, which is
    how the store recognizes a connection that died earlier."""
    def __init__(self): self.rollbacks = 0; self.closed = False
    def rollback(self): self.rollbacks += 1
    def close(self): self.closed = True


def test_write_retry_retries_transient_failures_then_succeeds():
    """A transient failure (deadlock, serialization, dropped connection) is
    retried rather than dropping the call, and each attempt rolls back first so
    a retry cannot double-write."""
    store = SQLStore(_StubConn(), _StubDialect, "run-1")
    attempts = []

    def _flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("transient")
        return "ok"

    assert store._with_write_retry(_flaky) == "ok"
    assert len(attempts) == 3
    assert store._conn.rollbacks == 2


def test_write_retry_gives_up_bounded_so_the_writer_never_hangs():
    """A permanently transient failure gives up after a FIXED number of
    attempts and re-raises into the fail-open guard — an unbounded retry would
    block the writer thread and, through it, `tracer.close()`."""
    store = SQLStore(_StubConn(), _StubDialect, "run-1")
    attempts = []

    def _always():
        attempts.append(1)
        raise RuntimeError("transient")

    with pytest.raises(RuntimeError):
        store._with_write_retry(_always)
    assert len(attempts) == 3


def test_write_retry_does_not_retry_a_non_transient_failure():
    """A bug, a bad value or a constraint violation re-raises IMMEDIATELY —
    retrying it would only delay the fail-open drop."""
    store = SQLStore(_StubConn(), _StubDialect, "run-1")
    attempts = []

    def _bad():
        attempts.append(1)
        raise RuntimeError("permanent")

    with pytest.raises(RuntimeError):
        store._with_write_retry(_bad)
    assert len(attempts) == 1


def test_postgres_classifies_transient_errors_by_sqlstate():
    """Postgres transients are recognized by SQLSTATE: serialization failure,
    deadlock, lock-not-available and the whole 08xxx connection class."""
    def err(state):
        exc = RuntimeError("boom")
        exc.sqlstate = state
        return exc
    assert _PostgresDialect.is_transient(err("40001"))   # serialization_failure
    assert _PostgresDialect.is_transient(err("40P01"))   # deadlock_detected
    assert _PostgresDialect.is_transient(err("55P03"))   # lock_not_available
    assert _PostgresDialect.is_transient(err("08006"))   # connection_failure
    assert not _PostgresDialect.is_transient(err("23505"))  # unique_violation
    assert not _PostgresDialect.is_transient(RuntimeError("no sqlstate"))


def test_mysql_classifies_transient_errors_by_errno():
    """MySQL transients are recognized by PyMySQL's `args[0]` errno: deadlock,
    lock-wait timeout, and the two connection-lost codes."""
    def err(code):
        return RuntimeError(code, "boom")
    assert _MySQLDialect.is_transient(err(1213))   # deadlock found
    assert _MySQLDialect.is_transient(err(1205))   # lock wait timeout
    assert _MySQLDialect.is_transient(err(2006))   # server has gone away
    assert _MySQLDialect.is_transient(err(2013))   # lost connection
    assert not _MySQLDialect.is_transient(err(1062))  # duplicate entry
    assert not _MySQLDialect.is_transient(RuntimeError("no errno"))


def test_write_retry_reopens_the_connection_before_retrying_a_lost_one():
    """A CONNECTION-level failure is retried on a NEW connection, not the dead
    one. Classifying a dropped connection as "transient" without reopening is
    inert — every attempt re-runs against the same dead socket, fails
    identically, and capture is over for the process. This is what a PgBouncer
    recycle, a failover or a `pg_terminate_backend` actually looks like."""
    dead, fresh = _StubConn(), _StubConn()
    store = SQLStore(dead, _StubDialect, "run-1", reconnect=lambda: fresh)
    attempts = []

    def _flaky():
        attempts.append(store._conn)
        if len(attempts) == 1:
            raise RuntimeError("lost")
        return "ok"

    assert store._with_write_retry(_flaky) == "ok"
    assert attempts == [dead, fresh]     # the retry ran on the NEW connection
    assert dead.closed is True           # ...and the old socket was released
    assert store._conn is fresh


def test_write_retry_recognizes_a_connection_that_already_closed_itself():
    """After the first failure a driver stops attaching an error code at all
    and simply reports "the connection is closed", so the store also treats a
    connection whose own liveness flag says dead as lost — otherwise the very
    error that proves reopening is needed looks like a non-retryable bug."""
    dead, fresh = _StubConn(), _StubConn()
    dead.closed = True                   # as psycopg marks it after a kill
    store = SQLStore(dead, _StubDialect, "run-1", reconnect=lambda: fresh)
    attempts = []

    def _flaky():
        attempts.append(store._conn)
        if len(attempts) == 1:
            raise RuntimeError("no error code here")   # not 'lost', not transient
        return "ok"

    assert store._with_write_retry(_flaky) == "ok"
    assert attempts == [dead, fresh]


def test_write_retry_gives_up_when_the_database_is_still_unreachable():
    """If reopening ALSO fails the server is genuinely down: give up now, into
    the fail-open guard, rather than burning the retry budget connecting to
    something that is not there."""
    def _no_server():
        raise OSError("connection refused")

    store = SQLStore(_StubConn(), _StubDialect, "run-1", reconnect=_no_server)
    attempts = []

    def _lost():
        attempts.append(1)
        raise RuntimeError("lost")

    with pytest.raises(RuntimeError, match="lost"):
        store._with_write_retry(_lost)
    assert len(attempts) == 1            # not retried against a missing server


# --- the auto-create race -----------------------------------------------------


def test_postgres_classifies_the_catalog_race_and_nothing_else():
    """`CREATE TABLE IF NOT EXISTS` is a check, not a lock: two backends can
    both pass it and the loser dies on a system index. Those codes — and ONLY
    those — make the DDL worth re-running (the winner has by then made it a
    no-op). An ordinary write-path 23505 must NOT be swept in here, which is
    why this classifier is separate from `is_transient` and is consulted only
    by `_ensure_schema`."""
    def err(state):
        exc = RuntimeError("boom")
        exc.sqlstate = state
        return exc
    assert _PostgresDialect.is_schema_race(err("23505"))   # unique_violation on pg_type
    assert _PostgresDialect.is_schema_race(err("42P07"))   # duplicate_table
    assert _PostgresDialect.is_schema_race(err("42710"))   # duplicate_object
    assert not _PostgresDialect.is_schema_race(err("40001"))
    assert not _PostgresDialect.is_schema_race(RuntimeError("no sqlstate"))
    # ...and the race codes stay OUT of the write-path classifier.
    assert not _PostgresDialect.is_transient(err("42P07"))


def test_postgres_classifies_a_killed_connection_as_lost():
    """`pg_terminate_backend` reports 57P01; a crash-restart 57P02; the whole
    08xxx class is a connection exception. All of them mean "reopen", which is
    a different question from "is this worth retrying"."""
    def err(state):
        exc = RuntimeError("boom")
        exc.sqlstate = state
        return exc
    assert _PostgresDialect.is_connection_lost(err("57P01"))
    assert _PostgresDialect.is_connection_lost(err("08006"))
    assert not _PostgresDialect.is_connection_lost(err("40001"))  # deadlock: same conn
    assert not _PostgresDialect.is_connection_lost(RuntimeError("no sqlstate"))


def test_mysql_classifies_a_killed_connection_as_lost():
    """PyMySQL reports a `KILL`/restart as 2006 or 2013."""
    def err(code):
        return RuntimeError(code, "boom")
    assert _MySQLDialect.is_connection_lost(err(2006))
    assert _MySQLDialect.is_connection_lost(err(2013))
    assert not _MySQLDialect.is_connection_lost(err(1213))   # deadlock: same conn
    assert not _MySQLDialect.is_connection_lost(RuntimeError("no errno"))


def test_ensure_schema_retries_the_catalog_race_once_then_succeeds(monkeypatch,
                                                                   tmp_path):
    """The cold-start race, in the small: the first DDL attempt loses on the
    catalog, the retry finds the table already there and returns. Without it,
    seven of eight replicas starting together against an empty database record
    NOTHING for their whole run."""
    backend, _ = _pg(monkeypatch, tmp_path)
    real_execute = fakedb.FakeCursor.execute
    attempted: list[str] = []

    def _lose_the_race_once(self, sql, params=()):
        if sql.startswith("CREATE TABLE"):
            attempted.append(sql)
            if len(attempted) == 1:
                exc = RuntimeError(
                    'duplicate key value violates unique constraint '
                    '"pg_type_typname_nsp_index"')
                exc.sqlstate = "23505"
                raise exc
        return real_execute(self, sql, params)

    monkeypatch.setattr(fakedb.FakeCursor, "execute", _lose_the_race_once)
    backend.open_session(project="p", provider="openai", started_at="t").close()

    # 1 statement that lost the race + all 4 re-run on the retry.
    assert len(attempted) == 5


def test_ensure_schema_does_not_retry_a_real_failure(monkeypatch, tmp_path):
    """A permission error, a bad type, a broken connection: re-running the DDL
    would only delay the fail-open drop, so anything that is not the catalog
    race raises on the first attempt."""
    backend, _ = _pg(monkeypatch, tmp_path)
    real_execute = fakedb.FakeCursor.execute
    attempted: list[str] = []

    def _always_fail(self, sql, params=()):
        if sql.startswith("CREATE TABLE"):
            attempted.append(sql)
            exc = RuntimeError("permission denied for schema public")
            exc.sqlstate = "42501"
            raise exc
        return real_execute(self, sql, params)

    monkeypatch.setattr(fakedb.FakeCursor, "execute", _always_fail)
    with pytest.raises(RuntimeError, match="permission denied"):
        backend.open_session(project="p", provider="openai", started_at="t")
    assert len(attempted) == 1


# --- the host thread never does store I/O ------------------------------------


class _SlowBackend:
    """A backend whose `open_session` takes a LONG time — the shape of a
    database that is reachable but wedged (a hung pooler, a partitioned
    network, a box under load). Records which thread called it, since where
    the connect happens is the whole point."""

    def __init__(self, delay: float = 3.0, fail: bool = False):
        self.delay = delay
        self.fail = fail
        self.opened_on: str | None = None
        self.store: _CountingStore | None = None
        self.entered = threading.Event()

    def open_session(self, project, provider, model="", started_at=""):
        self.opened_on = threading.current_thread().name
        self.entered.set()
        time.sleep(self.delay)
        if self.fail:
            raise OSError("connection timed out")
        self.store = _CountingStore()
        return self.store

    def open_reader(self):
        raise NotImplementedError


class _CountingStore:
    """A minimal `Store` that just counts what it was asked to persist."""

    def __init__(self): self.calls = []
    def record_call(self, seq, params, usage, latency_ms, error, call_blocks,
                    agent=None, step=None, provider=None):
        self.calls.append(seq)
        return f"call-{seq}"
    def note_model(self, model): pass
    def list_sessions(self): return []
    def get_run(self, session_id=None): raise NotImplementedError
    def get_calls(self, session_id=None): raise NotImplementedError
    def get_call_blocks(self, call_id): raise NotImplementedError
    def close(self): pass


def test_opening_the_session_never_happens_on_the_host_thread(tmp_path):
    """`wrap()` must not do store I/O — not a connect, not a handshake, not a
    schema check.

    A timeout is a bound on the DAMAGE, not a fix: with the open on the host
    thread, a healthy-but-slow database still costs the agent's first call up
    to the full connect timeout, and a database that completes its TCP
    handshake and then stops answering blocks past it entirely (a client
    connect timeout covers connect+auth only). Doing the open on the writer
    thread — the thread that already owns every write — removes the class of
    bug rather than shrinking it: there is no I/O left on the call path to
    bound."""
    backend = _SlowBackend(delay=2.0)
    t = trace.init("slow", store=backend)

    started = time.monotonic()
    wrapped = t.wrap(_FakeOpenAI())
    for _ in range(3):
        assert isinstance(wrapped.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}]), _Resp)
    host_elapsed = time.monotonic() - started

    assert backend.entered.wait(5)                  # the open really was attempted
    assert host_elapsed < 0.5                       # ...just not here
    assert backend.opened_on == "ctxdiff-writer"
    t.close()


def test_a_slow_store_still_records_every_call_once_it_opens(tmp_path):
    """Deferring the open loses nothing: calls made while the store is still
    connecting wait in the writer's queue and are persisted, in order, as soon
    as it is up."""
    backend = _SlowBackend(delay=0.4)
    t = trace.init("slow", store=backend)
    wrapped = t.wrap(_FakeOpenAI())
    for i in range(4):
        wrapped.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": f"q{i}"}])
    t.close()

    assert backend.store.calls == [1, 2, 3, 4]


def test_close_is_bounded_by_the_backends_statement_timeout(tmp_path, caplog):
    """`close()` must return in bounded time even when the writer is stuck on a
    database that has stopped answering. The bound is the backend's OWN
    statement timeout (plus a margin to let a legitimately-slow statement
    finish and the connection close), not a flat 30 seconds — a host that
    already lost its database should not also wait half a minute to exit."""
    backend = _SlowBackend(delay=30.0)
    caplog.set_level(logging.WARNING, logger="ctxdiff")
    backend.statement_timeout = 1              # as PostgresStore/MySQLStore expose
    t = trace.init("stuck", store=backend)
    t.wrap(_FakeOpenAI())
    assert backend.entered.wait(5)             # writer is now stuck in open_session

    started = time.monotonic()
    t.close()
    elapsed = time.monotonic() - started

    assert elapsed < 10                        # bounded, and well under the old 30s
    assert [r for r in caplog.records if "did not drain" in r.message]


def test_close_keeps_the_generous_bound_for_the_local_sqlite_store(tmp_path):
    """...but a backend with no statement timeout of its own (the local
    `.ctrace`) keeps the generous join. SQLite's own contention budget is a 5s
    busy timeout across 6 retries, so a shorter join would abandon writes that
    were about to succeed."""
    t = trace.init("local", path=str(tmp_path / "r.ctrace"))
    t.wrap(_FakeOpenAI())
    assert t._writer._close_timeout == 30.0
    t.close()


def test_postgres_keeps_a_wedged_connection_from_hanging_the_writer(monkeypatch,
                                                                    tmp_path):
    """psycopg gets TCP keepalives and `tcp_user_timeout` as well as
    `connect_timeout`.

    `connect_timeout` bounds connect+auth and NOTHING after it, and a
    server-side `statement_timeout` can never fire when the packets carrying it
    are being dropped — so a connection that completes its handshake and is
    then partitioned away (a wedged box, a hung pooler, a NAT that forgot the
    flow) leaves the writer blocked in `recv` indefinitely. `tcp_user_timeout`
    is the client-side bound on unacknowledged data, and the keepalives make an
    IDLE connection notice the same thing."""
    driver = fakedb.install(monkeypatch, "psycopg", str(tmp_path / "pg.sqlite"))
    backend = PostgresStore(dsn=PG_DSN, connect_timeout=3, statement_timeout=7)
    backend.open_session(project="p", provider="openai", started_at="t").close()

    conn = driver.connections[0]
    assert conn["connect_timeout"] == 3
    assert conn["keepalives"] == 1
    assert conn["keepalives_idle"] > 0
    assert conn["keepalives_interval"] > 0
    assert conn["keepalives_count"] > 0
    # Milliseconds, matching the statement bound: a statement that cannot get
    # its bytes through is dead by the time it would have timed out anyway.
    assert conn["tcp_user_timeout"] == 7000


def test_postgres_connects_without_tcp_user_timeout_where_it_is_unsupported(
        monkeypatch, tmp_path):
    """`tcp_user_timeout` is a Linux socket option. libpq REFUSES the whole
    connection when asked for it on a platform that lacks it, so a knob meant
    to harden the connection must never be able to prevent one: an
    unsupported-option failure retries once without it."""
    driver = fakedb.install(monkeypatch, "psycopg", str(tmp_path / "pg.sqlite"))
    real_connect = driver.connect
    attempts: list[dict] = []

    def _reject_tcp_user_timeout(*args, **kwargs):
        attempts.append(dict(kwargs))
        if kwargs.get("tcp_user_timeout"):
            raise RuntimeError("setsockopt(TCP_USER_TIMEOUT) not supported")
        return real_connect(*args, **kwargs)

    monkeypatch.setitem(sys.modules, "psycopg",
                        _module_with_connect(_reject_tcp_user_timeout))

    backend = PostgresStore(dsn=PG_DSN)
    store = backend.open_session(project="p", provider="openai", started_at="t")
    store.close()

    assert len(attempts) == 2                          # rejected, then retried
    assert "tcp_user_timeout" not in attempts[1]       # ...without the knob
    assert attempts[1]["keepalives"] == 1              # keepalives still set
    assert attempts[1]["connect_timeout"] == attempts[0]["connect_timeout"]


def _module_with_connect(connect):
    """A stand-in `psycopg` module exposing just the `connect` a test wants —
    enough for the adapter's real lazy `import psycopg` to resolve."""
    mod = types.ModuleType("fake_psycopg")
    mod.connect = connect
    mod.Error = Exception
    return mod


@pytest.fixture
def blackhole_server():
    """A TCP endpoint that ACCEPTS connections and then never answers — the
    real, unmocked shape of a wedged database box: the socket connects, so
    "unreachable" handling never triggers, and then nothing comes back.

    Yields `(host, port)` and holds every accepted socket open for the duration
    (closing them would turn the wedge into a clean disconnect, which is the
    easy case)."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    listener.settimeout(0.2)
    accepted: list[socket.socket] = []
    stop = threading.Event()

    def _accept_and_ignore():
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except (socket.timeout, TimeoutError):
                continue
            except OSError:
                break
            accepted.append(conn)          # held open, deliberately unanswered

    thread = threading.Thread(target=_accept_and_ignore, daemon=True)
    thread.start()
    try:
        yield listener.getsockname()
    finally:
        stop.set()
        thread.join(2)
        for conn in accepted:
            conn.close()
        listener.close()


def test_a_wedged_server_bounds_both_wrap_and_close(blackhole_server, caplog):
    """THE hang, end to end, against a REAL psycopg and a REAL socket that
    accepts and then says nothing.

    Both halves are asserted because both used to fail: `wrap()` blocked on the
    host thread for as long as the connect took (and, past the handshake, with
    no bound at all), and `close()` then burned its full 30-second join. The
    host's own calls must go through untouched throughout, and `close()` must
    return promptly — a program whose database has wedged still has to be able
    to exit."""
    pytest.importorskip(
        "psycopg", reason="the wedged-server test needs the real psycopg "
                          "(pip install 'ctxdiff[postgres]')")
    host, port = blackhole_server
    caplog.set_level(logging.WARNING, logger="ctxdiff")
    backend = PostgresStore(dsn=f"postgresql://u:p@{host}:{port}/ctxdiff",
                            connect_timeout=2, statement_timeout=2)

    t = trace.init("wedged", store=backend)
    client = _FakeOpenAI()

    started = time.monotonic()
    wrapped = t.wrap(client)
    for i in range(3):
        assert isinstance(wrapped.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": f"q{i}"}]),
            _Resp)
    host_elapsed = time.monotonic() - started

    closing = time.monotonic()
    t.close()
    close_elapsed = time.monotonic() - closing

    assert host_elapsed < 0.5      # the host never waits on the database at all
    assert close_elapsed < 8       # bounded by the statement timeout, not 30s
    assert len(client.chat.completions.calls) == 3     # every real call ran
    assert [r for r in caplog.records if "capture degraded" in r.message]
