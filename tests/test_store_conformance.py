"""The backend conformance suite: ONE set of semantics, asserted identically
against EVERY storage backend.

The point of a `Store` protocol is that swapping SQLite for Postgres or MySQL
changes nothing an analyzer can observe. That is only true if it is tested, so
every test in this file runs against every backend:

- `sqlite`        — always; the zero-config default.
- `postgres-stub` — always; the real `PostgresStore`, its real SQL and its real
  lazy `import psycopg`, executed against a stub driver (see `tests/fakedb.py`)
  so the adapter has genuine coverage in CI with no server.
- `mysql-stub`    — likewise for `MySQLStore` / PyMySQL.
- `postgres-live` — only when `CTXDIFF_TEST_POSTGRES_DSN` is set; otherwise an
  EXPLICIT, visible skip. Never a vacuous pass.
- `mysql-live`    — likewise, on `CTXDIFF_TEST_MYSQL_DSN`.
"""
from __future__ import annotations

import os
import threading

import pytest

from ctxdiff.models import Block, CallBlock, content_hash
from ctxdiff.store.mysql import MySQLStore
from ctxdiff.store.postgres import PostgresStore
from ctxdiff.store.sql import (
    BLOCK_TABLE,
    CALL_BLOCK_TABLE,
    CALL_TABLE,
    RUN_TABLE,
    SQLStore,
)
from ctxdiff.store.sqlite import SQLiteStore

from tests import fakedb

POSTGRES_DSN_ENV = "CTXDIFF_TEST_POSTGRES_DSN"
MYSQL_DSN_ENV = "CTXDIFF_TEST_MYSQL_DSN"

# The three timestamps the suite opens sessions with, named for the order they
# are INSERTED in — which is deliberately NOT their clock order.
#
# Sessions are ordered by insert order on every backend, and the suite is what
# holds that line. Hand-writing increasing timestamps (as this file used to)
# made every ordering assertion pass for the wrong reason: it hid that SQLite
# ordered by physical rowid while the network stores ordered by `started_at`,
# so the same three sessions came back in different orders depending on the
# backend. These values are DECREASING in wall-clock terms — the clock-skew
# case that is the flagship reason to point several containers at one shared
# database — so any backend that still sorts by timestamp returns them
# backwards and fails.
T_FIRST = "2026-07-25T12:00:00.000003+00:00"
T_SECOND = "2026-07-25T11:00:00.000002+00:00"
T_THIRD = "2026-07-25T10:00:00.000001+00:00"
# ...and a value SHARED by several sessions, for the other half of the same
# property: identical timestamps must still produce a deterministic order.
T_SAME = "2026-07-25T09:00:00+00:00"


# --- fixtures -----------------------------------------------------------------


def _drop_live_tables(backend) -> None:
    """Wipe ctxdiff's tables from a REAL server before a test, so a live run is
    as isolated as a tmp-file one. Dropped children-first because of the foreign
    keys; `IF EXISTS` makes the first-ever run a no-op."""
    conn = backend._connect()
    try:
        cur = conn.cursor()
        for table in (CALL_BLOCK_TABLE, CALL_TABLE, BLOCK_TABLE, RUN_TABLE):
            cur.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()
        cur.close()
    finally:
        conn.close()


def _live_backend(kind: str):
    """Build an EMPTY backend against a REAL server, or skip loudly naming the
    env var to set. Shared by the two live-capable fixtures so "live" means
    exactly one thing: a real server, wiped clean first."""
    env = POSTGRES_DSN_ENV if kind == "postgres-live" else MYSQL_DSN_ENV
    server = "PostgreSQL" if kind == "postgres-live" else "MySQL"
    dsn = os.environ.get(env)
    if not dsn:
        pytest.skip(f"set {env} to run this suite against a real {server} server")
    store = (PostgresStore(dsn=dsn) if kind == "postgres-live"
             else MySQLStore(dsn=dsn))
    _drop_live_tables(store)
    return store


@pytest.fixture(params=["sqlite", "postgres-stub", "mysql-stub",
                        "postgres-live", "mysql-live"])
def backend(request, tmp_path, monkeypatch):
    """Yield one configured, EMPTY backend per parameter. The two `-live`
    parameters skip loudly (with the env var to set) when no server is
    configured, so an unrun backend is visible in the report rather than
    silently passing."""
    kind = request.param
    if kind == "sqlite":
        return SQLiteStore(path=str(tmp_path / "conformance.ctrace"))
    if kind == "postgres-stub":
        fakedb.install(monkeypatch, "psycopg", str(tmp_path / "pg.sqlite"))
        return PostgresStore(dsn="postgresql://u:p@localhost:5432/ctxdiff")
    if kind == "mysql-stub":
        fakedb.install(monkeypatch, "pymysql", str(tmp_path / "my.sqlite"))
        return MySQLStore(dsn="mysql://u:p@localhost:3306/ctxdiff")
    return _live_backend(kind)


@pytest.fixture(params=["postgres-live", "mysql-live"])
def live_backend(request):
    """A REAL server only — no stub, no SQLite.

    Some guarantees are only meaningful against a server: a catalog race
    between processes creating the same table, a connection a DBA (or a pooler,
    or a restart) kills mid-run. The stub driver runs the adapters' real SQL,
    but it is a local SQLite file — it has no catalog to race on and no
    connection anyone can terminate — so tests that need those skip loudly
    rather than pretend."""
    return _live_backend(request.param)


# --- helpers ------------------------------------------------------------------


def _cb(text: str, position: int, role: str = "user", label: str = "user",
        label_source: str = "heuristic") -> CallBlock:
    """Build one CallBlock with a real content hash, so dedup is exercised by
    identical CONTENT rather than by a hand-written duplicate id."""
    block = Block(content_hash=content_hash(role, "message", text), role=role,
                  kind="message", text=text, token_count=len(text.split()),
                  token_method="estimate")
    return CallBlock(block=block, position=position, label=label,
                     label_source=label_source)


def _count_blocks(store) -> int:
    """How many DISTINCT block rows the store holds — the only assertion that
    cannot be phrased against the protocol (which deliberately exposes no
    'count blocks' method), so it reaches into each implementation's own
    connection. One tiny backend-aware helper beats widening the protocol with
    a method nothing in the product would call."""
    if isinstance(store, SQLStore):
        return store._query(f"SELECT COUNT(*) FROM {BLOCK_TABLE}")[0][0]
    return store._conn.execute("SELECT COUNT(*) FROM block").fetchone()[0]


# --- conformance --------------------------------------------------------------


def test_conformance_auto_creates_schema_and_appends_sessions(backend):
    """A brand-new store creates its own schema on first use (no migration
    step), and a SECOND session appends to the same store rather than replacing
    it — the two halves of "point ctxdiff at an empty database and it works"."""
    s1 = backend.open_session(project="p", provider="openai", started_at=T_FIRST)
    s1.close()
    s2 = backend.open_session(project="p", provider="openai", started_at=T_SECOND)
    try:
        sessions = s2.list_sessions()
    finally:
        s2.close()
    assert len(sessions) == 2
    assert [s.started_at for s in sessions] == [T_FIRST, T_SECOND]  # oldest first


def test_conformance_calls_are_isolated_per_session(backend):
    """Each session sees only its OWN calls. Same store, same tables, but
    `get_calls()` is bound to one session — the property multi-session project
    stores depend on."""
    s1 = backend.open_session(project="p", provider="openai", started_at=T_FIRST)
    s1.record_call(seq=1, params={"model": "gpt-4o"}, usage=None,
                   latency_ms=5, error=None, call_blocks=[_cb("one", 0)])
    s1_id = s1.get_run().id
    s1.close()

    s2 = backend.open_session(project="p", provider="openai", started_at=T_SECOND)
    s2.record_call(seq=1, params={"model": "gpt-4o"}, usage=None,
                   latency_ms=6, error=None, call_blocks=[_cb("two", 0)])
    s2.record_call(seq=2, params={"model": "gpt-4o"}, usage=None,
                   latency_ms=7, error=None, call_blocks=[_cb("three", 0)])
    try:
        assert len(s2.get_calls()) == 2
        assert len(s2.get_calls(session_id=s1_id)) == 1
        # ...and the other session's call is genuinely the other one.
        other = s2.get_calls(session_id=s1_id)[0]
        assert s2.get_call_blocks(other.id)[0].block.text == "one"
    finally:
        s2.close()


def test_conformance_blocks_dedup_by_content_hash(backend):
    """Identical content across calls is STORED ONCE and referenced twice —
    content-addressed dedup, the storage model's core invariant — while both
    calls still read back their full block list."""
    store = backend.open_session(project="p", provider="openai", started_at=T_FIRST)
    try:
        shared = "you are a helpful assistant"
        store.record_call(seq=1, params={}, usage=None, latency_ms=None,
                          error=None,
                          call_blocks=[_cb(shared, 0), _cb("first", 1)])
        store.record_call(seq=2, params={}, usage=None, latency_ms=None,
                          error=None,
                          call_blocks=[_cb(shared, 0), _cb("second", 1)])
        # 3 distinct texts across 4 memberships.
        assert _count_blocks(store) == 3
        calls = store.get_calls()
        first, second = store.get_call_blocks(calls[0].id), store.get_call_blocks(calls[1].id)
        assert len(first) == len(second) == 2
        assert first[0].block.content_hash == second[0].block.content_hash
        assert first[0].block.text == shared
    finally:
        store.close()


def test_conformance_membership_order_is_position_order(backend):
    """`get_call_blocks` returns a call's blocks in POSITION order, not
    insertion or hash order — the ordering every diff and token view assumes."""
    store = backend.open_session(project="p", provider="openai", started_at=T_FIRST)
    try:
        blocks = [_cb("sys", 0, role="system", label="system"),
                  _cb("hist", 1, role="assistant", label="history"),
                  _cb("q", 2)]
        # Hand them over shuffled: order must come from `position`, not input.
        store.record_call(seq=1, params={}, usage=None, latency_ms=None,
                          error=None, call_blocks=[blocks[2], blocks[0], blocks[1]])
        got = store.get_call_blocks(store.get_calls()[0].id)
        assert [cb.position for cb in got] == [0, 1, 2]
        assert [cb.block.text for cb in got] == ["sys", "hist", "q"]
        assert [cb.label for cb in got] == ["system", "history", "user"]
        assert [cb.label_source for cb in got] == ["heuristic"] * 3
    finally:
        store.close()


def test_conformance_call_fields_round_trip(backend):
    """Every stored call field survives the round trip byte-identically —
    params/usage JSON, latency, error, and the v2 attribution triple."""
    store = backend.open_session(project="p", provider="openai", started_at=T_FIRST)
    try:
        params = {"model": "gpt-4o", "temperature": 0.2,
                  "tools": [{"name": "search"}]}
        usage = {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}
        store.record_call(seq=7, params=params, usage=usage, latency_ms=123,
                          error="RateLimitError", call_blocks=[_cb("hi", 0)],
                          agent="planner", step="plan", provider="openai")
        call = store.get_calls()[0]
        assert call.seq == 7
        assert call.params == params
        assert call.usage == usage
        assert call.latency_ms == 123
        assert call.error == "RateLimitError"
        assert (call.agent, call.step, call.provider) == ("planner", "plan", "openai")
        # A call with no usage stores NULL, not an empty dict.
        store.record_call(seq=8, params={}, usage=None, latency_ms=None,
                          error=None, call_blocks=[])
        assert store.get_calls()[1].usage is None
    finally:
        store.close()


def test_conformance_long_free_text_values_are_stored_whole(backend):
    """Free-text fields hold whatever the host produced — a 400-character
    provider error, a long tagged label, a descriptive agent/step name, a real
    project slug — on EVERY backend.

    This is a whole-call guarantee, not a truncation one: the call and its
    blocks are written in ONE transaction, so a column too small to hold the
    value doesn't shorten a string, it raises (MySQL STRICT mode, error 1406)
    and loses the entire turn — on one backend only, which is precisely the
    divergence a conformance suite exists to catch."""
    long = "e" * 400
    store = backend.open_session(project=long, provider="openai",
                                 started_at=T_FIRST)
    try:
        # A latency wider than a 32-bit INT (~24 days in ms): stored, not
        # wrapped or rejected.
        store.record_call(seq=1, params={"model": long}, usage=None,
                          latency_ms=5_000_000_000, error=long,
                          call_blocks=[_cb("hi", 0, label=long,
                                           label_source=long)],
                          agent=long, step=long, provider=long)
        call = store.get_calls()[0]
        assert call.error == long
        assert (call.agent, call.step, call.provider) == (long, long, long)
        assert call.latency_ms == 5_000_000_000
        assert store.get_run().project == long
        blocks = store.get_call_blocks(call.id)
        assert blocks[0].label == long and blocks[0].label_source == long
    finally:
        store.close()


def test_conformance_content_hashes_are_compared_byte_exactly(backend):
    """Two blocks whose content hashes differ ONLY IN CASE are two DIFFERENT
    blocks, and each reads back its own text.

    Content-addressed storage is only sound if the key comparison is a BYTE
    comparison. Under a case-insensitive collation (MySQL's utf8mb4 default)
    the second insert collides with the first, the dedup upsert no-ops, and the
    second call silently reads back the FIRST block's text — corruption with no
    error anywhere. ctxdiff's own hashes are lowercase hex today, so this is a
    trap rather than a live bug; it is asserted because the storage model, not
    the current hash spelling, is the contract."""
    lower = Block(content_hash="abcdef0123456789", role="user", kind="message",
                  text="lower-cased hash", token_count=3,
                  token_method="estimate")
    upper = Block(content_hash="ABCDEF0123456789", role="user", kind="message",
                  text="UPPER-CASED HASH", token_count=3,
                  token_method="estimate")
    store = backend.open_session(project="p", provider="openai",
                                 started_at=T_FIRST)
    try:
        store.record_call(seq=1, params={}, usage=None, latency_ms=None,
                          error=None,
                          call_blocks=[CallBlock(block=lower, position=0,
                                                 label="user",
                                                 label_source="heuristic")])
        store.record_call(seq=2, params={}, usage=None, latency_ms=None,
                          error=None,
                          call_blocks=[CallBlock(block=upper, position=0,
                                                 label="user",
                                                 label_source="heuristic")])
        assert _count_blocks(store) == 2
        calls = store.get_calls()
        assert store.get_call_blocks(calls[0].id)[0].block.text == "lower-cased hash"
        assert store.get_call_blocks(calls[1].id)[0].block.text == "UPPER-CASED HASH"
    finally:
        store.close()


def test_conformance_list_sessions_reports_agents_and_turn_counts(backend):
    """`list_sessions()` summarizes each session with its distinct agents in
    FIRST-APPEARANCE order and its turn count — the session-picker query."""
    s1 = backend.open_session(project="p", provider="openai", started_at=T_FIRST)
    for seq, agent in ((1, "planner"), (2, "writer"), (3, "planner")):
        s1.record_call(seq=seq, params={}, usage=None, latency_ms=None,
                       error=None, call_blocks=[_cb(f"t{seq}", 0)], agent=agent)
    s1.close()

    s2 = backend.open_session(project="p", provider="openai", started_at=T_SECOND)
    s2.record_call(seq=1, params={}, usage=None, latency_ms=None, error=None,
                   call_blocks=[_cb("solo", 0)])
    try:
        first, second = s2.list_sessions()
        assert first.turn_count == 3
        assert first.agents == ["planner", "writer"]   # first-appearance order
        assert second.turn_count == 1
        assert second.agents == []                     # no agent attribution
        assert first.project == "p" and first.provider == "openai"
    finally:
        s2.close()


def test_conformance_reader_defaults_to_the_last_session_written(backend):
    """A reader opened with no session named binds to the session written LAST
    — so `ctxdiff diff` analyzes the run you just made, not the first one ever
    recorded into an accumulating project store.

    "Last" means last INSERTED, not largest timestamp. The three sessions here
    are written with DECREASING `started_at` values, the way two containers
    whose clocks disagree by a few seconds write into one shared database: a
    store that binds by timestamp hands back the FIRST session and the user
    analyzes somebody else's run."""
    for started, text in ((T_FIRST, "written-first"), (T_SECOND, "written-second"),
                          (T_THIRD, "written-last")):
        session = backend.open_session(project="p", provider="openai",
                                       started_at=started)
        session.record_call(seq=1, params={}, usage=None, latency_ms=None,
                            error=None, call_blocks=[_cb(text, 0)])
        session.close()

    reader = backend.open_reader()
    try:
        assert reader.get_run().started_at == T_THIRD
        blocks = reader.get_call_blocks(reader.get_calls()[0].id)
        assert blocks[0].block.text == "written-last"
    finally:
        reader.close()


def test_conformance_sessions_are_ordered_by_insert_order(backend):
    """`list_sessions()` returns sessions in the order they were WRITTEN, on
    every backend — even when their timestamps disagree with that order, and
    even when several share the exact same timestamp.

    Both halves of this used to diverge. With skewed clocks, SQLite (physical
    rowid) and the network stores (`ORDER BY started_at, id`) returned
    different orders for the same data. With IDENTICAL timestamps the network
    stores fell back to comparing random uuid4 hex — a coin flip that could
    reorder two sessions between one `ctxdiff runs` and the next. Insert order
    is the only tiebreak that is both stable and meaningful, and it is what
    SQLite's `rowid` gave the local store for free."""
    written = []
    for started in (T_FIRST, T_SAME, T_SECOND, T_SAME, T_THIRD):
        session = backend.open_session(project="p", provider="openai",
                                       started_at=started)
        written.append(session.get_run().id)
        session.close()

    reader = backend.open_reader()
    try:
        assert [s.id for s in reader.list_sessions()] == written
        # ...and it is STABLE: the same query twice gives the same order.
        assert [s.id for s in reader.list_sessions()] == written
        assert reader.get_run().id == written[-1]
    finally:
        reader.close()


def test_conformance_note_model_dedups_and_keeps_first_seen_order(backend):
    """The run's `models` list is backfilled from real call params, in
    first-seen order, with repeats deduped and blanks ignored — the store-level
    behavior that stops `run.models` being `['']` or a pile of duplicates."""
    store = backend.open_session(project="p", provider="openai", started_at=T_FIRST)
    try:
        assert store.get_run().models == []          # not seeded with a blank
        store.note_model("gpt-4o")
        store.note_model("gpt-4o")                   # repeat: no-op
        store.note_model(None)                       # ignored
        store.note_model("")                         # ignored
        store.note_model("claude-sonnet-4")
        assert store.get_run().models == ["gpt-4o", "claude-sonnet-4"]
        # record_call rolls the model up on its own, incl. bedrock's 'modelId'.
        store.record_call(seq=1, params={"modelId": "anthropic.claude-v2"},
                          usage=None, latency_ms=None, error=None,
                          call_blocks=[_cb("x", 0)])
        assert store.get_run().models == [
            "gpt-4o", "claude-sonnet-4", "anthropic.claude-v2"]
    finally:
        store.close()


def test_conformance_get_run_selects_by_session_id(backend):
    """`get_run(session_id=...)` reads ANY session's metadata from a handle
    bound to another one — how a session picker renders a list it did not
    open."""
    s1 = backend.open_session(project="alpha", provider="openai", started_at=T_FIRST)
    s1_id = s1.get_run().id
    s1.close()
    s2 = backend.open_session(project="beta", provider="anthropic", started_at=T_SECOND)
    try:
        assert s2.get_run().project == "beta"
        assert s2.get_run(session_id=s1_id).project == "alpha"
        assert s2.get_run(session_id=s1_id).provider == "openai"
    finally:
        s2.close()


def test_conformance_close_is_safe_to_call_twice(backend):
    """`close()` never raises — it runs on the writer thread's way out and in
    the CLI's `finally`, where an exception would be worse than a leaked
    socket."""
    store = backend.open_session(project="p", provider="openai", started_at=T_FIRST)
    store.close()
    store.close()


# --- live-server-only: races and failures a stub driver cannot stage ----------


def _kill_other_connections(backend) -> int:
    """Terminate every connection to the test database EXCEPT this one, the way
    a pooler recycle, a `pg_terminate_backend` from a DBA, or a server restart
    does. Returns how many were killed, so a test can assert the kill actually
    happened rather than passing because nothing was disturbed.

    Opened through the backend's own `_connect()`, so it kills using the same
    credentials and database the store under test is using."""
    conn = backend._connect()
    try:
        cur = conn.cursor()
        if backend.dialect.name == "postgres":
            cur.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = current_database() AND pid <> pg_backend_pid()")
            killed = len(cur.fetchall())
        else:
            cur.execute("SELECT id FROM information_schema.processlist "
                        "WHERE db = DATABASE() AND id <> CONNECTION_ID()")
            victims = [row[0] for row in cur.fetchall()]
            for victim in victims:
                try:
                    cur.execute(f"KILL {int(victim)}")
                except Exception:  # noqa: BLE001 — it may have gone on its own
                    pass
            killed = len(victims)
        conn.commit()
        cur.close()
        return killed
    finally:
        conn.close()


def test_live_capture_survives_the_connection_being_killed(live_backend):
    """A connection killed MID-RUN costs at most the in-flight write, never the
    rest of the run.

    A PgBouncer recycle, a failover, a `KILL`, a DBA's `pg_terminate_backend`,
    a database restart during a deploy — all of them drop a live connection
    under a long-running agent. Classifying that as "transient" and retrying on
    the SAME dead connection is inert: every retry fails identically, the store
    warns once, and capture is over for the process even though the database is
    healthy again seconds later. The retry has to REOPEN."""
    store = live_backend.open_session(project="p", provider="openai",
                                      started_at=T_FIRST)
    try:
        store.record_call(seq=1, params={"model": "gpt-4o"}, usage=None,
                          latency_ms=1, error=None, call_blocks=[_cb("before", 0)])

        assert _kill_other_connections(live_backend) >= 1   # the store's own

        for seq in (2, 3, 4):
            store.record_call(seq=seq, params={"model": "gpt-4o"}, usage=None,
                              latency_ms=seq, error=None,
                              call_blocks=[_cb(f"after-{seq}", 0)])

        calls = store.get_calls()
        assert [c.seq for c in calls] == [1, 2, 3, 4]
        texts = [store.get_call_blocks(c.id)[0].block.text for c in calls]
        assert texts == ["before", "after-2", "after-3", "after-4"]
        # The roll-up onto the run row survived the reconnect too.
        assert store.get_run().models == ["gpt-4o"]
    finally:
        store.close()


def test_live_concurrent_cold_start_records_every_session(live_backend):
    """Eight agents starting AT ONCE against an empty database all record.

    This is the cold-start shape of a real deployment: a service scales up and
    every replica's first LLM call opens its session within the same
    millisecond, against a database where ctxdiff's tables do not exist yet.
    `CREATE TABLE IF NOT EXISTS` does not make that safe on Postgres — two
    backends that pass the existence check together both insert into the system
    catalogs, and the losers die with a duplicate-key error on
    `pg_type_typname_nsp_index` (SQLSTATE 23505). Un-retried, 7 of 8 replicas
    silently record NOTHING for their whole run, once, on the deployment where
    it matters most.

    Threads released from a common barrier so the collision window is actually
    hit; each worker then writes a call, so the assertion covers the whole
    open-and-write path rather than just the DDL."""
    n = 8
    barrier = threading.Barrier(n)
    errors: list[str] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        try:
            barrier.wait(30)
            store = live_backend.open_session(project="cold", provider="openai",
                                              started_at=T_SAME)
            try:
                store.record_call(seq=1, params={"model": "gpt-4o"}, usage=None,
                                  latency_ms=1, error=None,
                                  call_blocks=[_cb(f"worker-{index}", 0)])
            finally:
                store.close()
        except Exception as exc:  # noqa: BLE001 — the capture loss under test
            with lock:
                errors.append(repr(exc))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)

    assert errors == []
    reader = live_backend.open_reader()
    try:
        sessions = reader.list_sessions()
        assert len(sessions) == n
        assert sum(s.turn_count for s in sessions) == n
    finally:
        reader.close()
