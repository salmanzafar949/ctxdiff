import pytest
from ctxdiff.models import Block, CallBlock
from ctxdiff.store.ctrace import CTrace, Run, Call


def _call_block(text, position, label="user", source="heuristic"):
    """Build a CallBlock with a Block whose hash is just the text (test helper)."""
    block = Block(
        content_hash=f"h-{text}", role="user", kind="message",
        text=text, token_count=len(text), token_method="tiktoken",
    )
    return CallBlock(block=block, position=position, label=label, label_source=source)


def test_create_writes_run_row(tmp_path):
    """create() persists a run row readable via get_run(). Passing a real
    model id at create time still seeds `models` with it directly (no call
    needed to populate it in this case)."""
    path = str(tmp_path / "r.ctrace")
    ct = CTrace.create(path, project="agent", provider="openai", model="gpt-4o")
    run = ct.get_run()
    assert run.project == "agent" and run.provider == "openai"
    assert run.models == ["gpt-4o"]
    ct.close()


def test_create_with_empty_model_starts_with_no_models(tmp_path):
    """create() with an empty/unknown model (the tracer's wrap()-time case —
    the model is only known once a call comes in) leaves `models` as an empty
    list, NOT `['']` — the bug this fix closes. An empty string and no model
    at all behave identically."""
    path = str(tmp_path / "r.ctrace")
    ct = CTrace.create(path, project="agent", provider="openai", model="")
    assert ct.get_run().models == []
    ct.close()


def test_note_model_dedups_preserves_order_and_ignores_empty(tmp_path):
    """note_model() appends new models in first-seen order, ignores a repeat
    of a model already recorded, and ignores None/"" so a call with no model
    param never pollutes the list with a blank entry."""
    path = str(tmp_path / "r.ctrace")
    ct = CTrace.create(path, project="agent", provider="openai", model="")
    ct.note_model("gpt-4o")
    ct.note_model("gpt-4o")       # repeat: ignored
    ct.note_model(None)           # ignored
    ct.note_model("")             # ignored
    ct.note_model("claude-sonnet-4-5")
    assert ct.get_run().models == ["gpt-4o", "claude-sonnet-4-5"]
    ct.close()


def test_record_call_rolls_up_distinct_models_from_params(tmp_path):
    """record_call() backfills run.models from each call's own params["model"]
    — two calls on different models roll up onto the run in call order, and a
    call whose model repeats an already-seen one doesn't duplicate it."""
    path = str(tmp_path / "r.ctrace")
    ct = CTrace.create(path, project="agent", provider="openai", model="")
    ct.record_call(seq=1, params={"model": "gpt-4o"}, usage=None,
                   latency_ms=1, error=None, call_blocks=[])
    ct.record_call(seq=2, params={"model": "gpt-4o"}, usage=None,
                   latency_ms=1, error=None, call_blocks=[])
    ct.record_call(seq=3, params={"model": "claude-sonnet-4-5"}, usage=None,
                   latency_ms=1, error=None, call_blocks=[])
    assert ct.get_run().models == ["gpt-4o", "claude-sonnet-4-5"]
    ct.close()


def test_record_call_and_read_back(tmp_path):
    """record_call() stores a call plus its ordered blocks; readers return them."""
    path = str(tmp_path / "r.ctrace")
    ct = CTrace.create(path, project="agent", provider="openai", model="gpt-4o")
    cbs = [_call_block("sys", 0, "system"), _call_block("hi", 1, "user")]
    call_id = ct.record_call(seq=1, params={"model": "gpt-4o"}, usage={"total_tokens": 5},
                             latency_ms=42, error=None, call_blocks=cbs)
    calls = ct.get_calls()
    assert len(calls) == 1 and calls[0].seq == 1 and calls[0].usage == {"total_tokens": 5}
    read = ct.get_call_blocks(call_id)
    assert [cb.position for cb in read] == [0, 1]
    assert [cb.block.text for cb in read] == ["sys", "hi"]
    assert read[0].label == "system"
    ct.close()


def test_block_dedup_across_calls(tmp_path):
    """A block with the same content_hash in two calls is stored once in `block`
    but referenced by both calls — the core storage-efficiency property."""
    path = str(tmp_path / "r.ctrace")
    ct = CTrace.create(path, project="agent", provider="openai", model="gpt-4o")
    stable = _call_block("system prompt", 0, "system")
    ct.record_call(1, {"model": "m"}, None, 1, None, [stable, _call_block("q1", 1)])
    ct.record_call(2, {"model": "m"}, None, 1, None, [stable, _call_block("q2", 1)])
    n_blocks = ct._conn.execute("SELECT COUNT(*) FROM block").fetchone()[0]
    assert n_blocks == 3  # 1 shared system + 2 distinct user blocks, not 4
    ct.close()


def test_open_rejects_wrong_schema_version(tmp_path):
    """A file whose schema_version is NEWER than supported raises a clear
    error, not a crash."""
    path = str(tmp_path / "r.ctrace")
    ct = CTrace.create(path, project="agent", provider="openai", model="gpt-4o")
    ct._conn.execute("UPDATE run SET schema_version = 999")
    ct._conn.commit(); ct.close()
    with pytest.raises(ValueError, match="schema version"):
        CTrace.open(path)


# --- v2 attribution: agent / step / provider ----------------------------------


def test_v2_roundtrip_agent_step_provider(tmp_path):
    """A v2 trace stores and reads back a call's agent/step/provider verbatim."""
    path = str(tmp_path / "r.ctrace")
    ct = CTrace.create(path, project="agent", provider="openai", model="gpt-4o")
    ct.record_call(seq=1, params={"model": "m"}, usage=None, latency_ms=1,
                   error=None, call_blocks=[_call_block("hi", 0)],
                   agent="researcher", step="retrieve", provider="openai")
    ct.close()

    ct = CTrace.open(path)
    call = ct.get_calls()[0]
    assert call.agent == "researcher"
    assert call.step == "retrieve"
    assert call.provider == "openai"
    ct.close()


def test_v2_attribution_defaults_to_none(tmp_path):
    """Recording a call without the v2 params leaves all three fields None."""
    path = str(tmp_path / "r.ctrace")
    ct = CTrace.create(path, project="agent", provider="openai", model="gpt-4o")
    ct.record_call(seq=1, params={"model": "m"}, usage=None, latency_ms=1,
                   error=None, call_blocks=[_call_block("hi", 0)])
    ct.close()
    ct = CTrace.open(path)
    call = ct.get_calls()[0]
    assert call.agent is None and call.step is None and call.provider is None
    ct.close()


# A verbatim copy of the v1 `call` DDL (no agent/step/provider columns), used to
# forge a REAL v1 file in-test so the v1-compatibility path is exercised against
# an actual on-disk v1 schema, not a mocked one.
_V1_DDL = """
CREATE TABLE run (
  id TEXT PRIMARY KEY, project TEXT NOT NULL, started_at TEXT NOT NULL,
  provider TEXT NOT NULL, models TEXT NOT NULL, ctxdiff_version TEXT NOT NULL,
  schema_version INTEGER NOT NULL);
CREATE TABLE call (
  id TEXT PRIMARY KEY, run_id TEXT NOT NULL, seq INTEGER NOT NULL,
  params TEXT NOT NULL, usage TEXT, latency_ms INTEGER, error TEXT,
  UNIQUE(run_id, seq));
CREATE TABLE block (
  content_hash TEXT PRIMARY KEY, role TEXT NOT NULL, kind TEXT NOT NULL,
  text TEXT NOT NULL, token_count INTEGER NOT NULL, token_method TEXT NOT NULL);
CREATE TABLE call_block (
  call_id TEXT NOT NULL, block_id TEXT NOT NULL, position INTEGER NOT NULL,
  label TEXT NOT NULL, label_source TEXT NOT NULL, PRIMARY KEY (call_id, position));
"""


def _build_v1_file(path):
    """Forge a real v1 .ctrace on disk: v1 DDL, schema_version=1, one run and
    one call written with the 7-column v1 call shape."""
    import sqlite3
    conn = sqlite3.connect(path)
    conn.executescript(_V1_DDL)
    conn.execute("INSERT INTO run VALUES ('run1','proj','2026-01-01','openai',"
                 "'[\"gpt-4o\"]','0.1.0',1)")
    conn.execute("INSERT INTO call VALUES ('call1','run1',1,'{\"model\":\"m\"}',"
                 "NULL,10,NULL)")
    conn.execute("INSERT INTO block VALUES ('h1','user','message','hi',2,'tiktoken')")
    conn.execute("INSERT INTO call_block VALUES ('call1','h1',0,'user','heuristic')")
    conn.commit(); conn.close()


def test_open_v1_file_surfaces_none_and_does_not_mutate(tmp_path):
    """Opening a REAL v1 file must (1) succeed, (2) surface the v2 fields as
    None, and (3) never rewrite the file — a debugger must not mutate the
    evidence it inspects, so the on-disk bytes are identical after open+read."""
    path = str(tmp_path / "v1.ctrace")
    _build_v1_file(path)
    before = open(path, "rb").read()

    ct = CTrace.open(path)
    calls = ct.get_calls()
    assert len(calls) == 1
    assert calls[0].seq == 1
    assert calls[0].agent is None
    assert calls[0].step is None
    assert calls[0].provider is None
    # blocks still read fine through the unchanged block/call_block tables
    assert ct.get_call_blocks(calls[0].id)[0].block.text == "hi"
    ct.close()

    after = open(path, "rb").read()
    assert before == after  # opening a v1 file left its bytes untouched


# --- hardening: parity fixes mirrored from the JS SDK -----------------------


def test_note_model_retries_after_a_failed_write(tmp_path, monkeypatch):
    """A model roll-up whose UPDATE FAILED must be retried by the next call that
    sees the same model. The old ordering marked the model seen BEFORE the write,
    so one exhausted retry budget left it permanently "already known" and
    `run.models` stayed [] for the life of the run — permanent, silent model
    loss. Here the first write is forced to fail; the second sighting must still
    persist it."""
    import sqlite3
    path = str(tmp_path / "m.ctrace")
    ct = CTrace.create(path, project="p", provider="openai", model="")

    seen = {"n": 0}
    real = CTrace._with_write_retry

    def flaky(fn):
        seen["n"] += 1
        if seen["n"] == 1:
            # Stands in for "the bounded retry budget was exhausted".
            raise sqlite3.OperationalError("database is locked")
        return real(fn)

    monkeypatch.setattr(CTrace, "_with_write_retry", staticmethod(flaky))
    with pytest.raises(sqlite3.OperationalError):
        ct.note_model("gpt-4o")
    monkeypatch.undo()

    ct.note_model("gpt-4o")  # second chance — must NOT be skipped as "seen"
    assert ct.get_run().models == ["gpt-4o"]
    ct.close()


def test_record_call_does_not_raise_when_model_rollup_fails(tmp_path, monkeypatch):
    """The model roll-up runs AFTER the call transaction has committed, so its
    failure must not escape `record_call` — otherwise the caller logs "failed to
    record call" for a call that WAS persisted."""
    path = str(tmp_path / "r.ctrace")
    ct = CTrace.create(path, project="p", provider="openai", model="")

    def boom(self, model):
        raise RuntimeError("roll-up boom")

    monkeypatch.setattr(CTrace, "note_model", boom)
    ct.record_call(seq=1, params={"model": "gpt-4o"}, usage=None, latency_ms=1,
                   error=None, call_blocks=[], agent="a")
    monkeypatch.undo()

    calls = ct.get_calls()
    assert len(calls) == 1 and calls[0].agent == "a"
    ct.close()


def test_append_into_v1_file_upgrades_and_preserves_attribution(tmp_path):
    """Appending a v2 session into a PHYSICALLY-v1 file must upgrade the `call`
    table in place. `CREATE TABLE IF NOT EXISTS` no-ops against the existing v1
    table, so without an explicit ALTER the new session was written in the
    7-column v1 shape — dropping agent/step/provider — while its run row was
    still stamped schema_version=2. Silent data loss, newly reachable now that
    the default path is a STABLE ./<project>.ctrace that can land on a
    pre-existing file."""
    path = str(tmp_path / "v1.ctrace")
    _build_v1_file(path)

    ct = CTrace.open_or_create_session(
        path, project="proj", provider="openai", model="",
        started_at="2026-07-25T00:00:00+00:00")
    ct.record_call(seq=1, params={"model": "gpt-4o"}, usage=None, latency_ms=5,
                   error=None, call_blocks=[], agent="planner", step="retrieve",
                   provider="openai")
    ct.close()

    r = CTrace.open(path)
    sessions = r.list_sessions()
    assert len(sessions) == 2  # the pre-existing v1 run plus the appended one
    appended = sessions[-1]
    calls = r.get_calls(appended.id)
    assert len(calls) == 1
    assert calls[0].agent == "planner"
    assert calls[0].step == "retrieve"
    assert calls[0].provider == "openai"
    assert appended.agents == ["planner"]
    # The pre-existing v1 call is untouched and still readable (NULL v2 fields).
    legacy = r.get_calls("run1")
    assert len(legacy) == 1 and legacy[0].agent is None
    r.close()
