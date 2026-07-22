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
    """create() persists a run row readable via get_run()."""
    path = str(tmp_path / "r.ctrace")
    ct = CTrace.create(path, project="agent", provider="openai", model="gpt-4o")
    run = ct.get_run()
    assert run.project == "agent" and run.provider == "openai"
    assert run.models == ["gpt-4o"]
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
    """A file whose schema_version differs raises a clear error, not a crash."""
    path = str(tmp_path / "r.ctrace")
    ct = CTrace.create(path, project="agent", provider="openai", model="gpt-4o")
    ct._conn.execute("UPDATE run SET schema_version = 999")
    ct._conn.commit(); ct.close()
    with pytest.raises(ValueError, match="schema version"):
        CTrace.open(path)
