"""Tests for `ctxdiff demo`'s trace builder: a genuinely-rich, deterministic
sample `.ctrace` built through the real public capture API (no network, no
provider SDK). Covers the builder's own contract (readable, rich, repeatable)
plus a smoke pass through each analyzer the dashboard relies on."""
from ctxdiff.analyze.cache import analyze_cache
from ctxdiff.analyze.differ import diff_turns
from ctxdiff.analyze.tokens import analyze_run
from ctxdiff.demo import build_demo_trace
from ctxdiff.store.ctrace import CTrace


def _open(path):
    """Build the demo trace at `path` and open it read/write, returning the
    handle. A thin helper so every test doesn't repeat the same two lines."""
    build_demo_trace(path)
    return CTrace.open(path)


# --- builder contract ---------------------------------------------------------


def test_builder_writes_a_readable_multi_call_multi_agent_trace(tmp_path):
    """The file CTrace.open can read back, with at least 5 calls spanning both
    named agents."""
    ct = _open(str(tmp_path / "demo.ctrace"))
    calls = ct.get_calls()
    ct.close()

    assert len(calls) >= 5
    agents = {c.agent for c in calls}
    assert agents == {"researcher", "writer"}


def test_builder_populates_run_models(tmp_path):
    """The demo's flagship header regression: run.models rolls up both real
    models the two agents actually called with — not `['']` — since every
    call in the builder goes through the real trace.wrap()/record_call()
    path (see build_demo_trace)."""
    ct = _open(str(tmp_path / "demo.ctrace"))
    models = ct.get_run().models
    ct.close()

    assert models  # non-empty: the bug this fix closes
    assert set(models) == {"gpt-4o", "claude-3-5-sonnet-20241022"}


def test_builder_includes_a_rag_tagged_block(tmp_path):
    """At least one call carries a block labeled 'rag' with a tagged source
    (not a role-based heuristic guess)."""
    ct = _open(str(tmp_path / "demo.ctrace"))
    calls = ct.get_calls()
    rag_blocks = [
        cb
        for c in calls
        for cb in ct.get_call_blocks(c.id)
        if cb.label == "rag"
    ]
    ct.close()

    assert len(rag_blocks) >= 1
    assert rag_blocks[0].label_source == "tagged"


def test_builder_includes_tool_schema_blocks(tmp_path):
    """At least one call registers a tool_schema block (the schema-bloat
    detector needs something to cross-reference)."""
    ct = _open(str(tmp_path / "demo.ctrace"))
    calls = ct.get_calls()
    schema_blocks = [
        cb
        for c in calls
        for cb in ct.get_call_blocks(c.id)
        if cb.label == "tool_schema"
    ]
    ct.close()

    assert len(schema_blocks) >= 1


def test_builder_reports_provider_usage_on_every_call(tmp_path):
    """Every call carries provider-reported usage (not None) — the fake
    clients' `.usage` objects were actually read by the adapters."""
    ct = _open(str(tmp_path / "demo.ctrace"))
    calls = ct.get_calls()
    ct.close()

    assert all(c.usage is not None for c in calls)
    assert all(c.usage for c in calls)  # non-empty dict, not just not-None


def test_builder_is_deterministic_across_two_builds(tmp_path):
    """Two independent builds produce logically identical traces: same seq
    count, same agent sequence, and the same set of block hashes per call —
    i.e. no hidden randomness anywhere in the scenario content."""
    ct1 = _open(str(tmp_path / "a.ctrace"))
    ct2 = _open(str(tmp_path / "b.ctrace"))

    calls1, calls2 = ct1.get_calls(), ct2.get_calls()
    assert [c.seq for c in calls1] == [c.seq for c in calls2]
    assert [c.agent for c in calls1] == [c.agent for c in calls2]

    hashes1 = [
        tuple(cb.block.content_hash for cb in ct1.get_call_blocks(c.id))
        for c in calls1
    ]
    hashes2 = [
        tuple(cb.block.content_hash for cb in ct2.get_call_blocks(c.id))
        for c in calls2
    ]
    ct1.close()
    ct2.close()
    assert hashes1 == hashes2


# --- analyzer smoke pass -------------------------------------------------------


def test_analyze_cache_flags_a_researcher_only_break(tmp_path):
    """`analyze_cache` reports at least one break attributed to 'researcher',
    and none attributed to 'writer' — the dynamic timestamp block only sits
    in the researcher's own turns."""
    ct = _open(str(tmp_path / "demo.ctrace"))
    report = analyze_cache(ct)
    ct.close()

    researcher_breaks = [b for b in report.breaks if b.agent == "researcher"]
    writer_breaks = [b for b in report.breaks if b.agent == "writer"]
    assert len(researcher_breaks) >= 1
    assert len(writer_breaks) == 0


def test_analyze_run_reports_schema_bloat_and_by_agent_usage(tmp_path):
    """`analyze_run` names the never-invoked tool as bloat (and not the used
    one), and its usage rollup breaks out both agents."""
    ct = _open(str(tmp_path / "demo.ctrace"))
    run_tokens = analyze_run(ct)
    ct.close()

    assert run_tokens.bloat is not None
    assert "delete_index" in run_tokens.bloat.unused_tools
    assert "search_web" not in run_tokens.bloat.unused_tools

    assert run_tokens.usage.by_agent is not None
    assert set(run_tokens.usage.by_agent) == {"researcher", "writer"}


def test_diff_turns_across_an_agent_handoff(tmp_path):
    """`diff_turns` computes cleanly across a researcher -> writer hand-off
    (turn 2 -> turn 3) without raising, despite the two calls sharing no
    common block content."""
    ct = _open(str(tmp_path / "demo.ctrace"))
    diff = diff_turns(ct, 2, 3)
    ct.close()

    assert diff.seq_old == 2
    assert diff.seq_new == 3
    assert len(diff.entries) > 0
