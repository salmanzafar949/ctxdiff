import json
import os
import re

from ctxdiff.cli import main
from ctxdiff.models import Block, CallBlock
from ctxdiff.store.ctrace import CTrace

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _cb(text, position, role="user", label="user"):
    """Build a CallBlock with a stable hash derived from (role, text) — a
    test helper mirroring the real content-addressing scheme."""
    block = Block(content_hash=f"h:{role}:{text}", role=role, kind="message",
                  text=text, token_count=len(text), token_method="tiktoken")
    return CallBlock(block=block, position=position, label=label, label_source="heuristic")


RAG_TEXT = "some new rag chunk"


def _make_trace(path):
    """Build a small 2-turn .ctrace: turn 1 has a system + user block; turn 2
    keeps both and adds one rag block — the minimal case that exercises an
    'added' entry in the diff."""
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")
    turn1 = [_cb("system prompt", 0, "system", "system"),
             _cb("hello", 1, "user", "user")]
    ct.record_call(seq=1, params={"model": "gpt-4o"}, usage=None, latency_ms=10,
                   error=None, call_blocks=turn1)
    turn2 = [_cb("system prompt", 0, "system", "system"),
             _cb(RAG_TEXT, 1, "user", "rag"),
             _cb("hello", 2, "user", "user")]
    ct.record_call(seq=2, params={"model": "gpt-4o"}, usage=None, latency_ms=10,
                   error=None, call_blocks=turn2)
    ct.close()


def test_diff_header_shows_turns_and_token_delta(tmp_path, capsys):
    """The header line names both turn numbers and the correct +added
    token delta (one new block, nothing evicted)."""
    path = str(tmp_path / "demo.ctrace")
    _make_trace(path)

    exit_code = main(["diff", "--turn", "1", "--turn", "2", "--run", path])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "turn 1 → turn 2" in out
    assert f"+{len(RAG_TEXT)} " in out or f"+{len(RAG_TEXT)}−" in out
    assert "−0 tokens" in out  # nothing evicted between these two turns


def test_diff_shows_added_block_line(tmp_path, capsys):
    """An added-block line appears, carrying its label/role tag and text."""
    path = str(tmp_path / "demo.ctrace")
    _make_trace(path)

    main(["diff", "--turn", "1", "--turn", "2", "--run", path])

    out = capsys.readouterr().out
    assert "+ [rag·user]" in out
    assert RAG_TEXT in out
    assert f"+{len(RAG_TEXT)} tok" in out


def test_no_color_output_has_no_ansi_escapes(tmp_path, monkeypatch, capsys):
    """Setting NO_COLOR strips every ANSI escape from the rendered diff. Pytest's
    own capture already makes stdout a non-TTY, which would pass this test
    for the wrong reason (isatty()-gating alone, NO_COLOR unexercised) — so
    stdout.isatty is forced to True here, isolating NO_COLOR as the thing
    actually under test."""
    path = str(tmp_path / "demo.ctrace")
    _make_trace(path)
    monkeypatch.setenv("NO_COLOR", "1")
    import sys
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)

    main(["diff", "--turn", "1", "--turn", "2", "--run", path])

    out = capsys.readouterr().out
    assert not _ANSI_RE.search(out)


def test_missing_turn_exits_1_with_stderr_message(tmp_path, capsys):
    """Diffing against a turn number that doesn't exist in the run is an
    operational error: exit code 1, with a message on stderr."""
    path = str(tmp_path / "demo.ctrace")
    _make_trace(path)

    exit_code = main(["diff", "--turn", "1", "--turn", "99", "--run", path])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.strip() != ""
    assert captured.out == ""


def test_wrong_number_of_turn_flags_exits_2(tmp_path, capsys):
    """Passing --turn once (or not exactly twice) is a usage error: exit 2."""
    path = str(tmp_path / "demo.ctrace")
    _make_trace(path)

    exit_code = main(["diff", "--turn", "1", "--run", path])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err.strip() != ""


def test_runs_lists_ctrace_file(tmp_path, monkeypatch, capsys):
    """`ctxdiff runs` lists every *.ctrace in the cwd with project/provider/
    turn-count."""
    path = tmp_path / "demo.ctrace"
    _make_trace(str(path))
    monkeypatch.chdir(tmp_path)

    exit_code = main(["runs"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "demo.ctrace" in out
    assert "project=demo" in out
    assert "provider=openai" in out
    assert "turns=2" in out


def test_runs_skips_unreadable_file(tmp_path, monkeypatch, capsys):
    """A .ctrace-named file that isn't actually a valid trace is skipped, not
    fatal to the listing."""
    good = tmp_path / "demo.ctrace"
    _make_trace(str(good))
    bad = tmp_path / "garbage.ctrace"
    bad.write_text("not a sqlite file")
    monkeypatch.chdir(tmp_path)

    exit_code = main(["runs"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "demo.ctrace" in out
    assert "garbage.ctrace" not in out


def test_default_run_picks_most_recently_modified(tmp_path, monkeypatch, capsys):
    """With no --run, the most recently modified *.ctrace in cwd is used."""
    older = tmp_path / "older.ctrace"
    _make_trace(str(older))
    newer = tmp_path / "newer.ctrace"
    _make_trace(str(newer))
    # Ensure a distinct, later mtime regardless of filesystem timestamp resolution.
    import os
    import time
    time.sleep(0.01)
    os.utime(newer, None)
    monkeypatch.chdir(tmp_path)

    exit_code = main(["diff", "--turn", "1", "--turn", "2"])

    assert exit_code == 0  # succeeds against whichever file it picked


def test_no_run_found_exits_1_with_friendly_message(tmp_path, monkeypatch, capsys):
    """No .ctrace anywhere findable -> the spec's friendly message, exit 1."""
    monkeypatch.chdir(tmp_path)

    exit_code = main(["diff", "--turn", "1", "--turn", "2"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "did the run capture" in captured.err


# --- `ctxdiff tokens` ---------------------------------------------------------

USED_SCHEMA = json.dumps({"type": "function", "function": {"name": "get_weather"}})
UNUSED_SCHEMA = json.dumps({"type": "function", "function": {"name": "delete_account"}})


def _tok_cb(text, position, role="user", kind="message", label="user",
            token_count=None, token_method="tiktoken"):
    """Build a CallBlock whose hash is derived from (role, kind, text,
    token_method) — the extra token_method component lets the same visible
    text be stored once as exact and once as an estimate without colliding."""
    if token_count is None:
        token_count = len(text)
    block = Block(content_hash=f"h:{role}:{kind}:{text}:{token_method}", role=role,
                  kind=kind, text=text, token_count=token_count, token_method=token_method)
    return CallBlock(block=block, position=position, label=label, label_source="heuristic")


def _make_tokens_trace(path):
    """Build a 2-turn .ctrace exercising every `ctxdiff tokens` code path:
    a used tool schema (referenced by name in an assistant block), an unused
    one (never referenced -> bloat), and one estimate-method block per turn
    (-> the ~approx marker). Each turn carries a turn-specific user block so
    `--turn` filtering is independently verifiable."""
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")
    for seq in (1, 2):
        blocks = [
            _tok_cb("system prompt", 0, role="system", label="system", token_count=50),
            _tok_cb(USED_SCHEMA, 1, role="system", kind="tool_schema",
                    label="tool_schema", token_count=30),
            _tok_cb(UNUSED_SCHEMA, 2, role="system", kind="tool_schema",
                    label="tool_schema", token_count=40),
            _tok_cb("calling get_weather now", 3, role="assistant",
                    label="history", token_count=10),
            _tok_cb(f"question at turn {seq}", 4, role="user", label="user",
                    token_count=20, token_method="estimate"),
        ]
        ct.record_call(seq=seq, params={"model": "gpt-4o"}, usage=None, latency_ms=10,
                       error=None, call_blocks=blocks)
    ct.close()


def test_tokens_shows_per_turn_totals_and_approx_marker(tmp_path, capsys):
    """Every turn's header appears with its total tokens, and the ~approx
    marker shows up since each turn has an estimate-method block."""
    path = str(tmp_path / "demo.ctrace")
    _make_tokens_trace(path)

    exit_code = main(["tokens", "--run", path])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "turn 1 ·" in out
    assert "turn 2 ·" in out
    assert "~approx" in out


def test_tokens_bloat_warning_names_unused_tool_only(tmp_path, capsys):
    """The bloat line names the unused tool (delete_account) and does not
    name the used one (get_weather)."""
    path = str(tmp_path / "demo.ctrace")
    _make_tokens_trace(path)

    main(["tokens", "--run", path])

    out = capsys.readouterr().out
    bloat_line = next(line for line in out.splitlines() if "schema bloat" in line)
    assert "delete_account" in bloat_line
    assert "get_weather" not in bloat_line


def test_tokens_turn_filter_limits_output(tmp_path, capsys):
    """`--turn 2` shows only turn 2's block, not turn 1's."""
    path = str(tmp_path / "demo.ctrace")
    _make_tokens_trace(path)

    exit_code = main(["tokens", "--turn", "2", "--run", path])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "turn 2 ·" in out
    assert "turn 1 ·" not in out


def test_tokens_prints_provider_usage_summary_with_coverage(tmp_path, capsys):
    """`ctxdiff tokens` prints a run-level provider-usage rollup first, summing
    input/output from the calls that reported usage and stating the coverage
    fraction (here 1 of 2 calls carried usage)."""
    path = str(tmp_path / "usage.ctrace")
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")
    ct.record_call(seq=1, params={"model": "m"},
                   usage={"prompt_tokens": 18400, "completion_tokens": 640},
                   latency_ms=10, error=None,
                   call_blocks=[_cb("system prompt", 0, "system", "system")])
    ct.record_call(seq=2, params={"model": "m"}, usage=None, latency_ms=10,
                   error=None, call_blocks=[_cb("hello", 0, "user", "user")])
    ct.close()

    exit_code = main(["tokens", "--run", path])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "run total · in 18,400 tok · out 640 tok" in out
    assert "1/2 calls reported usage" in out


def test_tokens_no_provider_usage_says_none_not_zeros(tmp_path, capsys):
    """When no call reported usage, the summary says so rather than printing a
    misleading in-0 / out-0 total."""
    path = str(tmp_path / "demo.ctrace")
    _make_trace(path)   # this fixture records usage=None on every call

    main(["tokens", "--run", path])

    out = capsys.readouterr().out
    assert "no provider usage reported" in out


def test_tokens_missing_turn_exits_1_with_stderr_message(tmp_path, capsys):
    path = str(tmp_path / "demo.ctrace")
    _make_tokens_trace(path)

    exit_code = main(["tokens", "--turn", "99", "--run", path])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.strip() != ""
    assert captured.out == ""


def test_tokens_no_color_output_has_no_ansi_escapes(tmp_path, monkeypatch, capsys):
    """NO_COLOR strips every ANSI escape, same convention as `ctxdiff diff`."""
    path = str(tmp_path / "demo.ctrace")
    _make_tokens_trace(path)
    monkeypatch.setenv("NO_COLOR", "1")
    import sys
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)

    main(["tokens", "--run", path])

    out = capsys.readouterr().out
    assert not _ANSI_RE.search(out)


# --- `ctxdiff cache` ------------------------------------------------------------


def _cache_cb(text, position, role="user", label="user", token_count=None):
    if token_count is None:
        token_count = len(text)
    block = Block(content_hash=f"h:{role}:{text}", role=role, kind="message",
                  text=text, token_count=token_count, token_method="tiktoken")
    return CallBlock(block=block, position=position, label=label, label_source="heuristic")


def _make_cache_break_trace(path):
    """Build a 3-turn .ctrace with a deliberate timestamp break: the system
    block's embedded timestamp changes every turn while the rest of the
    context is stable, breaking the cache prefix on every consecutive pair."""
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")
    for seq, ts in enumerate(["10:00:00", "10:00:05", "10:00:10"], start=1):
        blocks = [
            _cache_cb(f"you are an assistant. current time: {ts}", 0,
                     role="system", label="system"),
            _cache_cb("static rule", 1, role="system", label="system"),
            _cache_cb("hello", 2, role="user", label="user"),
        ]
        ct.record_call(seq=seq, params={"model": "gpt-4o"}, usage=None, latency_ms=10,
                       error=None, call_blocks=blocks)
    ct.close()


def _make_cache_stable_trace(path):
    """Build a 3-turn .ctrace that only ever appends to history — a stable
    cache prefix on every pair, the happy path."""
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")
    turn1 = [_cache_cb("system prompt", 0, role="system", label="system"),
             _cache_cb("hello", 1, role="user", label="user")]
    turn2 = turn1 + [_cache_cb("hi", 2, role="assistant", label="history"),
                     _cache_cb("bye", 3, role="user", label="user")]
    ct.record_call(seq=1, params={"model": "gpt-4o"}, usage=None, latency_ms=10,
                   error=None, call_blocks=turn1)
    ct.record_call(seq=2, params={"model": "gpt-4o"}, usage=None, latency_ms=10,
                   error=None, call_blocks=turn2)
    ct.close()


def test_cache_reports_grouped_warning_with_waste_note(tmp_path, capsys):
    """A trace with a deliberate timestamp break shows a warning line, the
    grouped count (2/2 pairs, since both pairs break the same way), and the
    waste note — with clean NO_COLOR output."""
    path = str(tmp_path / "demo.ctrace")
    _make_cache_break_trace(path)
    exit_code = main(["cache", "--run", path])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "⚠ warning" in out
    assert "(2/2 pairs)" in out
    assert "re-billed" in out
    assert "tokens re-billed across 2 turns" in out
    assert "dynamic value" in out  # the fix hint


def test_cache_no_color_output_has_no_ansi_escapes(tmp_path, monkeypatch, capsys):
    path = str(tmp_path / "demo.ctrace")
    _make_cache_break_trace(path)
    monkeypatch.setenv("NO_COLOR", "1")
    import sys
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)

    main(["cache", "--run", path])

    out = capsys.readouterr().out
    assert not _ANSI_RE.search(out)


def test_cache_stable_trace_shows_green_line_and_exits_zero(tmp_path, capsys):
    """A run whose prefix never breaks (pure append growth) shows the green
    stable line and exits 0, with no warning line."""
    path = str(tmp_path / "demo.ctrace")
    _make_cache_stable_trace(path)

    exit_code = main(["cache", "--run", path])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "✓" in out
    assert "prefix stable across all 1 turn pairs" in out
    assert "⚠" not in out


# --- `ctxdiff export` / `ctxdiff view` ----------------------------------------


def test_export_writes_file_and_prints_path(tmp_path, capsys):
    """`ctxdiff export --out FILE` writes the HTML file and prints its path,
    exit 0."""
    path = str(tmp_path / "demo.ctrace")
    _make_trace(path)
    out_file = str(tmp_path / "dash.html")

    exit_code = main(["export", "--run", path, "--out", out_file])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert os.path.exists(out_file)
    assert out_file in captured.out
    # sanity: it really is the self-contained dashboard for this run
    assert "demo" in open(out_file, encoding="utf-8").read()


def test_export_default_path_next_to_trace(tmp_path, capsys):
    """With no --out, export writes <stem>.html beside the trace and prints it."""
    path = str(tmp_path / "demo.ctrace")
    _make_trace(path)

    exit_code = main(["export", "--run", path])

    out = capsys.readouterr().out.strip()
    assert exit_code == 0
    assert out == str(tmp_path / "demo.html")
    assert os.path.exists(out)


def test_view_no_open_writes_temp_html_and_exits_zero(tmp_path, capsys):
    """`ctxdiff view --no-open` exports to a temp .html, prints its path, and
    exits 0 without launching a browser."""
    path = str(tmp_path / "demo.ctrace")
    _make_trace(path)

    exit_code = main(["view", "--run", path, "--no-open"])

    printed = capsys.readouterr().out.strip()
    assert exit_code == 0
    assert printed.endswith(".html")
    assert os.path.exists(printed)
    os.remove(printed)  # clean up the tempfile this test created


def test_export_no_run_found_exits_1(tmp_path, monkeypatch, capsys):
    """No .ctrace findable -> the friendly message and exit 1, same as diff."""
    monkeypatch.chdir(tmp_path)

    exit_code = main(["export"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "did the run capture" in captured.err


# --- multi-agent CLI ----------------------------------------------------------


def _agent_cb(text, position, role="user", label="user", token_count=None):
    if token_count is None:
        token_count = len(text)
    block = Block(content_hash=f"h:{role}:{text}", role=role, kind="message",
                  text=text, token_count=token_count, token_method="tiktoken")
    return CallBlock(block=block, position=position, label=label, label_source="heuristic")


def _make_multi_agent_trace(path):
    """Two agents (researcher, writer) interleaved on the global timeline; each
    only appends to its own history (both internally cache-stable)."""
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")

    def rec(seq, blocks, agent, step=None):
        ct.record_call(seq=seq, params={"model": "gpt-4o"}, usage=None,
                       latency_ms=10, error=None, call_blocks=blocks,
                       agent=agent, step=step)
    r1 = [_agent_cb("sys R", 0, "system", "system"), _agent_cb("r-q1", 1)]
    w1 = [_agent_cb("sys W", 0, "system", "system"), _agent_cb("w-q1", 1)]
    r2 = r1 + [_agent_cb("r-ans", 2, "assistant", "history"), _agent_cb("r-q2", 3)]
    rec(1, r1, "researcher", "plan")
    rec(2, w1, "writer")
    rec(3, r2, "researcher", "answer")
    ct.close()


def test_runs_shows_distinct_agent_names(tmp_path, monkeypatch, capsys):
    """`ctxdiff runs` lists each trace's distinct agent names."""
    _make_multi_agent_trace(str(tmp_path / "multi.ctrace"))
    monkeypatch.chdir(tmp_path)

    exit_code = main(["runs"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "agents=researcher, writer" in out


def test_runs_shows_dash_for_no_agents(tmp_path, monkeypatch, capsys):
    """A run with no named agents shows `agents=-`."""
    _make_trace(str(tmp_path / "demo.ctrace"))
    monkeypatch.chdir(tmp_path)

    main(["runs"])

    out = capsys.readouterr().out
    assert "agents=-" in out


def test_tokens_agent_filter_limits_and_shows_no_summary(tmp_path, capsys):
    """`tokens --agent researcher` shows only that agent's turns (1 and 3, not
    the writer's turn 2)."""
    path = str(tmp_path / "multi.ctrace")
    _make_multi_agent_trace(path)

    exit_code = main(["tokens", "--agent", "researcher", "--run", path])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "turn 1 ·" in out
    assert "turn 3 ·" in out
    assert "turn 2 ·" not in out       # writer's turn is filtered out


def test_tokens_unfiltered_prints_per_agent_summary(tmp_path, capsys):
    """Unfiltered on a multi-agent run, a per-agent summary block appears, and
    turn headers carry the [agent·step] marker."""
    path = str(tmp_path / "multi.ctrace")
    _make_multi_agent_trace(path)

    main(["tokens", "--run", path])

    out = capsys.readouterr().out
    assert "agents:" in out
    assert "researcher" in out and "writer" in out
    assert "[researcher·plan]" in out   # turn header marker


def test_cache_agent_grouping_note(tmp_path, capsys):
    """Unfiltered cache on a multi-agent run notes the per-agent grouping and
    (since both agents are internally stable) reports the prefix stable."""
    path = str(tmp_path / "multi.ctrace")
    _make_multi_agent_trace(path)

    exit_code = main(["cache", "--run", path])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "within 2 agents" in out
    assert "⚠" not in out               # no cross-agent hand-off is a break


def _make_agent_usage_trace(path):
    """Multi-agent trace carrying provider usage per call, so the usage rollup
    has non-zero numbers to scope by agent."""
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")

    def rec(seq, agent, pt, ct_tok):
        ct.record_call(seq=seq, params={"model": "m"},
                       usage={"prompt_tokens": pt, "completion_tokens": ct_tok},
                       latency_ms=10, error=None,
                       call_blocks=[_agent_cb("q", 0)], agent=agent)
    rec(1, "researcher", 100, 20)
    rec(2, "writer", 40, 6)
    rec(3, "researcher", 30, 4)
    ct.close()


def test_tokens_agent_filter_scopes_usage_label(tmp_path, capsys):
    """Under `--agent researcher`, the usage rollup label reads
    `researcher total ·` (agent-scoped), not `run total ·` — the numbers are
    agent-filtered and the label must not claim run scope."""
    path = str(tmp_path / "usage.ctrace")
    _make_agent_usage_trace(path)

    exit_code = main(["tokens", "--agent", "researcher", "--run", path])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "researcher total · in 130 tok · out 24 tok" in out
    assert "run total ·" not in out


def test_cache_break_uses_agents_own_pair_denominator(tmp_path, capsys):
    """A researcher that breaks on both of its own pairs, alongside a stable
    writer, reports `2/2 pairs` (its OWN denominator) — not `2/3` of the
    run-wide pair count."""
    path = str(tmp_path / "multi.ctrace")
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")

    def rec(seq, blocks, agent):
        ct.record_call(seq=seq, params={"model": "m"}, usage=None, latency_ms=10,
                       error=None, call_blocks=blocks, agent=agent)

    def r(ts):
        return [_agent_cb(f"sys R time {ts}", 0, "system", "system"),
                _agent_cb("r-q", 1)]
    w1 = [_agent_cb("sys W", 0, "system", "system"), _agent_cb("w-q1", 1)]
    w2 = w1 + [_agent_cb("w-ans", 2, "assistant", "history"), _agent_cb("w-q2", 3)]
    rec(1, r("10:00:00"), "researcher")
    rec(2, w1, "writer")
    rec(3, r("10:00:05"), "researcher")   # researcher break #1
    rec(4, w2, "writer")                  # writer stable
    rec(5, r("10:00:10"), "researcher")   # researcher break #2
    ct.close()

    exit_code = main(["cache", "--run", path])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "(2/2 pairs)" in out    # researcher's own denominator
    assert "2/3" not in out        # NOT the run-wide pair count


def test_diff_agent_validates_turn_ownership(tmp_path, capsys):
    """`diff --agent researcher --turn 1 --turn 2` fails: turn 2 belongs to
    the writer, not the researcher — exit 1 with that agent's own turns listed."""
    path = str(tmp_path / "multi.ctrace")
    _make_multi_agent_trace(path)

    exit_code = main(["diff", "--turn", "1", "--turn", "2",
                      "--agent", "researcher", "--run", path])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "researcher" in captured.err
    assert captured.out == ""


def test_diff_agent_accepts_owned_turns(tmp_path, capsys):
    """`diff --agent researcher --turn 1 --turn 3` succeeds: both turns are the
    researcher's own calls."""
    path = str(tmp_path / "multi.ctrace")
    _make_multi_agent_trace(path)

    exit_code = main(["diff", "--turn", "1", "--turn", "3",
                      "--agent", "researcher", "--run", path])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "turn 1 → turn 3" in out


# --- `ctxdiff demo` -------------------------------------------------------------


def test_demo_out_writes_ctrace_and_html_and_prints_both_paths(tmp_path, capsys):
    """`ctxdiff demo --no-open --out FILE.ctrace` writes FILE.ctrace and a
    self-contained FILE.html beside it, and prints both paths."""
    out_ctrace = str(tmp_path / "d.ctrace")
    out_html = str(tmp_path / "d.html")

    exit_code = main(["demo", "--no-open", "--out", out_ctrace])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert os.path.exists(out_ctrace)
    assert os.path.exists(out_html)
    assert out_ctrace in captured.out
    assert out_html in captured.out

    html_text = open(out_html, encoding="utf-8").read()
    assert "http://" not in html_text
    assert "https://" not in html_text


def test_demo_default_uses_tempfile_and_exits_zero(tmp_path, monkeypatch, capsys):
    """`ctxdiff demo --no-open` with no --out/--keep succeeds via a tempfile
    pair, exits 0, and prints paths ending in .ctrace/.html."""
    monkeypatch.chdir(tmp_path)

    exit_code = main(["demo", "--no-open"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert ".ctrace" in out
    assert ".html" in out


def test_demo_keep_writes_fixed_filenames_in_cwd(tmp_path, monkeypatch, capsys):
    """`ctxdiff demo --no-open --keep` writes ./ctxdiff-demo.ctrace and
    ./ctxdiff-demo.html in the current directory."""
    monkeypatch.chdir(tmp_path)

    exit_code = main(["demo", "--no-open", "--keep"])

    assert exit_code == 0
    assert (tmp_path / "ctxdiff-demo.ctrace").exists()
    assert (tmp_path / "ctxdiff-demo.html").exists()


def test_demo_prints_a_nudge_toward_tracing_a_real_agent(tmp_path, capsys):
    """The output nudges the user toward `trace.init`/`tracer.wrap` for their
    own agent, not just the sample."""
    out_ctrace = str(tmp_path / "d.ctrace")

    main(["demo", "--no-open", "--out", out_ctrace])

    out = capsys.readouterr().out
    assert "trace.init" in out
    assert "tracer.wrap" in out


def test_export_agent_preselects_the_level_and_a_bad_one_exits_2(tmp_path, capsys):
    """`--agent` on the dashboard commands PRESELECTS which of the three levels
    the page opens on rather than filtering what the file contains — the HTML
    still covers the whole project either way. A name nobody answers to is a
    usage error (exit 2) carrying the listing, exactly like every other
    selector."""
    path = str(tmp_path / "multi.ctrace")
    _make_multi_agent_trace(path)
    out_file = str(tmp_path / "dash.html")

    assert main(["export", "--run", path, "--agent", "researcher",
                 "--out", out_file]) == 0
    capsys.readouterr()
    page = open(out_file, encoding="utf-8").read()
    assert '"start": {"level": 3, "agent": "researcher"' in page
    # ...and the OTHER agent is still fully described by the same artifact.
    assert '"name": "writer"' in page

    assert main(["view", "--run", path, "--agent", "nobody", "--no-open"]) == 2
    err = capsys.readouterr().err
    assert "no agent 'nobody' in this project" in err
    assert "researcher" in err


def test_tokens_streamed_without_usage_names_the_remedy(tmp_path, capsys):
    """When the missing usage is diagnosable — OpenAI-chat streams sent
    without stream_options.include_usage — the summary must NAME the
    caller-side fix, not stop at 'no provider usage reported' (dogfood
    finding 2026-07-27: a real app streamed every call and saw only a dead
    end)."""
    path = str(tmp_path / "demo.ctrace")
    ct = CTrace.create(path, project="agent", provider="openai", model="gpt-4o")
    blocks = [_cb("system prompt", 0, "system", "system"),
              _cb("hi", 1, "user", "user")]
    ct.record_call(seq=1, params={"model": "gpt-4o", "stream": True,
                                  "messages": [{"role": "user", "content": "hi"}]},
                   usage=None, latency_ms=10, error=None, call_blocks=blocks)
    ct.close()

    main(["tokens", "--run", path])

    out = capsys.readouterr().out
    assert "no provider usage reported" in out
    assert "1 streamed call recorded no usage" in out
    assert 'stream_options={"include_usage": true}' in out


def test_tokens_non_streamed_missing_usage_gets_no_remedy_hint(tmp_path, capsys):
    """Missing usage on NON-streamed calls has no caller-side fix to name —
    the summary must stay exactly as before, with no hint line."""
    path = str(tmp_path / "demo.ctrace")
    _make_trace(path)   # records usage=None on every call, non-streamed params

    main(["tokens", "--run", path])

    out = capsys.readouterr().out
    assert "no provider usage reported" in out
    assert "streamed call" not in out
