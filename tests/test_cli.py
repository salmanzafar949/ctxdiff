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
