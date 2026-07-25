"""CLI-level tests for the SESSION/AGENT command surface.

What is pinned here: the two discovery commands (`sessions` with its hidden
`runs` alias, and `agents`), the ambiguity contract (several sessions and no
`--session` => exit 2 with the pickable listing), selector resolution by prefix,
and the two cross diffs — cross-SESSION (the regression case: same agent, two
runs) and cross-AGENT (two agents, one run) — including the scope header that
keeps `turn 3 → turn 3` from being ambiguous.
"""
import pytest

from ctxdiff.cli import main
from ctxdiff.models import Block, CallBlock
from ctxdiff.store.ctrace import CTrace


def _cb(text, position, role="user", label="user"):
    """A CallBlock with a stable hash derived from (role, text), mirroring the
    real content-addressing scheme."""
    block = Block(content_hash=f"h:{role}:{text}", role=role, kind="message",
                  text=text, token_count=len(text), token_method="tiktoken")
    return CallBlock(block=block, position=position, label=label,
                     label_source="heuristic")


def _write_project(path) -> tuple[str, str]:
    """Build the PROJECT fixture: one file, TWO sessions ("good" then "bad"),
    each with the same two agents on the same turn numbers and one block whose
    text differs between the runs. This is the regression shape the session CLI
    exists for, and it doubles as the ambiguity fixture — a two-session project
    is precisely what makes a bare `--session`-less command refuse to guess.
    Returns both session ids, oldest first."""
    r_sys = _cb("You are the RESEARCHER.", 0, "system", "system")
    w_sys = _cb("You are the WRITER.", 0, "system", "system")
    for started, tail, bump in (("2026-07-20T09:15:00+00:00", "good", 0),
                                ("2026-07-21T18:42:30+00:00", "bad", 1)):
        ct = CTrace.open_or_create_session(
            str(path), project="pipeline", provider="openai", model="",
            started_at=started)
        ct.record_call(seq=1, params={"model": "gpt-4o"},
                       usage={"prompt_tokens": 100 + bump, "completion_tokens": 20},
                       latency_ms=100, error=None,
                       call_blocks=[r_sys, _cb("Find facts about Mars.", 1)],
                       agent="researcher", step="gather")
        ct.record_call(seq=2, params={"model": "gpt-4o"},
                       usage={"prompt_tokens": 40, "completion_tokens": 8},
                       latency_ms=110, error=None,
                       call_blocks=[w_sys, _cb("Write an intro about Mars.", 1)],
                       agent="writer", step="compose")
        ct.record_call(seq=3, params={"model": "gpt-4o"}, usage=None,
                       latency_ms=120, error=None,
                       call_blocks=[r_sys, _cb("Find facts about Mars.", 1),
                                    _cb("Mars is the fourth planet.", 2,
                                        "assistant", "history"),
                                    _cb(f"More detail ({tail})?", 3)],
                       agent="researcher", step="gather")
        ct.close()
    ct = CTrace.open(str(path))
    try:
        return tuple(s.id for s in ct.list_sessions())  # type: ignore[return-value]
    finally:
        ct.close()


def _write_single(path) -> None:
    """A one-session, one-agent trace — the unambiguous case that needs no
    `--session` at all."""
    ct = CTrace.open_or_create_session(
        str(path), project="solo", provider="openai", model="",
        started_at="2026-07-01T00:00:00+00:00")
    ct.record_call(seq=1, params={"model": "gpt-4o"}, usage=None, latency_ms=10,
                   error=None, call_blocks=[_cb("hello", 0)], agent="solo")
    ct.close()


@pytest.fixture
def project(tmp_path):
    """The two-session project file plus its two session ids."""
    path = tmp_path / "project.ctrace"
    good, bad = _write_project(path)
    return str(path), good, bad


# --- `ctxdiff sessions` ---------------------------------------------------------


def test_sessions_lists_each_session_with_local_time_and_agents(project, capsys):
    """Every session of the project gets a row carrying its short id, a
    local-time column (date, time, offset), the turn count and the agents."""
    path, good, bad = project

    exit_code = main(["sessions", "--project", path])

    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert exit_code == 0
    assert len(lines) == 2
    # Two sessions in one file => each row is labeled <file>#<short id>.
    assert f"project.ctrace#{good[:12]}" in lines[0]
    assert f"project.ctrace#{bad[:12]}" in lines[1]
    for line in lines:
        assert "project=pipeline" in line
        assert "turns=3" in line
        assert "agents=researcher, writer" in line


def test_sessions_labels_a_single_session_file_by_filename(tmp_path, capsys):
    """The overwhelmingly common case — one session in a file — keeps the bare
    filename as its label, which is what a user recognizes."""
    _write_single(tmp_path / "solo.ctrace")

    exit_code = main(["sessions", "--project", str(tmp_path / "solo.ctrace")])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert out.startswith("solo.ctrace ")
    assert "#" not in out


def test_sessions_scans_the_whole_cwd_when_no_project_is_named(project, tmp_path,
                                                               monkeypatch, capsys):
    """Discovery is the one job where narrowing to the newest file would defeat
    the purpose: you cannot pick a project you were never shown."""
    _write_single(tmp_path / "solo.ctrace")
    monkeypatch.chdir(tmp_path)

    exit_code = main(["sessions"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "solo.ctrace" in out
    assert "project.ctrace#" in out


def test_runs_is_still_a_working_hidden_alias(project, capsys):
    """`runs` was renamed to `sessions`, but every existing script says `runs`
    — it must keep producing identical output, forever."""
    path, _, _ = project

    assert main(["sessions", "--project", path]) == 0
    via_sessions = capsys.readouterr()
    assert main(["runs", "--project", path]) == 0
    via_runs = capsys.readouterr()

    assert via_runs.out == via_sessions.out
    assert via_runs.err == via_sessions.err


def test_runs_is_hidden_from_the_help_listing(capsys):
    """Hidden means hidden: `runs` appears neither in the choices line nor in
    the per-command descriptions."""
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "sessions" in out
    assert "runs" not in out


def test_sessions_reports_an_empty_directory_and_exits_zero(tmp_path, monkeypatch,
                                                            capsys):
    monkeypatch.chdir(tmp_path)

    exit_code = main(["sessions"])

    assert exit_code == 0
    assert capsys.readouterr().out == "no .ctrace files in the current directory\n"


# --- `ctxdiff agents` -----------------------------------------------------------


def test_agents_aggregates_across_every_session_in_the_project(project, capsys):
    """An agent's footprint is a property of the PROJECT, not of whichever run
    happens to be newest: researcher = 2 sessions x 2 calls (only turn 1 of each
    reported usage: 120 + 121 = 241), writer = 2 sessions x 1 call (48 x 2)."""
    path, _, _ = project

    exit_code = main(["agents", "--project", path])

    lines = capsys.readouterr().out.strip().splitlines()
    assert exit_code == 0
    assert lines[0] == "researcher  sessions=2  calls=4  tokens=241"
    assert lines[1] == "writer  sessions=2  calls=2  tokens=96"


def test_agents_reports_dash_not_zero_when_no_usage_was_reported(tmp_path, capsys):
    """`tokens=0` would read as "this agent was free"; '-' says the truth, which
    is that the provider never told us."""
    _write_single(tmp_path / "solo.ctrace")

    exit_code = main(["agents", "--project", str(tmp_path / "solo.ctrace")])

    assert exit_code == 0
    assert capsys.readouterr().out == "solo  sessions=1  calls=1  tokens=-\n"


# --- ambiguity + selector resolution ---------------------------------------------


@pytest.mark.parametrize("argv", [
    ["tokens"],
    ["cache"],
    ["diff", "--turn", "1", "--turn", "3"],
    ["export"],
    ["view", "--no-open"],
])
def test_ambiguous_session_exits_2_and_lists_the_sessions(project, argv, capsys):
    """Every command that reads ONE session refuses to guess between two, and
    prints the listing the user needs to pick — usage error, exit 2."""
    path, good, bad = project

    exit_code = main([*argv, "--project", path])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.splitlines()[0] == (
        "ctxdiff: this project holds 2 sessions — pass --session to pick one:")
    assert good[:12] in captured.err
    assert bad[:12] in captured.err


def test_single_session_needs_no_flag(tmp_path, capsys):
    _write_single(tmp_path / "solo.ctrace")

    exit_code = main(["tokens", "--project", str(tmp_path / "solo.ctrace")])

    assert exit_code == 0
    assert "turn 1 ·" in capsys.readouterr().out


def test_session_prefix_resolves_and_an_unknown_one_lists(project, capsys):
    path, good, bad = project

    assert main(["tokens", "--project", path, "--session", bad[:12]]) == 0
    capsys.readouterr()

    exit_code = main(["tokens", "--project", path, "--session", "zzzz"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "no session 'zzzz' in this project — available sessions:" in captured.err


def test_session_scopes_the_analysis_to_that_run(project, capsys):
    """The chosen session's calls are what gets analyzed — the 'bad' run's turn
    3 carries its own text, not the 'good' run's."""
    path, good, bad = project

    assert main(["tokens", "--project", path, "--session", bad,
                 "--agent", "researcher", "--turn", "3"]) == 0
    out = capsys.readouterr().out
    assert "turn 3 ·" in out
    assert "turn 1 ·" not in out


@pytest.mark.parametrize("command", ["tokens", "cache"])
def test_unknown_agent_exits_2_and_lists_the_real_agents(project, command, capsys):
    """A typo'd agent used to filter every call away and exit 0 with "no calls
    in this run" — technically true, actively misleading. It is a bad flag."""
    path, good, _ = project

    exit_code = main([command, "--project", path, "--session", good,
                      "--agent", "nope"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err == (
        f"ctxdiff: no agent 'nope' in session {good[:12]} — available agents:"
        "\n  researcher\n  writer\n")


def test_run_flag_is_still_an_alias_for_project(project, capsys):
    path, _, _ = project

    assert main(["sessions", "--project", path]) == 0
    via_project = capsys.readouterr().out
    assert main(["sessions", "--run", path]) == 0
    via_run = capsys.readouterr().out

    assert via_run == via_project


# --- cross-session diff ----------------------------------------------------------


def test_cross_session_diff_compares_one_agent_across_two_runs(project, capsys):
    """The headline regression case: same agent, same turn, two runs. The scope
    header names both sessions (without it, `turn 3 → turn 3` is unreadable) and
    the diff shows exactly the one block that changed."""
    path, good, bad = project

    exit_code = main(["diff", "--project", path, "--session", f"{good}:3",
                      "--session", f"{bad}:3", "--agent", "researcher"])

    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert exit_code == 0
    assert lines[0] == (f"── {good[:12]} · researcher · turn 3  →  "
                        f"{bad[:12]} · researcher · turn 3 ──")
    assert "turn 3 → turn 3" in lines[1]
    assert "1 blocks changed" in lines[1]
    # Char-level inline diff: the shared trailing "d" of good/bad stays equal.
    assert "[-goo-]" in out
    assert "{+ba+}" in out
    assert "= 3 unchanged blocks" in out


def test_cross_session_one_turn_applies_to_both_sides(project, capsys):
    """`--turn 3` once means turn 3 on BOTH sides — the shape the regression
    case is actually typed in."""
    path, good, bad = project

    assert main(["diff", "--project", path, "--session", f"{good}:3",
                 "--session", f"{bad}:3", "--agent", "researcher"]) == 0
    via_suffix = capsys.readouterr().out
    assert main(["diff", "--project", path, "--session", good, "--session", bad,
                 "--turn", "3", "--agent", "researcher"]) == 0
    via_turn = capsys.readouterr().out

    assert via_turn == via_suffix


def test_cross_session_requires_agent_when_the_runs_hold_several(project, capsys):
    """Turn 3 is a global seq within each session, so with two agents
    interleaved it could mean different agents in the two runs. Rather than
    silently comparing unrelated contexts, ask."""
    path, good, bad = project

    exit_code = main(["diff", "--project", path, "--session", f"{good}:3",
                      "--session", f"{bad}:3"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err == (
        "ctxdiff: these sessions hold 2 agents — pass --agent to pick one:"
        "\n  researcher\n  writer\n")


def test_cross_session_turn_not_owned_by_the_agent_names_the_session(project, capsys):
    """"turn 2 not found" without saying WHERE is unusable when the diff spans
    two sessions."""
    path, good, bad = project

    exit_code = main(["diff", "--project", path, "--session", f"{good}:2",
                      "--session", f"{bad}:3", "--agent", "researcher"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err == (
        f"ctxdiff: session {good[:12]}: turn 2 is not a call of agent "
        "'researcher' (that agent's turns: [1, 3])\n")


def test_cross_session_without_any_turn_is_a_usage_error(project, capsys):
    path, good, bad = project

    exit_code = main(["diff", "--project", path, "--session", good,
                      "--session", bad, "--agent", "researcher"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err == (
        "ctxdiff: each side of a cross-session diff needs a turn — pass "
        "--session VALUE:TURN twice, or --turn N --turn M\n")


# --- cross-agent diff ------------------------------------------------------------


def test_cross_agent_diff_compares_two_agents_in_one_session(project, capsys):
    """Two agents, one run — and the identical session is DROPPED from the scope
    header, since only the axis that actually differs is worth showing."""
    path, good, _ = project

    exit_code = main(["diff", "--project", path, "--session", good,
                      "--agent", "researcher:1", "--agent", "writer:2"])

    lines = capsys.readouterr().out.strip().splitlines()
    assert exit_code == 0
    assert lines[0] == "── researcher · turn 1  →  writer · turn 2 ──"
    assert "turn 1 → turn 2" in lines[1]


def test_cross_agent_rejects_an_unknown_agent(project, capsys):
    path, good, _ = project

    exit_code = main(["diff", "--project", path, "--session", good,
                      "--agent", "researcher:1", "--agent", "ghost:2"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "no agent 'ghost' in session" in captured.err


def test_diff_refuses_to_mix_the_two_cross_axes(project, capsys):
    path, good, bad = project

    exit_code = main(["diff", "--project", path, "--session", good,
                      "--session", bad, "--agent", "researcher:1",
                      "--agent", "writer:2"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "compares along ONE axis" in captured.err


def test_diff_rejects_more_than_two_sides(project, capsys):
    path, good, bad = project

    exit_code = main(["diff", "--project", path, "--session", good,
                      "--session", bad, "--session", good, "--turn", "1"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "at most twice" in captured.err


def test_turn_suffix_is_rejected_on_an_ordinary_diff(project, capsys):
    """A `:TURN` suffix outside a cross diff is silently meaningless, so it is
    refused rather than ignored."""
    path, good, _ = project

    exit_code = main(["diff", "--project", path, "--session", f"{good}:1",
                      "--turn", "1", "--turn", "3"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "only means something on a cross-session or cross-agent diff" in captured.err

# --- flag surface: only the selectors a command really takes ---------------------


@pytest.mark.parametrize("argv", [
    ["cache", "--turn", "1"],
    ["sessions", "--session", "abc"],
    ["sessions", "--agent", "researcher"],
    ["sessions", "--turn", "1"],
    ["runs", "--agent", "researcher"],
    ["agents", "--agent", "researcher"],
    ["agents", "--session", "abc"],
    ["agents", "--turn", "1"],
])
def test_a_command_rejects_selectors_it_does_not_take(project, argv):
    """Each subparser registers ONLY the selectors it acts on, so a flag that
    command cannot honor is a usage error rather than a silently ignored one.

    Why it matters more than it looks: `ctxdiff agents --agent researcher` reads
    as "list only that agent". Accepting and ignoring it would print EVERY agent
    and exit 0 — indistinguishable from "the filter matched everything", so a
    script grepping that output is wrong forever and never learns. Exit 2 is the
    only honest answer. (argparse reports these from the TOP-level parser, which
    calls `parser.error()` -> SystemExit, hence the raises.)"""
    path, _, _ = project

    with pytest.raises(SystemExit) as excinfo:
        main([*argv, "--project", path])

    assert excinfo.value.code == 2


def test_turn_rejects_non_ascii_digits(project, capsys):
    """`--turn ٢` (ARABIC-INDIC DIGIT TWO) is rejected rather than silently
    read as 2. Bare `int()` accepts every Unicode decimal digit, which would
    make the flag mean something different from the `:TURN` selector suffix —
    `select.parse_selector` already requires ASCII there — and from the JS CLI's
    `/^[0-9]+$/`. One grammar for turns, stated the same way everywhere."""
    path, _, _ = project

    with pytest.raises(SystemExit) as excinfo:
        main(["tokens", "--project", path, "--turn", "٢"])

    assert excinfo.value.code == 2
    assert "invalid int value: '٢'" in capsys.readouterr().err


def test_turn_still_accepts_the_int_spellings_it_always_did(project, capsys):
    """The ASCII narrowing must not cost the ordinary spellings: surrounding
    whitespace, a leading `+`, leading zeros and a negative value all still
    parse, and all still normalize the way `int()` did."""
    path, good, _ = project

    for spelling in (" 3 ", "+3", "003"):
        assert main(["tokens", "--project", path, "--session", good,
                     "--turn", spelling]) == 0
        assert "turn 3 ·" in capsys.readouterr().out

    assert main(["tokens", "--project", path, "--session", good,
                 "--turn", "-1"]) == 1
    assert ("ctxdiff: turn -1 not found in this run"
            in capsys.readouterr().err)


def test_a_huge_turn_is_echoed_with_every_digit(project, capsys):
    """A turn larger than any session holds is reported back EXACTLY as typed —
    Python's ints are arbitrary-precision, and the JS CLI matches by carrying the
    raw text rather than a double that would print 1e+21."""
    path, good, _ = project

    exit_code = main(["tokens", "--project", path, "--session", good,
                      "--turn", "1000000000000000000000"])

    assert exit_code == 1
    assert ("ctxdiff: turn 1000000000000000000000 not found in this run"
            in capsys.readouterr().err)


# --- a DISCOVERED project is named in selector errors ----------------------------


def test_a_discovered_project_is_named_in_an_agent_error(tmp_path, monkeypatch,
                                                         capsys):
    """`ctxdiff agents` lists agents from EVERY .ctrace in the directory, so the
    obvious next command can name an agent that is real — just not in the one
    file the no-flag default picked. The error therefore names that file, not
    only a session short id that appears nowhere in the `sessions` listing (whose
    rows are labeled by filename). Without it the user has no hint that a
    different project was chosen, nor that `--project` is the fix."""
    _write_project(tmp_path / "one.ctrace")      # holds researcher + writer
    _write_single(tmp_path / "two.ctrace")       # newest -> the default
    monkeypatch.chdir(tmp_path)

    exit_code = main(["tokens", "--agent", "researcher"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err.startswith(
        "ctxdiff: no agent 'researcher' in two.ctrace (session ")
    assert "— available agents:" in captured.err


def test_a_named_project_is_not_re_labeled(project, capsys):
    """The filename appears ONLY when the project was discovered by scanning.
    A user who passed --project already knows which project they named, and the
    two CLIs' byte-identical error text depends on that staying stable."""
    path, good, _ = project

    exit_code = main(["tokens", "--project", path, "--session", good,
                      "--agent", "nope"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err.startswith(
        f"ctxdiff: no agent 'nope' in session {good[:12]} —")
