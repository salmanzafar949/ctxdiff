"""Unit tests for `ctxdiff.cli.select` — the session/agent resolution layer
behind `--project`/`--session`/`--agent`/`--turn`.

Four things are pinned here, all of which the CLI's byte-identity with the JS
SDK depends on: the `NAME:TURN` selector grammar, LOCAL-timezone timestamp
rendering (driven by `TZ`, including a DST-sensitive zone), the defaulting/
ambiguity rules and the exact error text they produce, and the cross-diff scope
header. Everything here is pure — no store is opened.
"""
import os
import time

import pytest

from ctxdiff.cli.select import (
    DiffSide,
    Selector,
    SelectionError,
    agent_picker,
    agents_text,
    choose_session,
    diff_scope_line,
    distinct_agent_names,
    format_local,
    parse_selector,
    require_agent,
    require_single_agent,
    session_line,
    short_id,
)
from ctxdiff.store.base import Call, Session


def _session(**over) -> Session:
    """A Session summary with sensible defaults, so each test states only the
    fields it actually cares about."""
    fields = dict(id="0123456789abcdef0123456789abcdef", project="demo",
                  started_at="2026-07-20T09:15:00+00:00", provider="openai",
                  models=["gpt-4o"], agents=[], turn_count=3)
    fields.update(over)
    return Session(**fields)


def _call(seq: int, agent: str | None) -> Call:
    """A Call carrying only what the selectors read: seq and agent."""
    return Call(id=f"c{seq}", run_id="r", seq=seq, params={}, usage=None,
                latency_ms=None, error=None, agent=agent)


@pytest.fixture
def local_tz(monkeypatch):
    """Set the process's local timezone for the duration of a test.

    `datetime.astimezone()` reads the C library's zone, which only re-reads `TZ`
    after `tzset()` — so both are done here, and pytest's monkeypatch restores
    the env var afterwards while a final `tzset()` restores the C-level state."""
    def _set(name: str) -> None:
        monkeypatch.setenv("TZ", name)
        time.tzset()
    yield _set
    os.environ.pop("TZ", None)
    time.tzset()


# --- selector grammar ---------------------------------------------------------


def test_parse_selector_splits_trailing_turn():
    """`name:N` splits into a name and an int turn."""
    assert parse_selector("researcher:8") == Selector("researcher", 8)
    assert parse_selector("4f3a2b1c9d8e:12") == Selector("4f3a2b1c9d8e", 12)


def test_parse_selector_leaves_a_bare_name_alone():
    assert parse_selector("researcher") == Selector("researcher", None)


def test_parse_selector_splits_on_the_last_colon():
    """A name may itself contain colons; only the final `:digits` is a turn."""
    assert parse_selector("tools:web:3") == Selector("tools:web", 3)
    assert parse_selector("tools:web") == Selector("tools:web", None)


@pytest.mark.parametrize("value", ["agent:٨", "agent:²", "agent:-1", "agent:", ":8"])
def test_parse_selector_rejects_non_ascii_digit_suffixes(value):
    """`str.isdigit()` accepts superscripts and Arabic-Indic digits; the
    selector grammar deliberately does not, so both SDKs parse the same strings
    the same way (JS tests `/^[0-9]+$/`)."""
    assert parse_selector(value) == Selector(value, None)


def test_short_id_and_agents_text():
    assert short_id("0123456789abcdef0123456789abcdef") == "0123456789ab"
    assert agents_text([]) == "-"
    assert agents_text(["a", "b"]) == "a, b"


# --- local-time rendering ------------------------------------------------------


def test_format_local_renders_in_the_local_zone_with_its_offset(local_tz):
    local_tz("Asia/Dubai")
    assert format_local("2026-07-20T09:15:00+00:00") == "2026-07-20 13:15:00 +04:00"
    local_tz("UTC")
    assert format_local("2026-07-20T09:15:00+00:00") == "2026-07-20 09:15:00 +00:00"


def test_format_local_uses_the_offset_at_that_instant_not_today(local_tz):
    """America/New_York is UTC-4 in July and UTC-5 in January: a fixed-offset
    renderer would get one of these two wrong."""
    local_tz("America/New_York")
    assert format_local("2026-07-20T09:15:00Z") == "2026-07-20 05:15:00 -04:00"
    assert format_local("2026-01-20T09:15:00Z") == "2026-01-20 04:15:00 -05:00"


def test_format_local_handles_a_half_hour_zone(local_tz):
    local_tz("Asia/Kolkata")
    assert format_local("2026-07-20T09:15:00Z") == "2026-07-20 14:45:00 +05:30"


def test_format_local_truncates_a_sub_minute_offset_toward_zero(local_tz):
    """Pre-standard-time zones carry LMT offsets with SECONDS in them —
    America/New_York was -04:56:02 until 1883, America/Los_Angeles -07:52:58.
    The offset is shown in whole minutes, and the rounding has to be toward zero
    to agree with V8's `Date.getTimezoneOffset()`; flooring would render -04:57
    for an instant Node calls -04:56."""
    local_tz("America/New_York")
    assert format_local("1800-06-01T12:00:00Z") == "1800-06-01 07:03:58 -04:56"
    local_tz("America/Los_Angeles")
    assert format_local("1800-06-01T12:00:00Z") == "1800-06-01 04:07:02 -07:52"


def test_format_local_treats_a_naive_stored_value_as_utc(local_tz):
    local_tz("UTC")
    assert format_local("2026-07-20T09:15:00") == "2026-07-20 09:15:00 +00:00"


def test_format_local_degrades_instead_of_raising():
    """An empty timestamp renders '-', an unparseable one is echoed back — a
    listing must never die over one odd row."""
    assert format_local("") == "-"
    assert format_local("   ") == "-"
    assert format_local("not a timestamp") == "not a timestamp"


def test_format_local_echoes_a_timestamp_local_time_cannot_represent(local_tz):
    """A stored value at the very edge of `datetime`'s range OVERFLOWS when the
    local offset shifts it past MINYEAR/MAXYEAR — `astimezone()` raises
    OverflowError, not ValueError, so a listing holding one such row used to die
    with a traceback and exit 1. It is echoed back like any other value that
    cannot be rendered."""
    local_tz("Asia/Dubai")  # east of UTC: 9999-12-31T23:59Z would become 10000
    assert format_local("9999-12-31T23:59:59+00:00") == "9999-12-31T23:59:59+00:00"
    local_tz("America/Los_Angeles")  # west of UTC: 0001-01-01T00:00Z becomes 0000
    assert format_local("0001-01-01T00:00:00+00:00") == "0001-01-01T00:00:00+00:00"


# --- session selection ---------------------------------------------------------

_A = _session(id="aaaa1111bbbb2222", turn_count=3, agents=["researcher"])
_B = _session(id="cccc3333dddd4444", turn_count=5)


def test_one_session_needs_no_flag():
    assert choose_session([_A], None, _A.id) == _A.id


def test_several_sessions_refuse_to_guess_and_list_them_all():
    """The ambiguity error names the count and prints every session as a
    pickable line — a usage error that doesn't show the options is a dead end."""
    with pytest.raises(SelectionError) as exc:
        choose_session([_A, _B], None, _B.id)
    message = str(exc.value)
    assert message.splitlines()[0] == (
        "ctxdiff: this project holds 2 sessions — pass --session to pick one:")
    assert "  " + session_line(_A) in message
    assert "  " + session_line(_B) in message


def test_exact_id_and_unambiguous_prefix_both_resolve():
    assert choose_session([_A, _B], _A.id, _B.id) == _A.id
    assert choose_session([_A, _B], "aaaa", _B.id) == _A.id


def test_unknown_session_reports_with_the_listing():
    with pytest.raises(SelectionError, match="no session 'zzzz' in this project"):
        choose_session([_A, _B], "zzzz", _B.id)


def test_ambiguous_prefix_lists_only_the_matches():
    other = _session(id="aaaa9999eeee0000")
    with pytest.raises(SelectionError) as exc:
        choose_session([_A, _B, other], "aaaa", _B.id)
    message = str(exc.value)
    assert message.splitlines()[0] == (
        "ctxdiff: session 'aaaa' is ambiguous — 2 sessions match:")
    assert short_id(_B.id) not in message


# --- agent selection -----------------------------------------------------------


def test_distinct_agent_names_are_named_only_and_first_seen_order():
    calls = [_call(1, "researcher"), _call(2, "writer"),
             _call(3, "researcher"), _call(4, None)]
    assert distinct_agent_names(calls) == ["researcher", "writer"]


def test_unknown_agent_lists_the_real_names():
    with pytest.raises(SelectionError) as exc:
        require_agent("nope", ["researcher", "writer"], "session abc")
    assert str(exc.value) == (
        "ctxdiff: no agent 'nope' in session abc — available agents:"
        "\n  researcher\n  writer")


def test_agent_picker_explains_an_agentless_session():
    """'available agents:' followed by nothing would read like a bug."""
    assert agent_picker([]) == "  (none — this session's calls carry no agent label)"
    with pytest.raises(SelectionError, match="carry no agent label"):
        require_agent("x", [], "session abc")


def test_single_agent_is_required_only_when_there_are_several():
    require_single_agent([], "these sessions")        # nothing to get wrong
    require_single_agent(["solo"], "these sessions")  # only one candidate
    with pytest.raises(SelectionError) as exc:
        require_single_agent(["a", "b"], "these sessions")
    assert str(exc.value) == (
        "ctxdiff: these sessions hold 2 agents — pass --agent to pick one:\n  a\n  b")


# --- the cross-diff scope header ------------------------------------------------


def test_scope_line_is_none_for_an_ordinary_diff():
    """Same session, same agent: `render_turn_diff`'s own header is already the
    complete truth, so nothing is prepended and existing output is unchanged."""
    assert diff_scope_line(DiffSide("s1", "r", 7), DiffSide("s1", "r", 8)) is None


def test_scope_line_names_both_sessions_across_runs():
    line = diff_scope_line(DiffSide("aaaa1111bbbb2222", "researcher", 8),
                           DiffSide("cccc3333dddd4444", "researcher", 8))
    assert line == ("── aaaa1111bbbb · researcher · turn 8  →  "
                    "cccc3333dddd · researcher · turn 8 ──")


def test_scope_line_drops_the_identical_session_on_a_cross_agent_diff():
    line = diff_scope_line(DiffSide("s1", "researcher", 1),
                           DiffSide("s1", "writer", 2))
    assert line == "── researcher · turn 1  →  writer · turn 2 ──"


def test_scope_line_omits_the_agent_when_none_was_selected():
    line = diff_scope_line(DiffSide("aaaa1111bbbb2222", None, 8),
                           DiffSide("cccc3333dddd4444", None, 8))
    assert line == "── aaaa1111bbbb · turn 8  →  cccc3333dddd · turn 8 ──"
