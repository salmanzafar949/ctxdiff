"""The context window: resolving it, and rendering a turn as a share of it.

The property this file defends is that ctxdiff never invents a denominator.
There is no model→context-window table in the package (the same decision that
keeps a price table out), so a percentage appears only when the user has stated
the window — as a flag or as `CTXDIFF_CONTEXT_WINDOW` — and with no window every
command prints exactly the bytes it printed before percentages existed.

The second property is that the flag and the environment variable are resolved
in ONE place, so `ctxdiff tokens`, `ctxdiff check` and the exported dashboard can
never be scored against two different windows on the same machine.
"""
from __future__ import annotations

import pytest

from ctxdiff.analyze.window import (
    CONTEXT_WINDOW_ALARM_PCT,
    CONTEXT_WINDOW_ENV,
    ContextWindowError,
    format_window_share,
    is_alarming,
    parse_context_window,
    resolve_context_window,
    window_pct,
)
from ctxdiff.cli import main
from ctxdiff.models import Block, CallBlock
from ctxdiff.store.ctrace import CTrace


def _cb(text, position, role="user", kind="message", label="user", tokens=10):
    """A CallBlock with an explicit token count, so a turn's total is a fixture
    constant rather than something the tokenizer decides."""
    block = Block(content_hash=f"h:{text}", role=role, kind=kind, text=text,
                  token_count=tokens, token_method="tiktoken")
    return CallBlock(block=block, position=position, label=label,
                     label_source="heuristic")


@pytest.fixture()
def trace(tmp_path):
    """A two-turn trace whose turn totals are exactly 40 and 90 tokens — chosen
    so that against a 100-token window one turn sits comfortably under the alarm
    threshold (40.0%) and the other is well past it (90.0%)."""
    path = str(tmp_path / "w.ctrace")
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")
    turn1 = [_cb("system", 0, role="system", label="system", tokens=25),
             _cb("hello", 1, tokens=15)]
    turn2 = turn1 + [_cb("reply", 2, role="assistant", label="history", tokens=30),
                     _cb("again", 3, tokens=20)]
    for seq, blocks in ((1, turn1), (2, turn2)):
        ct.record_call(seq=seq, params={"model": "gpt-4o"}, usage=None,
                       latency_ms=1, error=None, call_blocks=blocks)
    ct.close()
    return path


# --- resolution ----------------------------------------------------------------


def test_the_flag_wins_over_the_environment(monkeypatch):
    """A flag typed on this invocation is the most specific thing anyone said,
    so it beats an ambient variable rather than the other way round."""
    monkeypatch.setenv(CONTEXT_WINDOW_ENV, "200000")
    assert resolve_context_window(8000) == 8000


def test_the_environment_supplies_the_window_when_no_flag_does(monkeypatch):
    """The whole point of the variable: `tokens`, `check`, `view` and `export`
    agree without the window being retyped for each of them."""
    monkeypatch.setenv(CONTEXT_WINDOW_ENV, "200000")
    assert resolve_context_window(None) == 200000


def test_no_flag_and_no_variable_means_no_window(monkeypatch):
    """None is 'render nothing', never a guessed default — ctxdiff ships no
    model→window table by design."""
    monkeypatch.delenv(CONTEXT_WINDOW_ENV, raising=False)
    assert resolve_context_window(None) is None


def test_an_empty_variable_is_unset_rather_than_an_error(monkeypatch):
    """`CTXDIFF_CONTEXT_WINDOW= ctxdiff tokens` is how a shell unsets a variable
    for one command; failing there would make that idiom unusable."""
    monkeypatch.setenv(CONTEXT_WINDOW_ENV, "   ")
    assert resolve_context_window(None) is None


@pytest.mark.parametrize("value", ["abc", "200k", "1e5", "12.5", "٢", ""])
def test_an_unusable_variable_is_reported_not_ignored(monkeypatch, value):
    """A percentage that quietly stops rendering looks exactly like a percentage
    that is fine, so a malformed window is an error. The grammar is narrowed to
    ASCII digits — the same narrowing `--turn` uses — so a non-ASCII digit that
    `int()` would silently accept is refused in both SDKs."""
    if value == "":
        pytest.skip("empty means unset; covered above")
    monkeypatch.setenv(CONTEXT_WINDOW_ENV, value)
    with pytest.raises(ContextWindowError) as exc:
        resolve_context_window(None)
    assert CONTEXT_WINDOW_ENV in str(exc.value)


@pytest.mark.parametrize("value", ["0", "-5"])
def test_a_non_positive_window_is_refused(value):
    """A zero window is a division by zero and a negative one is not a window."""
    with pytest.raises(ContextWindowError):
        parse_context_window(value)


@pytest.mark.parametrize("value", [0, -5])
def test_a_non_positive_flag_is_refused_by_the_resolver_too(value, monkeypatch):
    """The rule belongs to the WINDOW, not to the place it was typed. The
    resolver used to hand the flag back unchecked, so the environment path
    rejected a zero and the flag path divided by it — `tokens --context-window 0`
    was a `ZeroDivisionError` traceback in Python and `⚠ Infinity%` in JS, and
    `--context-window -5` rendered `-260.0%` in both."""
    monkeypatch.delenv(CONTEXT_WINDOW_ENV, raising=False)
    with pytest.raises(ContextWindowError) as exc:
        resolve_context_window(value)
    assert f"--context-window must be greater than 0 (got {value})" in str(exc.value)


# --- rendering -------------------------------------------------------------------


def test_the_share_names_both_numbers_and_the_percentage():
    """`18,400 / 200,000 tok · 9.2%` — the numerator alone is what makes a token
    count unactionable, so the denominator travels with it."""
    assert format_window_share(18400, 200000) == "18,400 / 200,000 tok · 9.2%"


def test_the_alarm_marker_trips_at_the_documented_threshold():
    """The marker is compared against the DISPLAYED percentage, so a turn shown
    as 80.0% is marked and one shown as 79.9% is not — no invisible third number
    decides it."""
    assert is_alarming(CONTEXT_WINDOW_ALARM_PCT)
    assert not is_alarming(CONTEXT_WINDOW_ALARM_PCT - 0.1)
    assert "⚠" in format_window_share(80, 100)
    assert "⚠" not in format_window_share(79, 100)


def test_the_percentage_is_rounded_to_one_decimal():
    """One decimal, rounded before formatting — the split every ctxdiff
    percentage uses so CPython's round and the JS twin's `pyRound1` cannot
    disagree on a boundary."""
    assert window_pct(1, 3) == 33.3
    assert format_window_share(1, 3) == "1 / 3 tok · 33.3%"


def test_an_exactly_representable_tie_rounds_to_even():
    """The boundary that is not hypothetical. `61 / 80 * 100` is EXACTLY 76.25 —
    a double, not an approximation — so it is a real tie, and CPython's `round`
    takes it to even: 76.2, not 76.3.

    Pinned here because the JS twin used to disagree: `toFixed(1)` rounds half
    away from zero, so the same trace printed `76.3%` from the npm CLI and
    `76.2%` from the pip one. The ties that occur in ctxdiff all end `.25` or
    `.75` (they are `odd/4`, hence exactly representable), and they fall out of
    ordinary inputs — this one is a 61-token turn against an 80-token window, and
    `1 / 16 * 100` is a one-token slice of a sixteen-token turn."""
    assert window_pct(61, 80) == 76.2
    assert window_pct(1, 16) == 6.2
    assert window_pct(3, 16) == 18.8
    assert format_window_share(61, 80) == "61 / 80 tok · 76.2%"


# --- the CLI ----------------------------------------------------------------------


def test_tokens_without_a_window_is_unchanged(trace, capsys, monkeypatch):
    """The regression that matters most: with no window stated, the turn header
    is byte-for-byte what it always was."""
    monkeypatch.delenv(CONTEXT_WINDOW_ENV, raising=False)
    main(["tokens", "--project", trace])
    out = capsys.readouterr().out
    assert "turn 1 · 40 tokens" in out
    assert "/" not in out.split("\n")[2]


def test_tokens_with_a_window_renders_the_share(trace, capsys, monkeypatch):
    """Both turns, one quiet and one alarming, from one flag."""
    monkeypatch.delenv(CONTEXT_WINDOW_ENV, raising=False)
    main(["tokens", "--project", trace, "--context-window", "100"])
    out = capsys.readouterr().out
    assert "turn 1 · 40 / 100 tok · 40.0%" in out
    assert "turn 2 · 90 / 100 tok · ⚠ 90.0%" in out


def test_tokens_reads_the_window_from_the_environment(trace, capsys, monkeypatch):
    """No flag typed, and the dashboard/CLI still show percentages — which is
    the ergonomic reason the variable exists."""
    monkeypatch.setenv(CONTEXT_WINDOW_ENV, "100")
    main(["tokens", "--project", trace])
    assert "turn 1 · 40 / 100 tok · 40.0%" in capsys.readouterr().out


def test_a_bad_environment_window_is_a_usage_error(trace, capsys, monkeypatch):
    """Exit 2, not 1: the variable is part of how the command was invoked, not
    part of what it was pointed at — and it fails before any store is opened."""
    monkeypatch.setenv(CONTEXT_WINDOW_ENV, "lots")
    assert main(["tokens", "--project", trace]) == 2
    assert CONTEXT_WINDOW_ENV in capsys.readouterr().err


@pytest.mark.parametrize("command", ["tokens", "view", "export"])
@pytest.mark.parametrize("value", ["0", "-5"])
def test_the_display_commands_refuse_a_non_positive_flag(
        command, value, trace, tmp_path, capsys, monkeypatch):
    """All four commands inherit the rule from the ONE resolver, so none of them
    can render a percentage against a window that is not one. Exit 2 — a usage
    error, the same code the environment path has always returned — instead of a
    traceback, an `⚠ Infinity%` dashboard or a `-260.0%` turn header."""
    monkeypatch.delenv(CONTEXT_WINDOW_ENV, raising=False)
    argv = [command, "--project", trace, "--context-window", value]
    if command == "export":
        argv += ["--out", str(tmp_path / "d.html")]
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert f"ctxdiff: --context-window must be greater than 0 (got {value})" in err


def test_check_keeps_naming_itself_when_it_refuses_a_zero_window(
        trace, capsys, monkeypatch):
    """`check`'s usage errors are all prefixed `ctxdiff check:` so a CI log says
    which step spoke; the shared resolver must not take that sentence away from
    it."""
    monkeypatch.delenv(CONTEXT_WINDOW_ENV, raising=False)
    assert main(["check", "--project", trace, "--max-context-pct", "50",
                 "--context-window", "0"]) == 2
    assert ("ctxdiff check: --context-window must be greater than 0 (got 0)"
            in capsys.readouterr().err)


def test_check_and_tokens_resolve_the_same_window(trace, capsys, monkeypatch):
    """The reason both commands go through one resolver: a gate scored against a
    window a human's report never saw would be a gate nobody could audit. With
    the variable set, `--max-context-pct` needs no flag and the percentage it
    quotes is the one `tokens` printed."""
    monkeypatch.setenv(CONTEXT_WINDOW_ENV, "100")
    main(["tokens", "--project", trace])
    tokens_out = capsys.readouterr().out
    assert main(["check", "--project", trace, "--max-context-pct", "50"]) == 1
    check_out = capsys.readouterr().out
    assert "90.0%" in tokens_out
    assert "90.0% of 100 tok window" in check_out


def test_an_ambient_window_alone_never_triggers_the_unused_flag_error(
        trace, capsys, monkeypatch):
    """Rule 3 of `check`'s flag validation is about what the user TYPED. Failing
    a `--max-context` check because the shell happens to know the window would
    be absurd."""
    monkeypatch.setenv(CONTEXT_WINDOW_ENV, "100")
    assert main(["check", "--project", trace, "--max-context", "1000"]) == 0


def test_the_typed_flag_still_has_to_be_consumed(trace, capsys, monkeypatch):
    """...but a flag someone typed and nothing reads is still a usage error:
    silently ignoring it is how a CI gate asserts less than its author believes."""
    monkeypatch.delenv(CONTEXT_WINDOW_ENV, raising=False)
    assert main(["check", "--project", trace, "--max-context", "1000",
                 "--context-window", "100"]) == 2


def test_the_dashboard_embeds_the_window_and_every_turns_percentage(
        trace, tmp_path, capsys, monkeypatch):
    """The exported page carries the resolved window and each turn's PRECOMPUTED
    percentage, so the browser never rounds and the two SDKs emit the same
    bytes."""
    import json
    import re

    monkeypatch.setenv(CONTEXT_WINDOW_ENV, "100")
    out = str(tmp_path / "d.html")
    assert main(["export", "--project", trace, "--out", out]) == 0
    html = open(out, encoding="utf-8").read()
    island = re.search(
        r'<script id="ctxdiff-data" type="application/json">(.*?)</script>',
        html, re.S).group(1)
    payload = json.loads(island.replace("<\\/", "</"))
    assert payload["tokens"]["context_window"] == 100
    assert payload["tokens"]["window_alarm_pct"] == CONTEXT_WINDOW_ALARM_PCT
    assert [c["pct_of_window"] for c in payload["tokens"]["calls"]] == [40.0, 90.0]


def test_the_dashboard_shows_no_percentages_without_a_window(
        trace, tmp_path, monkeypatch):
    """No window, no invented denominator — the null is what the page checks
    before rendering any share."""
    import json
    import re

    monkeypatch.delenv(CONTEXT_WINDOW_ENV, raising=False)
    out = str(tmp_path / "d.html")
    assert main(["export", "--project", trace, "--out", out]) == 0
    island = re.search(
        r'<script id="ctxdiff-data" type="application/json">(.*?)</script>',
        open(out, encoding="utf-8").read(), re.S).group(1)
    payload = json.loads(island.replace("<\\/", "</"))
    assert payload["tokens"]["context_window"] is None
    assert all(c["pct_of_window"] is None for c in payload["tokens"]["calls"])
