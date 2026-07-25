"""The context-window resolver and the one place a "share of the window" is
phrased.

Why this module exists at all. `18,400 tok` is a number nobody can act on;
`18,400 / 200,000 tok · 9.2%` is, and `164,000 / 200,000 tok · ⚠ 82.0%` is the
alarm that explains the bug someone is actually chasing — the provider silently
dropping the oldest half of the conversation. Proximity to the limit is the
fact; the absolute count is only its numerator.

Why ctxdiff will not supply the denominator itself. There is deliberately NO
model→context-window table in this package, for the same reason there is no
price table: those numbers change per model, per provider, per deployment and
per month, and a stale one does not degrade — it LIES, quietly, in the exact
direction that makes a gate pass. So the window is the user's to state, and
every percentage in ctxdiff renders only when they have stated it. With no
window, every command falls back to precisely the output it produced before this
module existed.

The two ways to state it, in priority order (`resolve_context_window`):

1. `--context-window N` on the command — wins always, because a flag typed on
   this invocation is the most specific thing anyone said;
2. `CTXDIFF_CONTEXT_WINDOW=N` in the environment — set once in a shell profile
   or a CI job, it makes `tokens`, `check`, `view` and `export` all agree
   without re-typing, and it is what lets the exported dashboard show
   percentages with no flag at all. Same shape as the existing `CTXDIFF_STORE`
   convention.

There is deliberately no third source. In particular the window is NOT recorded
into the `.ctrace`: it never appears on the wire, so capture cannot observe it,
and a trace is evidence — a number baked into an evidence file that nobody can
verify from the file and that a debugger must not rewrite (`CTrace.open()` is
read-only by design) is a stale-metadata lie with no correction path. An
environment variable is corrected by typing a different one.
"""
from __future__ import annotations

import os
import re
from collections.abc import Mapping

# The environment variable that supplies a context window when no flag does.
# Named like `CTXDIFF_STORE` so the two read as one family of settings.
CONTEXT_WINDOW_ENV = "CTXDIFF_CONTEXT_WINDOW"

# At or above this percentage of the window, a turn is rendered with a warning
# marker rather than a bare number.
#
# 80% and not 90%: the thing being warned about is not "you exceeded the window"
# (the provider would have errored, and you would know) but the SILENT failure
# just below it — a framework's sliding-window trimmer, a `max_tokens` reservation
# for the response, or a single large tool result arriving next turn. By 80% the
# remaining headroom is smaller than one typical tool output, so the next turn is
# where content starts disappearing. Compared against the DISPLAYED (1-decimal)
# percentage, so the marker never contradicts the number printed beside it.
CONTEXT_WINDOW_ALARM_PCT = 80.0

# The marker prefixed to an alarming percentage. Same `⚠` the cache profiler and
# the bloat line use, so one glyph means "ctxdiff wants your attention" everywhere.
ALARM_MARKER = "⚠ "


class ContextWindowError(ValueError):
    """A context window that was supplied but cannot be used — a non-numeric or
    non-positive `CTXDIFF_CONTEXT_WINDOW`, or a non-positive `--context-window`.
    A `ValueError` subclass so existing broad handlers keep working; its own type
    so the CLI can report it as the usage error it is (exit 2) rather than as a
    failure to read a trace."""


# The integer grammar the environment variable accepts: optional surrounding
# whitespace, an optional sign, then ASCII decimal digits. Narrower than `int()`
# on purpose and IDENTICAL to the CLI's `--turn`/`--context-window` grammar —
# `int()` also accepts every Unicode decimal digit, so `CTXDIFF_CONTEXT_WINDOW=٢`
# would quietly mean 2 in Python and be rejected by the JS twin.
_INT_RE = re.compile(r"^\s*[+-]?[0-9]+\s*$")


def parse_context_window(text: str) -> int:
    """Parse an environment-supplied context window into a positive int.

    Raises `ContextWindowError` naming the variable and echoing what was found
    for anything that is not ASCII digits, or that parses to zero or less: a
    zero window is a division by zero and a negative one is not a window, and
    both are far better reported than silently ignored — a percentage that
    quietly stops rendering looks exactly like a percentage that is fine."""
    if not _INT_RE.match(text):
        raise ContextWindowError(
            f"ctxdiff: {CONTEXT_WINDOW_ENV} must be a whole number of tokens "
            f"(got {text!r})")
    value = int(text)
    if value <= 0:
        raise ContextWindowError(
            f"ctxdiff: {CONTEXT_WINDOW_ENV} must be greater than 0 "
            f"(got {value})")
    return value


def resolve_context_window(flag: int | None,
                           env: Mapping[str, str] | None = None) -> int | None:
    """The ONE window-resolution path, shared by `tokens`, `check`, `view` and
    `export` so no two commands can disagree about the denominator on the same
    machine: an explicit `--context-window` if given, else
    `CTXDIFF_CONTEXT_WINDOW` if set to something usable, else None ("no window
    known" — render nothing rather than guess).

    An empty or whitespace-only environment value is treated as UNSET rather
    than as an error: `CTXDIFF_CONTEXT_WINDOW=` is how a shell unsets a variable
    for one command, and failing there would make that idiom unusable. Any other
    unusable value raises `ContextWindowError`.

    The POSITIVITY rule belongs to the window itself, not to the place it was
    typed, so it is enforced here rather than in `parse_context_window` alone:
    a zero window is a division by zero and a negative one is not a window,
    whichever of the two sources produced it. Enforcing it in the one shared
    resolver is what makes all four commands inherit it — `tokens`, `view` and
    `export` used to take the flag on trust and render a traceback, an
    `⚠ Infinity%` dashboard or a `-260.0%` turn header respectively, while only
    `check` (which re-validated it for its own gate) and the environment path
    said no."""
    if flag is not None:
        if flag <= 0:
            raise ContextWindowError(
                f"ctxdiff: --context-window must be greater than 0 (got {flag})")
        return flag
    raw = (env if env is not None else os.environ).get(CONTEXT_WINDOW_ENV)
    if raw is None or not raw.strip():
        return None
    return parse_context_window(raw)


def round1(value: float) -> float:
    """`round(value, 1)` — the single named rounding step every ctxdiff
    percentage goes through before it is formatted, so the JS twin has one thing
    to mirror (`pyRound1`) instead of an inline `round` at each call site."""
    return round(value, 1)


def pct_text(value: float) -> str:
    """A percentage rendered to one decimal. Callers round through `round1`
    FIRST and format second, so CPython's round-half-to-even and JS's
    `toFixed(1)` cannot disagree on a boundary case."""
    return f"{value:.1f}"


def window_pct(total_tokens: int, window: int) -> float:
    """`total_tokens` as a percentage of `window`, rounded to one decimal. The
    ratio is computed with the same IEEE-754 operations in both SDKs, so only
    the rounding step needs matching."""
    return round1(total_tokens / window * 100)


def is_alarming(pct: float) -> bool:
    """Whether an already-rounded percentage has reached the alarm threshold.
    Compared against the DISPLAYED value so `80.0%` and its marker always agree
    — a turn shown as `80.0%` is marked, one shown as `79.9%` is not, with no
    invisible third number deciding it."""
    return pct >= CONTEXT_WINDOW_ALARM_PCT


def format_window_share(total_tokens: int, window: int) -> str:
    """One turn's context as a share of the window:
    `18,400 / 200,000 tok · 9.2%`, or `164,000 / 200,000 tok · ⚠ 82.0%` once the
    percentage reaches `CONTEXT_WINDOW_ALARM_PCT`.

    The marker sits on the PERCENTAGE rather than at the end of the line because
    the percentage is the thing that is alarming; a trailing glyph would read as
    a comment on the whole row and would collide with the `(~approx)` marker the
    caller appends after this string."""
    pct = window_pct(total_tokens, window)
    marker = ALARM_MARKER if is_alarming(pct) else ""
    return (f"{total_tokens:,} / {window:,} tok · {marker}{pct_text(pct)}%")
