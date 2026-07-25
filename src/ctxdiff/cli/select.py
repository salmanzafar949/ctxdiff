"""Session/agent selection — the resolution layer behind every read command's
`--project` / `--session` / `--agent` / `--turn` flags.

A project store now holds MANY sessions and each session MANY agents, so a read
command has to answer three questions before it can analyze anything: which
project, which session, which agent. `cli/main.py` owns the first (it knows the
store-opening rules); this module owns the other two, identically for every
command, so `diff`, `tokens`, `cache`, `export` and `view` can never disagree
about what `--session 4f3a2b1c` means.

Three ideas carry the whole module:

- A SELECTOR is a name with an OPTIONAL `:TURN` suffix — `researcher:8`,
  `4f3a2b1c:8`. That one piece of syntax is what makes a cross-session or
  cross-agent diff expressible with the flags that already exist
  (`--session GOOD:8 --session BAD:8`) instead of a bespoke sub-command.
- DEFAULTING is silent only when it cannot be wrong. Exactly one session in the
  project => no `--session` needed. Several => the user must say which, because
  quietly picking "the newest" would analyze a different run than the one they
  asked about and give them a confidently wrong answer.
- An unresolvable selector raises `SelectionError`, and its message ALREADY
  CONTAINS the listing of what could have been picked. The flag is a usage error
  (exit 2), and a usage error that doesn't show you the options is a dead end.

Everything here is pure: nothing is opened, nothing is queried, and the only
ambient input is the local timezone offset used to render a stored UTC
timestamp (see `format_local`).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from ctxdiff.store.base import Call, Session, parse_started_at

# How many leading characters of a session's uuid the CLI shows and accepts.
# 12 hex chars is 48 bits — collision-free for any realistic project DB, short
# enough to paste, and long enough that a prefix match is unambiguous.
SHORT_ID_LEN = 12

# The bucket name for calls that carry no agent label, shared with the token
# analyzer so one run never reports two different names for the same calls.
UNLABELED = "(unlabeled)"


class SelectionError(Exception):
    """A `--session`/`--agent` value that cannot be resolved, or a defaulting
    decision that would have to be a guess.

    Its own exception type (rather than ValueError) so `cli/main.py` can map it
    to argparse's usage-error exit code 2 and print it VERBATIM — the message is
    built here complete with the listing of available sessions/agents, because
    the only useful response to "which session?" is the list of sessions."""


@dataclass(frozen=True)
class Selector:
    """One `--session`/`--agent` value: a `name` plus the turn it pinned, if
    any. `turn` is None for a plain `--agent researcher`, and 8 for
    `--agent researcher:8` — the suffix form that lets one flag name both a
    side of a diff and the turn to take from it."""
    name: str
    turn: int | None


def parse_selector(value: str) -> Selector:
    """Split a selector into its name and optional `:TURN` suffix.

    How: partition on the LAST colon and accept the suffix as a turn only when
    it is a non-empty run of ASCII digits and something remains on the left.
    Splitting from the right means a name that itself contains a colon keeps it
    (`--agent tools:web:3` is agent `tools:web`, turn 3); requiring ASCII digits
    (not `str.isdigit()`, which accepts superscripts and Arabic-Indic digits)
    keeps this byte-identical to the JS `/^[0-9]+$/` test. Anything else is a
    plain name, so a bare session id or agent name round-trips untouched."""
    head, sep, tail = value.rpartition(":")
    if sep and head and tail.isascii() and tail.isdigit():
        return Selector(name=head, turn=int(tail))
    return Selector(name=value, turn=None)


def short_id(session_id: str) -> str:
    """The 12-character prefix of a session id used everywhere the CLI shows or
    labels a session. Prefix rather than a hash so what is printed is also what
    can be pasted straight back into `--session` (see `choose_session`, which
    accepts any unambiguous prefix)."""
    return session_id[:SHORT_ID_LEN]


def format_local(started_at: str) -> str:
    """Render a session's stored UTC `started_at` in the VIEWER's LOCAL
    timezone, with the offset shown: `2026-07-24 16:03:11 +04:00`.

    Why local: a stored timestamp is UTC (canonical, portable), but the person
    reading the listing is trying to match a session against "the run I did
    after lunch". Showing the offset keeps it unambiguous when the trace is
    shared across zones.

    How, and why it is byte-identical to the JS twin: `astimezone()` with no
    argument converts to the process's local zone — the SAME zone Node's
    `Date.getTimezoneOffset()` reports, both honoring `TZ` — and the offset in
    effect AT THAT INSTANT, so a summer timestamp renders in summer time.
    Every field is formatted by hand rather than via `strftime`, because
    `strftime`'s zero-padding of years and its `%z` spelling vary by platform
    while `f"{n:02d}"` does not.

    Degrades rather than raises: an empty `started_at` (a trace created without
    one) renders `-`, and any value that cannot be turned into a local time is
    echoed back unchanged, so a listing never dies over one odd row. Three
    failures count as "cannot": a value ISO parsing rejects (`ValueError`); one
    whose LOCAL shift lands outside `datetime`'s MINYEAR..MAXYEAR range, which
    raises `OverflowError` rather than ValueError — a stored `9999-12-31T23:59Z`
    read east of UTC, or `0001-01-01T00:00Z` read west of it — and a zone lookup
    the platform refuses (`OSError`, which `astimezone()` can surface from
    `time.localtime`). Every one of them used to escape as an unhandled
    traceback that killed the whole listing over a single row."""
    text = (started_at or "").strip()
    if not text:
        return "-"
    try:
        dt = parse_started_at(text).astimezone()
    except (ValueError, OverflowError, OSError):
        return text
    offset = dt.utcoffset() or timedelta(0)
    # Whole minutes, TRUNCATED TOWARD ZERO rather than floored. Only pre-standard
    # time zones have sub-minute offsets (America/New_York was LMT -04:56:02
    # until 1883), and for those the two roundings disagree: flooring -17762s
    # gives -297 minutes (-04:57), truncation -296 (-04:56). V8's
    # `Date.getTimezoneOffset()` truncates, and the JS CLI's twin of this line
    # has no choice about that — so this one matches it.
    seconds = int(offset.total_seconds())
    minutes = -((-seconds) // 60) if seconds < 0 else seconds // 60
    sign = "-" if minutes < 0 else "+"
    hours, mins = divmod(abs(minutes), 60)
    return (f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d} "
            f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d} "
            f"{sign}{hours:02d}:{mins:02d}")


def agents_text(names: list[str]) -> str:
    """Comma-join agent names for a listing, or `-` when a session has none (a
    single-agent or pre-v2 session). One helper so the `sessions` table, the
    ambiguity picker and the `runs` alias never disagree on the empty case."""
    return ", ".join(names) if names else "-"


def session_line(s: Session) -> str:
    """One session as a single picker line: short id, local start time, turn
    count, agents. Deliberately the same columns the `sessions` table shows,
    so the listing a user is shown when they must pick reads exactly like the
    listing they would have run `ctxdiff sessions` to get."""
    return (f"{short_id(s.id)}  {format_local(s.started_at)}  "
            f"turns={s.turn_count}  agents={agents_text(s.agents)}")


def session_picker(sessions: list[Session]) -> str:
    """The indented block of session lines appended to an ambiguity error."""
    return "\n".join("  " + session_line(s) for s in sessions)


def agent_picker(names: list[str]) -> str:
    """The indented block of agent names appended to an agent error, or an
    explanatory line when the session has no labeled agents at all — 'available
    agents:' followed by nothing would read like a bug in the tool."""
    if not names:
        return "  (none — this session's calls carry no agent label)"
    return "\n".join("  " + n for n in names)


def choose_session(sessions: list[Session], wanted: str | None,
                   default_id: str) -> str:
    """Resolve which session to read, or raise `SelectionError` carrying the
    listing of what could be picked.

    With `wanted`: an exact id wins; otherwise any session whose id STARTS WITH
    the value matches, so the 12-char id printed by `ctxdiff sessions` can be
    pasted straight back. Zero matches and several matches are both errors, and
    both print the listing — a prefix that matches two sessions is a question,
    not a failure.

    Without `wanted`: one session (or none — an empty store is reported
    elsewhere) means `default_id`, the handle's own binding, which is the newest
    session. More than one is AMBIGUOUS and refuses to guess: this is the case
    where silently defaulting would analyze yesterday's run and say nothing."""
    if wanted is not None:
        exact = [s for s in sessions if s.id == wanted]
        matches = exact or [s for s in sessions if s.id.startswith(wanted)]
        if len(matches) == 1:
            return matches[0].id
        if not matches:
            raise SelectionError(
                f"ctxdiff: no session '{wanted}' in this project — "
                f"available sessions:\n{session_picker(sessions)}")
        raise SelectionError(
            f"ctxdiff: session '{wanted}' is ambiguous — {len(matches)} "
            f"sessions match:\n{session_picker(matches)}")

    if len(sessions) <= 1:
        return default_id
    raise SelectionError(
        f"ctxdiff: this project holds {len(sessions)} sessions — pass "
        f"--session to pick one:\n{session_picker(sessions)}")


def distinct_agent_names(calls: list[Call]) -> list[str]:
    """The distinct NAMED agents across `calls`, in first-appearance order.
    Unlabeled calls contribute nothing (unlike the analyzers' `distinct_agents`,
    which counts a None bucket) because this list exists to be shown to a user
    picking an `--agent` value, and `--agent None` is not a thing they can
    type."""
    names: list[str] = []
    for c in calls:
        if c.agent and c.agent not in names:
            names.append(c.agent)
    return names


def require_agent(agent: str, names: list[str], where: str) -> None:
    """Assert that `agent` actually exists in a session, raising a
    `SelectionError` listing the real agent names when it doesn't. `where`
    describes the scope in the message ('session 4f3a2b1c', 'these sessions'),
    since the same check guards a single-session filter and both sides of a
    cross-session diff."""
    if agent in names:
        return
    raise SelectionError(
        f"ctxdiff: no agent '{agent}' in {where} — available agents:"
        f"\n{agent_picker(names)}")


def require_single_agent(names: list[str], where: str) -> None:
    """Refuse to compare two sessions that hold SEVERAL agents when no
    `--agent` was given.

    Why this one is ambiguous while a plain `--agent`-less command is not: a
    cross-session diff means "the same agent, in two runs" — the regression
    case. Turn 8 is a global sequence number within each session, so with two
    agents interleaved, turn 8 can be the researcher in one run and the writer
    in the other, and the diff would silently compare unrelated contexts. One
    agent (or none) leaves nothing to get wrong, so no flag is required."""
    if len(names) <= 1:
        return
    raise SelectionError(
        f"ctxdiff: {where} hold {len(names)} agents — pass --agent to pick "
        f"one:\n{agent_picker(names)}")


@dataclass(frozen=True)
class DiffSide:
    """One end of a diff: which session, which agent (None = no filter), which
    turn. A cross-session diff has two sides differing in `session_id`, a
    cross-agent diff two sides differing in `agent`, and an ordinary
    same-session diff two sides differing only in `turn`."""
    session_id: str
    agent: str | None
    turn: int


def diff_scope_line(left: DiffSide, right: DiffSide) -> str | None:
    """The header printed ABOVE a cross-session/cross-agent diff naming what is
    being compared, or None for an ordinary same-session same-agent diff.

    Why None in the ordinary case: `render_turn_diff`'s own header already says
    `turn 7 → turn 8`, and that is the complete truth when both turns come from
    one session and one agent — printing a scope line there would change every
    existing diff's output for no information. It is only when the two sides
    differ in session or agent that `turn 8 → turn 8` becomes actively
    misleading without one.

    Only the dimensions that MATTER are shown: the session appears on both
    sides only when the sides come from different sessions, and the agent only
    when one was selected. So a cross-session diff reads
    `── 4f3a2b1c9d8e · researcher · turn 8  →  9e8d7c6b5a4f · researcher · turn 8 ──`
    and a cross-agent one drops the (identical) session entirely."""
    if left.session_id == right.session_id and left.agent == right.agent:
        return None
    show_session = left.session_id != right.session_id

    def label(side: DiffSide) -> str:
        parts: list[str] = []
        if show_session:
            parts.append(short_id(side.session_id))
        if side.agent:
            parts.append(side.agent)
        parts.append(f"turn {side.turn}")
        return " · ".join(parts)

    return f"── {label(left)}  →  {label(right)} ──"
