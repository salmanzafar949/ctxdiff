"""ctxdiff's command-line entry point. Plain argparse + a small subcommand
registry (stdlib only, per CLAUDE.md's "runtime deps: tiktoken only" rule —
no click) so later milestones can each add one `_add_<name>_parser` function to
`_SUBCOMMANDS` without touching the others.

The command surface is SESSION- and AGENT-aware, because a project store holds
many sessions and each session many agents. Every read command takes the same
four selectors — `--project` (which store), `--session` (which run in it),
`--agent` (which agent's calls), `--turn` (which call) — resolved identically
for all of them by `cli/select.py`, so no two commands can disagree about what
`--session 4f3a2b1c` means. Two discovery commands, `sessions` and `agents`,
exist to tell the user what those selectors can be set to."""
from __future__ import annotations

import argparse
import glob
import importlib.util
import logging
import os
import re
import sys
import tempfile
import webbrowser
from dataclasses import replace
from typing import NamedTuple

from ctxdiff.analyze.cache import analyze_cache
from ctxdiff.analyze.check import Thresholds, analyze_check
from ctxdiff.analyze.differ import diff_calls, diff_turns
from ctxdiff.analyze.evictions import analyze_evictions
from ctxdiff.analyze.tokens import analyze_run, registered_tool_names, usage_totals
from ctxdiff.analyze.window import (
    CONTEXT_WINDOW_ENV,
    ContextWindowError,
    resolve_context_window,
)
from ctxdiff.cli.render import (
    render_agent_summary,
    render_agents_list,
    render_cache_report,
    render_check_report,
    render_run_tokens,
    render_sessions_list,
    render_turn_diff,
    render_usage_summary,
)
from ctxdiff.cli.select import (
    UNLABELED,
    DiffSide,
    SelectionError,
    agents_text,
    choose_session,
    diff_scope_line,
    distinct_agent_names,
    format_local,
    parse_selector,
    require_agent,
    require_single_agent,
    short_id,
)
from ctxdiff.demo import build_demo_trace
# The install hint only — `ctxdiff.mcp` imports no SDK, so this costs nothing on
# a plain install; the server itself is imported lazily inside `_cmd_mcp`.
from ctxdiff.mcp import MISSING_EXTRA_HINT
from ctxdiff.store import config as store_config
from ctxdiff.store.base import Call, EmptyStoreError, Session, Store
from ctxdiff.store.ctrace import CTrace
from ctxdiff.store.sqlite import SQLiteStore
from ctxdiff.viewer import export_html, export_store

_log = logging.getLogger("ctxdiff")


def _find_default_run(cwd: str) -> str | None:
    """Return the most recently modified `*.ctrace` file in `cwd`, or None
    if there isn't one. Backs --project's default so the common case (one
    project DB in the working directory) needs no flag at all."""
    candidates = glob.glob(os.path.join(cwd, "*.ctrace"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


class _NoTraceFound(Exception):
    """No trace to read: no `--project`, no configured backend, and no
    `*.ctrace` in the working directory. Its own type so callers can print the
    friendly "did the run capture?" line for this case while still reporting a
    genuine open FAILURE (corrupt file, unreachable database) as an error."""


def _discovery_backend(project: str | None):
    """The ONE project store a command should read, or None meaning "there is
    no single project here — scan the working directory instead".

    Resolution mirrors the write side's explicit-beats-ambient rule:
    1. `--project VALUE` — always that, whether it is a `.ctrace` path or a
       database DSN. `from_dsn` already knows both spellings (a value with no
       recognizable scheme is a filesystem path), so one flag covers both and a
       path still beats an ambient database, as it always has.
    2. A configured NETWORKED backend (`configure()` / `CTXDIFF_STORE`) —
       detected by the ABSENCE of `path_for`, the same file-backend capability
       check `Tracer` uses, rather than an isinstance chain over every backend.
    3. A configured SQLite backend pointing at a concrete FILE — that file.
    4. Anything else (nothing configured, or a backend naming a whole directory
       of traces, which is not one project) — None.

    Returning None rather than raising is what lets the discovery commands
    (`sessions`, `agents`) list every `.ctrace` in the cwd while the analysis
    commands narrow that to the newest one — see `_resolve_backend`."""
    if project:
        return store_config.from_dsn(project)
    backend = store_config.resolve()
    if backend is not None:
        if not hasattr(backend, "path_for"):
            return backend
        configured_path = getattr(backend, "path", None)
        if configured_path and not os.path.isdir(configured_path):
            return backend
    return None


class _ResolvedBackend(NamedTuple):
    """The store an analysis command will read, plus the path it was DISCOVERED
    at (None whenever the user named the project themselves). `discovered` is
    carried purely so selector errors can name the file — see `_session_where`."""
    backend: object
    discovered: str | None


def _resolve_backend(project: str | None) -> _ResolvedBackend:
    """The store an ANALYSIS command should read: whatever `_discovery_backend`
    resolves, else the most recently modified `*.ctrace` in the cwd — the
    zero-config default that has always made `ctxdiff diff` work with no flags
    in a directory holding one project DB. Raises `_NoTraceFound` when even that
    finds nothing.

    Only the cwd fall-through reports a `discovered` path: it is the one branch
    where the user never said which project this is, so it is the one branch
    whose errors have to."""
    backend = _discovery_backend(project)
    if backend is not None:
        return _ResolvedBackend(backend, None)
    path = _find_default_run(os.getcwd())
    if path is None:
        raise _NoTraceFound()
    return _ResolvedBackend(SQLiteStore(path=path), path)


def _session_where(session_id: str, discovered: str | None) -> str:
    """The scope label a selector error uses for one session: `session
    4f3a2b1c9d8e` normally, and `one.ctrace (session 4f3a2b1c9d8e)` when the
    project was DISCOVERED by scanning the working directory rather than named
    with `--project`.

    Why the filename earns its place exactly there: `ctxdiff agents` lists agents
    from EVERY .ctrace in the directory, so the obvious next command —
    `ctxdiff tokens --agent alpha` — can name an agent that really exists, just
    not in the one file the no-flag default happened to pick. Unqualified, the
    error then names a session short id that appears NOWHERE in the `sessions`
    listing (whose rows are labeled by filename), leaving no hint that a
    different project was chosen or that `--project` is the way out. Naming the
    file turns a dead end into a pointer."""
    where = f"session {short_id(session_id)}"
    return f"{os.path.basename(discovered)} ({where})" if discovered else where


def _html_default_for_backend(backend) -> str | None:
    """The dashboard path `export`/`view` default to for `backend`:
    `<trace-stem>.html` beside a file-backed store, and None for a networked one
    (a database has no filename to borrow, so those commands ask for `--out`)."""
    if not hasattr(backend, "path_for"):
        return None
    path = getattr(backend, "path", None)
    if path and not os.path.isdir(path):
        return _default_html_for(path)
    return None


def _default_html_for(ctrace_path: str) -> str:
    """The dashboard path `export`/`view` default to for a file-backed trace:
    `<trace-stem>.html` right beside the trace — unchanged from when the CLI
    passed the path straight to `export_html`."""
    stem = os.path.splitext(os.path.basename(ctrace_path))[0]
    return os.path.join(os.path.dirname(os.path.abspath(ctrace_path)), f"{stem}.html")


class _SessionView:
    """A read handle PINNED to one session of a (possibly many-session) store.

    Why it exists: every analyzer and the dashboard exporter call
    `get_run()`/`get_calls()` with no arguments and get the store's default
    binding — the NEWEST session. Once `--session` can name any session, those
    calls have to answer for the CHOSEN one instead. Rebinding the underlying
    store is not an option (a `CTrace` binds its run id at open time), so this
    thin forwarder substitutes the chosen session id on the two reads that take
    one and passes everything else straight through.

    It satisfies the same structural read contract the analyzers consume, so
    nothing downstream knows or cares that it is not the store itself."""

    def __init__(self, store: Store, session_id: str):
        """Pin `store` to `session_id`. Takes ownership of the handle: `close()`
        closes the underlying store, so callers keep their existing
        `try/finally: ct.close()` shape unchanged."""
        self._store = store
        self._session_id = session_id

    @property
    def session_id(self) -> str:
        """The session this view reads — used to label cross-session output."""
        return self._session_id

    def get_run(self, session_id: str | None = None):
        """The pinned session's run row (an explicit `session_id` still wins,
        so a caller that knows exactly what it wants is never overridden)."""
        return self._store.get_run(session_id or self._session_id)

    def get_calls(self, session_id: str | None = None):
        """The pinned session's calls, in turn order."""
        return self._store.get_calls(session_id or self._session_id)

    def get_call_blocks(self, call_id: str):
        """One call's blocks — call ids are globally unique, so no pinning."""
        return self._store.get_call_blocks(call_id)

    def list_sessions(self):
        """Every session in the underlying store (pinning scopes reads, not
        discovery)."""
        return self._store.list_sessions()

    def close(self) -> None:
        """Release the underlying store."""
        self._store.close()


class _OpenedProject(NamedTuple):
    """Everything a command needs to choose WITHIN an opened project: the
    reader, its full session list, the id of the session it is bound to by
    default (the newest), the default HTML output path, and the discovered
    project path (None unless the cwd was scanned)."""
    reader: Store
    sessions: list[Session]
    default_id: str
    html_default: str | None
    discovered: str | None


class _OpenedSession(NamedTuple):
    """One project opened and pinned to one session: the view every
    single-session command reads, its default HTML output path, and the
    discovered project path selector errors name."""
    view: _SessionView
    html_default: str | None
    discovered: str | None


def _open_project(project: str | None) -> _OpenedProject:
    """Open the resolved project store and report everything a command needs to
    choose within it.

    The session list is fetched eagerly because ambiguity has to be DETECTED
    before anything is analyzed — a command cannot know whether it may default
    to the newest session without knowing how many there are. That is one extra
    query against a networked store, paid once per command."""
    backend, discovered = _resolve_backend(project)
    reader = backend.open_reader()
    try:
        sessions = reader.list_sessions()
        default_id = reader.get_run().id
    except Exception:
        # Never leak the handle when the follow-up reads fail (dead connection,
        # unreadable file): the caller only gets a reader it can close on
        # success.
        reader.close()
        raise
    return _OpenedProject(reader, sessions, default_id,
                          _html_default_for_backend(backend), discovered)


class _OpenedDashboard(NamedTuple):
    """A project opened for the HTML dashboard: the view pinned to the FOCUS
    session, the default output path, and whether the user actually named that
    session — which is what decides the level the dashboard opens on."""
    view: _SessionView
    html_default: str | None
    session_selected: bool


def _project_agent_names(sessions: list[Session]) -> list[str]:
    """Every NAMED agent anywhere in the project, in first-appearance order,
    read from the session rows the listing already carries — so validating
    `--agent` for the dashboard costs no extra query. Unlabeled calls contribute
    nothing, exactly as `distinct_agent_names` does, because `--agent
    (unlabeled)` is not a value a user types."""
    names: list[str] = []
    for s in sessions:
        for n in s.agents:
            if n not in names:
                names.append(n)
    return names


def _open_dashboard(project: str | None, session: str | None,
                    agent: str | None) -> _OpenedDashboard:
    """Open a project for `export`/`view`, pinned to the session whose detail
    the dashboard focuses on.

    Why this is NOT `_open_session`: every other read command analyzes exactly
    one session, so several sessions and no `--session` is a question it must
    refuse to guess at. The dashboard is the opposite — it exists to show the
    WHOLE project, every agent and every session, and it opens on the agent
    listing precisely when there is more than one thing to choose from. So a
    bare `ctxdiff view` on a many-session project is not ambiguous any more; it
    focuses the newest session's detail and lands the user on level 1, where
    picking is what the page is for.

    `--session` still resolves through `choose_session` (unknown id, ambiguous
    prefix -> the same `SelectionError` with the same listing), and `--agent` is
    checked against the whole project's agents so a typo is caught here rather
    than silently opening on a level scoped to nobody."""
    opened = _open_project(project)
    try:
        chosen = (choose_session(opened.sessions, session, opened.default_id)
                  if session is not None else opened.default_id)
        if agent is not None:
            require_agent(agent, _project_agent_names(opened.sessions),
                          "this project")
    except Exception:
        opened.reader.close()
        raise
    return _OpenedDashboard(_SessionView(opened.reader, chosen),
                            opened.html_default, session is not None)


def _open_session(project: str | None, session: str | None) -> _OpenedSession:
    """Open the resolved project store bound to ONE session — the entry point
    every single-session command (`tokens`, `cache`, `export`, `view`, and the
    ordinary `diff`) uses. Raises `SelectionError` when the session cannot be
    resolved (unknown id, ambiguous prefix, or several sessions and no
    `--session`), always closing the handle on the way out."""
    opened = _open_project(project)
    try:
        chosen = choose_session(opened.sessions, session, opened.default_id)
    except Exception:
        opened.reader.close()
        raise
    return _OpenedSession(_SessionView(opened.reader, chosen),
                          opened.html_default, opened.discovered)


def _report_open_failure(exc: Exception) -> int:
    """Print the right message for a failed store open and return the exit
    code, so every read command reports identically: the friendly
    "did the run capture?" line (exit 1) when there was nothing to open, and
    `ctxdiff: <error>` (exit 1) for a genuine failure.

    The prefix is added only when the message does not ALREADY carry it: most
    errors ctxdiff raises itself are spelled `ctxdiff: ...` so they read
    correctly wherever they surface (a log line, a traceback, a warning), and
    blindly prepending here produced "ctxdiff: ctxdiff: no sessions recorded",
    which reads like a bug in the tool."""
    if isinstance(exc, _NoTraceFound):
        print("no .ctrace here — did the run capture?", file=sys.stderr)
    else:
        message = str(exc)
        if not message.startswith("ctxdiff:"):
            message = f"ctxdiff: {message}"
        print(message, file=sys.stderr)
    return 1


def _report_selection_error(exc: SelectionError) -> int:
    """Print an unresolvable `--session`/`--agent` and return exit code 2.

    Exit 2 (not 1) because this is a USAGE error in argparse's sense — the flags
    as given cannot be acted on — and the message is printed verbatim because
    `select.py` already built it complete with the listing of what could be
    picked instead."""
    print(str(exc), file=sys.stderr)
    return 2


# --- subcommand registration --------------------------------------------------

_PROJECT_HELP = ("project to read: a .ctrace path or a database DSN "
                 "(default: the store configured via CTXDIFF_STORE, else the "
                 "most recently modified *.ctrace in cwd)")

_SESSION_HELP = ("session id, or any unambiguous prefix of one (default: the "
                 "only session; REQUIRED when the project holds several — run "
                 "'ctxdiff sessions' to list them)")

_SESSION_HELP_DIFF = _SESSION_HELP + (". Pass TWICE, optionally as ID:TURN, to "
                                      "diff the same agent across two sessions")

_AGENT_HELP = "restrict to one agent's calls (default: all agents)"

_AGENT_HELP_DIFF = _AGENT_HELP + (". Pass TWICE, optionally as NAME:TURN, to "
                                  "diff two agents within one session")

# The dashboard's own selector help. Both flags are OPTIONAL there and mean
# something different than they do for the analysis commands: the HTML always
# covers the whole project, and these only PRESELECT which of its three levels
# it opens on (see `_open_dashboard`).
_DASHBOARD_SESSION_HELP = ("open the dashboard on this session's turn-by-turn "
                           "view (id or unambiguous prefix; default: land on "
                           "the agent listing and pick there)")

_DASHBOARD_AGENT_HELP = ("open the dashboard scoped to this agent — its session "
                         "list, or its turns when combined with --session "
                         "(default: land on the agent listing)")


# The `--turn` grammar, stated ONCE: optional surrounding whitespace, an
# optional sign, then ASCII decimal digits. `[0-9]` rather than `\d` on purpose
# — see `_turn_int`.
_TURN_RE = re.compile(r"^\s*[+-]?[0-9]+\s*$")


def _turn_int(value: str) -> int:
    """argparse `type=` for every `--turn`: `int()` NARROWED to ASCII digits.

    Why not plain `type=int`: `int()` accepts every Unicode decimal digit, so
    `--turn ٢` (ARABIC-INDIC DIGIT TWO) quietly means 2 — while the `:TURN`
    selector suffix in `select.parse_selector` already requires ASCII (it guards
    with `tail.isascii()`) and the JS CLI's `--turn` requires ASCII too. One flag
    accepting a spelling the selector beside it rejects is a difference nobody
    can predict, so the grammar is narrowed here to the one both sides share.

    Failures are raised as `ArgumentTypeError` carrying argparse's OWN wording
    for a bad int (`invalid int value: '...'`) rather than being left to argparse
    to phrase: it would otherwise name this function in the message
    (`invalid _turn_int value: ...`), leaking an internal name into user-facing
    output that the JS CLI has to match."""
    if not _TURN_RE.match(value):
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}")
    return int(value)


# The `--max-*-pct` grammar: optional whitespace, optional sign, then ASCII
# decimal digits with an optional fractional part (or a bare fraction). Narrower
# than `float()` on purpose — `float()` also accepts `inf`, `nan`, `1e4`,
# underscores and non-ASCII decimal digits, none of which are a percentage
# anyone means to type, and all of which the JS twin's parser would have to
# reproduce exactly for the two CLIs to reject the same strings.
_FLOAT_RE = re.compile(r"^\s*[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)\s*$")


def _pct_float(value: str) -> float:
    """argparse `type=` for every percentage flag: `float()` NARROWED to plain
    ASCII decimals (see `_FLOAT_RE`). Failures carry argparse's OWN wording for
    a bad float (`invalid float value: '...'`) rather than naming this function,
    exactly as `_turn_int` does for ints — an internal name in user-facing
    output is a difference the JS CLI would have to mirror for no benefit."""
    if not _FLOAT_RE.match(value):
        raise argparse.ArgumentTypeError(f"invalid float value: {value!r}")
    return float(value)


# The `--context-window` help, stated ONCE and shared by `tokens`, `view` and
# `export` (`check` keeps its own wording, because there the window is not a
# display denominator but the thing `--max-context-pct` is asserted against).
_CONTEXT_WINDOW_HELP = (
    "the model's context window, in tokens — the denominator every percentage "
    f"is taken against (default: ${CONTEXT_WINDOW_ENV}, if set). Yours to "
    "supply: ctxdiff ships no model→window table by design, so with no window "
    "no percentage is shown")


def _add_context_window_flag(p: argparse.ArgumentParser) -> None:
    """Register `--context-window N` on a command that DISPLAYS percentages.

    Kept as one function so `tokens`, `view` and `export` cannot drift apart in
    spelling, type or help text, and so every one of them feeds the same
    `_resolve_window` — the flag and the environment variable are resolved in
    exactly one place for the whole CLI (see `analyze.window`)."""
    p.add_argument("--context-window", type=_turn_int, default=None, metavar="N",
                   help=_CONTEXT_WINDOW_HELP)


def _resolve_window(args: argparse.Namespace) -> int | None:
    """The context window this invocation should render percentages against:
    `--context-window` if typed, else `CTXDIFF_CONTEXT_WINDOW`, else None.

    An unusable environment value is re-raised as a `SelectionError` so it
    reports as the usage error it is (exit 2, message printed verbatim) rather
    than as a failure to read a trace — the variable is part of how the command
    was invoked, not part of what it was pointed at."""
    try:
        return resolve_context_window(args.context_window)
    except ContextWindowError as exc:
        raise SelectionError(str(exc)) from exc


def _add_project_flags(p: argparse.ArgumentParser) -> None:
    """Register `--project` plus its hidden `--run` alias. Both write the same
    `project` dest, so every existing `--run PATH` invocation and script keeps
    working unchanged while help only documents the name that is true now (a
    `.ctrace` is a PROJECT holding many runs, not one run)."""
    p.add_argument("--project", default=None, help=_PROJECT_HELP)
    p.add_argument("--run", dest="project", default=None, help=argparse.SUPPRESS)


def _add_selector_flags(p: argparse.ArgumentParser, *,
                        repeatable: bool = False) -> None:
    """Register the selectors shared by every analysis command: `--project`
    (+ `--run`), `--session`, `--agent`. `repeatable` makes the last two
    `append` actions — only `diff` needs that, to name the two sides of a
    cross-session or cross-agent comparison."""
    _add_project_flags(p)
    if repeatable:
        p.add_argument("--session", action="append", dest="sessions",
                       default=None, help=_SESSION_HELP_DIFF)
        p.add_argument("--agent", action="append", dest="agents",
                       default=None, help=_AGENT_HELP_DIFF)
    else:
        p.add_argument("--session", default=None, help=_SESSION_HELP)
        p.add_argument("--agent", default=None, help=_AGENT_HELP)


def _add_diff_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register `ctxdiff diff`. `--turn`, `--session` and `--agent` all use
    action='append' so each can be passed twice; `_cmd_diff` validates the
    resulting combination itself, since argparse alone cannot express "this
    flag exactly twice" or "these two flags are alternatives"."""
    p = subparsers.add_parser(
        "diff", help="git-style block diff between two turns")
    p.add_argument("--turn", action="append", type=_turn_int, dest="turns",
                   help="turn (call seq) number; pass exactly twice, e.g. "
                        "--turn 7 --turn 8")
    _add_selector_flags(p, repeatable=True)
    p.set_defaults(func=_cmd_diff)


def _add_tokens_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register `ctxdiff tokens`. `--turn` is a single optional int (unlike
    `diff`'s `--turn` twice) — omitted, every turn in the session is shown;
    given, output is limited to that one turn."""
    p = subparsers.add_parser(
        "tokens", help="token heatmap + schema-bloat report")
    p.add_argument("--turn", type=_turn_int, default=None,
                    help="limit output to one turn (call seq) number")
    _add_context_window_flag(p)
    _add_selector_flags(p)
    p.set_defaults(func=_cmd_tokens)


def _add_cache_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register `ctxdiff cache`: no `--turn`, because the profiler works on
    consecutive-call PAIRS — a single turn has no pair to be stable against."""
    p = subparsers.add_parser(
        "cache", help="prefix-stability report + wasted-spend estimate")
    _add_selector_flags(p)
    p.set_defaults(func=_cmd_cache)


def _add_check_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register `ctxdiff check`: the CI gate. Every assertion is OPT-IN — the
    command asserts exactly what it was asked to and nothing else — and no
    assertion at all is a usage error rather than a pass (see `_check_thresholds`).

    No `--turn`: a budget is a property of a whole run, and a check scoped to a
    single turn could not answer "did any turn blow the budget", which is the
    question. `--agent` still narrows the run, so one pipeline can gate each of
    its agents against a different budget."""
    p = subparsers.add_parser(
        "check", help="assert context budgets and fail the build (CI gate)")
    p.add_argument("--max-context", type=_turn_int, default=None, metavar="N",
                   help="fail if any turn's context exceeds N tokens (the same "
                        "total 'ctxdiff tokens' prints for that turn)")
    p.add_argument("--context-window", type=_turn_int, default=None, metavar="N",
                   help="the model's context window, in tokens — the denominator "
                        "--max-context-pct is a percentage of. Required by it, and "
                        "supplied by you: ctxdiff ships no model→window table")
    p.add_argument("--max-context-pct", type=_pct_float, default=None, metavar="P",
                   help="fail if any turn's context exceeds P%% of --context-window")
    p.add_argument("--require-stable-prefix", action="store_true",
                   help="fail if the prompt-cache prefix breaks anywhere in the run")
    p.add_argument("--no-dead-schemas", action="store_true",
                   help="fail if a tool schema is registered but never invoked")
    p.add_argument("--no-tagged-eviction", action="store_true",
                   help="fail if a block you tagged with tracer.tag() entered "
                        "an agent's context and later left it for good")
    p.add_argument("--max-growth", type=_turn_int, default=None, metavar="N",
                   help="fail if the context grows by more than N tokens between "
                        "two consecutive turns of the same agent")
    p.add_argument("--max-growth-pct", type=_pct_float, default=None, metavar="P",
                   help="fail if the context grows by more than P%% between two "
                        "consecutive turns of the same agent")
    _add_selector_flags(p)
    p.set_defaults(func=_cmd_check)


def _add_sessions_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register `ctxdiff sessions`: the session picker's data source — what
    `--session` can be set to, with the local start time and agent list needed
    to tell one run from another."""
    p = subparsers.add_parser(
        "sessions", help="list the sessions ctxdiff can see")
    _add_project_flags(p)
    p.set_defaults(func=_cmd_sessions)


def _add_runs_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register `runs` as a HIDDEN alias of `sessions`. The command was renamed
    once a `.ctrace` became a project holding many runs, but every existing
    script, README and muscle memory says `runs` — so it keeps working, forever,
    silently.

    Hidden in two steps, because argparse has no single switch for it: passing
    no `help` keeps it out of the per-command description list, and the
    subparsers' explicit `metavar` (see `_build_parser`) keeps it out of the
    `{diff,tokens,...}` choices line. `help=argparse.SUPPRESS` does NOT work
    here — the subparser formatter prints the sentinel literally."""
    p = subparsers.add_parser("runs")
    _add_project_flags(p)
    p.set_defaults(func=_cmd_sessions)


def _add_agents_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register `ctxdiff agents`: the agent picker's data source — every agent
    in the project with its footprint aggregated across ALL sessions, which is
    the scale at which an agent's cost is actually a fact."""
    p = subparsers.add_parser(
        "agents", help="list every agent in the project, across all sessions")
    _add_project_flags(p)
    p.set_defaults(func=_cmd_agents)


def _add_export_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register `ctxdiff export [--out FILE.html]`: emit the self-contained HTML
    dashboard for the whole project. `--out` overrides the destination; by
    default the file is written as `<trace-stem>.html` beside the trace."""
    p = subparsers.add_parser(
        "export", help="write a self-contained HTML dashboard for the project")
    _add_project_flags(p)
    p.add_argument("--session", default=None, help=_DASHBOARD_SESSION_HELP)
    p.add_argument("--agent", default=None, help=_DASHBOARD_AGENT_HELP)
    _add_context_window_flag(p)
    p.add_argument("--out", default=None,
                    help="output .html path (default: <trace-stem>.html next to the trace)")
    p.set_defaults(func=_cmd_export)


def _add_demo_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register `ctxdiff demo [--out FILE] [--no-open] [--keep]`: the
    zero-friction first run — build a sample multi-agent `.ctrace` (no API
    keys, no network, no agent to wire up) and open its dashboard. `--out`
    picks a permanent `.ctrace` path (implying `--keep`'s no-tempfile
    behavior); `--keep` alone writes to a fixed `./ctxdiff-demo.{ctrace,html}`
    pair instead of a tempfile; `--no-open` skips the browser launch (CI/
    headless/screenshotting) but still writes and prints both paths."""
    p = subparsers.add_parser(
        "demo", help="build a sample multi-agent trace and open its dashboard "
                     "— no API keys, no setup")
    p.add_argument("--out", default=None,
                   help="write the demo trace to FILE.ctrace (and FILE.html "
                        "beside it) instead of a tempfile; implies --keep")
    p.add_argument("--no-open", action="store_true", dest="no_open",
                   help="write and print both paths but do not open a browser")
    p.add_argument("--keep", action="store_true",
                   help="write to ./ctxdiff-demo.ctrace + .html in the "
                        "current directory instead of a tempfile")
    p.set_defaults(func=_cmd_demo)


def _add_mcp_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register `ctxdiff mcp`: expose the analyzers to a coding agent over the
    Model Context Protocol, on stdio.

    Two flags and no more, both of them the OPERATOR's decisions rather than the
    connected model's (see `mcp/server.py`): `--runs-dir`, because an MCP
    server's working directory is whatever the editor launched it from and the
    newest-.ctrace-in-cwd default every other command uses is meaningless there;
    and `--redact`, the never-return-raw-text mode for anyone who does not want
    recorded prompt content reaching a cloud model."""
    p = subparsers.add_parser(
        "mcp", help="serve ctxdiff to a coding agent over MCP (stdio)")
    p.add_argument("--runs-dir", dest="runs_dir", default=None, metavar="DIR",
                   help="directory of .ctrace files to serve (default: the "
                        "store configured via CTXDIFF_STORE, else the current "
                        "directory — which for a client-launched server is "
                        "rarely where your traces are, so set this)")
    p.add_argument("--redact", action="store_true",
                   help="never return captured text — labels, content hashes, "
                        "token counts and structure only, including from "
                        "ctxdiff_block. Use when the MCP client's model is "
                        "remote and your traces hold sensitive prompts")
    p.set_defaults(func=_cmd_mcp)


def _add_view_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register `ctxdiff view [--no-open]`: export the project's dashboard to a
    temp file and open it in the default browser. `--no-open` skips the browser
    launch (used by tests and headless/CI environments) but still writes and
    prints the file path."""
    p = subparsers.add_parser(
        "view", help="open a self-contained HTML dashboard in your browser")
    _add_project_flags(p)
    p.add_argument("--session", default=None, help=_DASHBOARD_SESSION_HELP)
    p.add_argument("--agent", default=None, help=_DASHBOARD_AGENT_HELP)
    _add_context_window_flag(p)
    p.add_argument("--no-open", action="store_true", dest="no_open",
                    help="write and print the HTML path but do not open a browser")
    p.set_defaults(func=_cmd_view)


# Every registered subcommand's add_parser function, in help-listing order.
# Appending here is the ONLY change a new subcommand needs. `runs` is in the
# list but registers itself with help=SUPPRESS, so it dispatches without
# appearing.
_SUBCOMMANDS = [_add_diff_parser, _add_tokens_parser, _add_cache_parser,
                _add_check_parser, _add_sessions_parser, _add_agents_parser,
                _add_runs_parser, _add_export_parser, _add_view_parser,
                _add_demo_parser, _add_mcp_parser]

# The commands help ADVERTISES, in listing order. `runs` is deliberately absent:
# it still dispatches (see `_add_runs_parser`), it just isn't offered to anyone
# reading the help for the first time.
_VISIBLE_COMMANDS = ["diff", "tokens", "cache", "check", "sessions", "agents",
                     "export", "view", "demo", "mcp"]


def _build_parser() -> argparse.ArgumentParser:
    """Assemble the top-level parser and register every subcommand. The
    subparsers' `metavar` is set explicitly from `_VISIBLE_COMMANDS` rather than
    left to argparse, which would enumerate every registered choice and so
    re-expose the hidden `runs` alias in the usage line."""
    parser = argparse.ArgumentParser(
        prog="ctxdiff", description="git diff for your agent's context window")
    subparsers = parser.add_subparsers(
        dest="command", metavar="{" + ",".join(_VISIBLE_COMMANDS) + "}")
    for register in _SUBCOMMANDS:
        register(subparsers)
    return parser


# --- diff ----------------------------------------------------------------------


def _side_turn(selector, index: int, turns: list[int]) -> int | None:
    """Which turn one side of a cross diff should read, in priority order: the
    side's own `:TURN` suffix; else the matching one of two `--turn` flags; else
    a single `--turn` applied to BOTH sides (the common regression shape —
    "turn 8 in the good run vs turn 8 in the bad one"). None means the user
    named no turn for this side, which is a usage error the caller reports."""
    if selector.turn is not None:
        return selector.turn
    if len(turns) == 2:
        return turns[index]
    if len(turns) == 1:
        return turns[0]
    return None


def _side_blocks(reader: Store, side: DiffSide, discovered: str | None = None):
    """Load the ordered blocks of the ONE call a diff side names, validating
    that the turn exists (and belongs to the named agent) first. Raises
    ValueError with a session-qualified message — a cross diff spans two
    sessions, so "turn 8 not found" without saying WHERE is unusable, and it
    names the discovered file too when the project was never named (see
    `_session_where`)."""
    calls = reader.get_calls(side.session_id)
    where = _session_where(side.session_id, discovered)
    if side.agent is not None:
        owned = [c for c in calls if c.agent == side.agent]
        match = [c for c in owned if c.seq == side.turn]
        if not match:
            raise ValueError(
                f"{where}: turn {side.turn} is not a call of agent "
                f"'{side.agent}' (that agent's turns: {[c.seq for c in owned]})")
    else:
        match = [c for c in calls if c.seq == side.turn]
        if not match:
            raise ValueError(
                f"{where}: turn {side.turn} not found "
                f"(available turns: {sorted(c.seq for c in calls)})")
    return reader.get_call_blocks(match[0].id)


def _cross_session_sides(reader: Store, sessions: list[Session], default_id: str,
                         session_sels, agent_sels, turns,
                         discovered: str | None = None) -> list[DiffSide]:
    """Build both sides of a CROSS-SESSION diff — the regression case: the same
    agent, two runs. Resolves each `--session` value to a real session id, then
    settles the agent ONCE for both sides, since "the same agent in two runs" is
    the entire premise: an explicit `--agent` is validated against both
    sessions, and with none given the agent is inferred only when it cannot be
    wrong (exactly one agent, or none at all) — otherwise the user is asked."""
    ids = [choose_session(sessions, s.name, default_id) for s in session_sels]
    per_session_agents = [distinct_agent_names(reader.get_calls(i)) for i in ids]

    agent = agent_sels[0].name if agent_sels else None
    if agent is None:
        combined: list[str] = []
        for names in per_session_agents:
            for n in names:
                if n not in combined:
                    combined.append(n)
        require_single_agent(combined, "these sessions")
        agent = combined[0] if combined else None
    else:
        for session_id, names in zip(ids, per_session_agents):
            require_agent(agent, names, _session_where(session_id, discovered))

    return [_make_side(ids[i], agent, session_sels[i], i, turns, "session")
            for i in (0, 1)]


def _cross_agent_sides(reader: Store, sessions: list[Session], default_id: str,
                       session_sels, agent_sels, turns,
                       discovered: str | None = None) -> list[DiffSide]:
    """Build both sides of a CROSS-AGENT diff: two agents, one session — "why
    does the writer see a different context than the researcher?". The session
    is resolved exactly as for any single-session command (so it may be omitted
    when the project holds one), and both agent names are validated against
    that session before any turn is looked up."""
    wanted = session_sels[0].name if session_sels else None
    session_id = choose_session(sessions, wanted, default_id)
    names = distinct_agent_names(reader.get_calls(session_id))
    for sel in agent_sels:
        require_agent(sel.name, names, _session_where(session_id, discovered))
    return [_make_side(session_id, agent_sels[i].name, agent_sels[i], i, turns,
                       "agent")
            for i in (0, 1)]


def _make_side(session_id: str, agent: str | None, selector, index: int,
               turns: list[int], axis: str) -> DiffSide:
    """Assemble one `DiffSide`, turning "no turn was named for this side" into
    a `SelectionError` (exit 2) that spells out both ways to name one."""
    turn = _side_turn(selector, index, turns)
    if turn is None:
        raise SelectionError(
            f"ctxdiff: each side of a cross-{axis} diff needs a turn — pass "
            f"--{axis} VALUE:TURN twice, or --turn N --turn M")
    return DiffSide(session_id=session_id, agent=agent, turn=turn)


def _cmd_diff(args: argparse.Namespace) -> int:
    """Implements `ctxdiff diff` in its three shapes, dispatched purely on how
    many times `--session`/`--agent` were passed:

    - twice `--session` -> CROSS-SESSION (same agent, two runs);
    - twice `--agent`   -> CROSS-AGENT (two agents, one run);
    - otherwise         -> the ordinary two-turn diff within one session.

    Everything an argparse action cannot express is checked here: at most two
    of each flag, never both axes at once, and a `:TURN` suffix only where it
    means something. All of those are usage errors (exit 2)."""
    session_sels = [parse_selector(v) for v in (args.sessions or [])]
    agent_sels = [parse_selector(v) for v in (args.agents or [])]
    turns = args.turns or []

    if len(session_sels) > 2 or len(agent_sels) > 2:
        print("ctxdiff diff compares exactly two sides: pass --session (or "
              "--agent) at most twice", file=sys.stderr)
        return 2
    if len(session_sels) == 2 and len(agent_sels) == 2:
        print("ctxdiff diff compares along ONE axis: pass two --session values "
              "(cross-session) or two --agent values (cross-agent), not both",
              file=sys.stderr)
        return 2

    cross_session = len(session_sels) == 2
    cross_agent = len(agent_sels) == 2
    if not (cross_session or cross_agent):
        if any(s.turn is not None for s in session_sels + agent_sels):
            print("ctxdiff diff: a ':TURN' suffix only means something on a "
                  "cross-session or cross-agent diff — use --turn N --turn M "
                  "here", file=sys.stderr)
            return 2
        if len(turns) != 2:
            print("ctxdiff diff requires exactly two --turn flags, e.g. "
                  "'ctxdiff diff --turn 7 --turn 8'", file=sys.stderr)
            return 2
        return _diff_within_session(args, session_sels, agent_sels, turns)

    if len(turns) > 2:
        print("ctxdiff diff accepts at most two --turn flags", file=sys.stderr)
        return 2
    return _diff_cross(args, session_sels, agent_sels, turns,
                       cross_session=cross_session)


def _diff_within_session(args: argparse.Namespace, session_sels, agent_sels,
                         turns: list[int]) -> int:
    """The ordinary diff: two turns of ONE session, optionally filtered to one
    agent. Unchanged in behavior from before sessions existed — same ownership
    check, same messages, same exit codes — except that the session it reads is
    now chosen rather than assumed."""
    wanted = session_sels[0].name if session_sels else None
    agent = agent_sels[0].name if agent_sels else None
    try:
        ct, _, discovered = _open_session(args.project, wanted)
    except SelectionError as exc:
        return _report_selection_error(exc)
    except Exception as exc:  # noqa: BLE001 — any open failure is reported, not crashed
        return _report_open_failure(exc)

    try:
        try:
            _check_agent_filter(ct, agent, discovered)
        except SelectionError as exc:
            return _report_selection_error(exc)
        turn_old, turn_new = turns
        # --agent: validate BOTH turns belong to that agent before diffing.
        # Turns stay global seq numbers (what every view shows); we only check
        # ownership, then diff_turns resolves them against the whole session as
        # usual (the two calls happen to be this agent's).
        if agent is not None:
            agent_seqs = [c.seq for c in ct.get_calls() if c.agent == agent]
            bad = [t for t in (turn_old, turn_new) if t not in agent_seqs]
            if bad:
                print(f"ctxdiff: turn(s) {bad} are not calls of agent "
                      f"'{agent}' (that agent's turns: {agent_seqs})",
                      file=sys.stderr)
                return 1
        try:
            diff = diff_turns(ct, turn_old, turn_new)
        except ValueError as exc:
            print(f"ctxdiff: {exc}", file=sys.stderr)
            return 1
        print(render_turn_diff(diff))
    finally:
        ct.close()
    return 0


def _diff_cross(args: argparse.Namespace, session_sels, agent_sels,
                turns: list[int], *, cross_session: bool) -> int:
    """The cross-session / cross-agent diff. Reuses `diff_calls` verbatim — the
    differ aligns two ordered block lists by content hash and has never cared
    where they came from, so comparing turn 8 of a good run against turn 8 of a
    bad one is the SAME operation as comparing two turns of one run. All this
    adds is resolving which two calls those are, and a scope header naming them
    (without which two `turn 8`s in one header would be indistinguishable)."""
    try:
        reader, sessions, default_id, _, discovered = _open_project(args.project)
    except Exception as exc:  # noqa: BLE001 — any open failure is reported, not crashed
        return _report_open_failure(exc)

    try:
        builder = _cross_session_sides if cross_session else _cross_agent_sides
        try:
            left, right = builder(reader, sessions, default_id, session_sels,
                                  agent_sels, turns, discovered)
        except SelectionError as exc:
            return _report_selection_error(exc)
        try:
            old_blocks = _side_blocks(reader, left, discovered)
            new_blocks = _side_blocks(reader, right, discovered)
        except ValueError as exc:
            print(f"ctxdiff: {exc}", file=sys.stderr)
            return 1
        diff = diff_calls(old_blocks, new_blocks, seq_old=left.turn,
                          seq_new=right.turn)
        scope = diff_scope_line(left, right)
        if scope:
            print(scope)
        print(render_turn_diff(diff))
    finally:
        reader.close()
    return 0


# --- tokens / cache -------------------------------------------------------------


def _check_agent_filter(ct: _SessionView, agent: str | None,
                        discovered: str | None = None) -> None:
    """Reject an `--agent` naming nobody in the session under analysis, listing
    the agents that ARE there.

    Why this is worth a query: without it a typo'd or stale agent name filters
    every call away and the command cheerfully reports "no calls in this run" —
    technically true, actively misleading, and exit 0, so a CI check built on it
    would pass forever. Raising `SelectionError` turns it into what it is: a bad
    flag (exit 2) with the correct values printed underneath."""
    if agent is None:
        return
    require_agent(agent, distinct_agent_names(ct.get_calls()),
                  _session_where(ct.session_id, discovered))


def _cmd_tokens(args: argparse.Namespace) -> int:
    """Implements `ctxdiff tokens`. How: resolves and opens ONE session of the
    project (unresolvable selector -> exit 2 with the listing; nothing
    found/unreadable -> exit 1), runs the token attributor over it, then narrows
    to one turn when `--turn` is given (missing turn -> exit 1 with a message,
    mirroring `diff`'s behavior). When there IS unused-schema bloat to report, a
    second pass over the session's blocks derives "M" (how many tools are
    registered in total) for the "N of M" bloat message — cheap relative to the
    session size, and only paid when there's something to report.

    Two things beyond the attribution itself: the context window is resolved
    FIRST (before any store is opened, so a bad `CTXDIFF_CONTEXT_WINDOW` fails
    the same way with or without a trace present), and the tagged-eviction
    detector runs over the same session so a block the developer marked as
    load-bearing that silently left the context is named right under the turn
    that lost it."""
    try:
        context_window = _resolve_window(args)
    except SelectionError as exc:
        return _report_selection_error(exc)

    try:
        ct, _, discovered = _open_session(args.project, args.session)
    except SelectionError as exc:
        return _report_selection_error(exc)
    except Exception as exc:  # noqa: BLE001 — any open failure is reported, not crashed
        return _report_open_failure(exc)

    try:
        try:
            _check_agent_filter(ct, args.agent, discovered)
        except SelectionError as exc:
            return _report_selection_error(exc)
        run_tokens = analyze_run(ct, agent=args.agent)
        calls_by_seq = {c.seq: c for c in run_tokens.calls}

        if args.turn is not None:
            if args.turn not in calls_by_seq:
                where = (f"agent '{args.agent}'" if args.agent is not None
                         else "this run")
                print(f"ctxdiff: turn {args.turn} not found in {where} "
                      f"(available turns: {sorted(calls_by_seq)})", file=sys.stderr)
                return 1
            selected = [calls_by_seq[args.turn]]
        else:
            selected = run_tokens.calls

        total_tools = None
        if run_tokens.bloat is not None and run_tokens.bloat.unused_tools:
            all_blocks = [ct.get_call_blocks(c.id) for c in ct.get_calls()]
            total_tools = len(registered_tool_names(all_blocks))

        # The provider-usage rollup prints first (always — it self-describes as
        # "no provider usage reported" when nothing reported); the per-agent
        # block-token summary is only non-None for an unfiltered multi-agent run.
        usage_summary = render_usage_summary(run_tokens.usage, args.agent)
        agent_summary = render_agent_summary(run_tokens)
        # The detector always runs over the whole session — an eviction is a fact
        # about a PAIR of turns, so it cannot be computed from one turn's blocks
        # — but under `--turn N` the report shown beside it covers exactly that
        # turn, and a stanza saying "…was evicted at turn 3" under a turn-1
        # report names something the reader did not ask about and cannot see
        # above it. So the events are filtered to the turn that LOST the block.
        evictions = analyze_evictions(ct, agent=args.agent)
        if args.turn is not None:
            evictions = replace(evictions, evictions=[
                e for e in evictions.evictions if e.evicted_seq == args.turn])
        print(render_run_tokens(selected, run_tokens.bloat, total_tools,
                                agent_summary, usage_summary,
                                context_window, evictions))
    finally:
        ct.close()
    return 0


def _cmd_cache(args: argparse.Namespace) -> int:
    """Implements `ctxdiff cache`. How: resolves and opens ONE session
    (unresolvable selector -> exit 2, nothing found/unreadable -> exit 1, same
    convention as `diff`/`tokens`), runs the cache-prefix profiler over it, and
    prints the rendered report. Breaks found are informational (not a usage/
    operational failure), so a clean run and a run full of warnings both exit 0
    — same convention as `tokens`' schema-bloat warning."""
    try:
        ct, _, discovered = _open_session(args.project, args.session)
    except SelectionError as exc:
        return _report_selection_error(exc)
    except Exception as exc:  # noqa: BLE001 — any open failure is reported, not crashed
        return _report_open_failure(exc)

    try:
        try:
            _check_agent_filter(ct, args.agent, discovered)
        except SelectionError as exc:
            return _report_selection_error(exc)
        report = analyze_cache(ct, agent=args.agent)
        print(render_cache_report(report))
    finally:
        ct.close()
    return 0


# --- check ------------------------------------------------------------------------

# The assertion flags, in the order the "you asked for nothing" error lists
# them — the same order the report prints its rows in, so the message reads as
# a menu of what the command can do rather than an arbitrary list.
_ASSERTION_FLAGS = ("--max-context", "--max-context-pct", "--require-stable-prefix",
                    "--no-dead-schemas", "--no-tagged-eviction", "--max-growth",
                    "--max-growth-pct")


def _check_thresholds(args: argparse.Namespace) -> Thresholds:
    """Turn `check`'s flags into a `Thresholds`, rejecting every combination
    that cannot mean what it says. Raises `SelectionError` (exit 2) — these are
    usage errors in argparse's sense: the flags as given cannot be acted on.

    The four rules, and why each is an error rather than a shrug:

    1. NO assertion at all is refused. `ctxdiff check` with no thresholds would
       otherwise exit 0 having verified nothing — a green tick in CI that means
       "nobody asked a question", which is the exact failure this command
       exists to prevent.
    2. `--max-context-pct` with NO window at all is refused, because the
       percentage has no denominator. ctxdiff deliberately ships no
       model→window table (windows differ per model and per provider and change
       under you; a stale table would silently move everyone's threshold), so
       the window is the user's to supply — as `--context-window N` or as
       `CTXDIFF_CONTEXT_WINDOW`, resolved by the ONE `_resolve_window` path
       `ctxdiff tokens` and the dashboard use, so a gate and the report beside
       it can never be scored against two different windows.
    3. The `--context-window` FLAG without `--max-context-pct` is refused too —
       nothing would consume it, and silently ignoring a flag someone typed is
       how a CI gate ends up asserting less than its author believes. Only the
       flag: an inherited `CTXDIFF_CONTEXT_WINDOW` is ambient configuration for
       the whole shell, and failing a `--max-context` check because the
       environment happens to know the window would be absurd.
    4. A limit outside its legal range is refused: the two absolute budgets and
       the percentage must be POSITIVE (a zero context window is a division by
       zero, and a zero-token budget is not an assertion anyone means), while
       the two growth limits may legitimately be zero — "this context must not
       grow at all" is a real thing to assert."""
    # Rule 4 for the WINDOW FLAG is stated here, before the shared resolver gets
    # a chance to state it in its own words. `resolve_context_window` refuses a
    # non-positive flag too (so `tokens`/`view`/`export` inherit the rule), but
    # its message is the generic `ctxdiff:` one; every usage error `check` emits
    # is prefixed `ctxdiff check:`, which is what tells a CI log which step
    # spoke. Only the flag needs the early guard: an environment-supplied window
    # is already positive by the time `parse_context_window` returns it.
    if args.context_window is not None and args.context_window <= 0:
        raise SelectionError("ctxdiff check: --context-window must be greater "
                             f"than 0 (got {args.context_window})")
    context_window = _resolve_window(args)
    # Built first so "was anything asked for?" is answered by the value object's
    # own `any_requested` — the single definition of that question, shared with
    # the analyzer instead of restated as a second `any([...])` that could fall
    # out of step with it the next time a flag is added.
    thresholds = Thresholds(
        max_context=args.max_context,
        context_window=context_window,
        max_context_pct=args.max_context_pct,
        require_stable_prefix=args.require_stable_prefix,
        no_dead_schemas=args.no_dead_schemas,
        no_tagged_eviction=args.no_tagged_eviction,
        max_growth=args.max_growth,
        max_growth_pct=args.max_growth_pct,
    )
    if not thresholds.any_requested:
        raise SelectionError(
            "ctxdiff check: nothing to assert — pass at least one of "
            + ", ".join(_ASSERTION_FLAGS))
    if args.max_context_pct is not None and context_window is None:
        raise SelectionError(
            "ctxdiff check: --max-context-pct needs a denominator — pass "
            "--context-window N (ctxdiff ships no model→window table by design, "
            "so the window is yours to state)")
    if args.context_window is not None and args.max_context_pct is None:
        raise SelectionError(
            "ctxdiff check: --context-window is only used by --max-context-pct "
            "— pass that too, or use --max-context N for an absolute budget")
    # Integer and percentage limits are reported back in their own spelling —
    # a bare integer for the token budgets, one decimal for the percentages,
    # the same way every percentage in the report itself is written — so the
    # error echoes what the user typed without either CLI having to reproduce
    # the other's general-purpose float formatting.
    if args.max_context is not None and args.max_context <= 0:
        raise SelectionError("ctxdiff check: --max-context must be greater "
                             f"than 0 (got {args.max_context})")
    if args.max_context_pct is not None and args.max_context_pct <= 0:
        raise SelectionError("ctxdiff check: --max-context-pct must be greater "
                             f"than 0 (got {args.max_context_pct:.1f})")
    if args.max_growth is not None and args.max_growth < 0:
        raise SelectionError("ctxdiff check: --max-growth cannot be negative "
                             f"(got {args.max_growth})")
    if args.max_growth_pct is not None and args.max_growth_pct < 0:
        raise SelectionError("ctxdiff check: --max-growth-pct cannot be negative "
                             f"(got {args.max_growth_pct:.1f})")
    return thresholds


def _cmd_check(args: argparse.Namespace) -> int:
    """Implements `ctxdiff check`: assert a context budget, and exit non-zero
    when it is blown so CI fails the build.

    Exit codes, matching the convention every other subcommand follows —
    **0** every requested assertion passed; **1** at least one was violated (or
    the trace could not be read, or the session held nothing to check); **2** a
    usage error (an impossible flag combination, an unresolvable
    `--session`/`--agent`).

    Order matters: the flags are validated BEFORE any store is opened, so a
    malformed check fails the same way whether or not a trace happens to exist
    — otherwise the same command would report a flag problem on one machine and
    a missing `.ctrace` on another.

    An EMPTY session (or an `--agent` slice with no calls) exits 1 rather than
    reporting a table of vacuous passes: a check that examined zero turns has
    proved nothing, and a CI gate that greens on "there was no trace" is worse
    than no gate at all — it would keep passing forever the day capture
    silently broke."""
    try:
        thresholds = _check_thresholds(args)
    except SelectionError as exc:
        return _report_selection_error(exc)

    try:
        ct, _, discovered = _open_session(args.project, args.session)
    except SelectionError as exc:
        return _report_selection_error(exc)
    except Exception as exc:  # noqa: BLE001 — any open failure is reported, not crashed
        return _report_open_failure(exc)

    try:
        try:
            _check_agent_filter(ct, args.agent, discovered)
        except SelectionError as exc:
            return _report_selection_error(exc)
        report = analyze_check(ct, thresholds, agent=args.agent)
        if report.turns_analyzed == 0:
            where = (f"agent '{args.agent}'" if args.agent is not None
                     else "this session")
            print(f"ctxdiff check: no calls in {where} — nothing to check "
                  f"(did the run capture?)", file=sys.stderr)
            return 1
        # The header names the session (and, when the project was DISCOVERED
        # rather than named, the file it came from) with the same
        # `_session_where` spelling every selector error uses — so a CI log
        # records which trace produced the verdict, and the newest-.ctrace
        # default can never green a build on a file nobody meant to check.
        print(render_check_report(
            report, _session_where(ct.session_id, discovered)))
        return 0 if report.passed else 1
    finally:
        ct.close()


# --- discovery: sessions / agents ------------------------------------------------

_EMPTY_CWD = "no .ctrace files in the current directory"
_EMPTY_PROJECT = "no sessions in this project"
_EMPTY_STORE = "no sessions in the configured store"


def _file_session_rows(filename: str, sessions: list[Session]):
    """Turn one `.ctrace` file's sessions into `sessions`-table rows. The label
    is the bare filename when the file holds ONE session — the overwhelmingly
    common case, and what a user recognizes — and `<filename>#<short id>` when
    it holds several, so each row names something that can actually be selected
    (`--project <filename> --session <short id>`)."""
    multi = len(sessions) > 1
    return [(f"{filename}#{short_id(s.id)}" if multi else filename,
             format_local(s.started_at), s.project, s.provider,
             s.turn_count, agents_text(s.agents))
            for s in sessions]


def _read_sessions(backend) -> list[Session]:
    """Every session in `backend`, with the handle always released. An empty
    store is reported as an empty list rather than an error — a database that
    exists but holds nothing yet is a perfectly normal state, and the caller
    prints an empty listing for it exactly as it would for an empty directory."""
    try:
        reader = backend.open_reader()
    except EmptyStoreError:
        return []
    try:
        return reader.list_sessions()
    finally:
        reader.close()


def _cmd_sessions(args: argparse.Namespace) -> int:
    """Implements `ctxdiff sessions` (and its hidden `runs` alias): the answer
    to "what can I pass to --session?".

    Where it looks mirrors every other command with ONE deliberate exception:
    with no `--project` and nothing configured, it lists every `*.ctrace` in the
    working directory rather than narrowing to the newest. Discovery is the one
    job where narrowing would defeat the purpose — you cannot pick a project you
    were never shown. A file that fails to open (corrupt, wrong schema version,
    not actually a ctrace) is skipped rather than aborting the whole listing;
    one bad file shouldn't hide every good one."""
    try:
        backend = _discovery_backend(args.project)
    except Exception as exc:  # noqa: BLE001 — a bad DSN/dead DB is reported, not crashed
        return _report_open_failure(exc)

    if backend is None:
        rows = []
        for path in sorted(glob.glob(os.path.join(os.getcwd(), "*.ctrace"))):
            try:
                ct = CTrace.open(path)
            except Exception:  # noqa: BLE001 — skip unreadable files, don't crash the listing
                continue
            try:
                sessions = ct.list_sessions()
            finally:
                ct.close()
            rows.extend(_file_session_rows(os.path.basename(path), sessions))
        print(render_sessions_list(rows, empty=_EMPTY_CWD))
        return 0

    empty = _EMPTY_PROJECT if args.project else _EMPTY_STORE
    try:
        sessions = _read_sessions(backend)
    except Exception as exc:  # noqa: BLE001 — a dead DB is reported, not crashed
        return _report_open_failure(exc)

    path = getattr(backend, "path", None) if hasattr(backend, "path_for") else None
    if path:
        rows = _file_session_rows(os.path.basename(path), sessions)
    else:
        # A networked store has no filenames at all, so the short session id is
        # the only label there is — and it is exactly what `--session` takes.
        rows = [(short_id(s.id), format_local(s.started_at), s.project,
                 s.provider, s.turn_count, agents_text(s.agents))
                for s in sessions]
    print(render_sessions_list(rows, empty=empty))
    return 0


def _agent_rows(sessions_calls: list[list[Call]]):
    """Aggregate per-session call lists into the `agents` table's rows: for each
    agent, how many SESSIONS it appears in, how many calls it made in total, and
    its provider-reported token spend.

    How: one pass over every session's calls, bucketing by agent name (calls
    with no label share the same `(unlabeled)` bucket the token report uses) in
    first-appearance order, counting a session once per agent it contains. The
    token column is delegated to `usage_totals`, the same rollup `ctxdiff
    tokens` prints, so the two commands can never report different numbers for
    the same calls — and it renders '-' rather than 0 when NO call of that agent
    reported usage, because 0 would read as "this agent was free"."""
    order: list[str] = []
    calls_by_agent: dict[str, list[Call]] = {}
    session_counts: dict[str, int] = {}
    for calls in sessions_calls:
        seen_here: set[str] = set()
        for c in calls:
            name = c.agent if c.agent else UNLABELED
            if name not in calls_by_agent:
                order.append(name)
                calls_by_agent[name] = []
            calls_by_agent[name].append(c)
            if name not in seen_here:
                seen_here.add(name)
                session_counts[name] = session_counts.get(name, 0) + 1

    rows = []
    for name in order:
        calls = calls_by_agent[name]
        totals = usage_totals(calls)
        tokens = (f"{totals.input_tokens + totals.output_tokens:,}"
                  if totals.calls_with_usage else "-")
        rows.append((name, session_counts[name], len(calls), tokens))
    return rows


def _cmd_agents(args: argparse.Namespace) -> int:
    """Implements `ctxdiff agents`: every agent in the project with its
    footprint aggregated ACROSS all sessions.

    Why aggregated: "how much does the researcher cost" is a question about the
    project, not about whichever run happens to be newest — a per-session answer
    would change every time the user re-ran their agent. Project resolution is
    the same as `sessions`', including the cwd-wide scan when nothing is
    configured, so the two discovery commands always describe the same set of
    traces."""
    try:
        backend = _discovery_backend(args.project)
    except Exception as exc:  # noqa: BLE001 — a bad DSN/dead DB is reported, not crashed
        return _report_open_failure(exc)

    sessions_calls: list[list[Call]] = []
    try:
        if backend is None:
            for path in sorted(glob.glob(os.path.join(os.getcwd(), "*.ctrace"))):
                try:
                    ct = CTrace.open(path)
                except Exception:  # noqa: BLE001 — skip unreadable files
                    continue
                try:
                    sessions_calls.extend(
                        ct.get_calls(s.id) for s in ct.list_sessions())
                finally:
                    ct.close()
        else:
            try:
                reader = backend.open_reader()
            except EmptyStoreError:
                reader = None
            if reader is not None:
                try:
                    sessions_calls = [reader.get_calls(s.id)
                                      for s in reader.list_sessions()]
                finally:
                    reader.close()
    except Exception as exc:  # noqa: BLE001 — a dead DB is reported, not crashed
        return _report_open_failure(exc)

    print(render_agents_list(_agent_rows(sessions_calls),
                             empty="no agents in this project"))
    return 0


# --- viewer commands -------------------------------------------------------------


def _cmd_export(args: argparse.Namespace) -> int:
    """Implements `ctxdiff export`. How: opens the project focused on one
    session (unresolvable `--session`/`--agent` -> exit 2 with the listing;
    nothing found -> exit 1, same convention as the other subcommands), exports
    the whole project's three-level dashboard to HTML (any build/write failure
    -> exit 1 with a message), and prints the written path on success. With no
    `--out`, a file-backed trace writes `<stem>.html` beside itself and a
    database-backed one has no filename to borrow, so `--out` is required
    there.

    The context window is resolved before the project is opened (same order
    `tokens` uses, so a bad `CTXDIFF_CONTEXT_WINDOW` fails identically with or
    without a trace present) and embedded in the page, which is what lets a
    dashboard someone else opens show percentages with no flag of their own."""
    try:
        context_window = _resolve_window(args)
    except SelectionError as exc:
        return _report_selection_error(exc)
    try:
        ct, default_out, selected = _open_dashboard(args.project, args.session,
                                                    args.agent)
    except SelectionError as exc:
        return _report_selection_error(exc)
    except Exception as exc:  # noqa: BLE001 — any open failure is reported, not crashed
        return _report_open_failure(exc)
    try:
        out = export_store(ct, args.out or default_out, agent=args.agent,
                           session_selected=selected,
                           context_window=context_window)
    except Exception as exc:  # noqa: BLE001 — any export failure is reported, not crashed
        print(f"ctxdiff: {exc}", file=sys.stderr)
        return 1
    finally:
        ct.close()
    print(out)
    return 0


def _cmd_view(args: argparse.Namespace) -> int:
    """Implements `ctxdiff view`. How: opens the project focused on one session
    (unresolvable selector -> exit 2, nothing found -> exit 1), exports the
    three-level dashboard to a temp `.html` file, prints its path, and opens it
    in the default browser via a `file://` URL unless `--no-open` is set. The
    browser launch is wrapped so a failing/absent browser NEVER crashes the
    command — the path is already printed, so the user can always open the file
    themselves. `--context-window` (or `CTXDIFF_CONTEXT_WINDOW`) is resolved
    through the same `_resolve_window` every other command uses and embedded in
    the page, so the dashboard's percentages and `ctxdiff tokens`' percentages
    are always the same number."""
    try:
        context_window = _resolve_window(args)
    except SelectionError as exc:
        return _report_selection_error(exc)
    try:
        ct, _, selected = _open_dashboard(args.project, args.session, args.agent)
    except SelectionError as exc:
        return _report_selection_error(exc)
    except Exception as exc:  # noqa: BLE001 — any open failure is reported, not crashed
        return _report_open_failure(exc)

    # mkstemp creates and opens the file; we only need its path, so close the
    # descriptor immediately and let the exporter rewrite the file by path.
    fd, tmp = tempfile.mkstemp(suffix=".html")
    os.close(fd)
    try:
        out = export_store(ct, tmp, agent=args.agent, session_selected=selected,
                           context_window=context_window)
    except Exception as exc:  # noqa: BLE001 — any export failure is reported, not crashed
        print(f"ctxdiff: {exc}", file=sys.stderr)
        return 1
    finally:
        ct.close()

    print(out)
    if not args.no_open:
        try:
            webbrowser.open("file://" + os.path.abspath(out))
        except Exception:  # noqa: BLE001 — never let a browser failure crash view
            _log.warning("ctxdiff: could not open a browser; open %s manually", out)
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    """Implements `ctxdiff demo`: the zero-friction first run. How: resolves
    where the demo's two files go (see the three placement rules below),
    builds the sample trace via `build_demo_trace` and exports its dashboard
    (any build/write failure -> exit 1, same convention as `export`/`view`),
    prints both paths plus a one-line explanation of what they're looking at,
    opens the dashboard in a browser unless `--no-open`, and closes with a
    nudge toward tracing a real agent. The browser launch is wrapped exactly
    like `view`'s — a failing/absent browser never crashes the command, since
    both paths are already printed.

    Placement rules, in priority order: `--out FILE` writes the trace there
    (and `FILE`'s `.html` sibling next to it) and is never cleaned up, since
    the user named a permanent location; `--keep` (no `--out`) writes a fixed
    `./ctxdiff-demo.ctrace` + `.html` pair in the cwd, so a re-run overwrites
    the same two files the user can find again; the default uses a tempfile
    pair, same as `view` — printed for one-off use, never auto-deleted (same
    convention `view` already established)."""
    if args.out:
        ctrace_path = args.out
        html_path = os.path.splitext(ctrace_path)[0] + ".html"
    elif args.keep:
        ctrace_path = os.path.join(os.getcwd(), "ctxdiff-demo.ctrace")
        html_path = os.path.join(os.getcwd(), "ctxdiff-demo.html")
    else:
        fd, ctrace_path = tempfile.mkstemp(suffix=".ctrace")
        os.close(fd)
        fd, html_path = tempfile.mkstemp(suffix=".html")
        os.close(fd)

    try:
        build_demo_trace(ctrace_path)
        out = export_html(ctrace_path, html_path)
    except Exception as exc:  # noqa: BLE001 — any build/export failure is reported, not crashed
        print(f"ctxdiff: {exc}", file=sys.stderr)
        return 1

    print(f"sample trace  -> {ctrace_path}")
    print(f"dashboard     -> {out}")
    print("This is a sample multi-agent research-pipeline run (no API keys, "
          "no network) — it shows turn-by-turn diffs, token/schema-bloat "
          "detection, a cache-prefix break, and two agents on one timeline.")

    if not args.no_open:
        try:
            webbrowser.open("file://" + os.path.abspath(out))
        except Exception:  # noqa: BLE001 — never let a browser failure crash demo
            _log.warning("ctxdiff: could not open a browser; open %s manually", out)

    print('Trace your own agent next: tracer = trace.init("my-agent"); '
          "client = tracer.wrap(OpenAI())")
    return 0


# --- mcp --------------------------------------------------------------------------


def _mcp_sdk_installed() -> bool:
    """Whether the optional `mcp` SDK (the `ctxdiff[mcp]` extra) is importable.

    `find_spec` rather than a try/except around the import, because it answers
    the question without EXECUTING anything: the two failures are not the same,
    and a try/except would quietly translate an ImportError raised from inside
    ctxdiff's own server module — a bug — into "go install something", which is
    the one message guaranteed not to fix it.

    A named function, not an inline call, so a test can simulate an install
    WITHOUT the extra on a machine that has it."""
    return importlib.util.find_spec("mcp") is not None


def _cmd_mcp(args: argparse.Namespace) -> int:
    """Implements `ctxdiff mcp`: run the MCP server on stdio until the client
    disconnects.

    The one thing this function does beyond delegating is refuse to crash when
    the optional extra is absent. `ctxdiff[mcp]` pulls in the official `mcp`
    Python SDK, which the core deliberately does not depend on (runtime deps
    stay tiktoken-only), so a user who types `ctxdiff mcp` on a plain install
    must get one line telling them what to install — not a traceback ending in
    `ModuleNotFoundError: No module named 'mcp'`, which reads like ctxdiff is
    broken rather than like a feature is opt-in. See `_mcp_sdk_installed` for
    why the check is a spec lookup rather than a caught import.

    stdout belongs to the JSON-RPC protocol here, so the hint — like every
    other error in this CLI — goes to stderr."""
    if not _mcp_sdk_installed():
        print(MISSING_EXTRA_HINT, file=sys.stderr)
        return 1
    from ctxdiff.mcp.server import serve_stdio
    return serve_stdio(runs_dir=args.runs_dir, redact=args.redact)


# --- entry point ---------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """ctxdiff's console-script entry point. `argv` defaults to
    `sys.argv[1:]` (via argparse's own default) when None, so both the
    installed script and in-process tests — which pass argv explicitly —
    behave identically. Dispatches to the matched subcommand's `func` and
    returns its exit code; with no subcommand, prints help and exits 2
    (argparse's usage-error convention)."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
