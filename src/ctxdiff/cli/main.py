"""ctxdiff's command-line entry point. Plain argparse + a small subcommand
registry (stdlib only, per CLAUDE.md's "runtime deps: tiktoken only" rule —
no click) so later milestones (`tokens`, `cache`, `view`, `export`) can each
add one `_add_<name>_parser` function to `_SUBCOMMANDS` without touching the
others."""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import tempfile
import webbrowser

from ctxdiff.analyze.cache import analyze_cache
from ctxdiff.analyze.differ import diff_turns
from ctxdiff.analyze.tokens import analyze_run, registered_tool_names
from ctxdiff.cli.render import (
    render_agent_summary,
    render_cache_report,
    render_runs_list,
    render_run_tokens,
    render_turn_diff,
    render_usage_summary,
)
from ctxdiff.store.ctrace import CTrace
from ctxdiff.viewer import export_html

_log = logging.getLogger("ctxdiff")


def _find_default_run(cwd: str) -> str | None:
    """Return the most recently modified `*.ctrace` file in `cwd`, or None
    if there isn't one. Backs --run's default so the common case (one run in
    the working directory) needs no flag at all."""
    candidates = glob.glob(os.path.join(cwd, "*.ctrace"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _resolve_run_path(explicit: str | None) -> str | None:
    """Resolve the `.ctrace` path to open: the explicit `--run` value if
    given, else the most recently modified `*.ctrace` in the cwd. Returns
    None (rather than raising) when neither yields a path, so callers stay
    plain functions returning exit codes — `main()` must be safely callable
    in-process (tests do `main(argv=[...])` and check the return value), not
    only as a script that's allowed to exit the whole process."""
    if explicit:
        return explicit
    return _find_default_run(os.getcwd())


# --- subcommand registration --------------------------------------------------


def _add_diff_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register `ctxdiff diff --turn N --turn M [--run PATH]`. `--turn` uses
    action='append' so it can be passed twice; `_cmd_diff` validates the
    resulting count (exactly 2) itself, since argparse alone can't express
    "this flag exactly twice"."""
    p = subparsers.add_parser(
        "diff", help="git-style block diff between two turns")
    p.add_argument("--turn", action="append", type=int, dest="turns",
                   help="turn (call seq) number; pass exactly twice, e.g. "
                        "--turn 7 --turn 8")
    p.add_argument("--agent", default=None,
                   help="restrict to one agent: both --turn values must be "
                        "calls of this agent (turns still use global seq numbers)")
    p.add_argument("--run", default=None, help="path to a .ctrace file "
                    "(default: most recently modified *.ctrace in cwd)")
    p.set_defaults(func=_cmd_diff)


def _add_tokens_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register `ctxdiff tokens [--turn N] [--run PATH]`. `--turn` is a
    single optional int (unlike `diff`'s `--turn` twice) — omitted, every
    turn in the run is shown; given, output is limited to that one turn."""
    p = subparsers.add_parser(
        "tokens", help="token heatmap + schema-bloat report")
    p.add_argument("--turn", type=int, default=None,
                    help="limit output to one turn (call seq) number")
    p.add_argument("--agent", default=None,
                    help="restrict token attribution to one agent's calls")
    p.add_argument("--run", default=None, help="path to a .ctrace file "
                    "(default: most recently modified *.ctrace in cwd)")
    p.set_defaults(func=_cmd_tokens)


def _add_cache_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register `ctxdiff cache [--run PATH]`: no other arguments — the
    profiler always analyzes the whole run's consecutive-call pairs."""
    p = subparsers.add_parser(
        "cache", help="prefix-stability report + wasted-spend estimate")
    p.add_argument("--agent", default=None,
                    help="restrict prefix analysis to one agent's calls")
    p.add_argument("--run", default=None, help="path to a .ctrace file "
                    "(default: most recently modified *.ctrace in cwd)")
    p.set_defaults(func=_cmd_cache)


def _add_runs_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register `ctxdiff runs` (no arguments: it always lists the cwd)."""
    p = subparsers.add_parser(
        "runs", help="list .ctrace files in the working directory")
    p.set_defaults(func=_cmd_runs)


def _add_export_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register `ctxdiff export [--run PATH] [--out FILE.html]`: emit a
    self-contained HTML dashboard. `--out` overrides the destination; by
    default the file is written as `<trace-stem>.html` beside the trace."""
    p = subparsers.add_parser(
        "export", help="write a self-contained HTML dashboard for a run")
    p.add_argument("--run", default=None, help="path to a .ctrace file "
                    "(default: most recently modified *.ctrace in cwd)")
    p.add_argument("--out", default=None,
                    help="output .html path (default: <trace-stem>.html next to the trace)")
    p.set_defaults(func=_cmd_export)


def _add_view_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register `ctxdiff view [--run PATH] [--no-open]`: export the dashboard
    to a temp file and open it in the default browser. `--no-open` skips the
    browser launch (used by tests and headless/CI environments) but still
    writes and prints the file path."""
    p = subparsers.add_parser(
        "view", help="open a self-contained HTML dashboard in your browser")
    p.add_argument("--run", default=None, help="path to a .ctrace file "
                    "(default: most recently modified *.ctrace in cwd)")
    p.add_argument("--no-open", action="store_true", dest="no_open",
                    help="write and print the HTML path but do not open a browser")
    p.set_defaults(func=_cmd_view)


# Every registered subcommand's add_parser function, in help-listing order.
# Appending here is the ONLY change later milestones need to add a subcommand.
_SUBCOMMANDS = [_add_diff_parser, _add_tokens_parser, _add_cache_parser,
                _add_runs_parser, _add_export_parser, _add_view_parser]


def _build_parser() -> argparse.ArgumentParser:
    """Assemble the top-level parser and register every subcommand."""
    parser = argparse.ArgumentParser(
        prog="ctxdiff", description="git diff for your agent's context window")
    subparsers = parser.add_subparsers(dest="command")
    for register in _SUBCOMMANDS:
        register(subparsers)
    return parser


# --- subcommand implementations ------------------------------------------------


def _cmd_diff(args: argparse.Namespace) -> int:
    """Implements `ctxdiff diff`. How: validates `--turn` was passed exactly
    twice (a usage error -> exit 2), resolves and opens the `.ctrace`
    (missing/corrupt file -> operational error, exit 1), computes the diff
    via `diff_turns` (missing turn -> operational error, exit 1), and prints
    the rendered result on success."""
    turns = args.turns or []
    if len(turns) != 2:
        print("ctxdiff diff requires exactly two --turn flags, e.g. "
              "'ctxdiff diff --turn 7 --turn 8'", file=sys.stderr)
        return 2

    path = _resolve_run_path(args.run)
    if path is None:
        print("no .ctrace here — did the run capture?", file=sys.stderr)
        return 1
    try:
        ct = CTrace.open(path)
    except Exception as exc:  # noqa: BLE001 — any open failure is reported, not crashed
        print(f"ctxdiff: {exc}", file=sys.stderr)
        return 1

    try:
        turn_old, turn_new = turns
        # --agent: validate BOTH turns belong to that agent before diffing.
        # Turns stay global seq numbers (what every view shows); we only check
        # ownership, then diff_turns resolves them against the whole run as
        # usual (the two calls happen to be this agent's).
        if args.agent is not None:
            agent_seqs = [c.seq for c in ct.get_calls() if c.agent == args.agent]
            bad = [t for t in (turn_old, turn_new) if t not in agent_seqs]
            if bad:
                print(f"ctxdiff: turn(s) {bad} are not calls of agent "
                      f"'{args.agent}' (that agent's turns: {agent_seqs})",
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


def _cmd_tokens(args: argparse.Namespace) -> int:
    """Implements `ctxdiff tokens`. How: resolves and opens the `.ctrace`
    (missing/corrupt file -> exit 1, same as `diff`), runs the token
    attributor over the whole run, then narrows to one turn when `--turn` is
    given (missing turn -> exit 1 with a message, mirroring `diff`'s
    behavior). When there IS unused-schema bloat to report, a second pass
    over the run's blocks derives "M" (how many tools are registered in
    total) for the "N of M" bloat message — cheap relative to the run size,
    and only paid when there's something to report."""
    path = _resolve_run_path(args.run)
    if path is None:
        print("no .ctrace here — did the run capture?", file=sys.stderr)
        return 1
    try:
        ct = CTrace.open(path)
    except Exception as exc:  # noqa: BLE001 — any open failure is reported, not crashed
        print(f"ctxdiff: {exc}", file=sys.stderr)
        return 1

    try:
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
        print(render_run_tokens(selected, run_tokens.bloat, total_tools,
                                agent_summary, usage_summary))
    finally:
        ct.close()
    return 0


def _cmd_cache(args: argparse.Namespace) -> int:
    """Implements `ctxdiff cache`. How: resolves and opens the `.ctrace`
    (missing/corrupt file -> exit 1, same convention as `diff`/`tokens`),
    runs the cache-prefix profiler over the whole run, and prints the
    rendered report. Breaks found are informational (not a usage/operational
    failure), so a clean run and a run full of warnings both exit 0 — same
    convention as `tokens`' schema-bloat warning."""
    path = _resolve_run_path(args.run)
    if path is None:
        print("no .ctrace here — did the run capture?", file=sys.stderr)
        return 1
    try:
        ct = CTrace.open(path)
    except Exception as exc:  # noqa: BLE001 — any open failure is reported, not crashed
        print(f"ctxdiff: {exc}", file=sys.stderr)
        return 1

    try:
        report = analyze_cache(ct, agent=args.agent)
        print(render_cache_report(report))
    finally:
        ct.close()
    return 0


def _cmd_runs(args: argparse.Namespace) -> int:
    """Implements `ctxdiff runs`: lists every `*.ctrace` in the cwd with its
    project/provider/turn-count. A file that fails to open (corrupt, wrong
    schema version, not actually a ctrace) is skipped rather than aborting
    the whole listing — one bad file shouldn't hide every good one."""
    candidates = sorted(glob.glob(os.path.join(os.getcwd(), "*.ctrace")))
    rows: list[tuple[str, str, str, int, str]] = []
    for path in candidates:
        try:
            ct = CTrace.open(path)
        except Exception:  # noqa: BLE001 — skip unreadable files, don't crash the listing
            continue
        try:
            run = ct.get_run()
            calls = ct.get_calls()
            n_calls = len(calls)
            # Distinct named agents, in first-appearance order; '-' when the
            # run has none (a single-agent/pre-v2 run).
            names: list[str] = []
            for c in calls:
                if c.agent and c.agent not in names:
                    names.append(c.agent)
            agents = ", ".join(names) if names else "-"
        finally:
            ct.close()
        rows.append((os.path.basename(path), run.project, run.provider,
                     n_calls, agents))
    print(render_runs_list(rows))
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    """Implements `ctxdiff export`. How: resolves the `.ctrace` (missing ->
    exit 1, same convention as the other subcommands), exports it to HTML
    (any open/build/write failure -> exit 1 with a message), and prints the
    written path on success."""
    path = _resolve_run_path(args.run)
    if path is None:
        print("no .ctrace here — did the run capture?", file=sys.stderr)
        return 1
    try:
        out = export_html(path, args.out)
    except Exception as exc:  # noqa: BLE001 — any export failure is reported, not crashed
        print(f"ctxdiff: {exc}", file=sys.stderr)
        return 1
    print(out)
    return 0


def _cmd_view(args: argparse.Namespace) -> int:
    """Implements `ctxdiff view`. How: resolves the `.ctrace` (missing ->
    exit 1), exports the dashboard to a temp `.html` file, prints its path,
    and opens it in the default browser via a `file://` URL unless
    `--no-open` is set. The browser launch is wrapped so a failing/absent
    browser NEVER crashes the command — the path is already printed, so the
    user can always open the file themselves."""
    path = _resolve_run_path(args.run)
    if path is None:
        print("no .ctrace here — did the run capture?", file=sys.stderr)
        return 1

    # mkstemp creates and opens the file; we only need its path, so close the
    # descriptor immediately and let export_html rewrite the file by path.
    fd, tmp = tempfile.mkstemp(suffix=".html")
    os.close(fd)
    try:
        out = export_html(path, tmp)
    except Exception as exc:  # noqa: BLE001 — any export failure is reported, not crashed
        print(f"ctxdiff: {exc}", file=sys.stderr)
        return 1

    print(out)
    if not args.no_open:
        try:
            webbrowser.open("file://" + os.path.abspath(out))
        except Exception:  # noqa: BLE001 — never let a browser failure crash view
            _log.warning("ctxdiff: could not open a browser; open %s manually", out)
    return 0


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
