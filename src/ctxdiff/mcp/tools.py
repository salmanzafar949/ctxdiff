"""The six ctxdiff MCP tools, as plain functions over the SAME pure analyzers
the CLI, the HTML dashboard and `ctxdiff check` read.

Nothing here computes a number. `ctxdiff_diff` calls `diff_turns`,
`ctxdiff_tokens` calls `analyze_run` and `analyze_evictions`, `ctxdiff_cache`
calls `analyze_cache`, `ctxdiff_runs` calls `Store.list_sessions`, and
`ctxdiff_explain` calls all of them — the same functions, on the same store
handle (`_SessionView`), resolved through the same store resolver
(`_discovery_backend`). That is the whole design constraint: a debugger that
answers "18,400 tokens" in the terminal and "18,900 tokens" over MCP is worse
than no debugger, and the only way to make that impossible is to have one
implementation and four renderings of it.

The module is deliberately free of any `mcp` SDK import, so the result shaping
can be tested — and reasoned about — with the optional extra absent;
`server.py` is the thin layer that binds these functions to FastMCP and carries
the tool descriptions.

STATELESSNESS is a property, not an accident: every call resolves the source,
opens a store, reads, and closes. There is no daemon, no cached handle, no lock
held against the live writer, so a running agent can keep appending to the very
trace being inspected."""
from __future__ import annotations

import glob
import os
from collections import Counter
from dataclasses import dataclass

from ctxdiff.analyze.cache import analyze_cache, pairs_denominator
from ctxdiff.analyze.differ import diff_turns
from ctxdiff.analyze.evictions import analyze_evictions
from ctxdiff.analyze.tokens import analyze_run, registered_tool_names
from ctxdiff.cli.main import _SessionView, _discovery_backend
from ctxdiff.cli.select import SHORT_ID_LEN
from ctxdiff.mcp import payload as pay
from ctxdiff.mcp.payload import TextPolicy
from ctxdiff.store import config as store_config
from ctxdiff.store.base import EmptyStoreError, Session, parse_started_at
from ctxdiff.store.ctrace import CTrace

# How many turns a block-lookup result names as "present at", before it stops
# enumerating. A block that survives 200 turns is answered by "200 turns,
# first 20 listed" just as well, and far more cheaply.
_MAX_TURNS_LISTED = 20

# How many individual changed blocks `ctxdiff_explain` quotes inline. It is a
# SUMMARY tool — the agent that wants the twelfth change calls `ctxdiff_diff`.
_EXPLAIN_CHANGES = 5

# How much of a tool name / label / hash / agent name an error message or a
# finding row may quote. Long enough to recognize, short enough that forty of
# them still cost less than one preview.
_NAME_CHARS = 64

# How many turn numbers and agent names an error message enumerates before it
# switches to counting. The value of "available turns: [...]" is that the agent
# can correct itself without another call; the first few plus a total does that
# just as well as five thousand integers, and a 5,000-turn trace's full list is
# ~30 KB — twice the entire result cap, spent on a failure.
_MAX_TURNS_IN_ERROR = 40
_MAX_AGENTS_IN_ERROR = 20

# How many findings of one kind `_explain_summary` names before it counts the
# rest, and the hard ceiling on the assembled sentence. The summary is a plain
# string, so `fit()` can neither trim items from it nor drop it — anything
# unbounded here is unbounded in the result. See `_explain_summary`.
_SUMMARY_FINDINGS = 3
_SUMMARY_CHARS = 600


class ToolError(Exception):
    """A tool call that cannot be answered as asked: an unknown run, a turn the
    trace does not contain, an ambiguous content hash.

    Its own type so `server.py` can turn it into a clean one-line MCP error
    (which is what an agent can act on) while a genuine bug still surfaces as a
    real traceback. Every message it carries names the way forward — usually
    "call ctxdiff_runs" or the list of values that WOULD have worked — because
    an agent's next move is decided entirely by this string.

    The message is bounded and ANSI-stripped HERE, at the one place every tool
    error passes through, because an error result does not go through `fit()`:
    it is handed to the agent whole. Every value quoted below is also flattened
    at its own call site so the useful part is not crowded out — this is the
    backstop that holds when a future message forgets to."""

    # The cap on one error result. Two orders of magnitude under
    # MAX_RESULT_BYTES on purpose: an error carries no data, only a correction.
    MAX_CHARS = 1_000

    def __init__(self, message: str):
        super().__init__(pay.flatten(pay.strip_ansi(str(message)),
                                     self.MAX_CHARS))


# --- where traces come from ----------------------------------------------------


@dataclass(frozen=True)
class Source:
    """Where this server reads traces from: a DIRECTORY of `.ctrace` files, or a
    configured networked store (Postgres/MySQL). Exactly one is set.

    Why a directory and not "the newest .ctrace in cwd" (the CLI's zero-config
    default): an MCP server's working directory is whatever the client happened
    to launch it from — an editor's install root, `/`, the user's home — so
    "newest here" is not merely unreliable, it is usually wrong. `--runs-dir`
    makes the location explicit in the client's server config, and
    `ctxdiff_runs` is how the agent learns what is in it."""
    directory: str | None
    backend: object | None

    @property
    def label(self) -> str:
        """A short description of the source for result envelopes, so an agent
        (and the human reading over its shoulder) can see which store answered.
        A backend is named by CLASS, never by DSN — a DSN carries credentials,
        and this string is on its way to a cloud model."""
        if self.directory is not None:
            return self.directory
        return type(self.backend).__name__


def resolve_source(runs_dir: str | None) -> Source:
    """Decide where to read traces from, explicit-beats-ambient, exactly as
    every ctxdiff read command decides it:

    1. `--runs-dir DIR` — always that directory.
    2. A configured store (`ctxdiff.configure()` / `CTXDIFF_STORE`), resolved by
       the CLI's OWN `_discovery_backend`, so `ctxdiff mcp` and `ctxdiff tokens`
       can never disagree about which database is configured. That covers both a
       networked backend and a configured SQLite FILE.
    3. A configured SQLite backend naming a whole DIRECTORY of traces — which
       `_discovery_backend` reports as "no single project here" — is that
       directory.
    4. Nothing configured: the working directory, which for a server launched by
       an editor is a guess, hence rule 1."""
    if runs_dir is not None:
        return Source(directory=os.path.abspath(os.path.expanduser(runs_dir)),
                      backend=None)
    backend = _discovery_backend(None)
    if backend is not None:
        return Source(directory=None, backend=backend)
    configured = store_config.resolve()
    directory = getattr(configured, "path", None) if configured is not None else None
    return Source(directory=directory or os.getcwd(), backend=None)


@dataclass(frozen=True)
class RunRef:
    """One session this server can be asked about, and the handle that asks for
    it. `run` is the ONLY identifier the other five tools accept — it is a label
    minted from data this server already read (a filename in the runs directory,
    or a session id), never a path the caller supplies. That is deliberate: it
    means no tool argument can be steered into opening a file outside the
    configured source."""
    run: str
    session_id: str
    # The file this session lives in, if any, split into the name a human reads
    # and the path this server may open. They are separate because a CONFIGURED
    # SQLite store is read through its backend (never re-opened by path) but is
    # still recognizable — and typeable — by its filename.
    filename: str | None
    path: str | None
    project: str
    provider: str
    started_at: str
    turns: int
    agents: list[str]
    # The session-id prefix this listing quotes and disambiguates handles with.
    # `short_id`'s 12 characters in the overwhelmingly common case, longer only
    # when two sessions this server can see share those 12 — see
    # `_unique_prefix_len`.
    session_short: str


def _unique_prefix_len(session_ids: list[str]) -> int:
    """The shortest session-id prefix length (never below `short_id`'s) that
    tells every discovered session apart.

    Handles are minted from this prefix, and a handle that names two sessions
    names neither: `_match_run` correctly refuses an ambiguous one, so a file
    whose sessions collide becomes wholly unreachable over MCP — every call
    answering "matches 3 sessions: X, X, X". Not hypothetical: ctxdiff's own
    `spec/golden/corpus/tagged-eviction.json` seeds three sessions as
    `e900…0001/2/3`, which share 12 leading hex characters, and any id scheme
    with a common prefix does the same.

    Lengthening only on collision keeps the common case at exactly the 12
    characters `ctxdiff sessions` prints, so a handle copied from one surface
    still resolves in the other."""
    ids = list(session_ids)
    longest = max((len(i) for i in ids), default=0)
    distinct = len(set(ids))
    for length in range(min(SHORT_ID_LEN, longest), longest):
        if len({i[:length] for i in ids}) == distinct:
            return length
    return max(longest, SHORT_ID_LEN)


def _run_label(basename: str | None, prefix: str, multi: bool) -> str:
    """The handle for one session: the bare filename when its file holds one
    session (what a human recognizes and what an agent will guess),
    `file.ctrace#4f3a2b1c` when it holds several, and the bare session prefix
    when there is no file at all. Same rule `ctxdiff sessions` labels its rows
    by, so a handle copied out of either listing works in the other."""
    if basename is None:
        return prefix
    return f"{basename}#{prefix}" if multi else basename


def _ref(session: Session, run: str, filename: str | None, path: str | None,
         prefix: str) -> RunRef:
    """Build one discovery row from a store's own `Session` summary."""
    return RunRef(run=run, session_id=session.id, filename=filename, path=path,
                  project=session.project, provider=session.provider,
                  started_at=session.started_at, turns=session.turn_count,
                  agents=list(session.agents), session_short=prefix)


def list_runs(source: Source) -> list[RunRef]:
    """Every session `source` holds, newest first.

    A `.ctrace` that will not open (corrupt, a half-written file, something that
    merely ends in `.ctrace`) is SKIPPED rather than aborting the listing — the
    same choice `ctxdiff sessions` makes, and for the same reason: one bad file
    must not hide every good one from an agent that has no other way to find
    them."""
    rows: list[tuple[Session, str | None, str | None]] = []
    if source.backend is not None:
        try:
            reader = source.backend.open_reader()
        except EmptyStoreError:
            return []
        try:
            sessions = reader.list_sessions()
        finally:
            reader.close()
        # A configured SQLite store is a FILE, and `ctxdiff sessions` labels its
        # rows by that filename — so this listing does too, or the same session
        # would answer to two different names in two ctxdiff surfaces. A
        # networked store has no filename, and its short session id is both the
        # only label there is and exactly what `--session` takes.
        filename = _backend_filename(source.backend)
        rows = [(s, filename, None) for s in sessions]
    else:
        # RECURSIVE on purpose (dogfood finding 2026-07-27): a project often
        # holds its traces one level down (`app/py/*.ctrace`, `app/js/*.ctrace`),
        # and a top-level-only glob made `--runs-dir <project>` silently see
        # NOTHING — indistinguishable from "no traces exist". `**` includes the
        # top level itself, so flat layouts behave exactly as before; when the
        # same filename appears in two subdirectories the label is the path
        # RELATIVE to the runs dir (not the bare basename), so the two stay
        # distinguishable in the listing and addressable by `run`.
        pattern = os.path.join(source.directory, "**", "*.ctrace")
        for path in sorted(glob.glob(pattern, recursive=True)):
            try:
                ct = CTrace.open(path)
            except Exception:  # noqa: BLE001 — skip unreadable files, don't hide the rest
                continue
            try:
                sessions = ct.list_sessions()
            finally:
                ct.close()
            label = os.path.relpath(path, source.directory)
            rows.extend((s, label, path) for s in sessions)
    return _label_rows(rows)


def _label_rows(rows: list[tuple[Session, str | None, str | None]]) -> list[RunRef]:
    """Turn discovered (session, filename, path) rows into handled, sorted
    discovery rows.

    Labeling is done over ALL rows at once rather than file by file, because
    both things it decides are properties of the whole listing: whether a file
    holds several sessions (so its handle needs a disambiguator at all) and how
    long that disambiguator has to be before every session this server can see
    has its own."""
    prefix_len = _unique_prefix_len([s.id for s, _, _ in rows])
    per_file = Counter(filename for _, filename, _ in rows)
    refs = [_ref(s, _run_label(filename, s.id[:prefix_len],
                               per_file[filename] > 1),
                 filename, path, s.id[:prefix_len])
            for s, filename, path in rows]
    return sorted(refs, key=_started_key, reverse=True)


def _backend_filename(backend) -> str | None:
    """The filename a FILE-BACKED store should be labeled by, or None for a
    networked one. Detected by the absence of `path_for` — the same capability
    check `Tracer` and the CLI's own `_discovery_backend` use, rather than an
    isinstance chain that a new backend would have to be added to."""
    if not hasattr(backend, "path_for"):
        return None
    path = getattr(backend, "path", None)
    if path and not os.path.isdir(path):
        return os.path.basename(path)
    return None


def _started_key(ref: RunRef) -> float:
    """Sort key for "newest first": the session's start time as a timestamp, and
    0 for a row whose stored timestamp will not parse — an unreadable clock
    should push a run to the bottom of the list, never crash discovery."""
    try:
        return parse_started_at(ref.started_at).timestamp()
    except (ValueError, TypeError):
        return 0.0


def _match_run(refs: list[RunRef], run: str) -> RunRef:
    """Resolve a `run` argument to exactly one discovered session, accepting
    every spelling an agent might carry back: the exact handle
    (`trace.ctrace#4f3a2b1c`), a bare filename when the file holds one session,
    a filename whose stem was typed without `.ctrace`, or a session id prefix.

    Ambiguity is an ERROR listing the candidates, never a silent pick: choosing
    for the agent is how a debugging session ends up reasoning confidently about
    the wrong run."""
    wanted = run.strip()
    if not wanted:
        raise ToolError("run is required — call ctxdiff_runs to list the traces "
                        "this server can read")
    # What gets QUOTED back at the caller, bounded. The argument itself stays
    # whole for matching: an error result does not pass through `fit()`, so a
    # 100,000-character `run` would otherwise be echoed verbatim into the
    # agent's context — several times the entire result cap, to say "no match".
    shown = pay.flatten(wanted, _NAME_CHARS)

    exact = [r for r in refs if r.run == wanted]
    if len(exact) == 1:
        return exact[0]

    candidates = [
        r for r in refs
        if r.run == wanted
        or (r.filename is not None and r.filename == wanted)
        or (r.filename is not None
            and os.path.splitext(r.filename)[0] == wanted)
        or (len(wanted) >= 4 and r.session_id.startswith(wanted))
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ToolError(
            f"no run matching {shown!r} — call ctxdiff_runs for the list "
            f"(this server reads {_source_hint(refs)})")
    raise ToolError(
        f"{shown!r} matches {len(candidates)} sessions: "
        f"{', '.join(sorted(c.run for c in candidates))} — pass one of those")


def _source_hint(refs: list[RunRef]) -> str:
    """A short "where I looked" clause for a no-match error: how many traces the
    server can see at all, which is what distinguishes "you typed the wrong
    name" from "this server is pointed at an empty directory"."""
    return f"{len(refs)} trace(s)" if refs else "no traces at all"


def open_run(source: Source, run: str) -> tuple[_SessionView, RunRef]:
    """Resolve `run` and return a read handle pinned to that session, plus the
    row it was resolved from.

    The handle is the CLI's own `_SessionView` — the same pinned reader
    `ctxdiff tokens --session X` analyzes through — so the analyzers cannot tell
    they are being called by an MCP server, which is the point.

    Note what this function does NOT do: it never turns `run` into a filesystem
    path. Paths come only from the source's own directory glob (or from the
    configured backend), so no argument reaching this server can address a file
    it was not pointed at."""
    ref = _match_run(list_runs(source), run)
    if ref.path is not None:
        store = CTrace.open(ref.path)
    else:
        store = source.backend.open_reader()
    return _SessionView(store, ref.session_id), ref


# --- shared payload pieces -----------------------------------------------------


def _envelope(ref: RunRef, agent: str | None = None) -> dict:
    """The header every result carries: which run and session answered, and the
    agent filter in force. Small on purpose — it is repeated on every call, so
    it holds identifiers, not descriptions."""
    return {"run": ref.run, "session": ref.session_short,
            "project": ref.project, "agent": agent}


def _require_turn(view: _SessionView, turn: int, agent: str | None):
    """Resolve a turn number to its call, refusing an unknown one with the list
    of turns that DO exist (scoped to `agent` when one was given) — the same
    contract `ctxdiff diff` and `ctxdiff tokens` honor. An agent that guessed
    turn 9 on an 8-turn trace can fix itself from this message without a second
    discovery call."""
    calls = [c for c in view.get_calls() if agent is None or c.agent == agent]
    for call in calls:
        if call.seq == turn:
            return call
    where = (f"agent {pay.flatten(agent, _NAME_CHARS)!r}" if agent is not None
             else "this run")
    seqs = sorted(c.seq for c in calls)
    raise ToolError(f"turn {turn} not found in {where} (available turns: "
                    f"{_listed(seqs, _MAX_TURNS_IN_ERROR)})")


def _listed(values: list, limit: int) -> str:
    """Render `values` for an error message: the first `limit` of them, then how
    many there are in total.

    A correction, not a data dump. Enumerating all 5,000 turns of a long run
    told the agent nothing the first forty did not, and cost twice the entire
    result cap to do it — on a failure, which by definition returns no
    analysis."""
    head = ", ".join(str(v) for v in values[:limit])
    if len(values) <= limit:
        return f"[{head}]"
    return f"[{head}, … {len(values)} total]"


def _check_agent(view: _SessionView, agent: str | None) -> None:
    """Reject an `agent` naming nobody in this session, listing who IS there.

    Without this, a typo'd agent name filters every call away and each tool
    cheerfully reports an empty, entirely truthful, entirely useless result —
    and an agent reading "0 turns" concludes the trace is empty rather than that
    it misspelled a name."""
    if agent is None:
        return
    names = []
    for call in view.get_calls():
        if call.agent and call.agent not in names:
            names.append(call.agent)
    if agent not in names:
        # Both sides are bounded: the name the caller passed (it may be
        # enormous) and the roster echoed back (a fleet run can carry hundreds
        # of agents, each named by the developer at any length).
        shown = [pay.flatten(n, _NAME_CHARS) for n in names]
        here = (_listed(shown, _MAX_AGENTS_IN_ERROR) if names
                else "none — every call is unlabeled")
        raise ToolError(f"no agent named {pay.flatten(agent, _NAME_CHARS)!r} "
                        f"in this run (agents here: {here})")


def _changed_hunk(inline_diff, policy: TextPolicy) -> str | None:
    """Turn a modified block's char-level inline diff into a compact `-`/`+`
    hunk: only the segments that actually differ, each flattened to one line.

    This is the single biggest token saving in the whole surface. A modified
    system prompt is typically 2,000 tokens of which four characters changed;
    returning the block would cost the debugging agent its own context window to
    say "a timestamp moved". The equal segments are exactly what the agent
    already knows, so they are dropped and only the delta is quoted."""
    if inline_diff is None:
        return None
    removed = "".join(text for op, text in inline_diff if op == "delete")
    added = "".join(text for op, text in inline_diff if op == "insert")
    parts = []
    if removed:
        parts.append("- " + pay.flatten(removed, pay.HUNK_CHARS // 2))
    if added:
        parts.append("+ " + pay.flatten(added, pay.HUNK_CHARS // 2))
    if not parts:
        return None
    return policy.hunk("\n".join(parts))


# --- 1. ctxdiff_runs -----------------------------------------------------------


def ctxdiff_runs(source: Source, policy: TextPolicy) -> str:
    """Discovery: every trace this server can read, newest first.

    Carries no captured text at all (project names and agent names are the
    developer's own labels, not recorded conversation), so `--redact` changes
    nothing here beyond the standing note — which is intentional: an agent must
    still be able to FIND a run on a redacted server."""
    refs = list_runs(source)
    body = {
        "source": source.label,
        "count": len(refs),
        "runs": [{"run": r.run, "session": r.session_short,
                  "file": r.filename, "project": r.project,
                  "provider": r.provider, "started_at": r.started_at,
                  "turns": r.turns, "agents": r.agents or None}
                 for r in refs],
    }
    if not refs:
        body["hint"] = (
            "no traces here. This server reads only the directory it was "
            "started with (--runs-dir) or the configured CTXDIFF_STORE — not "
            "the user's current directory. Ask the user where their .ctrace "
            "files are and have them point the server at that directory.")
    else:
        body["hint"] = ("pass one 'run' value to ctxdiff_explain(run, turn) to "
                        "diagnose a bad turn")
    return pay.fit(pay.scrub(pay.compact(body), policy))


# --- 2. ctxdiff_diff -----------------------------------------------------------


def ctxdiff_diff(source: Source, policy: TextPolicy, run: str, turn_a: int,
                 turn_b: int, agent: str | None = None) -> str:
    """The block diff between two turns, via `diff_turns` — the same call
    `ctxdiff diff --turn A --turn B` makes.

    What comes back, and why: added/evicted/modified blocks individually (with
    hash, label, role, token count and a bounded excerpt), and UNCHANGED blocks
    only as a count and a token total. The unchanged blocks are the bulk of any
    context window and, by definition, not the answer to "what changed" — so
    enumerating them would spend most of the result restating what the agent
    already knows."""
    view, ref = open_run(source, run)
    try:
        _check_agent(view, agent)
        if agent is not None:
            _require_turn(view, turn_a, agent)
            _require_turn(view, turn_b, agent)
        diff = diff_turns(view, turn_a, turn_b)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    finally:
        view.close()

    # The four kinds partition the diff: added + evicted + modified + unchanged
    # is exactly the number of entries. `unchanged_moved` is deliberately NOT a
    # fifth kind — it is a SUBSET of `unchanged`, counting blocks that survived
    # the turn with the same content at a different position (history that slid
    # up as a system prompt grew). Named for what it is, because a bare "moved"
    # alongside the four reads as a fifth category and makes them look like they
    # do not add up; the CLI has no "moved" concept at all.
    counts = {"added": 0, "evicted": 0, "modified": 0, "unchanged": 0,
              "unchanged_moved": 0}
    changes = []
    unchanged_tokens = 0
    for entry in diff.entries:
        counts[entry.kind] += 1
        if entry.kind == "unchanged":
            unchanged_tokens += entry.block.token_count
            if entry.position_old != entry.position_new:
                counts["unchanged_moved"] += 1
            continue
        changes.append(_change_row(entry, policy))

    body = _envelope(ref, agent) | {
        "turn_a": diff.seq_old,
        "turn_b": diff.seq_new,
        "tokens_added": diff.tokens_added,
        "tokens_evicted": diff.tokens_evicted,
        "net_tokens": diff.tokens_added - diff.tokens_evicted,
        "counts": counts,
        "unchanged_tokens": unchanged_tokens,
        "changes": changes,
        "hint": ("content_hash values are prefixes — pass one to "
                 "ctxdiff_block(run, content_hash) for a block's full text"),
    }
    return pay.fit(pay.scrub(pay.compact(body), policy))


def _change_row(entry, policy: TextPolicy) -> dict:
    """One changed block, as the diff tools report it: what happened to it,
    what it was labeled, where it sat, what it cost, and the smallest excerpt
    that makes it recognizable — a changed hunk for a modified block, a
    one-line preview for an added or evicted one."""
    row = {
        "kind": entry.kind,
        "label": entry.label,
        "role": entry.block.role,
        "content_hash": pay.short_hash(entry.block.content_hash),
        "tokens": entry.block.token_count,
        "position": (entry.position_new if entry.position_new is not None
                     else entry.position_old),
    }
    if entry.kind == "modified":
        row["tokens_before"] = entry.old_block.token_count
        row["hunk"] = _changed_hunk(entry.inline_diff, policy)
    else:
        row["preview"] = policy.preview(entry.block.text)
    return row


# --- 3. ctxdiff_tokens ---------------------------------------------------------


def ctxdiff_tokens(source: Source, policy: TextPolicy, run: str,
                   turn: int | None = None, agent: str | None = None) -> str:
    """Token attribution, via `analyze_run` plus `analyze_evictions` — the same
    two calls `ctxdiff tokens` makes, in the same order, so the per-turn totals
    here are character-for-character the numbers the CLI prints."""
    view, ref = open_run(source, run)
    try:
        _check_agent(view, agent)
        if turn is not None:
            _require_turn(view, turn, agent)
        report = analyze_run(view, agent=agent)
        evictions = analyze_evictions(view, agent=agent)
        total_tools = None
        if report.bloat is not None and report.bloat.unused_tools:
            total_tools = len(registered_tool_names(
                [view.get_call_blocks(c.id) for c in view.get_calls()]))
    finally:
        view.close()

    calls = ([c for c in report.calls if c.seq == turn] if turn is not None
             else report.calls)
    peak = max(report.calls, key=lambda c: c.total_tokens, default=None)

    body = _envelope(ref, agent) | {
        "turns": [_turn_tokens_row(c) for c in calls],
        # The peak carries its own approximation flags. A per-turn row that says
        # "(~approx), 1 unmeasured block" and a peak that states the same number
        # bare invites the reader to treat the peak as exact — and the peak is
        # precisely the number a budget gets checked against, so a FLOOR
        # presented as a measurement is the worst place for that to happen.
        "peak": ({"turn": peak.seq, "total_tokens": peak.total_tokens,
                  "approximate": peak.approximate or None,
                  "unmeasured_blocks": peak.unmeasured_blocks or None}
                 if peak is not None else None),
        "by_agent": report.by_agent,
        "provider_usage": {
            "input_tokens": report.usage.input_tokens,
            "output_tokens": report.usage.output_tokens,
            "calls_with_usage": report.usage.calls_with_usage,
            "calls_total": report.usage.calls_total,
        },
        "schema_bloat": _bloat_row(report.bloat, total_tools, policy),
        "tagged_evictions": [_eviction_row(e, policy)
                             for e in evictions.evictions
                             if turn is None or e.evicted_seq == turn],
    }
    return pay.fit(pay.scrub(pay.compact(body), policy))


def _turn_tokens_row(call) -> dict:
    """One turn's attribution: its total, whether that total is approximate
    (any estimated block) or an outright FLOOR (`unmeasured_blocks` — content
    whose cost could not be known at all, e.g. an image we refused to fetch),
    and where the budget went by label. No captured text — a label breakdown is
    structure, which is exactly what a redacted server can still answer with."""
    return {
        "turn": call.seq,
        "agent": call.agent,
        "step": call.step,
        "total_tokens": call.total_tokens,
        "approximate": call.approximate or None,
        "unmeasured_blocks": call.unmeasured_blocks or None,
        "by_label": [{"label": s.label, "tokens": s.tokens, "pct": s.pct,
                      "blocks": s.block_count} for s in call.slices],
        "provider_prompt_tokens_delta": call.reconciliation_delta,
    }


def _bloat_row(bloat, total_tools: int | None, policy: TextPolicy) -> dict | None:
    """Dead tool schemas: registered on every single call, never invoked once —
    a recurring per-turn tax an agent can act on immediately by deleting them.
    None when the run registers no schemas at all (nothing to report about).

    A tool NAME is captured text, however much it looks like configuration. It
    is a JSON `"name"` lifted out of a recorded tool_schema block — whatever the
    traced application put on the wire, which may be whatever an attacker got it
    to put there. Returned raw, it was the one string in this whole surface that
    reached the debugging agent with no fence around it and no ANSI stripping,
    and it survived `--redact` besides. So each name is flattened, fenced and
    defanged like every other quoted excerpt, and under `--redact` only the
    COUNT is reported — which still names the finding and still sizes it, in
    tokens per call, without quoting anything."""
    if bloat is None or not bloat.unused_tools:
        return None
    row = {
        "unused_tools_count": len(bloat.unused_tools),
        "unused_tokens_per_call": bloat.unused_tokens_per_call,
        "pct_of_avg_context": bloat.pct_of_avg_context,
        "registered_tools": total_tools,
        "note": ("these tool schemas are re-sent on every call and never "
                 "invoked anywhere in the run"),
    }
    if not policy.redact:
        row["unused_tools"] = [pay.fence(pay.flatten(name, _NAME_CHARS))
                               for name in bloat.unused_tools]
    return row


def _eviction_row(eviction, policy: TextPolicy) -> dict:
    """One block the developer explicitly vouched for with `tracer.tag()` that
    entered the context and later left it for good — usually the direct answer
    to "the agent forgot the thing I gave it"."""
    return {
        "label": eviction.label,
        "agent": eviction.agent,
        "tagged_turn": eviction.tagged_seq,
        "entered_turn": eviction.entered_seq,
        "last_seen_turn": eviction.last_seen_seq,
        "evicted_turn": eviction.evicted_seq,
        "tokens": eviction.tokens,
        "role": eviction.role,
        "content_hash": pay.short_hash(eviction.content_hash),
        "preview": policy.preview(eviction.snippet),
    }


# --- 4. ctxdiff_cache ----------------------------------------------------------


def ctxdiff_cache(source: Source, policy: TextPolicy, run: str,
                  agent: str | None = None) -> str:
    """Prompt-cache prefix stability, via `analyze_cache` — the same call
    `ctxdiff cache` makes, including its per-agent grouping (a hand-off between
    two agents is never miscounted as a broken prefix)."""
    view, ref = open_run(source, run)
    try:
        _check_agent(view, agent)
        report = analyze_cache(view, agent=agent)
    finally:
        view.close()

    body = _envelope(ref, agent) | {
        "pairs_analyzed": report.pairs_analyzed,
        "agents_analyzed": report.agents_analyzed,
        "stable_prefix_tokens_min": report.stable_prefix_tokens_min,
        "rebilled_tokens_total": report.rebilled_tokens_total,
        "waste_note": report.estimated_waste_note,
        "fix_hint": report.fix_hint,
        "breaks": [_break_row(b, report, policy) for b in report.breaks],
    }
    if not report.breaks and report.pairs_analyzed:
        body["verdict"] = (f"prefix stable across all {report.pairs_analyzed} "
                           "turn pairs")
    return pay.fit(pay.scrub(pay.compact(body), policy))


def _break_row(brk, report, policy: TextPolicy) -> dict:
    """One prefix break: the pair it happened between, how much prefix survived,
    and which block is responsible. `detail` and `preview` quote CAPTURED text
    (that is the whole value — "the timestamp in your system prompt moved"), so
    both are fenced and both disappear under `--redact`; the position, kind and
    label survive, which is still enough to locate the culprit block."""
    return {
        "turn_prev": brk.seq_prev,
        "turn": brk.seq,
        "agent": brk.agent,
        "stable_blocks": brk.stable_blocks,
        "stable_tokens": brk.stable_tokens,
        "divergent_position": brk.divergent_position,
        "culprit_kind": brk.culprit_kind,
        "culprit_label": brk.culprit_label,
        "pairs_denominator": pairs_denominator(report, brk),
        "detail": policy.preview(brk.detail, pay.HUNK_CHARS),
        "preview": policy.preview(brk.culprit_snippet),
    }


# --- 5. ctxdiff_block ----------------------------------------------------------


def ctxdiff_block(source: Source, policy: TextPolicy, run: str,
                  content_hash: str, offset: int = 0,
                  max_chars: int = pay.BLOCK_CHARS_DEFAULT) -> str:
    """One block's full text, on demand and PAGED.

    This is the deliberate escape hatch from the token discipline everywhere
    else: the other tools return hashes and previews precisely so that reading a
    whole block is an explicit, bounded decision. `max_chars` is clamped to
    `BLOCK_CHARS_MAX`, the returned slice is shrunk further if JSON escaping
    would push the result past the cap, and `next_offset` tells the caller
    whether there is more — so a 50 KB retrieved document can be read in pieces
    by an agent that decides it needs to, and can never arrive unasked.

    Under `--redact` the metadata still answers "how big is it, what is it
    labeled, which turns carried it" while the text itself is withheld."""
    view, ref = open_run(source, run)
    try:
        found = _find_block(view, content_hash)
    finally:
        view.close()
    block, label, turns = found

    offset = max(0, offset)
    limit = max(1, min(int(max_chars), pay.BLOCK_CHARS_MAX))
    total = len(block.text)

    def build(chars: int) -> dict:
        """Assemble the payload for a slice of `chars` characters, keeping the
        paging fields honest about exactly what this slice contains — the
        fitting loop below calls it repeatedly with smaller values."""
        slice_text = block.text[offset:offset + chars]
        end = offset + len(slice_text)
        return pay.scrub(pay.compact(_envelope(ref) | {
            "content_hash": block.content_hash,
            "role": block.role,
            "kind": block.kind,
            "label": label,
            "tokens": block.token_count,
            "token_method": block.token_method,
            "chars_total": total,
            "offset": offset,
            "chars_returned": len(slice_text),
            "next_offset": end if end < total else None,
            "truncated": end < total or None,
            "turns_present": turns[:_MAX_TURNS_LISTED],
            "turns_present_total": len(turns),
            "text": policy.body(slice_text),
        }), policy)

    return _fit_block(build, limit)


def _fit_block(build, chars: int) -> str:
    """Encode `build(chars)`, shrinking the slice until the ENCODED result fits
    under the cap.

    Why the slice and not the generic shrinker: paging fields have to stay true.
    Halving a text field after the fact would leave `chars_returned` and
    `next_offset` describing a slice that was not returned, and an agent paging
    through a document on those numbers would silently skip the middle of it. So
    the whole payload is rebuilt at the smaller size instead. The step is
    proportional (scale by how far over we are, then take a little more off), so
    even text that escapes to several bytes per character converges in a couple
    of iterations."""
    while True:
        encoded = pay.encode(build(chars))
        over = pay.size(encoded)
        if over <= pay.MAX_RESULT_BYTES or chars <= 0:
            return encoded
        scaled = chars * pay.MAX_RESULT_BYTES // over
        chars = max(0, min(chars - 64, scaled - 64))


def _find_block(view: _SessionView, content_hash: str):
    """Locate one block in this session by content-hash PREFIX, returning it
    with its label and every turn that carried it.

    Prefix rather than full hash because every other tool quotes prefixes; a
    minimum length is enforced so a one-character "hash" cannot match half the
    run, and an ambiguous prefix is an error listing the candidates rather than
    a guess. The scan is over the whole session because the server is stateless
    — there is no index to keep warm, and a session is small enough that the
    walk costs less than a connection would."""
    wanted = content_hash.strip().lower()
    # Quoted back bounded, for the same reason `_match_run` bounds `run`: this
    # argument is caller-supplied and an error result is not capped by `fit()`.
    shown = pay.flatten(wanted, _NAME_CHARS)
    if len(wanted) < 6:
        raise ToolError("content_hash must be at least 6 hex characters — copy "
                        "one from a ctxdiff_diff or ctxdiff_tokens result")

    matches: dict[str, tuple] = {}
    turns: dict[str, list[int]] = {}
    for call in view.get_calls():
        for cb in view.get_call_blocks(call.id):
            digest = cb.block.content_hash
            if not digest.lower().startswith(wanted):
                continue
            matches.setdefault(digest, (cb.block, cb.label))
            seqs = turns.setdefault(digest, [])
            if call.seq not in seqs:
                seqs.append(call.seq)

    if not matches:
        raise ToolError(f"no block in this run whose content hash starts with "
                        f"{shown!r} — hashes are per-run; check the run "
                        f"argument, or re-read the diff that named it")
    if len(matches) > 1:
        raise ToolError(
            f"{shown!r} matches {len(matches)} distinct blocks "
            f"({_listed(sorted(pay.short_hash(h) for h in matches), 20)}) — "
            "pass more characters")

    digest, (block, label) = next(iter(matches.items()))
    return block, label, sorted(turns[digest])


# --- 6. ctxdiff_explain --------------------------------------------------------


def ctxdiff_explain(source: Source, policy: TextPolicy, run: str, turn: int,
                    agent: str | None = None) -> str:
    """The composite: all three analyses for one turn, in one call, summarized.

    This is the tool that gets invoked from "why did my agent break at turn 8",
    and it exists because agents overwhelmingly prefer one call to three —
    a surface that required `ctxdiff_diff` + `ctxdiff_tokens` + `ctxdiff_cache`
    to answer the only question anyone actually asks would mostly go unused.

    "The previous turn" is the previous turn of the SAME AGENT, not the previous
    global turn number: in an interleaved multi-agent run those differ, and
    diffing an agent's context against a different agent's context would report
    the entire window as changed — a spectacular, confident, useless answer."""
    view, ref = open_run(source, run)
    try:
        _check_agent(view, agent)
        target = _require_turn(view, turn, agent)
        scope = agent if agent is not None else target.agent
        previous = _previous_turn(view, target)

        report = analyze_run(view, agent=scope)
        cache = analyze_cache(view, agent=scope)
        evictions = analyze_evictions(view, agent=scope)
        diff = (diff_turns(view, previous.seq, turn) if previous is not None
                else None)
    finally:
        view.close()

    this_turn = next((c for c in report.calls if c.seq == turn), None)
    breaks_here = [b for b in cache.breaks if b.seq == turn]
    evicted_here = [e for e in evictions.evictions if e.evicted_seq == turn]

    diff_summary = None
    if diff is not None:
        counts = {"added": 0, "evicted": 0, "modified": 0, "unchanged": 0}
        for entry in diff.entries:
            counts[entry.kind] += 1
        changed = [e for e in diff.entries if e.kind != "unchanged"]
        # Biggest movers first: the agent gets the five blocks most likely to
        # explain the failure, not the five that happen to sit earliest.
        changed.sort(key=lambda e: e.block.token_count, reverse=True)
        diff_summary = {
            "vs_turn": previous.seq,
            "counts": counts,
            "tokens_added": diff.tokens_added,
            "tokens_evicted": diff.tokens_evicted,
            "top_changes": [_change_row(e, policy)
                            for e in changed[:_EXPLAIN_CHANGES]],
        }

    body = _envelope(ref, agent) | {
        "turn": turn,
        "agent_of_turn": target.agent,
        "summary": _explain_summary(turn, target, this_turn, diff, previous,
                                    breaks_here, evicted_here, report),
        "tokens": (_turn_tokens_row(this_turn) if this_turn is not None else None),
        "diff_vs_previous_turn": diff_summary,
        "cache": {
            "pairs_analyzed": cache.pairs_analyzed,
            "stable_prefix_tokens_min": cache.stable_prefix_tokens_min,
            "rebilled_tokens_total": cache.rebilled_tokens_total,
            "fix_hint": cache.fix_hint,
            "breaks_at_this_turn": [_break_row(b, cache, policy)
                                    for b in breaks_here],
        },
        "tagged_evictions_at_this_turn": [_eviction_row(e, policy)
                                          for e in evicted_here],
        "schema_bloat": _bloat_row(report.bloat, None, policy),
        "next_steps": [
            "ctxdiff_diff(run, turn_a, turn_b) for every changed block, not "
            "just the largest",
            "ctxdiff_block(run, content_hash) for a block's full text",
        ],
    }
    if previous is None:
        body["note"] = ("this is the agent's first turn in the run — there is "
                        "no previous context to diff it against")
    return pay.fit(pay.scrub(pay.compact(body), policy))


def _previous_turn(view: _SessionView, target):
    """The call this agent made immediately before `target`, or None when
    `target` is its first. Same-agent rather than same-session for the reason
    given in `ctxdiff_explain`'s docstring."""
    earlier = [c for c in view.get_calls()
               if c.agent == target.agent and c.seq < target.seq]
    return earlier[-1] if earlier else None


def _explain_summary(turn: int, target, this_turn, diff, previous,
                     breaks_here, evicted_here, report) -> str:
    """One sentence naming what changed at this turn and what, if anything,
    looks wrong with it.

    Composed from NUMBERS and ctxdiff's own vocabulary only — never from
    captured text — so it is safe to read on a redacted server and carries no
    injected instruction. It leads with the cost, then the delta, then the two
    findings that most often explain a bad turn (a tagged block that vanished,
    a broken cache prefix), then dead schemas.

    Bounded at both ends, and that is a correctness property rather than a
    nicety. This is a plain STRING: `fit()` cannot trim items out of it the way
    it trims a list, and it is not captured text, so nothing drops it either. A
    clause per eviction meant one turn that dropped a hundred tagged blocks
    produced a summary bigger than the whole result cap, with no way down —
    which is how a server ended up spinning instead of answering. So the
    findings are named up to `_SUMMARY_FINDINGS` and then COUNTED, developer
    labels are flattened, and the assembled sentence is cut to
    `_SUMMARY_CHARS`."""
    parts = []
    if this_turn is not None:
        total = f"{this_turn.total_tokens:,} tok"
        if this_turn.approximate:
            # Spelled exactly as `ctxdiff tokens` prints it (cli/render.py).
            # Four renderers, one vocabulary: a reader comparing the terminal
            # to the MCP result must not have to wonder whether `(approx)` and
            # `(~approx)` mean two different things.
            total += " (~approx)"
        parts.append(f"turn {turn} carried {total}")
    if diff is not None and previous is not None:
        net = diff.tokens_added - diff.tokens_evicted
        counts = {}
        for entry in diff.entries:
            counts[entry.kind] = counts.get(entry.kind, 0) + 1
        parts.append(
            f"{net:+,} tok vs turn {previous.seq} "
            f"({counts.get('added', 0)} added, {counts.get('evicted', 0)} "
            f"evicted, {counts.get('modified', 0)} modified)")
    else:
        parts.append("no previous turn for this agent to compare against")
    for e in evicted_here[:_SUMMARY_FINDINGS]:
        parts.append(f"the block tagged "
                     f"'{pay.flatten(e.label or '', _NAME_CHARS)}' at turn "
                     f"{e.tagged_seq} ({e.tokens:,} tok) was evicted here")
    if len(evicted_here) > _SUMMARY_FINDINGS:
        parts.append(f"{len(evicted_here) - _SUMMARY_FINDINGS} more tagged "
                     f"block(s) were evicted at this turn — see "
                     f"tagged_evictions_at_this_turn")
    for b in breaks_here[:_SUMMARY_FINDINGS]:
        parts.append(f"the prompt-cache prefix broke at position "
                     f"{b.divergent_position} "
                     f"([{pay.flatten(b.culprit_label or '', _NAME_CHARS)}·"
                     f"{b.culprit_kind}]), re-billing everything after it")
    if len(breaks_here) > _SUMMARY_FINDINGS:
        parts.append(f"{len(breaks_here) - _SUMMARY_FINDINGS} more prefix "
                     f"break(s) at this turn")
    if report.bloat is not None and report.bloat.unused_tools:
        parts.append(f"{len(report.bloat.unused_tools)} tool schema(s) are "
                     f"re-sent every call but never invoked "
                     f"({report.bloat.unused_tokens_per_call:,} tok/call)")
    return pay.flatten("; ".join(parts), _SUMMARY_CHARS)
