r"""The exporter: turn a `.ctrace` (or a database-backed project) into one
self-contained, THREE-LEVEL HTML dashboard.

Three pieces live here, deliberately split:

- `build_payload(ct)` is a PURE function that assembles every fact ONE
  session's detail view (level 3) needs into one JSON-serializable dict. It
  embeds PRECOMPUTED analyzer output (the block differ, the token attributor,
  the cache profiler — the same Python functions the CLI uses) rather than
  re-implementing any of that logic in JavaScript, so there is exactly one
  source of truth for every number the dashboard shows. Being pure, it is
  trivially unit-testable without touching the filesystem or the template.

- `build_project_payload(reader, ...)` wraps that with the PROJECT index the
  three-level dashboard navigates: level 1 lists every agent across every
  session, level 2 lists the sessions one agent appeared in, level 3 is the
  detail `build_payload` produces. The focus session's detail sits at the TOP
  LEVEL (so the payload of a single-session project is exactly what it always
  was, plus one `project` key), and the other embedded sessions' details hang
  off `project.details`.

- `export_html(...)` / `export_store(...)` are the thin I/O shells: open the
  trace, build the payload, serialize it, inject it into the page template, and
  write the file.

WHAT IS AND ISN'T EMBEDDED (the scale decision). A project database can hold
thousands of sessions and the artifact must stay ONE file with no network, so
the two axes are separated by cost:

- Levels 1 and 2 are AGGREGATES over every session, computed from the session
  list and each session's CALL rows alone — no block reads. Their cost is
  O(calls), the same read `ctxdiff agents` already does, so every session in
  the project is always listed and no agent is ever hidden.
- Level 3 is the block-level detail, which is what actually has size, so it is
  embedded only for the `_DETAIL_SESSIONS` most RECENT sessions (plus the focus
  session, always, so `--session` can name an old run and still get its detail).
  Older sessions still appear in level 2 with their aggregates; the page marks
  them and states the cap rather than pretending they are missing.

Security note (enforced here, tested in test_viewer): the payload is embedded
as a JSON island inside a `<script type="application/json">` tag, and every
`</` in the serialized JSON is escaped to `<\/`. That single substitution makes
it impossible for block text containing `</script>` (or any other markup) to
break out of the tag and become live HTML — the browser hands the island back
verbatim via `.textContent`, and the page renders all block text with
`textContent` (never `innerHTML`), so untrusted trace data can never execute.
The same holds for AGENT NAMES and session labels, which levels 1/2 and the
breadcrumb render: every one of them reaches the DOM through `textContent`."""
from __future__ import annotations

import html
import json
import os

from ctxdiff.analyze.cache import CacheReport, analyze_cache
from ctxdiff.analyze.differ import TurnDiff, diff_turns, distinct_agents
from ctxdiff.analyze.tokens import RunTokens, analyze_run, usage_totals
from ctxdiff.store.ctrace import CTrace
from ctxdiff.store.base import Call, Session, Store
from ctxdiff.viewer.template import render_page

# How much of a diff entry's block text to embed as its snippet, and how much
# of each inline-diff segment to keep — the dashboard shows previews, not whole
# documents, and capping here keeps the embedded JSON from ballooning on runs
# with very large blocks.
_SNIPPET_CHARS = 200
_INLINE_SEG_CHARS = 500

# How many sessions get their FULL level-3 detail (every block of every turn,
# every diff, the cache and token analyses) embedded in the artifact — the most
# recent ones, plus the focus session whether or not it is among them.
#
# Why a cap at all: the dashboard is one self-contained file with no network, so
# every byte of detail is paid up front. A 40-turn session's detail is on the
# order of a few hundred KB, so 25 keeps a project dashboard comfortably in the
# megabytes even when the database holds thousands of sessions — while still
# covering "the runs I did today", which is what anyone opening a dashboard is
# actually looking at. Levels 1 and 2 are NOT capped: every session in the
# project is listed, and the page says so when a listed session's detail was not
# embedded.
_DETAIL_SESSIONS = 25

# The bucket name for calls that carry no agent label. Duplicated from
# `cli.select.UNLABELED` rather than imported because the viewer must not depend
# on the CLI; the two are asserted equal in the tests, so one run can never
# report two different names for the same calls.
_UNLABELED = "(unlabeled)"


# --- per-entity serializers ----------------------------------------------------


def _serialize_block(cb) -> dict:
    """One CallBlock -> a JSON dict for the blocks table. Carries the full text
    (the page truncates for preview and reveals the rest on demand) plus an
    8-char hash prefix — enough to eyeball block identity/dedup without showing
    a 64-char digest."""
    b = cb.block
    return {
        "position": cb.position,
        "label": cb.label,
        "label_source": cb.label_source,
        "role": b.role,
        "kind": b.kind,
        "token_count": b.token_count,
        "token_method": b.token_method,
        "text": b.text,
        "content_hash_short": b.content_hash[:8],
    }


def _serialize_call(call, call_blocks) -> dict:
    """One Call + its blocks -> a JSON dict. `params` is reduced to the model
    id ALONE — the raw params dict may carry temperature, api keys, or other
    sensitive fields, and none of that belongs in a shareable artifact — while
    provider `usage` (token counts only) is passed through for reconciliation."""
    return {
        "seq": call.seq,
        "latency_ms": call.latency_ms,
        "error": call.error,
        "agent": call.agent,
        "step": call.step,
        "params": {"model": (call.params or {}).get("model")},
        "usage": call.usage,
        "blocks": [_serialize_block(cb) for cb in call_blocks],
    }


def _serialize_diff_entry(e) -> dict:
    """One DiffEntry -> a JSON dict. Text is capped: the block snippet to
    _SNIPPET_CHARS, and each inline-diff segment to _INLINE_SEG_CHARS.
    `inline_diff` is null for anything but a 'modified' entry (only modifieds
    carry a char-level diff)."""
    inline = None
    if e.inline_diff is not None:
        inline = [[op, text[:_INLINE_SEG_CHARS]] for op, text in e.inline_diff]
    return {
        "kind": e.kind,
        "label": e.label,
        "role": e.block.role,
        "position_old": e.position_old,
        "position_new": e.position_new,
        "token_count": e.block.token_count,
        "snippet": e.block.text[:_SNIPPET_CHARS],
        "inline_diff": inline,
    }


def _serialize_diff(td: TurnDiff) -> dict:
    """One TurnDiff -> a JSON dict: the two turn numbers, the net token deltas,
    and every entry (unchanged included, so the panel can report how many
    blocks held steady, not only what moved)."""
    return {
        "seq_old": td.seq_old,
        "seq_new": td.seq_new,
        "tokens_added": td.tokens_added,
        "tokens_evicted": td.tokens_evicted,
        "entries": [_serialize_diff_entry(e) for e in td.entries],
    }


def _serialize_tokens(rt: RunTokens) -> dict:
    """A RunTokens -> a JSON dict: per-call label slices (with pct and the
    approximate flag) plus the run-level bloat report (null when the run has no
    tool schemas to critique)."""
    return {
        "calls": [
            {
                "seq": c.seq,
                "total": c.total_tokens,
                "approximate": c.approximate,
                "slices": [
                    {"label": s.label, "tokens": s.tokens, "pct": s.pct}
                    for s in c.slices
                ],
                "reconciliation_delta": c.reconciliation_delta,
            }
            for c in rt.calls
        ],
        "bloat": None if rt.bloat is None else {
            "unused_tools": rt.bloat.unused_tools,
            "unused_tokens_per_call": rt.bloat.unused_tokens_per_call,
            "pct_of_avg_context": rt.bloat.pct_of_avg_context,
        },
        # NB: per-agent BLOCK-token totals are intentionally NOT embedded — the
        # template renders per-agent numbers from stats.agents (tokens) and
        # stats.usage.by_agent (provider in/out), so a tokens.by_agent field
        # would be dead weight in the payload.
    }


def _serialize_cache(cr: CacheReport) -> dict:
    """A CacheReport -> a JSON dict: each prefix break with its culprit and the
    run-level stable/rebilled totals, plus the price-free waste note and the
    optional fix hint (both composed by the analyzer, embedded verbatim)."""
    return {
        "pairs_analyzed": cr.pairs_analyzed,
        "breaks": [
            {
                "seq_prev": b.seq_prev,
                "seq": b.seq,
                "stable_tokens": b.stable_tokens,
                "divergent_position": b.divergent_position,
                "culprit_kind": b.culprit_kind,
                "culprit_label": b.culprit_label,
                "culprit_snippet": b.culprit_snippet,
                "detail": b.detail,
                "agent": b.agent,
            }
            for b in cr.breaks
        ],
        "stable_prefix_tokens_min": cr.stable_prefix_tokens_min,
        "rebilled_tokens_total": cr.rebilled_tokens_total,
        "waste_note": cr.estimated_waste_note,
        "fix_hint": cr.fix_hint,
        "agents_analyzed": cr.agents_analyzed,
    }


# --- payload assembly ----------------------------------------------------------


def build_payload(ct: Store) -> dict:
    """Assemble every fact the dashboard renders into one JSON-serializable
    dict (spec §7.2, amended). Pure: it reads from `ct` and the three analyzers
    and returns a dict — no filesystem, no template, no HTML.

    How: load the run, its calls, and each call's blocks once; then delegate
    the hard analysis to the existing pure analyzers (diff_turns per adjacent
    pair, analyze_run once, analyze_cache once) and serialize their frozen
    dataclasses into plain dicts. `stats` is computed here directly — distinct
    block count vs. total references (the dedup story) and per-turn context
    growth (peak/trend for the scrubber and the growth chart)."""
    run = ct.get_run()
    calls = ct.get_calls()
    blocks_by_call = {c.id: ct.get_call_blocks(c.id) for c in calls}

    # calls, in send order
    calls_out = [_serialize_call(c, blocks_by_call[c.id]) for c in calls]

    # diffs: one per adjacent (N-1, N) turn pair, in global seq order. Each
    # entry gains `cross_agent` (the pair spans two agents — the UI renders it
    # as an agent hand-off, not a normal diff). When it IS a hand-off, and the
    # newer call's agent has an earlier call of its OWN, also precompute
    # `same_agent_diff` (the diff vs that same-agent predecessor) so the panel
    # can show a meaningful "vs this agent's previous turn" diff; it is omitted
    # when there is no earlier same-agent call to compare against.
    diffs_out = []
    for i in range(1, len(calls)):
        prev, cur = calls[i - 1], calls[i]
        d = _serialize_diff(diff_turns(ct, prev.seq, cur.seq))
        d["cross_agent"] = (prev.agent != cur.agent)
        if d["cross_agent"]:
            same_prev = next((calls[j] for j in range(i - 1, -1, -1)
                              if calls[j].agent == cur.agent), None)
            if same_prev is not None:
                d["same_agent_diff"] = _serialize_diff(
                    diff_turns(ct, same_prev.seq, cur.seq))
        diffs_out.append(d)

    # precomputed analyzer output (single source of truth). analyze_cache with
    # agent=None auto-groups per agent, so cross-agent hand-offs are never
    # miscounted as cache breaks.
    run_tokens = analyze_run(ct)
    tokens_out = _serialize_tokens(run_tokens)
    cache_out = _serialize_cache(analyze_cache(ct))

    # stats: dedup ratio + context growth
    distinct: set[str] = set()
    total_refs = 0
    for cbs in blocks_by_call.values():
        for cb in cbs:
            distinct.add(cb.block.content_hash)
            total_refs += 1
    # context growth reuses the token attributor's per-call totals so the
    # scrubber, the growth chart, and the token panel all agree on one number.
    growth = [c.total_tokens for c in run_tokens.calls]

    # per-agent stats for the header chips: name (unlabeled calls bucketed),
    # call count, and total block tokens — in first-appearance order.
    tokens_by_call = {c.id: sum(cb.block.token_count for cb in blocks_by_call[c.id])
                      for c in calls}
    agents_stats = []
    for label in distinct_agents(calls):
        group = [c for c in calls if c.agent == label]
        agents_stats.append({
            "name": label if label is not None else "(unlabeled)",
            "calls": len(group),
            "tokens": sum(tokens_by_call[c.id] for c in group),
        })

    # run-level provider-usage rollup (input/output token totals, coverage as
    # [reported, total], and per-agent [in, out] when multi-agent) — surfaced
    # in the header beside the block-token total.
    u = run_tokens.usage
    usage_stats = {
        "input": u.input_tokens,
        "output": u.output_tokens,
        "coverage": [u.calls_with_usage, u.calls_total],
        "by_agent": None if u.by_agent is None else {
            name: [inp, outp] for name, (inp, outp) in u.by_agent.items()
        },
    }

    return {
        "run": {
            "project": run.project,
            "provider": run.provider,
            "started_at": run.started_at,
            "ctxdiff_version": run.ctxdiff_version,
            "models": run.models,
        },
        "calls": calls_out,
        "diffs": diffs_out,
        "tokens": tokens_out,
        "cache": cache_out,
        "stats": {
            "distinct_blocks": len(distinct),
            "total_block_refs": total_refs,
            "context_growth": growth,
            "agents": agents_stats,
            "usage": usage_stats,
        },
    }


# --- project index (levels 1 and 2) ---------------------------------------------


class _PinnedReader:
    """A read handle PINNED to one session of a many-session store.

    Why it exists: `build_payload` and all three analyzers call
    `get_run()`/`get_calls()` with NO argument and get whatever session the
    handle is bound to. Building the detail of a session OTHER than that one
    therefore needs a reader whose no-argument reads answer for the session we
    chose — which is exactly this, a two-line forwarder that substitutes the
    session id on the two reads that take one and passes the rest through.

    It deliberately does not close anything: the caller owns the underlying
    reader and hands out several pins over its lifetime."""

    def __init__(self, reader, session_id: str):
        self._reader = reader
        self._session_id = session_id

    def get_run(self, session_id: str | None = None):
        """The pinned session's run row (an explicit id still wins)."""
        return self._reader.get_run(session_id or self._session_id)

    def get_calls(self, session_id: str | None = None):
        """The pinned session's calls, in turn order."""
        return self._reader.get_calls(session_id or self._session_id)

    def get_call_blocks(self, call_id: str):
        """One call's blocks — call ids are globally unique, so no pinning."""
        return self._reader.get_call_blocks(call_id)


def _agent_name(call: Call) -> str:
    """One call's agent bucket: its label, or `(unlabeled)` — the same bucket
    name the token analyzer and `ctxdiff agents` use, so the dashboard's level-1
    listing and the CLI's can never disagree about who made a call."""
    return call.agent if call.agent else _UNLABELED


def _usage_fields(calls: list[Call]) -> dict:
    """Provider-reported token spend for `calls`, as the three payload fields
    levels 1 and 2 render: summed input, summed output, and how many calls
    actually REPORTED usage.

    `reported` earns its place because 0 and "nobody told us" are different
    facts: a row showing `0 tokens` reads as "this agent was free", which is a
    lie about a provider that simply returned no usage block. The page renders
    `-` when `reported` is 0, exactly as `ctxdiff agents` does."""
    totals = usage_totals(calls)
    return {
        "input": totals.input_tokens,
        "output": totals.output_tokens,
        "reported": totals.calls_with_usage,
    }


def _agent_index(sessions: list[Session], calls_by_session: dict[str, list[Call]]) -> list[dict]:
    """LEVEL 1: every agent in the project with its footprint aggregated across
    ALL sessions — how many sessions it appears in, how many calls it made, what
    it cost, and the span of time it ran over.

    Order is first appearance scanning sessions oldest-first, the same order
    `ctxdiff agents` lists them in, so the CLI listing and the dashboard's
    landing view read identically.

    `first_seen`/`last_seen` are the RAW stored UTC timestamps of the oldest and
    newest session the agent appears in; the page converts them to the viewer's
    local zone at render time (see the template's `localTime`), because the
    person opening the file may well be in a different zone than the machine
    that captured it.

    They are the MIN and MAX of the collected timestamps, not the ends of the
    list. The list is in INSERT order, and insert order is not chronological
    order the moment two capturing machines' clocks disagree or a session is
    backfilled — which showed as `first seen 2026-05-01 ... last seen
    2026-03-01`, a span running backwards. (Session ORDERING is a different
    question and is deliberately still insert-based: see `_session_index`.)"""
    order: list[str] = []
    calls_by_agent: dict[str, list[Call]] = {}
    session_counts: dict[str, int] = {}
    spans: dict[str, list[str]] = {}
    for s in sessions:
        seen_here: set[str] = set()
        for c in calls_by_session.get(s.id, []):
            name = _agent_name(c)
            if name not in calls_by_agent:
                order.append(name)
                calls_by_agent[name] = []
            calls_by_agent[name].append(c)
            if name not in seen_here:
                seen_here.add(name)
                session_counts[name] = session_counts.get(name, 0) + 1
                spans.setdefault(name, []).append(s.started_at)

    rows = []
    for name in order:
        calls = calls_by_agent[name]
        seen = spans[name]
        rows.append({
            "name": name,
            "sessions": session_counts[name],
            "calls": len(calls),
            **_usage_fields(calls),
            "first_seen": min(seen),
            "last_seen": max(seen),
        })
    return rows


def _session_index(sessions: list[Session], calls_by_session: dict[str, list[Call]],
                   detailed: set[str]) -> list[dict]:
    """LEVEL 2: every session in the project, NEWEST FIRST, with the per-agent
    breakdown that makes a row answer "what did THIS agent do in that run".

    `detail` says whether this session's block-level level-3 view is embedded in
    this file (see `_DETAIL_SESSIONS`). A row without it is still shown with all
    of its aggregates — hiding an old run entirely would make the project look
    smaller than it is — and the page marks it as not drillable and names the
    cap, so the omission is a stated fact rather than a mystery.

    `started_at` stays the RAW stored UTC string; the page renders it in the
    VIEWER's local zone (see `_agent_index`)."""
    rows = []
    for s in reversed(sessions):
        calls = calls_by_session.get(s.id, [])
        order: list[str] = []
        by_agent: dict[str, list[Call]] = {}
        for c in calls:
            name = _agent_name(c)
            if name not in by_agent:
                order.append(name)
                by_agent[name] = []
            by_agent[name].append(c)
        rows.append({
            "id": s.id,
            "started_at": s.started_at,
            "provider": s.provider,
            "models": s.models,
            "turn_count": s.turn_count,
            "detail": s.id in detailed,
            "agents": [{"name": name, "turns": len(by_agent[name]),
                        **_usage_fields(by_agent[name])}
                       for name in order],
        })
    return rows


def _start_level(agent: str | None, session_selected: bool,
                 agent_rows: list[dict], session_rows: list[dict],
                 focus_id: str) -> dict:
    """Which of the three levels the dashboard OPENS on, and what it opens
    scoped to.

    The rule is "never make someone click through a choice they don't have, and
    never guess one they do":

    - `--agent` AND `--session` name a single detail view: open on level 3,
      scoped to that agent.
    - `--agent` alone: open on its session list (level 2) — unless it ran in
      exactly one session, where that list has one row and would be pure
      friction, so open straight on level 3 scoped to it.
    - `--session` alone: open on that session's detail (level 3), all agents.
    - Neither: level 1, the project landing view — except for the common
      single-agent single-session project, which has nothing to choose between
      at either level above and so opens straight on level 3.

    `session` is carried only for the levels that have one, so the page never
    has to guess whether a session id in `start` is meaningful. When there IS
    one it is `focus_id` — the session this whole payload was built around, and
    the only one whose detail is guaranteed to be embedded. It must not be read
    off `session_rows`, which is NEWEST FIRST: doing that opened every
    `--session <older run>` export on the newest session instead, sending the
    page's `openSession(start.session)` at a run the user never named."""
    multi = len(session_rows) > 1 or len(agent_rows) > 1
    if agent is not None:
        agent_sessions = sum(1 for s in session_rows
                             if any(a["name"] == agent for a in s["agents"]))
        level = 3 if (session_selected or agent_sessions <= 1) else 2
    elif session_selected or not multi:
        level = 3
    else:
        level = 1
    return {"level": level, "agent": agent,
            "session": focus_id if level == 3 else None}


def build_project_payload(reader, *, agent: str | None = None,
                          session_selected: bool = False,
                          focus_session_id: str | None = None) -> dict:
    """Assemble the THREE-LEVEL dashboard payload: the focus session's detail at
    the top level (exactly the dict `build_payload` returns), plus one `project`
    key carrying the level-1 agent index, the level-2 session index, and the
    embedded details of the other sessions within the cap.

    Why the focus session's detail is at the TOP level rather than inside
    `project.details`: it keeps this payload a strict SUPERSET of the
    single-session one, so every existing consumer, test and assertion about
    `run`/`calls`/`diffs`/`tokens`/`cache`/`stats` keeps meaning what it meant,
    and the focus session's (large) detail is never serialized twice.

    `reader` must answer `list_sessions()`, `get_run()`, `get_calls(session_id)`
    and `get_call_blocks(call_id)` — satisfied by a `CTrace`, by the CLI's
    session-pinned view over any backend, and by an in-memory project snapshot
    of a Postgres/MySQL store. A reader that CANNOT list sessions (an older
    single-session snapshot) degrades to a one-session project built from the
    focus session alone rather than failing to export.

    Pure apart from the reads: no filesystem, no template, no HTML."""
    focus_run = reader.get_run(focus_session_id) if focus_session_id else reader.get_run()
    focus_id = focus_run.id

    sessions = _list_sessions_or_focus(reader, focus_run)
    # Every session's CALL rows — the input to both aggregate levels. Blocks are
    # NOT read here: that is the expensive axis and it is paid only for the
    # sessions whose detail is actually embedded (see the module docstring).
    calls_by_session = {s.id: reader.get_calls(s.id) for s in sessions}

    # The most recent `_DETAIL_SESSIONS` sessions get full detail, and so does
    # the focus session whether or not it made that cut — `--session <old run>`
    # must always land on a working level 3.
    newest_first = list(reversed(sessions))
    detailed = {s.id for s in newest_first[:_DETAIL_SESSIONS]}
    detailed.add(focus_id)

    agent_rows = _agent_index(sessions, calls_by_session)
    session_rows = _session_index(sessions, calls_by_session, detailed)

    # Non-focus details, newest first — the focus session's own detail is the
    # payload's top level and is never duplicated here.
    details = {s["id"]: build_payload(_PinnedReader(reader, s["id"]))
               for s in session_rows if s["detail"] and s["id"] != focus_id}

    payload = build_payload(_PinnedReader(reader, focus_id))
    payload["project"] = {
        "name": focus_run.project,
        "sessions_total": len(sessions),
        "detail_cap": _DETAIL_SESSIONS,
        "focus": focus_id,
        "start": _start_level(agent, session_selected, agent_rows, session_rows,
                              focus_id),
        "agents": agent_rows,
        "sessions": session_rows,
        "details": details,
    }
    return payload


def _list_sessions_or_focus(reader, focus_run) -> list[Session]:
    """Every session in the project, oldest first — or a one-element list
    describing just the focus session when the reader cannot list them.

    The fallback is what keeps `export_store` total: a reader that was
    materialized for ONE session (an in-memory snapshot taken without the
    session list) can still produce a perfectly correct dashboard of that one
    session, and refusing to export at all would be a strictly worse answer than
    a dashboard that shows what it has."""
    try:
        sessions = reader.list_sessions()
    except Exception:  # noqa: BLE001 — any listing failure degrades to the focus session
        sessions = []
    if sessions:
        return sessions
    calls = reader.get_calls(focus_run.id)
    return [Session(id=focus_run.id, project=focus_run.project,
                    started_at=focus_run.started_at, provider=focus_run.provider,
                    models=focus_run.models,
                    agents=[n for n in dict.fromkeys(
                        c.agent for c in calls if c.agent)],
                    turn_count=len(calls))]


# --- serialization + write -----------------------------------------------------


def _embed_json(payload: dict) -> str:
    """Serialize `payload` for embedding in a `<script type="application/json">`
    island. `ensure_ascii=False` keeps unicode readable rather than escaped;
    then every `</` becomes `<\\/` so no block text — not even a literal
    `</script>` — can terminate the script tag early. `<\\/` is a valid JSON
    escape of a solidus, so `JSON.parse` (and Python's json.loads) restores the
    original `</` exactly: the escaping is invisible to the data, load-bearing
    only for the parser boundary."""
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


def export_html(ctrace_path: str, out_path: str | None = None, *,
                agent: str | None = None, session_selected: bool = False) -> str:
    """Export the trace at `ctrace_path` to a self-contained HTML dashboard and
    return the written path. `out_path` overrides the destination; by default
    the file is written as `<trace-stem>.html` right next to the trace. Opens
    the trace read-only, builds the payload, embeds it in the page template
    (title = "ctxdiff — {project}"), and writes UTF-8.

    The dashboard covers the WHOLE project the file holds — every agent, every
    session — with the newest session in focus; `agent`/`session_selected`
    preselect which level it opens on (see `_start_level`).

    The file-specific half of exporting: it resolves the default output path
    from the TRACE's own path (which only a file-backed store has) and owns the
    handle's lifetime, then hands the open store to `export_store` — which is
    what a networked backend uses instead."""
    ct = CTrace.open(ctrace_path)
    try:
        if out_path is None:
            stem = os.path.splitext(os.path.basename(ctrace_path))[0]
            out_path = os.path.join(
                os.path.dirname(os.path.abspath(ctrace_path)), f"{stem}.html")
        return export_store(ct, out_path, agent=agent,
                            session_selected=session_selected)
    finally:
        ct.close()


def export_store(ct: Store, out_path: str | None = None, *,
                 agent: str | None = None, session_selected: bool = False) -> str:
    """Export an ALREADY-OPEN store handle to a self-contained HTML dashboard
    and return the written path — the backend-agnostic export path, used when
    the trace lives in Postgres/MySQL and there is no file to name.

    The handle's BOUND session becomes the dashboard's focus (its level-3
    detail); the project index around it covers every session the handle can
    list. `agent` and `session_selected` say which selectors the user actually
    passed, which is all `_start_level` needs to decide where to open.

    The caller owns the handle (this never closes it), because the caller is the
    one that knows whether the same handle is about to be used again. With no
    `out_path`, the dashboard is written as `./<project>.html`, the only
    sensible default when the source has no filename to borrow one from."""
    payload = build_project_payload(ct, agent=agent,
                                    session_selected=session_selected)
    project = payload["run"]["project"]
    document = render_page(project_title=html.escape(f"ctxdiff — {project}"),
                           data_json=_embed_json(payload))

    if out_path is None:
        out_path = f"{project}.html"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(document)
    return out_path
