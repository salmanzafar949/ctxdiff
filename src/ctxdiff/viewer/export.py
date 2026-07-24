r"""The exporter: turn a `.ctrace` into one self-contained HTML dashboard.

Two pieces live here, deliberately split:

- `build_payload(ct)` is a PURE function that assembles every fact the page
  needs into one JSON-serializable dict. It embeds PRECOMPUTED analyzer output
  (the block differ, the token attributor, the cache profiler — the same
  Python functions the CLI uses) rather than re-implementing any of that logic
  in JavaScript, so there is exactly one source of truth for every number the
  dashboard shows. Being pure, it is trivially unit-testable without touching
  the filesystem or the template.

- `export_html(...)` is the thin I/O shell: open the trace, build the payload,
  serialize it, inject it into the page template, and write the file.

Security note (enforced here, tested in test_viewer): the payload is embedded
as a JSON island inside a `<script type="application/json">` tag, and every
`</` in the serialized JSON is escaped to `<\/`. That single substitution makes
it impossible for block text containing `</script>` (or any other markup) to
break out of the tag and become live HTML — the browser hands the island back
verbatim via `.textContent`, and the page renders all block text with
`textContent` (never `innerHTML`), so untrusted trace data can never execute."""
from __future__ import annotations

import html
import json
import os

from ctxdiff.analyze.cache import CacheReport, analyze_cache
from ctxdiff.analyze.differ import TurnDiff, diff_turns, distinct_agents
from ctxdiff.analyze.tokens import RunTokens, analyze_run
from ctxdiff.store.ctrace import CTrace
from ctxdiff.viewer.template import render_page

# How much of a diff entry's block text to embed as its snippet, and how much
# of each inline-diff segment to keep — the dashboard shows previews, not whole
# documents, and capping here keeps the embedded JSON from ballooning on runs
# with very large blocks.
_SNIPPET_CHARS = 200
_INLINE_SEG_CHARS = 500


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


def build_payload(ct: CTrace) -> dict:
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


def export_html(ctrace_path: str, out_path: str | None = None) -> str:
    """Export the trace at `ctrace_path` to a self-contained HTML dashboard and
    return the written path. `out_path` overrides the destination; by default
    the file is written as `<trace-stem>.html` right next to the trace. Opens
    the trace read-only, builds the payload, embeds it in the page template
    (title = "ctxdiff — {project}"), and writes UTF-8."""
    ct = CTrace.open(ctrace_path)
    try:
        payload = build_payload(ct)
    finally:
        ct.close()

    project = payload["run"]["project"]
    document = render_page(project_title=html.escape(f"ctxdiff — {project}"),
                           data_json=_embed_json(payload))

    if out_path is None:
        stem = os.path.splitext(os.path.basename(ctrace_path))[0]
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(ctrace_path)), f"{stem}.html")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(document)
    return out_path
