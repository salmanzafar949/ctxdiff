"""The prompt-cache alignment profiler (spec §6.4): a pure function that
finds where a run's provider-side cache prefix breaks between consecutive
calls, attributes the break to a specific block, and quantifies the wasted
re-billing. Like differ.py and tokens.py, this module does no I/O and no
color — it reads store data (Call/CallBlock) and returns frozen dataclasses
for the CLI (`ctxdiff cache`) to render.

Why a *separate* alignment from the block differ: providers cache a request
prefix byte-for-byte, so "is the prefix still stable" is a strictly
positional question — position 0 must equal position 0, position 1 must
equal position 1, and so on — not the LCS-style "did this content move"
question the differ answers for human-readable diffing. The differ's richer
alignment IS reused here, but only *after* the raw positional walk has found
where the prefix breaks, to explain *what* changed at that exact position
(reusing diff_calls means this module never re-implements alignment)."""
from __future__ import annotations

from dataclasses import dataclass

from ctxdiff.analyze.differ import (
    TurnDiff,
    diff_calls,
    distinct_agents,
    filter_calls,
)
from ctxdiff.models import CallBlock
from ctxdiff.store.ctrace import Call, CTrace

# --- value types -------------------------------------------------------------


@dataclass(frozen=True)
class PrefixBreak:
    """One consecutive-call pair whose cache prefix diverges. `seq_prev`/`seq`
    are the two calls' turn numbers. `stable_blocks`/`stable_tokens` count the
    leading run of positionally-identical blocks (the part a cache-aware
    provider still hits). `divergent_position` is the 0-indexed slot where
    the two calls first differ — numerically equal to `stable_blocks` (the
    first N blocks are stable, so the break sits at index N), kept as a
    separate field because the two numbers mean different things to a reader
    (a count vs. a position). `culprit_kind` is one of: 'modified' | 'added'
    | 'evicted' (the block differ's own vocabulary, reused directly for a
    same-slot content edit / insertion / eviction at the divergence
    position); 'reordered' (the divergence is a pure position swap — the
    same content the differ folds into two 'unchanged'-but-moved entries,
    which byte-for-byte cache matching still treats as a break since the
    positions themselves changed); or 'changed' (a final, defensive
    catch-all for any other structural shape none of the above classifies —
    kept honest rather than mis-labeling something unrecognized).
    `culprit_label`/`culprit_snippet` describe the block responsible;
    `detail` is a terse human-readable explanation (for 'modified', it names
    the first differing character range; for 'added'/'evicted' it says a
    block was inserted/removed at that slot; for 'reordered', it names the
    old/new positions the block moved between)."""
    seq_prev: int
    seq: int
    stable_blocks: int
    stable_tokens: int
    divergent_position: int
    culprit_kind: str
    culprit_label: str
    culprit_snippet: str
    detail: str
    agent: str | None = None  # which agent's timeline this break sits in (None
    # for an unlabeled/single-agent run); a break is only ever computed between
    # two calls of the SAME agent, so this is unambiguous


@dataclass(frozen=True)
class CacheReport:
    """A whole run's cache-prefix analysis. `stable_prefix_tokens_min` is the
    smallest stable-prefix token count seen across all pairs — the effective
    guaranteed-cacheable prefix for the run as a whole, since a cache-aware
    provider can only ever count on the worst pair's stability.
    `rebilled_tokens_total` sums, over every pair, the tokens in the NEWER
    call from the divergence point onward (the tokens that would have been
    cache-priced had the prefix stayed stable). `estimated_waste_note` is a
    neutral, price-free quantification string (composed here so the CLI just
    prints it verbatim). `fix_hint` is populated only when a specific,
    actionable pattern is detected (see `_fix_hint`); None otherwise."""
    pairs_analyzed: int
    breaks: list[PrefixBreak]
    stable_prefix_tokens_min: int
    rebilled_tokens_total: int
    estimated_waste_note: str
    fix_hint: str | None
    agents_analyzed: int | None = None  # number of distinct agent groups the
    # run was split into when analyzed unfiltered with >1 agent (so a
    # cross-agent adjacent pair is never counted as a break); None when the run
    # was analyzed as a single timeline (one agent, or an --agent filter)
    pairs_by_agent: dict[str, int] | None = None  # per-agent analyzed-pair
    # counts (keyed by agent label) when grouped, so a break attributed to an
    # agent can be reported against THAT agent's own pair count as the
    # denominator; None when the run was a single timeline. '(unlabeled)' keys
    # the None-agent group.


# --- snippet formatting -------------------------------------------------------


def _flatten_snippet(text: str, limit: int = 80) -> str:
    """Collapse a block's text to a single flattened, truncated line for
    display (newlines/repeated whitespace -> single spaces, then hard-cut at
    `limit` chars with an ellipsis marker). Kept plain (no quoting/repr) —
    that's a rendering choice left to the CLI, which may format it however
    it formats other snippets."""
    flat = " ".join(text.split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


def _truncate(text: str, limit: int = 40) -> str:
    """Shorter truncation for the differing substring embedded in a
    'modified' break's `detail` string — the point there is to show just
    enough of the change to recognize it (e.g. a timestamp), not the whole
    block."""
    return text[:limit] + ("…" if len(text) > limit else "")


# --- first-difference extraction (from the differ's inline diff) --------------


def _first_diff_segment(inline_diff: list[tuple[str, str]]) -> tuple[int, str, str]:
    """Given a 'modified' DiffEntry's char-level inline_diff (old_text ->
    new_text segments from differ._inline_diff), return (char_offset,
    old_part, new_part) for the FIRST run of non-equal segments: char_offset
    is where old_text and new_text first diverge (the summed length of the
    leading 'equal' segments, since equal segments are byte-identical on both
    sides up to that point); old_part/new_part are the deleted/inserted text
    of that first differing run — a 'replace' opcode surfaces as one
    delete-then-insert pair here, so a same-position substitution (e.g. an
    old timestamp for a new one) reads as one before/after pair, not two
    unrelated segments."""
    offset = 0
    idx = 0
    while idx < len(inline_diff) and inline_diff[idx][0] == "equal":
        offset += len(inline_diff[idx][1])
        idx += 1
    old_part = ""
    new_part = ""
    while idx < len(inline_diff) and inline_diff[idx][0] != "equal":
        op, seg = inline_diff[idx]
        if op == "delete":
            old_part += seg
        else:  # 'insert'
            new_part += seg
        idx += 1
    return offset, old_part, new_part


def _is_dynamic_change(inline_diff: list[tuple[str, str]]) -> bool:
    """Heuristic for 'a small volatile substring inside otherwise-stable
    text' (e.g. a timestamp) as opposed to a wholesale rewrite: true when the
    diff has SOME shared (equal) text at all, and the total changed text is
    shorter than the total shared text — i.e. most of the block survived
    unchanged. A block that changed entirely (no shared text, or more
    changed than shared) doesn't fit the "dynamic field" story."""
    equal_len = sum(len(seg) for op, seg in inline_diff if op == "equal")
    changed_len = sum(len(seg) for op, seg in inline_diff if op != "equal")
    return equal_len > 0 and changed_len < equal_len


# --- break attribution ---------------------------------------------------------


def _attribute_break(
    turn_diff: TurnDiff, position: int, old: list[CallBlock], new: list[CallBlock]
) -> tuple[str, str, str, str, bool]:
    """Explain what happened at `position` (the first index where old/new
    hashes diverge), by looking up how the block differ classified that slot.
    Returns (culprit_kind, culprit_label, culprit_snippet, detail,
    is_dynamic_change). How: reuses `turn_diff` (already computed by
    diff_calls) rather than re-deriving alignment — a 'modified' entry whose
    OLD and NEW positions both equal `position` is a same-slot content edit;
    otherwise an 'added' entry landing at this new-position is an insertion,
    and an 'evicted' entry leaving from this old-position is a removal.

    A pure position swap (e.g. old=[A,B], new=[B,A], same content) is a
    real, reachable shape that is none of the above: diff_calls' move-
    reconciliation folds BOTH A and B into 'unchanged' entries (same
    content, different position — see differ.py's `_reconcile_moves`), so no
    modified/added/evicted entry lines up with `position` at all. Byte-for-
    byte cache matching still calls this a break (the bytes at this
    position genuinely changed, even though the differ correctly says
    "nothing was added or removed"), so it's detected separately: an
    'unchanged' entry whose position_old != position_new (it moved) and
    whose new-or-old position equals the divergence index is classified
    'reordered'.

    Any other shape falls back to a generic, still-honest 'changed'
    description rather than mis-labeling it — no fixture reaching this
    final fallback is known; it exists purely as a non-crashing safety net."""
    modified = next(
        (e for e in turn_diff.entries if e.kind == "modified"
         and e.position_old == position and e.position_new == position),
        None,
    )
    if modified is not None:
        offset, old_part, new_part = _first_diff_segment(modified.inline_diff or [])
        detail = (f"modified {modified.label} block — first difference at "
                  f"char {offset}: '{_truncate(old_part)}' → '{_truncate(new_part)}'")
        is_dynamic = _is_dynamic_change(modified.inline_diff or [])
        return ("modified", modified.label, _flatten_snippet(modified.block.text),
                detail, is_dynamic)

    added = next(
        (e for e in turn_diff.entries if e.kind == "added" and e.position_new == position),
        None,
    )
    if added is not None:
        detail = (f"block inserted at position {position} — {added.label}/"
                  f"{added.block.role} block not present in the previous turn")
        return ("added", added.label, _flatten_snippet(added.block.text), detail, False)

    evicted = next(
        (e for e in turn_diff.entries if e.kind == "evicted" and e.position_old == position),
        None,
    )
    if evicted is not None:
        detail = (f"block evicted at position {position} — {evicted.label}/"
                  f"{evicted.block.role} block from the previous turn is missing here")
        return ("evicted", evicted.label, _flatten_snippet(evicted.block.text), detail, False)

    # A pure position swap: the differ's move-reconciliation folds both sides
    # of the swap into 'unchanged' entries (same content_hash, just a
    # different position), so nothing above matched — but the divergence is
    # real from a byte-for-byte cache-matching point of view. Detect it
    # directly: an 'unchanged' entry that moved (its old and new positions
    # differ) and whose new-or-old position is this divergence index is the
    # block now sitting where a different block used to be (or vice versa).
    reordered = next(
        (e for e in turn_diff.entries if e.kind == "unchanged"
         and e.position_old != e.position_new
         and (e.position_new == position or e.position_old == position)),
        None,
    )
    if reordered is not None:
        detail = (f"block reordered — {reordered.label}/{reordered.block.role} moved "
                  f"from position {reordered.position_old} to {reordered.position_new}, "
                  f"breaking the byte-for-byte prefix match at position {position}")
        return ("reordered", reordered.label, _flatten_snippet(reordered.block.text),
                detail, False)

    # Final fallback: some other structural shape neither classified above
    # nor a detected reorder landed exactly at this position. No known
    # fixture reaches this branch; it exists so an unanticipated shape is
    # described honestly instead of crashing or being mis-labeled.
    new_side = new[position] if position < len(new) else None
    old_side = old[position] if position < len(old) else None
    side = new_side or old_side
    text = side.block.text if side else ""
    label = side.label if side else "unknown"
    detail = f"context diverges at position {position} (not a simple modify/insert/evict/reorder)"
    return ("changed", label, _flatten_snippet(text), detail, False)


# --- waste note + fix hint ------------------------------------------------------


def _waste_note(rebilled_tokens_total: int, pairs_analyzed: int) -> str:
    """Compose the neutral, price-free wasted-spend note (spec §6.4): no
    hardcoded per-token price or per-provider discount figure (providers set
    and change these independently — e.g. Anthropic's ~10% is not OpenAI's or
    Gemini's), just an honest quantification of what a stable prefix would
    have saved, with a pointer to check actual current rates."""
    turn_word = "turn" if pairs_analyzed == 1 else "turns"
    return (
        f"{rebilled_tokens_total} tokens re-billed across {pairs_analyzed} {turn_word} "
        "that a stable prefix would have served from cache (cached input is "
        "typically billed at a fraction of the full input price — check your "
        "provider's current rates)"
    )


_DYNAMIC_FIELD_HINT = (
    "a dynamic value inside an early system block breaks the prefix every "
    "turn — move volatile content below the stable blocks"
)


def _fix_hint(breaks: list[PrefixBreak], dynamic_flags: list[bool]) -> str | None:
    """Detect the one actionable pattern spec §6.4 calls out: a dynamic value
    (e.g. a timestamp) baked into an early system block, breaking the prefix
    identically every turn. All of the following must hold: there's at least
    one break; every break shares the same divergent_position AND
    culprit_kind (the same slot is responsible every time, not a rotating
    cast of culprits); that kind is 'modified' (a dynamic-field problem is a
    content edit, not an insert/evict); the position is among the first
    three (0/1/2 — an "early" block); the culprit's label is 'system'; and
    every one of those modifications looks like a small change inside mostly
    stable text (`dynamic_flags`, computed alongside each break by
    `_attribute_break`) rather than a wholesale rewrite. Anything else
    returns None rather than guessing."""
    if not breaks:
        return None
    first = breaks[0]
    same_culprit_every_time = all(
        b.divergent_position == first.divergent_position and b.culprit_kind == first.culprit_kind
        for b in breaks
    )
    if not same_culprit_every_time:
        return None
    if first.culprit_kind != "modified":
        return None
    if first.culprit_label != "system":
        return None
    if first.divergent_position > 2:  # "first 3 positions" == indices 0,1,2
        return None
    if not all(dynamic_flags):
        return None
    return _DYNAMIC_FIELD_HINT


# --- core algorithm --------------------------------------------------------------


def _analyze_group(
    calls: list[Call],
    blocks_by_call_id: dict[str, list[CallBlock]],
    agent_label: str | None,
) -> tuple[list[PrefixBreak], list[bool], list[int], int]:
    """Walk one agent's calls (already in seq order) for prefix stability and
    return (breaks, dynamic_flags, stable_tokens_per_pair, rebilled_total).
    This is the per-pair core, factored out so both the single-timeline and the
    per-agent-grouped paths share ONE positional walk. `agent_label` is stamped
    onto every PrefixBreak this group produces.

    For each consecutive pair (call N-1, call N):

    1. Walk both calls' block-hash lists position-by-position from index 0
       (NOT the differ's LCS alignment — cache stability is a strictly
       positional question: does byte K of the request match byte K of the
       last request). The walk stops at the first index where the hashes
       differ, or at the end of the shorter list.
    2. If the walk reached the end of the shorter list, the shorter call's
       entire block list is an exact leading prefix of the longer one — pure
       history growth (or pure trailing truncation), the cache-friendly happy
       path. This is NOT a break: no PrefixBreak, no rebilled tokens.
    3. Otherwise there's a genuine divergence at that index: the blocks before
       it are the stable prefix (their tokens are never rebilled); everything
       in the NEWER call from that index onward is rebilled. `diff_calls`
       (already-tested alignment) is reused to explain *what* happened at that
       exact position."""
    breaks: list[PrefixBreak] = []
    dynamic_flags: list[bool] = []
    stable_tokens_per_pair: list[int] = []
    rebilled_total = 0

    for prev_call, call in zip(calls, calls[1:]):
        old = blocks_by_call_id[prev_call.id]
        new = blocks_by_call_id[call.id]
        old_hashes = [cb.block.content_hash for cb in old]
        new_hashes = [cb.block.content_hash for cb in new]

        shorter_len = min(len(old_hashes), len(new_hashes))
        i = 0
        while i < shorter_len and old_hashes[i] == new_hashes[i]:
            i += 1

        stable_tokens = sum(cb.block.token_count for cb in new[:i])
        stable_tokens_per_pair.append(stable_tokens)

        if i == shorter_len:
            # Exact leading prefix — pure append growth / trailing truncation.
            continue

        rebilled = sum(cb.block.token_count for cb in new[i:])
        rebilled_total += rebilled

        turn_diff = diff_calls(old, new, seq_old=prev_call.seq, seq_new=call.seq)
        culprit_kind, culprit_label, culprit_snippet, detail, is_dynamic = (
            _attribute_break(turn_diff, position=i, old=old, new=new)
        )

        breaks.append(PrefixBreak(
            seq_prev=prev_call.seq, seq=call.seq,
            stable_blocks=i, stable_tokens=stable_tokens,
            divergent_position=i,
            culprit_kind=culprit_kind, culprit_label=culprit_label,
            culprit_snippet=culprit_snippet, detail=detail,
            agent=agent_label,
        ))
        dynamic_flags.append(is_dynamic)

    return breaks, dynamic_flags, stable_tokens_per_pair, rebilled_total


def analyze_cache(ct: CTrace, agent: str | None = None) -> CacheReport:
    """Analyze a run for cache-prefix stability (spec §6.4), agent-aware.

    Grouping semantics (the correctness fix this feature exposes): cache
    stability is only meaningful BETWEEN CALLS OF THE SAME AGENT — two agents
    interleaved on the global timeline each carry their own prefix, and an
    adjacent cross-agent pair sharing nothing is NOT a "break" (the pre-v2 code
    would have flagged every hand-off as a cache miss).

    - `agent` given: analyze only that agent's calls (consecutive within its
      own filtered timeline).
    - `agent` None with a SINGLE distinct agent (or an all-unlabeled run):
      behaves exactly as before — one timeline, consecutive global pairs.
    - `agent` None with MULTIPLE distinct agents: split the calls into
      per-agent groups (each preserving seq order), analyze prefix stability
      WITHIN each group, and merge; a cross-agent adjacent pair is never
      considered. `pairs_analyzed` is the sum of per-group pair counts, and
      `agents_analyzed` records how many groups were merged.

    `stable_prefix_tokens_min` is the smallest stable-prefix token count across
    EVERY analyzed pair — the effective guaranteed-cacheable prefix is only as
    good as the worst pair. A run (or filtered agent) with fewer than 2
    analyzable pairs returns an empty report with an explanatory note rather
    than crashing on an empty pair list."""
    calls = filter_calls(ct.get_calls(), agent)
    blocks_by_call_id = {c.id: ct.get_call_blocks(c.id) for c in calls}

    # Decide the grouping. Only an unfiltered run with >1 distinct agent is
    # split; everything else is one timeline.
    labels = distinct_agents(calls)
    grouped = agent is None and len(labels) > 1
    if grouped:
        groups = [(lbl, [c for c in calls if c.agent == lbl]) for lbl in labels]
        agents_analyzed: int | None = len(labels)
    else:
        # A single group: its label is the filtered agent (when given) or the
        # sole label seen (None for an unlabeled single-agent run).
        sole = agent if agent is not None else (labels[0] if labels else None)
        groups = [(sole, calls)]
        agents_analyzed = None

    all_breaks: list[PrefixBreak] = []
    all_dynamic: list[bool] = []
    all_stable: list[int] = []
    rebilled_total = 0
    # per-agent pair counts (only meaningful when grouped) so a break can be
    # reported against its OWN agent's denominator, not the run-wide total.
    pairs_by_agent: dict[str, int] = {}
    for label, group_calls in groups:
        breaks, dyn, stable, rebilled = _analyze_group(
            group_calls, blocks_by_call_id, label)
        all_breaks.extend(breaks)
        all_dynamic.extend(dyn)
        all_stable.extend(stable)
        rebilled_total += rebilled
        if grouped:
            key = label if label is not None else "(unlabeled)"
            pairs_by_agent[key] = len(stable)

    pairs_analyzed = len(all_stable)  # one stable-tokens entry per analyzed pair
    if pairs_analyzed == 0:
        return CacheReport(
            pairs_analyzed=0, breaks=[], stable_prefix_tokens_min=0,
            rebilled_tokens_total=0,
            estimated_waste_note="fewer than 2 calls in this run — nothing to analyze",
            fix_hint=None, agents_analyzed=agents_analyzed,
            pairs_by_agent=pairs_by_agent or None,
        )

    # Compute the fix hint BEFORE reordering: _fix_hint reads all(dynamic_flags)
    # (order-independent) and compares every break to breaks[0], so the flags
    # need only stay a valid collection, not stay positionally paired.
    fix_hint = _fix_hint(all_breaks, all_dynamic)

    # Merge order: by the newer call's seq, so breaks read in timeline order
    # even after per-agent grouping shuffled them.
    all_breaks.sort(key=lambda b: b.seq)

    return CacheReport(
        pairs_analyzed=pairs_analyzed,
        breaks=all_breaks,
        stable_prefix_tokens_min=min(all_stable) if all_stable else 0,
        rebilled_tokens_total=rebilled_total,
        estimated_waste_note=_waste_note(rebilled_total, pairs_analyzed),
        fix_hint=fix_hint,
        agents_analyzed=agents_analyzed,
        pairs_by_agent=pairs_by_agent or None,
    )
