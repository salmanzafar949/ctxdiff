"""The tagged-eviction detector: "the block you tagged 'rag' at turn 3 was
evicted at turn 6".

Why this is its own report. "The agent forgot the thing I told it" is the single
most common bug an agent developer brings to a context debugger, and it almost
always has the same mechanical cause: a block that WAS in the context is not in
it any more — trimmed by a sliding window, dropped by a summarizer, rebuilt from
a shorter history. ctxdiff already sees that happen (the differ classifies it as
`evicted`) and already knows which blocks the developer considered load-bearing
(`tracer.tag()` marks them, `label_source == 'tagged'`). This module is the
join: it names WHERE the block entered and WHERE it disappeared, in those words.

Nothing here re-implements eviction detection. Every disappearance is one of
`differ.diff_calls`'s own `evicted` entries — which matters beyond tidiness,
because a naive "was this hash present last turn?" scan would report a block
whose text was EDITED as an eviction, while the differ correctly classifies a
same-slot content change as `modified` and a pure reorder as `unchanged`. Both
of those are things developers do to their prompts on purpose, and reporting
either as "your RAG block was evicted" would burn the signal on the first run.

One consequence of that reuse is worth stating plainly, because it is a BLIND
SPOT and it is deliberate: a tagged block REPLACED IN THE SAME SLOT by different
text of the same role and kind reads to the differ as an edit (`modified`), so no
eviction is reported for it. Teaching this module to second-guess that would mean
a similarity heuristic of its own — a second opinion about something the differ
already decided — and the two would then disagree in `ctxdiff diff` and `ctxdiff
tokens` over the same trace. Whatever the differ calls it, this report calls it.

Three deliberate narrowings, each there to keep a warning worth reading:

1. TAGGED ONLY. A block qualifies only when its label came from `tracer.tag()`
   (`label_source == 'tagged'`). Heuristic labels are never included. Every
   multi-turn agent evicts heuristically-labeled history constantly — that is
   what a context window IS — so including them would produce a wall of
   warnings about the intended behaviour of every framework in existence. A tag
   is the developer's own statement that this particular content is supposed to
   stay, which is exactly the assertion an eviction violates.

   Taggedness is a property of the CONTENT, not of one call, and that is not a
   convenience — it is what makes the feature work at all. `tracer.tag()` is
   NEXT-CALL-ONLY by design: it buffers labels for the next recorded call and
   clears itself, so a block tagged once is stored `tagged` on exactly ONE call
   and `heuristic` on every later call that still carries the same text. An
   eviction is always detected on the pair (last turn that had it, first turn
   that did not), so asking "was it tagged on the older side of THIS pair?"
   answers no for everything tagged earlier than the turn before it vanished —
   which silenced the module on its own headline example ("tagged at turn 1 …
   evicted at turn 5"). So the question asked here is "did the developer EVER
   vouch for this content hash, anywhere in this agent's timeline?", and the
   label and the turn quoted back are the ones from the FIRST call that did.

2. PER AGENT. Calls are grouped by agent and only consecutive calls of the SAME
   agent are compared — the rule `analyze_cache` established for the same
   reason. On an interleaved multi-agent timeline the researcher's block is
   "missing" from the writer's very next call because it was never in the
   writer's context at all; that is a hand-off, not an eviction, and flagging it
   would make the report useless on precisely the runs it matters most for.

3. PERMANENT ONLY. An eviction is reported only when the block never appears
   again in any later turn of that agent. A block that is absent for one turn
   and back the next was not forgotten — it was re-retrieved, re-ranked, or
   momentarily crowded out — and the report would be claiming a loss that did
   not happen. A block that leaves twice (out, back, then out for good) is
   reported once, for the departure it did not return from.

Like the other analyzers this module does no I/O beyond reading the store and
emits no color: it returns frozen dataclasses whose fields `cli/render.py` and
the dashboard lay out.
"""
from __future__ import annotations

from dataclasses import dataclass

from ctxdiff.analyze.cache import flatten_snippet
from ctxdiff.analyze.differ import diff_calls, distinct_agents, filter_calls
from ctxdiff.models import CallBlock
from ctxdiff.store.base import Call, Store

# --- value types -------------------------------------------------------------


@dataclass(frozen=True)
class TaggedEviction:
    """One tagged block's disappearance from an agent's context.

    `label` is the tag the developer gave it (`tracer.tag('rag', ...)`), which
    is also what the report calls it, and `tagged_seq` is the turn they gave it
    on — the FIRST call of this agent that carried this content as `tagged`.
    The two always come from the same call, because quoting a label against a
    turn that did not carry it describes a tag that never existed: a block
    present from turn 1 but tagged from turn 2 was not tagged at turn 1, and a
    block re-tagged under a second name later was not called that at the start.

    `entered_seq` is the first turn of this agent that carried the CONTENT at
    all, tagged or not — where it entered the context. It is usually the same
    turn as `tagged_seq` and is a separate field precisely for when it is not.
    `last_seen_seq` is the last turn that still carried it and `evicted_seq` the
    first that did not; those two are adjacent within the agent's own timeline
    but can be far apart in global turn numbers when agents interleave, which is
    why both are reported.

    `tokens`/`role`/`snippet` describe the block itself, so a reader can tell
    which of three tagged blocks this was without opening the trace.
    `content_hash` is carried for identity (and as the deterministic tiebreak in
    the report's sort order), never displayed."""
    label: str
    agent: str | None
    tagged_seq: int
    entered_seq: int
    last_seen_seq: int
    evicted_seq: int
    tokens: int
    role: str
    snippet: str
    content_hash: str


@dataclass(frozen=True)
class EvictionReport:
    """Every tagged eviction in a run, in timeline order.

    `pairs_analyzed` is how many consecutive same-agent turn pairs were actually
    compared — the honest denominator, and the number that distinguishes "no
    tagged block was evicted" from "there was nothing to compare" (a one-turn
    run, or a fan-out where every agent has a single turn). `agents_analyzed` is
    the number of per-agent groups the run was split into when analyzed
    unfiltered with more than one agent, and None when it was analyzed as a
    single timeline — the same field, with the same meaning, `CacheReport`
    carries. `tagged_blocks` counts the distinct tagged blocks seen at all, so a
    report can say "nothing was tagged" rather than implying it checked
    something; distinct means distinct PER AGENT TIMELINE, the same scope every
    other number here uses, so that it stays the denominator `evictions` is the
    numerator of — one event per tagged block per agent, one count per tagged
    block per agent."""
    evictions: list[TaggedEviction]
    pairs_analyzed: int
    agents_analyzed: int | None = None
    tagged_blocks: int = 0


# --- per-agent core -----------------------------------------------------------


def _by_position(blocks: list[CallBlock]) -> dict[int, CallBlock]:
    """Index one call's blocks by their position, so a differ entry's
    `position_old` can be resolved back to the CallBlock that produced it.

    Position rather than content hash: a call may legitimately carry the same
    text twice under different labels, and only the position identifies which
    membership the differ was talking about. The index is UNCONDITIONAL — every
    block, not just the ones this call happened to store as `tagged` — because
    whether the developer vouched for the content is decided over the whole
    timeline by `_tagged_labels`, not by the single call the eviction pair
    happens to start from (see narrowing 1 in the module docstring)."""
    return {cb.position: cb for cb in blocks}


def _tagged_labels(
    calls: list[Call],
    blocks_by_call_id: dict[str, list[CallBlock]],
) -> dict[str, tuple[str, int]]:
    """Map each content hash the developer EVER tagged in this group to the
    (label, turn) of the first call that tagged it.

    First and not last: a re-tag under a second name later must not make the
    report quote the newer name — the sentence names one turn, and the label it
    quotes has to be the one that turn actually carried. Walking `calls` in seq
    order and refusing to overwrite is the whole rule."""
    labels: dict[str, tuple[str, int]] = {}
    for c in calls:
        for cb in blocks_by_call_id[c.id]:
            if cb.label_source == "tagged":
                labels.setdefault(cb.block.content_hash, (cb.label, c.seq))
    return labels


def _analyze_group(
    calls: list[Call],
    blocks_by_call_id: dict[str, list[CallBlock]],
    agent_label: str | None,
) -> tuple[list[TaggedEviction], int, set[str]]:
    """Find one agent's tagged evictions. Returns (evictions, pairs analyzed,
    the distinct tagged content hashes seen).

    How, in three passes over data that is already loaded:

    1. Walk the group's calls once in seq order to record, per content hash,
       the FIRST turn it appeared in and the LAST — the first becomes
       `entered_seq`, and the last is what makes the permanence test a
       comparison rather than a re-scan — and, via `_tagged_labels`, the label
       and turn of the first call that tagged it.
    2. For each consecutive pair, ask `diff_calls` what happened. Only its
       `evicted` entries are considered, so an edit (`modified`) or a reorder
       (`unchanged` with moved positions) can never be mistaken for a loss.
    3. Keep an evicted entry when the content it names was tagged SOMEWHERE in
       this group and its last appearance anywhere in the group is that older
       turn — i.e. it never comes back.

    A block that is evicted, returns, and is evicted again therefore yields
    exactly one event: the first departure fails the permanence test (its last
    appearance is later), the second passes. A block that a single call carried
    TWICE also yields exactly one event, via `reported`: the differ correctly
    reports two `evicted` entries because two memberships really did disappear,
    but the developer lost one block, and counting memberships is what made the
    report print the same stanza twice and `check` say "2 tagged blocks evicted
    of 1". The duplicates are byte-identical anyway — same content hash means
    same text, role and token count, and the label and turn now come from
    `_tagged_labels` rather than from the individual membership."""
    first_seen: dict[str, int] = {}
    last_seen: dict[str, int] = {}
    for c in calls:
        for cb in blocks_by_call_id[c.id]:
            h = cb.block.content_hash
            first_seen.setdefault(h, c.seq)
            last_seen[h] = c.seq
    tagged_label = _tagged_labels(calls, blocks_by_call_id)

    evictions: list[TaggedEviction] = []
    reported: set[str] = set()
    pairs = 0
    for prev_call, call in zip(calls, calls[1:]):
        pairs += 1
        if not tagged_label:
            continue  # nothing the developer vouched for could have been lost
        old = blocks_by_call_id[prev_call.id]
        new = blocks_by_call_id[call.id]
        old_by_position = _by_position(old)
        turn_diff = diff_calls(old, new, seq_old=prev_call.seq, seq_new=call.seq)
        for entry in turn_diff.entries:
            if entry.kind != "evicted" or entry.position_old is None:
                continue
            cb = old_by_position.get(entry.position_old)
            if cb is None:
                continue
            h = cb.block.content_hash
            if h not in tagged_label:
                continue  # evicted, but never tagged — ordinary history churn
            if last_seen.get(h, prev_call.seq) > prev_call.seq:
                continue  # it comes back later: absent, but not forgotten
            if h in reported:
                continue  # a second membership of the same lost block
            reported.add(h)
            label, tagged_seq = tagged_label[h]
            evictions.append(TaggedEviction(
                label=label,
                agent=agent_label,
                tagged_seq=tagged_seq,
                entered_seq=first_seen[h],
                last_seen_seq=prev_call.seq,
                evicted_seq=call.seq,
                tokens=cb.block.token_count,
                role=cb.block.role,
                snippet=flatten_snippet(cb.block.text),
                content_hash=h,
            ))
    return evictions, pairs, set(tagged_label)


# --- the entry point -----------------------------------------------------------


def analyze_evictions(ct: Store, agent: str | None = None) -> EvictionReport:
    """Find every tagged block that entered an agent's context and later left it
    for good (see the module docstring for the three narrowings).

    Grouping mirrors `analyze_cache` exactly, because the question has the same
    shape — "what happened between two turns of the same agent":

    - `agent` given: that agent's calls only, consecutive within its own slice;
    - `agent` None with a single distinct agent (or an all-unlabeled run): one
      timeline, consecutive global pairs;
    - `agent` None with several agents: split per agent, analyze within each,
      merge. A cross-agent adjacent pair is never compared, so a hand-off can
      never read as an eviction.

    Events are merged in a deterministic order — by the turn the block
    disappeared, then the turn it entered, then its content hash — so the two
    SDKs list them identically even though they arrive grouped by agent."""
    calls = filter_calls(ct.get_calls(), agent)
    blocks_by_call_id = {c.id: ct.get_call_blocks(c.id) for c in calls}

    labels = distinct_agents(calls)
    grouped = agent is None and len(labels) > 1
    if grouped:
        groups = [(lbl, [c for c in calls if c.agent == lbl]) for lbl in labels]
        agents_analyzed: int | None = len(labels)
    else:
        sole = agent if agent is not None else (labels[0] if labels else None)
        groups = [(sole, calls)]
        agents_analyzed = None

    all_evictions: list[TaggedEviction] = []
    pairs_analyzed = 0
    # Keyed by (agent, hash) rather than by hash alone, so the count stays the
    # denominator the eviction list is the numerator of: a group emits at most
    # one event per tagged hash, so the same text tagged in two agents' contexts
    # is two tagged blocks that can be lost twice — and collapsing them here
    # would reintroduce, across agents, the very "N evicted of fewer than N" the
    # per-group dedupe removes within one.
    tagged_hashes: set[tuple[str | None, str]] = set()
    for label, group_calls in groups:
        evictions, pairs, tagged = _analyze_group(
            group_calls, blocks_by_call_id, label)
        all_evictions.extend(evictions)
        pairs_analyzed += pairs
        tagged_hashes |= {(label, h) for h in tagged}

    # Sorted by the turn the block disappeared, then the turn it entered, then
    # its content hash. The hash and not the tag as the final tiebreak on
    # purpose: it is unique per block (so the order is total), and it is lowercase
    # hex — which JS and Python compare identically, while an arbitrary user tag
    # could contain astral characters that UTF-16 and code-point ordering disagree
    # about.
    all_evictions.sort(
        key=lambda e: (e.evicted_seq, e.entered_seq, e.content_hash))

    return EvictionReport(
        evictions=all_evictions,
        pairs_analyzed=pairs_analyzed,
        agents_analyzed=agents_analyzed,
        tagged_blocks=len(tagged_hashes),
    )
