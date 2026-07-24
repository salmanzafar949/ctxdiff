"""The block differ (spec §6.2): a pure function that turns two calls' ordered
block lists into a structured, git-style diff. No I/O and no color live here —
the CLI (`ctxdiff diff`) and the viewer are the only things that render this
output; this module just computes it so both can share the same algorithm."""
from __future__ import annotations

import difflib
from dataclasses import dataclass

from ctxdiff.models import Block, CallBlock
from ctxdiff.store.ctrace import CTrace

# --- value types -------------------------------------------------------------


@dataclass(frozen=True)
class DiffEntry:
    """One block's fate between two turns. `kind` is 'added' | 'evicted' |
    'modified' | 'unchanged'. `block` is always the side worth showing by
    default — the new block for added/modified/unchanged, the old block for
    evicted (there is no new side to prefer). `old_block` is populated only
    for 'modified', carrying the pre-edit block so callers can show a
    before/after. `inline_diff` is populated only for 'modified': a list of
    (op, text) segments, op in {'equal','delete','insert'}, from a char-level
    diff of old_block.text -> block.text. `position_old`/`position_new` are
    the block's position within each call's block list (None on the side it
    doesn't appear); for 'unchanged' both are set, and a mismatch between them
    is a moved-not-changed block."""
    kind: str
    block: Block
    label: str
    position_old: int | None
    position_new: int | None
    inline_diff: list[tuple[str, str]] | None
    old_block: Block | None = None


@dataclass(frozen=True)
class TurnDiff:
    """The full diff between call `seq_old` and call `seq_new`: every block's
    DiffEntry plus token deltas. `tokens_added` sums token_count over 'added'
    blocks plus the new side of every 'modified' block; `tokens_evicted` sums
    'evicted' blocks plus the old side of every 'modified' block — a content
    edit is accounted as "evict the old text, add the new text" for budgeting
    purposes even though it renders as one inline diff, not two lines."""
    seq_old: int
    seq_new: int
    entries: list[DiffEntry]
    tokens_added: int
    tokens_evicted: int


# --- inline (char-level) diff -------------------------------------------------


def _inline_diff(old_text: str, new_text: str) -> list[tuple[str, str]]:
    """Char-level diff between a modified block's old and new text, via
    difflib.SequenceMatcher. Returns a flat list of (op, text) segments in
    old->new order. SequenceMatcher's own opcodes already include 'equal' /
    'delete' / 'insert' verbatim; its 'replace' opcode (a same-position swap
    of old chars for new chars) is split into a delete-then-insert pair here
    so every segment in the output maps to exactly one side's text — callers
    never have to special-case a fifth op."""
    sm = difflib.SequenceMatcher(None, old_text, new_text, autojunk=False)
    segments: list[tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            segments.append(("equal", old_text[i1:i2]))
        elif tag == "delete":
            segments.append(("delete", old_text[i1:i2]))
        elif tag == "insert":
            segments.append(("insert", new_text[j1:j2]))
        elif tag == "replace":
            segments.append(("delete", old_text[i1:i2]))
            segments.append(("insert", new_text[j1:j2]))
    return segments


# --- moved-block reconciliation ----------------------------------------------


def _reconcile_moves(entries: list[DiffEntry]) -> list[DiffEntry]:
    """Fix up pure moves that the LCS alignment below reports as evict+add.

    SequenceMatcher aligns the two hash lists by finding the longest common
    subsequence; a block that kept its exact content (same hash) but moved to
    a position that breaks that subsequence's order does NOT get folded into
    an 'equal' opcode — it surfaces as one 'delete' and one 'insert' of the
    same hash, on either side of the reordering. Left alone that would show up
    as spurious added/evicted entries (and token deltas) for content that in
    fact didn't change at all.

    Fix: index every 'evicted' and 'added' entry by content_hash; wherever the
    same hash appears on both sides, pair them off (first evicted with first
    added, in encounter order — FIFO, since blocks of identical content are
    interchangeable) and replace the pair with a single 'unchanged' entry
    carrying both positions. Any hash with an evicted/added count mismatch
    (a real net add or removal of repeated content) leaves its leftovers as
    genuine evicted/added entries."""
    evicted_by_hash: dict[str, list[DiffEntry]] = {}
    added_by_hash: dict[str, list[DiffEntry]] = {}
    for e in entries:
        if e.kind == "evicted":
            evicted_by_hash.setdefault(e.block.content_hash, []).append(e)
        elif e.kind == "added":
            added_by_hash.setdefault(e.block.content_hash, []).append(e)

    moved: list[DiffEntry] = []
    consumed: set[int] = set()  # id() of entries folded into a move
    for h, evicted_list in evicted_by_hash.items():
        for ev, ad in zip(evicted_list, added_by_hash.get(h, [])):
            consumed.add(id(ev))
            consumed.add(id(ad))
            moved.append(DiffEntry(
                kind="unchanged", block=ad.block, label=ad.label,
                position_old=ev.position_old, position_new=ad.position_new,
                inline_diff=None,
            ))

    kept = [e for e in entries if id(e) not in consumed]
    kept.extend(moved)
    return kept


# --- core algorithm ------------------------------------------------------------


def diff_calls(old: list[CallBlock], new: list[CallBlock],
                seq_old: int, seq_new: int) -> TurnDiff:
    """Diff two ordered CallBlock lists (one call's context each) into a
    TurnDiff. Algorithm (spec §6.2, "sequence alignment over hashes"):

    1. Sequence-align the two ordered hash lists with difflib.SequenceMatcher
       — the LCS-style alignment the spec calls for, so an insertion or
       deletion produces exactly one added/evicted entry rather than
       cascading into "everything after it changed" (a naive index-by-index
       zip would do the latter).
    2. 'equal' opcodes -> unchanged entries, recording both positions (a
       moved-but-identical block is visible via a position_old/position_new
       delta even though it's still 'unchanged').
    3. 'delete' opcodes -> evicted; 'insert' opcodes -> added.
    4. 'replace' opcodes -> the two spans don't line up 1:1 by identity (that
       is exactly why SequenceMatcher called them 'replace' and not
       'equal'), so pair them up positionally within the span. A pair in the
       same "logical slot" — same role AND same kind — is treated as one
       block's content changing in place -> 'modified', with a char-level
       inline diff. A pair with mismatched role/kind, or a leftover item when
       the two spans differ in length, is not a content edit -> evicted (old
       side) / added (new side) instead.
    5. Post-pass: reconcile pure moves that step 1-4 would otherwise report
       as spurious evict+add of identical content (see _reconcile_moves).

    Labels: an entry's `label` always comes from the CallBlock on the side
    being shown (new side for added/modified/unchanged, old side for
    evicted) — label is a property of *how a call used a block*, not of the
    block itself, so there is no single "right" label to inherit across
    calls."""
    old_hashes = [cb.block.content_hash for cb in old]
    new_hashes = [cb.block.content_hash for cb in new]
    sm = difflib.SequenceMatcher(None, old_hashes, new_hashes, autojunk=False)

    entries: list[DiffEntry] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                o, n = old[i], new[j]
                entries.append(DiffEntry(
                    kind="unchanged", block=n.block, label=n.label,
                    position_old=o.position, position_new=n.position,
                    inline_diff=None,
                ))
        elif tag == "delete":
            for i in range(i1, i2):
                o = old[i]
                entries.append(DiffEntry(
                    kind="evicted", block=o.block, label=o.label,
                    position_old=o.position, position_new=None,
                    inline_diff=None,
                ))
        elif tag == "insert":
            for j in range(j1, j2):
                n = new[j]
                entries.append(DiffEntry(
                    kind="added", block=n.block, label=n.label,
                    position_old=None, position_new=n.position,
                    inline_diff=None,
                ))
        elif tag == "replace":
            old_slice = old[i1:i2]
            new_slice = new[j1:j2]
            paired = min(len(old_slice), len(new_slice))
            for k in range(paired):
                o, n = old_slice[k], new_slice[k]
                same_slot = o.block.role == n.block.role and o.block.kind == n.block.kind
                if same_slot:
                    entries.append(DiffEntry(
                        kind="modified", block=n.block, label=n.label,
                        position_old=o.position, position_new=n.position,
                        inline_diff=_inline_diff(o.block.text, n.block.text),
                        old_block=o.block,
                    ))
                else:
                    entries.append(DiffEntry(
                        kind="evicted", block=o.block, label=o.label,
                        position_old=o.position, position_new=None,
                        inline_diff=None,
                    ))
                    entries.append(DiffEntry(
                        kind="added", block=n.block, label=n.label,
                        position_old=None, position_new=n.position,
                        inline_diff=None,
                    ))
            # length mismatch inside the replace span: extra items on either
            # side have nothing to pair with, so they're straight evict/add.
            for o in old_slice[paired:]:
                entries.append(DiffEntry(
                    kind="evicted", block=o.block, label=o.label,
                    position_old=o.position, position_new=None,
                    inline_diff=None,
                ))
            for n in new_slice[paired:]:
                entries.append(DiffEntry(
                    kind="added", block=n.block, label=n.label,
                    position_old=None, position_new=n.position,
                    inline_diff=None,
                ))

    entries = _reconcile_moves(entries)

    # Modified counts as evicted-old + added-new for budgeting purposes (see
    # TurnDiff docstring): the old text's tokens left the context, the new
    # text's tokens entered it, even though the UI shows one inline diff.
    tokens_added = sum(
        e.block.token_count for e in entries if e.kind in ("added", "modified")
    )
    tokens_evicted = sum(
        e.block.token_count for e in entries if e.kind == "evicted"
    ) + sum(
        e.old_block.token_count for e in entries if e.kind == "modified"
    )

    # Stable, human-legible order for rendering: by the position a block ends
    # up at in the new call, falling back to its old position for pure
    # evictions (which have no new position).
    entries.sort(key=lambda e: e.position_new if e.position_new is not None
                 else e.position_old)

    return TurnDiff(seq_old=seq_old, seq_new=seq_new, entries=entries,
                     tokens_added=tokens_added, tokens_evicted=tokens_evicted)


def diff_turns(ct: CTrace, turn_old: int, turn_new: int) -> TurnDiff:
    """Convenience wrapper: resolve two turn numbers (call.seq) to their calls
    in `ct`, load each call's blocks, and delegate to diff_calls. Raises
    ValueError with a clear message if either turn has no matching call in
    this run, rather than letting a KeyError/IndexError leak out."""
    calls_by_seq = {c.seq: c for c in ct.get_calls()}
    missing = [seq for seq in (turn_old, turn_new) if seq not in calls_by_seq]
    if missing:
        raise ValueError(
            f"turn(s) {missing} not found in this run "
            f"(available turns: {sorted(calls_by_seq)})"
        )
    old_call = calls_by_seq[turn_old]
    new_call = calls_by_seq[turn_new]
    old_blocks = ct.get_call_blocks(old_call.id)
    new_blocks = ct.get_call_blocks(new_call.id)
    return diff_calls(old_blocks, new_blocks, seq_old=turn_old, seq_new=turn_new)
