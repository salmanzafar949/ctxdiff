import random

from ctxdiff.analyze.differ import diff_calls
from ctxdiff.models import Block, CallBlock


def _block(text, role="user", kind="message", token_count=None):
    """Build a Block whose content_hash is derived from (role, kind, text) so
    identical (role, kind, text) always produces the identical hash — mirrors
    real content-addressing without pulling in the store."""
    if token_count is None:
        token_count = len(text)
    return Block(
        content_hash=f"h:{role}:{kind}:{text}", role=role, kind=kind,
        text=text, token_count=token_count, token_method="tiktoken",
    )


def _cbs(specs):
    """Build an ordered CallBlock list from (text, role, label) tuples,
    positions assigned in list order — the shape diff_calls consumes."""
    result = []
    for i, spec in enumerate(specs):
        text, role, label = spec
        result.append(CallBlock(block=_block(text, role=role), position=i,
                                label=label, label_source="heuristic"))
    return result


def _kinds(diff):
    return [e.kind for e in diff.entries]


BASE = [
    ("system prompt", "system", "system"),
    ("rule one", "system", "system"),
    ("hi there", "user", "user"),
    ("how can I help", "assistant", "history"),
]


def test_insert_in_middle_is_one_added_no_cascade():
    """Inserting one block in the middle of an otherwise-identical list
    produces exactly one 'added' entry; everything else stays unchanged —
    the LCS alignment must not cascade the insertion into 'everything after
    it looks different'."""
    old = _cbs(BASE)
    new_specs = BASE[:2] + [("new rag chunk", "user", "rag")] + BASE[2:]
    new = _cbs(new_specs)

    diff = diff_calls(old, new, seq_old=1, seq_new=2)

    kinds = _kinds(diff)
    assert kinds.count("added") == 1
    assert kinds.count("evicted") == 0
    assert kinds.count("modified") == 0
    assert kinds.count("unchanged") == len(BASE)
    added = [e for e in diff.entries if e.kind == "added"][0]
    assert added.block.text == "new rag chunk"
    assert diff.tokens_added == len("new rag chunk")
    assert diff.tokens_evicted == 0


def test_delete_one_is_one_evicted_no_cascade():
    """Deleting one block produces exactly one 'evicted' entry; the rest stay
    unchanged."""
    old = _cbs(BASE)
    new_specs = BASE[:1] + BASE[2:]  # drop "rule one"
    new = _cbs(new_specs)

    diff = diff_calls(old, new, seq_old=1, seq_new=2)

    kinds = _kinds(diff)
    assert kinds.count("evicted") == 1
    assert kinds.count("added") == 0
    assert kinds.count("modified") == 0
    assert kinds.count("unchanged") == len(BASE) - 1
    evicted = [e for e in diff.entries if e.kind == "evicted"][0]
    assert evicted.block.text == "rule one"
    assert diff.tokens_evicted == len("rule one")
    assert diff.tokens_added == 0


def test_same_slot_text_change_is_modified_with_inline_diff():
    """Replacing a block's text in the same role/kind slot yields exactly one
    'modified' entry whose inline_diff contains the changed words as
    delete/insert segments (and the unchanged words as equal segments)."""
    old = _cbs(BASE)
    new_specs = list(BASE)
    new_specs[2] = ("hi there friend", "user", "user")  # same role, new text
    new = _cbs(new_specs)

    diff = diff_calls(old, new, seq_old=1, seq_new=2)

    kinds = _kinds(diff)
    assert kinds.count("modified") == 1
    assert kinds.count("added") == 0
    assert kinds.count("evicted") == 0

    modified = [e for e in diff.entries if e.kind == "modified"][0]
    assert modified.old_block is not None
    assert modified.old_block.text == "hi there"
    assert modified.block.text == "hi there friend"
    assert modified.inline_diff is not None

    inserted_text = "".join(t for op, t in modified.inline_diff if op == "insert")
    deleted_text = "".join(t for op, t in modified.inline_diff if op == "delete")
    equal_text = "".join(t for op, t in modified.inline_diff if op == "equal")
    assert "friend" in inserted_text
    assert deleted_text == ""  # pure append: nothing removed
    assert equal_text == "hi there"

    # Token budgeting: modified counts as evicted-old + added-new.
    assert diff.tokens_evicted == len("hi there")
    assert diff.tokens_added == len("hi there friend")


def test_replace_pair_with_different_roles_is_evict_plus_add_not_modified():
    """A replace pair whose old/new items have different roles is NOT a
    content edit — it must show as one evicted + one added, never
    'modified'."""
    old = _cbs(BASE)
    new_specs = list(BASE)
    # Swap the user turn for an assistant turn in the same slot -> role changes.
    new_specs[2] = ("assistant took over", "assistant", "history")
    new = _cbs(new_specs)

    diff = diff_calls(old, new, seq_old=1, seq_new=2)

    kinds = _kinds(diff)
    assert kinds.count("modified") == 0
    assert kinds.count("evicted") == 1
    assert kinds.count("added") == 1
    evicted = [e for e in diff.entries if e.kind == "evicted"][0]
    added = [e for e in diff.entries if e.kind == "added"][0]
    assert evicted.block.text == "hi there"
    assert added.block.text == "assistant took over"


def test_identical_lists_are_all_unchanged_zero_deltas():
    """Diffing a list against an identical copy of itself: everything is
    unchanged, and token deltas are exactly zero."""
    old = _cbs(BASE)
    new = _cbs(BASE)

    diff = diff_calls(old, new, seq_old=1, seq_new=2)

    kinds = _kinds(diff)
    assert kinds.count("unchanged") == len(BASE)
    assert kinds.count("added") == 0
    assert kinds.count("evicted") == 0
    assert kinds.count("modified") == 0
    assert diff.tokens_added == 0
    assert diff.tokens_evicted == 0


def test_moved_block_is_unchanged_not_added_evicted():
    """A block that kept its exact content but moved position (reordered
    list, same hashes) must produce no added/evicted entries and zero token
    deltas for that content — a pure move, not a change."""
    old = _cbs(BASE)
    # Move "rule one" (index 1) to the end; everything else keeps its order.
    reordered = [BASE[0], BASE[2], BASE[3], BASE[1]]
    new = _cbs(reordered)

    diff = diff_calls(old, new, seq_old=1, seq_new=2)

    kinds = _kinds(diff)
    assert kinds.count("added") == 0
    assert kinds.count("evicted") == 0
    assert kinds.count("modified") == 0
    assert kinds.count("unchanged") == len(BASE)
    assert diff.tokens_added == 0
    assert diff.tokens_evicted == 0

    moved = [e for e in diff.entries if e.block.text == "rule one"][0]
    assert moved.position_old == 1
    assert moved.position_new == 3


def test_duplicate_hash_net_removal():
    """Old has the SAME block (identical text/role/kind -> identical hash) at
    two positions; new has it once. This must resolve to exactly one
    'evicted' entry for that hash (the net-removed copy), not two, and the
    token-delta invariant must hold."""
    dup = ("shared rule", "system", "system")
    other = ("hi there", "user", "user")
    old = _cbs([dup, other, dup])   # dup appears twice
    new = _cbs([dup, other])        # dup appears once

    diff = diff_calls(old, new, seq_old=1, seq_new=2)

    kinds = _kinds(diff)
    assert kinds.count("evicted") == 1
    assert kinds.count("added") == 0
    assert kinds.count("modified") == 0
    # Two unchanged entries survive: the "other" block and the ONE surviving
    # copy of the duplicate (not both copies -- there's only one in `new`).
    assert kinds.count("unchanged") == 2

    evicted = [e for e in diff.entries if e.kind == "evicted"][0]
    assert evicted.block.text == "shared rule"
    assert diff.tokens_evicted == len("shared rule")
    assert diff.tokens_added == 0

    old_sum = sum(cb.block.token_count for cb in old)
    new_sum = sum(cb.block.token_count for cb in new)
    assert diff.tokens_added - diff.tokens_evicted == new_sum - old_sum


def test_duplicate_hash_net_addition():
    """Mirror of the removal case: old has the block once, new has it twice.
    Exactly one 'added' entry for that hash, and the other occurrence is
    accounted for as unchanged (not a second spurious 'added')."""
    dup = ("shared rule", "system", "system")
    other = ("hi there", "user", "user")
    old = _cbs([dup, other])        # dup appears once
    new = _cbs([dup, other, dup])   # dup appears twice

    diff = diff_calls(old, new, seq_old=1, seq_new=2)

    kinds = _kinds(diff)
    assert kinds.count("added") == 1
    assert kinds.count("evicted") == 0
    assert kinds.count("modified") == 0
    assert kinds.count("unchanged") == 2  # "other" + the surviving dup copy

    added = [e for e in diff.entries if e.kind == "added"][0]
    assert added.block.text == "shared rule"
    assert diff.tokens_added == len("shared rule")
    assert diff.tokens_evicted == 0

    old_sum = sum(cb.block.token_count for cb in old)
    new_sum = sum(cb.block.token_count for cb in new)
    assert diff.tokens_added - diff.tokens_evicted == new_sum - old_sum


def test_move_reconciliation_adjusts_token_totals():
    """A pure move that SequenceMatcher's LCS alignment splits into a
    delete+insert of the same hash (rather than folding into 'equal') must
    be fully reconciled by the post-pass: no evicted/added entries survive
    for that hash, and — the direct proof totals are computed AFTER
    reconciliation, not before — tokens_added == tokens_evicted == 0 for the
    whole diff, even though the raw opcode pass produced a delete and an
    insert before the post-pass ran."""
    old = _cbs(BASE)
    # Move "rule one" (index 1) to the end; everything else keeps its
    # relative order, which is exactly the shape that breaks the LCS match
    # and forces the moved block through the delete+insert path (see
    # test_moved_block_is_unchanged_not_added_evicted for the same shape).
    reordered = [BASE[0], BASE[2], BASE[3], BASE[1]]
    new = _cbs(reordered)

    diff = diff_calls(old, new, seq_old=1, seq_new=2)

    moved_hash = old[1].block.content_hash  # "rule one"
    survivors = [e for e in diff.entries
                 if e.block.content_hash == moved_hash and e.kind in ("evicted", "added")]
    assert survivors == []

    assert diff.tokens_added == 0
    assert diff.tokens_evicted == 0


def test_property_token_delta_invariant_over_random_shuffles():
    """For random shuffle/insert/delete/replace mutations of a base list
    (seeded — never unseeded, per the test plan), the invariant
    tokens_added - tokens_evicted == sum(new tokens) - sum(old tokens) must
    hold, since unchanged/moved blocks contribute equally to both sides and
    cancel out."""
    rng = random.Random(42)
    pool = [
        ("alpha block", "system", "system"),
        ("beta block", "user", "user"),
        ("gamma block", "assistant", "history"),
        ("delta block", "user", "rag"),
        ("epsilon block", "tool", "tool_output"),
        ("zeta block", "system", "system"),
        ("eta block", "user", "user"),
        ("theta block", "assistant", "history"),
    ]

    for trial in range(30):
        old_specs = rng.sample(pool, k=rng.randint(2, len(pool)))
        new_specs = list(old_specs)
        rng.shuffle(new_specs)
        # Randomly insert, delete, or mutate a couple of entries.
        for _ in range(rng.randint(0, 3)):
            action = rng.choice(["insert", "delete", "replace"])
            if action == "insert" and len(new_specs) < len(pool):
                candidate = rng.choice([s for s in pool if s not in new_specs] or pool)
                new_specs.insert(rng.randint(0, len(new_specs)), candidate)
            elif action == "delete" and new_specs:
                new_specs.pop(rng.randint(0, len(new_specs) - 1))
            elif action == "replace" and new_specs:
                idx = rng.randint(0, len(new_specs) - 1)
                text, role, label = new_specs[idx]
                new_specs[idx] = (text + f" edited-{trial}", role, label)

        old = _cbs(old_specs)
        new = _cbs(new_specs)

        diff = diff_calls(old, new, seq_old=trial, seq_new=trial + 1)

        old_sum = sum(cb.block.token_count for cb in old)
        new_sum = sum(cb.block.token_count for cb in new)
        assert diff.tokens_added - diff.tokens_evicted == new_sum - old_sum
