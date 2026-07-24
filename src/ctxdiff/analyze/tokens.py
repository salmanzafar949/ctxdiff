"""The token attributor (spec §6.3): pure functions that turn a call's blocks
into "where did the budget go" data, and cross-reference tool_schema blocks
against actual tool usage to flag wasted (never-invoked) schema tokens. Like
differ.py, this module does no I/O and no color — it reads store data
(Call/CallBlock) and returns frozen dataclasses for the CLI/viewer to render."""
from __future__ import annotations

import json
from dataclasses import dataclass

from ctxdiff.models import CallBlock
from ctxdiff.store.ctrace import Call, CTrace

# --- value types -------------------------------------------------------------


@dataclass(frozen=True)
class LabelSlice:
    """One label's share of a call's token budget. `pct` is `tokens` as a
    percentage of the call's total block tokens, rounded to 1 decimal —
    slices across a call therefore sum to ~100.0 (rounding may drift it by a
    tenth or two, which is fine for a heatmap, not a ledger)."""
    label: str
    tokens: int
    block_count: int
    pct: float


@dataclass(frozen=True)
class CallTokens:
    """One call's full token attribution. `approximate` is True iff ANY block
    in the call used the 'estimate' token method — a single estimated block
    is enough to make the whole call's total not-exact, so the UI must mark
    the whole thing rather than imply false precision. `slices` is sorted by
    `tokens` descending (biggest spender first). `provider_usage` is the raw
    usage dict from the call's response (provider-shaped, may be None).
    `reconciliation_delta` is provider-reported prompt-side tokens minus our
    own summed total, when a recognizable prompt-token key is present in
    `provider_usage`; None when there's no usage or no recognizable key."""
    seq: int
    total_tokens: int
    approximate: bool
    slices: list[LabelSlice]
    provider_usage: dict | None
    reconciliation_delta: int | None


@dataclass(frozen=True)
class BloatReport:
    """Cross-run schema-bloat summary (spec §6.3). `unused_tools` names every
    registered tool whose schema is never referenced anywhere in the run.
    `unused_tokens_per_call` is the token cost of those unused schemas in ONE
    call (schemas are re-sent verbatim every call, so this is the recurring
    per-turn tax, not a run-wide sum). `calls_analyzed` is how many calls the
    detector looked at. `pct_of_avg_context` expresses that recurring cost as
    a percentage of the average call's total context size."""
    unused_tools: list[str]
    unused_tokens_per_call: int
    calls_analyzed: int
    pct_of_avg_context: float


@dataclass(frozen=True)
class RunTokens:
    """A whole run's token attribution: every call's CallTokens plus the
    run-level BloatReport (None when the run has no tool_schema blocks at
    all — there is nothing to report bloat about)."""
    calls: list[CallTokens]
    bloat: BloatReport | None


# --- provider usage reconciliation --------------------------------------------

# Prompt-side token count key, by provider wire shape. Order matters: the
# first key present (with a non-None value) in a call's usage dict wins,
# since a usage dict only ever carries one provider's shape at a time.
_PROMPT_TOKEN_KEYS = ("prompt_tokens", "input_tokens", "prompt_token_count", "inputTokens")


def _reconciliation_delta(usage: dict | None, total_tokens: int) -> int | None:
    """Provider-reported prompt-side tokens minus our own summed total, or
    None when there's no usage dict or none of the known prompt-token keys is
    present with a real value. A key present but mapped to None (e.g. an
    adapter's getattr-with-default-None when the SDK object lacks the
    attribute) is treated as absent — only the next candidate key is tried,
    not "reconciled against None"."""
    if not usage:
        return None
    for key in _PROMPT_TOKEN_KEYS:
        value = usage.get(key)
        if value is not None:
            return value - total_tokens
    return None


# --- schema-name extraction ---------------------------------------------------

# Nested containers a tool's name might live under, by provider wire shape:
# OpenAI wraps as {"function": {"name": ...}}; Bedrock's raw Converse toolSpec
# shape is {"toolSpec": {"name": ...}} (today's adapter already unwraps this
# before storing, but detection stays defensive in case that ever changes, or
# a hand-built .ctrace carries the raw shape).
_NESTED_NAME_CONTAINERS = ("function", "toolSpec")

# Sentinel returned when a schema's name can't be determined — from a JSON
# parse failure or an unrecognized shape. Never appears in a "used" set, so a
# sentinel schema can also never be reported as unused (used=unknown).
UNPARSED_TOOL_NAME = "<unparsed>"


def extract_tool_name(schema_text: str) -> str:
    """Defensively pull a tool's name out of a stored tool_schema block's
    JSON text. Tries, in order: a top-level "name" (Anthropic's/Gemini's bare
    {"name": ...} shape), then "name" nested under "function" (OpenAI) or
    "toolSpec" (Bedrock's raw wire shape). Any parse failure, non-dict JSON,
    or shape none of the above recognize returns UNPARSED_TOOL_NAME rather
    than raising — a malformed schema must never crash bloat detection."""
    try:
        parsed = json.loads(schema_text)
    except (json.JSONDecodeError, TypeError):
        return UNPARSED_TOOL_NAME
    if not isinstance(parsed, dict):
        return UNPARSED_TOOL_NAME

    name = parsed.get("name")
    if isinstance(name, str) and name:
        return name

    for container_key in _NESTED_NAME_CONTAINERS:
        container = parsed.get(container_key)
        if isinstance(container, dict):
            nested_name = container.get("name")
            if isinstance(nested_name, str) and nested_name:
                return nested_name

    return UNPARSED_TOOL_NAME


def registered_tool_names(all_calls_with_blocks: list[list[CallBlock]]) -> set[str]:
    """Every distinct tool name that appears in a tool_schema block anywhere
    in the run (excluding the unparsed sentinel). Separate from
    `detect_bloat` so callers that only need "how many tools are registered"
    (e.g. the CLI's "N of M" bloat message) don't have to re-derive it from a
    BloatReport, which only carries the unused subset."""
    names: set[str] = set()
    for call_blocks in all_calls_with_blocks:
        for cb in call_blocks:
            if cb.label == "tool_schema":
                name = extract_tool_name(cb.block.text)
                if name != UNPARSED_TOOL_NAME:
                    names.add(name)
    return names


# --- core analysis -------------------------------------------------------------


def analyze_call(call: Call, call_blocks: list[CallBlock]) -> CallTokens:
    """Attribute one call's blocks to CallTokens: group `token_count` by
    `label`, compute each slice's share of the call's total, and reconcile
    against provider usage when available. How: a single pass over
    `call_blocks` accumulates per-label token/count totals and notes whether
    any block used the 'estimate' token method; slices are then built from
    those totals and sorted biggest-first."""
    label_tokens: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    approximate = False
    for cb in call_blocks:
        label_tokens[cb.label] = label_tokens.get(cb.label, 0) + cb.block.token_count
        label_counts[cb.label] = label_counts.get(cb.label, 0) + 1
        if cb.block.token_method == "estimate":
            approximate = True

    total_tokens = sum(label_tokens.values())
    slices = [
        LabelSlice(
            label=label,
            tokens=tokens,
            block_count=label_counts[label],
            pct=round(tokens / total_tokens * 100, 1) if total_tokens else 0.0,
        )
        for label, tokens in label_tokens.items()
    ]
    slices.sort(key=lambda s: s.tokens, reverse=True)

    return CallTokens(
        seq=call.seq,
        total_tokens=total_tokens,
        approximate=approximate,
        slices=slices,
        provider_usage=call.usage,
        reconciliation_delta=_reconciliation_delta(call.usage, total_tokens),
    )


def detect_bloat(all_calls_with_blocks: list[list[CallBlock]]) -> BloatReport | None:
    """Cross-reference every tool_schema block in the run against tool names
    actually referenced anywhere else in the run (spec §6.3). How:

    1. Collect every distinct tool_schema block (deduped by content_hash,
       since the same schema is re-sent — and therefore re-stored/referenced
       — every call) and its parsed name + token cost.
    2. Return None immediately if there are no tool_schema blocks at all —
       there is nothing to report bloat about.
    3. Build one search haystack from every block that could plausibly carry
       a tool invocation: assistant-role blocks (where a tool call would be
       serialized) and blocks labeled 'tool_output' (where a tool result
       would be). A schema's name is "used" if it appears anywhere in that
       haystack — substring match, since tool-call/result wire shapes vary
       enough across providers that exact structural parsing isn't reliable
       here, and this is documented as an accepted approximation.
    4. A schema whose name is the UNPARSED_TOOL_NAME sentinel is skipped
       entirely (used=unknown) rather than assumed unused — reporting a
       schema we couldn't even name as "bloat" would be a false alarm we
       can't back up.
    5. unused_tokens_per_call sums the token cost of the unused schemas
       ONCE (not once per call) since they're the same schema resent every
       call; pct_of_avg_context expresses that against the average call's
       total token size across the whole run."""
    schemas_by_hash: dict[str, tuple[str, int]] = {}
    for call_blocks in all_calls_with_blocks:
        for cb in call_blocks:
            if cb.label == "tool_schema":
                schemas_by_hash.setdefault(
                    cb.block.content_hash,
                    (extract_tool_name(cb.block.text), cb.block.token_count),
                )

    if not schemas_by_hash:
        return None

    haystack_parts = [
        cb.block.text
        for call_blocks in all_calls_with_blocks
        for cb in call_blocks
        if cb.block.role == "assistant" or cb.label == "tool_output"
    ]
    haystack = "\n".join(haystack_parts)

    unused_tools: list[str] = []
    unused_tokens = 0
    for name, token_count in schemas_by_hash.values():
        if name == UNPARSED_TOOL_NAME:
            continue
        if name not in haystack:
            unused_tools.append(name)
            unused_tokens += token_count

    call_totals = [
        sum(cb.block.token_count for cb in call_blocks)
        for call_blocks in all_calls_with_blocks
    ]
    avg_total = sum(call_totals) / len(call_totals) if call_totals else 0
    pct = round(unused_tokens / avg_total * 100, 1) if avg_total else 0.0

    return BloatReport(
        unused_tools=unused_tools,
        unused_tokens_per_call=unused_tokens,
        calls_analyzed=len(all_calls_with_blocks),
        pct_of_avg_context=pct,
    )


def analyze_run(ct: CTrace) -> RunTokens:
    """Convenience wrapper: load every call and its blocks from `ct`, attribute
    each call's tokens, and run bloat detection once across the whole run."""
    calls = ct.get_calls()
    all_calls_with_blocks = [ct.get_call_blocks(c.id) for c in calls]
    call_tokens = [
        analyze_call(call, blocks)
        for call, blocks in zip(calls, all_calls_with_blocks)
    ]
    bloat = detect_bloat(all_calls_with_blocks)
    return RunTokens(calls=call_tokens, bloat=bloat)
