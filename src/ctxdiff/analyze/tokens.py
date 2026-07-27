"""The token attributor (spec §6.3): pure functions that turn a call's blocks
into "where did the budget go" data, and cross-reference tool_schema blocks
against actual tool usage to flag wasted (never-invoked) schema tokens. Like
differ.py, this module does no I/O and no color — it reads store data
(Call/CallBlock) and returns frozen dataclasses for the CLI/viewer to render."""
from __future__ import annotations

import json
from dataclasses import dataclass

from ctxdiff.analyze.differ import distinct_agents, filter_calls
from ctxdiff.models import CallBlock
from ctxdiff.store.base import Call, Store

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
    `provider_usage`; None when there's no usage or no recognizable key.

    `unmeasured_blocks` counts the blocks in this call whose cost is not merely
    estimated but UNKNOWN: non-empty content that the 'estimate' method priced
    at zero. The estimator never returns zero for non-empty text (it rounds up
    to at least one token), so the only way to land here is a block that
    declined to guess at all — an image whose bytes we refused to fetch, a
    `file_id` reference, a format the sniffer does not recognize. Those cost the
    provider real tokens that this total does not contain, which makes
    `total_tokens` a FLOOR rather than a measurement for such a call. Kept as a
    count rather than a bool so a report can say how many blocks are missing."""
    seq: int
    total_tokens: int
    approximate: bool
    slices: list[LabelSlice]
    provider_usage: dict | None
    reconciliation_delta: int | None
    agent: str | None = None  # v2 attribution passthrough from the Call, so the
    step: str | None = None   # CLI/viewer can label a turn's panels [agent·step]
    unmeasured_blocks: int = 0  # blocks priced at 0 by the estimator despite
    # carrying content — see the class docstring; `check` refuses to certify a
    # budget from a total that contains one


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
class UsageTotals:
    """Run-level rollup of PROVIDER-REPORTED usage (not our own block counts):
    summed input/output tokens across the run, normalized over the four
    provider key shapes. Calls that reported no usage are skipped from the sums
    but still counted in `calls_total`, so coverage can be reported honestly
    ('5/6 calls reported usage') rather than implying a complete total from a
    partial one. `by_agent` maps each agent label to an (input, output) tuple
    ('(unlabeled)' for None), populated only when the run spans ≥2 distinct
    agent labels (else None), and only for agents that actually reported
    usage."""
    input_tokens: int
    output_tokens: int
    calls_with_usage: int
    calls_total: int
    by_agent: dict[str, tuple[int, int]] | None
    # How many of the no-usage calls are DIAGNOSABLE as OpenAI-style streams
    # sent without `stream_options: {"include_usage": true}` — the one case
    # where the missing usage has a one-line caller-side fix. Recognized off
    # the recorded params (wire shape, not provider label): a `messages` list
    # plus a truthy `stream` and no `include_usage`. ctxdiff never injects the
    # option itself (wire-truth), so the honest move is to NAME the remedy in
    # the renderer instead of leaving "no provider usage reported" as a dead
    # end (dogfood finding 2026-07-27).
    streamed_without_usage: int = 0


@dataclass(frozen=True)
class RunTokens:
    """A whole run's token attribution: every call's CallTokens, the run-level
    BloatReport (None when the run has no tool_schema blocks at all — there is
    nothing to report bloat about), and the provider-usage rollup (`usage`).
    `by_agent` maps each agent label to its total BLOCK tokens (None-labeled
    calls collected under '(unlabeled)') — but only when the analyzed calls
    span ≥2 distinct agent labels; it is None for a single-agent run or an
    --agent-filtered analysis, where a per-agent breakdown would be trivial
    noise."""
    calls: list[CallTokens]
    bloat: BloatReport | None
    usage: UsageTotals
    by_agent: dict[str, int] | None = None


# --- provider usage reconciliation --------------------------------------------

# Prompt-side token count key, by provider wire shape. Order matters: the
# first key present (with a non-None value) in a call's usage dict wins,
# since a usage dict only ever carries one provider's shape at a time.
_PROMPT_TOKEN_KEYS = ("prompt_tokens", "input_tokens", "prompt_token_count", "inputTokens")

# Output/completion-side token count key, by provider wire shape — the mirror of
# _PROMPT_TOKEN_KEYS. Same first-present-wins rule (a usage dict only ever
# carries one provider's shape at a time).
_OUTPUT_TOKEN_KEYS = ("completion_tokens", "output_tokens", "candidates_token_count", "outputTokens")


def _first_present(usage: dict | None, keys: tuple[str, ...]) -> int | None:
    """First key in `keys` present in `usage` with a non-None value, or None.
    A key mapped to None (an adapter's getattr-default when the SDK object
    lacked the attribute) is treated as absent — the next candidate is tried."""
    if not usage:
        return None
    for key in keys:
        value = usage.get(key)
        if value is not None:
            return value
    return None


def _reconciliation_delta(usage: dict | None, total_tokens: int) -> int | None:
    """Provider-reported prompt-side tokens minus our own summed total, or
    None when there's no usage dict or none of the known prompt-token keys is
    present with a real value. A key present but mapped to None (e.g. an
    adapter's getattr-with-default-None when the SDK object lacks the
    attribute) is treated as absent — only the next candidate key is tried,
    not "reconciled against None"."""
    value = _first_present(usage, _PROMPT_TOKEN_KEYS)
    return None if value is None else value - total_tokens


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
    those totals and sorted biggest-first.

    The same pass counts UNMEASURED blocks — an 'estimate' block that priced
    non-empty content at zero tokens. `_estimate_count` rounds any non-empty
    text up to at least one token, so a zero there is never a small estimate:
    it is the image pipeline saying "this cost cannot be known" (a remote URL
    we refused to fetch, a `file_id`, an unrecognized format). Counting them
    here, in the one place a call's blocks are already walked, is what lets
    `ctxdiff check` refuse to certify a budget against a total it knows is a
    floor."""
    label_tokens: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    approximate = False
    unmeasured = 0
    for cb in call_blocks:
        label_tokens[cb.label] = label_tokens.get(cb.label, 0) + cb.block.token_count
        label_counts[cb.label] = label_counts.get(cb.label, 0) + 1
        if cb.block.token_method == "estimate":
            approximate = True
            if cb.block.token_count == 0 and cb.block.text:
                unmeasured += 1

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
        agent=call.agent,
        step=call.step,
        unmeasured_blocks=unmeasured,
    )


# Bucket label for calls with no agent, used only in the by_agent breakdown so
# unlabeled calls still surface as their own line rather than vanishing.
_UNLABELED = "(unlabeled)"


def _by_agent_totals(
    calls: list[Call], all_blocks: list[list[CallBlock]]
) -> dict[str, int] | None:
    """Total block tokens per agent label across `calls`, or None when the run
    spans fewer than 2 distinct agent labels (nothing worth breaking down).
    None-labeled calls accumulate under '(unlabeled)'. Insertion order follows
    first appearance, so the breakdown reads in the order agents entered the
    run."""
    if len(distinct_agents(calls)) < 2:
        return None
    totals: dict[str, int] = {}
    for c, blocks in zip(calls, all_blocks):
        key = c.agent if c.agent is not None else _UNLABELED
        totals[key] = totals.get(key, 0) + sum(cb.block.token_count for cb in blocks)
    return totals


def usage_totals(calls: list[Call]) -> UsageTotals:
    """Roll up PROVIDER-REPORTED usage across `calls` into a UsageTotals. How:
    for each call, pull the input and output token counts by trying the four
    provider key shapes in order (first-present wins, mirroring reconciliation);
    a call that yields NEITHER an input nor an output number reported no usage —
    it is skipped from the sums but still counted in `calls_total`, so coverage
    stays honest. Per-agent (input, output) tuples are accumulated in parallel
    and surfaced only when the run spans ≥2 distinct agent labels (else None),
    keyed '(unlabeled)' for None-agent calls and limited to agents that
    actually reported usage, in first-appearance order."""
    input_total = 0
    output_total = 0
    with_usage = 0
    per_in: dict[str, int] = {}
    per_out: dict[str, int] = {}
    streamed_without_usage = 0
    for c in calls:
        in_v = _first_present(c.usage, _PROMPT_TOKEN_KEYS)
        out_v = _first_present(c.usage, _OUTPUT_TOKEN_KEYS)
        if in_v is None and out_v is None:
            # No recognizable provider usage. Before skipping, check whether
            # this is the fixable case: an OpenAI-chat-shaped streamed request
            # (`messages` + `stream`) that never asked for usage via
            # `stream_options.include_usage` — OpenAI's streams only emit a
            # usage chunk when the caller opts in, and ctxdiff won't inject
            # the option. Counted here so the renderer can print the remedy.
            p = c.params or {}
            opts = p.get("stream_options") or {}
            if (p.get("stream") and isinstance(p.get("messages"), list)
                    and not (isinstance(opts, dict) and opts.get("include_usage"))):
                streamed_without_usage += 1
            continue  # skipped from sums; still counted in calls_total
        with_usage += 1
        input_total += in_v or 0
        output_total += out_v or 0
        key = c.agent if c.agent is not None else _UNLABELED
        per_in[key] = per_in.get(key, 0) + (in_v or 0)
        per_out[key] = per_out.get(key, 0) + (out_v or 0)

    by_agent: dict[str, tuple[int, int]] | None = None
    if len(distinct_agents(calls)) >= 2:
        # first-appearance order, only agents that reported usage
        by_agent = {}
        for label in distinct_agents(calls):
            key = label if label is not None else _UNLABELED
            if key in per_in or key in per_out:
                by_agent[key] = (per_in.get(key, 0), per_out.get(key, 0))

    return UsageTotals(
        input_tokens=input_total, output_tokens=output_total,
        calls_with_usage=with_usage, calls_total=len(calls),
        by_agent=by_agent, streamed_without_usage=streamed_without_usage,
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


def analyze_run(ct: Store, agent: str | None = None) -> RunTokens:
    """Convenience wrapper: load the run's calls (filtered to `agent` when
    given, else all), attribute each call's tokens, run bloat detection once
    across the analyzed calls, and compute the per-agent token breakdown
    (`by_agent`, non-None only for an unfiltered multi-agent run)."""
    calls = filter_calls(ct.get_calls(), agent)
    all_calls_with_blocks = [ct.get_call_blocks(c.id) for c in calls]
    call_tokens = [
        analyze_call(call, blocks)
        for call, blocks in zip(calls, all_calls_with_blocks)
    ]
    bloat = detect_bloat(all_calls_with_blocks)
    by_agent = _by_agent_totals(calls, all_calls_with_blocks)
    usage = usage_totals(calls)
    return RunTokens(calls=call_tokens, bloat=bloat, usage=usage, by_agent=by_agent)
