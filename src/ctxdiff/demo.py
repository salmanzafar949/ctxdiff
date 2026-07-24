r"""Builds a sample, self-describing multi-agent `.ctrace` — the payload behind
`ctxdiff demo` (see `ctxdiff.cli.main`). The whole point is zero-friction: no
API keys, no network, no provider SDK installed, and byte-for-byte the same
demo content on every run — so `pip install ctxdiff && ctxdiff demo` works
offline, instantly, for every user.

How it stays honest to the real capture path: rather than hand-writing rows
into a `.ctrace` (as most of this project's OWN test fixtures do, for speed),
this module drives the actual PUBLIC API — `trace.init()` / `tracer.wrap()` /
`tracer.tag()` — through two tiny fake clients built inline below. Each fake
mimics just enough of its real SDK's shape for `trace.wrap()`'s provider
detection and the OpenAI/Anthropic adapters' `extract_usage()` to work
(`__module__` set to the real package name, and a `.usage` object with the
real attribute names) — everything else (the request content) is plain
dicts/strings passed as `**kwargs`, exactly what a real call would look like.
The result is a trace that exercises the SAME recorder/adapter/store code a
real integration would, not a shortcut that could silently drift from it.

The scenario: a two-agent research pipeline summarizing evidence on
prompt-cache pricing across providers — `researcher` (OpenAI) searches and
verifies, `writer` (Anthropic) turns the findings into a leadership summary,
handing off back and forth. It is deliberately built to light up every panel
of the dashboard at once:

- a stable system prompt reused verbatim across the researcher's turns (block
  dedup), sitting right behind a small dynamic "current session time" block
  that changes every researcher turn — a realistic misconfiguration (a
  framework stamping a live clock ahead of the static instructions) that
  breaks the provider's cache prefix on every researcher turn while the
  writer's own prefix (no dynamic block) stays perfectly stable — so
  `ctxdiff cache` reports a break attributed to `researcher` ONLY, with a fix
  hint, while `writer` shows none.
- two tool schemas registered on every researcher turn, `search_web` (actually
  invoked, with a real tool_call + tool result recorded) and `delete_index`
  (registered but never called) — schema-bloat detection has something real
  to flag.
- a `tracer.tag("rag", ...)` chunk on the writer's synthesis turn, carrying
  the researcher's findings forward with an explicit provenance label instead
  of a role-based guess.
- growing per-agent history across six interleaved turns, so the context-
  growth chart and the run's token/usage rollup (researcher heavier on input
  from carrying tool schemas + search results; writer moderate) both have a
  real shape to show.

Determinism: every piece of scenario text is a literal, fixed string — no
`datetime.now()`, no `random`, no wall-clock timestamps — so calling
`build_demo_trace` twice produces two files with identical blocks, hashes,
and call structure (the only differences are the store's own inherent
per-file randomness — `run.id`/`call.id` UUIDs and `run.started_at`, which is
real wall-clock time stamped by `trace.wrap()` itself so the dashboard header
shows a genuine date — neither of which is scenario content)."""
from __future__ import annotations

import json

from ctxdiff import trace

# --- fake provider clients ---------------------------------------------------
#
# No provider SDK is imported or required to be installed: `trace.wrap()`
# detects a provider purely from `type(client).__module__` (see
# `trace._detect_provider`), and the adapters' `extract_usage()` only ever
# duck-types a response object's `.usage` attribute. So each fake below is the
# minimum shape that satisfies both: a class whose `__module__` is spoofed to
# the real package name, and a response object carrying a plain `.usage` with
# the real provider's attribute names. No network call is possible because
# nothing here does I/O — `create()` just returns the next canned response.


class _OpenAIUsage:
    """Duck-types `openai`'s Chat Completions usage shape (see
    `capture.openai.OpenAIAdapter.extract_usage`)."""

    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _OpenAIResponse:
    """The only attribute the adapter reads off a response: `.usage`. Request
    content lives entirely in the call's kwargs (mirroring the wire — a real
    response's `.choices[].message` would echo the request back, but ctxdiff
    only records what was SENT, not what came back, so there is nothing to
    reconstruct here)."""

    def __init__(self, usage: _OpenAIUsage) -> None:
        self.usage = usage


class _FakeChatCompletions:
    """Stands in for `client.chat.completions`. Returns the next usage tuple
    from a fixed, pre-scripted list — one per call, in call order — so each
    turn's usage numbers are deliberate and reproducible rather than
    computed."""

    def __init__(self, usages: list[tuple[int, int]]) -> None:
        self._usages = list(usages)
        self._i = 0

    def create(self, **kwargs: object) -> _OpenAIResponse:
        prompt_tokens, completion_tokens = self._usages[self._i]
        self._i += 1
        return _OpenAIResponse(_OpenAIUsage(prompt_tokens, completion_tokens))


class _FakeChat:
    def __init__(self, usages: list[tuple[int, int]]) -> None:
        self.completions = _FakeChatCompletions(usages)


class _FakeOpenAIClient:
    """An OpenAI-shaped fake client: `__module__` is the real package root so
    `trace._detect_provider` resolves it to the `openai` adapter exactly as it
    would a real `OpenAI()` instance."""

    __module__ = "openai"

    def __init__(self, usages: list[tuple[int, int]]) -> None:
        self.chat = _FakeChat(usages)


class _AnthropicUsage:
    """Duck-types Anthropic's Messages usage shape (input_tokens/
    output_tokens — see `capture.anthropic.AnthropicAdapter.extract_usage`)."""

    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _AnthropicResponse:
    def __init__(self, usage: _AnthropicUsage) -> None:
        self.usage = usage


class _FakeMessages:
    """Stands in for `client.messages`; same scripted-usage-list contract as
    `_FakeChatCompletions` above."""

    def __init__(self, usages: list[tuple[int, int]]) -> None:
        self._usages = list(usages)
        self._i = 0

    def create(self, **kwargs: object) -> _AnthropicResponse:
        input_tokens, output_tokens = self._usages[self._i]
        self._i += 1
        return _AnthropicResponse(_AnthropicUsage(input_tokens, output_tokens))


class _FakeAnthropicClient:
    """An Anthropic-shaped fake client — same `__module__`-spoofing trick as
    `_FakeOpenAIClient`, resolving to the `anthropic` adapter."""

    __module__ = "anthropic"

    def __init__(self, usages: list[tuple[int, int]]) -> None:
        self.messages = _FakeMessages(usages)


# --- scenario content ---------------------------------------------------------
#
# Every string below is a fixed literal — no clock reads, no randomness — so
# the trace this module builds is deterministic across runs (see the module
# docstring). Three timestamps stand in for a research session's real
# clock — chosen fixed rather than computed so re-running the demo builder
# never produces a different byte of scenario content.

_TS1 = "2026-07-24T09:58:03Z"
_TS2 = "2026-07-24T09:59:47Z"
_TS3 = "2026-07-24T10:02:15Z"

# Researcher's stable instructions — identical text every researcher turn, so
# it's stored once and referenced three times (the block-dedup story). Kept as
# its own message SEPARATE from the dynamic timestamp block below it, which
# occupies the position right in front of it — see the module docstring for
# why that specific ordering is the point.
_RESEARCHER_SYSTEM = (
    "You are a research analyst agent in a two-agent pipeline. Your job is "
    "to find and verify evidence from public vendor documentation, then hand "
    "off precise, sourced findings to a writer agent. Always name your "
    "source (vendor + doc section). Do not speculate beyond what a source "
    "states."
)

_WRITER_SYSTEM = (
    "You are a technical writer agent. Turn the research findings you're "
    "given into a clear, well-cited summary for engineering leadership. Keep "
    "it tight and never introduce a claim the research didn't supply."
)

_RESEARCH_QUESTION = (
    "Summarize the most recent evidence on prompt-cache pricing across "
    "OpenAI, Anthropic, and Gemini — specifically how cache writes vs cache "
    "reads are billed, and whether the discount changes with prefix length."
)

_SEARCH_WEB_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search public documentation for a query and return top results.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}

# Registered but never invoked anywhere in the scenario — the schema-bloat
# detector (analyze.tokens.detect_bloat) should name exactly this tool.
_DELETE_INDEX_SCHEMA = {
    "type": "function",
    "function": {
        "name": "delete_index",
        "description": "Permanently delete a named search index. Destructive; requires confirmation.",
        "parameters": {
            "type": "object",
            "properties": {"index_name": {"type": "string"}},
            "required": ["index_name"],
        },
    },
}

_TOOLS = [_SEARCH_WEB_SCHEMA, _DELETE_INDEX_SCHEMA]

# The model's own tool invocation, echoed back verbatim on every later turn
# (a real agent loop replays the assistant's tool_calls message as history) —
# reused as the SAME dict object so its serialized text (and therefore its
# content hash) is identical everywhere it appears.
_TOOL_CALL_SEARCH = {
    "id": "call_search_1",
    "type": "function",
    "function": {
        "name": "search_web",
        "arguments": json.dumps(
            {"query": "prompt cache pricing OpenAI Anthropic Gemini cache write vs read 2026"},
            sort_keys=True,
        ),
    },
}

_SEARCH_RESULTS = (
    "Search results:\n"
    "1. OpenAI docs, 'Prompt Caching' (2026-06): cached input tokens billed "
    "at 50% of standard input price; a cache hit requires an identical "
    "prefix of at least 1024 tokens.\n"
    "2. Anthropic docs, 'Prompt caching' (2026-05): cache writes cost ~25% "
    "MORE than a normal input token (a one-time write premium); cache reads "
    "cost roughly 10% of a normal input token; entries expire after 5 "
    "minutes of inactivity.\n"
    "3. Google Gemini docs, 'Context caching' (2026-06): a flat per-hour "
    "storage fee plus a reduced per-token rate on cache hits; the minimum "
    "cacheable content size varies by model."
)

_ANALYSIS_1 = (
    "Findings so far: all three vendors discount cache HITS relative to a "
    "fresh input token, but the shapes differ — OpenAI's discount is a flat "
    "~50% with a 1024-token minimum prefix; Anthropic charges a write "
    "premium (~+25%) but drops reads to ~10% of normal price; Gemini bills a "
    "separate hourly storage fee on top of a reduced hit rate. Sourced from "
    "each vendor's own caching docs. Handing off to the writer agent for a "
    "leadership-ready summary."
)

_FOLLOWUP_QUESTION = (
    "Can you confirm Anthropic's cache write premium applies per write, not "
    "per token stored — and cite the doc section?"
)

# Fed to the writer via tracer.tag("rag", [_RAG_FINDINGS]) — this exact text is
# embedded verbatim inside _SYNTHESIS_PROMPT below, so the tag's substring
# match labels that whole block "rag" instead of the role-based "user" guess.
_RAG_FINDINGS = (
    "OpenAI: ~50% discount on cache hits, 1024-token minimum prefix, no "
    "write premium. Anthropic: ~25% write premium, cache reads ~10% of "
    "normal input price, 5-minute TTL. Gemini: flat hourly storage fee plus "
    "a reduced per-token hit rate."
)

_SYNTHESIS_PROMPT = (
    "Turn the research findings below into a concise, engineering-"
    "leadership-ready summary (3-4 sentences, no jargon) comparing how "
    "OpenAI, Anthropic, and Gemini price prompt-cache hits and writes.\n\n"
    f"Findings:\n{_RAG_FINDINGS}"
)

_DRAFT_V1 = (
    "Prompt caching now meaningfully changes unit economics for high-"
    "repetition agent workloads. OpenAI gives a flat ~50% discount on cache "
    "hits once a request's shared prefix passes 1024 tokens, with no extra "
    "charge to populate the cache. Anthropic instead charges a one-time "
    "~25% premium to write into cache but drops the price of a cache hit to "
    "roughly a tenth of normal input, and evicts unused entries after five "
    "minutes. Gemini's model is the most different: a flat hourly storage "
    "fee on top of a reduced per-token rate on hits, which favors long-"
    "lived, high-traffic caches over bursty ones."
)

_NEW_FINDING_FOR_WRITER = (
    "Confirmed with the research agent: Anthropic's ~25% cache-write "
    "premium is a one-time, per-request charge (not per-token storage) — "
    "see 'Prompt caching' docs, Pricing section."
)

_REVISION_REQUEST = (
    "Please tighten the summary to under 120 words and add one caveat "
    "sentence noting that Anthropic's write premium is a one-time, per-"
    "request charge, not ongoing storage — per the research agent's "
    "confirmation above."
)

_DRAFT_V2 = (
    "Prompt caching now materially changes the economics of high-repetition "
    "agent workloads. OpenAI discounts cache hits ~50% once a shared prefix "
    "exceeds 1024 tokens, at no extra write cost. Anthropic charges a one-"
    "time ~25% premium to populate the cache but discounts hits to roughly "
    "a tenth of normal input price, with a five-minute idle TTL. Gemini "
    "instead bills a flat hourly storage fee plus a reduced per-token hit "
    "rate, favoring long-lived caches. Caveat: Anthropic's write premium is "
    "a one-time, per-request charge, not ongoing storage."
)

_FINALIZE_REQUEST = (
    "This reads well — finalize it as the closing paragraph of the report; "
    "flag anything that still needs a citation."
)

# Usage figures, in call order per agent — fixed constants, chosen so the
# researcher (carrying tool schemas + search results every turn) reads
# heavier on input than the writer, and both grow turn over turn as their
# own history grows.
_RESEARCHER_USAGES: list[tuple[int, int]] = [
    (850, 60),      # turn 1: opening question, no history yet
    (1400, 180),    # turn 2: + tool call/result now in context
    (1900, 150),    # turn 4: + prior analysis + follow-up question
]
_WRITER_USAGES: list[tuple[int, int]] = [
    (520, 210),     # turn 3: first synthesis
    (760, 240),     # turn 5: + draft + revision request
    (980, 190),     # turn 6: + second draft + finalize request
]


def build_demo_trace(path: str) -> str:
    """Build a realistic, deterministic multi-agent `.ctrace` at `path` via
    the real public capture API (`trace.init`/`wrap`/`tag`) and return `path`.
    No network access and no provider SDK import — see the module docstring
    for the fake-client mechanism and the module-level constants above for
    the (fixed, non-random) scenario content this drives through it.

    Six calls, interleaved across two agents (see the module docstring for
    what each is designed to demonstrate): researcher opens the investigation
    and gets a tool result back (turns 1-2), writer drafts a first synthesis
    off a tagged rag chunk (turn 3), researcher confirms one more detail
    (turn 4), and writer revises then finalizes (turns 5-6) — two agent
    hand-offs on the timeline, each turn's request built from the fixed
    strings above so the trace is identical in content across builds."""
    tracer = trace.init("research-pipeline-demo", path=path)
    researcher = tracer.wrap(_FakeOpenAIClient(_RESEARCHER_USAGES), agent="researcher")
    writer = tracer.wrap(_FakeAnthropicClient(_WRITER_USAGES), agent="writer")

    # turn 1 (seq 1) — researcher opens the investigation and calls search_web.
    researcher.chat.completions.create(
        model="gpt-4o",
        tools=_TOOLS,
        messages=[
            {"role": "system", "content": f"Current session time: {_TS1}"},
            {"role": "system", "content": _RESEARCHER_SYSTEM},
            {"role": "user", "content": _RESEARCH_QUESTION},
        ],
    )

    # turn 2 (seq 2) — the tool result comes back; researcher produces a first
    # analysis. The dynamic timestamp block changed (TS1 -> TS2): this is the
    # researcher's first cache-prefix break.
    researcher.chat.completions.create(
        model="gpt-4o",
        tools=_TOOLS,
        messages=[
            {"role": "system", "content": f"Current session time: {_TS2}"},
            {"role": "system", "content": _RESEARCHER_SYSTEM},
            {"role": "user", "content": _RESEARCH_QUESTION},
            {"role": "assistant", "content": None, "tool_calls": [_TOOL_CALL_SEARCH]},
            {"role": "tool", "tool_call_id": "call_search_1", "content": _SEARCH_RESULTS},
        ],
    )

    # turn 3 (seq 3) — hand-off to the writer: a rag-tagged synthesis request
    # carrying the researcher's findings forward with an explicit provenance
    # label. tag() applies to this next recorded call only.
    tracer.tag("rag", [_RAG_FINDINGS])
    writer.messages.create(
        model="claude-3-5-sonnet-20241022",
        system=_WRITER_SYSTEM,
        messages=[{"role": "user", "content": _SYNTHESIS_PROMPT}],
    )

    # turn 4 (seq 4) — hand-off back to the researcher for one more
    # confirmation. TS2 -> TS3: the researcher's second cache-prefix break.
    researcher.chat.completions.create(
        model="gpt-4o",
        tools=_TOOLS,
        messages=[
            {"role": "system", "content": f"Current session time: {_TS3}"},
            {"role": "system", "content": _RESEARCHER_SYSTEM},
            {"role": "user", "content": _RESEARCH_QUESTION},
            {"role": "assistant", "content": None, "tool_calls": [_TOOL_CALL_SEARCH]},
            {"role": "tool", "tool_call_id": "call_search_1", "content": _SEARCH_RESULTS},
            {"role": "assistant", "content": _ANALYSIS_1},
            {"role": "user", "content": _FOLLOWUP_QUESTION},
        ],
    )

    # turn 5 (seq 5) — hand-off to the writer again: revise with the
    # researcher's confirmation folded in. Pure append after the writer's
    # stable prefix (system + synthesis prompt) — no cache break here.
    writer.messages.create(
        model="claude-3-5-sonnet-20241022",
        system=_WRITER_SYSTEM,
        messages=[
            {"role": "user", "content": _SYNTHESIS_PROMPT},
            {"role": "assistant", "content": _DRAFT_V1},
            {"role": "user", "content": f"{_NEW_FINDING_FOR_WRITER}\n\n{_REVISION_REQUEST}"},
        ],
    )

    # turn 6 (seq 6) — writer finalizes. Same agent as turn 5 (no hand-off),
    # pure append again — still cache-stable.
    writer.messages.create(
        model="claude-3-5-sonnet-20241022",
        system=_WRITER_SYSTEM,
        messages=[
            {"role": "user", "content": _SYNTHESIS_PROMPT},
            {"role": "assistant", "content": _DRAFT_V1},
            {"role": "user", "content": f"{_NEW_FINDING_FOR_WRITER}\n\n{_REVISION_REQUEST}"},
            {"role": "assistant", "content": _DRAFT_V2},
            {"role": "user", "content": _FINALIZE_REQUEST},
        ],
    )

    tracer.close()
    return path
