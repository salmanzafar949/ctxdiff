import json

import pytest

from ctxdiff.analyze.tokens import (
    analyze_call,
    analyze_run,
    detect_bloat,
    extract_tool_name,
    usage_totals,
)
from ctxdiff.models import Block, CallBlock
from ctxdiff.store.ctrace import Call, CTrace


def _block(text, role="user", kind="message", token_count=None, token_method="tiktoken"):
    """Build a Block whose content_hash is derived from (role, kind, text) —
    mirrors real content-addressing without pulling in the store."""
    if token_count is None:
        token_count = len(text)
    return Block(
        content_hash=f"h:{role}:{kind}:{text}", role=role, kind=kind,
        text=text, token_count=token_count, token_method=token_method,
    )


def _cb(text, position, role="user", kind="message", label="user",
        token_count=None, token_method="tiktoken"):
    """Build a CallBlock at `position` with the given label."""
    block = _block(text, role=role, kind=kind, token_count=token_count,
                    token_method=token_method)
    return CallBlock(block=block, position=position, label=label, label_source="heuristic")


def _call(seq=1, usage=None):
    return Call(id=f"call-{seq}", run_id="run", seq=seq, params={}, usage=usage,
                latency_ms=10, error=None)


# --- analyze_call: label grouping + pct math ----------------------------------


def test_analyze_call_groups_by_label_with_pct():
    """Two system blocks and one user block group into two label slices whose
    pct is each label's share of the call's total tokens, rounded to 1dp."""
    call_blocks = [
        _cb("system prompt", 0, role="system", label="system", token_count=100),
        _cb("rule two", 1, role="system", label="system", token_count=50),
        _cb("hi", 2, role="user", label="user", token_count=50),
    ]

    ct = analyze_call(_call(seq=1), call_blocks)

    assert ct.seq == 1
    assert ct.total_tokens == 200
    by_label = {s.label: s for s in ct.slices}
    assert by_label["system"].tokens == 150
    assert by_label["system"].block_count == 2
    assert by_label["system"].pct == 75.0
    assert by_label["user"].tokens == 50
    assert by_label["user"].block_count == 1
    assert by_label["user"].pct == 25.0


def test_analyze_call_pct_rounds_to_one_decimal():
    """A three-way split that doesn't divide evenly still rounds each slice's
    pct to exactly 1 decimal place."""
    call_blocks = [
        _cb("a", 0, label="user", token_count=1),
        _cb("b", 1, label="history", token_count=1),
        _cb("c", 2, label="system", token_count=1),
    ]

    ct = analyze_call(_call(), call_blocks)

    for s in ct.slices:
        assert s.pct == round(s.pct, 1)
    assert sum(s.tokens for s in ct.slices) == 3


def test_slices_sorted_by_tokens_desc():
    """Slices come back ordered biggest-spender-first, regardless of the
    input blocks' order."""
    call_blocks = [
        _cb("a", 0, role="user", label="user", token_count=10),
        _cb("b", 1, role="system", label="system", token_count=90),
        _cb("c", 2, role="tool", label="tool_output", token_count=40),
    ]

    ct = analyze_call(_call(), call_blocks)

    assert [s.label for s in ct.slices] == ["system", "tool_output", "user"]


def test_empty_call_has_zero_total_and_no_slices():
    """A call with no blocks has zero total tokens and an empty slice list —
    no division-by-zero crash on pct."""
    ct = analyze_call(_call(), [])

    assert ct.total_tokens == 0
    assert ct.slices == []
    assert ct.approximate is False


# --- approximate flag ----------------------------------------------------------


def test_approximate_false_when_all_blocks_exact():
    call_blocks = [_cb("a", 0, token_method="tiktoken"),
                   _cb("b", 1, token_method="tiktoken")]

    ct = analyze_call(_call(), call_blocks)

    assert ct.approximate is False


def test_approximate_true_when_any_block_is_estimate():
    """A single estimated block is enough to mark the whole call approximate,
    even when every other block is exact."""
    call_blocks = [_cb("a", 0, token_method="tiktoken"),
                   _cb("b", 1, token_method="estimate")]

    ct = analyze_call(_call(), call_blocks)

    assert ct.approximate is True


# --- reconciliation_delta: per-provider usage-key shape -----------------------


@pytest.mark.parametrize("usage,expected_delta", [
    ({"prompt_tokens": 120}, 20),                 # openai
    ({"input_tokens": 90}, -10),                  # anthropic
    ({"prompt_token_count": 105}, 5),              # gemini
    ({"inputTokens": 100}, 0),                     # bedrock
    (None, None),                                  # no usage at all
    ({}, None),                                    # empty usage dict
    ({"completion_tokens": 50}, None),             # no recognized prompt-side key
    ({"prompt_tokens": None, "input_tokens": 90}, -10),  # first key present but None -> fall through
])
def test_reconciliation_delta_per_provider_shape(usage, expected_delta):
    """reconciliation_delta = provider prompt-side tokens - our summed total,
    trying each provider's key in order and falling through a present-but-None
    key to the next candidate; None when nothing usable is found."""
    call_blocks = [_cb("a", 0, token_count=100)]

    ct = analyze_call(_call(usage=usage), call_blocks)

    assert ct.reconciliation_delta == expected_delta
    assert ct.provider_usage == usage


# --- extract_tool_name: defensive JSON name parsing ----------------------------


def test_extract_tool_name_top_level():
    assert extract_tool_name(json.dumps({"name": "get_weather"})) == "get_weather"


def test_extract_tool_name_nested_under_function():
    text = json.dumps({"type": "function", "function": {"name": "get_weather"}})
    assert extract_tool_name(text) == "get_weather"


def test_extract_tool_name_nested_under_toolspec():
    text = json.dumps({"toolSpec": {"name": "get_weather"}})
    assert extract_tool_name(text) == "get_weather"


def test_extract_tool_name_malformed_json_returns_sentinel():
    """Malformed JSON never raises — it returns the unparsed sentinel."""
    assert extract_tool_name("not json {{{") == "<unparsed>"


def test_extract_tool_name_unrecognized_shape_returns_sentinel():
    assert extract_tool_name(json.dumps({"description": "no name anywhere"})) == "<unparsed>"


def test_extract_tool_name_non_dict_json_returns_sentinel():
    assert extract_tool_name(json.dumps(["not", "a", "dict"])) == "<unparsed>"


# --- detect_bloat ----------------------------------------------------------------


def _schema_cb(position, name, token_count=30, shape="plain"):
    """Build a tool_schema CallBlock with the tool's name serialized in one
    of the three recognized wire shapes."""
    if shape == "plain":
        text = json.dumps({"name": name, "description": "d"})
    elif shape == "function":
        text = json.dumps({"type": "function", "function": {"name": name}})
    else:
        raise ValueError(shape)
    return _cb(text, position, role="system", kind="tool_schema",
               label="tool_schema", token_count=token_count)


def test_bloat_reports_only_the_unused_tool_with_correct_token_math():
    """One used tool (referenced by name inside a tool_output block) and one
    unused tool: only the unused one is reported, with its token count and
    pct-of-avg-context computed correctly."""
    used_schema = _schema_cb(0, "get_weather", token_count=40)
    unused_schema = _schema_cb(1, "delete_account", token_count=60)
    tool_output = _cb('{"tool": "get_weather", "result": "sunny"}', 2,
                       role="tool", label="tool_output", token_count=10)
    user_msg = _cb("hi", 3, role="user", label="user", token_count=5)
    call_blocks = [used_schema, unused_schema, tool_output, user_msg]

    report = detect_bloat([call_blocks])

    assert report is not None
    assert report.unused_tools == ["delete_account"]
    assert report.unused_tokens_per_call == 60
    assert report.calls_analyzed == 1
    total = 40 + 60 + 10 + 5
    assert report.pct_of_avg_context == round(60 / total * 100, 1)


def test_bloat_used_tool_via_assistant_role_block():
    """A tool referenced in an assistant-role block's text (rather than a
    tool_output block) is also recognized as used."""
    used_schema = _schema_cb(0, "get_weather", token_count=40)
    assistant_call = _cb("calling get_weather now", 1, role="assistant",
                          label="history", token_count=5)
    call_blocks = [used_schema, assistant_call]

    report = detect_bloat([call_blocks])

    assert report.unused_tools == []
    assert report.unused_tokens_per_call == 0


def test_bloat_schema_repeated_across_calls_counted_once_per_call_not_summed():
    """The same schema (identical text -> identical hash) resent in every
    call must contribute its token cost once, not once per call — the report
    describes the recurring per-turn tax, not a run total."""
    unused_schema_1 = _schema_cb(0, "delete_account", token_count=60)
    unused_schema_2 = _schema_cb(0, "delete_account", token_count=60)
    user_msg = _cb("hi", 1, role="user", label="user", token_count=5)

    report = detect_bloat([[unused_schema_1, user_msg], [unused_schema_2, user_msg]])

    assert report.unused_tools == ["delete_account"]
    assert report.unused_tokens_per_call == 60
    assert report.calls_analyzed == 2


def test_bloat_none_when_run_has_no_tool_schema_blocks():
    call_blocks = [_cb("hi", 0, role="user", label="user")]

    assert detect_bloat([call_blocks]) is None


def test_bloat_all_tools_used_returns_empty_unused_list():
    """When every registered tool is used, a BloatReport is still returned
    (there ARE tool_schema blocks), just with an empty unused list."""
    used_schema = _schema_cb(0, "get_weather", token_count=40)
    tool_output = _cb('{"tool": "get_weather"}', 1, role="tool",
                       label="tool_output", token_count=10)

    report = detect_bloat([[used_schema, tool_output]])

    assert report is not None
    assert report.unused_tools == []
    assert report.unused_tokens_per_call == 0
    assert report.pct_of_avg_context == 0.0


def test_bloat_malformed_schema_json_no_crash_sentinel_excluded():
    """A malformed tool_schema block never crashes detection, and is never
    reported as unused (used=unknown, not assumed wasted)."""
    bad_schema = _cb("not json {{{", 0, role="system", kind="tool_schema",
                      label="tool_schema", token_count=20)
    good_unused = _schema_cb(1, "delete_account", token_count=60)

    report = detect_bloat([[bad_schema, good_unused]])

    assert report is not None
    assert "<unparsed>" not in report.unused_tools
    assert report.unused_tools == ["delete_account"]


def test_bloat_recognizes_openai_tool_call_content_part_as_used():
    """End-to-end-ish: build blocks the way the FIXED OpenAI adapter now
    produces them for an assistant message with tool_calls — a 'content_part'
    block (role=assistant) whose text is the tool_call dict's JSON, which
    contains the function name nested under "function". A second, genuinely
    unreferenced tool's schema must still be reported as unused. This is the
    regression test for the capture bug: before the fix, tool_calls were
    dropped entirely and every OpenAI tool call would read as 'unused'."""
    used_schema = _schema_cb(0, "get_weather", token_count=40, shape="function")
    never_used_schema = _schema_cb(1, "never_used", token_count=60, shape="function")
    tool_call_text = json.dumps({
        "id": "c1", "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city":"Dubai"}'},
    }, sort_keys=True, ensure_ascii=False)
    tool_call_block = _cb(tool_call_text, 2, role="assistant", kind="content_part",
                           label="history", token_count=15)
    call_blocks = [used_schema, never_used_schema, tool_call_block]

    report = detect_bloat([call_blocks])

    assert report is not None
    assert report.unused_tools == ["never_used"]
    assert report.unused_tokens_per_call == 60


# --- analyze_run: end-to-end over a real CTrace --------------------------------


def _record(ct, seq, blocks, usage=None):
    ct.record_call(seq=seq, params={"model": "gpt-4o"}, usage=usage,
                   latency_ms=10, error=None, call_blocks=blocks)


# --- usage_totals: provider-usage rollup --------------------------------------


def _usage_call(seq, usage, agent=None):
    return Call(id=f"c-{seq}", run_id="run", seq=seq, params={}, usage=usage,
                latency_ms=10, error=None, agent=agent)


@pytest.mark.parametrize("usage,exp_in,exp_out", [
    ({"prompt_tokens": 100, "completion_tokens": 20}, 100, 20),          # openai
    ({"input_tokens": 90, "output_tokens": 15}, 90, 15),                 # anthropic
    ({"prompt_token_count": 80, "candidates_token_count": 12}, 80, 12),  # gemini
    ({"inputTokens": 70, "outputTokens": 8}, 70, 8),                     # bedrock
])
def test_usage_totals_normalizes_each_provider_shape(usage, exp_in, exp_out):
    """usage_totals sums input/output across the four provider key shapes,
    first-present wins for each side."""
    totals = usage_totals([_usage_call(1, usage)])
    assert totals.input_tokens == exp_in
    assert totals.output_tokens == exp_out
    assert totals.calls_with_usage == 1
    assert totals.calls_total == 1


def test_usage_totals_mixed_run_reports_coverage():
    """A run where only some calls carry usage: sums come only from the ones
    that reported, but calls_total counts every call (honest coverage)."""
    calls = [
        _usage_call(1, {"prompt_tokens": 100, "completion_tokens": 20}),
        _usage_call(2, None),                                  # no usage
        _usage_call(3, {"prompt_tokens": 50, "completion_tokens": 10}),
        _usage_call(4, {"completion_tokens": 5}),              # output only
    ]
    totals = usage_totals(calls)
    assert totals.input_tokens == 150
    assert totals.output_tokens == 35
    assert totals.calls_with_usage == 3   # calls 1, 3, 4 reported something
    assert totals.calls_total == 4


def test_usage_totals_no_usage_run_is_all_zero_zero_coverage():
    """A run where NO call reported usage sums to zero with zero coverage — the
    caller (CLI/dashboard) uses this to say 'no provider usage reported'."""
    totals = usage_totals([_usage_call(1, None), _usage_call(2, {})])
    assert totals.input_tokens == 0
    assert totals.output_tokens == 0
    assert totals.calls_with_usage == 0
    assert totals.calls_total == 2
    assert totals.by_agent is None


def test_usage_totals_by_agent_when_multiple_agents():
    """by_agent carries (input, output) per agent when the run spans ≥2 agents;
    single-agent runs get None."""
    calls = [
        _usage_call(1, {"prompt_tokens": 100, "completion_tokens": 20}, agent="researcher"),
        _usage_call(2, {"input_tokens": 40, "output_tokens": 6}, agent="writer"),
        _usage_call(3, {"prompt_tokens": 30, "completion_tokens": 4}, agent="researcher"),
    ]
    totals = usage_totals(calls)
    assert totals.by_agent == {"researcher": (130, 24), "writer": (40, 6)}

    single = usage_totals([_usage_call(1, {"prompt_tokens": 5}, agent="solo")])
    assert single.by_agent is None


def test_usage_totals_by_agent_buckets_unlabeled():
    """A named agent mixed with unlabeled calls counts as multi-agent; the
    unlabeled calls' usage accumulates under '(unlabeled)'."""
    calls = [
        _usage_call(1, {"prompt_tokens": 10, "completion_tokens": 2}, agent="named"),
        _usage_call(2, {"prompt_tokens": 7, "completion_tokens": 1}),  # no agent
    ]
    totals = usage_totals(calls)
    assert totals.by_agent == {"named": (10, 2), "(unlabeled)": (7, 1)}


def test_analyze_run_carries_usage_totals(tmp_path):
    """analyze_run threads the provider-usage rollup onto RunTokens.usage."""
    path = str(tmp_path / "run.ctrace")
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")
    ct.record_call(seq=1, params={"model": "m"}, usage={"prompt_tokens": 200, "completion_tokens": 30},
                   latency_ms=10, error=None, call_blocks=[_cb("hi", 0, token_count=5)])
    ct.record_call(seq=2, params={"model": "m"}, usage=None,
                   latency_ms=10, error=None, call_blocks=[_cb("bye", 0, token_count=5)])
    run_tokens = analyze_run(ct)
    ct.close()

    assert run_tokens.usage.input_tokens == 200
    assert run_tokens.usage.output_tokens == 30
    assert run_tokens.usage.calls_with_usage == 1
    assert run_tokens.usage.calls_total == 2


def _record_agent(ct, seq, blocks, agent=None, step=None):
    ct.record_call(seq=seq, params={"model": "gpt-4o"}, usage=None, latency_ms=10,
                   error=None, call_blocks=blocks, agent=agent, step=step)


def test_by_agent_totals_when_multiple_agents(tmp_path):
    """by_agent sums block tokens per agent label when the run spans ≥2
    distinct agents; each CallTokens also carries its agent/step passthrough."""
    path = str(tmp_path / "run.ctrace")
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")
    _record_agent(ct, 1, [_cb("aa", 0, token_count=10)], agent="researcher", step="plan")
    _record_agent(ct, 2, [_cb("bb", 0, token_count=30)], agent="writer")
    _record_agent(ct, 3, [_cb("cc", 0, token_count=5)], agent="researcher")

    run_tokens = analyze_run(ct)
    ct.close()

    assert run_tokens.by_agent == {"researcher": 15, "writer": 30}
    first = run_tokens.calls[0]
    assert first.agent == "researcher" and first.step == "plan"


def test_by_agent_none_for_single_agent_run(tmp_path):
    """A run with a single agent label (or all-unlabeled) has by_agent None —
    a per-agent breakdown would be trivial noise."""
    path = str(tmp_path / "run.ctrace")
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")
    _record_agent(ct, 1, [_cb("aa", 0, token_count=10)], agent="solo")
    _record_agent(ct, 2, [_cb("bb", 0, token_count=30)], agent="solo")

    run_tokens = analyze_run(ct)
    ct.close()

    assert run_tokens.by_agent is None


def test_by_agent_buckets_unlabeled_calls(tmp_path):
    """A run mixing a named agent and unlabeled calls counts as multiple
    agents; the unlabeled calls collect under '(unlabeled)'."""
    path = str(tmp_path / "run.ctrace")
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")
    _record_agent(ct, 1, [_cb("aa", 0, token_count=10)], agent="named")
    _record_agent(ct, 2, [_cb("bb", 0, token_count=7)])  # no agent

    run_tokens = analyze_run(ct)
    ct.close()

    assert run_tokens.by_agent == {"named": 10, "(unlabeled)": 7}


def test_analyze_run_agent_filter(tmp_path):
    """analyze_run(ct, agent=NAME) restricts to that agent's calls, and a
    single-agent filtered view has by_agent None."""
    path = str(tmp_path / "run.ctrace")
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")
    _record_agent(ct, 1, [_cb("aa", 0, token_count=10)], agent="researcher")
    _record_agent(ct, 2, [_cb("bb", 0, token_count=30)], agent="writer")

    run_tokens = analyze_run(ct, agent="writer")
    ct.close()

    assert [c.seq for c in run_tokens.calls] == [2]
    assert run_tokens.by_agent is None


def test_analyze_run_combines_per_call_tokens_and_bloat(tmp_path):
    path = str(tmp_path / "run.ctrace")
    ct = CTrace.create(path, project="demo", provider="openai", model="gpt-4o")

    used_schema = _schema_cb(0, "get_weather", token_count=40)
    unused_schema = _schema_cb(1, "delete_account", token_count=60)
    tool_output = _cb('{"tool": "get_weather"}', 2, role="tool",
                       label="tool_output", token_count=10)
    _record(ct, 1, [used_schema, unused_schema, tool_output],
            usage={"prompt_tokens": 120})
    _record(ct, 2, [used_schema, unused_schema, tool_output],
            usage={"prompt_tokens": 120})

    run_tokens = analyze_run(ct)
    ct.close()

    assert [c.seq for c in run_tokens.calls] == [1, 2]
    assert run_tokens.bloat is not None
    assert run_tokens.bloat.unused_tools == ["delete_account"]
    assert run_tokens.bloat.calls_analyzed == 2
    for c in run_tokens.calls:
        assert c.reconciliation_delta == 120 - c.total_tokens


# --- usage_totals: the streamed-without-include_usage diagnosis ---------------


def _stream_call(seq, params, usage=None):
    return Call(id=f"c-{seq}", run_id="run", seq=seq, params=params,
                usage=usage, latency_ms=10, error=None)


def test_usage_totals_flags_openai_streams_missing_include_usage():
    """A no-usage call whose recorded params are OpenAI-chat-shaped and
    streamed WITHOUT stream_options.include_usage is the one diagnosable,
    caller-fixable cause of missing usage — counted so the renderer can name
    the remedy instead of dead-ending at 'no provider usage reported'."""
    calls = [
        _stream_call(1, {"model": "gpt-4o", "stream": True,
                         "messages": [{"role": "user", "content": "hi"}]}),
        _stream_call(2, {"model": "gpt-4o", "stream": True,
                         "messages": [{"role": "user", "content": "yo"}]}),
    ]
    totals = usage_totals(calls)
    assert totals.calls_with_usage == 0
    assert totals.streamed_without_usage == 2


def test_usage_totals_streamed_flag_skips_calls_that_did_report_usage():
    """A streamed call that DID report usage (caller opted in, or the compat
    endpoint sent it anyway) needs no remedy and must not be counted."""
    calls = [_stream_call(
        1,
        {"model": "gpt-4o", "stream": True,
         "stream_options": {"include_usage": True},
         "messages": [{"role": "user", "content": "hi"}]},
        usage={"prompt_tokens": 5, "completion_tokens": 2},
    )]
    totals = usage_totals(calls)
    assert totals.streamed_without_usage == 0


def test_usage_totals_streamed_flag_ignores_non_streamed_and_non_chat_shapes():
    """Non-streamed calls and non-OpenAI-chat wire shapes (no `messages`
    list) can be missing usage for reasons no caller-side option fixes —
    they must not trigger the hint."""
    calls = [
        # Non-streamed, no usage: not diagnosable as the include_usage case.
        _stream_call(1, {"model": "gpt-4o",
                         "messages": [{"role": "user", "content": "hi"}]}),
        # Streamed but not chat-shaped (e.g. Gemini native contents).
        _stream_call(2, {"model": "gemini-2.5-flash", "stream": True,
                         "contents": [{"parts": [{"text": "hi"}]}]}),
    ]
    totals = usage_totals(calls)
    assert totals.streamed_without_usage == 0
