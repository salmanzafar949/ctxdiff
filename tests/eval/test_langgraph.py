"""A REAL LangGraph run, captured through the ctxdiff callback handler.

This is the test that matters most for the handler's reason to exist. Nobody
hand-writes `ChatOpenAI(callbacks=[...])` for a production agent; they build a
graph and invoke it. LangGraph PROPAGATES the callbacks it is given at invoke
time down through every node — so the handler is attached ONCE, to
`graph.invoke(..., config={"callbacks": [handler]})`, and never to the model
at all. If that propagation ever stopped working (or ctxdiff's handler stopped
being a real `BaseCallbackHandler` and got rejected by LangChain's pydantic
validation), this file fails and the framework people actually use silently
stops being covered.

It also covers the full tool-calling round trip — tool schemas, an assistant
message whose content is a tool CALL, and a tool RESULT fed back in — checked
against the exact JSON body LangChain put on the wire, since the second turn's
context is precisely where a normalization bug would hide.
"""
from __future__ import annotations

import json
from typing import Annotated, TypedDict

import httpx
import pytest

from ctxdiff import trace
from ctxdiff.capture.recorder import build_block
from ctxdiff.store.ctrace import CTrace

langchain_openai = pytest.importorskip("langchain_openai")
langgraph = pytest.importorskip("langgraph")

from langchain_core.tools import tool  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402

_OPENAI_URL = "https://api.openai.com/v1/chat/completions"

_TOOL_CALL_RESPONSE = {
    "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "gpt-4o",
    "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "call_1", "type": "function", "function": {
            "name": "get_weather", "arguments": '{"city": "Dubai"}'}}]}}],
    "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
}

_FINAL_RESPONSE = {
    "id": "chatcmpl-2", "object": "chat.completion", "created": 1, "model": "gpt-4o",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {
        "role": "assistant", "content": "It is sunny in Dubai."}}],
    "usage": {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35},
}


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return "sunny"


class _State(TypedDict):
    """The graph's state: the running message list, reduced by LangGraph's
    own `add_messages` so each node appends rather than replaces."""
    messages: Annotated[list, add_messages]


def _build_graph(llm):
    """A minimal but genuine ReAct graph: a model node, a tool node, and a
    conditional edge that loops back once the model asks for a tool. Built by
    hand rather than with a prebuilt helper so the test depends only on
    LangGraph's stable core (`StateGraph`/`ToolNode`) — and so the callback
    propagation being exercised is visibly the graph's, not a helper's."""
    bound = llm.bind_tools([get_weather])

    def call_model(state: _State) -> dict:
        return {"messages": [bound.invoke(state["messages"])]}

    def route(state: _State) -> str:
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    graph = StateGraph(_State)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode([get_weather]))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")
    return graph.compile()


def _openai_hashes(wire: dict) -> list[str]:
    """The content hashes a DIRECT capture of `wire` would store — the real
    OpenAI adapter's blocks through the real `build_block`."""
    from ctxdiff.capture.openai import OpenAIAdapter

    return [build_block(raw, "openai").content_hash
            for raw in OpenAIAdapter().extract_blocks(wire)]


def test_langgraph_run_is_captured_turn_by_turn(respx_mock, tmp_ctrace_path):
    """The whole point: a handler attached ONLY to `graph.invoke(...,
    config={"callbacks": [...]})` — never to the model — captures every LLM
    turn the graph makes, in order, each with its own usage.

    Two turns here: the model asks for a tool, the tool runs, the model
    answers. Turn 2's context is a superset of turn 1's, which is exactly the
    growth `ctxdiff diff` exists to show."""
    sent: list[dict] = []

    def _respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=_TOOL_CALL_RESPONSE if len(sent) == 1
                              else _FINAL_RESPONSE)

    respx_mock.post(_OPENAI_URL).mock(side_effect=_respond)

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    graph = _build_graph(langchain_openai.ChatOpenAI(model="gpt-4o", api_key="x"))

    result = graph.invoke({"messages": [("user", "weather in Dubai?")]},
                          config={"callbacks": [tracer.langchain_handler(agent="weatherbot")]})
    assert result["messages"][-1].content == "It is sunny in Dubai."
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    assert len(calls) == 2                              # callbacks reached both nodes
    assert [c.seq for c in calls] == [1, 2]
    assert all(c.agent == "weatherbot" for c in calls)
    assert calls[0].usage == {"prompt_tokens": 20, "completion_tokens": 8,
                              "total_tokens": 28}
    assert calls[1].usage == {"prompt_tokens": 30, "completion_tokens": 5,
                              "total_tokens": 35}

    first = ct.get_call_blocks(calls[0].id)
    assert [(b.block.role, b.block.kind) for b in first] == [
        ("system", "tool_schema"), ("user", "message")]
    assert "get_weather" in first[0].block.text
    assert first[1].block.text == "weather in Dubai?"

    second = ct.get_call_blocks(calls[1].id)
    # The tool schema and the original question are the SAME blocks as turn 1
    # (same hashes) — a stable cache prefix, not a rewritten context.
    assert [b.block.content_hash for b in second[:2]] \
        == [b.block.content_hash for b in first]
    # ...followed by the assistant's tool call and the tool's result.
    assert [(b.block.role, b.block.kind) for b in second[2:]] == [
        ("assistant", "content_part"), ("tool", "message")]
    assert json.loads(second[2].block.text)["function"]["name"] == "get_weather"
    assert second[3].block.text == "sunny"
    ct.close()


def test_langgraph_tool_turn_hashes_match_the_wire_body(respx_mock, tmp_ctrace_path):
    """Hash identity for the hard turn. The second request — tool schemas, an
    assistant message carrying a tool call, a tool result — is compared block
    for block against the ACTUAL JSON body LangChain sent, run through the
    OpenAI adapter. This is what pins the tool-call normalization — the
    NORMALIZED `.tool_calls` rebuilt with the same `json.dumps` LangChain
    itself uses, with `additional_kwargs["tool_calls"]` kept only as the
    FALLBACK for an integration that carries no normalized form — to the wire
    rather than to ctxdiff's own idea of it. (The other order looks more
    faithful and is not: the raw form holds the PROVIDER's original JSON
    text, whose whitespace LangChain does not preserve when it re-serializes,
    so the block would differ from what actually went out.)

    It is also where the LIMIT of cross-SDK identity lives: `json.dumps`
    emits `{"city": "Dubai"}` where `JSON.stringify` emits
    `{"city":"Dubai"}`, so a tool-call block hashes differently in the two
    SDKs — each faithful to its own framework's wire. That divergence is
    pinned deliberately in
    `test_langchain_handler.test_cross_sdk_tool_call_hashes_are_pinned_as_known_divergent`."""
    sent: list[dict] = []

    def _respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=_TOOL_CALL_RESPONSE if len(sent) == 1
                              else _FINAL_RESPONSE)

    respx_mock.post(_OPENAI_URL).mock(side_effect=_respond)

    tracer = trace.init(project="p", path=tmp_ctrace_path)
    graph = _build_graph(langchain_openai.ChatOpenAI(model="gpt-4o", api_key="x"))
    graph.invoke({"messages": [("user", "weather in Dubai?")]},
                 config={"callbacks": [tracer.langchain_handler()]})
    tracer.close()

    ct = CTrace.open(tmp_ctrace_path)
    calls = ct.get_calls()
    for call, wire in zip(calls, sent):
        assert [b.block.content_hash for b in ct.get_call_blocks(call.id)] \
            == _openai_hashes(wire)
    ct.close()
