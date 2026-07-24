"""Tests for the self-contained HTML dashboard viewer (M5). Covers the pure
payload builder (build_payload) and the HTML exporter (export_html), including
the two security guarantees the exporter must hold: no external requests
(nothing in the file may look like an http(s) URL) and no </script> breakout
(embedded trace data — even data that literally contains </script> or raw HTML
— must stay inside the JSON island and never become live markup)."""
import json
import os
import re

from ctxdiff.models import Block, CallBlock
from ctxdiff.store.ctrace import CTrace
from ctxdiff.viewer import build_payload, export_html

# Regex that pulls the embedded JSON island back out of the exported HTML,
# stopping at the FIRST literal </script> — which only works because the
# exporter escapes </ inside the payload, so a </script> in block text can
# never masquerade as the real closing tag.
_DATA_RE = re.compile(
    r'<script id="ctxdiff-data" type="application/json">(.*?)</script>',
    re.DOTALL,
)


def _cb(text, position, role="user", kind="message", label="user",
        label_source="heuristic", token_count=None, token_method="tiktoken"):
    """Build a CallBlock with a stable content hash derived from its fields —
    mirrors real content-addressing without pulling in a tokenizer. The
    token_method is folded into the hash so the same visible text can be stored
    once exact and once estimated without colliding."""
    if token_count is None:
        token_count = max(1, len(text) // 4)
    block = Block(content_hash=f"h:{role}:{kind}:{text}:{token_method}",
                  role=role, kind=kind, text=text,
                  token_count=token_count, token_method=token_method)
    return CallBlock(block=block, position=position, label=label,
                     label_source=label_source)


TOOL_SCHEMA = json.dumps({"type": "function", "function": {"name": "search_docs"}})
UNUSED_SCHEMA = json.dumps({"type": "function", "function": {"name": "delete_everything"}})


def _make_trace(path):
    """Build a 3-turn trace exercising every payload section: a stable system
    block whose embedded timestamp changes each turn (a cache break), a used +
    an unused tool schema (bloat), a tagged rag block, growing history, and one
    estimate-method block per turn (approximate token totals). Params carry
    sensitive keys (temperature, api_key) so the model-only guard is testable."""
    ct = CTrace.create(path, project="rag-support-agent", provider="anthropic",
                       model="claude-sonnet-4-5")
    for seq, ts in enumerate(["09:00:00", "09:00:07", "09:00:15"], start=1):
        blocks = [
            _cb(f"You are a support agent. Current time: {ts}.", 0,
                role="system", label="system", token_count=25),
            _cb(TOOL_SCHEMA, 0, role="system", kind="tool_schema",
                label="tool_schema", token_count=30),
            _cb(UNUSED_SCHEMA, 0, role="system", kind="tool_schema",
                label="tool_schema", token_count=35),
            _cb(f"Retrieved billing doc chunk #{seq}", 0, role="user",
                label="rag", label_source="tagged", token_count=40),
            _cb("call search_docs", 0, role="assistant", label="history",
                token_count=8),
            _cb(f"User question number {seq}", 0, role="user", label="user",
                token_count=15, token_method="estimate"),
        ]
        for h in range(1, seq):
            blocks.append(_cb(f"assistant answer {h}", 0, role="assistant",
                              label="history", token_count=12))
        # Reindex positions to the final send-order (0..n-1).
        blocks = [CallBlock(block=b.block, position=i, label=b.label,
                            label_source=b.label_source)
                  for i, b in enumerate(blocks)]
        ct.record_call(seq=seq,
                       params={"model": "claude-sonnet-4-5", "temperature": 0.7,
                               "api_key": "sk-secret"},
                       usage={"input_tokens": 200}, latency_ms=120, error=None,
                       call_blocks=blocks)
    ct.close()


def _payload_for(path):
    """Open the trace at `path`, build its payload, and close — the common
    setup every build_payload test needs."""
    _make_trace(path)
    ct = CTrace.open(path)
    try:
        return build_payload(ct)
    finally:
        ct.close()


# --- build_payload ------------------------------------------------------------


def test_payload_has_all_top_level_sections(tmp_path):
    """The payload carries every section the page renders from."""
    payload = _payload_for(str(tmp_path / "r.ctrace"))
    for key in ("run", "calls", "diffs", "tokens", "cache", "stats"):
        assert key in payload


def test_run_section_fields(tmp_path):
    """The run section carries the header's identity fields."""
    payload = _payload_for(str(tmp_path / "r.ctrace"))
    run = payload["run"]
    assert run["project"] == "rag-support-agent"
    assert run["provider"] == "anthropic"
    for key in ("started_at", "ctxdiff_version", "models"):
        assert key in run


def test_blocks_carry_text_and_short_hash(tmp_path):
    """Each serialized block carries its full text plus an 8-char hash prefix."""
    payload = _payload_for(str(tmp_path / "r.ctrace"))
    block = payload["calls"][0]["blocks"][0]
    assert isinstance(block["text"], str) and block["text"]
    assert len(block["content_hash_short"]) == 8
    for key in ("position", "label", "label_source", "role", "kind",
                "token_count", "token_method"):
        assert key in block


def test_diffs_length_is_turns_minus_one(tmp_path):
    """One diff per adjacent turn pair: N turns -> N-1 diffs."""
    payload = _payload_for(str(tmp_path / "r.ctrace"))
    assert len(payload["diffs"]) == len(payload["calls"]) - 1


def test_params_contain_only_model(tmp_path):
    """The sensitive-params guard: params expose the model and NOTHING else —
    no temperature, no api_key — since full params may carry secrets."""
    payload = _payload_for(str(tmp_path / "r.ctrace"))
    for call in payload["calls"]:
        assert set(call["params"].keys()) == {"model"}


def test_tokens_and_cache_sections_populated(tmp_path):
    """The embedded analyzer output is present: per-call token slices, a bloat
    report naming the unused tool, and a cache report with the timestamp break."""
    payload = _payload_for(str(tmp_path / "r.ctrace"))
    assert payload["tokens"]["calls"][0]["slices"]
    assert "delete_everything" in payload["tokens"]["bloat"]["unused_tools"]
    assert payload["cache"]["breaks"], "the timestamp change should break the prefix"
    assert payload["cache"]["fix_hint"]


def test_stats_dedup_and_growth(tmp_path):
    """Stats carry the dedup counts and per-turn context growth."""
    payload = _payload_for(str(tmp_path / "r.ctrace"))
    stats = payload["stats"]
    assert stats["distinct_blocks"] < stats["total_block_refs"]  # dedup happened
    assert len(stats["context_growth"]) == len(payload["calls"])


def test_payload_roundtrips_json(tmp_path):
    """The whole payload survives a json.dumps/loads round-trip unchanged —
    i.e. it is fully JSON-serializable with no exotic values."""
    payload = _payload_for(str(tmp_path / "r.ctrace"))
    assert json.loads(json.dumps(payload)) == payload


# --- export_html --------------------------------------------------------------


def test_export_writes_file_next_to_trace(tmp_path):
    """With no out_path, the HTML is written as <stem>.html beside the trace."""
    path = str(tmp_path / "r.ctrace")
    _make_trace(path)
    out = export_html(path)
    assert out == str(tmp_path / "r.html")
    assert os.path.exists(out)


def test_export_contains_project_name(tmp_path):
    """The exported file mentions the project name (title + embedded data)."""
    path = str(tmp_path / "r.ctrace")
    _make_trace(path)
    text = open(export_html(path), encoding="utf-8").read()
    assert "rag-support-agent" in text


def test_embedded_json_parses_back(tmp_path):
    """The JSON island extracts and parses back to the same payload shape."""
    path = str(tmp_path / "r.ctrace")
    _make_trace(path)
    text = open(export_html(path), encoding="utf-8").read()
    raw = _DATA_RE.search(text).group(1)
    payload = json.loads(raw)
    assert payload["run"]["project"] == "rag-support-agent"
    assert len(payload["calls"]) == 3


def test_self_contained_no_external_requests(tmp_path):
    """The no-external-requests guarantee, asserted literally: the file must
    contain no http:// or https:// substring anywhere — no CDN, font, image,
    or even an SVG xmlns URL."""
    path = str(tmp_path / "r.ctrace")
    _make_trace(path)
    text = open(export_html(path), encoding="utf-8").read()
    assert "http://" not in text
    assert "https://" not in text


def test_script_breakout_impossible(tmp_path):
    """A block whose text literally contains </script> (and a full breakout
    payload) must survive the round-trip intact: the exporter escapes </ so the
    data can never terminate the script tag early. If it leaked, either the
    JSON island would be truncated (parse error) or the extracted text would
    differ from what went in."""
    path = str(tmp_path / "r.ctrace")
    ct = CTrace.create(path, project="x", provider="openai", model="gpt-4o")
    evil = 'oops </script><script>alert(1)</script> done'
    ct.record_call(seq=1, params={"model": "gpt-4o"}, usage=None, latency_ms=1,
                   error=None, call_blocks=[_cb(evil, 0)])
    ct.record_call(seq=2, params={"model": "gpt-4o"}, usage=None, latency_ms=1,
                   error=None, call_blocks=[_cb(evil + " more", 0)])
    ct.close()
    text = open(export_html(path), encoding="utf-8").read()
    raw = _DATA_RE.search(text).group(1)
    payload = json.loads(raw)
    assert payload["calls"][0]["blocks"][0]["text"] == evil


def test_html_injection_stays_inside_json_island(tmp_path):
    """Raw HTML in block text appears in the embedded JSON but NEVER as live
    markup outside the script tag — the page renders block text via textContent
    at run time, so nothing is escaped/injected into the document at build time."""
    path = str(tmp_path / "r.ctrace")
    ct = CTrace.create(path, project="x", provider="openai", model="gpt-4o")
    xss = "<img src=x onerror=alert(1)>"
    ct.record_call(seq=1, params={"model": "gpt-4o"}, usage=None, latency_ms=1,
                   error=None, call_blocks=[_cb(xss, 0)])
    ct.close()
    text = open(export_html(path), encoding="utf-8").read()
    raw = _DATA_RE.search(text).group(1)
    assert xss in raw                      # present inside the JSON island
    assert xss not in text.replace(raw, "")  # absent everywhere else


# --- multi-agent payload ------------------------------------------------------


def _make_multi_agent_trace(path):
    """Two agents interleaved on the global timeline: researcher (turns 1,3)
    and writer (turn 2). Turn 2->3 and 1->2 are cross-agent hand-offs."""
    ct = CTrace.create(path, project="multi", provider="openai", model="gpt-4o")

    def rec(seq, blocks, agent, step=None):
        ct.record_call(seq=seq, params={"model": "gpt-4o"}, usage=None,
                       latency_ms=10, error=None, call_blocks=blocks,
                       agent=agent, step=step)
    r1 = [_cb("sys R", 0, role="system", label="system", token_count=5),
          _cb("r-q1", 1, token_count=3)]
    w1 = [_cb("sys W", 0, role="system", label="system", token_count=5),
          _cb("w-q1", 1, token_count=3)]
    r2 = r1 + [_cb("r-ans", 2, role="assistant", label="history", token_count=4),
               _cb("r-q2", 3, token_count=3)]
    rec(1, r1, "researcher", "plan")
    rec(2, w1, "writer")
    rec(3, r2, "researcher", "answer")
    ct.close()


def test_payload_carries_agent_stats_and_fields(tmp_path):
    """stats.agents lists each agent with call count and tokens; calls carry
    agent/step passthrough."""
    path = str(tmp_path / "multi.ctrace")
    _make_multi_agent_trace(path)
    ct = CTrace.open(path)
    try:
        payload = build_payload(ct)
    finally:
        ct.close()

    agents = {a["name"]: a for a in payload["stats"]["agents"]}
    assert agents["researcher"]["calls"] == 2
    assert agents["writer"]["calls"] == 1
    assert agents["researcher"]["tokens"] > 0
    assert payload["calls"][0]["agent"] == "researcher"
    assert payload["calls"][0]["step"] == "plan"
    # per-agent block tokens live on stats.agents (the template's source); the
    # payload carries no redundant tokens.by_agent field.
    assert "by_agent" not in payload["tokens"]


def test_payload_carries_usage_rollup(tmp_path):
    """stats.usage carries the provider-usage rollup: summed input/output,
    coverage [reported, total], and per-agent [in, out] on a multi-agent run.
    The _make_trace fixture in test_viewer records input_tokens per call, so
    the totals are non-zero and full-coverage."""
    path = str(tmp_path / "r.ctrace")
    payload = _payload_for(path)   # single-agent fixture, input_tokens=200 x3
    usage = payload["stats"]["usage"]
    assert usage["input"] == 600           # 200 * 3 calls
    assert usage["coverage"] == [3, 3]     # every call reported usage
    assert usage["by_agent"] is None       # single-agent run


def test_payload_usage_by_agent_multi_agent(tmp_path):
    """A multi-agent run with provider usage carries per-agent [in, out]."""
    path = str(tmp_path / "multi.ctrace")
    ct = CTrace.create(path, project="multi", provider="openai", model="gpt-4o")
    ct.record_call(seq=1, params={"model": "m"},
                   usage={"prompt_tokens": 100, "completion_tokens": 20},
                   latency_ms=10, error=None, call_blocks=[_cb("a", 0)],
                   agent="researcher")
    ct.record_call(seq=2, params={"model": "m"},
                   usage={"input_tokens": 40, "output_tokens": 6},
                   latency_ms=10, error=None, call_blocks=[_cb("b", 0)],
                   agent="writer")
    ct.close()
    ct = CTrace.open(path)
    try:
        payload = build_payload(ct)
    finally:
        ct.close()
    assert payload["stats"]["usage"]["by_agent"] == {
        "researcher": [100, 20], "writer": [40, 6]}


def test_payload_marks_cross_agent_diffs(tmp_path):
    """Each adjacent-pair diff carries cross_agent; a hand-off pair that has a
    same-agent predecessor also carries a precomputed same_agent_diff."""
    path = str(tmp_path / "multi.ctrace")
    _make_multi_agent_trace(path)
    ct = CTrace.open(path)
    try:
        payload = build_payload(ct)
    finally:
        ct.close()

    diffs = payload["diffs"]  # index 0 = turns 1->2, index 1 = turns 2->3
    assert diffs[0]["cross_agent"] is True   # researcher -> writer
    assert diffs[1]["cross_agent"] is True   # writer -> researcher
    # turn 3 (researcher) has an earlier researcher turn (1) to diff against
    assert "same_agent_diff" in diffs[1]
    assert diffs[1]["same_agent_diff"]["seq_old"] == 1
    assert diffs[1]["same_agent_diff"]["seq_new"] == 3


def test_hostile_agent_name_stays_inside_json_island(tmp_path):
    """An agent name with hostile markup survives to the payload but appears
    ONLY inside the JSON island — the page renders agent names via textContent,
    so a hostile name can never become live markup in the document."""
    path = str(tmp_path / "r.ctrace")
    ct = CTrace.create(path, project="x", provider="openai", model="gpt-4o")
    evil = "</script><img onerror=alert(1)>"
    ct.record_call(seq=1, params={"model": "gpt-4o"}, usage=None, latency_ms=1,
                   error=None, call_blocks=[_cb("hi", 0)], agent=evil)
    ct.record_call(seq=2, params={"model": "gpt-4o"}, usage=None, latency_ms=1,
                   error=None, call_blocks=[_cb("bye", 0)], agent="other")
    ct.close()
    text = open(export_html(path), encoding="utf-8").read()
    raw = _DATA_RE.search(text).group(1)
    # The </script> is escaped as <\/script> inside the island; compare on the
    # unescaped form the browser would parse back.
    assert evil in raw.replace("<\\/", "</")
    assert evil not in text.replace(raw, "")  # nowhere in the static document
