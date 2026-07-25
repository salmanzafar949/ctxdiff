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
from ctxdiff.viewer import build_payload, build_project_payload, export_html

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


def test_export_omits_model_fallback_when_models_empty(tmp_path):
    """The header must not render a dangling ' \\u00b7 ? \\u00b7 ' placeholder for
    the model segment when run.models is empty (a run whose call(s) never
    reported a model param). The old `(r.models || []).join(", ") || "?"`
    fallback — which rendered a literal "?" in that case — must be gone from
    the header-building JS embedded in the exported page; the payload's
    run.models stays an accurate empty list either way."""
    path = str(tmp_path / "r.ctrace")
    ct = CTrace.create(path, project="empty-model-run", provider="openai", model="")
    ct.record_call(seq=1, params={}, usage=None, latency_ms=1, error=None,
                   call_blocks=[_cb("hi", 0)])
    ct.close()
    text = open(export_html(path), encoding="utf-8").read()
    raw = _DATA_RE.search(text).group(1)
    payload = json.loads(raw)
    assert payload["run"]["models"] == []
    assert '(r.models || []).join(", ") || "?"' not in text


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


# --- the three-level project dashboard ------------------------------------------
#
# LEVEL 1 lists every agent across every session, LEVEL 2 the sessions one agent
# appeared in, LEVEL 3 that session's turn-by-turn detail. The `project` section
# is what the page navigates; the top level stays exactly the single-session
# payload it always was (the focus session's detail), which is what keeps every
# test above meaningful.


def _write_project(path, *, sessions=2, project="pipeline"):
    """A PROJECT trace: one file, several sessions, two agents in each, with
    fixed UTC timestamps a day apart so 'newest first' is unambiguous and the
    local-time column is a pure function of TZ. The writer sits out the LAST
    session, so an agent's session count is genuinely not the project's."""
    for i in range(sessions):
        ct = CTrace.open_or_create_session(
            path, project=project, provider="openai", model="gpt-4o",
            started_at=f"2026-07-{i + 1:02d}T09:15:00+00:00")
        ct.record_call(seq=1, params={"model": "gpt-4o"},
                       usage={"prompt_tokens": 100, "completion_tokens": 20},
                       latency_ms=10, error=None,
                       call_blocks=[_cb("sys R", 0, role="system", label="system"),
                                    _cb(f"research {i}", 1)],
                       agent="researcher", step="gather")
        ct.record_call(seq=2, params={"model": "gpt-4o"},
                       usage={"prompt_tokens": 40, "completion_tokens": 8},
                       latency_ms=10, error=None,
                       call_blocks=[_cb("sys R", 0, role="system", label="system"),
                                    _cb(f"research {i}", 1),
                                    _cb("more", 2, role="assistant", label="history")],
                       agent="researcher", step="gather")
        if i < sessions - 1:
            ct.record_call(seq=3, params={"model": "gpt-4o"}, usage=None,
                           latency_ms=10, error=None,
                           call_blocks=[_cb("sys W", 0, role="system", label="system"),
                                        _cb(f"write {i}", 1)],
                           agent="writer", step="compose")
        ct.close()
    return path


def _project_payload(path, **kwargs):
    """Open `path` and build its full three-level payload."""
    ct = CTrace.open(path)
    try:
        return build_project_payload(ct, **kwargs)
    finally:
        ct.close()


def test_project_section_sits_beside_the_untouched_session_payload(tmp_path):
    """The project payload is a strict SUPERSET of the single-session one: the
    focus session's detail is still the top level, verbatim, with `project`
    appended. That is what lets one page render both shapes and every
    single-session assertion keep its meaning."""
    path = _write_project(str(tmp_path / "p.ctrace"))
    payload = _project_payload(path)

    for key in ("run", "calls", "diffs", "tokens", "cache", "stats", "project"):
        assert key in payload
    ct = CTrace.open(path)
    try:
        assert {k: v for k, v in payload.items() if k != "project"} == build_payload(ct)
    finally:
        ct.close()


def test_level1_aggregates_each_agent_across_every_session(tmp_path):
    """LEVEL 1: one row per agent, counting the SESSIONS it ran in and the calls
    it made across all of them — the number that does not change when the user
    re-runs their agent."""
    path = _write_project(str(tmp_path / "p.ctrace"), sessions=3)
    agents = {a["name"]: a for a in _project_payload(path)["project"]["agents"]}

    assert list(agents) == ["researcher", "writer"]   # first-appearance order
    assert agents["researcher"]["sessions"] == 3
    assert agents["researcher"]["calls"] == 6         # 2 per session
    assert agents["researcher"]["input"] == 3 * 140   # 100 + 40 per session
    assert agents["researcher"]["output"] == 3 * 28
    assert agents["researcher"]["reported"] == 6
    # The writer sits out the newest session, and reported NO provider usage —
    # `reported == 0` is what makes the page print "-" instead of a false 0.
    assert agents["writer"]["sessions"] == 2
    assert agents["writer"]["calls"] == 2
    assert agents["writer"]["reported"] == 0
    # first_seen/last_seen span the sessions the agent actually appeared in.
    assert agents["researcher"]["first_seen"].startswith("2026-07-01")
    assert agents["researcher"]["last_seen"].startswith("2026-07-03")
    assert agents["writer"]["last_seen"].startswith("2026-07-02")


def test_level2_lists_every_session_newest_first_with_per_agent_turns(tmp_path):
    """LEVEL 2: every session in the project, newest first, each carrying the
    per-agent breakdown that answers 'what did THIS agent do in that run'."""
    path = _write_project(str(tmp_path / "p.ctrace"), sessions=3)
    sessions = _project_payload(path)["project"]["sessions"]

    assert [s["started_at"][:10] for s in sessions] == [
        "2026-07-03", "2026-07-02", "2026-07-01"]
    newest, middle = sessions[0], sessions[1]
    assert newest["turn_count"] == 2                    # writer sat this one out
    assert [a["name"] for a in newest["agents"]] == ["researcher"]
    assert [a["name"] for a in middle["agents"]] == ["researcher", "writer"]
    assert {a["name"]: a["turns"] for a in middle["agents"]} == {
        "researcher": 2, "writer": 1}
    assert newest["provider"] == "openai"
    assert newest["models"] == ["gpt-4o"]


def test_level2_timestamps_stay_utc_for_the_browser_to_localize(tmp_path):
    """Timestamps are embedded as the RAW stored UTC strings, never preformatted
    at export: the file is meant to be shared, so the conversion to a wall clock
    has to happen in the VIEWER's zone at render time. The page's `localTime`
    does it, and the same bytes therefore read differently in two zones —
    which is the point."""
    path = _write_project(str(tmp_path / "p.ctrace"))
    project = _project_payload(path)["project"]

    for s in project["sessions"]:
        assert s["started_at"].endswith("+00:00")
    for a in project["agents"]:
        assert a["first_seen"].endswith("+00:00")
        assert a["last_seen"].endswith("+00:00")
    # ...and the page carries the renderer that converts them client-side.
    from ctxdiff.viewer.template import _PAGE
    assert "function localTime(value)" in _PAGE
    assert "getTimezoneOffset()" in _PAGE


def test_single_agent_single_session_project_opens_straight_on_level3(tmp_path):
    """The fast path: a project with nothing to choose between at either level
    above skips the listings entirely, so the common case never clicks twice."""
    path = str(tmp_path / "solo.ctrace")
    _make_trace(path)                       # one session, no agent labels
    project = _project_payload(path)["project"]

    assert project["sessions_total"] == 1
    assert len(project["agents"]) == 1      # the (unlabeled) bucket
    assert project["start"]["level"] == 3
    assert project["start"]["session"] == project["focus"]


def test_multi_agent_project_opens_on_the_agent_listing(tmp_path):
    """More than one agent (or more than one session) means there IS a choice,
    so the dashboard lands on level 1 rather than guessing."""
    path = _write_project(str(tmp_path / "p.ctrace"))
    project = _project_payload(path)["project"]

    assert project["start"] == {"level": 1, "agent": None, "session": None}


def test_selectors_preselect_the_level_the_dashboard_opens_on(tmp_path):
    """`--agent` opens that agent's session list, `--session` opens a detail
    view, and both together open the detail scoped to the agent. An agent that
    ran in exactly ONE session skips its own one-row listing."""
    path = _write_project(str(tmp_path / "p.ctrace"), sessions=3)

    only_agent = _project_payload(path, agent="researcher")["project"]["start"]
    assert only_agent == {"level": 2, "agent": "researcher", "session": None}

    only_session = _project_payload(path, session_selected=True)["project"]["start"]
    assert only_session["level"] == 3
    assert only_session["agent"] is None

    both = _project_payload(path, agent="writer", session_selected=True)["project"]
    assert both["start"]["level"] == 3
    assert both["start"]["agent"] == "writer"

    # An agent with a single session has no listing worth showing.
    solo = _write_project(str(tmp_path / "solo.ctrace"), sessions=1)
    assert _project_payload(solo, agent="researcher")["project"]["start"]["level"] == 3


def test_detail_is_embedded_only_for_the_most_recent_sessions(tmp_path):
    """The scale decision, asserted: every session is LISTED (levels 1 and 2 are
    aggregates and are never capped), but only the `_DETAIL_SESSIONS` most recent
    ones carry the block-level detail that gives the artifact its size."""
    from ctxdiff.viewer.export import _DETAIL_SESSIONS
    total = _DETAIL_SESSIONS + 3
    path = _write_project(str(tmp_path / "big.ctrace"), sessions=total)
    project = _project_payload(path)["project"]

    assert project["sessions_total"] == total          # nothing hidden
    assert len(project["sessions"]) == total
    assert project["detail_cap"] == _DETAIL_SESSIONS
    detailed = [s for s in project["sessions"] if s["detail"]]
    assert len(detailed) == _DETAIL_SESSIONS
    # ...and they are the NEWEST ones, in order.
    assert [s["id"] for s in project["sessions"][:_DETAIL_SESSIONS]] == [
        s["id"] for s in detailed]
    # The focus session's detail is the payload's top level; the rest hang off
    # project.details, and it is never duplicated into them.
    assert project["focus"] not in project["details"]
    assert len(project["details"]) == _DETAIL_SESSIONS - 1


def test_an_explicitly_named_old_session_always_gets_its_detail(tmp_path):
    """`--session <a run older than the cap>` must land on a WORKING level 3, so
    the focus session is embedded whether or not it made the recency cut."""
    from ctxdiff.viewer.export import _DETAIL_SESSIONS
    path = _write_project(str(tmp_path / "big.ctrace"),
                          sessions=_DETAIL_SESSIONS + 2)
    ct = CTrace.open(path)
    try:
        oldest = ct.list_sessions()[0].id
        project = build_project_payload(ct, focus_session_id=oldest,
                                        session_selected=True)["project"]
    finally:
        ct.close()

    assert project["focus"] == oldest
    row = next(s for s in project["sessions"] if s["id"] == oldest)
    assert row["detail"] is True
    # ...and the page OPENS on it: embedding the detail is only half the job if
    # `start` sends the page somewhere else (see the test below).
    assert project["start"]["session"] == oldest
    # It is the top-level detail, so one MORE session than the cap is embedded.
    assert len(project["details"]) == _DETAIL_SESSIONS


def test_the_named_session_is_the_one_the_dashboard_opens_on(tmp_path):
    """`start.session` must be the FOCUS session, not whichever session happens
    to be listed first.

    `project.sessions` is newest-first, so reading its head named the newest run
    for every project with more than one session: the page boots with
    `openSession(start.session)`, which repoints the whole level-3 view — the
    breadcrumb, the header and the blocks table — at a run the user never asked
    for, while the focus session's embedded detail sits unread. Asserted for
    `--session` alone and for `--agent` + `--session`, which take different
    branches to the same level 3."""
    path = _write_project(str(tmp_path / "big.ctrace"), sessions=4)
    ct = CTrace.open(path)
    try:
        sessions = ct.list_sessions()
        oldest, newest = sessions[0].id, sessions[-1].id
        by_session = build_project_payload(
            ct, focus_session_id=oldest, session_selected=True)["project"]
        by_both = build_project_payload(
            ct, agent="researcher", focus_session_id=oldest,
            session_selected=True)["project"]
    finally:
        ct.close()

    assert oldest != newest
    for project in (by_session, by_both):
        assert project["focus"] == oldest
        assert project["start"]["level"] == 3
        assert project["start"]["session"] == oldest
        # The newest session is what the buggy head-of-list read returned.
        assert project["start"]["session"] != newest
    assert by_both["start"]["agent"] == "researcher"


def test_agent_spans_are_chronological_not_write_ordered(tmp_path):
    """`first_seen`/`last_seen` are the OLDEST and NEWEST timestamps an agent
    ran at — which is not the same as the first and last session WRITTEN once
    two capturing machines' clocks disagree (or a session is backfilled).

    Sessions written 2026-05-01, then 2026-01-01, then 2026-03-01 must report
    the span 2026-01-01 .. 2026-05-01; taking the ends of the insertion-ordered
    list reported `first seen 2026-05-01 ... last seen 2026-03-01`, a range
    running backwards that also excluded its own earliest run."""
    path = str(tmp_path / "skew.ctrace")
    for stamp in ("2026-05-01T09:00:00+00:00", "2026-01-01T09:00:00+00:00",
                  "2026-03-01T09:00:00+00:00"):
        ct = CTrace.open_or_create_session(path, project="skew",
                                           provider="openai", model="gpt-4o",
                                           started_at=stamp)
        ct.record_call(seq=1, params={"model": "gpt-4o"}, usage=None,
                       latency_ms=1, error=None, call_blocks=[_cb("hi", 0)],
                       agent="researcher")
        ct.close()

    project = _project_payload(path)["project"]
    row = project["agents"][0]
    assert row["first_seen"].startswith("2026-01-01")
    assert row["last_seen"].startswith("2026-05-01")
    assert row["first_seen"] <= row["last_seen"]
    # Session ORDERING is a separate question and deliberately unchanged: rows
    # stay in reverse INSERT order, which is what "newest first" means for a
    # store whose only reliable clock is the order it was written in.
    assert [s["started_at"][:10] for s in project["sessions"]] == [
        "2026-03-01", "2026-01-01", "2026-05-01"]


def test_a_project_named_like_the_data_marker_keeps_its_title(tmp_path):
    """The two page markers are filled in ONE pass, so a project whose NAME
    contains `__CTXDIFF_DATA__` cannot make the title swallow the payload.

    Substituting the title first and then the data with `str.replace` replaced
    EVERY occurrence of the data marker — including the one the title had just
    introduced — so `<title>` became the entire JSON payload."""
    path = str(tmp_path / "marker.ctrace")
    ct = CTrace.open_or_create_session(path, project="__CTXDIFF_DATA__",
                                       provider="openai", model="gpt-4o",
                                       started_at="2026-05-01T09:00:00+00:00")
    ct.record_call(seq=1, params={"model": "gpt-4o"}, usage=None, latency_ms=1,
                   error=None, call_blocks=[_cb("hi", 0)])
    ct.close()

    text = open(export_html(path), encoding="utf-8").read()
    assert "<title>ctxdiff — __CTXDIFF_DATA__</title>" in text
    # ...and the island is still the payload, parseable and complete.
    island = _DATA_RE.search(text).group(1)
    assert json.loads(island)["run"]["project"] == "__CTXDIFF_DATA__"
    # No marker survives unfilled anywhere in the document (the two occurrences
    # left are the project name, in the title and in the payload).
    assert text.count("__CTXDIFF_TITLE__") == 0


def test_prototype_named_agents_get_a_real_color_in_the_page(tmp_path):
    """An agent literally named `__proto__` must still get a color dot.

    The page's per-agent color map was a plain `{}`, and assigning a STRING to
    `__proto__` on one is a silent no-op — the palette entry vanished and
    `agentColor("__proto__")` returned `Object.prototype`, which the browser
    ignores as a style value. A null-prototype object takes `__proto__` as an
    ordinary own key. The same applies to the provider-usage lookup behind the
    chip tooltip, which must ask for an OWN property rather than inheriting one.
    (The behavior is asserted against the booted page in the JS suite, which has
    a DOM to run it in; here we assert the shipped page carries the fix.)"""
    from ctxdiff.viewer.template import _PAGE
    assert "const AGENT_COLOR = Object.create(null);" in _PAGE
    assert "Object.prototype.hasOwnProperty.call(uba, a.name)" in _PAGE
    assert "const AGENT_COLOR = {};" not in _PAGE


def test_a_prototype_named_block_label_gets_a_real_color_in_the_page(tmp_path):
    """A block labeled `__proto__` must fall back to the unknown color, not to a
    CSS variable that does not exist.

    `KNOWN_LABELS` is a plain object literal, so `KNOWN_LABELS[label]` INHERITS a
    truthy value for `__proto__` / `constructor` / `toString` — the label reads as
    known, `labelColor` emits `var(--c-__proto__)`, and since the page's CSS
    declares no such custom property the swatch renders with no color at all.
    Labels are user-controlled (`tracer.tag()` takes arbitrary text), so the
    lookup has to ask for an OWN property. (The behavior is asserted against the
    booted page in the JS suite, which has a DOM to run it in; here we assert the
    shipped page carries the fix — and that the fall-through it now takes names a
    variable the stylesheet actually defines.)"""
    from ctxdiff.viewer.template import _PAGE
    assert "Object.prototype.hasOwnProperty.call(KNOWN_LABELS, label)" in _PAGE
    assert "KNOWN_LABELS[label] ?" not in _PAGE
    # Every label the guard admits, plus the fall-through, is a declared CSS
    # custom property — the reason an inherited key is a visible bug.
    for label in ("system", "tool_schema", "rag", "history", "user",
                  "tool_output", "unknown"):
        assert f"--c-{label}:" in _PAGE
    assert "--c-__proto__" not in _PAGE

    # ...and a real export carrying such a block still renders (the label reaches
    # the DOM as text either way; only its swatch was at stake).
    path = str(tmp_path / "protolabel.ctrace")
    ct = CTrace.create(path, project="p", provider="openai", model="gpt-4o")
    ct.record_call(seq=1, params={"model": "gpt-4o"}, usage=None, latency_ms=1,
                   error=None,
                   call_blocks=[_cb("tagged text", 0, label="__proto__",
                                    label_source="tagged")])
    ct.close()
    island = _DATA_RE.search(open(export_html(path), encoding="utf-8").read()).group(1)
    assert json.loads(island)["calls"][0]["blocks"][0]["label"] == "__proto__"


def test_a_reader_that_cannot_list_sessions_still_exports(tmp_path):
    """A reader materialized for ONE session (an in-memory snapshot taken
    without the listing) degrades to a one-session project rather than failing
    to export — a dashboard of what it has beats no dashboard at all."""
    path = str(tmp_path / "r.ctrace")
    _make_trace(path)

    class _NoListing:
        """A reader with the analyzer surface and no session listing at all."""
        def __init__(self, ct):
            self._ct = ct
        def get_run(self, session_id=None):
            return self._ct.get_run(session_id)
        def get_calls(self, session_id=None):
            return self._ct.get_calls(session_id)
        def get_call_blocks(self, call_id):
            return self._ct.get_call_blocks(call_id)

    ct = CTrace.open(path)
    try:
        project = build_project_payload(_NoListing(ct))["project"]
    finally:
        ct.close()

    assert project["sessions_total"] == 1
    assert project["start"]["level"] == 3
    assert len(project["sessions"][0]["agents"]) == 1


def test_the_unlabeled_bucket_matches_the_cli(tmp_path):
    """The viewer must not invent a second name for calls with no agent label —
    one project reporting '(unlabeled)' in `ctxdiff agents` and something else
    in the dashboard would be a bug nobody could explain."""
    from ctxdiff.cli.select import UNLABELED
    from ctxdiff.viewer.export import _UNLABELED
    assert _UNLABELED == UNLABELED

    path = str(tmp_path / "r.ctrace")
    _make_trace(path)                       # records no agent labels at all
    agents = _project_payload(path)["project"]["agents"]
    assert [a["name"] for a in agents] == [UNLABELED]


def test_hostile_agent_and_session_labels_stay_inside_the_island(tmp_path):
    """The XSS guarantee extended to the two NEW levels. An agent name is
    attacker-influenced text that levels 1 and 2 and the breadcrumb all render,
    and a session's provider/model strings are rendered by level 2 — none of
    them may become live markup anywhere in the document."""
    path = str(tmp_path / "evil.ctrace")
    evil_agent = "</script><img src=x onerror=alert('L1')>"
    evil_provider = "</script><svg onload=alert('L2')>"
    for i, agent in enumerate([evil_agent, "plain"]):
        ct = CTrace.open_or_create_session(
            path, project="proj", provider=evil_provider,
            model="</script><script>alert('model')</script>",
            started_at=f"2026-07-0{i + 1}T00:00:00+00:00")
        ct.record_call(seq=1, params={"model": "m"}, usage=None, latency_ms=1,
                       error=None, call_blocks=[_cb("hi", 0)], agent=agent)
        ct.close()

    text = open(export_html(path), encoding="utf-8").read()
    raw = _DATA_RE.search(text).group(1)
    outside = text.replace(raw, "")
    unescaped = raw.replace("<\\/", "</")

    # Every hostile string reached the payload (levels 1/2 render them)...
    assert evil_agent in unescaped
    assert evil_provider in unescaped
    # ...as inert JSON only: no `</script` survives inside the island to close
    # it early, and no live element appears anywhere outside it.
    assert "</script" not in raw
    assert "onerror" not in outside
    assert "onload" not in outside
    assert not re.search(r"<img[^>]*onerror", outside, re.I)
    assert not re.search(r"<svg[^>]*onload", outside, re.I)
    # Exactly two real </script> closers: the data island and the page script.
    assert text.count("</script>") == 2


def test_project_dashboard_is_still_self_contained(tmp_path):
    """The self-containment guarantee holds for the multi-session artifact too:
    zero external requests, everything inline."""
    path = _write_project(str(tmp_path / "p.ctrace"), sessions=3)
    text = open(export_html(path), encoding="utf-8").read()

    assert not re.search(r"https?://", text)
    assert not re.search(r'src=["\']//', text)
    assert not re.search(r'href=["\']//', text)
    assert "//cdn" not in text
    assert text.startswith("<!DOCTYPE html>")
