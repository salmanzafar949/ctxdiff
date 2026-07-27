"""OpenAI-compatible endpoint attribution (dogfood finding 2026-07-27).

An OpenAI-SDK client pointed at another vendor's OpenAI-compatible endpoint
(Gemini's `generativelanguage.googleapis.com`, Anthropic's
`api.anthropic.com`, an Ollama/vLLM box, ...) used to record
`provider=openai` for every call — confidently-wrong metadata on the run
header and each call row. The ADAPTER must stay `openai` (the wire shape is
OpenAI's; capture mechanics key off the adapter), but the recorded provider
LABEL is now refined from the client's `base_url` host: named vendors get
their name, OpenAI/Azure keep `openai`, and unrecognized hosts are labeled
`openai-compatible` rather than guessed.
"""
import pytest

from ctxdiff import trace
from ctxdiff.store.ctrace import CTrace
from ctxdiff.trace import _openai_compat_label


class _Usage:
    prompt_tokens = 3; completion_tokens = 1; total_tokens = 4
class _Resp:
    usage = _Usage()


class _FakeCompletions:
    def create(self, **kwargs):
        return _Resp()


class _FakeChat:
    def __init__(self): self.completions = _FakeCompletions()


class _FakeOpenAI:
    """Duck-typed OpenAI client: module 'openai' (what _detect_provider keys
    off) plus the `base_url` attribute the real SDK exposes."""
    __module__ = "openai"

    def __init__(self, base_url=None):
        self.chat = _FakeChat()
        self.base_url = base_url


# --- the label function itself -----------------------------------------------

@pytest.mark.parametrize("base_url,expected", [
    # No base_url at all (the default client) → the real OpenAI.
    (None, "openai"),
    # OpenAI's own hosts stay openai.
    ("https://api.openai.com/v1", "openai"),
    # Azure OpenAI has ALWAYS recorded 'openai'; relabeling would churn
    # existing users' traces, so it is exempted explicitly.
    ("https://myco.openai.azure.com/openai", "openai"),
    # Named vendors' compat endpoints get the vendor's name.
    ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini"),
    ("https://api.anthropic.com/v1/", "anthropic"),
    # Anything else is compat traffic to a host we cannot name — truthful
    # without guessing (Ollama/vLLM/OpenRouter/LiteLLM proxies land here).
    ("http://localhost:11434/v1", "openai-compatible"),
    ("https://openrouter.ai/api/v1", "openai-compatible"),
])
def test_label_maps_base_url_hosts(base_url, expected):
    assert _openai_compat_label(_FakeOpenAI(base_url)) == expected


def test_label_is_fail_open_on_unreadable_base_url():
    """A base_url that explodes on inspection must yield the unrefined
    'openai', never an exception — labeling can't be allowed to break wrap()."""
    class _Bomb:
        @property
        def host(self):
            raise RuntimeError("boom")

        def __str__(self):
            raise RuntimeError("boom")

    assert _openai_compat_label(_FakeOpenAI(_Bomb())) == "openai"


# --- end-to-end: the label lands on the run AND each call --------------------

def test_compat_client_records_vendor_label_on_run_and_calls(tmp_path):
    """Wrap a fake OpenAI client pointed at Gemini's compat endpoint, make a
    call, and the persisted trace must attribute BOTH the session and the call
    to gemini — while capture still went through the openai adapter (the call
    recorded fine, params intact)."""
    path = str(tmp_path / "proj.ctrace")
    t = trace.init("proj", path=path)
    client = t.wrap(_FakeOpenAI(
        "https://generativelanguage.googleapis.com/v1beta/openai/"))
    client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "hi"}])
    t.close()

    ct = CTrace.open(path)
    try:
        assert ct.get_run().provider == "gemini"
        calls = ct.get_calls()
        assert len(calls) == 1
        assert calls[0].provider == "gemini"
        assert calls[0].params["model"] == "gemini-2.5-flash"
    finally:
        ct.close()


def test_plain_openai_client_still_records_openai(tmp_path):
    """The default client (no base_url) keeps the historical label."""
    path = str(tmp_path / "proj.ctrace")
    t = trace.init("proj", path=path)
    client = t.wrap(_FakeOpenAI())
    client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    t.close()

    ct = CTrace.open(path)
    try:
        assert ct.get_run().provider == "openai"
        assert ct.get_calls()[0].provider == "openai"
    finally:
        ct.close()
