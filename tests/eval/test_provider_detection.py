"""Provider detection against five REAL client constructions (spike §1). No
HTTP happens here — `_detect_provider` is synchronous and side-effect-free, so
these tests only need the client objects to exist, not to be called."""
from __future__ import annotations

import anthropic
import openai
import pytest

from ctxdiff import trace

langchain_openai = pytest.importorskip("langchain_openai")


def _make_openai():
    """A plain OpenAI client; module is `openai`, root `openai`."""
    return openai.OpenAI(api_key="x")


def _make_azure_openai():
    """An AzureOpenAI client; module is `openai.lib.azure`, root still
    `openai` — proving Azure needs no special-casing at detection time."""
    return openai.AzureOpenAI(api_key="x", azure_endpoint="https://ex.openai.azure.com",
                              api_version="2024-02-01")


def _make_anthropic():
    """A plain Anthropic client; module is `anthropic`, root `anthropic`."""
    return anthropic.Anthropic(api_key="x")


def _make_oss_openai_compatible():
    """An `openai.OpenAI` pointed at a non-OpenAI base_url (Ollama-style OSS
    endpoint). Same class/module as real OpenAI, so detection is identical —
    this is the point of the test: OSS models "just work" via the OpenAI
    adapter because the wire protocol, not the base_url, decides the adapter."""
    return openai.OpenAI(api_key="x", base_url="http://localhost:11434/v1")


def _make_chatopenai():
    """A LangChain `ChatOpenAI`; module is `langchain_openai.chat_models.base`,
    root `langchain_openai` — not in `_ADAPTERS`, so detection must raise."""
    return langchain_openai.ChatOpenAI(api_key="x", model="gpt-4o")


@pytest.mark.parametrize("make_client,expected_provider", [
    (_make_openai, "openai"),
    (_make_azure_openai, "openai"),
    (_make_anthropic, "anthropic"),
    (_make_oss_openai_compatible, "openai"),
])
def test_detect_provider_maps_to_adapter(make_client, expected_provider):
    """Each of the four non-LangChain constructions maps to the expected
    adapter name via `ctxdiff.trace._detect_provider`."""
    client = make_client()
    assert trace._detect_provider(client) == expected_provider


def test_detect_provider_raises_on_chatopenai():
    """`ChatOpenAI` lives under `langchain_openai`, not `openai`/`anthropic`,
    so `_detect_provider` must fail loudly at setup time rather than silently
    misdetecting or defaulting to some adapter at record time."""
    client = _make_chatopenai()
    with pytest.raises(ValueError, match="unrecognized client module"):
        trace._detect_provider(client)
