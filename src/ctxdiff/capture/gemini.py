"""Google Gemini adapter (google-genai SDK). Reads the request kwargs a caller
passes to `client.models.generate_content(...)` — plain dicts/strings, or the
SDK's own typed objects duck-typed via getattr — and the response object, with
no dependency on the google-genai SDK itself.

Two shapes this adapter absorbs that OpenAI/Anthropic don't have: `contents`
(no top-level `messages` key — a string, or a list of strings/dicts-with-parts/
SDK Content objects) and `config` (a dict OR a `GenerateContentConfig` object
that carries both the system instruction/tools AND sampling params like
temperature, in one bag)."""
from __future__ import annotations

import json

from ctxdiff.models import RawBlock

# Request keys that carry block *content* rather than sampling params; excluded
# from params so content is never stored twice. `config` also carries sampling
# fields (temperature etc.) — those are pulled back out in extract_params.
_CONTENT_KEYS = {"contents", "config"}

# Non-content sampling fields that may live on `config`, alongside its
# content-bearing system_instruction/tools fields.
_CONFIG_PARAM_FIELDS = ("temperature", "max_output_tokens", "top_p", "top_k")

_ROLE_MAP = {"model": "assistant"}


def _cfg_get(config: object, name: str):
    """Read `name` off `config`, which may be a plain dict (`.get`) or an SDK
    `GenerateContentConfig`-style object (`getattr`) — callers pass either.
    Missing/absent fields duck-type to None either way."""
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


class GeminiAdapter:
    """Normalize Gemini generate_content requests into the block model."""

    provider = "gemini"
    create_path = ("models", "generate_content")

    def extract_blocks(self, kwargs: dict) -> list[RawBlock]:
        """Flatten a request into ordered RawBlocks: `config.system_instruction`
        first (one 'system'-role 'message' block per instruction string — it
        sits at the very front of the sent context), then tool schemas from
        `config.tools` (one 'tool_schema' block each, JSON-serialized so their
        text is stable and diffable), then `contents`. A plain string
        `contents` becomes a single 'user' 'message' block. A list `contents`
        yields, per entry: a string → one 'user' 'message' block; a dict with
        'parts' → one 'content_part' block per part (part text if the part is
        a str or a dict carrying a 'text' key, else stable JSON), with role
        taken from the entry's 'role' ('model' → 'assistant', anything else
        passed through, defaulting to 'user' when absent)."""
        blocks: list[RawBlock] = []
        config = kwargs.get("config")

        system_instruction = _cfg_get(config, "system_instruction")
        if isinstance(system_instruction, str) and system_instruction:
            blocks.append(RawBlock(role="system", kind="message", text=system_instruction))
        elif isinstance(system_instruction, list):
            for part in system_instruction:
                text = part if isinstance(part, str) else part.get("text", "")
                blocks.append(RawBlock(role="system", kind="message", text=text))

        for tool in _cfg_get(config, "tools") or []:
            blocks.append(RawBlock(
                role="system", kind="tool_schema",
                text=json.dumps(tool, sort_keys=True, ensure_ascii=False)))

        contents = kwargs.get("contents")
        if isinstance(contents, str):
            if contents:
                blocks.append(RawBlock(role="user", kind="message", text=contents))
        elif isinstance(contents, list):
            for entry in contents:
                if isinstance(entry, str):
                    blocks.append(RawBlock(role="user", kind="message", text=entry))
                    continue
                role = entry.get("role") if isinstance(entry, dict) else None
                role = _ROLE_MAP.get(role, role or "user")
                for part in (entry.get("parts") or []) if isinstance(entry, dict) else []:
                    if isinstance(part, str):
                        text = part
                    elif isinstance(part, dict) and "text" in part:
                        text = part["text"]
                    else:
                        text = json.dumps(part, sort_keys=True, ensure_ascii=False)
                    blocks.append(RawBlock(role=role, kind="content_part", text=text))
        return blocks

    def extract_params(self, kwargs: dict) -> dict:
        """Return every request kwarg except the content-bearing 'contents'/
        'config' keys, plus any non-content sampling fields (temperature,
        max_output_tokens, top_p, top_k) duck-typed off `config` when present
        — so 'model' and sampling settings are captured without duplicating
        block content."""
        params = {k: v for k, v in kwargs.items() if k not in _CONTENT_KEYS}
        config = kwargs.get("config")
        for field in _CONFIG_PARAM_FIELDS:
            value = _cfg_get(config, field)
            if value is not None:
                params[field] = value
        return params

    def extract_usage(self, response: object) -> dict | None:
        """Pull provider-reported usage off `response.usage_metadata` (duck-
        typed) into a plain dict. Returns None when the response carries no
        usage (e.g. an error path), so downstream code can treat usage as
        optional.

        Unlike OpenAI/Anthropic, Gemini has no raw-response wrapper hop to
        fall back through (no `with_raw_response`-style resource in
        google-genai) — a plain getattr chain with a None default is
        sufficient here."""
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return None
        return {
            "prompt_token_count": getattr(usage, "prompt_token_count", None),
            "candidates_token_count": getattr(usage, "candidates_token_count", None),
            "total_token_count": getattr(usage, "total_token_count", None),
        }
