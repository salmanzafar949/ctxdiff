"""Core value types plus the pure functions that give a block its identity and
its label. Kept dependency-free so every other module can import it freely."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

# --- value types -----------------------------------------------------------

@dataclass(frozen=True)
class RawBlock:
    """One context unit as an adapter extracts it from a request payload,
    before token counting or hashing. `kind` is 'message' | 'content_part' |
    'tool_schema' | 'image'; `role` is 'system' | 'user' | 'assistant' |
    'tool'.

    The three optional fields exist for blocks whose stored text is a STAND-IN
    rather than the content itself — today that means image blocks (see
    `ctxdiff.images`), whose `text` is a descriptor like
    `[image 1024×768 · ~765 tok]`. For those, hashing the descriptor would make
    two different images collide whenever they happened to share a size, and
    tokenizing it would measure the descriptor instead of the picture. So an
    adapter may override both:

      * `hash_input` — what to hash INSTEAD of `text` for identity (for an
        image, a digest of the image bytes, so the same picture is one block
        however it was wrapped and however many turns it survives);
      * `token_count` / `token_method` — a pre-computed count INSTEAD of
        running the tokenizer over `text` (for an image, the provider's
        documented vision-token formula, always marked `'estimate'`).

    All three default to None, which means "behave exactly as before": hash the
    text, tokenize the text. Every pre-existing adapter call site is therefore
    unchanged."""
    role: str
    kind: str
    text: str
    hash_input: str | None = None
    token_count: int | None = None
    token_method: str | None = None


@dataclass(frozen=True)
class Block:
    """A stored, content-addressed context unit. Identity is `content_hash`,
    so equal text (same role+kind) is stored once and referenced many times."""
    content_hash: str
    role: str
    kind: str
    text: str
    token_count: int
    token_method: str  # 'tiktoken' | 'estimate'  ('estimate' for every image)


@dataclass(frozen=True)
class CallBlock:
    """Membership of a `Block` in one call: where it sat (`position`) and how it
    was labeled. Label lives here, not on Block, because the same text can be
    labeled differently depending on tagging — identity must not depend on it."""
    block: Block
    position: int
    label: str
    label_source: str  # 'heuristic' | 'tagged'


# --- pure functions --------------------------------------------------------

def normalize_text(text: object) -> str:
    """Return a canonical string for hashing. Strings pass through verbatim to
    preserve wire truth; anything else (multi-part content dicts/lists) is
    JSON-serialized with sorted keys so semantically-equal content always maps
    to the same string, and therefore the same hash."""
    if isinstance(text, str):
        return text
    return json.dumps(text, sort_keys=True, ensure_ascii=False)


def content_hash(role: str, kind: str, text: str) -> str:
    """Compute a block's identity as sha256 over role, kind, and normalized
    text, joined by a NUL separator that cannot appear in the fields — so
    ('a','bc') and ('ab','c') never collide. Returns a 64-char hex digest."""
    joined = "\x00".join((role, kind, normalize_text(text)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


# Role → coarse label. The richer heuristic (e.g. detecting large non-recent
# user blocks as RAG) is an analyzer added in a later milestone; M1 stores this
# honest, cheap mapping so `call_block.label` is never null.
_ROLE_LABEL = {
    "system": "system",
    "tool": "tool_output",
    "user": "user",
    "assistant": "history",
}


def basic_label(
    role: str, kind: str, text: str, tagged: list[tuple[str, str]]
) -> tuple[str, str]:
    """Decide a block's (label, source). A developer tag wins first: if any
    tagged text is a substring of this block, return that tag's label with
    source 'tagged' (first tag registered wins). Otherwise a tool schema is
    'tool_schema' and everything else maps by role. Falls back to the raw role
    string for unknown roles so no input can crash labeling."""
    for label, needle in tagged:
        if needle and needle in text:
            return (label, "tagged")
    if kind == "tool_schema":
        return ("tool_schema", "heuristic")
    return (_ROLE_LABEL.get(role, role), "heuristic")
