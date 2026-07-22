from ctxdiff.models import (
    RawBlock, Block, CallBlock,
    normalize_text, content_hash, basic_label,
)

def test_normalize_text_passes_strings_through():
    """Plain strings are hashed verbatim — capture must preserve wire truth."""
    assert normalize_text("hello") == "hello"

def test_normalize_text_serializes_non_strings_stably():
    """Multi-part content (dict/list) is serialized with sorted keys so equal
    content always yields an equal string (and therefore an equal hash)."""
    a = normalize_text({"type": "text", "text": "hi"})
    b = normalize_text({"text": "hi", "type": "text"})
    assert a == b

def test_content_hash_is_deterministic_and_role_sensitive():
    """Identity = sha256(role + kind + normalized_text): same inputs → same hash,
    and a role change → a different hash (a user 'hi' ≠ an assistant 'hi')."""
    h1 = content_hash("user", "message", "hi")
    h2 = content_hash("user", "message", "hi")
    h3 = content_hash("assistant", "message", "hi")
    assert h1 == h2 and h1 != h3
    assert len(h1) == 64  # hex sha256

def test_basic_label_maps_roles_to_labels():
    """Untagged blocks get a coarse role-based label with source 'heuristic'."""
    assert basic_label("system", "message", "x", []) == ("system", "heuristic")
    assert basic_label("tool", "message", "x", []) == ("tool_output", "heuristic")
    assert basic_label("user", "message", "x", []) == ("user", "heuristic")
    assert basic_label("assistant", "message", "x", []) == ("history", "heuristic")
    assert basic_label("system", "tool_schema", "x", []) == ("tool_schema", "heuristic")

def test_basic_label_tag_override_wins_by_substring():
    """A tagged text that is a substring of the block marks it with the tagged
    label and source 'tagged' — tags override the role heuristic."""
    tagged = [("rag", "Enterprise pricing FAQ")]
    label, source = basic_label(
        "user", "message", "Context: Enterprise pricing FAQ — 2026 edition", tagged
    )
    assert (label, source) == ("rag", "tagged")
