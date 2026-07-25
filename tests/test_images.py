"""Tests for the image block representation — `ctxdiff.images` plus the four
adapters that route content parts through it.

The thing under test, stated once: an image content part must NOT reach the
store as its base64 payload. It must become a block whose text is a short
descriptor, whose identity is a digest of the image BYTES, and whose token
count is the provider's published vision formula marked as an estimate. Every
test below is one clause of that sentence, or one way it is allowed to degrade.
"""
from __future__ import annotations

import base64
import hashlib
import struct
import zlib

import pytest

from ctxdiff.capture.anthropic import AnthropicAdapter
from ctxdiff.capture.bedrock import BedrockAdapter
from ctxdiff.capture.gemini import GeminiAdapter
from ctxdiff.capture.openai import OpenAIAdapter
from ctxdiff.capture.recorder import build_block
from ctxdiff.images import (
    detect_image_part,
    estimate_image_tokens,
    format_image_text,
    format_token_estimate,
    image_raw_block,
    sniff_dimensions,
)
from ctxdiff.models import CallBlock

# --- tiny synthetic images ---------------------------------------------------
#
# Header-accurate blobs rather than real photographs: the sniffer reads only the
# first few dozen bytes of each format, so a hand-built header exercises exactly
# the code path a 4 MB screenshot would, in 40 bytes and with no binary fixture
# checked into the repo.


def _png(width: int, height: int, payload: bytes = b"\x00\x00\x00") -> bytes:
    """A structurally valid PNG declaring `width`x`height` in its IHDR.
    `payload` varies the IDAT so two images of the SAME size can be given
    DIFFERENT bytes — which is how the dedup tests tell identity (the pixels)
    apart from the descriptor (the label)."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    ihdr = struct.pack(">II", width, height) + bytes([8, 2, 0, 0, 0])
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(payload)) + chunk(b"IEND", b""))


def _jpeg(width: int, height: int) -> bytes:
    """A JPEG carrying a JFIF APP0 segment before its SOF0, so the sniffer has
    to actually WALK the marker chain rather than read a fixed offset."""
    app0 = (b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + bytes([1, 1, 0])
            + struct.pack(">HH", 1, 1) + bytes([0, 0]))
    sof0 = (b"\xff\xc0" + struct.pack(">H", 17) + bytes([8])
            + struct.pack(">HH", height, width) + bytes([3])
            + bytes([1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1]))
    return b"\xff\xd8" + app0 + sof0 + b"\xff\xd9"


def _gif(width: int, height: int) -> bytes:
    """A GIF89a whose logical screen descriptor carries the size."""
    return b"GIF89a" + struct.pack("<HH", width, height) + bytes([0xF0, 0, 0]) + b"\x00" * 6 + b";"


def _webp_lossy(width: int, height: int) -> bytes:
    """A RIFF/WEBP container whose first chunk is a lossy `VP8 ` frame."""
    frame = bytes([0, 0, 0]) + b"\x9d\x01\x2a" + struct.pack("<HH", width, height) + b"\x00" * 8
    payload = b"VP8 " + struct.pack("<I", len(frame)) + frame
    return b"RIFF" + struct.pack("<I", 4 + len(payload)) + b"WEBP" + payload


def _webp_lossless(width: int, height: int) -> bytes:
    """A RIFF/WEBP container whose first chunk is a lossless `VP8L` frame —
    dimensions packed as two 14-bit fields, a completely different layout from
    the lossy one."""
    bits = (width - 1) | ((height - 1) << 14)
    body = bytes([0x2F]) + struct.pack("<I", bits) + b"\x00" * 12
    payload = b"VP8L" + struct.pack("<I", len(body)) + body
    return b"RIFF" + struct.pack("<I", 4 + len(payload)) + b"WEBP" + payload


def _webp_extended(width: int, height: int) -> bytes:
    """A RIFF/WEBP container whose first chunk is a `VP8X` extended header —
    the shape an animated or alpha-channel WebP uses."""
    body = bytes([0x10, 0, 0, 0]) + (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little")
    payload = b"VP8X" + struct.pack("<I", len(body)) + body
    return b"RIFF" + struct.pack("<I", 4 + len(payload)) + b"WEBP" + payload + b"\x00" * 16


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _uri(data: bytes, media_type: str = "image/png") -> str:
    return f"data:{media_type};base64,{_b64(data)}"


PNG_1024x768 = _png(1024, 768)
PNG_1024x768_ALT = _png(1024, 768, b"\x01\x02\x03")


# --- the dimension sniffer ----------------------------------------------------


@pytest.mark.parametrize("data,expected", [
    (_png(1024, 768), (1024, 768)),
    (_png(1, 1), (1, 1)),
    (_jpeg(800, 600), (800, 600)),
    (_gif(640, 480), (640, 480)),
    (_webp_lossy(500, 400), (500, 400)),
    (_webp_lossless(300, 200), (300, 200)),
    (_webp_extended(1280, 720), (1280, 720)),
])
def test_sniff_dimensions_reads_every_supported_format(data, expected):
    """PNG (IHDR), JPEG (SOFn after a walked marker chain), GIF (screen
    descriptor) and all three WebP encodings are read straight out of the header
    bytes — no decoder, and therefore no new dependency."""
    assert sniff_dimensions(data) == expected


@pytest.mark.parametrize("data", [
    b"",
    b"BM" + b"\x00" * 60,                       # BMP: a real format we don't sniff
    b"\x89PNG\r\n\x1a\n" + b"\x00" * 4,         # PNG signature, truncated before IHDR
    b"\xff\xd8\xff",                            # JPEG SOI with nothing after it
    b"RIFF" + b"\x00" * 8 + b"WEBP",            # WebP header with no chunk
    b"GIF89a",                                  # GIF header with no screen descriptor
    b"not an image at all, just prose",
])
def test_sniff_dimensions_degrades_on_unknown_or_truncated_headers(data):
    """An unrecognized format or a truncated header returns None rather than
    guessing or raising. None is not a failure — it is what makes the descriptor
    degrade to `[image]` and the estimate to "unknown"."""
    assert sniff_dimensions(data) is None


def test_sniff_dimensions_rejects_a_zero_dimension():
    """A header declaring a 0-pixel side is malformed, not a real size, so it
    reads as unknown instead of producing a nonsensical `[image 0×768]`."""
    assert sniff_dimensions(_png(0, 768)) is None


@pytest.mark.parametrize("width,height", [
    (0xFFFFFFFF, 0xFFFFFFFF),   # a truncated download / fuzzer / hostile header
    (65536, 768),               # one past the largest expressible dimension
    (768, 100_000),
])
def test_sniff_dimensions_rejects_an_implausible_dimension(width, height):
    """PNG's IHDR is a pair of uint32s, so a corrupt or hostile header can
    declare 4294967295×4294967295 — 18 exapixels — and be structurally valid.
    Trusting it turns one block into 8,068,951,256,159,688 estimated tokens,
    which makes every other block in the run round to 0.0% and the run total
    meaningless. Anything above 65535 (the largest size JPEG and GIF can even
    express) is a broken header, not a picture, so it reads as unknown."""
    assert sniff_dimensions(_png(width, height)) is None


def test_the_largest_expressible_dimension_is_still_read():
    """The clamp is a plausibility bound, not a size limit: 65535 is exactly what
    a uint16 field can hold, so it stays a real, readable size."""
    assert sniff_dimensions(_png(65535, 65535)) == (65535, 65535)


def test_a_jpeg_whose_frame_header_is_past_the_scan_limit_reads_as_unknown():
    """The marker walk is bounded. A payload that never resynchronizes degrades
    to byte-at-a-time resync, so an unbounded walk over a multi-megabyte corrupt
    JPEG stalls the HOST application's thread (capture runs synchronously inside
    the caller's call). This pins the bound by putting a perfectly good SOF0
    just past it: the sniffer must stop before reaching it."""
    from ctxdiff.images import _JPEG_MAX_SCAN

    tail = _jpeg(800, 600)[2:]  # everything after the SOI marker
    assert sniff_dimensions(b"\xff\xd8" + bytes(64) + tail) == (800, 600)
    assert sniff_dimensions(b"\xff\xd8" + bytes(_JPEG_MAX_SCAN) + tail) is None


def test_a_large_corrupt_jpeg_sniffs_in_bounded_time():
    """The guardrail this protects is "never break the host app": a corrupt or
    truncated multi-MB screenshot must not stall the agent that is being traced.
    Unbounded, this exact payload took ~985 ms (≈30 ms per MB); bounded it is a
    fixed ~30 ms no matter how large the payload grows."""
    import time

    payload = b"\xff\xd8" + bytes(32 << 20)  # 32 MiB that never resynchronizes
    started = time.perf_counter()
    assert sniff_dimensions(payload) is None
    assert time.perf_counter() - started < 0.25


# --- provider token formulas --------------------------------------------------


@pytest.mark.parametrize("width,height,detail,expected", [
    # detail:low is a flat 85 regardless of size — no tiling at all.
    (1024, 768, "low", 85),
    (4096, 4096, "low", 85),
    # detail:high — 85 base + 170 per 512px tile after the 2048/768 fit.
    (1024, 768, "high", 85 + 170 * 4),      # 2x2 tiles, no scaling needed
    (512, 512, "high", 85 + 170),           # exactly one tile
    (2000, 1200, "high", 85 + 170 * 6),     # fits 2048; shortest 1200 -> 768
    # Absent and "auto" are treated as high — the API's own behavior for any
    # image big enough to matter.
    (1024, 768, None, 85 + 170 * 4),
    (1024, 768, "auto", 85 + 170 * 4),
])
def test_openai_vision_tokens_follow_the_published_tiling_formula(width, height, detail, expected):
    """OpenAI publishes its vision cost exactly: 85 flat for low detail, else 85
    + 170 per 512x512 tile after fitting into 2048x2048 and reducing the shortest
    side to 768. This pins that arithmetic."""
    assert estimate_image_tokens("openai", width, height, detail) == expected


@pytest.mark.parametrize("width,height,expected", [
    (800, 600, 640),        # 480000 / 750
    (1024, 768, 1049),      # 786432 / 750, rounded up
    (10, 10, 1),            # never free
    (3000, 2000, 1568 * 1045 // 750 + 1),  # long edge clamped to 1568 first
])
def test_anthropic_vision_tokens_follow_the_published_pixel_rule(width, height, expected):
    """Anthropic publishes tokens ≈ (w × h) / 750, applied after resizing so the
    long edge is at most 1568px."""
    assert estimate_image_tokens("anthropic", width, height, None) == expected


def test_bedrock_shares_the_anthropic_formula():
    """A Bedrock Converse image is overwhelmingly headed for a Claude model, so
    it is costed with Anthropic's published rule rather than OpenAI's tiling."""
    assert (estimate_image_tokens("bedrock", 800, 600, None)
            == estimate_image_tokens("anthropic", 800, 600, None))


@pytest.mark.parametrize("width,height,expected", [
    (384, 384, 258),        # both sides <= 384: one flat tile
    (200, 100, 258),
    (640, 480, 258 * 4),    # tile = 480*2//3 = 320 -> 2x2 tiles
    (1024, 768, 258 * 4),   # tile = 512 -> 2x2 tiles
])
def test_gemini_vision_tokens_follow_the_published_tiling_formula(width, height, expected):
    """Gemini publishes 258 tokens for an image inside 384x384, else 258 per tile
    of `min(w,h)/1.5` clamped to 256..768."""
    assert estimate_image_tokens("gemini", width, height, None) == expected


def test_an_unknown_provider_falls_back_to_the_openai_formula():
    """Azure OpenAI and the OpenAI-compatible OSS endpoints speak the same wire
    format, so an unrecognized provider id is costed with OpenAI's tiling rather
    than left at zero."""
    assert (estimate_image_tokens("some-oss-gateway", 1024, 768, None)
            == estimate_image_tokens("openai", 1024, 768, None))


def test_unknown_dimensions_estimate_nothing_rather_than_guessing():
    """With no dimensions there is no honest number, so the estimate is 0 — a
    visible gap against provider usage — never a fabricated figure that would be
    indistinguishable from a measured one."""
    assert estimate_image_tokens("openai", None, None, "high") == 0
    assert estimate_image_tokens("anthropic", None, None, None) == 0
    assert estimate_image_tokens("gemini", None, None, None) == 0


def test_low_detail_is_the_one_cost_knowable_without_dimensions():
    """`detail: "low"` is a flat 85 tokens BY DEFINITION, so a remote low-detail
    image can still be costed honestly without fetching it."""
    assert estimate_image_tokens("openai", None, None, "low") == 85


# --- the descriptor -----------------------------------------------------------


@pytest.mark.parametrize("tokens,expected", [
    (0, "0"), (85, "85"), (765, "765"), (999, "999"),
    (1000, "1k"), (1049, "1k"), (1105, "1.1k"), (1150, "1.2k"),
    (1950, "2k"), (12345, "12.3k"),
])
def test_token_estimate_formatting(tokens, expected):
    """Exact below 1000, one decimal of thousands above it, bare `k` when the
    decimal rounds to zero. Rounded with floor(n/100 + 0.5) so Python's banker's
    rounding and JS's half-up cannot disagree."""
    assert format_token_estimate(tokens) == expected


def test_descriptor_carries_size_and_estimate():
    """The headline form the user chose: size, a separator, and a tilde-marked
    token estimate. No format or MIME type — the point is context cost, not file
    metadata."""
    assert format_image_text(1024, 768, 765) == "[image 1024×768 · ~765 tok]"


def test_descriptor_degrades_one_field_at_a_time():
    """The descriptor says exactly as much as is known and no more: cost without
    size when only the cost is knowable, and a bare `[image]` when neither is."""
    assert format_image_text(None, None, 85) == "[image · ~85 tok]"
    assert format_image_text(None, None, 0) == "[image]"
    assert format_image_text(1024, 768, 0) == "[image 1024×768]"


# --- per-provider part shapes -------------------------------------------------


def _one_block(adapter, kwargs):
    """Extract blocks and return the single image block among them, asserting
    there is exactly one — so a test that accidentally matched two shapes fails
    loudly instead of silently checking the wrong block."""
    images = [b for b in adapter.extract_blocks(kwargs) if b.kind == "image"]
    assert len(images) == 1
    return images[0]


def test_openai_chat_image_url_data_uri():
    """OpenAI Chat Completions `{"type": "image_url", "image_url": {"url":
    "data:...", "detail": ...}}` — the commonest vision shape there is."""
    block = _one_block(OpenAIAdapter(), {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": _uri(PNG_1024x768), "detail": "high"}},
    ]}]})
    assert block.text == "[image 1024×768 · ~765 tok]"
    assert block.token_count == 765
    assert block.token_method == "estimate"
    assert block.hash_input.startswith("image:sha256:")
    # The base64 must appear NOWHERE in what gets stored.
    assert _b64(PNG_1024x768)[:32] not in block.text


def test_openai_chat_image_url_as_a_bare_string():
    """Hand-rolled clients and older examples put the URL directly under
    `image_url` instead of nesting it; that still captures as an image."""
    block = _one_block(OpenAIAdapter(), {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": _uri(PNG_1024x768)},
    ]}]})
    assert block.text == "[image 1024×768 · ~765 tok]"


def test_openai_responses_input_image_data_uri():
    """OpenAI Responses spells it `input_image` with `image_url` as a plain
    STRING and `detail` at the top level — a different shape from Chat
    Completions that must still produce the same block."""
    block = _one_block(OpenAIAdapter(), {"input": [{"role": "user", "content": [
        {"type": "input_image", "image_url": _uri(PNG_1024x768), "detail": "high"},
    ]}]})
    assert block.text == "[image 1024×768 · ~765 tok]"
    assert block.token_count == 765


def test_openai_responses_input_image_file_id_degrades():
    """An image already uploaded to the Files API has no bytes we can reach, so
    it degrades to a bare `[image]` identified by the file id."""
    block = _one_block(OpenAIAdapter(), {"input": [{"role": "user", "content": [
        {"type": "input_image", "file_id": "file-3d9a17c04be84e2fb0c5"},
    ]}]})
    assert block.text == "[image]"
    assert block.token_count == 0
    assert block.hash_input == "image:ref:file-3d9a17c04be84e2fb0c5"


def test_anthropic_base64_source():
    """Anthropic's classic shape: `{"type": "image", "source": {"type":
    "base64", "media_type": ..., "data": ...}}`, costed with Anthropic's own
    published formula rather than OpenAI's."""
    block = _one_block(AnthropicAdapter(), {"messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                     "data": _b64(_jpeg(800, 600))}},
    ]}]})
    assert block.text == "[image 800×600 · ~640 tok]"
    assert block.token_count == 640


def test_anthropic_url_source_degrades_without_fetching():
    """The newer `{"type": "url"}` source names a remote image. ctxdiff will not
    fetch it, so the block degrades to `[image]` — identified by the URL, which
    still dedups a repeated reference."""
    block = _one_block(AnthropicAdapter(), {"messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "url", "url": "https://cdn.example.com/a.png"}},
    ]}]})
    assert block.text == "[image]"
    assert block.hash_input == "image:url:https://cdn.example.com/a.png"


def test_anthropic_file_source_degrades():
    """A Files API `{"type": "file"}` source is a provider-side handle — no
    bytes, no dimensions, so `[image]` keyed on the file id."""
    block = _one_block(AnthropicAdapter(), {"messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "file", "file_id": "file_011CQrsTuVwXyZ"}},
    ]}]})
    assert block.text == "[image]"
    assert block.hash_input == "image:ref:file_011CQrsTuVwXyZ"


@pytest.mark.parametrize("part", [
    {"inline_data": {"mime_type": "image/gif", "data": _b64(_gif(640, 480))}},
    {"inlineData": {"mimeType": "image/gif", "data": _b64(_gif(640, 480))}},
])
def test_gemini_inline_data_snake_and_camel_case(part):
    """google-genai spells the key `inline_data`/`mime_type` in Python and
    `inlineData`/`mimeType` in JS. Both reach the same block, so a trace captured
    in either SDK is comparable."""
    block = _one_block(GeminiAdapter(), {"contents": [{"role": "user", "parts": [part]}]})
    assert block.text == "[image 640×480 · ~1k tok]"
    assert block.token_count == 258 * 4


def test_gemini_inline_data_with_raw_bytes():
    """The Python google-genai SDK hands raw `bytes` for `inline_data.data`
    rather than base64; those are hashed and sniffed directly."""
    block = _one_block(GeminiAdapter(), {"contents": [{"role": "user", "parts": [
        {"inline_data": {"mime_type": "image/gif", "data": _gif(640, 480)}},
    ]}]})
    assert block.text == "[image 640×480 · ~1k tok]"


def test_gemini_non_image_inline_data_keeps_the_json_path():
    """`inline_data` also carries audio, video and PDFs. Only an image MIME type
    is rerouted; everything else keeps the pre-existing stable-JSON
    `content_part` behavior, byte for byte."""
    blocks = GeminiAdapter().extract_blocks({"contents": [{"role": "user", "parts": [
        {"inline_data": {"mime_type": "audio/wav", "data": "UklGRiQAAABXQVZF"}},
    ]}]})
    assert [b.kind for b in blocks] == ["content_part"]
    assert "audio/wav" in blocks[0].text


@pytest.mark.parametrize("part", [
    {"file_data": {"mime_type": "image/jpeg", "file_uri": "https://gen.googleapis.com/v1/files/7k2m"}},
    {"fileData": {"mimeType": "image/jpeg", "fileUri": "https://gen.googleapis.com/v1/files/7k2m"}},
])
def test_gemini_file_data_degrades_without_fetching(part):
    """A `file_data` URI points at the Files API. Never fetched, so `[image]`
    keyed on the URI — in both the snake_case and camelCase spellings."""
    block = _one_block(GeminiAdapter(), {"contents": [{"role": "user", "parts": [part]}]})
    assert block.text == "[image]"
    assert block.hash_input == "image:ref:https://gen.googleapis.com/v1/files/7k2m"


def test_bedrock_converse_image_bytes():
    """Bedrock Converse nests the image under an `image` key with a `format` and
    a `source.bytes` that botocore hands over as raw bytes."""
    block = _one_block(BedrockAdapter(), {"messages": [{"role": "user", "content": [
        {"image": {"format": "png", "source": {"bytes": PNG_1024x768}}},
    ]}]})
    assert block.text == "[image 1024×768 · ~1k tok]"   # Anthropic formula
    assert block.token_count == 1049


def test_bedrock_s3_location_degrades():
    """An `s3Location` source names an object in the caller's own bucket, which
    ctxdiff will not read — so `[image]`, keyed on the S3 URI."""
    block = _one_block(BedrockAdapter(), {"messages": [{"role": "user", "content": [
        {"image": {"format": "png", "source": {"s3Location": {"uri": "s3://shots/frame-1.png"}}}},
    ]}]})
    assert block.text == "[image]"
    assert block.hash_input == "image:ref:s3://shots/frame-1.png"


# --- degradation and the no-network guarantee ---------------------------------


def test_a_remote_http_url_degrades_and_is_never_fetched(monkeypatch):
    """THE local-first guarantee, asserted rather than assumed: capturing a
    remote image URL must make no network request of any kind.

    How this proves it: every outbound path in the stdlib funnels through
    `socket.socket` (urllib, http.client, requests, httpx and everything else
    build on it), so replacing the constructor with a raiser turns any attempted
    connection — direct or through a library — into an immediate test failure."""
    import socket

    def _no_network(*args, **kwargs):
        raise AssertionError("ctxdiff attempted a network connection while capturing an image")

    monkeypatch.setattr(socket, "socket", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)

    block = _one_block(OpenAIAdapter(), {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://cdn.example.com/huge.png"}},
    ]}]})
    assert block.text == "[image]"
    assert block.token_count == 0
    assert block.hash_input == "image:url:https://cdn.example.com/huge.png"


def test_a_remote_low_detail_url_still_reports_its_flat_cost():
    """A remote image is unmeasurable EXCEPT under `detail: "low"`, whose cost is
    a constant. The descriptor then carries the estimate without a size."""
    block = _one_block(OpenAIAdapter(), {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://cdn.example.com/a.png", "detail": "low"}},
    ]}]})
    assert block.text == "[image · ~85 tok]"
    assert block.token_count == 85


def test_an_unknown_image_format_still_becomes_an_image_block():
    """A BMP (or a future format) is not sniffable, but it is still an image and
    still must not be tokenized as base64 prose — so it becomes `[image]` with
    the bytes as its identity."""
    block = _one_block(OpenAIAdapter(), {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": _uri(b"BM" + b"\x00" * 60, "image/bmp")}},
    ]}]})
    assert block.text == "[image]"
    assert block.hash_input.startswith("image:sha256:")


def test_an_implausible_header_degrades_to_a_bare_image_block():
    """End to end for the clamp: a hostile PNG header still becomes an image
    block (it IS an image, and its bytes are still its identity) — but with no
    size and no estimate, so one broken header cannot dominate the run's token
    attribution or its descriptor."""
    monster = _png(0xFFFFFFFF, 0xFFFFFFFF)
    raw = image_raw_block("user", {"type": "image_url", "image_url": {
        "url": _uri(monster)}}, "gemini")
    assert raw.text == "[image]"
    assert raw.token_count == 0
    assert raw.hash_input.startswith("image:sha256:")


def test_an_undecodable_payload_falls_back_to_the_previous_behavior():
    """When there are no bytes to be had at all, the part is NOT claimed as an
    image: it keeps the old stable-JSON `content_part` path. Falling back is
    always safe; inventing an empty-bytes image block would make every malformed
    image in a trace dedup into one bogus row."""
    blocks = AnthropicAdapter().extract_blocks({"messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "data": "!!!!"}},
    ]}]})
    assert [b.kind for b in blocks] == ["content_part"]


def test_detect_image_part_never_raises_on_garbage():
    """The detector runs inside the fail-open recorder but stays defensive on
    its own: no shape of malformed input may raise out of it."""
    for part in [None, 42, "a string", [], {}, {"type": "image_url"},
                 {"type": "image", "source": None}, {"type": "image", "source": {}},
                 {"inline_data": "not a dict"}, {"image": {"source": 7}},
                 {"type": "image_url", "image_url": {"url": 12345}}]:
        assert detect_image_part(part) is None


# --- identity: dedup and non-collision ----------------------------------------


def test_the_same_image_sent_twice_is_one_block():
    """The dedup promise. The same screenshot re-sent on the next turn must land
    on the SAME content hash, so the diff can report it unchanged and the store
    holds one copy."""
    part = {"type": "image_url", "image_url": {"url": _uri(PNG_1024x768), "detail": "high"}}
    first = build_block(image_raw_block("user", part, "openai"), "openai")
    second = build_block(image_raw_block("user", dict(part), "openai"), "openai")
    assert first.content_hash == second.content_hash


#: The same PNG in all four wrappers, each paired with the provider that
#: actually emits that wrapper — so a test using this exercises the real
#: cross-provider case (different formulas, therefore different counts) rather
#: than four shapes pretending to be one provider.
_SAME_PNG_ACROSS_PROVIDERS = [
    ("openai", {"type": "image_url", "image_url": {"url": _uri(PNG_1024x768)}}),
    ("anthropic", {"type": "image", "source": {
        "type": "base64", "media_type": "image/png", "data": _b64(PNG_1024x768)}}),
    ("gemini", {"inline_data": {"mime_type": "image/png", "data": _b64(PNG_1024x768)}}),
    ("bedrock", {"image": {"format": "png", "source": {"bytes": PNG_1024x768}}}),
]


def test_the_same_bytes_dedup_across_providers_and_wrappers():
    """Identity is the PIXELS, not the envelope. The same PNG wrapped as an
    OpenAI data URI, an Anthropic base64 source, a Gemini `inline_data` and a
    Bedrock `source.bytes` is one block — which is what makes a screenshot
    traceable through a multi-provider agent.

    Each shape is built under the provider that actually emits it, so this
    really is the cross-provider case: the four vision formulas disagree (765 /
    1049 / 1032 / 1049 tokens for the same 1024×768 PNG) and the descriptor
    embeds whichever count was computed — yet all four land on ONE hash."""
    blocks = [build_block(image_raw_block("user", part, provider), provider)
              for provider, part in _SAME_PNG_ACROSS_PROVIDERS]
    assert len({b.content_hash for b in blocks}) == 1
    assert [b.token_count for b in blocks] == [765, 1049, 1032, 1049]
    assert len({b.text for b in blocks}) > 1  # the counts really do differ


def test_a_cross_provider_dedup_keeps_the_first_writers_count_and_text(tmp_path):
    """The documented consequence of content-addressed dedup, asserted rather
    than waved away (see `spec/ctrace-schema.md`, "Image blocks"): when the same
    screenshot is sent to OpenAI and then to Anthropic, the block is written
    ONCE, and `INSERT OR IGNORE` keeps the first writer's `token_count` AND the
    count embedded in its descriptor — so the Anthropic turn renders OpenAI's
    765 where Anthropic's own formula says 1049.

    This is not specific to images: a text block deduped across an exact and an
    estimating provider has always behaved this way. It is pinned here because
    for an image the number is also visible in `text`, where a reader might take
    it for a per-call measurement."""
    from ctxdiff.store.ctrace import CTrace

    path = tmp_path / "cross.ctrace"
    ct = CTrace.create(str(path), project="vision", provider="openai", model="gpt-4o",
                       started_at="2026-04-11T09:30:00+00:00")
    for seq, (provider, part) in enumerate(_SAME_PNG_ACROSS_PROVIDERS[:2], start=1):
        raw = image_raw_block("user", part, provider)
        block = build_block(raw, provider)
        ct.record_call(seq=seq, params={"model": "m"}, usage=None, latency_ms=1, error=None,
                       call_blocks=[CallBlock(block=block, position=0, label="user",
                                              label_source="heuristic")])
    ct.close()

    ct = CTrace.open(str(path))
    stored = [ct.get_call_blocks(c.id)[0].block for c in ct.get_calls()]
    ct.close()
    assert stored[0].content_hash == stored[1].content_hash          # one block…
    assert stored[1].token_count == 765                              # …OpenAI's count
    assert stored[1].text == "[image 1024×768 · ~765 tok]"           # …and its text


# --- identity: the cost-affecting envelope ------------------------------------


def test_the_same_screenshot_at_two_detail_levels_is_two_blocks():
    """THE computer-use pattern: the same screenshot sits in history at
    `detail:"low"` and is re-sent for the current turn at `detail:"high"`.
    `detail` changes the cost NINE-fold (85 vs 765) and changes the descriptor,
    so the two must be two blocks. Collapsed into one, `INSERT OR IGNORE` keeps
    whichever was written first and `ctxdiff tokens` prints 85 where the truth is
    765 — a silent Δ +689 against the provider's own usage."""
    def block_for(detail):
        part = {"type": "image_url", "image_url": {"url": _uri(PNG_1024x768), "detail": detail}}
        return build_block(image_raw_block("user", part, "openai"), "openai")

    low, high = block_for("low"), block_for("high")
    assert low.content_hash != high.content_hash
    assert (low.token_count, high.token_count) == (85, 765)
    assert low.text == "[image 1024×768 · ~85 tok]"
    assert high.text == "[image 1024×768 · ~765 tok]"


def test_a_detail_change_reads_as_a_modified_block_in_the_diff(tmp_path):
    """The user-visible half of the same defect: `ctxdiff diff` must report the
    low→high promotion as a CHANGE. While `detail` was outside the identity the
    two turns shared one block and the diff called a 9× cost increase
    'unchanged'."""
    from ctxdiff.analyze.differ import diff_turns
    from ctxdiff.store.ctrace import CTrace

    def call_blocks(detail):
        part = {"type": "image_url", "image_url": {"url": _uri(PNG_1024x768), "detail": detail}}
        block = build_block(image_raw_block("user", part, "openai"), "openai")
        return [CallBlock(block=block, position=0, label="user", label_source="heuristic")]

    path = tmp_path / "detail.ctrace"
    ct = CTrace.create(str(path), project="vision", provider="openai", model="gpt-4o",
                       started_at="2026-04-11T09:30:00+00:00")
    for seq, detail in enumerate(("low", "high"), start=1):
        ct.record_call(seq=seq, params={"model": "gpt-4o"}, usage=None, latency_ms=1,
                       error=None, call_blocks=call_blocks(detail))
    ct.close()

    ct = CTrace.open(str(path))
    diff = diff_turns(ct, 1, 2)
    ct.close()
    assert [e.kind for e in diff.entries] == ["modified"]
    assert diff.tokens_added == 765 and diff.tokens_evicted == 85


def test_a_cache_control_breakpoint_changes_an_images_identity():
    """Anthropic's `cache_control` is the single most cache-relevant key a
    content part can carry, and it rides as a SIBLING of the image payload.
    Rebuilding the block from the payload alone dropped it, so adding or
    removing a cache breakpoint on an image stopped changing the block's hash
    and `ctxdiff cache` reported a stable prefix across a real caching change —
    the exact failure class the profiler exists to catch, and asymmetric with a
    text part, whose hash DOES move."""
    source = {"type": "base64", "media_type": "image/png", "data": _b64(PNG_1024x768)}
    plain = image_raw_block("user", {"type": "image", "source": source}, "anthropic")
    marked = image_raw_block("user", {"type": "image", "source": source,
                                      "cache_control": {"type": "ephemeral"}}, "anthropic")
    again = image_raw_block("user", {"type": "image", "source": source,
                                     "cache_control": {"type": "ephemeral"}}, "anthropic")

    assert plain.hash_input != marked.hash_input   # the breakpoint is identity…
    assert marked.hash_input == again.hash_input   # …but a stable one
    assert plain.text == marked.text               # and it changes nothing else
    assert plain.token_count == marked.token_count


def test_an_unrecognized_sibling_key_folds_into_identity():
    """The general rule behind the `cache_control` fix: every key ctxdiff
    understands is already represented in the block (the payload IS the hash;
    `detail` is appended to it; the media type is deliberately excluded).
    Anything left over is a modifier ctxdiff cannot interpret and therefore must
    not silently discard — a provider-side behavior change with no trace in the
    trace is worse than an over-eager hash."""
    part = {"type": "image_url", "image_url": {"url": _uri(PNG_1024x768)}}
    plain = image_raw_block("user", part, "openai")
    extra = image_raw_block("user", {**part, "x_provider_hint": "grounding"}, "openai")
    assert plain.hash_input != extra.hash_input


def test_the_standard_shapes_carry_no_leftover_envelope():
    """The other side of the rule: folding the remainder in must NOT defeat
    dedup for the shapes providers actually emit. Every documented wrapper has
    an empty remainder, so a plain OpenAI data URI and a plain Anthropic base64
    source still hash identically — the property the test above this one
    depends on."""
    for provider, part in _SAME_PNG_ACROSS_PROVIDERS:
        assert image_raw_block("user", part, provider).hash_input == (
            "image:sha256:" + hashlib.sha256(PNG_1024x768).hexdigest())


def test_two_different_images_of_the_same_size_are_two_blocks():
    """The other half of identity: two DIFFERENT pictures that happen to share a
    size render the SAME descriptor text, and must still be two blocks. If
    identity were the descriptor they would collide and a changed screenshot
    would silently read as unchanged."""
    def block_for(data):
        part = {"type": "image_url", "image_url": {"url": _uri(data), "detail": "high"}}
        return build_block(image_raw_block("user", part, "openai"), "openai")

    a, b = block_for(PNG_1024x768), block_for(PNG_1024x768_ALT)
    assert a.text == b.text                      # same label…
    assert a.content_hash != b.content_hash      # …different block


def test_base64_formatting_differences_do_not_change_identity():
    """Padding, embedded newlines and the URL-safe alphabet are formatting, not
    content. All three must decode to the same bytes and therefore the same
    hash — otherwise one client's line-wrapping would defeat dedup."""
    plain = _b64(PNG_1024x768)
    variants = [plain, plain.rstrip("="), "\n".join(plain[i:i + 60] for i in range(0, len(plain), 60)),
                plain.replace("+", "-").replace("/", "_")]
    hashes = {
        build_block(image_raw_block(
            "user", {"type": "image", "source": {"type": "base64", "data": v}}, "anthropic"),
            "anthropic").content_hash
        for v in variants
    }
    assert len(hashes) == 1


# --- how the block reaches the store ------------------------------------------


def test_build_block_uses_the_image_overrides_not_the_tokenizer():
    """`build_block` is the single definition of a stored block. For an image it
    must hash the byte digest and take the pre-computed vision estimate — never
    tokenize the `[image …]` descriptor, which would report about seven tokens
    for a full-page screenshot."""
    raw = image_raw_block("user", {"type": "image_url", "image_url": {
        "url": _uri(PNG_1024x768), "detail": "high"}}, "openai")
    block = build_block(raw, "openai")
    assert block.kind == "image"
    assert block.token_count == 765
    assert block.token_method == "estimate"
    from ctxdiff.models import content_hash
    assert block.content_hash == content_hash("user", "image", raw.hash_input)


def test_build_block_leaves_ordinary_blocks_exactly_as_before():
    """The overrides are opt-in. A RawBlock without them hashes and tokenizes its
    text, which is what every pre-existing block in every existing `.ctrace`
    did — this is the back-compatibility of the writer path."""
    from ctxdiff.models import RawBlock, content_hash
    from ctxdiff.tokenize.counter import count_tokens

    raw = RawBlock(role="user", kind="message", text="hello world")
    block = build_block(raw, "openai")
    assert block.content_hash == content_hash("user", "message", "hello world")
    assert (block.token_count, block.token_method) == count_tokens("hello world", "openai")


def test_a_vision_call_marks_the_whole_turn_approximate(tmp_path):
    """End to end through the recorder and the tokens analyzer: because an image
    block carries `token_method="estimate"`, the call it belongs to is reported
    as approximate. This is the honesty guarantee — a vision estimate can never
    be rendered as an exact tiktoken count, in this reader or in an older one."""
    from ctxdiff.analyze.tokens import analyze_call
    from ctxdiff.capture.recorder import Recorder
    from ctxdiff.store.ctrace import CTrace

    path = tmp_path / "vision.ctrace"
    ct = CTrace.create(str(path), project="vision", provider="openai", model="gpt-4o",
                       started_at="2026-04-11T09:30:00+00:00")
    Recorder(ct, OpenAIAdapter(), None).record(
        seq=1,
        kwargs={"model": "gpt-4o", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": _uri(PNG_1024x768), "detail": "high"}},
        ]}]},
        response=None, latency_ms=5, error=None, tagged=[])
    ct.close()

    ct = CTrace.open(str(path))
    call = ct.get_calls()[0]
    blocks = ct.get_call_blocks(call.id)
    image = [cb for cb in blocks if cb.block.kind == "image"][0]
    assert image.block.text == "[image 1024×768 · ~765 tok]"
    assert analyze_call(call, blocks).approximate is True
    ct.close()


def test_the_stored_ctrace_never_contains_the_base64(tmp_path):
    """The size half of the fix: the payload must not reach the file at all. A
    100 KB screenshot used to be written verbatim into `block.text` (and into
    every HTML export of that trace); now the row holds a 27-character
    descriptor."""
    from ctxdiff.capture.recorder import Recorder
    from ctxdiff.store.ctrace import CTrace

    big = _png(1024, 768, b"\x00" * 200_000)
    path = tmp_path / "big.ctrace"
    ct = CTrace.create(str(path), project="vision", provider="openai", model="gpt-4o",
                       started_at="2026-04-11T09:30:00+00:00")
    Recorder(ct, OpenAIAdapter(), None).record(
        seq=1,
        kwargs={"model": "gpt-4o", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _uri(big)}},
        ]}]},
        response=None, latency_ms=5, error=None, tagged=[])
    ct.close()

    raw = path.read_bytes()
    assert _b64(big)[:64].encode() not in raw
    assert "[image 1024×768".encode() in raw


# --- back-compatibility -------------------------------------------------------


def test_a_pre_existing_v2_file_with_a_json_dumped_image_still_opens(tmp_path):
    """BACK-COMPAT, the load-bearing test. Files captured before this change hold
    the image as a `content_part` whose text is the JSON-serialized part,
    tiktoken-counted. Nothing migrates them — a debugger must not rewrite the
    evidence it inspects — so they must keep opening and rendering exactly as
    they did, with their original hash, kind, text and token method intact."""
    import json
    import sqlite3

    from ctxdiff.models import basic_label, content_hash
    from ctxdiff.store.ctrace import CTrace
    from ctxdiff.store.schema import DDL, SCHEMA_VERSION
    from ctxdiff.tokenize.counter import count_tokens

    legacy_text = json.dumps(
        {"type": "image_url", "image_url": {"url": _uri(PNG_1024x768), "detail": "high"}},
        sort_keys=True, ensure_ascii=False)
    chash = content_hash("user", "content_part", legacy_text)
    tokens, method = count_tokens(legacy_text, "openai")
    label, label_source = basic_label("user", "content_part", legacy_text, [])

    path = tmp_path / "legacy.ctrace"
    conn = sqlite3.connect(str(path))
    conn.executescript(DDL)
    with conn:
        conn.execute("INSERT INTO run VALUES (?,?,?,?,?,?,?)",
                     ("a" * 32, "legacy", "2026-01-01T00:00:00+00:00", "openai",
                      json.dumps(["gpt-4o"]), "0.1.0", SCHEMA_VERSION))
        conn.execute("INSERT INTO call VALUES (?,?,?,?,?,?,?,?,?,?)",
                     ("b" * 32, "a" * 32, 1, json.dumps({"model": "gpt-4o"}), None,
                      10, None, None, None, "openai"))
        conn.execute("INSERT INTO block VALUES (?,?,?,?,?,?)",
                     (chash, "user", "content_part", legacy_text, tokens, method))
        conn.execute("INSERT INTO call_block VALUES (?,?,?,?,?)",
                     ("b" * 32, chash, 0, label, label_source))
    conn.close()

    ct = CTrace.open(str(path))
    blocks = ct.get_call_blocks(ct.get_calls()[0].id)
    assert len(blocks) == 1
    assert blocks[0].block.kind == "content_part"
    assert blocks[0].block.text == legacy_text
    assert blocks[0].block.content_hash == chash
    assert blocks[0].block.token_method == "tiktoken"
    ct.close()


def test_an_image_block_is_written_at_the_unchanged_schema_version(tmp_path):
    """The other direction of back-compat: a file containing image blocks is
    still SCHEMA_VERSION 2. The change is additive — a new `kind` value and a new
    text convention in columns that already exist — so an older ctxdiff opens the
    file and simply shows the descriptor. No version bump, no rejection."""
    import sqlite3

    from ctxdiff.capture.recorder import Recorder
    from ctxdiff.store.ctrace import CTrace
    from ctxdiff.store.schema import SCHEMA_VERSION

    path = tmp_path / "img.ctrace"
    ct = CTrace.create(str(path), project="vision", provider="openai", model="gpt-4o",
                       started_at="2026-04-11T09:30:00+00:00")
    Recorder(ct, OpenAIAdapter(), None).record(
        seq=1,
        kwargs={"model": "gpt-4o", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _uri(PNG_1024x768)}},
        ]}]},
        response=None, latency_ms=5, error=None, tagged=[])
    ct.close()

    # Read the version straight off the row: `Run` does not surface it, and the
    # claim under test is about what was WRITTEN to the file.
    conn = sqlite3.connect(str(path))
    stored_version = conn.execute("SELECT schema_version FROM run").fetchone()[0]
    conn.close()
    assert stored_version == SCHEMA_VERSION == 2

    ct = CTrace.open(str(path))
    block = ct.get_call_blocks(ct.get_calls()[0].id)[0].block
    assert block.kind == "image"
    # An older reader labels by role, and `basic_label` has no image branch — so
    # the label an old ctxdiff computes is the same one this ctxdiff stores.
    assert basic_label_for(block) == "user"
    ct.close()


def basic_label_for(block):
    """The label a reader with no knowledge of the `image` kind would compute —
    used above to show that an old ctxdiff labels an image block identically."""
    from ctxdiff.models import basic_label

    return basic_label(block.role, block.kind, block.text, [])[0]
