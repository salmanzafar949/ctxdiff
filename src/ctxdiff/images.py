"""Image content parts — detection, header sniffing, vision-token estimation
and the one-line descriptor that stands in for the pixels.

WHY THIS EXISTS. Before this module, an image content part reached the block
model through the same path as any other non-string part: `json.dumps(part)`.
For `{"type": "image_url", "image_url": {"url": "data:image/png;base64,<100k
chars>"}}` that meant the base64 blob became the block's `text` — tokenized by
tiktoken as if it were prose (tens of thousands of phantom tokens for an image
that really costs a few hundred), and written verbatim into the `.ctrace`.
Token attribution was therefore wrong for exactly the vision/computer-use
agents that most need context auditing.

WHAT IT DOES INSTEAD. An image part becomes a block whose

  * `kind`  is `"image"` (a distinct kind, so readers can tell it apart);
  * `text`  is a short descriptor — `[image 1024×768 · ~765 tok]` — never the
            bytes;
  * identity is sha256 over the image BYTES (see `image_hash_input`), so the
            same picture dedups across turns, across sessions and even across
            providers (an OpenAI data URI and an Anthropic base64 source
            carrying the same bytes are one block);
  * `token_count` is the provider's DOCUMENTED vision-token formula applied to
            the sniffed dimensions, always marked `token_method="estimate"` —
            a vision cost is never reported as an exact tiktoken count.

LOCAL-FIRST, ALWAYS. Nothing here ever performs I/O. A remote `http(s)` image
URL is identified by its URL and degrades to `[image]` with no token estimate;
fetching it to learn its size would turn a local debugging tool into a network
client, leak the trace subject's URLs, and change the numbers depending on
whether the host was online. Same for a provider-side file id.

Pure stdlib, no new dependency: dimensions are read from the first bytes of the
file (PNG IHDR, JPEG SOFn, GIF screen descriptor, WebP VP8/VP8L/VP8X) by the
small sniffer below.

PARITY: every function here has a byte-identical twin in `js/src/images.ts`.
Same descriptor text, same hash input, same token numbers. All arithmetic is
integer-only (`//`, not `/`) precisely so the two languages cannot round apart.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import re
from dataclasses import dataclass, replace

from ctxdiff.models import RawBlock, normalize_text

# The kind stored on an image block. Additive: readers that predate this kind
# see an ordinary block whose text is the descriptor, which is exactly what we
# want them to show.
IMAGE_KIND = "image"

# Image blocks are ALWAYS estimates, and they reuse the existing `"estimate"`
# marker rather than inventing a new one. Every reader already written —
# including an older ctxdiff opening a file this SDK writes today — tests
# `token_method == "estimate"` to decide whether to print the "~est" marker
# (see `analyze/tokens.py` and the viewer template). A new value would slip
# past those tests and render a vision estimate as if it were exact, which is
# the one thing this module exists to prevent.
IMAGE_TOKEN_METHOD = "estimate"

# The multiplication sign and middle dot used in the descriptor. Named so the
# JS twin can be compared against them literally.
_TIMES = "×"  # ×
_DOT = "·"  # ·


@dataclass(frozen=True)
class ImageRef:
    """What an adapter could learn about one image part WITHOUT doing any I/O.

    Exactly one of `data` / `url` / `ref` is normally set, in that order of
    preference: `data` when the part carried the bytes inline (a data URI, an
    Anthropic base64 source, a Gemini `inline_data`, a Bedrock image source),
    `url` when it referenced a remote image we deliberately will not fetch, and
    `ref` when it named a provider-side object (an OpenAI `file_id`, a Gemini
    `file_uri`) that only the provider can resolve.

    `detail` is OpenAI's per-image fidelity hint (`"low"` / `"high"` /
    `"auto"`); it changes the token cost — 85 vs 765 for the same 1024×768
    screenshot — so it is part of the identity (see `image_hash_input`) and is
    None for providers that have no such concept. `media_type` is carried for
    diagnostics only — it is deliberately NOT part of the descriptor or the
    hash.

    `modifiers` is the stable-JSON of everything ELSE the content part carried
    (see `_part_modifiers`): the keys ctxdiff does not interpret but the
    provider does, chiefly Anthropic's `cache_control`. It is filled in by
    `detect_image_part` and is None both when the part had no leftovers (every
    documented shape) and when an ImageRef was built directly by a caller."""
    data: bytes | None = None
    url: str | None = None
    ref: str | None = None
    detail: str | None = None
    media_type: str | None = None
    modifiers: str | None = None


# --- base64 / data URI ------------------------------------------------------


_NON_B64 = re.compile(r"[^A-Za-z0-9+/]")


def _b64_decode(value: object) -> bytes | None:
    """Decode base64 that came off the wire, or return None when there are no
    bytes to be had.

    Tolerant on purpose: SDK payloads arrive with missing `=` padding, embedded
    newlines (from a shell `base64` invocation) and occasionally in the
    URL-safe alphabet. Each of those is a formatting difference, not a
    different image. `bytes`/`bytearray` values pass straight through — the
    google-genai SDK hands raw bytes for `inline_data`.

    The normalization is spelled out step by step rather than left to the two
    languages' decoders, because those decoders disagree on malformed input and
    the bytes ARE the block's identity — a divergence here would be a
    cross-SDK hash divergence. So: map the URL-safe alphabet back, DELETE every
    character outside the base64 alphabet (including all `=`), drop a lone
    trailing character that cannot encode a whole byte, then re-pad. Both SDKs
    perform exactly these steps, so both see the same input by the time a
    decoder is involved.

    Returning None — for a non-string, an undecodable string, or anything that
    decodes to zero bytes — is what makes the caller fall back to the old
    JSON-serialization path. That matters: without the empty-bytes guard, every
    malformed image in a trace would share the digest of the empty string and
    dedup into a single bogus block."""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value) or None
    if not isinstance(value, str) or not value:
        return None
    cleaned = _NON_B64.sub("", value.replace("-", "+").replace("_", "/"))
    if len(cleaned) % 4 == 1:
        cleaned = cleaned[:-1]  # a lone trailing sextet encodes no whole byte
    cleaned += "=" * (-len(cleaned) % 4)
    if not cleaned:
        return None
    try:
        return base64.b64decode(cleaned, validate=False) or None
    except (binascii.Error, ValueError):
        return None


def _parse_data_uri(url: str) -> tuple[bytes | None, str | None]:
    """Split a `data:` URI into (bytes, media type), or (None, None) when it is
    not one. Handles the only form providers emit — `data:<mime>;base64,<b64>`
    — and returns the media type lowercased. A non-base64 data URI (percent-
    encoded text) yields no bytes; there is no such thing as a percent-encoded
    image in these APIs, and guessing would be worse than degrading."""
    if not url.startswith("data:"):
        return (None, None)
    head, sep, payload = url[5:].partition(",")
    if not sep:
        return (None, None)
    params = head.split(";")
    media_type = params[0].strip().lower() or None
    if "base64" not in [p.strip().lower() for p in params[1:]]:
        return (None, media_type)
    return (_b64_decode(payload), media_type)


# --- header sniffing --------------------------------------------------------


def _be32(data: bytes, off: int) -> int:
    """Big-endian uint32 at `off` — PNG's byte order."""
    return (data[off] << 24) | (data[off + 1] << 16) | (data[off + 2] << 8) | data[off + 3]


def _le16(data: bytes, off: int) -> int:
    """Little-endian uint16 at `off` — GIF's and WebP's byte order."""
    return data[off] | (data[off + 1] << 8)


def _png_size(data: bytes) -> tuple[int, int] | None:
    """PNG: the 8-byte signature is followed immediately by the IHDR chunk,
    whose first two fields are width and height as big-endian uint32 at fixed
    offsets 16 and 20. Requiring the literal `IHDR` tag rejects a file that
    merely starts with the signature."""
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        return None
    return (_be32(data, 16), _be32(data, 20))


def _gif_size(data: bytes) -> tuple[int, int] | None:
    """GIF: after the 6-byte `GIF87a`/`GIF89a` header comes the logical screen
    descriptor, whose first four bytes are width then height, little-endian."""
    if len(data) < 10 or data[:6] not in (b"GIF87a", b"GIF89a"):
        return None
    return (_le16(data, 6), _le16(data, 8))


# JPEG markers that are NOT start-of-frame even though they fall in the 0xC0..
# 0xCF range: DHT (define Huffman table), JPG (reserved), DAC (define
# arithmetic coding). Everything else in that range is an SOFn carrying the
# frame dimensions.
_JPEG_NON_SOF = (0xC4, 0xC8, 0xCC)

# How far into a payload the marker walk will look for the start-of-frame.
#
# WHY A BOUND AT ALL. A well-formed JPEG is walked segment by segment (`i += 2
# + seg_len`), so it reaches SOF in a handful of iterations however large the
# file is. A payload that merely BEGINS `FF D8` — a truncated download, a
# corrupt screenshot, a fuzzer's output — desynchronizes and degrades to
# byte-at-a-time resync over the whole buffer: ~30 ms per MB in Python, so a
# 100 MB blob burned 5.4 seconds. `extract_blocks` runs synchronously on the
# host application's thread during capture, so that is the traced agent
# stalling, which violates the "never break the host app" guardrail.
#
# WHY 1 MiB. The bound must sit above the largest legitimate run of pre-SOF
# metadata. A single JPEG segment is capped at 65533 bytes by the format, but a
# file may chain them: a full EXIF APP1 plus a multi-segment APP2 ICC profile is
# the realistic worst case and lands well under 1 MiB. Past the bound the
# sniffer reports "unknown" — an honest `[image]` — rather than scanning on.
_JPEG_MAX_SCAN = 1 << 20


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    """JPEG: walk the marker segments from SOI until a start-of-frame (SOFn),
    whose payload holds precision, then height and width as big-endian uint16.

    How the walk works: every marker is `0xFF <code>`; standalone markers (SOI,
    EOI, RSTn, TEM) carry no length, all others are followed by a 2-byte
    segment length that INCLUDES those two bytes. Fill bytes (runs of `0xFF`)
    are skipped. The walk stops at `_JPEG_MAX_SCAN` (see above) rather than at
    the end of the buffer, so the work is bounded by a constant instead of by
    the size of a possibly-adversarial payload; the SOF payload itself is still
    read against the real length, so a frame header found just before the bound
    is not lost to it."""
    n = len(data)
    if n < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return None
    limit = min(n, _JPEG_MAX_SCAN)
    i = 2
    while i + 3 < limit:
        if data[i] != 0xFF:  # desynchronized — resync on the next 0xFF
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xFF:  # fill byte
            i += 1
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD9:  # standalone, no length
            i += 2
            continue
        seg_len = (data[i + 2] << 8) | data[i + 3]
        if seg_len < 2:
            return None
        if 0xC0 <= marker <= 0xCF and marker not in _JPEG_NON_SOF:
            if i + 9 >= n:
                return None
            height = (data[i + 5] << 8) | data[i + 6]
            width = (data[i + 7] << 8) | data[i + 8]
            return (width, height) if width and height else None
        i += 2 + seg_len
    return None


def _webp_size(data: bytes) -> tuple[int, int] | None:
    """WebP: a RIFF container whose first chunk identifies one of three
    encodings, each storing the canvas size differently.

      * `VP8 ` (lossy) — a 3-byte frame tag, then the 3-byte start code
        `9D 01 2A`, then width and height as 14-bit little-endian fields.
      * `VP8L` (lossless) — signature byte `0x2F`, then 14 bits of width-1 and
        14 bits of height-1 packed into a little-endian uint32.
      * `VP8X` (extended: animation/alpha/ICC) — canvas width-1 and height-1 as
        two 24-bit little-endian fields."""
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None
    chunk = data[12:16]
    if chunk == b"VP8 ":
        if data[23:26] != b"\x9d\x01\x2a":
            return None
        return (_le16(data, 26) & 0x3FFF, _le16(data, 28) & 0x3FFF)
    if chunk == b"VP8L":
        if data[20] != 0x2F:
            return None
        bits = data[21] | (data[22] << 8) | (data[23] << 16) | (data[24] << 24)
        return ((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
    if chunk == b"VP8X":
        width = (data[24] | (data[25] << 8) | (data[26] << 16)) + 1
        height = (data[27] | (data[28] << 8) | (data[29] << 16)) + 1
        return (width, height)
    return None


# The largest side length a header is believed when it declares.
#
# Headers are read, not verified — the pixels are never decoded — so a corrupt
# or hostile file declares whatever it likes. PNG's IHDR is a pair of uint32s
# and WebP's VP8X canvas is 24-bit, so `0xFFFFFFFF × 0xFFFFFFFF` (18 exapixels)
# is a structurally VALID header, and trusting it produced a block estimated at
# 8,068,951,256,159,688 tokens: every other block in the run rounds to 0.0% and
# the run total becomes meaningless. 65535 is the ceiling because it is exactly
# what JPEG's and GIF's uint16 size fields can express — anything larger is a
# broken header, not a picture, and reads as unknown.
_MAX_SNIFFABLE_DIMENSION = 65535


def sniff_dimensions(data: bytes | None) -> tuple[int, int] | None:
    """Return `(width, height)` read from an image's header bytes, or None when
    the format is not one of the four this sniffer knows (PNG, JPEG, GIF,
    WebP), the header is truncated, or the size it declares is not plausible.

    Header-only by design: these four formats all declare their size in the
    first few dozen bytes, so no decoder and therefore no new dependency (no
    Pillow, no imagesize) is needed — and a partial capture still measures. A
    zero dimension is treated as unknown, since a 0-pixel image is a malformed
    header rather than a real size, and so is anything above
    `_MAX_SNIFFABLE_DIMENSION`. None here is not a failure: it degrades the
    descriptor to `[image]` and the token estimate to "unknown", which is the
    honest answer — and the single guard fixes both the number and the text,
    since both are derived from what this function returns."""
    if not data:
        return None
    for sniff in (_png_size, _gif_size, _jpeg_size, _webp_size):
        try:
            size = sniff(data)
        except (IndexError, ValueError):  # a truncated header, not a crash
            size = None
        if size and all(0 < side <= _MAX_SNIFFABLE_DIMENSION for side in size):
            return size
    return None


# --- provider vision-token formulas ----------------------------------------
#
# Every formula below is the provider's own PUBLISHED cost model, applied to
# the sniffed dimensions. They are estimates and are labeled as such: the real
# bill depends on the model, on server-side resampling, and (for OpenAI) on a
# per-model multiplier this deliberately does not try to guess. What they do
# guarantee is the right ORDER OF MAGNITUDE — a few hundred tokens for a
# screenshot, not the fifty thousand the base64 blob used to report.
#
# All arithmetic is integer. `a * b // c` and `(a + b - 1) // c` (ceiling
# division) behave identically in Python and in JS's `Math.floor`, so the two
# SDKs cannot disagree on a rounding boundary.


def _scale_to_fit(width: int, height: int, longest: int) -> tuple[int, int]:
    """Shrink `(width, height)` proportionally so the LONGER side is at most
    `longest`, leaving a smaller image untouched. Integer floor division, with
    a floor of 1 px so an extreme aspect ratio cannot produce a zero side."""
    longest_side = max(width, height)
    if longest_side <= longest:
        return (width, height)
    return (max(1, width * longest // longest_side),
            max(1, height * longest // longest_side))


def _scale_shortest_to(width: int, height: int, shortest: int) -> tuple[int, int]:
    """Shrink `(width, height)` proportionally so the SHORTER side is at most
    `shortest`. Only ever downscales — OpenAI's second scaling step never
    enlarges an image that is already small enough."""
    shortest_side = min(width, height)
    if shortest_side <= shortest:
        return (width, height)
    return (max(1, width * shortest // shortest_side),
            max(1, height * shortest // shortest_side))


def _ceil_div(a: int, b: int) -> int:
    """Ceiling division for non-negative integers, without floats."""
    return (a + b - 1) // b


# OpenAI's published tiling model for the GPT-4o/4.1 vision family: a flat base
# cost, plus a per-512px-tile cost, after the image is fitted into 2048×2048
# and then reduced so its shortest side is 768. `detail: "low"` skips tiling
# entirely and always costs the base+one-tile-equivalent flat rate of 85.
_OPENAI_LOW_DETAIL_TOKENS = 85
_OPENAI_BASE_TOKENS = 85
_OPENAI_TILE_TOKENS = 170
_OPENAI_TILE_PX = 512


def _openai_image_tokens(width: int, height: int, detail: str | None) -> int:
    """OpenAI's documented vision cost: 85 flat for `detail: "low"`; otherwise
    85 + 170 per 512×512 tile, counted after fitting the image into a 2048×2048
    box and then scaling so its shortest side is 768px.

    `"auto"`, an unrecognized value and an absent `detail` are all treated as
    high detail — that is what the API itself does for any image large enough
    to matter, and over-reporting a small image is far less damaging than
    silently under-reporting a screenshot."""
    if detail == "low":
        return _OPENAI_LOW_DETAIL_TOKENS
    width, height = _scale_to_fit(width, height, 2048)
    width, height = _scale_shortest_to(width, height, 768)
    tiles = _ceil_div(width, _OPENAI_TILE_PX) * _ceil_div(height, _OPENAI_TILE_PX)
    return _OPENAI_BASE_TOKENS + _OPENAI_TILE_TOKENS * tiles


# Anthropic publishes a single closed-form approximation — tokens ≈ (w × h) /
# 750 — and states that images with a long edge over 1568px are resized down
# before that is applied.
_ANTHROPIC_PIXELS_PER_TOKEN = 750
_ANTHROPIC_MAX_EDGE = 1568


def _anthropic_image_tokens(width: int, height: int) -> int:
    """Anthropic's documented approximation: resize so the longest edge is at
    most 1568px, then charge one token per 750 pixels, rounded up (an image is
    never free)."""
    width, height = _scale_to_fit(width, height, _ANTHROPIC_MAX_EDGE)
    return max(1, _ceil_div(width * height, _ANTHROPIC_PIXELS_PER_TOKEN))


# Gemini's published model: an image fitting inside 384×384 is a flat 258
# tokens; anything larger is cropped into tiles of `min(w,h)/1.5` px, clamped
# to [256, 768], each tile costing the same 258.
_GEMINI_TILE_TOKENS = 258
_GEMINI_SMALL_EDGE = 384
_GEMINI_MIN_TILE = 256
_GEMINI_MAX_TILE = 768


def _gemini_image_tokens(width: int, height: int) -> int:
    """Gemini's documented tiling: 258 tokens flat when both sides are ≤384px;
    otherwise the image is cut into tiles whose side is the shorter dimension
    divided by 1.5 (clamped to 256..768), and each tile costs 258.

    The `/1.5` is computed as `* 2 // 3` so it is exact integer arithmetic in
    both SDKs rather than a float that could round differently."""
    if width <= _GEMINI_SMALL_EDGE and height <= _GEMINI_SMALL_EDGE:
        return _GEMINI_TILE_TOKENS
    tile = min(width, height) * 2 // 3
    tile = min(max(tile, _GEMINI_MIN_TILE), _GEMINI_MAX_TILE)
    tiles = _ceil_div(width, tile) * _ceil_div(height, tile)
    return _GEMINI_TILE_TOKENS * tiles


def estimate_image_tokens(provider: str, width: int | None, height: int | None,
                          detail: str | None) -> int:
    """Estimate what one image costs the provider, in tokens. Returns 0 when
    the cost genuinely cannot be known — an image whose dimensions we refused
    to fetch, or a format the sniffer does not recognize.

    Zero rather than a guess, deliberately. A fabricated number would be
    indistinguishable from a measured one in every view, whereas a zero shows
    up immediately as a gap between the call's block total and the provider's
    reported `usage`, which is exactly the signal a user should act on. The one
    dimension-free case that IS knowable is OpenAI's `detail: "low"`, which is
    a flat 85 tokens by definition and needs no pixels.

    Provider dispatch: `anthropic` and `bedrock` share Anthropic's formula (a
    Bedrock Converse image is overwhelmingly headed for a Claude model);
    `gemini` uses Gemini's tiling; everything else — `openai`, Azure OpenAI and
    the OpenAI-compatible OSS endpoints that speak the same wire format — uses
    OpenAI's tiling."""
    if width is None or height is None:
        return _OPENAI_LOW_DETAIL_TOKENS if (provider == "openai" and detail == "low") else 0
    if provider in ("anthropic", "bedrock"):
        return _anthropic_image_tokens(width, height)
    if provider == "gemini":
        return _gemini_image_tokens(width, height)
    return _openai_image_tokens(width, height, detail)


# --- the descriptor ---------------------------------------------------------


def format_token_estimate(tokens: int) -> str:
    """Render a token estimate compactly for the descriptor: exact below 1000
    (`765`), one decimal of thousands above it (`1.1k`), with a bare `k` when
    the decimal is zero (`2k`, not `2.0k`).

    Rounding is done as `floor(n / 100 + 0.5)` on tenths-of-a-thousand rather
    than with a language rounding function, because Python's `round()` is
    banker's rounding and JS's `Math.round()` is half-up — they disagree on
    exactly the .5 cases this would hit."""
    if tokens < 1000:
        return str(tokens)
    tenths = int(tokens / 100 + 0.5)
    whole, remainder = divmod(tenths, 10)
    return f"{whole}k" if remainder == 0 else f"{whole}.{remainder}k"


def format_image_text(width: int | None, height: int | None, tokens: int) -> str:
    """The block text that stands in for the image: `[image 1024×768 · ~765
    tok]`.

    Degrades one field at a time, so the descriptor always says exactly as much
    as is actually known: `[image · ~85 tok]` when the cost is known but the
    size is not (OpenAI `detail: "low"` on a remote URL), and a bare `[image]`
    when neither is. The `~` is not decoration — it is the same "this is an
    estimate" claim the block's `token_method` makes, restated where a human
    reads it."""
    size = f"{width}{_TIMES}{height}" if width and height else None
    cost = f"~{format_token_estimate(tokens)} tok" if tokens else None
    if size and cost:
        return f"[image {size} {_DOT} {cost}]"
    if size:
        return f"[image {size}]"
    if cost:
        return f"[image {_DOT} {cost}]"
    return "[image]"


def image_hash_input(ref: ImageRef) -> str:
    """The string hashed (with role and kind) to give an image block its
    identity: a payload term, then the terms that change what the payload COSTS
    or DOES.

        image:sha256:<hex>[;detail=<detail>][;part=<stable-json>]

    Bytes first: `image:sha256:<hex>` over the raw image bytes means the SAME
    picture is ONE block no matter how it was wrapped — a data URI in an OpenAI
    request, a base64 source in an Anthropic one, `inline_data` in a Gemini
    one, sent once or re-sent on every turn of a long agent loop. That is what
    makes an image dedup like any other block, and what makes "this screenshot
    has been in context for 12 turns" a question the diff can answer.

    Without bytes we fall back to the reference itself — the URL, or the
    provider-side file id — which still dedups a repeated reference to the same
    remote image. The `image:` prefix and the explicit `sha256:` / `url:` /
    `ref:` tags keep the three namespaces from ever colliding with each other
    (or with a plain text block that happens to spell a hex digest).

    THE PIXELS ARE NOT THE WHOLE REQUEST, though, and the two suffixes are the
    difference between an image and a request FOR an image:

      * `;detail=` — OpenAI's fidelity hint changes the cost nine-fold (85 at
        `"low"`, 765 at `"high"` for a 1024×768 screenshot) and changes the
        descriptor with it. Leaving it out collapsed the standard computer-use
        pattern — the same screenshot at `"low"` in history and `"high"` for
        the current turn — into one block, where `INSERT OR IGNORE` kept the
        first count written and the diff called a 9× cost change "unchanged".
      * `;part=` — everything else the content part carried (see
        `_part_modifiers`), chiefly Anthropic's `cache_control`. A cache
        breakpoint moving on or off an image is precisely what the cache
        profiler exists to report; with it outside the identity the hash did
        not move and the profiler called the prefix stable.

    Both suffixes are absent for the common case, so the documented shapes
    still dedup byte for byte across providers and wrappers."""
    if ref.data is not None:
        base = "image:sha256:" + hashlib.sha256(ref.data).hexdigest()
    elif ref.url:
        base = "image:url:" + ref.url
    elif ref.ref:
        base = "image:ref:" + ref.ref
    else:
        base = "image:unknown"
    if ref.detail:
        base += ";detail=" + ref.detail
    if ref.modifiers:
        base += ";part=" + ref.modifiers
    return base


# --- provider part shapes ---------------------------------------------------


def _str_or_none(value: object) -> str | None:
    """Return `value` when it is a non-empty string, else None — so a
    malformed payload (a number where a URL belongs) degrades instead of
    propagating a wrong type into the hash."""
    return value if isinstance(value, str) and value else None


def _from_url(url: str, detail: str | None) -> ImageRef:
    """Build an ImageRef from a URL that is either a `data:` URI (inline bytes
    we can hash and sniff) or a remote address we will NOT fetch."""
    data, media_type = _parse_data_uri(url)
    if data is not None:
        return ImageRef(data=data, detail=detail, media_type=media_type)
    return ImageRef(url=url, detail=detail, media_type=media_type)


def _openai_chat_image(part: dict) -> ImageRef | None:
    """OpenAI Chat Completions: `{"type": "image_url", "image_url": {"url":
    ..., "detail": "high"}}`. The nested object is the documented shape; a bare
    string under `image_url` is accepted too, because hand-rolled clients and
    older examples emit it."""
    image_url = part.get("image_url")
    if isinstance(image_url, str):
        return _from_url(image_url, None)
    if isinstance(image_url, dict):
        url = _str_or_none(image_url.get("url"))
        detail = _str_or_none(image_url.get("detail"))
        if url:
            return _from_url(url, detail)
        file_id = _str_or_none(image_url.get("file_id"))
        if file_id:
            return ImageRef(ref=file_id, detail=detail)
    return None


def _openai_responses_image(part: dict) -> ImageRef | None:
    """OpenAI Responses: `{"type": "input_image", "image_url": "data:...",
    "detail": "high"}` — note `image_url` is a plain STRING here, unlike Chat
    Completions — or `{"type": "input_image", "file_id": "file-..."}` for an
    image already uploaded to the Files API (no bytes reachable locally)."""
    detail = _str_or_none(part.get("detail"))
    url = _str_or_none(part.get("image_url"))
    if url:
        return _from_url(url, detail)
    if isinstance(part.get("image_url"), dict):  # defensive: chat shape reused
        nested = _str_or_none(part["image_url"].get("url"))
        if nested:
            return _from_url(nested, detail)
    file_id = _str_or_none(part.get("file_id"))
    if file_id:
        return ImageRef(ref=file_id, detail=detail)
    return None


def _anthropic_image(part: dict) -> ImageRef | None:
    """Anthropic Messages: `{"type": "image", "source": {...}}`, where the
    source is one of three documented shapes — `{"type": "base64",
    "media_type": "image/png", "data": "<b64>"}`, the newer `{"type": "url",
    "url": "https://..."}`, or `{"type": "file", "file_id": "..."}` for the
    Files API. A source missing its `type` discriminator is read by which key
    it carries, so a slightly-off payload still records as an image."""
    source = part.get("source")
    if not isinstance(source, dict):
        return None
    media_type = _str_or_none(source.get("media_type"))
    data = _b64_decode(source.get("data"))
    if data is not None:
        return ImageRef(data=data, media_type=media_type)
    url = _str_or_none(source.get("url"))
    if url:
        return _from_url(url, None)
    file_id = _str_or_none(source.get("file_id"))
    if file_id:
        return ImageRef(ref=file_id, media_type=media_type)
    return None


def _is_image_media_type(media_type: str | None) -> bool:
    """Whether a declared MIME type names an image. Used to keep Gemini's
    generic `inline_data`/`file_data` carriers — which also transport audio,
    video and PDFs — from being rewritten as image blocks."""
    return bool(media_type) and media_type.lower().startswith("image/")


def _gemini_inline_image(inline: dict) -> ImageRef | None:
    """Gemini `inline_data` / `inlineData`: `{"mime_type": "image/png",
    "data": "<b64 or bytes>"}` (the JS SDK spells the key `mimeType`).

    Gated on the MIME type because the same carrier also delivers audio, video
    and PDFs; a non-image keeps the pre-existing JSON-serialization path
    untouched. When the MIME type is absent entirely, the header sniffer
    decides — if the bytes really are a PNG/JPEG/GIF/WebP we treat them as the
    image they are, otherwise we leave the part alone."""
    media_type = _str_or_none(inline.get("mime_type")) or _str_or_none(inline.get("mimeType"))
    data = _b64_decode(inline.get("data"))
    if data is None:
        return None
    if media_type is None:
        return ImageRef(data=data) if sniff_dimensions(data) else None
    if not _is_image_media_type(media_type):
        return None
    return ImageRef(data=data, media_type=media_type)


def _gemini_file_image(file_data: dict) -> ImageRef | None:
    """Gemini `file_data` / `fileData`: `{"mime_type": "image/jpeg",
    "file_uri": "https://generativelanguage.googleapis.com/..."}` (`mimeType` /
    `fileUri` in the JS SDK). A URI only — never fetched — so this degrades to
    `[image]`. Requires an image MIME type, since the same shape carries video
    and PDFs and there are no bytes to sniff."""
    media_type = _str_or_none(file_data.get("mime_type")) or _str_or_none(file_data.get("mimeType"))
    if not _is_image_media_type(media_type):
        return None
    uri = _str_or_none(file_data.get("file_uri")) or _str_or_none(file_data.get("fileUri"))
    if not uri:
        return None
    return ImageRef(ref=uri, media_type=media_type)


def _bedrock_image(image: dict) -> ImageRef | None:
    """Bedrock Converse: `{"image": {"format": "png", "source": {"bytes":
    b"..."}}}`. botocore hands `bytes` directly; a base64 string is accepted
    too for hand-built payloads. An `s3Location` source names an object in the
    caller's bucket that we will not read, so it degrades to a reference."""
    source = image.get("source")
    if not isinstance(source, dict):
        return None
    fmt = _str_or_none(image.get("format"))
    media_type = f"image/{fmt.lower()}" if fmt else None
    data = _b64_decode(source.get("bytes"))
    if data is not None:
        return ImageRef(data=data, media_type=media_type)
    s3 = source.get("s3Location")
    if isinstance(s3, dict):
        uri = _str_or_none(s3.get("uri"))
        if uri:
            return ImageRef(ref=uri, media_type=media_type)
    return None


def _detect_shape(part: dict) -> ImageRef | None:
    """Dispatch one content part to the provider shape that owns it, returning
    what the payload itself says. The shapes are disjoint on their
    discriminator key (`type: image_url` / `input_image` / `image`, or the
    presence of `inline_data` / `file_data` / `image`), so a single dispatch is
    unambiguous — and a shared detector is the only way the two SDKs, five
    providers and the golden harness can be guaranteed to agree on what counts
    as an image. `detect_image_part` wraps this to add the part's remainder."""
    part_type = part.get("type")
    if part_type == "image_url" or ("image_url" in part and part_type is None):
        return _openai_chat_image(part)
    if part_type == "input_image":
        return _openai_responses_image(part)
    if part_type == "image" and isinstance(part.get("source"), dict):
        return _anthropic_image(part)
    for key in ("inline_data", "inlineData"):
        if isinstance(part.get(key), dict):
            return _gemini_inline_image(part[key])
    for key in ("file_data", "fileData"):
        if isinstance(part.get(key), dict):
            return _gemini_file_image(part[key])
    if isinstance(part.get("image"), dict):
        return _bedrock_image(part["image"])
    return None


# Keys the shape functions above have ALREADY accounted for. Each is either the
# payload itself (`data`/`url`/`bytes`/`file_id`/`file_uri`/`s3Location`), the
# discriminator that selected the shape (`type`, `format`), a hint promoted
# into the ImageRef (`detail`), or a field deliberately excluded from identity
# so the same picture dedups however it was labeled (the media type).
_ACCOUNTED_KEYS = frozenset({
    "type", "detail", "url", "data", "bytes", "file_id", "file_uri", "fileUri",
    "media_type", "mime_type", "mimeType", "format", "s3Location",
})

# Keys whose VALUE is one of the nested carriers a shape wraps its payload in,
# and therefore the only places worth looking one level deeper. Recursion is
# restricted to these on purpose: `cache_control: {"type": "ephemeral"}` must
# keep its own `type`, which is a cache mode and not a part discriminator.
_CARRIER_KEYS = frozenset({
    "image_url", "source", "inline_data", "inlineData", "file_data", "fileData",
    "image",
})


def _is_json_safe(value: object) -> bool:
    """Whether `value` is something both SDKs serialize identically — strings,
    numbers, booleans, null, and lists/dicts of those.

    Anything else (raw `bytes` under an unrecognized key, an SDK object) is
    dropped from the remainder rather than serialized: Python's `json.dumps`
    raises on it while JS's `stableStringify` would happily render a byte array
    as `{"0": 137, …}`, and a hash that depends on which SDK captured the call
    is worse than a hash that ignores an exotic value."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json_safe(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_safe(v) for k, v in value.items())
    return False


def _unaccounted_keys(obj: dict) -> dict:
    """Return the part's remainder: everything `_detect_shape` did NOT consume,
    with the nested carriers descended into and pruned when they empty out.

    The rule is "we understand it, or it counts": every key ctxdiff reads is in
    `_ACCOUNTED_KEYS` and is already represented in the block, so whatever is
    left is a modifier ctxdiff cannot interpret but the provider can — and a
    provider-side behavior change that leaves no trace in the trace is the one
    outcome a context debugger must not produce."""
    out: dict = {}
    for key, value in obj.items():
        if not isinstance(key, str) or key in _ACCOUNTED_KEYS:
            continue
        if key in _CARRIER_KEYS:
            if isinstance(value, dict):
                nested = _unaccounted_keys(value)
                if nested:
                    out[key] = nested
            continue  # a non-dict carrier IS the payload (a bare `image_url`)
        if _is_json_safe(value):
            out[key] = value
    return out


def _part_modifiers(part: dict) -> str | None:
    """The remainder of a content part as a stable, sorted-key JSON string, or
    None when there is none.

    None for every documented shape — an OpenAI `image_url`, an Anthropic
    `source`, a Gemini `inline_data`, a Bedrock `image` — which is what keeps
    cross-provider dedup intact. Non-None exactly when the part carried
    something extra, the motivating case being `cache_control`:

        {"type": "image", "source": {…}, "cache_control": {"type": "ephemeral"}}
        ->  '{"cache_control": {"type": "ephemeral"}}'

    Serialized through the same `normalize_text` every other block hashes with,
    so the JS twin produces the identical bytes."""
    remainder = _unaccounted_keys(part)
    return normalize_text(remainder) if remainder else None


def detect_image_part(part: object) -> ImageRef | None:
    """Recognize one content part as an image across every provider shape
    ctxdiff captures, returning what could be learned about it without I/O — or
    None when the part is not an image, in which case the caller keeps its
    existing behavior verbatim.

    Two steps: `_detect_shape` reads the payload, then the part's unaccounted
    remainder is attached, so identity covers the whole content part and not
    just the pixels inside it. Splitting them keeps the per-provider readers
    free of any knowledge of the remainder rule.

    Never raises: a malformed part returns None and falls back to the previous
    JSON-serialization path, which is always a safe (if verbose) answer."""
    if not isinstance(part, dict):
        return None
    try:
        ref = _detect_shape(part)
        if ref is None:
            return None
        return replace(ref, modifiers=_part_modifiers(part))
    except Exception:  # noqa: BLE001 — a weird payload must never break capture
        return None


def image_block_fields(ref: ImageRef, provider: str) -> tuple[str, str, int, str]:
    """Turn a detected image into the four fields a block needs:
    `(text, hash_input, token_count, token_method)`.

    Kept separate from `detect_image_part` so the golden harness and the tests
    can drive the measurement half directly, with an ImageRef they built
    themselves, without going through a provider payload."""
    width, height = sniff_dimensions(ref.data) or (None, None)
    tokens = estimate_image_tokens(provider, width, height, ref.detail)
    return (format_image_text(width, height, tokens), image_hash_input(ref),
            tokens, IMAGE_TOKEN_METHOD)


def image_raw_block(role: str, part: object, provider: str) -> RawBlock | None:
    """THE adapter entry point: if `part` is an image in any provider shape,
    return the `RawBlock` that should stand in for it; otherwise None, meaning
    "not an image — do whatever you did before".

    Every adapter calls exactly this, first, in its content-part loop. Routing
    all five providers through one function is what makes the block a vision
    request produces independent of which SDK captured it: the same picture in
    an OpenAI request and in an Anthropic one produces the same hash, the same
    descriptor and (per that provider's published formula) an honest cost."""
    ref = detect_image_part(part)
    if ref is None:
        return None
    text, hash_input, tokens, method = image_block_fields(ref, provider)
    return RawBlock(role=role, kind=IMAGE_KIND, text=text, hash_input=hash_input,
                    token_count=tokens, token_method=method)
