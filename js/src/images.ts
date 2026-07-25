/**
 * Image content parts — detection, header sniffing, vision-token estimation
 * and the one-line descriptor that stands in for the pixels.
 *
 * WHY THIS EXISTS. Before this module, an image content part reached the block
 * model through the same path as any other non-string part: stable-JSON of the
 * whole part. For `{type: "image_url", image_url: {url: "data:image/png;base64,
 * <100k chars>"}}` that meant the base64 blob became the block's `text` —
 * tokenized as if it were prose (tens of thousands of phantom tokens for an
 * image that really costs a few hundred), and written verbatim into the
 * `.ctrace`. Token attribution was therefore wrong for exactly the
 * vision/computer-use agents that most need context auditing.
 *
 * WHAT IT DOES INSTEAD. An image part becomes a block whose
 *
 *   - `kind` is `"image"` (a distinct kind, so readers can tell it apart);
 *   - `text` is a short descriptor — `[image 1024×768 · ~765 tok]` — never the
 *     bytes;
 *   - identity is sha256 over the image BYTES (see `imageHashInput`), so the
 *     same picture dedups across turns, across sessions and even across
 *     providers;
 *   - `tokenCount` is the provider's DOCUMENTED vision-token formula applied to
 *     the sniffed dimensions, always marked `tokenMethod: "estimate"` — a
 *     vision cost is never reported as an exact tiktoken count.
 *
 * LOCAL-FIRST, ALWAYS. Nothing here ever performs I/O. A remote `http(s)` image
 * URL is identified by its URL and degrades to `[image]` with no token
 * estimate; fetching it would turn a local debugging tool into a network
 * client, leak the trace subject's URLs, and make the numbers depend on whether
 * the host was online. Same for a provider-side file id.
 *
 * No new dependency: dimensions are read from the first bytes of the file (PNG
 * IHDR, JPEG SOFn, GIF screen descriptor, WebP VP8/VP8L/VP8X) by the small
 * sniffer below.
 *
 * PARITY: every function here has a byte-identical twin in `src/ctxdiff/
 * images.py`. Same descriptor text, same hash input, same token numbers. All
 * arithmetic is integer-only (`Math.floor`, never a bare `/` that survives into
 * a result) precisely so the two languages cannot round apart.
 */
import { createHash } from "node:crypto";
import { normalizeText } from "./models.js";
import type { RawBlock } from "./models.js";

/**
 * The kind stored on an image block. Additive: readers that predate this kind
 * see an ordinary block whose text is the descriptor, which is exactly what we
 * want them to show.
 */
export const IMAGE_KIND = "image";

/**
 * Image blocks are ALWAYS estimates, and they reuse the existing `"estimate"`
 * marker rather than inventing a new one. Every reader already written —
 * including an older ctxdiff opening a file this SDK writes today — tests
 * `tokenMethod === "estimate"` to decide whether to print the "~est" marker
 * (see `analyze/tokens.ts` and the viewer template). A new value would slip past
 * those tests and render a vision estimate as if it were exact, which is the one
 * thing this module exists to prevent.
 */
export const IMAGE_TOKEN_METHOD = "estimate";

// The multiplication sign and middle dot used in the descriptor. Named so the
// Python twin can be compared against them literally.
const TIMES = "×"; // ×
const DOT = "·"; // ·

/**
 * What an adapter could learn about one image part WITHOUT doing any I/O.
 *
 * Exactly one of `data` / `url` / `ref` is normally set, in that order of
 * preference: `data` when the part carried the bytes inline (a data URI, an
 * Anthropic base64 source, a Gemini `inlineData`, a Bedrock image source), `url`
 * when it referenced a remote image we deliberately will not fetch, and `ref`
 * when it named a provider-side object (an OpenAI `file_id`, a Gemini
 * `fileUri`) that only the provider can resolve.
 *
 * `detail` is OpenAI's per-image fidelity hint (`"low"` / `"high"` / `"auto"`);
 * it changes the token cost — 85 vs 765 for the same 1024×768 screenshot — so
 * it is part of the identity (see `imageHashInput`) and is null for providers
 * with no such concept. `mediaType` is carried for diagnostics only — it is
 * deliberately NOT part of the descriptor or the hash.
 *
 * `modifiers` is the stable-JSON of everything ELSE the content part carried
 * (see `partModifiers`): the keys ctxdiff does not interpret but the provider
 * does, chiefly Anthropic's `cache_control`. It is filled in by
 * `detectImagePart` and is null both when the part had no leftovers (every
 * documented shape) and when an ImageRef was built directly by a caller.
 */
export interface ImageRef {
  data: Uint8Array | null;
  url: string | null;
  ref: string | null;
  detail: string | null;
  mediaType: string | null;
  modifiers: string | null;
}

/** Build an ImageRef with every field defaulted, so callers only name what they
 * actually learned. Mirrors the Python dataclass's field defaults. */
function makeRef(fields: Partial<ImageRef>): ImageRef {
  return {
    data: fields.data ?? null,
    url: fields.url ?? null,
    ref: fields.ref ?? null,
    detail: fields.detail ?? null,
    mediaType: fields.mediaType ?? null,
    modifiers: fields.modifiers ?? null,
  };
}

function isRecord(x: unknown): x is Record<string, unknown> {
  return typeof x === "object" && x !== null && !Array.isArray(x);
}

// --- base64 / data URI ------------------------------------------------------

const NON_B64 = /[^A-Za-z0-9+/]/g;

/**
 * Decode base64 that came off the wire, or return null when there are no bytes
 * to be had.
 *
 * Tolerant on purpose: SDK payloads arrive with missing `=` padding, embedded
 * newlines (from a shell `base64` invocation) and occasionally in the URL-safe
 * alphabet. Each of those is a formatting difference, not a different image. A
 * `Uint8Array`/`Buffer` passes straight through — some SDK call sites hand raw
 * bytes rather than base64.
 *
 * The normalization is spelled out step by step rather than left to the two
 * languages' decoders, because those decoders disagree on malformed input and
 * the bytes ARE the block's identity — a divergence here would be a cross-SDK
 * hash divergence. So: map the URL-safe alphabet back, DELETE every character
 * outside the base64 alphabet (including all `=`), drop a lone trailing
 * character that cannot encode a whole byte, then re-pad. Both SDKs perform
 * exactly these steps, so both see the same input by the time a decoder is
 * involved.
 *
 * Returning null — for a non-string, an undecodable string, or anything that
 * decodes to zero bytes — is what makes the caller fall back to the old
 * JSON-serialization path. That matters: without the empty-bytes guard, every
 * malformed image in a trace would share the digest of the empty string and
 * dedup into a single bogus block.
 */
function b64Decode(value: unknown): Uint8Array | null {
  if (value instanceof Uint8Array) return value.length > 0 ? value : null;
  if (typeof value !== "string" || value.length === 0) return null;
  let cleaned = value.replace(/-/g, "+").replace(/_/g, "/").replace(NON_B64, "");
  if (cleaned.length % 4 === 1) cleaned = cleaned.slice(0, -1);
  cleaned += "=".repeat((4 - (cleaned.length % 4)) % 4);
  if (cleaned.length === 0) return null;
  try {
    const buf = Buffer.from(cleaned, "base64");
    return buf.length > 0 ? new Uint8Array(buf) : null;
  } catch {
    return null;
  }
}

/**
 * Split a `data:` URI into [bytes, media type], or [null, null] when it is not
 * one. Handles the only form providers emit — `data:<mime>;base64,<b64>` — and
 * lowercases the media type. A non-base64 data URI (percent-encoded text)
 * yields no bytes; there is no such thing as a percent-encoded image in these
 * APIs, and guessing would be worse than degrading.
 */
function parseDataUri(url: string): [Uint8Array | null, string | null] {
  if (!url.startsWith("data:")) return [null, null];
  const rest = url.slice(5);
  const comma = rest.indexOf(",");
  if (comma < 0) return [null, null];
  const params = rest.slice(0, comma).split(";");
  const mediaType = params[0].trim().toLowerCase() || null;
  const isBase64 = params.slice(1).some((p) => p.trim().toLowerCase() === "base64");
  if (!isBase64) return [null, mediaType];
  return [b64Decode(rest.slice(comma + 1)), mediaType];
}

// --- header sniffing --------------------------------------------------------

/** Big-endian uint32 at `off` — PNG's byte order. */
function be32(d: Uint8Array, off: number): number {
  return ((d[off] << 24) | (d[off + 1] << 16) | (d[off + 2] << 8) | d[off + 3]) >>> 0;
}

/** Little-endian uint16 at `off` — GIF's and WebP's byte order. */
function le16(d: Uint8Array, off: number): number {
  return d[off] | (d[off + 1] << 8);
}

/** Whether `d` starts (at `off`) with the given ASCII tag. */
function tagAt(d: Uint8Array, off: number, tag: string): boolean {
  for (let i = 0; i < tag.length; i++) {
    if (d[off + i] !== tag.charCodeAt(i)) return false;
  }
  return true;
}

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

/**
 * PNG: the 8-byte signature is followed immediately by the IHDR chunk, whose
 * first two fields are width and height as big-endian uint32 at fixed offsets 16
 * and 20. Requiring the literal `IHDR` tag rejects a file that merely starts
 * with the signature.
 */
function pngSize(d: Uint8Array): [number, number] | null {
  if (d.length < 24) return null;
  for (let i = 0; i < 8; i++) if (d[i] !== PNG_SIGNATURE[i]) return null;
  if (!tagAt(d, 12, "IHDR")) return null;
  return [be32(d, 16), be32(d, 20)];
}

/**
 * GIF: after the 6-byte `GIF87a`/`GIF89a` header comes the logical screen
 * descriptor, whose first four bytes are width then height, little-endian.
 */
function gifSize(d: Uint8Array): [number, number] | null {
  if (d.length < 10) return null;
  if (!tagAt(d, 0, "GIF87a") && !tagAt(d, 0, "GIF89a")) return null;
  return [le16(d, 6), le16(d, 8)];
}

// JPEG markers that are NOT start-of-frame even though they fall in the
// 0xC0..0xCF range: DHT (define Huffman table), JPG (reserved), DAC (define
// arithmetic coding). Everything else in that range is an SOFn carrying the
// frame dimensions.
const JPEG_NON_SOF = new Set([0xc4, 0xc8, 0xcc]);

/**
 * How far into a payload the marker walk will look for the start-of-frame.
 *
 * WHY A BOUND AT ALL. A well-formed JPEG is walked segment by segment (`i += 2 +
 * segLen`), so it reaches SOF in a handful of iterations however large the file
 * is. A payload that merely BEGINS `FF D8` — a truncated download, a corrupt
 * screenshot, a fuzzer's output — desynchronizes and degrades to byte-at-a-time
 * resync over the whole buffer. `extractBlocks` runs synchronously on the host
 * application's thread during capture, so that is the traced agent stalling,
 * which violates the "never break the host app" guardrail.
 *
 * WHY 1 MiB. The bound must sit above the largest legitimate run of pre-SOF
 * metadata. A single JPEG segment is capped at 65533 bytes by the format, but a
 * file may chain them: a full EXIF APP1 plus a multi-segment APP2 ICC profile is
 * the realistic worst case and lands well under 1 MiB. Past the bound the
 * sniffer reports "unknown" — an honest `[image]` — rather than scanning on.
 */
export const JPEG_MAX_SCAN = 1 << 20;

/**
 * JPEG: walk the marker segments from SOI until a start-of-frame (SOFn), whose
 * payload holds precision, then height and width as big-endian uint16.
 *
 * How the walk works: every marker is `0xFF <code>`; standalone markers (SOI,
 * EOI, RSTn, TEM) carry no length, all others are followed by a 2-byte segment
 * length that INCLUDES those two bytes. Fill bytes (runs of `0xFF`) are skipped.
 * The walk stops at `JPEG_MAX_SCAN` (see above) rather than at the end of the
 * buffer, so the work is bounded by a constant instead of by the size of a
 * possibly-adversarial payload; the SOF payload itself is still read against the
 * real length, so a frame header found just before the bound is not lost to it.
 */
function jpegSize(d: Uint8Array): [number, number] | null {
  const n = d.length;
  if (n < 4 || d[0] !== 0xff || d[1] !== 0xd8) return null;
  const limit = Math.min(n, JPEG_MAX_SCAN);
  let i = 2;
  while (i + 3 < limit) {
    if (d[i] !== 0xff) {
      i += 1; // desynchronized — resync on the next 0xFF
      continue;
    }
    const marker = d[i + 1];
    if (marker === 0xff) {
      i += 1; // fill byte
      continue;
    }
    if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd9)) {
      i += 2; // standalone, no length
      continue;
    }
    const segLen = (d[i + 2] << 8) | d[i + 3];
    if (segLen < 2) return null;
    if (marker >= 0xc0 && marker <= 0xcf && !JPEG_NON_SOF.has(marker)) {
      if (i + 9 >= n) return null;
      const height = (d[i + 5] << 8) | d[i + 6];
      const width = (d[i + 7] << 8) | d[i + 8];
      return width && height ? [width, height] : null;
    }
    i += 2 + segLen;
  }
  return null;
}

/**
 * WebP: a RIFF container whose first chunk identifies one of three encodings,
 * each storing the canvas size differently.
 *
 *   - `VP8 ` (lossy) — a 3-byte frame tag, then the 3-byte start code
 *     `9D 01 2A`, then width and height as 14-bit little-endian fields.
 *   - `VP8L` (lossless) — signature byte `0x2F`, then 14 bits of width-1 and 14
 *     bits of height-1 packed into a little-endian uint32.
 *   - `VP8X` (extended: animation/alpha/ICC) — canvas width-1 and height-1 as
 *     two 24-bit little-endian fields.
 */
function webpSize(d: Uint8Array): [number, number] | null {
  if (d.length < 30 || !tagAt(d, 0, "RIFF") || !tagAt(d, 8, "WEBP")) return null;
  if (tagAt(d, 12, "VP8 ")) {
    if (d[23] !== 0x9d || d[24] !== 0x01 || d[25] !== 0x2a) return null;
    return [le16(d, 26) & 0x3fff, le16(d, 28) & 0x3fff];
  }
  if (tagAt(d, 12, "VP8L")) {
    if (d[20] !== 0x2f) return null;
    const bits = (d[21] | (d[22] << 8) | (d[23] << 16) | (d[24] << 24)) >>> 0;
    return [(bits & 0x3fff) + 1, ((bits >>> 14) & 0x3fff) + 1];
  }
  if (tagAt(d, 12, "VP8X")) {
    return [
      (d[24] | (d[25] << 8) | (d[26] << 16)) + 1,
      (d[27] | (d[28] << 8) | (d[29] << 16)) + 1,
    ];
  }
  return null;
}

/**
 * The largest side length a header is believed when it declares.
 *
 * Headers are read, not verified — the pixels are never decoded — so a corrupt
 * or hostile file declares whatever it likes. PNG's IHDR is a pair of uint32s
 * and WebP's VP8X canvas is 24-bit, so `0xFFFFFFFF × 0xFFFFFFFF` (18 exapixels)
 * is a structurally VALID header, and trusting it produced a block estimated at
 * 8,068,951,256,159,688 tokens: every other block in the run rounds to 0.0% and
 * the run total becomes meaningless. 65535 is the ceiling because it is exactly
 * what JPEG's and GIF's uint16 size fields can express — anything larger is a
 * broken header, not a picture, and reads as unknown.
 */
const MAX_SNIFFABLE_DIMENSION = 65535;

/**
 * Return `[width, height]` read from an image's header bytes, or null when the
 * format is not one of the four this sniffer knows (PNG, JPEG, GIF, WebP), the
 * header is truncated, or the size it declares is not plausible.
 *
 * Header-only by design: these four formats all declare their size in the first
 * few dozen bytes, so no decoder and therefore no new dependency is needed — and
 * a partial capture still measures. A zero dimension is treated as unknown,
 * since a 0-pixel image is a malformed header rather than a real size, and so is
 * anything above `MAX_SNIFFABLE_DIMENSION`. Null here is not a failure: it
 * degrades the descriptor to `[image]` and the token estimate to "unknown",
 * which is the honest answer — and the single guard fixes both the number and
 * the text, since both are derived from what this function returns.
 */
export function sniffDimensions(data: Uint8Array | null): [number, number] | null {
  if (!data || data.length === 0) return null;
  for (const sniff of [pngSize, gifSize, jpegSize, webpSize]) {
    let size: [number, number] | null = null;
    try {
      size = sniff(data);
    } catch {
      size = null; // a truncated header, not a crash
    }
    if (size && size.every((side) => side > 0 && side <= MAX_SNIFFABLE_DIMENSION)) return size;
  }
  return null;
}

// --- provider vision-token formulas ----------------------------------------
//
// Every formula below is the provider's own PUBLISHED cost model, applied to the
// sniffed dimensions. They are estimates and are labeled as such: the real bill
// depends on the model, on server-side resampling, and (for OpenAI) on a
// per-model multiplier this deliberately does not try to guess. What they do
// guarantee is the right ORDER OF MAGNITUDE — a few hundred tokens for a
// screenshot, not the fifty thousand the base64 blob used to report.
//
// All arithmetic is integer. `Math.floor(a * b / c)` and `ceilDiv` behave
// identically to Python's `//` and `(a + b - 1) // b`, so the two SDKs cannot
// disagree on a rounding boundary.

/**
 * Shrink `[width, height]` proportionally so the LONGER side is at most
 * `longest`, leaving a smaller image untouched. Integer floor division, with a
 * floor of 1px so an extreme aspect ratio cannot produce a zero side.
 */
function scaleToFit(width: number, height: number, longest: number): [number, number] {
  const longestSide = Math.max(width, height);
  if (longestSide <= longest) return [width, height];
  return [
    Math.max(1, Math.floor((width * longest) / longestSide)),
    Math.max(1, Math.floor((height * longest) / longestSide)),
  ];
}

/**
 * Shrink `[width, height]` proportionally so the SHORTER side is at most
 * `shortest`. Only ever downscales — OpenAI's second scaling step never enlarges
 * an image that is already small enough.
 */
function scaleShortestTo(width: number, height: number, shortest: number): [number, number] {
  const shortestSide = Math.min(width, height);
  if (shortestSide <= shortest) return [width, height];
  return [
    Math.max(1, Math.floor((width * shortest) / shortestSide)),
    Math.max(1, Math.floor((height * shortest) / shortestSide)),
  ];
}

/** Ceiling division for non-negative integers, without floats. */
function ceilDiv(a: number, b: number): number {
  return Math.floor((a + b - 1) / b);
}

// OpenAI's published tiling model for the GPT-4o/4.1 vision family: a flat base
// cost, plus a per-512px-tile cost, after the image is fitted into 2048×2048 and
// then reduced so its shortest side is 768. `detail: "low"` skips tiling
// entirely and always costs the flat rate of 85.
const OPENAI_LOW_DETAIL_TOKENS = 85;
const OPENAI_BASE_TOKENS = 85;
const OPENAI_TILE_TOKENS = 170;
const OPENAI_TILE_PX = 512;

/**
 * OpenAI's documented vision cost: 85 flat for `detail: "low"`; otherwise 85 +
 * 170 per 512×512 tile, counted after fitting the image into a 2048×2048 box and
 * then scaling so its shortest side is 768px.
 *
 * `"auto"`, an unrecognized value and an absent `detail` are all treated as high
 * detail — that is what the API itself does for any image large enough to
 * matter, and over-reporting a small image is far less damaging than silently
 * under-reporting a screenshot.
 */
function openaiImageTokens(width: number, height: number, detail: string | null): number {
  if (detail === "low") return OPENAI_LOW_DETAIL_TOKENS;
  let [w, h] = scaleToFit(width, height, 2048);
  [w, h] = scaleShortestTo(w, h, 768);
  const tiles = ceilDiv(w, OPENAI_TILE_PX) * ceilDiv(h, OPENAI_TILE_PX);
  return OPENAI_BASE_TOKENS + OPENAI_TILE_TOKENS * tiles;
}

// Anthropic publishes a single closed-form approximation — tokens ≈ (w × h) /
// 750 — and states that images with a long edge over 1568px are resized down
// before that is applied.
const ANTHROPIC_PIXELS_PER_TOKEN = 750;
const ANTHROPIC_MAX_EDGE = 1568;

/**
 * Anthropic's documented approximation: resize so the longest edge is at most
 * 1568px, then charge one token per 750 pixels, rounded up (an image is never
 * free).
 */
function anthropicImageTokens(width: number, height: number): number {
  const [w, h] = scaleToFit(width, height, ANTHROPIC_MAX_EDGE);
  return Math.max(1, ceilDiv(w * h, ANTHROPIC_PIXELS_PER_TOKEN));
}

// Gemini's published model: an image fitting inside 384×384 is a flat 258
// tokens; anything larger is cropped into tiles of `min(w,h)/1.5` px, clamped to
// [256, 768], each tile costing the same 258.
const GEMINI_TILE_TOKENS = 258;
const GEMINI_SMALL_EDGE = 384;
const GEMINI_MIN_TILE = 256;
const GEMINI_MAX_TILE = 768;

/**
 * Gemini's documented tiling: 258 tokens flat when both sides are ≤384px;
 * otherwise the image is cut into tiles whose side is the shorter dimension
 * divided by 1.5 (clamped to 256..768), and each tile costs 258.
 *
 * The `/1.5` is computed as `* 2 / 3` floored so it is exact integer arithmetic
 * in both SDKs rather than a float that could round differently.
 */
function geminiImageTokens(width: number, height: number): number {
  if (width <= GEMINI_SMALL_EDGE && height <= GEMINI_SMALL_EDGE) return GEMINI_TILE_TOKENS;
  let tile = Math.floor((Math.min(width, height) * 2) / 3);
  tile = Math.min(Math.max(tile, GEMINI_MIN_TILE), GEMINI_MAX_TILE);
  const tiles = ceilDiv(width, tile) * ceilDiv(height, tile);
  return GEMINI_TILE_TOKENS * tiles;
}

/**
 * Estimate what one image costs the provider, in tokens. Returns 0 when the cost
 * genuinely cannot be known — an image whose dimensions we refused to fetch, or
 * a format the sniffer does not recognize.
 *
 * Zero rather than a guess, deliberately. A fabricated number would be
 * indistinguishable from a measured one in every view, whereas a zero shows up
 * immediately as a gap between the call's block total and the provider's
 * reported `usage`, which is exactly the signal a user should act on. The one
 * dimension-free case that IS knowable is OpenAI's `detail: "low"`, a flat 85
 * tokens by definition, which needs no pixels.
 *
 * Provider dispatch: `anthropic` and `bedrock` share Anthropic's formula (a
 * Bedrock Converse image is overwhelmingly headed for a Claude model); `gemini`
 * uses Gemini's tiling; everything else — `openai`, Azure OpenAI and the
 * OpenAI-compatible OSS endpoints that speak the same wire format — uses
 * OpenAI's tiling.
 */
export function estimateImageTokens(
  provider: string,
  width: number | null,
  height: number | null,
  detail: string | null,
): number {
  if (width === null || height === null) {
    return provider === "openai" && detail === "low" ? OPENAI_LOW_DETAIL_TOKENS : 0;
  }
  if (provider === "anthropic" || provider === "bedrock") {
    return anthropicImageTokens(width, height);
  }
  if (provider === "gemini") return geminiImageTokens(width, height);
  return openaiImageTokens(width, height, detail);
}

// --- the descriptor ---------------------------------------------------------

/**
 * Render a token estimate compactly for the descriptor: exact below 1000
 * (`765`), one decimal of thousands above it (`1.1k`), with a bare `k` when the
 * decimal is zero (`2k`, not `2.0k`).
 *
 * Rounding is `Math.floor(n / 100 + 0.5)` on tenths-of-a-thousand rather than a
 * language rounding function, because Python's `round()` is banker's rounding
 * and JS's `Math.round()` is half-up — they disagree on exactly the .5 cases
 * this would hit.
 */
export function formatTokenEstimate(tokens: number): string {
  if (tokens < 1000) return String(tokens);
  const tenths = Math.floor(tokens / 100 + 0.5);
  const whole = Math.floor(tenths / 10);
  const remainder = tenths % 10;
  return remainder === 0 ? `${whole}k` : `${whole}.${remainder}k`;
}

/**
 * The block text that stands in for the image: `[image 1024×768 · ~765 tok]`.
 *
 * Degrades one field at a time, so the descriptor always says exactly as much as
 * is actually known: `[image · ~85 tok]` when the cost is known but the size is
 * not (OpenAI `detail: "low"` on a remote URL), and a bare `[image]` when
 * neither is. The `~` is not decoration — it is the same "this is an estimate"
 * claim the block's `tokenMethod` makes, restated where a human reads it.
 */
export function formatImageText(
  width: number | null,
  height: number | null,
  tokens: number,
): string {
  const size = width && height ? `${width}${TIMES}${height}` : null;
  const cost = tokens ? `~${formatTokenEstimate(tokens)} tok` : null;
  if (size && cost) return `[image ${size} ${DOT} ${cost}]`;
  if (size) return `[image ${size}]`;
  if (cost) return `[image ${DOT} ${cost}]`;
  return "[image]";
}

/**
 * The string hashed (with role and kind) to give an image block its identity: a
 * payload term, then the terms that change what the payload COSTS or DOES.
 *
 *     image:sha256:<hex>[;detail=<detail>][;part=<stable-json>]
 *
 * Bytes first: `image:sha256:<hex>` over the raw image bytes means the SAME
 * picture is ONE block no matter how it was wrapped — a data URI in an OpenAI
 * request, a base64 source in an Anthropic one, `inlineData` in a Gemini one,
 * sent once or re-sent on every turn of a long agent loop. That is what makes an
 * image dedup like any other block, and what makes "this screenshot has been in
 * context for 12 turns" a question the diff can answer.
 *
 * Without bytes we fall back to the reference itself — the URL, or the
 * provider-side file id — which still dedups a repeated reference to the same
 * remote image. The `image:` prefix and the explicit `sha256:` / `url:` / `ref:`
 * tags keep the three namespaces from ever colliding with each other (or with a
 * plain text block that happens to spell a hex digest).
 *
 * THE PIXELS ARE NOT THE WHOLE REQUEST, though, and the two suffixes are the
 * difference between an image and a request FOR an image:
 *
 *   - `;detail=` — OpenAI's fidelity hint changes the cost nine-fold (85 at
 *     `"low"`, 765 at `"high"` for a 1024×768 screenshot) and changes the
 *     descriptor with it. Leaving it out collapsed the standard computer-use
 *     pattern — the same screenshot at `"low"` in history and `"high"` for the
 *     current turn — into one block, where INSERT OR IGNORE kept the first count
 *     written and the diff called a 9× cost change "unchanged".
 *   - `;part=` — everything else the content part carried (see `partModifiers`),
 *     chiefly Anthropic's `cache_control`. A cache breakpoint moving on or off
 *     an image is precisely what the cache profiler exists to report; with it
 *     outside the identity the hash did not move and the profiler called the
 *     prefix stable.
 *
 * Both suffixes are absent for the common case, so the documented shapes still
 * dedup byte for byte across providers and wrappers.
 */
export function imageHashInput(ref: ImageRef): string {
  let base: string;
  if (ref.data !== null) {
    base = "image:sha256:" + createHash("sha256").update(ref.data).digest("hex");
  } else if (ref.url) {
    base = "image:url:" + ref.url;
  } else if (ref.ref) {
    base = "image:ref:" + ref.ref;
  } else {
    base = "image:unknown";
  }
  if (ref.detail) base += ";detail=" + ref.detail;
  if (ref.modifiers) base += ";part=" + ref.modifiers;
  return base;
}

// --- provider part shapes ---------------------------------------------------

/** Return `value` when it is a non-empty string, else null — so a malformed
 * payload (a number where a URL belongs) degrades instead of propagating a wrong
 * type into the hash. */
function strOrNull(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

/** Build an ImageRef from a URL that is either a `data:` URI (inline bytes we
 * can hash and sniff) or a remote address we will NOT fetch. */
function fromUrl(url: string, detail: string | null): ImageRef {
  const [data, mediaType] = parseDataUri(url);
  if (data !== null) return makeRef({ data, detail, mediaType });
  return makeRef({ url, detail, mediaType });
}

/**
 * OpenAI Chat Completions: `{type: "image_url", image_url: {url: ..., detail:
 * "high"}}`. The nested object is the documented shape; a bare string under
 * `image_url` is accepted too, because hand-rolled clients and older examples
 * emit it.
 */
function openaiChatImage(part: Record<string, unknown>): ImageRef | null {
  const imageUrl = part["image_url"];
  if (typeof imageUrl === "string") return fromUrl(imageUrl, null);
  if (isRecord(imageUrl)) {
    const url = strOrNull(imageUrl["url"]);
    const detail = strOrNull(imageUrl["detail"]);
    if (url) return fromUrl(url, detail);
    const fileId = strOrNull(imageUrl["file_id"]);
    if (fileId) return makeRef({ ref: fileId, detail });
  }
  return null;
}

/**
 * OpenAI Responses: `{type: "input_image", image_url: "data:...", detail:
 * "high"}` — note `image_url` is a plain STRING here, unlike Chat Completions —
 * or `{type: "input_image", file_id: "file-..."}` for an image already uploaded
 * to the Files API (no bytes reachable locally).
 */
function openaiResponsesImage(part: Record<string, unknown>): ImageRef | null {
  const detail = strOrNull(part["detail"]);
  const url = strOrNull(part["image_url"]);
  if (url) return fromUrl(url, detail);
  const nested = part["image_url"];
  if (isRecord(nested)) {
    const inner = strOrNull(nested["url"]); // defensive: chat shape reused
    if (inner) return fromUrl(inner, detail);
  }
  const fileId = strOrNull(part["file_id"]);
  if (fileId) return makeRef({ ref: fileId, detail });
  return null;
}

/**
 * Anthropic Messages: `{type: "image", source: {...}}`, where the source is one
 * of three documented shapes — `{type: "base64", media_type: "image/png", data:
 * "<b64>"}`, the newer `{type: "url", url: "https://..."}`, or `{type: "file",
 * file_id: "..."}` for the Files API. A source missing its `type` discriminator
 * is read by which key it carries, so a slightly-off payload still records as an
 * image.
 */
function anthropicImage(part: Record<string, unknown>): ImageRef | null {
  const source = part["source"];
  if (!isRecord(source)) return null;
  const mediaType = strOrNull(source["media_type"]);
  const data = b64Decode(source["data"]);
  if (data !== null) return makeRef({ data, mediaType });
  const url = strOrNull(source["url"]);
  if (url) return fromUrl(url, null);
  const fileId = strOrNull(source["file_id"]);
  if (fileId) return makeRef({ ref: fileId, mediaType });
  return null;
}

/** Whether a declared MIME type names an image. Used to keep Gemini's generic
 * `inlineData`/`fileData` carriers — which also transport audio, video and PDFs
 * — from being rewritten as image blocks. */
function isImageMediaType(mediaType: string | null): boolean {
  return !!mediaType && mediaType.toLowerCase().startsWith("image/");
}

/**
 * Gemini `inline_data` / `inlineData`: `{mime_type: "image/png", data: "<b64 or
 * bytes>"}` (the JS SDK spells the key `mimeType`).
 *
 * Gated on the MIME type because the same carrier also delivers audio, video and
 * PDFs; a non-image keeps the pre-existing JSON-serialization path untouched.
 * When the MIME type is absent entirely, the header sniffer decides — if the
 * bytes really are a PNG/JPEG/GIF/WebP we treat them as the image they are,
 * otherwise we leave the part alone.
 */
function geminiInlineImage(inline: Record<string, unknown>): ImageRef | null {
  const mediaType = strOrNull(inline["mime_type"]) ?? strOrNull(inline["mimeType"]);
  const data = b64Decode(inline["data"]);
  if (data === null) return null;
  if (mediaType === null) return sniffDimensions(data) ? makeRef({ data }) : null;
  if (!isImageMediaType(mediaType)) return null;
  return makeRef({ data, mediaType });
}

/**
 * Gemini `file_data` / `fileData`: `{mime_type: "image/jpeg", file_uri:
 * "https://generativelanguage.googleapis.com/..."}` (`mimeType` / `fileUri` in
 * the JS SDK). A URI only — never fetched — so this degrades to `[image]`.
 * Requires an image MIME type, since the same shape carries video and PDFs and
 * there are no bytes to sniff.
 */
function geminiFileImage(fileData: Record<string, unknown>): ImageRef | null {
  const mediaType = strOrNull(fileData["mime_type"]) ?? strOrNull(fileData["mimeType"]);
  if (!isImageMediaType(mediaType)) return null;
  const uri = strOrNull(fileData["file_uri"]) ?? strOrNull(fileData["fileUri"]);
  if (!uri) return null;
  return makeRef({ ref: uri, mediaType });
}

/**
 * Bedrock Converse: `{image: {format: "png", source: {bytes: <bytes>}}}`. The
 * AWS SDK hands a `Uint8Array` directly; a base64 string is accepted too for
 * hand-built payloads. An `s3Location` source names an object in the caller's
 * bucket that we will not read, so it degrades to a reference.
 */
function bedrockImage(image: Record<string, unknown>): ImageRef | null {
  const source = image["source"];
  if (!isRecord(source)) return null;
  const fmt = strOrNull(image["format"]);
  const mediaType = fmt ? `image/${fmt.toLowerCase()}` : null;
  const data = b64Decode(source["bytes"]);
  if (data !== null) return makeRef({ data, mediaType });
  const s3 = source["s3Location"];
  if (isRecord(s3)) {
    const uri = strOrNull(s3["uri"]);
    if (uri) return makeRef({ ref: uri, mediaType });
  }
  return null;
}

/**
 * Dispatch one content part to the provider shape that owns it, returning what
 * the payload itself says. The shapes are disjoint on their discriminator key
 * (`type: image_url` / `input_image` / `image`, or the presence of `inlineData`
 * / `fileData` / `image`), so a single dispatch is unambiguous — and a shared
 * detector is the only way the two SDKs, five providers and the golden harness
 * can be guaranteed to agree on what counts as an image. `detectImagePart` wraps
 * this to add the part's remainder.
 */
function detectShape(part: Record<string, unknown>): ImageRef | null {
  const partType = part["type"];
  if (partType === "image_url" || ("image_url" in part && partType === undefined)) {
    return openaiChatImage(part);
  }
  if (partType === "input_image") return openaiResponsesImage(part);
  if (partType === "image" && isRecord(part["source"])) return anthropicImage(part);
  for (const key of ["inline_data", "inlineData"]) {
    const inline = part[key];
    if (isRecord(inline)) return geminiInlineImage(inline);
  }
  for (const key of ["file_data", "fileData"]) {
    const fileData = part[key];
    if (isRecord(fileData)) return geminiFileImage(fileData);
  }
  const image = part["image"];
  if (isRecord(image)) return bedrockImage(image);
  return null;
}

/**
 * Keys the shape functions above have ALREADY accounted for. Each is either the
 * payload itself (`data`/`url`/`bytes`/`file_id`/`file_uri`/`s3Location`), the
 * discriminator that selected the shape (`type`, `format`), a hint promoted into
 * the ImageRef (`detail`), or a field deliberately excluded from identity so the
 * same picture dedups however it was labeled (the media type).
 */
const ACCOUNTED_KEYS = new Set([
  "type", "detail", "url", "data", "bytes", "file_id", "file_uri", "fileUri",
  "media_type", "mime_type", "mimeType", "format", "s3Location",
]);

/**
 * Keys whose VALUE is one of the nested carriers a shape wraps its payload in,
 * and therefore the only places worth looking one level deeper. Recursion is
 * restricted to these on purpose: `cache_control: {type: "ephemeral"}` must keep
 * its own `type`, which is a cache mode and not a part discriminator.
 */
const CARRIER_KEYS = new Set([
  "image_url", "source", "inline_data", "inlineData", "file_data", "fileData",
  "image",
]);

/**
 * Whether `value` is something both SDKs serialize identically — strings,
 * numbers, booleans, null, and arrays/objects of those.
 *
 * Anything else (raw bytes under an unrecognized key, an SDK object) is dropped
 * from the remainder rather than serialized: Python's `json.dumps` raises on it
 * while `stableStringify` would happily render a byte array as `{"0": 137, …}`,
 * and a hash that depends on which SDK captured the call is worse than a hash
 * that ignores an exotic value.
 */
function isJsonSafe(value: unknown): boolean {
  if (value === null) return true;
  const t = typeof value;
  if (t === "string" || t === "boolean") return true;
  if (t === "number") return Number.isFinite(value);
  if (Array.isArray(value)) return value.every(isJsonSafe);
  if (value instanceof Uint8Array) return false; // Python cannot serialize bytes
  if (isRecord(value)) return Object.values(value).every(isJsonSafe);
  return false;
}

/**
 * Return the part's remainder: everything `detectShape` did NOT consume, with
 * the nested carriers descended into and pruned when they empty out.
 *
 * The rule is "we understand it, or it counts": every key ctxdiff reads is in
 * `ACCOUNTED_KEYS` and is already represented in the block, so whatever is left
 * is a modifier ctxdiff cannot interpret but the provider can — and a
 * provider-side behavior change that leaves no trace in the trace is the one
 * outcome a context debugger must not produce.
 */
function unaccountedKeys(obj: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const key of Object.keys(obj)) {
    if (ACCOUNTED_KEYS.has(key)) continue;
    const value = obj[key];
    if (CARRIER_KEYS.has(key)) {
      if (isRecord(value)) {
        const nested = unaccountedKeys(value);
        if (Object.keys(nested).length > 0) out[key] = nested;
      }
      continue; // a non-object carrier IS the payload (a bare `image_url`)
    }
    // `undefined` has no Python counterpart, so it is treated as absent rather
    // than serialized (`stableStringify` would render it as `null`).
    if (value !== undefined && isJsonSafe(value)) out[key] = value;
  }
  return out;
}

/**
 * The remainder of a content part as a stable, sorted-key JSON string, or null
 * when there is none.
 *
 * Null for every documented shape — an OpenAI `image_url`, an Anthropic
 * `source`, a Gemini `inlineData`, a Bedrock `image` — which is what keeps
 * cross-provider dedup intact. Non-null exactly when the part carried something
 * extra, the motivating case being `cache_control`:
 *
 *     {type: "image", source: {…}, cache_control: {type: "ephemeral"}}
 *     ->  '{"cache_control": {"type": "ephemeral"}}'
 *
 * Serialized through the same `normalizeText` every other block hashes with, so
 * the Python twin produces the identical bytes.
 */
function partModifiers(part: Record<string, unknown>): string | null {
  const remainder = unaccountedKeys(part);
  return Object.keys(remainder).length > 0 ? normalizeText(remainder) : null;
}

/**
 * Recognize one content part as an image across every provider shape ctxdiff
 * captures, returning what could be learned about it without I/O — or null when
 * the part is not an image, in which case the caller keeps its existing behavior
 * verbatim.
 *
 * Two steps: `detectShape` reads the payload, then the part's unaccounted
 * remainder is attached, so identity covers the whole content part and not just
 * the pixels inside it. Splitting them keeps the per-provider readers free of
 * any knowledge of the remainder rule.
 *
 * Never throws: a malformed part returns null and falls back to the previous
 * JSON-serialization path, which is always a safe (if verbose) answer.
 */
export function detectImagePart(part: unknown): ImageRef | null {
  if (!isRecord(part)) return null;
  try {
    const ref = detectShape(part);
    if (ref === null) return null;
    return { ...ref, modifiers: partModifiers(part) };
  } catch {
    return null; // a weird payload must never break capture
  }
}

/**
 * Turn a detected image into the four fields a block needs: `[text, hashInput,
 * tokenCount, tokenMethod]`.
 *
 * Kept separate from `detectImagePart` so the golden harness and the tests can
 * drive the measurement half directly, with an ImageRef they built themselves,
 * without going through a provider payload.
 */
export function imageBlockFields(
  ref: ImageRef,
  provider: string,
): [string, string, number, string] {
  const size = sniffDimensions(ref.data);
  const width = size ? size[0] : null;
  const height = size ? size[1] : null;
  const tokens = estimateImageTokens(provider, width, height, ref.detail);
  return [
    formatImageText(width, height, tokens),
    imageHashInput(ref),
    tokens,
    IMAGE_TOKEN_METHOD,
  ];
}

/**
 * THE adapter entry point: if `part` is an image in any provider shape, return
 * the `RawBlock` that should stand in for it; otherwise null, meaning "not an
 * image — do whatever you did before".
 *
 * Every adapter calls exactly this, first, in its content-part loop. Routing all
 * five providers through one function is what makes the block a vision request
 * produces independent of which SDK captured it: the same picture in an OpenAI
 * request and in an Anthropic one produces the same hash, the same descriptor
 * and (per that provider's published formula) an honest cost.
 */
export function imageRawBlock(
  role: string,
  part: unknown,
  provider: string,
): RawBlock | null {
  const ref = detectImagePart(part);
  if (ref === null) return null;
  const [text, hashInput, tokenCount, tokenMethod] = imageBlockFields(ref, provider);
  return { role, kind: IMAGE_KIND, text, hashInput, tokenCount, tokenMethod };
}
