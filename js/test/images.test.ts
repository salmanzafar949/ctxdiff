/**
 * Tests for the image block representation — `src/images.ts` plus the three
 * adapters that route content parts through it. The mirror of Python's
 * `tests/test_images.py`, assertion for assertion.
 *
 * The thing under test, stated once: an image content part must NOT reach the
 * store as its base64 payload. It must become a block whose text is a short
 * descriptor, whose identity is a digest of the image BYTES, and whose token
 * count is the provider's published vision formula marked as an estimate. Every
 * test below is one clause of that sentence, or one way it is allowed to
 * degrade.
 */
import { describe, it, expect } from "vitest";
import { deflateSync } from "node:zlib";
import { readFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { createHash, randomUUID } from "node:crypto";
import { OpenAIAdapter } from "../src/capture/openai.js";
import { AnthropicAdapter } from "../src/capture/anthropic.js";
import { GeminiAdapter } from "../src/capture/gemini.js";
import { buildBlock, Recorder } from "../src/capture/recorder.js";
import { CTrace } from "../src/store/ctrace.js";
import { DDL, SCHEMA_VERSION } from "../src/store/schema.js";
import { analyzeCall } from "../src/analyze/tokens.js";
import { diffTurns } from "../src/analyze/diff.js";
import { basicLabel, contentHash, stableStringify } from "../src/models.js";
import type { RawBlock } from "../src/models.js";
import { countTokens } from "../src/tokenize.js";
import {
  JPEG_MAX_SCAN,
  detectImagePart,
  estimateImageTokens,
  formatImageText,
  formatTokenEstimate,
  imageRawBlock,
  sniffDimensions,
} from "../src/images.js";

// --- tiny synthetic images ---------------------------------------------------
//
// Header-accurate blobs rather than real photographs: the sniffer reads only the
// first few dozen bytes of each format, so a hand-built header exercises exactly
// the code path a 4 MB screenshot would, in 40 bytes and with no binary fixture
// checked into the repo.

function u32be(n: number): Uint8Array {
  return new Uint8Array([(n >>> 24) & 0xff, (n >>> 16) & 0xff, (n >>> 8) & 0xff, n & 0xff]);
}
function u16be(n: number): Uint8Array {
  return new Uint8Array([(n >>> 8) & 0xff, n & 0xff]);
}
function u16le(n: number): Uint8Array {
  return new Uint8Array([n & 0xff, (n >>> 8) & 0xff]);
}
function u32le(n: number): Uint8Array {
  return new Uint8Array([n & 0xff, (n >>> 8) & 0xff, (n >>> 16) & 0xff, (n >>> 24) & 0xff]);
}
function cat(...parts: (Uint8Array | number[] | string)[]): Uint8Array {
  const bufs = parts.map((p) =>
    typeof p === "string" ? Buffer.from(p, "latin1") : Buffer.from(p as Uint8Array),
  );
  return new Uint8Array(Buffer.concat(bufs));
}

/** CRC-32 as PNG chunks require it — the same polynomial Python's zlib.crc32
 * uses, so the two SDKs' synthetic PNGs are byte-identical for a given size. */
function crc32(data: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < data.length; i++) {
    c ^= data[i];
    for (let k = 0; k < 8; k++) c = c & 1 ? (c >>> 1) ^ 0xedb88320 : c >>> 1;
  }
  return (c ^ 0xffffffff) >>> 0;
}

function pngChunk(tag: string, data: Uint8Array): Uint8Array {
  const body = cat(tag, data);
  return cat(u32be(data.length), body, u32be(crc32(body)));
}

/**
 * A structurally valid PNG declaring `width`x`height` in its IHDR. `payload`
 * varies the IDAT so two images of the SAME size can be given DIFFERENT bytes —
 * which is how the dedup tests tell identity (the pixels) apart from the
 * descriptor (the label).
 */
function png(width: number, height: number, payload = new Uint8Array([0, 0, 0])): Uint8Array {
  const ihdr = cat(u32be(width), u32be(height), [8, 2, 0, 0, 0]);
  return cat(
    [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a],
    pngChunk("IHDR", ihdr),
    pngChunk("IDAT", new Uint8Array(deflateSync(Buffer.from(payload)))),
    pngChunk("IEND", new Uint8Array(0)),
  );
}

/** A JPEG carrying a JFIF APP0 segment before its SOF0, so the sniffer has to
 * actually WALK the marker chain rather than read a fixed offset. */
function jpeg(width: number, height: number): Uint8Array {
  const app0 = cat([0xff, 0xe0], u16be(16), "JFIF\0", [1, 1, 0], u16be(1), u16be(1), [0, 0]);
  const sof0 = cat(
    [0xff, 0xc0],
    u16be(17),
    [8],
    u16be(height),
    u16be(width),
    [3],
    [1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1],
  );
  return cat([0xff, 0xd8], app0, sof0, [0xff, 0xd9]);
}

/** A GIF89a whose logical screen descriptor carries the size. */
function gif(width: number, height: number): Uint8Array {
  return cat("GIF89a", u16le(width), u16le(height), [0xf0, 0, 0], new Uint8Array(6), ";");
}

/** A RIFF/WEBP container whose first chunk is a lossy `VP8 ` frame. */
function webpLossy(width: number, height: number): Uint8Array {
  const frame = cat([0, 0, 0], [0x9d, 0x01, 0x2a], u16le(width), u16le(height), new Uint8Array(8));
  const payload = cat("VP8 ", u32le(frame.length), frame);
  return cat("RIFF", u32le(4 + payload.length), "WEBP", payload);
}

/** A RIFF/WEBP container whose first chunk is a lossless `VP8L` frame —
 * dimensions packed as two 14-bit fields, a completely different layout. */
function webpLossless(width: number, height: number): Uint8Array {
  const bits = ((width - 1) | ((height - 1) << 14)) >>> 0;
  const body = cat([0x2f], u32le(bits), new Uint8Array(12));
  const payload = cat("VP8L", u32le(body.length), body);
  return cat("RIFF", u32le(4 + payload.length), "WEBP", payload);
}

/** A RIFF/WEBP container whose first chunk is a `VP8X` extended header — the
 * shape an animated or alpha-channel WebP uses. */
function webpExtended(width: number, height: number): Uint8Array {
  const w = width - 1;
  const h = height - 1;
  const body = cat(
    [0x10, 0, 0, 0],
    [w & 0xff, (w >>> 8) & 0xff, (w >>> 16) & 0xff],
    [h & 0xff, (h >>> 8) & 0xff, (h >>> 16) & 0xff],
  );
  const payload = cat("VP8X", u32le(body.length), body);
  return cat("RIFF", u32le(4 + payload.length), "WEBP", payload, new Uint8Array(16));
}

function b64(data: Uint8Array): string {
  return Buffer.from(data).toString("base64");
}
function uri(data: Uint8Array, mediaType = "image/png"): string {
  return `data:${mediaType};base64,${b64(data)}`;
}

const PNG_1024x768 = png(1024, 768);
const PNG_1024x768_ALT = png(1024, 768, new Uint8Array([1, 2, 3]));

/**
 * The same PNG in all four wrappers, each paired with the provider that
 * actually emits that wrapper — so a test using this exercises the real
 * cross-provider case (different formulas, therefore different counts) rather
 * than four shapes pretending to be one provider.
 */
const SAME_PNG_ACROSS_PROVIDERS: [string, unknown][] = [
  ["openai", { type: "image_url", image_url: { url: uri(PNG_1024x768) } }],
  [
    "anthropic",
    { type: "image", source: { type: "base64", media_type: "image/png", data: b64(PNG_1024x768) } },
  ],
  ["gemini", { inline_data: { mime_type: "image/png", data: b64(PNG_1024x768) } }],
  ["bedrock", { image: { format: "png", source: { bytes: PNG_1024x768 } } }],
];

// --- the dimension sniffer ----------------------------------------------------

describe("dimension sniffing (header bytes only, no decoder, no dependency)", () => {
  it.each([
    ["png", png(1024, 768), [1024, 768]],
    ["png 1x1", png(1, 1), [1, 1]],
    ["jpeg", jpeg(800, 600), [800, 600]],
    ["gif", gif(640, 480), [640, 480]],
    ["webp lossy", webpLossy(500, 400), [500, 400]],
    ["webp lossless", webpLossless(300, 200), [300, 200]],
    ["webp extended", webpExtended(1280, 720), [1280, 720]],
  ])("reads %s dimensions from the header", (_name, data, expected) => {
    expect(sniffDimensions(data as Uint8Array)).toEqual(expected);
  });

  it.each([
    ["empty", new Uint8Array(0)],
    ["bmp (a real format we do not sniff)", cat("BM", new Uint8Array(60))],
    ["png truncated before IHDR", cat([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a], new Uint8Array(4))],
    ["jpeg SOI with nothing after", new Uint8Array([0xff, 0xd8, 0xff])],
    ["webp header with no chunk", cat("RIFF", new Uint8Array(8), "WEBP")],
    ["gif header with no screen descriptor", cat("GIF89a")],
    ["plain prose", cat("not an image at all, just prose")],
  ])("degrades to null on %s", (_name, data) => {
    // Null is not a failure — it is what makes the descriptor degrade to
    // `[image]` and the estimate to "unknown", which is the honest answer.
    expect(sniffDimensions(data as Uint8Array)).toBeNull();
  });

  it("rejects a zero dimension as malformed rather than reporting [image 0×768]", () => {
    expect(sniffDimensions(png(0, 768))).toBeNull();
  });

  it.each([
    ["4294967295×4294967295 (a truncated download / fuzzer / hostile header)", 0xffffffff, 0xffffffff],
    ["one past the largest expressible dimension", 65536, 768],
    ["100000 high", 768, 100000],
  ])("rejects an implausible dimension: %s", (_name, width, height) => {
    // PNG's IHDR is a pair of uint32s, so a corrupt header can declare 18
    // exapixels and be structurally valid. Trusting it turns ONE block into
    // 8,068,951,256,159,688 estimated tokens, which rounds every other block in
    // the run to 0.0% and makes the run total meaningless. Anything above 65535
    // — the largest size JPEG and GIF can even express — is a broken header.
    expect(sniffDimensions(png(width as number, height as number))).toBeNull();
  });

  it("still reads the largest expressible dimension", () => {
    // The clamp is a plausibility bound, not a size limit.
    expect(sniffDimensions(png(65535, 65535))).toEqual([65535, 65535]);
  });

  it("stops the jpeg marker walk at the scan limit", () => {
    // The walk is bounded. A payload that never resynchronizes degrades to
    // byte-at-a-time resync, so an unbounded walk over a multi-megabyte corrupt
    // JPEG stalls the HOST application's thread (capture runs synchronously
    // inside the caller's call). Pinned by putting a perfectly good SOF0 just
    // past the bound: the sniffer must stop before reaching it.
    const tail = jpeg(800, 600).slice(2); // everything after the SOI marker
    expect(sniffDimensions(cat([0xff, 0xd8], new Uint8Array(64), tail))).toEqual([800, 600]);
    expect(sniffDimensions(cat([0xff, 0xd8], new Uint8Array(JPEG_MAX_SCAN), tail))).toBeNull();
  });

  it("sniffs a large corrupt jpeg in bounded time", () => {
    // The guardrail this protects is "never break the host app": a corrupt or
    // truncated multi-MB screenshot must not stall the agent being traced.
    const payload = cat([0xff, 0xd8], new Uint8Array(32 << 20)); // never resyncs
    const started = performance.now();
    expect(sniffDimensions(payload)).toBeNull();
    expect(performance.now() - started).toBeLessThan(250);
  });
});

// --- provider token formulas --------------------------------------------------

describe("vision-token estimates follow each provider's published formula", () => {
  it.each([
    [1024, 768, "low", 85],
    [4096, 4096, "low", 85],
    [1024, 768, "high", 85 + 170 * 4],
    [512, 512, "high", 85 + 170],
    [2000, 1200, "high", 85 + 170 * 6],
    [1024, 768, null, 85 + 170 * 4],
    [1024, 768, "auto", 85 + 170 * 4],
  ])("openai %ix%i detail=%s -> %i tokens", (w, h, detail, expected) => {
    // 85 flat for low detail, else 85 + 170 per 512x512 tile after fitting into
    // 2048x2048 and reducing the shortest side to 768.
    expect(estimateImageTokens("openai", w as number, h as number, detail as string | null)).toBe(
      expected,
    );
  });

  it.each([
    [800, 600, 640],
    [1024, 768, 1049],
    [10, 10, 1],
  ])("anthropic %ix%i -> %i tokens ((w*h)/750, long edge capped at 1568)", (w, h, expected) => {
    expect(estimateImageTokens("anthropic", w, h, null)).toBe(expected);
  });

  it("bedrock shares the anthropic formula (a Converse image is headed for Claude)", () => {
    expect(estimateImageTokens("bedrock", 800, 600, null)).toBe(
      estimateImageTokens("anthropic", 800, 600, null),
    );
  });

  it.each([
    [384, 384, 258],
    [200, 100, 258],
    [640, 480, 258 * 4],
    [1024, 768, 258 * 4],
  ])("gemini %ix%i -> %i tokens (258 per tile of min(w,h)/1.5)", (w, h, expected) => {
    expect(estimateImageTokens("gemini", w, h, null)).toBe(expected);
  });

  it("an unknown provider falls back to the openai formula", () => {
    // Azure OpenAI and the OpenAI-compatible OSS endpoints speak the same wire
    // format, so an unrecognized provider id is costed rather than left at zero.
    expect(estimateImageTokens("some-oss-gateway", 1024, 768, null)).toBe(
      estimateImageTokens("openai", 1024, 768, null),
    );
  });

  it("estimates nothing rather than guessing when the dimensions are unknown", () => {
    // 0 is a visible gap against provider usage; a fabricated number would be
    // indistinguishable from a measured one in every view.
    expect(estimateImageTokens("openai", null, null, "high")).toBe(0);
    expect(estimateImageTokens("anthropic", null, null, null)).toBe(0);
    expect(estimateImageTokens("gemini", null, null, null)).toBe(0);
  });

  it("detail:low is the one cost knowable without any dimensions", () => {
    expect(estimateImageTokens("openai", null, null, "low")).toBe(85);
  });
});

// --- the descriptor -----------------------------------------------------------

describe("the descriptor", () => {
  it.each([
    [0, "0"],
    [85, "85"],
    [765, "765"],
    [999, "999"],
    [1000, "1k"],
    [1049, "1k"],
    [1105, "1.1k"],
    [1150, "1.2k"],
    [1950, "2k"],
    [12345, "12.3k"],
  ])("formats %i tokens as %s", (tokens, expected) => {
    // Rounded with floor(n/100 + 0.5) so Python's banker's rounding and JS's
    // half-up cannot disagree.
    expect(formatTokenEstimate(tokens)).toBe(expected);
  });

  it("carries size and estimate in the chosen form", () => {
    expect(formatImageText(1024, 768, 765)).toBe("[image 1024×768 · ~765 tok]");
  });

  it("degrades one field at a time, saying exactly as much as is known", () => {
    expect(formatImageText(null, null, 85)).toBe("[image · ~85 tok]");
    expect(formatImageText(null, null, 0)).toBe("[image]");
    expect(formatImageText(1024, 768, 0)).toBe("[image 1024×768]");
  });
});

// --- per-provider part shapes -------------------------------------------------

/** Extract blocks and return the single image block among them, asserting there
 * is exactly one — so a test that accidentally matched two shapes fails loudly
 * instead of silently checking the wrong block. */
function oneImageBlock(
  adapter: { extractBlocks(k: Record<string, unknown>): RawBlock[] },
  kwargs: Record<string, unknown>,
): RawBlock {
  const images = adapter.extractBlocks(kwargs).filter((b) => b.kind === "image");
  expect(images).toHaveLength(1);
  return images[0];
}

describe("per-provider image part shapes", () => {
  it("openai chat image_url with a data URI", () => {
    const block = oneImageBlock(new OpenAIAdapter(), {
      messages: [
        {
          role: "user",
          content: [{ type: "image_url", image_url: { url: uri(PNG_1024x768), detail: "high" } }],
        },
      ],
    });
    expect(block.text).toBe("[image 1024×768 · ~765 tok]");
    expect(block.tokenCount).toBe(765);
    expect(block.tokenMethod).toBe("estimate");
    expect(block.hashInput?.startsWith("image:sha256:")).toBe(true);
    // The base64 must appear NOWHERE in what gets stored.
    expect(block.text).not.toContain(b64(PNG_1024x768).slice(0, 32));
  });

  it("openai chat image_url given as a bare string", () => {
    const block = oneImageBlock(new OpenAIAdapter(), {
      messages: [{ role: "user", content: [{ type: "image_url", image_url: uri(PNG_1024x768) }] }],
    });
    expect(block.text).toBe("[image 1024×768 · ~765 tok]");
  });

  it("openai responses input_image with a data URI (image_url is a plain string here)", () => {
    const block = oneImageBlock(new OpenAIAdapter(), {
      input: [
        {
          role: "user",
          content: [{ type: "input_image", image_url: uri(PNG_1024x768), detail: "high" }],
        },
      ],
    });
    expect(block.text).toBe("[image 1024×768 · ~765 tok]");
    expect(block.tokenCount).toBe(765);
  });

  it("openai responses input_image with only a file_id degrades", () => {
    const block = oneImageBlock(new OpenAIAdapter(), {
      input: [{ role: "user", content: [{ type: "input_image", file_id: "file-3d9a17c04be84e2fb0c5" }] }],
    });
    expect(block.text).toBe("[image]");
    expect(block.tokenCount).toBe(0);
    expect(block.hashInput).toBe("image:ref:file-3d9a17c04be84e2fb0c5");
  });

  it("anthropic base64 source, costed with anthropic's own formula", () => {
    const block = oneImageBlock(new AnthropicAdapter(), {
      messages: [
        {
          role: "user",
          content: [
            {
              type: "image",
              source: { type: "base64", media_type: "image/jpeg", data: b64(jpeg(800, 600)) },
            },
          ],
        },
      ],
    });
    expect(block.text).toBe("[image 800×600 · ~640 tok]");
    expect(block.tokenCount).toBe(640);
  });

  it("anthropic url source degrades without fetching", () => {
    const block = oneImageBlock(new AnthropicAdapter(), {
      messages: [
        {
          role: "user",
          content: [{ type: "image", source: { type: "url", url: "https://cdn.example.com/a.png" } }],
        },
      ],
    });
    expect(block.text).toBe("[image]");
    expect(block.hashInput).toBe("image:url:https://cdn.example.com/a.png");
  });

  it("anthropic file source degrades to the file id", () => {
    const block = oneImageBlock(new AnthropicAdapter(), {
      messages: [
        {
          role: "user",
          content: [{ type: "image", source: { type: "file", file_id: "file_011CQrsTuVwXyZ" } }],
        },
      ],
    });
    expect(block.text).toBe("[image]");
    expect(block.hashInput).toBe("image:ref:file_011CQrsTuVwXyZ");
  });

  it.each([
    ["snake_case", { inline_data: { mime_type: "image/gif", data: b64(gif(640, 480)) } }],
    ["camelCase", { inlineData: { mimeType: "image/gif", data: b64(gif(640, 480)) } }],
  ])("gemini inline data in %s reaches the same block", (_name, part) => {
    // google-genai spells the key one way in Python and the other in JS; both
    // must produce the same block or a trace stops being comparable.
    const block = oneImageBlock(new GeminiAdapter(), {
      contents: [{ role: "user", parts: [part] }],
    });
    expect(block.text).toBe("[image 640×480 · ~1k tok]");
    expect(block.tokenCount).toBe(258 * 4);
  });

  it("gemini inline data given as raw bytes", () => {
    const block = oneImageBlock(new GeminiAdapter(), {
      contents: [{ role: "user", parts: [{ inline_data: { mime_type: "image/gif", data: gif(640, 480) } }] }],
    });
    expect(block.text).toBe("[image 640×480 · ~1k tok]");
  });

  it("gemini non-image inline data keeps the stable-JSON content_part path", () => {
    // `inlineData` also carries audio, video and PDFs. Only an image MIME type
    // is rerouted; everything else keeps the pre-existing behavior byte for byte.
    const blocks = new GeminiAdapter().extractBlocks({
      contents: [{ role: "user", parts: [{ inline_data: { mime_type: "audio/wav", data: "UklGRiQAAABXQVZF" } }] }],
    });
    expect(blocks.map((b) => b.kind)).toEqual(["content_part"]);
    expect(blocks[0].text).toContain("audio/wav");
  });

  it.each([
    ["snake_case", { file_data: { mime_type: "image/jpeg", file_uri: "https://gen.googleapis.com/v1/files/7k2m" } }],
    ["camelCase", { fileData: { mimeType: "image/jpeg", fileUri: "https://gen.googleapis.com/v1/files/7k2m" } }],
  ])("gemini file data in %s degrades without fetching", (_name, part) => {
    const block = oneImageBlock(new GeminiAdapter(), { contents: [{ role: "user", parts: [part] }] });
    expect(block.text).toBe("[image]");
    expect(block.hashInput).toBe("image:ref:https://gen.googleapis.com/v1/files/7k2m");
  });

  it("bedrock converse image bytes (via the shared detector)", () => {
    // There is no JS Bedrock adapter yet, but the detector is shared and a
    // Converse payload must already resolve — so a future adapter inherits it.
    const block = imageRawBlock(
      "user",
      { image: { format: "png", source: { bytes: PNG_1024x768 } } },
      "bedrock",
    );
    expect(block?.text).toBe("[image 1024×768 · ~1k tok]");
    expect(block?.tokenCount).toBe(1049);
  });
});

// --- degradation and the no-network guarantee ---------------------------------

describe("degradation, and the promise never to fetch", () => {
  it("a remote http URL degrades and is never fetched", async () => {
    // THE local-first guarantee, asserted rather than assumed. Every outbound
    // path in Node — `fetch`, `http(s).request`, undici, axios, anything —
    // ultimately opens a TCP connection through `net.Socket.prototype.connect`,
    // so replacing that one prototype method (plus `fetch`, which may reach the
    // network through internal bindings rather than the public socket) turns any
    // attempted connection into an immediate failure. The ESM namespace objects
    // themselves are frozen and cannot be reassigned, which is why the patch
    // goes on the prototype.
    const net = await import("node:net");
    const boom = () => {
      throw new Error("ctxdiff attempted a network connection while capturing an image");
    };
    const savedFetch = globalThis.fetch;
    const savedConnect = net.Socket.prototype.connect;
    globalThis.fetch = boom as unknown as typeof fetch;
    net.Socket.prototype.connect = boom as unknown as typeof net.Socket.prototype.connect;
    try {
      const block = oneImageBlock(new OpenAIAdapter(), {
        messages: [
          {
            role: "user",
            content: [{ type: "image_url", image_url: { url: "https://cdn.example.com/huge.png" } }],
          },
        ],
      });
      expect(block.text).toBe("[image]");
      expect(block.tokenCount).toBe(0);
      expect(block.hashInput).toBe("image:url:https://cdn.example.com/huge.png");
    } finally {
      globalThis.fetch = savedFetch;
      net.Socket.prototype.connect = savedConnect;
    }
  });

  it("a remote low-detail URL still reports its flat cost", () => {
    const block = oneImageBlock(new OpenAIAdapter(), {
      messages: [
        {
          role: "user",
          content: [
            { type: "image_url", image_url: { url: "https://cdn.example.com/a.png", detail: "low" } },
          ],
        },
      ],
    });
    expect(block.text).toBe("[image · ~85 tok]");
    expect(block.tokenCount).toBe(85);
  });

  it("an unknown image format still becomes an image block, keyed on its bytes", () => {
    const block = oneImageBlock(new OpenAIAdapter(), {
      messages: [
        {
          role: "user",
          content: [
            { type: "image_url", image_url: { url: uri(cat("BM", new Uint8Array(60)), "image/bmp") } },
          ],
        },
      ],
    });
    expect(block.text).toBe("[image]");
    expect(block.hashInput?.startsWith("image:sha256:")).toBe(true);
  });

  it("an implausible header degrades to a bare image block", () => {
    // End to end for the clamp: a hostile PNG header still becomes an image
    // block (it IS an image, and its bytes are still its identity) — but with no
    // size and no estimate, so one broken header cannot dominate the run's token
    // attribution or its descriptor.
    const monster = png(0xffffffff, 0xffffffff);
    const raw = imageRawBlock("user", { type: "image_url", image_url: { url: uri(monster) } }, "gemini")!;
    expect(raw.text).toBe("[image]");
    expect(raw.tokenCount).toBe(0);
    expect(raw.hashInput?.startsWith("image:sha256:")).toBe(true);
  });

  it("an undecodable payload falls back to the previous content_part behavior", () => {
    // Falling back is always safe; inventing an empty-bytes image block would
    // make every malformed image in a trace dedup into one bogus row.
    const blocks = new AnthropicAdapter().extractBlocks({
      messages: [{ role: "user", content: [{ type: "image", source: { type: "base64", data: "!!!!" } }] }],
    });
    expect(blocks.map((b) => b.kind)).toEqual(["content_part"]);
  });

  it("the detector never throws on garbage", () => {
    for (const part of [
      null,
      42,
      "a string",
      [],
      {},
      { type: "image_url" },
      { type: "image", source: null },
      { type: "image", source: {} },
      { inline_data: "not an object" },
      { image: { source: 7 } },
      { type: "image_url", image_url: { url: 12345 } },
    ]) {
      expect(detectImagePart(part)).toBeNull();
    }
  });
});

// --- identity: dedup and non-collision ----------------------------------------

describe("identity is the pixels", () => {
  const blockFor = (part: unknown, provider = "openai") =>
    buildBlock(imageRawBlock("user", part, provider)!, provider);

  it("the same image sent twice is one block", () => {
    const part = { type: "image_url", image_url: { url: uri(PNG_1024x768), detail: "high" } };
    expect(blockFor(part).contentHash).toBe(blockFor({ ...part }).contentHash);
  });

  it("the same bytes dedup across providers and wrappers", () => {
    // Identity is the PIXELS, not the envelope — which is what makes a
    // screenshot traceable through a multi-provider agent. Each shape is built
    // under the provider that actually emits it, so this really is the
    // cross-provider case: the four vision formulas disagree (765 / 1049 / 1032
    // / 1049 for the same 1024×768 PNG) and the descriptor embeds whichever
    // count was computed — yet all four land on ONE hash.
    const blocks = SAME_PNG_ACROSS_PROVIDERS.map(([provider, part]) =>
      blockFor(part, provider),
    );
    expect(new Set(blocks.map((b) => b.contentHash)).size).toBe(1);
    expect(blocks.map((b) => b.tokenCount)).toEqual([765, 1049, 1032, 1049]);
    expect(new Set(blocks.map((b) => b.text)).size).toBeGreaterThan(1);
  });

  it("a cross-provider dedup keeps the first writer's count and text", () => {
    // The documented consequence of content-addressed dedup, asserted rather
    // than waved away (see `spec/ctrace-schema.md`, "Image blocks"): the same
    // screenshot sent to OpenAI and then to Anthropic is written ONCE, and
    // `INSERT OR IGNORE` keeps the first writer's `tokenCount` AND the count
    // embedded in its descriptor — so the Anthropic turn renders OpenAI's 765
    // where Anthropic's own formula says 1049. Not image-specific (a text block
    // deduped across an exact and an estimating provider has always behaved this
    // way); pinned here because for an image the number is also in `text`, where
    // a reader might take it for a per-call measurement.
    const path = join(tmpdir(), `ctxdiff-img-cross-${randomUUID()}.ctrace`);
    try {
      const ct = CTrace.create(path, "vision", "openai", "gpt-4o", "2026-04-11T09:30:00+00:00");
      SAME_PNG_ACROSS_PROVIDERS.slice(0, 2).forEach(([provider, part], i) => {
        ct.recordCall({
          seq: i + 1,
          params: { model: "m" },
          usage: null,
          latencyMs: 1,
          error: null,
          callBlocks: [
            { block: blockFor(part, provider), position: 0, label: "user", labelSource: "heuristic" },
          ],
        });
      });
      ct.close();

      const read = CTrace.open(path);
      const stored = read.getCalls().map((c) => read.getCallBlocks(c.id)[0].block);
      read.close();
      expect(stored[0].contentHash).toBe(stored[1].contentHash); // one block…
      expect(stored[1].tokenCount).toBe(765); // …OpenAI's count
      expect(stored[1].text).toBe("[image 1024×768 · ~765 tok]"); // …and its text
    } finally {
      for (const suffix of ["", "-wal", "-shm"]) rmSync(path + suffix, { force: true });
    }
  });

  it("two different images of the same size are two blocks", () => {
    // If identity were the descriptor they would collide and a CHANGED
    // screenshot would silently read as unchanged.
    const a = blockFor({ type: "image_url", image_url: { url: uri(PNG_1024x768), detail: "high" } });
    const b = blockFor({
      type: "image_url",
      image_url: { url: uri(PNG_1024x768_ALT), detail: "high" },
    });
    expect(a.text).toBe(b.text);
    expect(a.contentHash).not.toBe(b.contentHash);
  });

  it("base64 formatting differences do not change identity", () => {
    // Padding, embedded newlines and the URL-safe alphabet are formatting, not
    // content — otherwise one client's line-wrapping would defeat dedup.
    const plain = b64(PNG_1024x768);
    const variants = [
      plain,
      plain.replace(/=+$/, ""),
      (plain.match(/.{1,60}/g) ?? []).join("\n"),
      plain.replace(/\+/g, "-").replace(/\//g, "_"),
    ];
    const hashes = new Set(
      variants.map(
        (v) => blockFor({ type: "image", source: { type: "base64", data: v } }, "anthropic").contentHash,
      ),
    );
    expect(hashes.size).toBe(1);
  });
});

// --- identity: the cost-affecting envelope ------------------------------------

describe("identity covers what the payload COSTS and DOES", () => {
  const blockFor = (part: unknown, provider = "openai") =>
    buildBlock(imageRawBlock("user", part, provider)!, provider);

  it("the same screenshot at two detail levels is two blocks", () => {
    // THE computer-use pattern: the same screenshot sits in history at
    // detail:"low" and is re-sent for the current turn at detail:"high".
    // `detail` changes the cost NINE-fold (85 vs 765) and changes the
    // descriptor, so the two must be two blocks. Collapsed into one, INSERT OR
    // IGNORE keeps whichever was written first and `ctxdiff tokens` prints 85
    // where the truth is 765 — a silent Δ +689 against the provider's usage.
    const forDetail = (detail: string) =>
      blockFor({ type: "image_url", image_url: { url: uri(PNG_1024x768), detail } });
    const low = forDetail("low");
    const high = forDetail("high");
    expect(low.contentHash).not.toBe(high.contentHash);
    expect([low.tokenCount, high.tokenCount]).toEqual([85, 765]);
    expect(low.text).toBe("[image 1024×768 · ~85 tok]");
    expect(high.text).toBe("[image 1024×768 · ~765 tok]");
  });

  it("a detail change reads as a modified block in the diff", () => {
    // The user-visible half of the same defect: `ctxdiff diff` must report the
    // low→high promotion as a CHANGE. While `detail` was outside the identity
    // the two turns shared one block and the diff called a 9× cost increase
    // 'unchanged'.
    const path = join(tmpdir(), `ctxdiff-img-detail-${randomUUID()}.ctrace`);
    try {
      const ct = CTrace.create(path, "vision", "openai", "gpt-4o", "2026-04-11T09:30:00+00:00");
      ["low", "high"].forEach((detail, i) => {
        const block = blockFor({ type: "image_url", image_url: { url: uri(PNG_1024x768), detail } });
        ct.recordCall({
          seq: i + 1,
          params: { model: "gpt-4o" },
          usage: null,
          latencyMs: 1,
          error: null,
          callBlocks: [{ block, position: 0, label: "user", labelSource: "heuristic" }],
        });
      });
      ct.close();

      const read = CTrace.open(path);
      const diff = diffTurns(read, 1, 2);
      read.close();
      expect(diff.entries.map((e) => e.kind)).toEqual(["modified"]);
      expect([diff.tokensAdded, diff.tokensEvicted]).toEqual([765, 85]);
    } finally {
      for (const suffix of ["", "-wal", "-shm"]) rmSync(path + suffix, { force: true });
    }
  });

  it("a cache_control breakpoint changes an image's identity", () => {
    // Anthropic's `cache_control` is the single most cache-relevant key a
    // content part can carry, and it rides as a SIBLING of the image payload.
    // Rebuilding the block from the payload alone dropped it, so adding or
    // removing a cache breakpoint on an image stopped changing the block's hash
    // and `ctxdiff cache` reported a stable prefix across a real caching change
    // — the exact failure class the profiler exists to catch, and asymmetric
    // with a text part, whose hash DOES move.
    const source = { type: "base64", media_type: "image/png", data: b64(PNG_1024x768) };
    const marked = { type: "image", source, cache_control: { type: "ephemeral" } };
    const plain = imageRawBlock("user", { type: "image", source }, "anthropic")!;
    const withBreakpoint = imageRawBlock("user", marked, "anthropic")!;
    const again = imageRawBlock("user", { ...marked }, "anthropic")!;

    expect(plain.hashInput).not.toBe(withBreakpoint.hashInput); // identity…
    expect(withBreakpoint.hashInput).toBe(again.hashInput); // …but a stable one
    expect(plain.text).toBe(withBreakpoint.text); // and it changes nothing else
    expect(plain.tokenCount).toBe(withBreakpoint.tokenCount);
  });

  it("an unrecognized sibling key folds into identity", () => {
    // The general rule behind the cache_control fix: every key ctxdiff
    // understands is already represented in the block (the payload IS the hash;
    // `detail` is appended to it; the media type is deliberately excluded).
    // Anything left over is a modifier ctxdiff cannot interpret and therefore
    // must not silently discard — a provider-side behavior change with no trace
    // in the trace is worse than an over-eager hash.
    const part = { type: "image_url", image_url: { url: uri(PNG_1024x768) } };
    const plain = imageRawBlock("user", part, "openai")!;
    const extra = imageRawBlock("user", { ...part, x_provider_hint: "grounding" }, "openai")!;
    expect(plain.hashInput).not.toBe(extra.hashInput);
  });

  it("the standard shapes carry no leftover envelope", () => {
    // The other side of the rule: folding the remainder in must NOT defeat
    // dedup for the shapes providers actually emit. Every documented wrapper has
    // an empty remainder, so a plain OpenAI data URI and a plain Anthropic
    // base64 source still hash identically.
    const digest = "image:sha256:" + createHash("sha256").update(PNG_1024x768).digest("hex");
    for (const [provider, part] of SAME_PNG_ACROSS_PROVIDERS) {
      expect(imageRawBlock("user", part, provider)!.hashInput).toBe(digest);
    }
  });
});

// --- how the block reaches the store ------------------------------------------

describe("buildBlock, the single definition of a stored block", () => {
  it("uses the image overrides and not the tokenizer", () => {
    // Tokenizing the `[image …]` descriptor would report about seven tokens for
    // a full-page screenshot.
    const raw = imageRawBlock(
      "user",
      { type: "image_url", image_url: { url: uri(PNG_1024x768), detail: "high" } },
      "openai",
    )!;
    const block = buildBlock(raw, "openai");
    expect(block.kind).toBe("image");
    expect(block.tokenCount).toBe(765);
    expect(block.tokenMethod).toBe("estimate");
    expect(block.contentHash).toBe(contentHash("user", "image", raw.hashInput!));
  });

  it("leaves ordinary blocks exactly as before", () => {
    // The overrides are opt-in. This is the back-compatibility of the writer
    // path: every pre-existing block hashes and tokenizes its own text.
    const block = buildBlock({ role: "user", kind: "message", text: "hello world" }, "openai");
    expect(block.contentHash).toBe(contentHash("user", "message", "hello world"));
    expect([block.tokenCount, block.tokenMethod]).toEqual(countTokens("hello world", "openai"));
  });
});

describe("end to end through the recorder and the store", () => {
  it("a vision call marks the whole turn approximate", () => {
    // The honesty guarantee: because an image block carries
    // tokenMethod="estimate", the call is reported as approximate — so a vision
    // estimate can never be rendered as an exact tiktoken count, in this reader
    // or in an older one.
    const path = join(tmpdir(), `ctxdiff-img-${randomUUID()}.ctrace`);
    try {
      const ct = CTrace.create(path, "vision", "openai", "gpt-4o", "2026-04-11T09:30:00+00:00");
      new Recorder(ct, new OpenAIAdapter(), null).record({
        seq: 1,
        kwargs: {
          model: "gpt-4o",
          messages: [
            {
              role: "user",
              content: [
                { type: "text", text: "what is this?" },
                { type: "image_url", image_url: { url: uri(PNG_1024x768), detail: "high" } },
              ],
            },
          ],
        },
        response: null,
        latencyMs: 5,
        error: null,
        tagged: [],
      });
      ct.close();

      const read = CTrace.open(path);
      const call = read.getCalls()[0];
      const blocks = read.getCallBlocks(call.id);
      const image = blocks.find((cb) => cb.block.kind === "image")!;
      expect(image.block.text).toBe("[image 1024×768 · ~765 tok]");
      expect(analyzeCall(call, blocks).approximate).toBe(true);
      read.close();
    } finally {
      for (const suffix of ["", "-wal", "-shm"]) rmSync(path + suffix, { force: true });
    }
  });

  it("the stored .ctrace never contains the base64", () => {
    // The size half of the fix: a 100 KB screenshot used to be written verbatim
    // into `block.text` (and into every HTML export of that trace); now the row
    // holds a 27-character descriptor.
    const big = png(1024, 768, new Uint8Array(200_000));
    const path = join(tmpdir(), `ctxdiff-img-big-${randomUUID()}.ctrace`);
    try {
      const ct = CTrace.create(path, "vision", "openai", "gpt-4o", "2026-04-11T09:30:00+00:00");
      new Recorder(ct, new OpenAIAdapter(), null).record({
        seq: 1,
        kwargs: {
          model: "gpt-4o",
          messages: [{ role: "user", content: [{ type: "image_url", image_url: { url: uri(big) } }] }],
        },
        response: null,
        latencyMs: 5,
        error: null,
        tagged: [],
      });
      ct.close();

      // Compared as raw BYTES, not as a decoded string: the file is a SQLite
      // database (arbitrary binary) and the descriptor's `×` is multi-byte
      // UTF-8, so any single text decoding would mangle one side or the other.
      const raw = readFileSync(path);
      expect(raw.includes(Buffer.from(b64(big).slice(0, 64), "ascii"))).toBe(false);
      expect(raw.includes(Buffer.from("[image 1024×768", "utf8"))).toBe(true);
    } finally {
      for (const suffix of ["", "-wal", "-shm"]) rmSync(path + suffix, { force: true });
    }
  });
});

// --- back-compatibility -------------------------------------------------------

describe("back-compatibility", () => {
  it("a pre-existing v2 file with a JSON-dumped image block still opens and renders", async () => {
    // Files captured before this change hold the image as a `content_part` whose
    // text is the JSON-serialized part, tiktoken-counted. Nothing migrates them
    // — a debugger must not rewrite the evidence it inspects — so they must keep
    // opening with their original hash, kind, text and token method intact.
    const { DatabaseSync } = await import("node:sqlite");
    const path = join(tmpdir(), `ctxdiff-legacy-${randomUUID()}.ctrace`);
    try {
      const legacyText = stableStringify({
        type: "image_url",
        image_url: { url: uri(PNG_1024x768), detail: "high" },
      });
      const hash = contentHash("user", "content_part", legacyText);
      const [tokens, method] = countTokens(legacyText, "openai");
      const [label, labelSource] = basicLabel("user", "content_part", legacyText, []);

      const db = new DatabaseSync(path);
      db.exec(DDL);
      db.prepare("INSERT INTO run VALUES (?,?,?,?,?,?,?)").run(
        "a".repeat(32),
        "legacy",
        "2026-01-01T00:00:00+00:00",
        "openai",
        JSON.stringify(["gpt-4o"]),
        "0.1.0",
        SCHEMA_VERSION,
      );
      db.prepare("INSERT INTO call VALUES (?,?,?,?,?,?,?,?,?,?)").run(
        "b".repeat(32),
        "a".repeat(32),
        1,
        JSON.stringify({ model: "gpt-4o" }),
        null,
        10,
        null,
        null,
        null,
        "openai",
      );
      db.prepare("INSERT INTO block VALUES (?,?,?,?,?,?)").run(
        hash,
        "user",
        "content_part",
        legacyText,
        tokens,
        method,
      );
      db.prepare("INSERT INTO call_block VALUES (?,?,?,?,?)").run(
        "b".repeat(32),
        hash,
        0,
        label,
        labelSource,
      );
      db.close();

      const ct = CTrace.open(path);
      const blocks = ct.getCallBlocks(ct.getCalls()[0].id);
      expect(blocks).toHaveLength(1);
      expect(blocks[0].block.kind).toBe("content_part");
      expect(blocks[0].block.text).toBe(legacyText);
      expect(blocks[0].block.contentHash).toBe(hash);
      expect(blocks[0].block.tokenMethod).toBe("tiktoken");
      ct.close();
    } finally {
      for (const suffix of ["", "-wal", "-shm"]) rmSync(path + suffix, { force: true });
    }
  });

  it("an image block is written at the unchanged schema version 2", async () => {
    // The change is additive — a new `kind` value and a new text convention in
    // columns that already exist — so an older ctxdiff opens the file and simply
    // shows the descriptor. No version bump, no rejection.
    const { DatabaseSync } = await import("node:sqlite");
    const path = join(tmpdir(), `ctxdiff-imgver-${randomUUID()}.ctrace`);
    try {
      const ct = CTrace.create(path, "vision", "openai", "gpt-4o", "2026-04-11T09:30:00+00:00");
      new Recorder(ct, new OpenAIAdapter(), null).record({
        seq: 1,
        kwargs: {
          model: "gpt-4o",
          messages: [
            { role: "user", content: [{ type: "image_url", image_url: { url: uri(PNG_1024x768) } }] },
          ],
        },
        response: null,
        latencyMs: 5,
        error: null,
        tagged: [],
      });
      ct.close();

      const db = new DatabaseSync(path);
      const row = db.prepare("SELECT schema_version FROM run").get() as { schema_version: number };
      db.close();
      expect(row.schema_version).toBe(SCHEMA_VERSION);
      expect(SCHEMA_VERSION).toBe(2);

      const read = CTrace.open(path);
      const block = read.getCallBlocks(read.getCalls()[0].id)[0].block;
      expect(block.kind).toBe("image");
      // A reader with no knowledge of the `image` kind labels by role, and
      // `basicLabel` has no image branch — so an OLD ctxdiff computes the same
      // label this one stores.
      expect(basicLabel(block.role, block.kind, block.text, [])[0]).toBe("user");
      read.close();
    } finally {
      for (const suffix of ["", "-wal", "-shm"]) rmSync(path + suffix, { force: true });
    }
  });
});
