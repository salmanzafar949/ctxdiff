import { describe, it, expect } from "vitest";
import {
  contentHash,
  normalizeText,
  stableStringify,
  basicLabel,
} from "../src/models.js";

describe("stableStringify (Python json.dumps parity)", () => {
  it("sorts object keys recursively and uses ', '/': ' separators", () => {
    // Byte-for-byte what CPython's json.dumps(sort_keys=True) emits.
    expect(stableStringify({ b: 2, a: 1, z: [3, 2, { y: 1, x: 2 }] })).toBe(
      '{"a": 1, "b": 2, "z": [3, 2, {"x": 2, "y": 1}]}',
    );
  });

  it("preserves array order", () => {
    expect(stableStringify([3, 1, 2])).toBe("[3, 1, 2]");
  });

  it("does not escape non-ASCII (ensure_ascii=False)", () => {
    expect(stableStringify({ name: "café", emoji: "😀" })).toBe(
      '{"emoji": "😀", "name": "café"}',
    );
  });

  it("spells booleans/null like Python", () => {
    expect(stableStringify({ a: null, b: true, c: false })).toBe(
      '{"a": null, "b": true, "c": false}',
    );
  });
});

describe("normalizeText", () => {
  it("passes strings through verbatim", () => {
    expect(normalizeText("hi")).toBe("hi");
    expect(normalizeText("café 😀")).toBe("café 😀");
  });
  it("stable-serializes non-strings", () => {
    expect(normalizeText({ b: 1, a: 2 })).toBe('{"a": 2, "b": 1}');
  });
});

describe("contentHash", () => {
  it("GOLDEN: contentHash('user','message','hi')", () => {
    expect(contentHash("user", "message", "hi")).toBe(
      "4e6c4093072114cd3ec3641653e12f750391cded3515bf460ccd07162c647685",
    );
  });

  it("is invariant to object key order (dict/multipart case)", () => {
    const a = contentHash("user", "content_part", { a: 1, b: 2, c: 3 });
    const b = contentHash("user", "content_part", { c: 3, b: 2, a: 1 });
    expect(a).toBe(b);
  });

  it("cannot collide across the role/kind boundary (NUL separator)", () => {
    // ('a','bc') vs ('ab','c') must differ.
    expect(contentHash("a", "bc", "x")).not.toBe(contentHash("ab", "c", "x"));
  });

  it("is a 64-char lowercase hex digest", () => {
    expect(contentHash("system", "message", "anything")).toMatch(
      /^[0-9a-f]{64}$/,
    );
  });
});

describe("basicLabel", () => {
  it("maps roles heuristically", () => {
    expect(basicLabel("system", "message", "x", [])).toEqual([
      "system",
      "heuristic",
    ]);
    expect(basicLabel("tool", "message", "x", [])).toEqual([
      "tool_output",
      "heuristic",
    ]);
    expect(basicLabel("user", "message", "x", [])).toEqual([
      "user",
      "heuristic",
    ]);
    expect(basicLabel("assistant", "message", "x", [])).toEqual([
      "history",
      "heuristic",
    ]);
  });

  it("labels tool schemas as tool_schema", () => {
    expect(basicLabel("system", "tool_schema", "{}", [])).toEqual([
      "tool_schema",
      "heuristic",
    ]);
  });

  it("falls back to the raw role for unknown roles", () => {
    expect(basicLabel("weird", "message", "x", [])).toEqual([
      "weird",
      "heuristic",
    ]);
  });

  it("a tag substring match wins with source 'tagged'", () => {
    expect(
      basicLabel("user", "message", "please retrieve DOC-42 now", [
        ["rag", "DOC-42"],
      ]),
    ).toEqual(["rag", "tagged"]);
  });

  it("first registered tag wins", () => {
    expect(
      basicLabel("user", "message", "abc", [
        ["first", "b"],
        ["second", "c"],
      ]),
    ).toEqual(["first", "tagged"]);
  });
});
