import { describe, it, expect } from "vitest";
import { SequenceMatcher } from "../src/analyze/sequence-matcher.js";

/** Turn opcodes into the [tag, i1, i2, j1, j2] tuples Python's
 * `SequenceMatcher.get_opcodes()` returns, for direct comparison. */
function tuples(a: string[], b: string[]) {
  return new SequenceMatcher(a, b).getOpcodes().map((o) => [o.tag, o.i1, o.i2, o.j1, o.j2]);
}

describe("SequenceMatcher port (matches difflib.SequenceMatcher opcodes)", () => {
  // Reference values captured from CPython difflib (autojunk=False).
  it("replace in the middle", () => {
    expect(tuples(["a", "b", "c"], ["a", "x", "c"])).toEqual([
      ["equal", 0, 1, 0, 1],
      ["replace", 1, 2, 1, 2],
      ["equal", 2, 3, 2, 3],
    ]);
  });

  it("delete then equal", () => {
    expect(tuples(["a", "b", "c", "d"], ["a", "c", "d"])).toEqual([
      ["equal", 0, 1, 0, 1],
      ["delete", 1, 2, 1, 1],
      ["equal", 2, 4, 1, 3],
    ]);
  });

  it("swap surfaces as insert + equal + delete (not a move)", () => {
    expect(tuples(["a", "b"], ["b", "a"])).toEqual([
      ["insert", 0, 0, 0, 1],
      ["equal", 0, 1, 1, 2],
      ["delete", 1, 2, 2, 2],
    ]);
  });

  it("kitten -> sitting (classic)", () => {
    expect(tuples("kitten".split(""), "sitting".split(""))).toEqual([
      ["replace", 0, 1, 0, 1],
      ["equal", 1, 4, 1, 4],
      ["replace", 4, 5, 4, 5],
      ["equal", 5, 6, 5, 6],
      ["insert", 6, 6, 6, 7],
    ]);
  });

  it("leading replace, trailing equal run", () => {
    expect(tuples(["s", "t1", "t2", "u"], ["s2", "t1", "t2", "u"])).toEqual([
      ["replace", 0, 1, 0, 1],
      ["equal", 1, 4, 1, 4],
    ]);
  });

  it("identical sequences are one equal block", () => {
    expect(tuples(["x", "y", "z"], ["x", "y", "z"])).toEqual([["equal", 0, 3, 0, 3]]);
  });

  it("empty vs non-empty is a pure insert", () => {
    expect(tuples([], ["a", "b"])).toEqual([["insert", 0, 0, 0, 2]]);
  });
});
