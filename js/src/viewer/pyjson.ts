/**
 * A JSON serializer that matches Python's `json.dumps(obj, ensure_ascii=False)`
 * byte-for-byte, plus an HTML escaper matching Python's `html.escape`. These let
 * the JS viewer embed the exact same JSON island and title the Python viewer
 * does, so the rendered dashboards are byte-identical.
 *
 * Why not `JSON.stringify`: CPython's json emits `", "` and `": "` separators
 * (with spaces) and preserves dict INSERTION order; `JSON.stringify` uses
 * compact `,`/`:` — so the embedded payload would differ. String escaping and
 * integer formatting already match between the two, so string leaves reuse
 * `JSON.stringify`; only container spacing and float formatting need care.
 */

/**
 * Wrapper marking a value that must serialize as a Python float (e.g. `100.0`,
 * not `100`). JS has no runtime int/float distinction, but Python's json emits
 * `float.__repr__` — so an integer-valued float like a `100.0` percentage would
 * differ (`100` vs `100.0`). Wrap the payload's float fields (percentages) in
 * this so `pyJsonDumps` reproduces Python's spelling exactly.
 */
export class PyFloat {
  constructor(public readonly value: number) {}
}

/** Format a number as Python's json would. Integers print bare; a `PyFloat`
 * whose value is integral gets a trailing `.0` (Python float repr), otherwise
 * the shortest round-tripping decimal (which JS and Python agree on).
 *
 * `String(n)` matches Python `repr` for every value the payload can hold —
 * bounded integers (token counts, positions, seqs) and 1-decimal percentages.
 * The only known divergence is exponent zero-padding for extreme magnitudes
 * (`String(1e-7)` → "1e-7" vs Python "1e-07"), which no payload field can reach,
 * so no guard is needed. */
function numberToPy(n: number): string {
  return String(n);
}

/**
 * Serialize `value` exactly like `json.dumps(value, ensure_ascii=False)`:
 * objects keep insertion order with `", "`/`": "` separators, arrays use `", "`,
 * strings reuse `JSON.stringify` (identical escaping + non-ASCII passthrough),
 * integers/booleans/null print as Python spells them, and `PyFloat` reproduces
 * Python's float repr. Unlike the hashing `stableStringify`, keys are NOT sorted
 * — the payload's key order already mirrors the Python builder's.
 */
export function pyJsonDumps(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "string") return JSON.stringify(value);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value instanceof PyFloat) {
    const v = value.value;
    if (!Number.isFinite(v)) return "null";
    return Number.isInteger(v) ? v.toFixed(1) : String(v);
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "null";
    return numberToPy(value);
  }
  if (Array.isArray(value)) {
    return "[" + value.map((v) => pyJsonDumps(v)).join(", ") + "]";
  }
  if (typeof value === "object") {
    const obj = value as Record<string, unknown>;
    const parts: string[] = [];
    for (const k of Object.keys(obj)) {
      parts.push(JSON.stringify(k) + ": " + pyJsonDumps(obj[k]));
    }
    return "{" + parts.join(", ") + "}";
  }
  return "null";
}

/**
 * Escape a string for HTML text/attribute context exactly like Python's
 * `html.escape(s)` (quote=True default): `&`→`&amp;`, `<`→`&lt;`, `>`→`&gt;`,
 * `"`→`&quot;`, `'`→`&#x27;`, in that order (ampersand first). Used for the
 * `<title>` text.
 */
export function htmlEscape(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#x27;");
}
