/**
 * CPython-compatible rounding, number FORMATTING and value REPR — the small set
 * of "spell it exactly the way Python would" primitives the analyzers and
 * renderers share. Everything here has exactly one job: make a value that
 * crossed a language boundary print the same on both sides.
 * Parity requires matching Python's `round`, which is round-HALF-TO-EVEN on the
 * true value of the double (JS `Math.round` is half-up, so it diverges on ties).
 *
 * The inputs here are all `x = someInt / someInt * scale` computed with the
 * SAME IEEE-754 operations in both languages, so the double `x` is bit-identical
 * across JS and Python — only the rounding step must be matched.
 */

/**
 * `round(x, 1)` — round to 1 decimal place, CPython's way: to nearest, TIES TO
 * EVEN, on the exact value of the double.
 *
 * Ties are rare but they are real, and this function used to get them wrong. The
 * old reasoning was that a tie needs `x === (2n+1)/20`, which is never dyadic
 * and so never exactly a double — but that is only true of ties ending in a
 * lone `5` at the second decimal with an odd first decimal. The ties that DO
 * occur are the ones whose exact value ends `.25` or `.75`: those are `odd/4`,
 * perfectly representable, and they come out of ordinary ctxdiff arithmetic —
 * `61 / 80 * 100` is exactly `76.25`, and a 1-token slice of a 16-token turn is
 * exactly `6.25`. On those, `toFixed(1)` rounds half AWAY from zero (`76.3`,
 * `6.3`) while CPython rounds half to even (`76.2`, `6.2`), and the two SDKs
 * printed different percentages for the same trace.
 *
 * The fix, in two branches:
 *
 * - `x * 4` an ODD INTEGER is exactly the tie condition. A double whose exact
 *   decimal expansion terminates at the second place with a `5` must be `m/100`
 *   with `m` ending in 25 or 75, i.e. `odd/4`; and multiplying a double by 4 is
 *   exact, so the test itself introduces no error. `x * 10` is then exact too
 *   — SO LONG AS the result still fits the 53-bit significand: `odd/4 * 10` is
 *   the half-integer `odd*5/2`, which is representable up to about 2^52 and
 *   rounds above it (`2251799813685247.75 * 10` lands on `…2476`, and this
 *   branch would then answer `…247.5` where CPython says `…247.8`). So the
 *   branch is entered only when `scaled * 2` — the smallest integer that pins
 *   the half — is still a safe integer. Inside that domain, flooring `scaled`
 *   and picking the EVEN neighbour reproduces CPython's tie rule, including for
 *   negatives, where `Math.floor` already steps the right way.
 * - everything else — every non-tie, and the huge ties the guard rejects — has a
 *   uniquely closest 1-decimal value that `toFixed(1)` selects from the exact
 *   value of `x` (no error-prone `* 10`), which is what CPython's `round`
 *   returns. `toFixed` switches to exponential notation at |x| ≥ 1e21, but
 *   `parseFloat` reads that back to the same double, so the VALUE this function
 *   returns is still CPython's; only a caller that formats the result with
 *   `toFixed` again would see `1e+21` where Python's `:.1f` writes the digits
 *   out. No ctxdiff caller can reach there: every input is a percentage of a
 *   positive window (see `analyze/window.ts`) or a share of a turn's tokens.
 *
 * `-0` is returned as `-0`, matching CPython's `round(-0.0, 1) == -0.0`. It has
 * to be special-cased because `(-0).toFixed(1)` is the string `"0.0"` — the one
 * place `toFixed` loses the sign, since `(-0.04).toFixed(1)` correctly gives
 * `"-0.0"`.
 */
export function pyRound1(x: number): number {
  if (!Number.isFinite(x)) return x;
  if (x === 0) return x; // preserves -0, which `toFixed` would flatten to "0.0"
  const quarters = x * 4;
  if (Number.isInteger(quarters) && Math.abs(quarters) % 2 === 1) {
    const scaled = x * 10; // a half-integer — exact only while it fits
    if (Number.isSafeInteger(scaled * 2)) {
      const lower = Math.floor(scaled);
      // `lower % 2` is 0, 1 or -1; only an exactly-even lower neighbour wins.
      const chosen = lower % 2 === 0 ? lower : lower + 1;
      return chosen / 10;
    }
  }
  return parseFloat(x.toFixed(1));
}

/**
 * `round(x)` — round to the nearest integer, ties to even. Unlike the 1-decimal
 * case, a tie (`x === k + 0.5`) IS exactly representable (0.5 is dyadic), so the
 * tie must be resolved to the even neighbor to match CPython. Inputs here are
 * non-negative (a proportional bar length), so only the positive branch matters.
 */
export function pyRoundHalfEven(x: number): number {
  const floor = Math.floor(x);
  const frac = x - floor;
  if (frac < 0.5) return floor;
  if (frac > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

/**
 * Integer thousands-separator formatting, matching Python's `{n:,}` — `1234` →
 * `"1,234"`, `-1234` → `"-1,234"`. The magnitude is truncated toward zero
 * first, so a float that reached here by arithmetic renders as the integer
 * Python's `:,` on an int would.
 *
 * Lives here, beside the rounding helpers, because it is the same KIND of thing
 * — a CPython number-formatting rule reproduced once — and because three
 * separate copies of it (the renderer, the CLI's agent table and now the check
 * report) is three chances for one of them to drift on the negative branch and
 * break byte-identity in exactly one command.
 */
export function pyComma(n: number): string {
  const neg = n < 0;
  const grouped = Math.abs(Math.trunc(n))
    .toString()
    .replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  return neg ? "-" + grouped : grouped;
}

// A code point is non-printable (in Python's `str.isprintable` sense) when its
// Unicode category is "Other" (C*: control, format, surrogate, private-use,
// unassigned) or "Separator" (Z*: line/paragraph/space) — EXCEPT the ASCII
// space U+0020, which is printable. This covers C0/C1 controls, DEL, NBSP,
// soft-hyphen, the bidi/zero-width format marks (U+200B–200F, U+202A–202E,
// U+2060, U+FEFF) and the line/paragraph separators, exactly as CPython does.
const NON_PRINTABLE = /[\p{C}\p{Z}]/u;
function isNonPrintable(ch: string): boolean {
  return ch !== " " && NON_PRINTABLE.test(ch);
}

/**
 * Python `repr(str)`: choose single quotes unless the string contains a single
 * quote and no double quote (then double); escape the quote and backslash, use
 * the `\n`/`\r`/`\t` shorthands, escape any non-printable code point (see
 * `isNonPrintable`) using Python's width-appropriate form — `\xNN` below 0x100,
 * `\uNNNN` in the BMP, `\UNNNNNNNN` for astral — and pass every printable
 * character (including café/emoji) through. Iterates code points (`for..of`)
 * like Python, so astral characters are handled as single units.
 *
 * Exported because diff snippets are not the only place Python's repr quoting
 * has to be reproduced: `SQLiteStore.openReader` interpolates a path with `!r`
 * on the Python side, and `JSON.stringify` there would print double quotes
 * where Python prints single ones.
 */
export function pyRepr(s: string): string {
  const quote = s.includes("'") && !s.includes('"') ? '"' : "'";
  let out = quote;
  for (const ch of s) {
    const o = ch.codePointAt(0)!;
    if (ch === quote || ch === "\\") out += "\\" + ch;
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (ch === "\t") out += "\\t";
    else if (isNonPrintable(ch)) {
      if (o < 0x100) out += "\\x" + o.toString(16).padStart(2, "0");
      else if (o < 0x10000) out += "\\u" + o.toString(16).padStart(4, "0");
      else out += "\\U" + o.toString(16).padStart(8, "0");
    } else out += ch;
  }
  return out + quote;
}
