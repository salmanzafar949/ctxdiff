/**
 * CPython-compatible rounding for the two shapes the analyzers/renderers need.
 * Parity requires matching Python's `round`, which is round-HALF-TO-EVEN on the
 * true value of the double (JS `Math.round` is half-up, so it diverges on ties).
 *
 * The inputs here are all `x = someInt / someInt * scale` computed with the
 * SAME IEEE-754 operations in both languages, so the double `x` is bit-identical
 * across JS and Python — only the rounding step must be matched.
 */

/**
 * `round(x, 1)` — round to 1 decimal place. A true tie at a `.?5` boundary
 * would require `x` to exactly equal `(2n+1)/20`, which is never dyadic and so
 * never exactly representable as a double; therefore there is never a real tie
 * at 1 decimal, and the uniquely-closest value is correct regardless of
 * half-to-even vs half-up. `toFixed(1)` selects that closest value from the
 * exact value of `x` (no error-prone `*10`), and parsing it back yields exactly
 * what CPython's `round(x, 1)` returns.
 */
export function pyRound1(x: number): number {
  if (!Number.isFinite(x)) return x;
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
