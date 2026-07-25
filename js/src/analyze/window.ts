/**
 * The context-window resolver and the one place a "share of the window" is
 * phrased. A faithful port of Python `ctxdiff.analyze.window` — same priority
 * order, same environment variable, same alarm threshold, same rounding, same
 * strings.
 *
 * Why it exists: `18,400 tok` is a number nobody can act on;
 * `18,400 / 200,000 tok · 9.2%` is, and `164,000 / 200,000 tok · ⚠ 82.0%` is the
 * alarm that explains the bug someone is actually chasing — the provider
 * silently dropping the oldest half of the conversation.
 *
 * Why ctxdiff will not supply the denominator itself: there is deliberately NO
 * model→context-window table in this package, for the same reason there is no
 * price table. Those numbers change per model, per provider, per deployment and
 * per month, and a stale one does not degrade — it LIES, in the exact direction
 * that makes a gate pass. So the window is the user's to state, in one of two
 * places (`resolveContextWindow`):
 *
 * 1. `--context-window N` on the command — the most specific thing anyone said;
 * 2. `CTXDIFF_CONTEXT_WINDOW=N` in the environment — set once in a shell profile
 *    or a CI job, it makes `tokens`, `check`, `view` and `export` agree without
 *    re-typing, and it is what lets an exported dashboard show percentages with
 *    no flag at all. Same shape as the existing `CTXDIFF_STORE` convention.
 *
 * There is deliberately no third source. In particular the window is NOT
 * recorded into the `.ctrace`: it never appears on the wire, so capture cannot
 * observe it, and a trace is evidence — a number baked into an evidence file
 * that nobody can verify from the file, and that a read-only debugger must not
 * rewrite, is a stale-metadata lie with no correction path.
 */
import { pyComma, pyRepr, pyRound1 } from "./pyround.js";

/** The environment variable that supplies a context window when no flag does.
 * Named like `CTXDIFF_STORE` so the two read as one family of settings. */
export const CONTEXT_WINDOW_ENV = "CTXDIFF_CONTEXT_WINDOW";

/**
 * At or above this percentage of the window, a turn is rendered with a warning
 * marker rather than a bare number.
 *
 * 80% and not 90%: the thing being warned about is not "you exceeded the window"
 * (the provider would have errored, and you would know) but the SILENT failure
 * just below it — a framework's sliding-window trimmer, a `max_tokens`
 * reservation for the response, or a single large tool result arriving next
 * turn. By 80% the remaining headroom is smaller than one typical tool output,
 * so the next turn is where content starts disappearing. Compared against the
 * DISPLAYED (1-decimal) percentage, so the marker never contradicts the number
 * printed beside it.
 */
export const CONTEXT_WINDOW_ALARM_PCT = 80.0;

/** The marker prefixed to an alarming percentage. The same `⚠` the cache
 * profiler and the bloat line use, so one glyph means "ctxdiff wants your
 * attention" everywhere. */
export const ALARM_MARKER = "⚠ ";

/**
 * A context window that was supplied but cannot be used — a non-numeric or
 * non-positive `CTXDIFF_CONTEXT_WINDOW`, or a non-positive `--context-window`.
 * Its own class so the CLI can report it as the usage error it is (exit 2)
 * rather than as a failure to read a trace.
 */
export class ContextWindowError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ContextWindowError";
  }
}

/**
 * The integer grammar the environment variable accepts: optional surrounding
 * whitespace, an optional sign, then ASCII decimal digits. IDENTICAL to the
 * CLI's `--turn`/`--context-window` grammar, and narrower than a bare
 * `parseInt` — which would happily read `200k` as 200 and `1e5` as 1.
 */
const INT_RE = /^\s*[+-]?[0-9]+\s*$/;

/**
 * Parse an environment-supplied context window into a positive integer. Throws
 * `ContextWindowError` naming the variable and echoing what was found for
 * anything that is not ASCII digits, or that parses to zero or less: a zero
 * window is a division by zero and a negative one is not a window, and both are
 * far better reported than silently ignored — a percentage that quietly stops
 * rendering looks exactly like a percentage that is fine.
 */
export function parseContextWindow(text: string): number {
  if (!INT_RE.test(text)) {
    throw new ContextWindowError(
      `ctxdiff: ${CONTEXT_WINDOW_ENV} must be a whole number of tokens ` +
        `(got ${pyRepr(text)})`,
    );
  }
  const value = parseInt(text.trim(), 10);
  if (value <= 0) {
    throw new ContextWindowError(
      `ctxdiff: ${CONTEXT_WINDOW_ENV} must be greater than 0 (got ${value})`,
    );
  }
  return value;
}

/**
 * The ONE window-resolution path, shared by `tokens`, `check`, `view` and
 * `export` so no two commands can disagree about the denominator on the same
 * machine: an explicit `--context-window` if given, else
 * `CTXDIFF_CONTEXT_WINDOW` if set to something usable, else null ("no window
 * known" — render nothing rather than guess).
 *
 * An empty or whitespace-only environment value is treated as UNSET rather than
 * as an error: `CTXDIFF_CONTEXT_WINDOW=` is how a shell unsets a variable for
 * one command, and failing there would make that idiom unusable.
 *
 * The POSITIVITY rule belongs to the window itself, not to the place it was
 * typed, so it is enforced here rather than in `parseContextWindow` alone: a
 * zero window is a division by zero and a negative one is not a window,
 * whichever source produced it. Enforcing it in the one shared resolver is what
 * makes all four commands inherit it — `tokens`, `view` and `export` used to
 * take the flag on trust and render `⚠ Infinity%` (a `-260.0%` turn header for a
 * negative one), while only `check` and the environment path said no.
 */
export function resolveContextWindow(
  flag: number | null,
  env: Record<string, string | undefined> = process.env,
): number | null {
  if (flag != null) {
    if (flag <= 0) {
      throw new ContextWindowError(
        `ctxdiff: --context-window must be greater than 0 (got ${flag})`,
      );
    }
    return flag;
  }
  const raw = env[CONTEXT_WINDOW_ENV];
  if (raw == null || raw.trim() === "") return null;
  return parseContextWindow(raw);
}

/**
 * `total` as a percentage of `window`, rounded to one decimal the way CPython's
 * `round` does. The ratio is computed with the same IEEE-754 operations in both
 * SDKs, so only the rounding step needs matching.
 */
export function windowPct(totalTokens: number, window: number): number {
  return pyRound1((totalTokens / window) * 100);
}

/**
 * Whether an already-rounded percentage has reached the alarm threshold.
 * Compared against the DISPLAYED value so `80.0%` and its marker always agree —
 * a turn shown as `80.0%` is marked, one shown as `79.9%` is not, with no
 * invisible third number deciding it.
 */
export function isAlarming(pct: number): boolean {
  return pct >= CONTEXT_WINDOW_ALARM_PCT;
}

/**
 * One turn's context as a share of the window: `18,400 / 200,000 tok · 9.2%`,
 * or `164,000 / 200,000 tok · ⚠ 82.0%` once the percentage reaches
 * `CONTEXT_WINDOW_ALARM_PCT`.
 *
 * The marker sits on the PERCENTAGE rather than at the end of the line because
 * the percentage is the thing that is alarming; a trailing glyph would read as a
 * comment on the whole row and would collide with the `(~approx)` marker the
 * caller appends after this string.
 */
export function formatWindowShare(totalTokens: number, window: number): string {
  const pct = windowPct(totalTokens, window);
  const marker = isAlarming(pct) ? ALARM_MARKER : "";
  return `${pyComma(totalTokens)} / ${pyComma(window)} tok · ${marker}${pct.toFixed(1)}%`;
}
