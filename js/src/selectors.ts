/**
 * Session/agent selection — the resolution layer behind every read command's
 * `--project` / `--session` / `--agent` / `--turn` flags. A faithful port of
 * Python `ctxdiff.cli.select`: same selector syntax, same defaulting rules, and
 * byte-identical error messages, because the two CLIs must be indistinguishable
 * from the outside.
 *
 * A project store holds MANY sessions and each session MANY agents, so a read
 * command has to answer three questions before it can analyze anything: which
 * project, which session, which agent. `cli.ts` owns the first (it knows the
 * store-opening rules); this module owns the other two, identically for every
 * command, so `diff`, `tokens`, `cache`, `export` and `view` can never disagree
 * about what `--session 4f3a2b1c` means.
 *
 * Three ideas carry the whole module:
 *
 * - A SELECTOR is a name with an OPTIONAL `:TURN` suffix — `researcher:8`,
 *   `4f3a2b1c:8`. That one piece of syntax is what makes a cross-session or
 *   cross-agent diff expressible with the flags that already exist
 *   (`--session GOOD:8 --session BAD:8`) instead of a bespoke sub-command.
 * - DEFAULTING is silent only when it cannot be wrong. Exactly one session in
 *   the project => no `--session` needed. Several => the user must say which,
 *   because quietly picking "the newest" would analyze a different run than the
 *   one they asked about and give them a confidently wrong answer.
 * - An unresolvable selector throws `SelectionError`, and its message ALREADY
 *   CONTAINS the listing of what could have been picked. The flag is a usage
 *   error (exit 2), and a usage error that doesn't show you the options is a
 *   dead end.
 *
 * Everything here is pure: nothing is opened, nothing is queried, and the only
 * ambient input is the local timezone offset used to render a stored UTC
 * timestamp (see `formatLocal`).
 */
import type { Call, Session } from "./models.js";
import { parseStartedAt } from "./store/ctrace.js";

/** How many leading characters of a session's uuid the CLI shows and accepts.
 * 12 hex chars is 48 bits — collision-free for any realistic project DB, short
 * enough to paste, and long enough that a prefix match is unambiguous. */
export const SHORT_ID_LEN = 12;

/** The bucket name for calls that carry no agent label, shared with the token
 * analyzer so one run never reports two different names for the same calls. */
export const UNLABELED = "(unlabeled)";

/**
 * A `--session`/`--agent` value that cannot be resolved, or a defaulting
 * decision that would have to be a guess.
 *
 * Its own error class so `cli.ts` can map it to argparse's usage-error exit
 * code 2 and print it VERBATIM — the message is built here complete with the
 * listing of available sessions/agents, because the only useful response to
 * "which session?" is the list of sessions. Mirrors Python `SelectionError`.
 */
export class SelectionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SelectionError";
  }
}

/**
 * A turn number as the user asked for it: the value calls are looked up by, and
 * the exact text to echo back when that lookup fails.
 *
 * Why both. Python's `--turn` is an arbitrary-precision `int`, so
 * `--turn 1000000000000000000000` comes back in "turn N not found" with all 22
 * digits. JS has only IEEE-754 doubles, which round that to `1e+21` — a number
 * the user never typed, in the one message whose whole job is to quote what they
 * did type. `text` is `String(BigInt(raw))`, which is exactly Python's
 * `str(int(raw))`: a leading `+`, surrounding whitespace and leading zeros
 * normalize away and every digit survives. `value` stays a plain number because
 * it is only ever compared against a stored `call.seq`, and a turn too large to
 * be an exact double is a turn no session has.
 */
export interface TurnArg {
  value: number;
  text: string;
}

/** Build a `TurnArg` from an already-validated decimal string (optional
 * surrounding whitespace, optional sign, ASCII digits). */
export function turnArg(raw: string): TurnArg {
  const trimmed = raw.trim();
  return { value: Number.parseInt(trimmed, 10), text: BigInt(trimmed).toString() };
}

/** Format a list of turn numbers like Python's list repr: `[1, 2, 3]` (space
 * after each comma), unlike `JSON.stringify` which omits the spaces. Accepts
 * `TurnArg`s so a turn the user typed is echoed with its own digits rather than
 * a double's rendering of them. */
export function turnList(turns: Array<TurnArg | number>): string {
  return "[" + turns.map((t) => (typeof t === "number" ? t : t.text)).join(", ") + "]";
}

/** One `--session`/`--agent` value: a `name` plus the turn it pinned, if any.
 * `turn` is null for a plain `--agent researcher`, and turn 8 for
 * `--agent researcher:8` — the suffix form that lets one flag name both a side
 * of a diff and the turn to take from it. */
export interface Selector {
  name: string;
  turn: TurnArg | null;
}

/**
 * Split a selector into its name and optional `:TURN` suffix.
 *
 * How: partition on the LAST colon and accept the suffix as a turn only when it
 * is a non-empty run of ASCII digits and something remains on the left.
 * Splitting from the right means a name that itself contains a colon keeps it
 * (`--agent tools:web:3` is agent `tools:web`, turn 3); the ASCII-digit test
 * matches Python's `tail.isascii() and tail.isdigit()`, which deliberately
 * rejects the superscripts and Arabic-Indic digits bare `str.isdigit()` accepts.
 * Anything else is a plain name, so a bare session id or agent name round-trips
 * untouched.
 */
export function parseSelector(value: string): Selector {
  const at = value.lastIndexOf(":");
  if (at > 0) {
    const tail = value.slice(at + 1);
    if (/^[0-9]+$/.test(tail)) {
      return { name: value.slice(0, at), turn: turnArg(tail) };
    }
  }
  return { name: value, turn: null };
}

/** The 12-character prefix of a session id used everywhere the CLI shows or
 * labels a session. Prefix rather than a hash so what is printed is also what
 * can be pasted straight back into `--session` (see `chooseSession`, which
 * accepts any unambiguous prefix). */
export function shortId(sessionId: string): string {
  return sessionId.slice(0, SHORT_ID_LEN);
}

/** Zero-pad an integer to `width`, matching Python's `f"{n:0Nd}"`. */
function pad(n: number, width = 2): string {
  return String(Math.abs(n)).padStart(width, "0");
}

/**
 * Render a session's stored UTC `startedAt` in the VIEWER's LOCAL timezone,
 * with the offset shown: `2026-07-24 16:03:11 +04:00`.
 *
 * Why local: a stored timestamp is UTC (canonical, portable), but the person
 * reading the listing is trying to match a session against "the run I did after
 * lunch". Showing the offset keeps it unambiguous when the trace is shared
 * across zones.
 *
 * How, and why it is byte-identical to the Python twin: the `Date` getters used
 * here (`getFullYear`, `getHours`, …) are the LOCAL-time getters and
 * `getTimezoneOffset()` reports the offset in effect AT THAT INSTANT — the same
 * zone and the same instant-sensitive offset Python's `astimezone()` uses, both
 * honoring `TZ`, so a summer timestamp renders in summer time on both sides.
 * `getTimezoneOffset()` returns minutes WEST of UTC (positive for the Americas),
 * so its sign is inverted to produce the conventional `+04:00`/`-05:00`.
 *
 * Degrades rather than throws: an empty `startedAt` (a trace created without
 * one) renders `-`, and a value that cannot be rendered as a local time is
 * echoed back unchanged, so a listing never dies over one odd row. That covers
 * one case `Date` alone would happily get "right": a stored timestamp whose
 * LOCAL shift lands outside year 1..9999. Python's `datetime` cannot represent
 * those at all (`astimezone()` raises OverflowError there and the raw text is
 * echoed), while `Date` spans ±273790 years and would print `10000-01-01` —
 * a rendering the other CLI can never produce. The range is enforced here so
 * both echo the same bytes, which is also the more honest answer.
 */
export function formatLocal(startedAt: string): string {
  const text = (startedAt ?? "").trim();
  if (!text) return "-";
  let d: Date;
  try {
    d = parseStartedAt(text);
  } catch {
    return text;
  }
  const year = d.getFullYear();
  if (year < 1 || year > 9999) return text;
  // `getTimezoneOffset()` reports WHOLE MINUTES west of UTC, truncated toward
  // zero for the sub-minute LMT offsets pre-1900 zones carry (America/New_York
  // was -04:56:02 until 1883). Python's twin truncates the same way rather than
  // flooring, which would put it a minute out for exactly those instants.
  const minutes = -d.getTimezoneOffset();
  const sign = minutes < 0 ? "-" : "+";
  const hours = Math.floor(Math.abs(minutes) / 60);
  const mins = Math.abs(minutes) % 60;
  return (
    `${pad(year, 4)}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ` +
    `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())} ` +
    `${sign}${pad(hours)}:${pad(mins)}`
  );
}

/** Comma-join agent names for a listing, or `-` when a session has none (a
 * single-agent or pre-v2 session). One helper so the `sessions` table, the
 * ambiguity picker and the `runs` alias never disagree on the empty case. */
export function agentsText(names: string[]): string {
  return names.length ? names.join(", ") : "-";
}

/** One session as a single picker line: short id, local start time, turn count,
 * agents. Deliberately the same columns the `sessions` table shows, so the
 * listing a user is shown when they must pick reads exactly like the listing
 * they would have run `ctxdiff sessions` to get. */
export function sessionLine(s: Session): string {
  return (
    `${shortId(s.id)}  ${formatLocal(s.startedAt)}  ` +
    `turns=${s.turnCount}  agents=${agentsText(s.agents)}`
  );
}

/** The indented block of session lines appended to an ambiguity error. */
export function sessionPicker(sessions: Session[]): string {
  return sessions.map((s) => "  " + sessionLine(s)).join("\n");
}

/** The indented block of agent names appended to an agent error, or an
 * explanatory line when the session has no labeled agents at all — 'available
 * agents:' followed by nothing would read like a bug in the tool. */
export function agentPicker(names: string[]): string {
  if (!names.length) return "  (none — this session's calls carry no agent label)";
  return names.map((n) => "  " + n).join("\n");
}

/**
 * Resolve which session to read, or throw `SelectionError` carrying the listing
 * of what could be picked.
 *
 * With `wanted`: an exact id wins; otherwise any session whose id STARTS WITH
 * the value matches, so the 12-char id printed by `ctxdiff sessions` can be
 * pasted straight back. Zero matches and several matches are both errors, and
 * both print the listing — a prefix that matches two sessions is a question, not
 * a failure.
 *
 * Without `wanted`: one session (or none — an empty store is reported elsewhere)
 * means `defaultId`, the handle's own binding, which is the newest session. More
 * than one is AMBIGUOUS and refuses to guess: this is the case where silently
 * defaulting would analyze yesterday's run and say nothing.
 */
export function chooseSession(
  sessions: Session[],
  wanted: string | null,
  defaultId: string,
): string {
  if (wanted !== null) {
    const exact = sessions.filter((s) => s.id === wanted);
    const matches = exact.length ? exact : sessions.filter((s) => s.id.startsWith(wanted));
    if (matches.length === 1) return matches[0].id;
    if (!matches.length) {
      throw new SelectionError(
        `ctxdiff: no session '${wanted}' in this project — ` +
          `available sessions:\n${sessionPicker(sessions)}`,
      );
    }
    throw new SelectionError(
      `ctxdiff: session '${wanted}' is ambiguous — ${matches.length} ` +
        `sessions match:\n${sessionPicker(matches)}`,
    );
  }

  if (sessions.length <= 1) return defaultId;
  throw new SelectionError(
    `ctxdiff: this project holds ${sessions.length} sessions — pass ` +
      `--session to pick one:\n${sessionPicker(sessions)}`,
  );
}

/** The distinct NAMED agents across `calls`, in first-appearance order.
 * Unlabeled calls contribute nothing (unlike the analyzers' `distinctAgents`,
 * which counts a null bucket) because this list exists to be shown to a user
 * picking an `--agent` value, and `--agent null` is not a thing they can type. */
export function distinctAgentNames(calls: Call[]): string[] {
  const names: string[] = [];
  for (const c of calls) {
    if (c.agent && !names.includes(c.agent)) names.push(c.agent);
  }
  return names;
}

/** Assert that `agent` actually exists in a session, throwing a
 * `SelectionError` listing the real agent names when it doesn't. `where`
 * describes the scope in the message ('session 4f3a2b1c', 'these sessions'),
 * since the same check guards a single-session filter and both sides of a
 * cross-session diff. */
export function requireAgent(agent: string, names: string[], where: string): void {
  if (names.includes(agent)) return;
  throw new SelectionError(
    `ctxdiff: no agent '${agent}' in ${where} — available agents:\n${agentPicker(names)}`,
  );
}

/**
 * Refuse to compare two sessions that hold SEVERAL agents when no `--agent` was
 * given.
 *
 * Why this one is ambiguous while a plain `--agent`-less command is not: a
 * cross-session diff means "the same agent, in two runs" — the regression case.
 * Turn 8 is a global sequence number within each session, so with two agents
 * interleaved, turn 8 can be the researcher in one run and the writer in the
 * other, and the diff would silently compare unrelated contexts. One agent (or
 * none) leaves nothing to get wrong, so no flag is required.
 */
export function requireSingleAgent(names: string[], where: string): void {
  if (names.length <= 1) return;
  throw new SelectionError(
    `ctxdiff: ${where} hold ${names.length} agents — pass --agent to pick ` +
      `one:\n${agentPicker(names)}`,
  );
}

/** One end of a diff: which session, which agent (null = no filter), which
 * turn. A cross-session diff has two sides differing in `sessionId`, a
 * cross-agent diff two sides differing in `agent`, and an ordinary same-session
 * diff two sides differing only in `turn`.
 *
 * `turn` is a `TurnArg` rather than the plain int Python's `DiffSide` carries,
 * for the one reason Python does not need it: a turn the user typed has to be
 * echoed back with the digits they typed, and a double cannot promise that. */
export interface DiffSide {
  sessionId: string;
  agent: string | null;
  turn: TurnArg;
}

/**
 * The header printed ABOVE a cross-session/cross-agent diff naming what is being
 * compared, or null for an ordinary same-session same-agent diff.
 *
 * Why null in the ordinary case: `renderTurnDiff`'s own header already says
 * `turn 7 → turn 8`, and that is the complete truth when both turns come from
 * one session and one agent — printing a scope line there would change every
 * existing diff's output for no information. It is only when the two sides
 * differ in session or agent that `turn 8 → turn 8` becomes actively misleading
 * without one.
 *
 * Only the dimensions that MATTER are shown: the session appears on both sides
 * only when the sides come from different sessions, and the agent only when one
 * was selected. So a cross-session diff reads
 * `── 4f3a2b1c9d8e · researcher · turn 8  →  9e8d7c6b5a4f · researcher · turn 8 ──`
 * and a cross-agent one drops the (identical) session entirely.
 */
export function diffScopeLine(left: DiffSide, right: DiffSide): string | null {
  if (left.sessionId === right.sessionId && left.agent === right.agent) return null;
  const showSession = left.sessionId !== right.sessionId;
  const label = (side: DiffSide): string => {
    const parts: string[] = [];
    if (showSession) parts.push(shortId(side.sessionId));
    if (side.agent) parts.push(side.agent);
    parts.push(`turn ${side.turn.text}`);
    return parts.join(" · ");
  };
  return `── ${label(left)}  →  ${label(right)} ──`;
}
