/**
 * Core value types plus the pure functions that give a block its identity and
 * its label. Kept dependency-free (only node:crypto) so every other module can
 * import it freely.
 *
 * PARITY: these primitives are the cross-language contract. They MUST produce
 * byte-identical output to the Python SDK's `ctxdiff.models` so a `.ctrace`
 * written here opens in the Python `ctxdiff view`. See spec/ctrace-schema.md.
 */
import { createHash } from "node:crypto";

// --- value types -----------------------------------------------------------

/**
 * One context unit as an adapter extracts it from a request payload, before
 * token counting or hashing. `kind` is 'message' | 'content_part' |
 * 'tool_schema'; `role` is 'system' | 'user' | 'assistant' | 'tool'.
 */
export interface RawBlock {
  role: string;
  kind: string;
  text: string;
}

/**
 * A stored, content-addressed context unit. Identity is `contentHash`, so
 * equal text (same role+kind) is stored once and referenced many times.
 */
export interface Block {
  contentHash: string;
  role: string;
  kind: string;
  text: string;
  tokenCount: number;
  tokenMethod: string; // 'tiktoken' | 'estimate'
}

/**
 * Membership of a `Block` in one call: where it sat (`position`) and how it was
 * labeled. Label lives here, not on Block, because the same text can be labeled
 * differently depending on tagging — identity must not depend on it.
 */
export interface CallBlock {
  block: Block;
  position: number;
  label: string;
  labelSource: string; // 'heuristic' | 'tagged'
}

/** One agent execution, as stored in the `run` table. */
export interface Run {
  id: string;
  project: string;
  startedAt: string;
  provider: string;
  models: string[];
  ctxdiffVersion: string;
}

/**
 * A one-line summary of one session (one `run` row) in a project `.ctrace`, as
 * returned by `CTrace.listSessions()` — the shape a session picker lists from.
 * `agents` is the set of distinct agent labels seen on this session's calls, in
 * first-appearance order (`[]` for a single-agent/pre-v2 session); `turnCount`
 * is how many calls it holds. `startedAt` is the raw stored string — use
 * `parseStartedAt()` for a tz-aware Date. Mirrors Python's `Session`.
 */
export interface Session {
  id: string;
  project: string;
  startedAt: string;
  provider: string;
  models: string[];
  agents: string[];
  turnCount: number;
}

/** One LLM request/response ('turn'), as stored in the `call` table. */
export interface Call {
  id: string;
  runId: string;
  seq: number;
  params: Record<string, unknown>;
  usage: Record<string, unknown> | null;
  latencyMs: number | null;
  error: string | null;
  agent: string | null;
  step: string | null;
  provider: string | null;
}

// --- pure functions --------------------------------------------------------

/**
 * Serialize a value to a string byte-identical to Python's
 * `json.dumps(value, sort_keys=True, ensure_ascii=False)`. This is the single
 * hardest parity requirement in the SDK: JS `JSON.stringify` does NOT sort
 * object keys and uses no spacing, whereas CPython's json emits `", "` and
 * `": "` separators and sorts keys. We reproduce Python's exact bytes:
 *
 *   - objects: keys sorted (lexicographically, by code unit — matching
 *     Python's default string comparison for the ASCII/BMP keys that appear in
 *     LLM payloads), separated by `", "`, key/value by `": "`.
 *   - arrays: order preserved, separated by `", "`.
 *   - strings: `JSON.stringify` of the string — JS and CPython escape control
 *     chars, quotes and backslashes identically and BOTH leave non-ASCII
 *     literal (ensure_ascii=False), so the bytes match.
 *   - integers/booleans/null: identical spelling in both languages.
 *
 * One cross-language edge (see spec): an integer-valued float in a JSON-Schema
 * numeric keyword (e.g. `default: 3.0` inside a tool schema) normalizes
 * differently — Python `json.dumps(3.0)` -> "3.0" vs JS `JSON.stringify(3.0)` ->
 * "3", because JS has no runtime int/float distinction — so the SAME tool schema
 * authored in both SDKs would hash differently. This affects only a
 * cross-language DIFF of the same app captured in both SDKs; it does NOT affect
 * JS->Python reads (readers store and return hashes verbatim, never re-hash),
 * and within a single language dedup is fully consistent.
 */
export function stableStringify(value: unknown): string {
  if (typeof value === "string") return JSON.stringify(value);
  if (value === null || value === undefined) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "null"; // Python json emits NaN/Infinity, but they never occur in payloads; null is the safe parity-neutral choice.
    return String(value);
  }
  if (Array.isArray(value)) {
    return "[" + value.map((v) => stableStringify(v)).join(", ") + "]";
  }
  if (typeof value === "object") {
    const obj = value as Record<string, unknown>;
    const keys = Object.keys(obj).sort();
    const parts = keys.map(
      (k) => JSON.stringify(k) + ": " + stableStringify(obj[k]),
    );
    return "{" + parts.join(", ") + "}";
  }
  // Fallback for exotic types (functions, symbols): Python would raise, but the
  // fail-open recorder swallows; emit "null" so hashing never itself throws.
  return "null";
}

/**
 * Return a canonical string for hashing. Strings pass through verbatim to
 * preserve wire truth; anything else (multi-part content dicts/lists) is
 * serialized with sorted keys so semantically-equal content always maps to the
 * same string, and therefore the same hash. Mirrors Python `normalize_text`.
 */
export function normalizeText(text: unknown): string {
  if (typeof text === "string") return text;
  return stableStringify(text);
}

/**
 * Compute a block's identity as sha256 over role, kind, and normalized text,
 * joined by a NUL separator that cannot appear in the fields — so ('a','bc')
 * and ('ab','c') never collide. Returns a 64-char lowercase hex digest.
 * Mirrors Python `content_hash`.
 */
export function contentHash(role: string, kind: string, text: unknown): string {
  const joined = [role, kind, normalizeText(text)].join("\x00");
  return createHash("sha256").update(joined, "utf-8").digest("hex");
}

// Role → coarse label. Mirrors Python's `_ROLE_LABEL`.
const ROLE_LABEL: Record<string, string> = {
  system: "system",
  tool: "tool_output",
  user: "user",
  assistant: "history",
};

/**
 * Decide a block's [label, source]. A developer tag wins first: if any tagged
 * text is a substring of this block, return that tag's label with source
 * 'tagged' (first tag registered wins). Otherwise a tool schema is
 * 'tool_schema' and everything else maps by role, falling back to the raw role
 * string for unknown roles so no input can crash labeling. Mirrors Python
 * `basic_label`.
 */
export function basicLabel(
  role: string,
  kind: string,
  text: string,
  tagged: [string, string][],
): [string, string] {
  for (const [label, needle] of tagged) {
    if (needle && text.includes(needle)) return [label, "tagged"];
  }
  if (kind === "tool_schema") return ["tool_schema", "heuristic"];
  return [ROLE_LABEL[role] ?? role, "heuristic"];
}
