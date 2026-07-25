/**
 * The exporter: turn a `.ctrace` (or a database-backed project) into one
 * self-contained, THREE-LEVEL HTML dashboard. A faithful port of Python
 * `ctxdiff.viewer.export` — `buildPayload` assembles the exact same
 * JSON-serializable payload for one session's detail view (same fields, same key
 * order, embedding the same precomputed analyzer output), `buildProjectPayload`
 * wraps it with the same project index (level 1 = all agents, level 2 = one
 * agent's sessions, level 3 = that detail), and `exportHtml` embeds the result
 * in the lifted-verbatim template. Together with `pyJsonDumps` this makes the
 * emitted HTML byte-identical to the Python viewer's for the same trace.
 *
 * WHAT IS AND ISN'T EMBEDDED (the scale decision, identical to Python's): levels
 * 1 and 2 are aggregates over EVERY session, computed from the session list and
 * each session's call rows alone — no block reads, so their cost is O(calls) and
 * no session or agent is ever hidden. Level 3 is the block-level detail, which
 * is what actually has size, so it is embedded only for the `DETAIL_SESSIONS`
 * most recent sessions plus the focus session. The page lists the rest with
 * their aggregates and states the cap.
 *
 * Security (audited): the payload is embedded as a JSON island inside a
 * `<script type="application/json">` tag, and every `</` in the serialized JSON
 * is escaped to `<\/`, so block text containing `</script>` (or any markup)
 * cannot break out of the tag. The template's runtime reads the island with
 * `.textContent` and renders all trace-derived text — block text, and the AGENT
 * NAMES and session labels levels 1/2 and the breadcrumb show — with
 * `.textContent` (never `.innerHTML`), so untrusted trace data can never
 * execute. Zero external requests: every byte (CSS/JS/data) is inline; no URL of
 * any kind.
 */
import { writeFileSync } from "node:fs";
import { basename, dirname, resolve, join } from "node:path";
import { CTrace } from "../store/ctrace.js";
import type { ReadableStore, Run, Session } from "../store/base.js";
import type { Call, CallBlock } from "../models.js";
import { diffTurns, distinctAgents, type DiffEntry, type TurnDiff } from "../analyze/diff.js";
import { analyzeRun, usageTotals, type RunTokens } from "../analyze/tokens.js";
import { analyzeCache, type CacheReport } from "../analyze/cache.js";
import { analyzeEvictions, type EvictionReport } from "../analyze/evictions.js";
import { CONTEXT_WINDOW_ALARM_PCT, windowPct } from "../analyze/window.js";
import { compareCodePoints } from "../analyze/sessions.js";
import { renderPage } from "./template.js";
import { PyFloat, pyJsonDumps, htmlEscape } from "./pyjson.js";

// How much of a diff entry's block text to embed, and how much of each inline
// segment to keep — previews, not whole documents (matches Python).
const SNIPPET_CHARS = 200;
const INLINE_SEG_CHARS = 500;

/**
 * How many sessions get their FULL level-3 detail embedded — the most recent
 * ones, plus the focus session whether or not it made the cut. Must equal
 * Python's `_DETAIL_SESSIONS`: a different cap on either side would embed a
 * different set of sessions and break byte-identity outright.
 *
 * Why a cap at all: the dashboard is one self-contained file with no network, so
 * every byte of detail is paid up front. 25 keeps a project dashboard in the
 * megabytes even against a database holding thousands of sessions, while still
 * covering "the runs I did today". Levels 1 and 2 are NOT capped.
 */
const DETAIL_SESSIONS = 25;

/** The bucket name for calls that carry no agent label — the same one the token
 * analyzer and `ctxdiff agents` use, so one project never reports two different
 * names for the same calls. */
const UNLABELED = "(unlabeled)";

/**
 * What `buildProjectPayload` reads: the synchronous analyzer surface plus, when
 * the reader has one, the store's session list. `listSessions` is OPTIONAL
 * because a reader materialized for a SINGLE session (an in-memory snapshot
 * taken without the listing) must still export — it degrades to a one-session
 * project rather than failing.
 */
export interface ProjectReader extends ReadableStore {
  listSessions?(): Session[];
}

/** Slice a string to `n` CODE POINTS (like Python `text[:n]`), not UTF-16 code
 * units, so multi-byte text truncates identically to Python. */
function sliceCp(text: string, n: number): string {
  return Array.from(text).slice(0, n).join("");
}

// --- per-entity serializers ----------------------------------------------------

/** One CallBlock → a JSON dict for the blocks table. Mirrors Python
 * `_serialize_block` (same keys, same order). */
function serializeBlock(cb: CallBlock): Record<string, unknown> {
  const b = cb.block;
  return {
    position: cb.position,
    label: cb.label,
    label_source: cb.labelSource,
    role: b.role,
    kind: b.kind,
    token_count: b.tokenCount,
    token_method: b.tokenMethod,
    text: b.text,
    content_hash_short: b.contentHash.slice(0, 8),
  };
}

/** One Call + its blocks → a JSON dict. `params` is reduced to the model id
 * ALONE (the raw params may carry temperature/keys — not for a shareable
 * artifact); provider `usage` passes through. Mirrors Python `_serialize_call`. */
function serializeCall(call: Call, callBlocks: CallBlock[]): Record<string, unknown> {
  return {
    seq: call.seq,
    latency_ms: call.latencyMs,
    error: call.error,
    agent: call.agent,
    step: call.step,
    params: { model: (call.params ?? {})["model"] ?? null },
    usage: call.usage,
    blocks: callBlocks.map(serializeBlock),
  };
}

/** One DiffEntry → a JSON dict (text capped). Mirrors Python
 * `_serialize_diff_entry`. */
function serializeDiffEntry(e: DiffEntry): Record<string, unknown> {
  const inline =
    e.inlineDiff !== null
      ? e.inlineDiff.map(([op, text]) => [op, sliceCp(text, INLINE_SEG_CHARS)])
      : null;
  return {
    kind: e.kind,
    label: e.label,
    role: e.block.role,
    position_old: e.positionOld,
    position_new: e.positionNew,
    token_count: e.block.tokenCount,
    snippet: sliceCp(e.block.text, SNIPPET_CHARS),
    inline_diff: inline,
  };
}

/** One TurnDiff → a JSON dict. Mirrors Python `_serialize_diff`. */
function serializeDiff(td: TurnDiff): Record<string, unknown> {
  return {
    seq_old: td.seqOld,
    seq_new: td.seqNew,
    tokens_added: td.tokensAdded,
    tokens_evicted: td.tokensEvicted,
    entries: td.entries.map(serializeDiffEntry),
  };
}

/**
 * A RunTokens → a JSON dict (per-call slices + run-level bloat). Percentages are
 * wrapped in PyFloat so they serialize as Python floats. Mirrors Python
 * `_serialize_tokens`.
 *
 * `contextWindow` and each call's `pct_of_window` carry the share-of-window
 * story into the page. The percentage is computed HERE rather than in the
 * template's JavaScript on purpose: `windowPct` rounds the way CPython's `round`
 * does and the Python exporter rounds the same way, so both SDKs emit the same
 * digits into the same file. A `toFixed` in the browser would put a third
 * rounding rule in the path and make the exported HTML's hash depend on which
 * SDK produced it. Both fields are null when no window was supplied, and the
 * page then renders exactly as it always has.
 *
 * `window_alarm_pct` travels with them so the page's warning styling and the
 * CLI's `⚠` marker trip at the same number, from one constant.
 */
function serializeTokens(
  rt: RunTokens,
  contextWindow: number | null,
): Record<string, unknown> {
  return {
    context_window: contextWindow,
    window_alarm_pct: new PyFloat(CONTEXT_WINDOW_ALARM_PCT),
    calls: rt.calls.map((c) => ({
      seq: c.seq,
      total: c.totalTokens,
      approximate: c.approximate,
      pct_of_window:
        contextWindow === null ? null : new PyFloat(windowPct(c.totalTokens, contextWindow)),
      slices: c.slices.map((s) => ({ label: s.label, tokens: s.tokens, pct: new PyFloat(s.pct) })),
      reconciliation_delta: c.reconciliationDelta,
    })),
    bloat:
      rt.bloat === null
        ? null
        : {
            unused_tools: rt.bloat.unusedTools,
            unused_tokens_per_call: rt.bloat.unusedTokensPerCall,
            pct_of_avg_context: new PyFloat(rt.bloat.pctOfAvgContext),
          },
  };
}

/** A CacheReport → a JSON dict. Mirrors Python `_serialize_cache`. */
function serializeCache(cr: CacheReport): Record<string, unknown> {
  return {
    pairs_analyzed: cr.pairsAnalyzed,
    breaks: cr.breaks.map((b) => ({
      seq_prev: b.seqPrev,
      seq: b.seq,
      stable_tokens: b.stableTokens,
      divergent_position: b.divergentPosition,
      culprit_kind: b.culpritKind,
      culprit_label: b.culpritLabel,
      culprit_snippet: b.culpritSnippet,
      detail: b.detail,
      agent: b.agent,
    })),
    stable_prefix_tokens_min: cr.stablePrefixTokensMin,
    rebilled_tokens_total: cr.rebilledTokensTotal,
    waste_note: cr.estimatedWasteNote,
    fix_hint: cr.fixHint,
    agents_analyzed: cr.agentsAnalyzed,
  };
}

/**
 * An EvictionReport → a JSON dict: each tagged block that entered an agent's
 * context and left it for good, with the turn it was tagged on, the turn its
 * content entered, the turn it disappeared and enough of the block to recognize
 * it. `tagged_seq` and `entered_seq` are two different facts and both travel, so
 * the page can name the tag against the turn that carried it. Mirrors Python
 * `_serialize_evictions`.
 *
 * Only TAGGED blocks are ever in here, so the dashboard can render every entry
 * as a warning without a threshold of its own. `pairs_analyzed`/`tagged_blocks`
 * travel along so the panel can tell "nothing was tagged" apart from "nothing
 * was lost" — two very different reassurances.
 */
function serializeEvictions(er: EvictionReport): Record<string, unknown> {
  return {
    evictions: er.evictions.map((e) => ({
      label: e.label,
      agent: e.agent,
      tagged_seq: e.taggedSeq,
      entered_seq: e.enteredSeq,
      last_seen_seq: e.lastSeenSeq,
      evicted_seq: e.evictedSeq,
      tokens: e.tokens,
      role: e.role,
      snippet: e.snippet,
    })),
    pairs_analyzed: er.pairsAnalyzed,
    tagged_blocks: er.taggedBlocks,
  };
}

// --- payload assembly ----------------------------------------------------------

/**
 * Assemble every fact the dashboard renders into one JSON-serializable object.
 * Pure: reads from `ct` and the three analyzers and returns a plain object — no
 * filesystem, no template, no HTML. Delegates the hard analysis to the existing
 * pure analyzers (single source of truth) and serializes their results with the
 * SAME field order as the Python builder. Mirrors Python `build_payload`.
 */
export function buildPayload(
  ct: ReadableStore,
  contextWindow: number | null = null,
): Record<string, unknown> {
  const run = ct.getRun();
  const calls = ct.getCalls();
  const blocksByCall = new Map<string, CallBlock[]>();
  for (const c of calls) blocksByCall.set(c.id, ct.getCallBlocks(c.id));

  const callsOut = calls.map((c) => serializeCall(c, blocksByCall.get(c.id)!));

  // diffs: one per adjacent (N-1, N) turn pair, in global seq order; each gains
  // `cross_agent`, and when it IS a hand-off with an earlier same-agent call,
  // also `same_agent_diff` — appended AFTER the base keys, matching Python.
  const diffsOut: Record<string, unknown>[] = [];
  for (let i = 1; i < calls.length; i++) {
    const prev = calls[i - 1];
    const cur = calls[i];
    const d = serializeDiff(diffTurns(ct, prev.seq, cur.seq));
    d.cross_agent = prev.agent !== cur.agent;
    if (d.cross_agent) {
      let samePrev: Call | null = null;
      for (let j = i - 1; j >= 0; j--) {
        if (calls[j].agent === cur.agent) {
          samePrev = calls[j];
          break;
        }
      }
      if (samePrev !== null) {
        d.same_agent_diff = serializeDiff(diffTurns(ct, samePrev.seq, cur.seq));
      }
    }
    diffsOut.push(d);
  }

  const runTokens = analyzeRun(ct);
  const tokensOut = serializeTokens(runTokens, contextWindow);
  const cacheOut = serializeCache(analyzeCache(ct));
  // analyzeEvictions groups per agent for the same reason analyzeCache does, so
  // a hand-off is never embedded in the page as a lost block.
  const evictionsOut = serializeEvictions(analyzeEvictions(ct));

  // stats: dedup ratio + context growth
  const distinct = new Set<string>();
  let totalRefs = 0;
  for (const cbs of blocksByCall.values()) {
    for (const cb of cbs) {
      distinct.add(cb.block.contentHash);
      totalRefs += 1;
    }
  }
  const growth = runTokens.calls.map((c) => c.totalTokens);

  const tokensByCall = new Map<string, number>();
  for (const c of calls) {
    tokensByCall.set(
      c.id,
      blocksByCall.get(c.id)!.reduce((acc, cb) => acc + cb.block.tokenCount, 0),
    );
  }
  const agentsStats = distinctAgents(calls).map((label) => {
    const group = calls.filter((c) => c.agent === label);
    return {
      name: label ?? "(unlabeled)",
      calls: group.length,
      tokens: group.reduce((acc, c) => acc + tokensByCall.get(c.id)!, 0),
    };
  });

  // The one place the payload is keyed by AGENT NAME rather than by position,
  // so the one place a name that means something to JavaScript can go missing:
  // `{}` inherits Object.prototype, and `obj["__proto__"] = io` on it runs the
  // prototype SETTER, which ignores non-objects and leaves NO own property — so
  // `Object.keys` (and therefore `pyJsonDumps`) dropped that agent entirely
  // while Python's dict kept it, diverging the two exports by whole bytes and
  // showing the agent's chip as `in 0 · out 0`. A null-prototype object has no
  // magic keys: `__proto__` is stored as data, in insertion order, like any
  // other name. (Every other name-keyed structure in this builder is a `Map`,
  // which was never affected; `details` is keyed by generated session ids.)
  const u = runTokens.usage;
  let byAgent: Record<string, [number, number]> | null = null;
  if (u.byAgent !== null) {
    byAgent = Object.create(null) as Record<string, [number, number]>;
    for (const [name, io] of u.byAgent) byAgent[name] = io;
  }
  const usageStats = {
    input: u.inputTokens,
    output: u.outputTokens,
    coverage: [u.callsWithUsage, u.callsTotal],
    by_agent: byAgent,
  };

  return {
    run: {
      project: run.project,
      provider: run.provider,
      started_at: run.startedAt,
      ctxdiff_version: run.ctxdiffVersion,
      models: run.models,
    },
    calls: callsOut,
    diffs: diffsOut,
    tokens: tokensOut,
    cache: cacheOut,
    evictions: evictionsOut,
    stats: {
      distinct_blocks: distinct.size,
      total_block_refs: totalRefs,
      context_growth: growth,
      agents: agentsStats,
      usage: usageStats,
    },
  };
}

// --- project index (levels 1 and 2) ---------------------------------------------

/**
 * A read handle PINNED to one session of a many-session store.
 *
 * Why it exists: `buildPayload` and all three analyzers call `getRun()`/
 * `getCalls()` with NO argument and get whatever session the handle is bound to.
 * Building the detail of a session OTHER than that one therefore needs a reader
 * whose no-argument reads answer for the session we chose — which is exactly
 * this. Mirrors Python `_PinnedReader`; it closes nothing, because the caller
 * owns the underlying reader and hands out several pins over its lifetime.
 */
class PinnedReader implements ReadableStore {
  constructor(
    private readonly reader: ReadableStore,
    private readonly sessionId: string,
  ) {}
  getRun(sessionId?: string): Run {
    return this.reader.getRun(sessionId ?? this.sessionId);
  }
  getCalls(sessionId?: string): Call[] {
    return this.reader.getCalls(sessionId ?? this.sessionId);
  }
  getCallBlocks(callId: string): CallBlock[] {
    return this.reader.getCallBlocks(callId);
  }
}

/** One call's agent bucket: its label, or `(unlabeled)`. Mirrors Python
 * `_agent_name`. */
function agentName(call: Call): string {
  return call.agent ? call.agent : UNLABELED;
}

/** Provider-reported spend for `calls` as the three payload fields levels 1 and
 * 2 render. `reported` earns its place because 0 and "nobody told us" are
 * different facts — the page shows `-` for the latter, exactly as `ctxdiff
 * agents` does. Mirrors Python `_usage_fields`. */
function usageFields(calls: Call[]): { input: number; output: number; reported: number } {
  const t = usageTotals(calls);
  return { input: t.inputTokens, output: t.outputTokens, reported: t.callsWithUsage };
}

/**
 * Which sessions get their level-3 detail embedded: the `DETAIL_SESSIONS` most
 * recent ones (`sessions` arrives OLDEST first, as every store lists it), plus
 * the focus session whether or not it made the cut — `--session <old run>` must
 * always land on a working detail view.
 *
 * Exported because the DECISION is needed in two places that must not drift: the
 * payload builder, which serializes those details, and the networked-store
 * snapshot, which decides whose BLOCKS to read over the wire in the first place
 * (see `store/snapshot.ts`). Python has no twin for this — its stores are
 * synchronous, so the payload builder reads whatever it needs directly.
 */
export function detailSessionIds(sessions: Session[], focusId: string): string[] {
  const ids = [...sessions].reverse().slice(0, DETAIL_SESSIONS).map((s) => s.id);
  if (!ids.includes(focusId)) ids.push(focusId);
  return ids;
}

/** LEVEL 1: every agent in the project with its footprint aggregated across ALL
 * sessions, in first-appearance order scanning sessions oldest-first — the same
 * order `ctxdiff agents` lists them in. `first_seen`/`last_seen` stay RAW stored
 * UTC; the page converts them to the viewer's local zone at render time. Mirrors
 * Python `_agent_index`.
 *
 * The span is the MIN and MAX of the collected timestamps, not the ends of the
 * list: the list is in INSERT order, which stops being chronological the moment
 * two capturing machines' clocks disagree or a session is backfilled — it read
 * `first seen 2026-05-01 ... last seen 2026-03-01`, a range running backwards.
 * `compareCodePoints` rather than `<` so ties and non-ASCII order the way
 * Python's `min`/`max` over `str` do (UTF-16 code UNITS misorder astral
 * characters), keeping the two exports byte-identical. */
function agentIndex(
  sessions: Session[],
  callsBySession: Map<string, Call[]>,
): Record<string, unknown>[] {
  const order: string[] = [];
  const callsByAgent = new Map<string, Call[]>();
  const sessionCounts = new Map<string, number>();
  const spans = new Map<string, string[]>();
  for (const s of sessions) {
    const seenHere = new Set<string>();
    for (const c of callsBySession.get(s.id) ?? []) {
      const name = agentName(c);
      if (!callsByAgent.has(name)) {
        order.push(name);
        callsByAgent.set(name, []);
      }
      callsByAgent.get(name)!.push(c);
      if (!seenHere.has(name)) {
        seenHere.add(name);
        sessionCounts.set(name, (sessionCounts.get(name) ?? 0) + 1);
        if (!spans.has(name)) spans.set(name, []);
        spans.get(name)!.push(s.startedAt);
      }
    }
  }
  return order.map((name) => {
    const calls = callsByAgent.get(name)!;
    const seen = spans.get(name)!;
    // Python's `min`/`max` keep the FIRST extreme they meet; so does this.
    let first = seen[0];
    let last = seen[0];
    for (const at of seen) {
      if (compareCodePoints(at, first) < 0) first = at;
      if (compareCodePoints(at, last) > 0) last = at;
    }
    return {
      name,
      sessions: sessionCounts.get(name)!,
      calls: calls.length,
      ...usageFields(calls),
      first_seen: first,
      last_seen: last,
    };
  });
}

/** LEVEL 2: every session in the project, NEWEST FIRST, with the per-agent
 * breakdown. `detail` says whether this session's level-3 view is embedded in
 * this file; a row without it is still listed with all its aggregates, and the
 * page names the cap. Mirrors Python `_session_index`. */
function sessionIndex(
  sessions: Session[],
  callsBySession: Map<string, Call[]>,
  detailed: Set<string>,
): Record<string, unknown>[] {
  const rows: Record<string, unknown>[] = [];
  for (let i = sessions.length - 1; i >= 0; i--) {
    const s = sessions[i];
    const order: string[] = [];
    const byAgent = new Map<string, Call[]>();
    for (const c of callsBySession.get(s.id) ?? []) {
      const name = agentName(c);
      if (!byAgent.has(name)) {
        order.push(name);
        byAgent.set(name, []);
      }
      byAgent.get(name)!.push(c);
    }
    rows.push({
      id: s.id,
      started_at: s.startedAt,
      provider: s.provider,
      models: s.models,
      turn_count: s.turnCount,
      detail: detailed.has(s.id),
      agents: order.map((name) => ({
        name,
        turns: byAgent.get(name)!.length,
        ...usageFields(byAgent.get(name)!),
      })),
    });
  }
  return rows;
}

/** Which of the three levels the dashboard OPENS on, and what it opens scoped
 * to — "never make someone click through a choice they don't have, and never
 * guess one they do". Mirrors Python `_start_level` exactly (see its docstring
 * for the rule table).
 *
 * When the start level HAS a session it is `focusId` — the session this whole
 * payload was built around, and the only one whose detail is guaranteed to be
 * embedded. It must not be read off `sessionRows`, which is NEWEST FIRST:
 * doing that opened every `--session <older run>` export on the newest session
 * instead, pointing the page's `openSession(start.session)` at a run the user
 * never named. */
function startLevel(
  agent: string | null,
  sessionSelected: boolean,
  agentRows: Record<string, unknown>[],
  sessionRows: Record<string, unknown>[],
  focusId: string,
): Record<string, unknown> {
  const multi = sessionRows.length > 1 || agentRows.length > 1;
  let level: number;
  if (agent !== null) {
    const agentSessions = sessionRows.filter((s) =>
      (s.agents as { name: string }[]).some((a) => a.name === agent),
    ).length;
    level = sessionSelected || agentSessions <= 1 ? 3 : 2;
  } else if (sessionSelected || !multi) {
    level = 3;
  } else {
    level = 1;
  }
  return { level, agent, session: level === 3 ? focusId : null };
}

/** Every session in the project, oldest first — or a one-element list describing
 * just the focus session when the reader cannot list them. The fallback is what
 * keeps `exportStore` total for a single-session snapshot. Mirrors Python
 * `_list_sessions_or_focus`. */
function listSessionsOrFocus(reader: ProjectReader, focusRun: Run): Session[] {
  let sessions: Session[] = [];
  try {
    if (typeof reader.listSessions === "function") sessions = reader.listSessions();
  } catch {
    /* any listing failure degrades to the focus session */
  }
  if (sessions.length) return sessions;
  const calls = reader.getCalls(focusRun.id);
  const agents: string[] = [];
  for (const c of calls) if (c.agent && !agents.includes(c.agent)) agents.push(c.agent);
  return [
    {
      id: focusRun.id,
      project: focusRun.project,
      startedAt: focusRun.startedAt,
      provider: focusRun.provider,
      models: focusRun.models,
      agents,
      turnCount: calls.length,
    },
  ];
}

/** Which selectors the user passed — all `startLevel` needs to decide where the
 * dashboard opens. */
export interface ProjectPayloadOptions {
  /** `--agent NAME`, or null/undefined when none was given. */
  agent?: string | null;
  /** Whether `--session` explicitly named the focus session (as opposed to it
   * being defaulted to the newest). */
  sessionSelected?: boolean;
  /** Build the project around THIS session instead of the reader's bound one. */
  focusSessionId?: string;
  /** The context window every percentage in the page is taken against, resolved
   * by `resolveContextWindow` (flag, then `CTXDIFF_CONTEXT_WINDOW`). Omitted or
   * null means no window is known and the page shows no percentages — ctxdiff
   * ships no model→window table by design. */
  contextWindow?: number | null;
}

/**
 * Assemble the THREE-LEVEL dashboard payload: the focus session's detail at the
 * top level (exactly the object `buildPayload` returns), plus one `project` key
 * carrying the level-1 agent index, the level-2 session index, and the embedded
 * details of the other sessions within the cap.
 *
 * The focus session's detail sits at the TOP level rather than inside
 * `project.details` so this payload is a strict SUPERSET of the single-session
 * one — every existing assertion about `run`/`calls`/`diffs`/`tokens`/`cache`/
 * `stats` keeps meaning what it meant, and the focus session's (large) detail is
 * never serialized twice. Key order matches the Python builder's dict insertion
 * order throughout, which is what `pyJsonDumps` needs to reproduce Python's
 * bytes. Mirrors Python `build_project_payload`.
 */
export function buildProjectPayload(
  reader: ProjectReader,
  opts: ProjectPayloadOptions = {},
): Record<string, unknown> {
  const agent = opts.agent ?? null;
  const focusRun = opts.focusSessionId ? reader.getRun(opts.focusSessionId) : reader.getRun();
  const focusId = focusRun.id;

  const sessions = listSessionsOrFocus(reader, focusRun);
  // Every session's CALL rows — the input to both aggregate levels. Blocks are
  // NOT read here: that is the expensive axis, paid only for embedded details.
  const callsBySession = new Map<string, Call[]>();
  for (const s of sessions) callsBySession.set(s.id, reader.getCalls(s.id));

  const detailed = new Set(detailSessionIds(sessions, focusId));

  const agentRows = agentIndex(sessions, callsBySession);
  const sessionRows = sessionIndex(sessions, callsBySession, detailed);

  // Non-focus details, newest first — the focus session's own detail is the
  // payload's top level and is never duplicated here.
  // The window is threaded into EVERY embedded session, not just the focus one:
  // the dashboard's level-3 view is reachable for each of them, and a page where
  // three sessions show percentages and the fourth silently does not would read
  // as a bug in the data rather than as one export flag.
  const contextWindow = opts.contextWindow ?? null;
  const details: Record<string, unknown> = {};
  for (const s of sessionRows) {
    const sid = s.id as string;
    if (s.detail === true && sid !== focusId) {
      details[sid] = buildPayload(new PinnedReader(reader, sid), contextWindow);
    }
  }

  const payload = buildPayload(new PinnedReader(reader, focusId), contextWindow);
  payload.project = {
    name: focusRun.project,
    sessions_total: sessions.length,
    detail_cap: DETAIL_SESSIONS,
    focus: focusId,
    start: startLevel(agent, opts.sessionSelected === true, agentRows, sessionRows, focusId),
    agents: agentRows,
    sessions: sessionRows,
    details,
  };
  return payload;
}

// --- serialization + write -----------------------------------------------------

/** Serialize `payload` for the JSON island: `pyJsonDumps` (Python-identical),
 * then escape every `</` to `<\/` so no block text can terminate the script
 * tag early. Mirrors Python `_embed_json`. */
function embedJson(payload: Record<string, unknown>): string {
  return pyJsonDumps(payload).replace(/<\//g, "<\\/");
}

/**
 * Export the trace at `ctracePath` to a self-contained HTML dashboard and
 * return the written path. `outPath` overrides the destination; by default the
 * file is written as `<trace-stem>.html` next to the trace.
 *
 * The dashboard covers the WHOLE project the file holds — every agent, every
 * session — with the newest session in focus; `opts` preselects which level it
 * opens on. Mirrors Python `export_html`.
 */
export function exportHtml(
  ctracePath: string,
  outPath?: string,
  opts: ProjectPayloadOptions = {},
): string {
  const ct = CTrace.open(ctracePath);
  let payload: Record<string, unknown>;
  try {
    payload = buildProjectPayload(ct, opts);
  } finally {
    ct.close();
  }

  let out = outPath;
  if (out === undefined) {
    const abs = resolve(ctracePath);
    const stem = basename(abs).replace(/\.[^.]*$/, "");
    out = join(dirname(abs), `${stem}.html`);
  }

  writeFileSync(out, renderPayload(payload), "utf-8");
  return out;
}

/**
 * Export an already-open READER — a `CTrace`, or an in-memory snapshot of a
 * Postgres/MySQL session (see `store/snapshot.ts`) — to a self-contained HTML
 * dashboard at `outPath`, returning the path written.
 *
 * The counterpart to `exportHtml` for stores that have no file: `--out` (or
 * `view`'s temp file) is REQUIRED here precisely because there is no trace path
 * to derive `<stem>.html` from. The reader's BOUND session becomes the
 * dashboard's focus; the project index around it covers every session the reader
 * can list. Everything downstream of `buildProjectPayload` is shared, so a
 * dashboard rendered from a database is byte-identical to one rendered from the
 * equivalent `.ctrace`. Mirrors Python `export_store`.
 */
export function exportStore(
  reader: ProjectReader,
  outPath: string,
  opts: ProjectPayloadOptions = {},
): string {
  writeFileSync(outPath, renderPayload(buildProjectPayload(reader, opts)), "utf-8");
  return outPath;
}

/** Render one payload into the full standalone document — the last step both
 * export entry points share, kept in one place so a file-backed and a
 * database-backed dashboard can never drift. */
function renderPayload(payload: Record<string, unknown>): string {
  const project = (payload.run as { project: string }).project;
  return renderPage(htmlEscape(`ctxdiff — ${project}`), embedJson(payload));
}
