/**
 * The exporter: turn a `.ctrace` into one self-contained HTML dashboard. A
 * faithful port of Python `ctxdiff.viewer.export` — `buildPayload` assembles the
 * exact same JSON-serializable payload (same fields, same key order, embedding
 * the same precomputed analyzer output), and `exportHtml` embeds it in the
 * lifted-verbatim template. Together with `pyJsonDumps` this makes the emitted
 * HTML byte-identical to the Python viewer's for the same trace.
 *
 * Security (audited): the payload is embedded as a JSON island inside a
 * `<script type="application/json">` tag, and every `</` in the serialized JSON
 * is escaped to `<\/`, so block text containing `</script>` (or any markup)
 * cannot break out of the tag. The template's runtime reads the island with
 * `.textContent` and renders all block text with `.textContent` (never
 * `.innerHTML`), so untrusted trace data can never execute. Zero external
 * requests: every byte (CSS/JS/data) is inline; no URL of any kind.
 */
import { writeFileSync } from "node:fs";
import { basename, dirname, resolve, join } from "node:path";
import { CTrace } from "../store/ctrace.js";
import type { ReadableStore } from "../store/base.js";
import type { Call, CallBlock } from "../models.js";
import { diffTurns, distinctAgents, type DiffEntry, type TurnDiff } from "../analyze/diff.js";
import { analyzeRun, type RunTokens } from "../analyze/tokens.js";
import { analyzeCache, type CacheReport } from "../analyze/cache.js";
import { renderPage } from "./template.js";
import { PyFloat, pyJsonDumps, htmlEscape } from "./pyjson.js";

// How much of a diff entry's block text to embed, and how much of each inline
// segment to keep — previews, not whole documents (matches Python).
const SNIPPET_CHARS = 200;
const INLINE_SEG_CHARS = 500;

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

/** A RunTokens → a JSON dict (per-call slices + run-level bloat). Percentages
 * are wrapped in PyFloat so they serialize as Python floats. Mirrors Python
 * `_serialize_tokens`. */
function serializeTokens(rt: RunTokens): Record<string, unknown> {
  return {
    calls: rt.calls.map((c) => ({
      seq: c.seq,
      total: c.totalTokens,
      approximate: c.approximate,
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

// --- payload assembly ----------------------------------------------------------

/**
 * Assemble every fact the dashboard renders into one JSON-serializable object.
 * Pure: reads from `ct` and the three analyzers and returns a plain object — no
 * filesystem, no template, no HTML. Delegates the hard analysis to the existing
 * pure analyzers (single source of truth) and serializes their results with the
 * SAME field order as the Python builder. Mirrors Python `build_payload`.
 */
export function buildPayload(ct: ReadableStore): Record<string, unknown> {
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
  const tokensOut = serializeTokens(runTokens);
  const cacheOut = serializeCache(analyzeCache(ct));

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

  const u = runTokens.usage;
  let byAgent: Record<string, [number, number]> | null = null;
  if (u.byAgent !== null) {
    byAgent = {};
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
    stats: {
      distinct_blocks: distinct.size,
      total_block_refs: totalRefs,
      context_growth: growth,
      agents: agentsStats,
      usage: usageStats,
    },
  };
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
 * file is written as `<trace-stem>.html` next to the trace. Mirrors Python
 * `export_html`.
 */
export function exportHtml(ctracePath: string, outPath?: string): string {
  const ct = CTrace.open(ctracePath);
  let payload: Record<string, unknown>;
  try {
    payload = buildPayload(ct);
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
 * to derive `<stem>.html` from. Everything downstream of `buildPayload` is
 * shared, so a dashboard rendered from a database is byte-identical to one
 * rendered from the equivalent `.ctrace`. Mirrors Python `export_store`.
 */
export function exportStore(reader: ReadableStore, outPath: string): string {
  writeFileSync(outPath, renderPayload(buildPayload(reader)), "utf-8");
  return outPath;
}

/** Render one payload into the full standalone document — the last step both
 * export entry points share, kept in one place so a file-backed and a
 * database-backed dashboard can never drift. */
function renderPayload(payload: Record<string, unknown>): string {
  const project = (payload.run as { project: string }).project;
  return renderPage(htmlEscape(`ctxdiff — ${project}`), embedJson(payload));
}
