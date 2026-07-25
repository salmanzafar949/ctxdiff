/**
 * The tagged-eviction detector: "the block you tagged 'rag' at turn 3 was
 * evicted at turn 6". A faithful port of Python `ctxdiff.analyze.evictions` —
 * same reuse of the differ, same three narrowings, same merge order.
 *
 * Why this is its own report. "The agent forgot the thing I told it" is the
 * single most common bug an agent developer brings to a context debugger, and it
 * almost always has the same mechanical cause: a block that WAS in the context
 * is not in it any more — trimmed by a sliding window, dropped by a summarizer,
 * rebuilt from a shorter history. ctxdiff already sees that happen (the differ
 * classifies it as `evicted`) and already knows which blocks the developer
 * considered load-bearing (`tracer.tag()` marks them, `labelSource === "tagged"`).
 * This module is the join: it names WHERE the block entered and WHERE it
 * disappeared, in those words.
 *
 * Nothing here re-implements eviction detection. Every disappearance is one of
 * `diffCalls`'s own `evicted` entries — which matters beyond tidiness, because a
 * naive "was this hash present last turn?" scan would report a block whose text
 * was EDITED as an eviction, while the differ correctly classifies a same-slot
 * content change as `modified` and a pure reorder as `unchanged`. Both of those
 * are things developers do to their prompts on purpose.
 *
 * One consequence of that reuse is worth stating plainly, because it is a BLIND
 * SPOT and it is deliberate: a tagged block REPLACED IN THE SAME SLOT by
 * different text of the same role and kind reads to the differ as an edit
 * (`modified`), so no eviction is reported for it. Teaching this module to
 * second-guess that would mean a similarity heuristic of its own — a second
 * opinion about something the differ already decided — and the two would then
 * disagree in `ctxdiff diff` and `ctxdiff tokens` over the same trace. Whatever
 * the differ calls it, this report calls it.
 *
 * Three deliberate narrowings, each there to keep a warning worth reading:
 *
 * 1. TAGGED ONLY (`labelSource === "tagged"`). Heuristic labels are never
 *    included: every multi-turn agent evicts heuristically-labeled history
 *    constantly — that is what a context window IS — so including them would
 *    produce a wall of warnings about the intended behaviour of every framework
 *    in existence. A tag is the developer's own statement that this content is
 *    supposed to stay, which is exactly the assertion an eviction violates.
 *
 *    Taggedness is a property of the CONTENT, not of one call, and that is what
 *    makes the feature work at all. `tracer.tag()` is NEXT-CALL-ONLY by design:
 *    it buffers labels for the next recorded call and clears itself, so a block
 *    tagged once is stored `tagged` on exactly ONE call and `heuristic` on every
 *    later call that still carries the same text. An eviction is always detected
 *    on the pair (last turn that had it, first turn that did not), so asking
 *    "was it tagged on the older side of THIS pair?" answers no for everything
 *    tagged earlier than the turn before it vanished — which silenced the module
 *    on its own headline example ("tagged at turn 1 … evicted at turn 5"). The
 *    question asked here is "did the developer EVER vouch for this content hash,
 *    anywhere in this agent's timeline?", and the label and turn quoted back are
 *    the ones from the FIRST call that did.
 * 2. PER AGENT, the rule `analyzeCache` established. On an interleaved timeline
 *    the researcher's block is "missing" from the writer's very next call
 *    because it was never in the writer's context at all; that is a hand-off,
 *    not an eviction.
 * 3. PERMANENT ONLY. A block that is absent for one turn and back the next was
 *    not forgotten. A block that leaves twice is reported once, for the
 *    departure it did not return from.
 *
 * No I/O beyond reading the store, no color.
 */
import { flattenSnippet } from "./cache.js";
import { diffCalls, distinctAgents, filterCalls } from "./diff.js";
import type { Call, CallBlock } from "../models.js";
import type { ReadableStore } from "../store/base.js";

// --- value types -------------------------------------------------------------

/**
 * One tagged block's disappearance from an agent's context. Mirrors Python
 * `TaggedEviction`.
 *
 * `label` is the tag the developer gave it, which is also what the report calls
 * it, and `taggedSeq` is the turn they gave it on — the FIRST call of this agent
 * that carried this content as `tagged`. The two always come from the same call,
 * because quoting a label against a turn that did not carry it describes a tag
 * that never existed.
 *
 * `enteredSeq` is the first turn of this agent that carried the CONTENT at all,
 * tagged or not — usually the same turn as `taggedSeq`, and a separate field
 * precisely for when it is not. `lastSeenSeq` is the last turn that still
 * carried it and `evictedSeq` the first that did not; those two are adjacent
 * within the agent's own timeline but can be far apart in global turn numbers
 * when agents interleave, which is why both are reported. `contentHash` is
 * carried for identity (and as the deterministic tiebreak in the sort order),
 * never displayed.
 */
export interface TaggedEviction {
  label: string;
  agent: string | null;
  taggedSeq: number;
  enteredSeq: number;
  lastSeenSeq: number;
  evictedSeq: number;
  tokens: number;
  role: string;
  snippet: string;
  contentHash: string;
}

/**
 * Every tagged eviction in a run, in timeline order. Mirrors Python
 * `EvictionReport`.
 *
 * `pairsAnalyzed` is how many consecutive same-agent turn pairs were actually
 * compared — the honest denominator, and the number that distinguishes "no
 * tagged block was evicted" from "there was nothing to compare".
 * `agentsAnalyzed` is the number of per-agent groups an unfiltered multi-agent
 * run was split into, and null when the run was analyzed as a single timeline.
 * `taggedBlocks` counts the distinct tagged blocks seen at all, so a report can
 * say "nothing was tagged" rather than implying it checked something; distinct
 * means distinct PER AGENT TIMELINE, the same scope every other number here
 * uses, so it stays the denominator `evictions` is the numerator of.
 */
export interface EvictionReport {
  evictions: TaggedEviction[];
  pairsAnalyzed: number;
  agentsAnalyzed: number | null;
  taggedBlocks: number;
}

// --- per-agent core -----------------------------------------------------------

/**
 * Index one call's blocks by their position, so a differ entry's `positionOld`
 * can be resolved back to the CallBlock that produced it. Position rather than
 * content hash: a call may legitimately carry the same text twice under
 * different labels, and only the position identifies which membership the differ
 * was talking about. The index is UNCONDITIONAL — every block, not just the ones
 * this call happened to store as `tagged` — because whether the developer
 * vouched for the content is decided over the whole timeline by `taggedLabels`,
 * not by the single call the eviction pair happens to start from.
 */
function byPosition(blocks: CallBlock[]): Map<number, CallBlock> {
  const out = new Map<number, CallBlock>();
  for (const cb of blocks) out.set(cb.position, cb);
  return out;
}

/**
 * Map each content hash the developer EVER tagged in this group to the [label,
 * turn] of the first call that tagged it. First and not last: a re-tag under a
 * second name later must not make the report quote the newer name, because the
 * sentence names one turn and the label it quotes has to be the one that turn
 * actually carried.
 */
function taggedLabels(
  calls: Call[],
  blocksByCallId: Map<string, CallBlock[]>,
): Map<string, [string, number]> {
  const labels = new Map<string, [string, number]>();
  for (const c of calls) {
    for (const cb of blocksByCallId.get(c.id) ?? []) {
      if (cb.labelSource === "tagged" && !labels.has(cb.block.contentHash)) {
        labels.set(cb.block.contentHash, [cb.label, c.seq]);
      }
    }
  }
  return labels;
}

/**
 * Find one agent's tagged evictions. Returns the events, how many pairs were
 * compared, and the distinct tagged content hashes seen.
 *
 * Three passes over data that is already loaded: record each hash's first and
 * last appearance and the label/turn of the first call that tagged it; ask
 * `diffCalls` what happened between each consecutive pair; keep an `evicted`
 * entry when the content it names was tagged SOMEWHERE in this group and its
 * last appearance anywhere in the group is that older turn.
 *
 * `reported` collapses the one case that yields two events for one loss: a call
 * that carried the same text twice. The differ correctly reports two `evicted`
 * entries — two memberships really did disappear — but the developer lost one
 * block, and counting memberships is what printed the same stanza twice and made
 * `check` say "2 tagged blocks evicted of 1". The duplicates are byte-identical
 * anyway: same content hash means same text, role and token count, and the label
 * and turn come from `taggedLabels` rather than from the membership.
 */
function analyzeGroup(
  calls: Call[],
  blocksByCallId: Map<string, CallBlock[]>,
  agentLabel: string | null,
): { evictions: TaggedEviction[]; pairs: number; tagged: Set<string> } {
  const firstSeen = new Map<string, number>();
  const lastSeen = new Map<string, number>();
  for (const c of calls) {
    for (const cb of blocksByCallId.get(c.id) ?? []) {
      const h = cb.block.contentHash;
      if (!firstSeen.has(h)) firstSeen.set(h, c.seq);
      lastSeen.set(h, c.seq);
    }
  }
  const tagged = taggedLabels(calls, blocksByCallId);

  const evictions: TaggedEviction[] = [];
  const reported = new Set<string>();
  let pairs = 0;
  for (let i = 1; i < calls.length; i++) {
    const prevCall = calls[i - 1];
    const call = calls[i];
    pairs += 1;
    if (tagged.size === 0) continue; // nothing vouched for could have been lost
    const old = blocksByCallId.get(prevCall.id) ?? [];
    const next = blocksByCallId.get(call.id) ?? [];
    const oldByPosition = byPosition(old);
    const turnDiff = diffCalls(old, next, prevCall.seq, call.seq);
    for (const entry of turnDiff.entries) {
      if (entry.kind !== "evicted" || entry.positionOld == null) continue;
      const cb = oldByPosition.get(entry.positionOld);
      if (cb === undefined) continue;
      const h = cb.block.contentHash;
      const vouched = tagged.get(h);
      if (vouched === undefined) continue; // evicted, but never tagged
      if ((lastSeen.get(h) ?? prevCall.seq) > prevCall.seq) continue; // it comes back
      if (reported.has(h)) continue; // a second membership of the same lost block
      reported.add(h);
      const [label, taggedSeq] = vouched;
      evictions.push({
        label,
        agent: agentLabel,
        taggedSeq,
        enteredSeq: firstSeen.get(h)!,
        lastSeenSeq: prevCall.seq,
        evictedSeq: call.seq,
        tokens: cb.block.tokenCount,
        role: cb.block.role,
        snippet: flattenSnippet(cb.block.text),
        contentHash: h,
      });
    }
  }
  return { evictions, pairs, tagged: new Set(tagged.keys()) };
}

// --- the entry point -----------------------------------------------------------

/**
 * Find every tagged block that entered an agent's context and later left it for
 * good (see the module docstring for the three narrowings). Grouping mirrors
 * `analyzeCache` exactly, because the question has the same shape — "what
 * happened between two turns of the same agent".
 *
 * Events are merged in a deterministic order — by the turn the block
 * disappeared, then the turn it entered, then its content hash — so the two
 * SDKs list them identically even though they arrive grouped by agent.
 */
export function analyzeEvictions(
  ct: ReadableStore,
  agent: string | null = null,
): EvictionReport {
  const calls = filterCalls(ct.getCalls(), agent);
  const blocksByCallId = new Map<string, CallBlock[]>();
  for (const c of calls) blocksByCallId.set(c.id, ct.getCallBlocks(c.id));

  const labels = distinctAgents(calls);
  const grouped = agent === null && labels.length > 1;
  let groups: [string | null, Call[]][];
  let agentsAnalyzed: number | null;
  if (grouped) {
    groups = labels.map((lbl) => [lbl, calls.filter((c) => c.agent === lbl)]);
    agentsAnalyzed = labels.length;
  } else {
    const sole = agent !== null ? agent : labels.length ? labels[0] : null;
    groups = [[sole, calls]];
    agentsAnalyzed = null;
  }

  const allEvictions: TaggedEviction[] = [];
  let pairsAnalyzed = 0;
  // Keyed by agent AND hash rather than by hash alone, so the count stays the
  // denominator the eviction list is the numerator of: a group emits at most one
  // event per tagged hash, so the same text tagged in two agents' contexts is two
  // tagged blocks that can be lost twice — and collapsing them here would
  // reintroduce, across agents, the very "N evicted of fewer than N" the
  // per-group dedupe removes within one.
  const taggedHashes = new Set<string>();
  for (const [label, groupCalls] of groups) {
    const { evictions, pairs, tagged } = analyzeGroup(groupCalls, blocksByCallId, label);
    allEvictions.push(...evictions);
    pairsAnalyzed += pairs;
    // `JSON.stringify` of the pair is the injective string key Python gets
    // for free from a tuple (a JS Set compares arrays by identity, so two
    // equal pairs would both be kept).
    for (const h of tagged) taggedHashes.add(JSON.stringify([label, h]));
  }

  // By the turn the block disappeared, then the turn it entered, then its
  // content hash. The hash and not the tag as the final tiebreak on purpose: it
  // is unique per block (so the order is total), and it is lowercase hex — which
  // JS and Python compare identically, while an arbitrary user tag could contain
  // astral characters that UTF-16 and code-point ordering disagree about.
  allEvictions.sort(
    (a, b) =>
      a.evictedSeq - b.evictedSeq ||
      a.enteredSeq - b.enteredSeq ||
      (a.contentHash < b.contentHash ? -1 : a.contentHash > b.contentHash ? 1 : 0),
  );

  return {
    evictions: allEvictions,
    pairsAnalyzed,
    agentsAnalyzed,
    taggedBlocks: taggedHashes.size,
  };
}
