/**
 * ctxdiff — a local-first context-window debugger for LLM agents (JS/TS SDK).
 *
 * Public surface is deliberately tiny:
 *
 *   import { trace } from "ctxdiff";
 *   const tracer = trace.init("my-project");
 *   const client = tracer.wrap(new OpenAI());
 *   // ...use `client` exactly like an OpenAI client...
 *   tracer.close();
 *
 * The `.ctrace` file it writes opens in the Python `ctxdiff view`.
 */
import { init, Tracer } from "./trace.js";

export { init, Tracer } from "./trace.js";
export type { InitOptions, WrapOptions } from "./trace.js";
export { CTrace } from "./store/ctrace.js";
export { SCHEMA_VERSION, DDL } from "./store/schema.js";
export { VERSION } from "./version.js";
export {
  normalizeText,
  stableStringify,
  contentHash,
  basicLabel,
} from "./models.js";
export type { Block, RawBlock, CallBlock, Run, Call } from "./models.js";
export { countTokens } from "./tokenize.js";
export type { Adapter } from "./capture/base.js";
export { OpenAIAdapter } from "./capture/openai.js";
export { AnthropicAdapter } from "./capture/anthropic.js";
export { GeminiAdapter } from "./capture/gemini.js";
export { Recorder, type RedactHook } from "./capture/recorder.js";

// --- read-side analyzers (parity with Python's `ctxdiff.analyze`) -----------
export {
  diffCalls,
  diffTurns,
  filterCalls,
  agentCalls,
  distinctAgents,
} from "./analyze/diff.js";
export type { DiffEntry, DiffKind, TurnDiff, InlineSegment } from "./analyze/diff.js";
export {
  analyzeRun,
  analyzeCall,
  usageTotals,
  detectBloat,
  extractToolName,
  registeredToolNames,
} from "./analyze/tokens.js";
export type {
  LabelSlice,
  CallTokens,
  BloatReport,
  UsageTotals,
  RunTokens,
} from "./analyze/tokens.js";
export { analyzeCache } from "./analyze/cache.js";
export type { PrefixBreak, CacheReport } from "./analyze/cache.js";
export { listRuns } from "./analyze/runs.js";
export type { RunRow } from "./analyze/runs.js";
export {
  renderTurnDiff,
  renderRunTokens,
  renderCallTokens,
  renderBloat,
  renderUsageSummary,
  renderAgentSummary,
  renderCacheReport,
  renderRunsList,
} from "./render.js";

/** Namespace mirroring the Python `from ctxdiff import trace` entry point, so
 * `trace.init(...)` reads the same across both SDKs. */
export const trace = { init, Tracer };

export default trace;
