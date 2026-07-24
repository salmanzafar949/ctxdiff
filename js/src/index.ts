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
export { CTrace, parseStartedAt } from "./store/ctrace.js";
export { SCHEMA_VERSION, DDL } from "./store/schema.js";

// --- pluggable storage (parity with Python's `ctxdiff.store`) ---------------
// Local-first stays the default: with nothing configured, `trace.init(project)`
// writes `./<project>.ctrace` exactly as before. `configure({ store })` — or
// `CTXDIFF_STORE` with no code change at all — points every later `init()` at a
// database you already run. The drivers (`pg`, `mysql2`) are optional peer
// dependencies, imported only when a connection is actually opened.
export { configure, configured, resolve as resolveStore, fromDsn, ENV_VAR } from "./store/config.js";
export type { ConfigureOptions } from "./store/config.js";
export { SQLiteStore } from "./store/sqlite.js";
export { PostgresStore } from "./store/postgres.js";
export type { PostgresStoreOptions } from "./store/postgres.js";
export { MySQLStore } from "./store/mysql.js";
export type { MySQLStoreOptions } from "./store/mysql.js";
export { EmptyStoreError, isFileBackend } from "./store/base.js";
export type {
  Awaitable,
  OpenSessionArgs,
  ReadableStore,
  RecordCallArgs,
  Store,
  StoreBackend,
} from "./store/base.js";
export { snapshotStore, StoreSnapshot, type SnapshotOptions } from "./store/snapshot.js";
export { VERSION } from "./version.js";
export {
  normalizeText,
  stableStringify,
  contentHash,
  basicLabel,
} from "./models.js";
export type { Block, RawBlock, CallBlock, Run, Call, Session } from "./models.js";
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

// --- viewer + demo (parity with Python's `ctxdiff.viewer` / `ctxdiff.demo`) --
export { buildPayload, exportHtml, exportStore } from "./viewer/export.js";
export { renderPage, PAGE } from "./viewer/template.js";
export { buildDemoTrace } from "./demo.js";

/** Namespace mirroring the Python `from ctxdiff import trace` entry point, so
 * `trace.init(...)` reads the same across both SDKs. */
export const trace = { init, Tracer };

export default trace;
