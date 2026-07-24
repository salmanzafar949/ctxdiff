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
export { Recorder, type RedactHook } from "./capture/recorder.js";

/** Namespace mirroring the Python `from ctxdiff import trace` entry point, so
 * `trace.init(...)` reads the same across both SDKs. */
export const trace = { init, Tracer };

export default trace;
