/**
 * The JAVASCRIPT half of ctxdiff's cross-SDK golden harness — the mirror of
 * `spec/golden/harness.py`, reading the SAME `spec/golden/manifest.json`, the
 * SAME `spec/golden/corpus/*.json` scenarios and the SAME committed
 * expectations under `spec/golden/expected/`.
 *
 * What it is for. ctxdiff's two SDKs count tokens with two different libraries
 * — `gpt-tokenizer` here, `tiktoken` in Python — that reimplement the same
 * `o200k_base` BPE table on independent release cadences. A `.ctrace` stays
 * portable regardless (token counts are not part of a block's content hash),
 * but every RENDERED number is downstream of a token count. The live
 * conformance suites compare the two SDKs against each other at whatever
 * versions happen to be installed, so a drift that hits BOTH — or one that
 * happens on a machine where only one SDK runs — is invisible there. The
 * committed goldens close that gap: each SDK must reproduce a number that was
 * written down, reviewed and frozen.
 *
 * Why the fixtures are rebuilt here instead of a committed binary `.ctrace`
 * being read: a stored `.ctrace` carries token counts baked in by whoever
 * generated it, so the tokenizer would never run and the check would be
 * vacuous. Blocks are hashed, counted and labeled by this SDK's own
 * `contentHash` / `countTokens` / `basicLabel` — the same three the recorder
 * uses — so the tokenizer is genuinely exercised at check time.
 *
 * Why rows are written with plain SQL rather than `CTrace.recordCall`: the
 * goldens must be byte-stable, and the writer path mints `uuid4` ids and stamps
 * a wall clock, both of which reach the rendered output (`ctxdiff sessions`
 * prints a short session id and a local-time column). The scenarios therefore
 * carry explicit ids and `startedAt` values. The writer path is not left
 * untested — `test/conformance.test.ts` drives it across the language boundary;
 * the goldens deliberately target the READ side, where tokenizer drift shows.
 */
import { DatabaseSync } from "node:sqlite";
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { basicLabel, contentHash } from "../../src/models.js";
import { countTokens } from "../../src/tokenize.js";
import { DDL, SCHEMA_VERSION } from "../../src/store/schema.js";

// vitest runs with cwd = the js/ package dir; the shared corpus lives at the
// repo root so BOTH SDKs read one copy of it.
const repoRoot = resolve(process.cwd(), "..");
export const GOLDEN_DIR = join(repoRoot, "spec", "golden");
const EXPECTED_DIR = join(GOLDEN_DIR, "expected");

export interface Manifest {
  tz: string;
  no_color?: boolean;
  ctxdiff_version: string;
  tokenizers: Record<string, { package: string; pinned_version: string; encoding: string }>;
  fixtures: { id: string; corpus: string; ctrace: string }[];
  cli_cases: { name: string; fixture: string; argv: string[] }[];
  html_cases: { name: string; fixture: string }[];
}

interface CorpusCall {
  call_id: string;
  seq: number;
  agent: string | null;
  step: string | null;
  provider: string | null;
  params: Record<string, unknown>;
  usage: Record<string, unknown> | null;
  latency_ms: number | null;
  error: string | null;
  tags?: [string, string][];
  blocks: [string, string, string][];
}

interface CorpusSession {
  run_id: string;
  project: string;
  started_at: string;
  provider: string;
  models: string[];
  calls: CorpusCall[];
}

interface Corpus {
  id: string;
  sessions: CorpusSession[];
}

/** Read the shared manifest — the single description of the corpus, the cases
 * and the tokenizer pins that both SDKs consume. */
export function loadManifest(): Manifest {
  return JSON.parse(readFileSync(join(GOLDEN_DIR, "manifest.json"), "utf8")) as Manifest;
}

/** Read one fixture scenario by id, resolving its path through the manifest so
 * the manifest stays the only place a fixture is named. */
function loadCorpus(manifest: Manifest, fixtureId: string): Corpus {
  const entry = manifest.fixtures.find((f) => f.id === fixtureId);
  if (!entry) throw new Error(`no fixture ${fixtureId} in the golden manifest`);
  return JSON.parse(readFileSync(join(GOLDEN_DIR, entry.corpus), "utf8")) as Corpus;
}

/**
 * Materialize one corpus fixture into a real `.ctrace` under `outDir`, exactly
 * as `harness.build_ctrace` does on the Python side, and return its path.
 *
 * The file is named from the manifest (`unicode.ctrace`, …) because the name is
 * USER-VISIBLE: `ctxdiff sessions` labels each row with the trace's basename, so
 * a random temp name would leak straight into the compared output.
 *
 * `ctxdiff_version` comes from the manifest rather than this package's own
 * VERSION constant: the two SDKs version independently and that field is
 * embedded verbatim in the HTML payload, so without the pin the HTML hash could
 * never be shared between them.
 */
export function buildCtrace(manifest: Manifest, fixtureId: string, outDir: string): string {
  const entry = manifest.fixtures.find((f) => f.id === fixtureId)!;
  const corpus = loadCorpus(manifest, fixtureId);
  const path = join(outDir, entry.ctrace);

  const db = new DatabaseSync(path);
  try {
    db.exec(DDL);
    const insertRun = db.prepare("INSERT INTO run VALUES (?,?,?,?,?,?,?)");
    const insertCall = db.prepare("INSERT INTO call VALUES (?,?,?,?,?,?,?,?,?,?)");
    const insertBlock = db.prepare("INSERT OR IGNORE INTO block VALUES (?,?,?,?,?,?)");
    const insertCallBlock = db.prepare("INSERT INTO call_block VALUES (?,?,?,?,?)");

    for (const session of corpus.sessions) {
      insertRun.run(
        session.run_id,
        session.project,
        session.started_at,
        session.provider,
        JSON.stringify(session.models),
        manifest.ctxdiff_version,
        SCHEMA_VERSION,
      );
      for (const call of session.calls) {
        insertCall.run(
          call.call_id,
          session.run_id,
          call.seq,
          pyParamsJson(call.params),
          call.usage === null || call.usage === undefined ? null : pyParamsJson(call.usage),
          call.latency_ms ?? null,
          call.error ?? null,
          call.agent ?? null,
          call.step ?? null,
          call.provider ?? null,
        );
        // `tags` mirrors what `tracer.tag(label, needle)` registers: a
        // (label, needle) pair whose needle, found inside a block, overrides
        // the role-based label. Routing it through `basicLabel` keeps the
        // tagged path on the compared surface.
        const tagged = (call.tags ?? []) as [string, string][];
        const provider = call.provider ?? session.provider;
        call.blocks.forEach(([role, kind, text], position) => {
          const [tokenCount, tokenMethod] = countTokens(text, provider);
          const hash = contentHash(role, kind, text);
          const [label, labelSource] = basicLabel(role, kind, text, tagged);
          insertBlock.run(hash, role, kind, text, tokenCount, tokenMethod);
          insertCallBlock.run(call.call_id, hash, position, label, labelSource);
        });
      }
    }
  } finally {
    db.close();
  }
  return path;
}

/**
 * Serialize a params/usage object the way CPython's `json.dumps` does, so the
 * `.ctrace` this builder writes is byte-identical to the one the Python builder
 * writes from the same scenario.
 *
 * Only the container spacing differs between the two languages' defaults
 * (`", "` / `": "` vs `,` / `:`), and only for the two JSON columns this builder
 * fills by hand. Both readers `json.loads`/`JSON.parse` these columns before
 * using them, so the spacing never reaches rendered output — but the raw bytes
 * are what a byte-comparison of the two builders' files would catch, and having
 * them match keeps that comparison meaningful. Values here are plain
 * strings/numbers/booleans/null from a JSON scenario, so no float-repr shim is
 * needed (see `src/viewer/pyjson.ts` for the case that does need one).
 */
function pyParamsJson(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "string") return JSON.stringify(value);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") return String(value);
  if (Array.isArray(value)) return "[" + value.map(pyParamsJson).join(", ") + "]";
  return (
    "{" +
    Object.entries(value as Record<string, unknown>)
      .map(([k, v]) => `${JSON.stringify(k)}: ${pyParamsJson(v)}`)
      .join(", ") +
    "}"
  );
}

/** Build every fixture into `outDir`, returning {fixture id -> path}. */
export function buildAll(manifest: Manifest, outDir: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const f of manifest.fixtures) out[f.id] = buildCtrace(manifest, f.id, outDir);
  return out;
}

/** Resolve a manifest case's argv, substituting `{trace}` with the built
 * fixture's path — the same placeholder contract the Python harness honors. */
export function caseArgv(
  c: { fixture: string; argv: string[] },
  traces: Record<string, string>,
): string[] {
  return c.argv.map((a) => a.replace("{trace}", traces[c.fixture]));
}

/** Where one CLI case's committed stdout lives. */
export function cliGoldenPath(name: string): string {
  return join(EXPECTED_DIR, "cli", `${name}.txt`);
}

/** Where one HTML case's committed {sha256, bytes} lives. */
export function htmlGoldenPath(name: string): string {
  return join(EXPECTED_DIR, "html", `${name}.json`);
}

/** The committed stdout for a CLI case. A missing file THROWS rather than
 * returning "": "the expectation does not exist" must fail the check, not
 * quietly compare against nothing. */
export function readCliGolden(name: string): string {
  const path = cliGoldenPath(name);
  if (!existsSync(path)) {
    throw new Error(
      `no golden for CLI case '${name}' at ${path} — run ` +
        "`npm run golden:regen` and review the diff",
    );
  }
  return readFileSync(path, "utf8");
}

/** The committed {sha256, bytes} for an HTML case; missing throws, for the same
 * reason as `readCliGolden`. */
export function readHtmlGolden(name: string): { sha256: string; bytes: number } {
  const path = htmlGoldenPath(name);
  if (!existsSync(path)) {
    throw new Error(
      `no golden for HTML case '${name}' at ${path} — run ` +
        "`npm run golden:regen` and review the diff",
    );
  }
  return JSON.parse(readFileSync(path, "utf8")) as { sha256: string; bytes: number };
}

/** The {sha256, bytes} shape a rendered dashboard is compared by. A hash plus a
 * size rather than the whole document: see the rationale in
 * `spec/golden/harness.py:render_html_case` — the same decision, applied to the
 * same files, has to be spelled the same way on both sides. */
export function hashHtml(path: string): { sha256: string; bytes: number } {
  const data = readFileSync(path);
  return { sha256: createHash("sha256").update(data).digest("hex"), bytes: data.length };
}

/** The gpt-tokenizer version actually installed under `js/node_modules`, used to
 * assert the environment matches the manifest pin. Read from the package's own
 * `package.json` (the INSTALLED artifact) rather than from our dependency range,
 * because a drifting resolver is exactly what changes the former and not the
 * latter. */
export function installedTokenizerVersion(): string {
  const path = join(process.cwd(), "node_modules", "gpt-tokenizer", "package.json");
  return (JSON.parse(readFileSync(path, "utf8")) as { version: string }).version;
}
