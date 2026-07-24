/**
 * Read/write access to a `.ctrace` file — a plain SQLite database holding one
 * run, its calls, and content-addressed blocks. Writers dedup blocks by hash;
 * readers reconstruct ordered CallBlocks. No analysis lives here.
 *
 * Storage is `node:sqlite`'s built-in `DatabaseSync` (zero runtime deps). The
 * file format is byte-compatible with the Python SDK's `CTrace`: same DDL, same
 * SCHEMA_VERSION, same JSON column encodings, same dedup + per-run model
 * roll-up semantics — so a file written here opens in the Python reader and
 * vice-versa. `startedAt` is passed in by the caller (the tracer) rather than
 * read from the clock here, keeping the store a pure I/O layer.
 */
import { DatabaseSync } from "node:sqlite";
import { randomUUID } from "node:crypto";
import type { Block, Call, CallBlock, Run } from "../models.js";
import { DDL, SCHEMA_VERSION } from "./schema.js";
import { VERSION } from "../version.js";

/** A uuid4 hex (32 chars, no dashes) — matches Python's `uuid.uuid4().hex`. */
function uuidHex(): string {
  return randomUUID().replace(/-/g, "");
}

export class CTrace {
  private db: DatabaseSync;
  private runId: string;
  private hasV2Cols: boolean;
  private models: string[];
  private modelsSeen: Set<string>;

  /**
   * Wrap an already-open, already-initialized database for one run. Snapshots
   * the `call` table's column set ONCE (via PRAGMA table_info) so reads/writes
   * can adapt to a v1 file (which lacks the agent/step/provider columns) without
   * a PRAGMA per query, and seeds an in-memory mirror of the run's `models`
   * column so `noteModel` gates on a cheap set lookup instead of a SELECT per
   * call. Construct via `create()`/`open()` — never directly.
   */
  private constructor(db: DatabaseSync, runId: string) {
    this.db = db;
    this.runId = runId;
    const cols = new Set(
      (db.prepare("PRAGMA table_info(call)").all() as { name: string }[]).map(
        (r) => r.name,
      ),
    );
    this.hasV2Cols = ["agent", "step", "provider"].every((c) => cols.has(c));
    const row = db
      .prepare("SELECT models FROM run WHERE id = ?")
      .get(runId) as { models: string } | undefined;
    this.models = row ? (JSON.parse(row.models) as string[]) : [];
    this.modelsSeen = new Set(this.models);
  }

  // --- construction --------------------------------------------------------

  /**
   * Create a fresh `.ctrace` at `path`, apply the schema, and write the single
   * run row. Foreign keys are enabled so referential integrity holds.
   * `models` starts as `[model]` when a real model id is passed, but as an
   * EMPTY list when `model` is falsy: the run's model is really a per-CALL fact
   * (`wrap()` doesn't know it yet at run-creation time), so seeding a
   * placeholder would just store a bogus blank forever. `noteModel()` populates
   * `models` from real call params as they arrive. Mirrors Python `create`.
   */
  static create(
    path: string,
    project: string,
    provider: string,
    model: string,
    startedAt = "",
  ): CTrace {
    const db = new DatabaseSync(path);
    try {
      db.exec("PRAGMA foreign_keys = ON");
      db.exec(DDL);
      const runId = uuidHex();
      const models = model ? [model] : [];
      db.prepare("INSERT INTO run VALUES (?,?,?,?,?,?,?)").run(
        runId,
        project,
        startedAt,
        provider,
        JSON.stringify(models),
        VERSION,
        SCHEMA_VERSION,
      );
      return new CTrace(db, runId);
    } catch (err) {
      // DDL/insert failed after the connection opened; close it so we don't
      // leak a file handle/lock on the way out.
      db.close();
      throw err;
    }
  }

  /**
   * Open an existing `.ctrace` read/write, accepting ANY schema version this
   * build understands. A v1 file (Python-written, pre-attribution) is read
   * as-is and never migrated — a debugger must not rewrite the evidence it
   * inspects; its missing agent/step/provider columns surface as null. Only a
   * file whose version is NEWER than supported is rejected, with a clear error.
   * Mirrors Python `open`.
   */
  static open(path: string): CTrace {
    const db = new DatabaseSync(path);
    try {
      db.exec("PRAGMA foreign_keys = ON");
      const r = db
        .prepare("SELECT id, schema_version FROM run LIMIT 1")
        .get() as { id: string; schema_version: number } | undefined;
      if (r === undefined) {
        throw new Error(`${path}: not a ctrace file (no run row)`);
      }
      if (r.schema_version > SCHEMA_VERSION) {
        throw new Error(
          `${path}: schema version ${r.schema_version} is newer than supported ` +
            `${SCHEMA_VERSION} — upgrade ctxdiff to read this file`,
        );
      }
      if (r.schema_version < 1) {
        throw new Error(
          `${path}: schema version ${r.schema_version} is not a recognized ctrace version`,
        );
      }
      return new CTrace(db, r.id);
    } catch (err) {
      db.close();
      throw err;
    }
  }

  // --- writing -------------------------------------------------------------

  /**
   * Persist one call and its ordered blocks in a single transaction. Each block
   * is upserted by content_hash (stored once, ignored if already present — the
   * dedup mechanism); each membership is written to call_block with its position
   * and label. `agent`/`step`/`provider` are the v2 attribution fields
   * (nullable). Returns the new call id. Mirrors Python `record_call`, including
   * the per-call model roll-up onto the run.
   */
  recordCall(args: {
    seq: number;
    params: Record<string, unknown>;
    usage: Record<string, unknown> | null;
    latencyMs: number | null;
    error: string | null;
    callBlocks: CallBlock[];
    agent?: string | null;
    step?: string | null;
    provider?: string | null;
  }): string {
    const callId = uuidHex();
    const {
      seq,
      params,
      usage,
      latencyMs,
      error,
      callBlocks,
      agent = null,
      step = null,
      provider = null,
    } = args;

    // All-or-nothing transaction, matching Python's `with self._conn:`.
    this.db.exec("BEGIN");
    try {
      if (this.hasV2Cols) {
        this.db
          .prepare("INSERT INTO call VALUES (?,?,?,?,?,?,?,?,?,?)")
          .run(
            callId,
            this.runId,
            seq,
            JSON.stringify(params),
            usage !== null ? JSON.stringify(usage) : null,
            latencyMs,
            error,
            agent,
            step,
            provider,
          );
      } else {
        this.db
          .prepare("INSERT INTO call VALUES (?,?,?,?,?,?,?)")
          .run(
            callId,
            this.runId,
            seq,
            JSON.stringify(params),
            usage !== null ? JSON.stringify(usage) : null,
            latencyMs,
            error,
          );
      }
      const insBlock = this.db.prepare(
        "INSERT OR IGNORE INTO block VALUES (?,?,?,?,?,?)",
      );
      const insCallBlock = this.db.prepare(
        "INSERT INTO call_block VALUES (?,?,?,?,?)",
      );
      for (const cb of callBlocks) {
        const b = cb.block;
        // INSERT OR IGNORE: first writer of a hash wins; repeats are no-ops —
        // exactly content-addressed dedup.
        insBlock.run(
          b.contentHash,
          b.role,
          b.kind,
          b.text,
          b.tokenCount,
          b.tokenMethod,
        );
        insCallBlock.run(
          callId,
          b.contentHash,
          cb.position,
          cb.label,
          cb.labelSource,
        );
      }
      this.db.exec("COMMIT");
    } catch (err) {
      this.db.exec("ROLLBACK");
      throw err;
    }

    // Roll this call's model up onto the run — `model` covers openai/anthropic/
    // gemini; `modelId` covers bedrock's Converse shape.
    const model =
      (params["model"] as string | undefined) ??
      (params["modelId"] as string | undefined) ??
      null;
    this.noteModel(model);
    return callId;
  }

  /**
   * Append `model` to the run's `models` list the first time it's seen,
   * preserving first-seen order and deduping repeats; ignores null/empty so a
   * call with no model never pollutes the list with a blank entry. The JSON is
   * re-serialized and the run row UPDATEd only on an actual new model, not every
   * call. Mirrors Python `note_model`.
   */
  noteModel(model: string | null | undefined): void {
    if (!model || this.modelsSeen.has(model)) return;
    this.modelsSeen.add(model);
    this.models.push(model);
    this.db
      .prepare("UPDATE run SET models = ? WHERE id = ?")
      .run(JSON.stringify(this.models), this.runId);
  }

  // --- reading -------------------------------------------------------------

  /** Return the run row as a Run, decoding the models JSON array. */
  getRun(): Run {
    const r = this.db
      .prepare(
        "SELECT id, project, started_at, provider, models, ctxdiff_version " +
          "FROM run WHERE id = ?",
      )
      .get(this.runId) as {
      id: string;
      project: string;
      started_at: string;
      provider: string;
      models: string;
      ctxdiff_version: string;
    };
    return {
      id: r.id,
      project: r.project,
      startedAt: r.started_at,
      provider: r.provider,
      models: JSON.parse(r.models) as string[],
      ctxdiffVersion: r.ctxdiff_version,
    };
  }

  /**
   * Return all calls for this run ordered by turn sequence. Selects the v2
   * attribution columns only when they exist (a v1 file lacks them); for a v1
   * file those three fields surface as null on every Call. Mirrors Python
   * `get_calls`.
   */
  getCalls(): Call[] {
    const base = "id, run_id, seq, params, usage, latency_ms, error";
    if (this.hasV2Cols) {
      const rows = this.db
        .prepare(
          `SELECT ${base}, agent, step, provider FROM call WHERE run_id = ? ORDER BY seq`,
        )
        .all(this.runId) as Record<string, unknown>[];
      return rows.map((r) => this.rowToCall(r, true));
    }
    const rows = this.db
      .prepare(`SELECT ${base} FROM call WHERE run_id = ? ORDER BY seq`)
      .all(this.runId) as Record<string, unknown>[];
    return rows.map((r) => this.rowToCall(r, false));
  }

  private rowToCall(r: Record<string, unknown>, v2: boolean): Call {
    return {
      id: r.id as string,
      runId: r.run_id as string,
      seq: r.seq as number,
      params: JSON.parse(r.params as string) as Record<string, unknown>,
      usage:
        r.usage !== null && r.usage !== undefined
          ? (JSON.parse(r.usage as string) as Record<string, unknown>)
          : null,
      latencyMs: (r.latency_ms as number | null) ?? null,
      error: (r.error as string | null) ?? null,
      agent: v2 ? ((r.agent as string | null) ?? null) : null,
      step: v2 ? ((r.step as string | null) ?? null) : null,
      provider: v2 ? ((r.provider as string | null) ?? null) : null,
    };
  }

  /**
   * Reconstruct one call's blocks in position order by joining call_block to
   * block. Mirrors Python `get_call_blocks`.
   */
  getCallBlocks(callId: string): CallBlock[] {
    const rows = this.db
      .prepare(
        "SELECT cb.position, cb.label, cb.label_source, " +
          "b.content_hash, b.role, b.kind, b.text, b.token_count, b.token_method " +
          "FROM call_block cb JOIN block b ON b.content_hash = cb.block_id " +
          "WHERE cb.call_id = ? ORDER BY cb.position",
      )
      .all(callId) as {
      position: number;
      label: string;
      label_source: string;
      content_hash: string;
      role: string;
      kind: string;
      text: string;
      token_count: number;
      token_method: string;
    }[];
    return rows.map((r) => {
      const block: Block = {
        contentHash: r.content_hash,
        role: r.role,
        kind: r.kind,
        text: r.text,
        tokenCount: r.token_count,
        tokenMethod: r.token_method,
      };
      return {
        block,
        position: r.position,
        label: r.label,
        labelSource: r.label_source,
      };
    });
  }

  /** Close the underlying SQLite connection. */
  close(): void {
    this.db.close();
  }
}
