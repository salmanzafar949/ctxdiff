/**
 * Shared `.ctrace` fixture builders for the analyzer parity tests. Each fixture
 * is written straight through the CTrace store (deterministic, no network) and
 * exercises a specific analyzer path. Blocks are hashed/counted/labeled exactly
 * like the recorder, so fixtures look like real captures — and the SAME files
 * feed the cross-language comparison against the Python CLI.
 */
import { CTrace } from "../../src/store/ctrace.js";
import { buildBlock } from "../../src/capture/recorder.js";
import { imageRawBlock } from "../../src/images.js";
import { contentHash, basicLabel, type CallBlock } from "../../src/models.js";
import { countTokens } from "../../src/tokenize.js";
import { join } from "node:path";

type Row = [string, string, string]; // [role, kind, text]

/** Build a CallBlock list from [role, kind, text] tuples, mirroring the recorder.
 * `tags` is the (label, needle) list `tracer.tag()` registers: a needle found
 * inside a block overrides its role-based label and marks the membership
 * `tagged`, which is the ONLY thing the eviction detector looks at. */
function blocks(
  rows: Row[],
  provider = "openai",
  tags: [string, string][] = [],
): CallBlock[] {
  return rows.map(([role, kind, text], position) => {
    const [tokenCount, tokenMethod] = countTokens(text, provider);
    const [label, labelSource] = basicLabel(role, kind, text, tags);
    return {
      block: { contentHash: contentHash(role, kind, text), role, kind, text, tokenCount, tokenMethod },
      position,
      label,
      labelSource,
    };
  });
}

/** A multi-turn trace: growing history (cache-friendly), a turn that modifies
 * the system block + a user block (diff), unused tool schemas (bloat), usage. */
export function writeMultiturn(dir: string): string {
  const path = join(dir, "multiturn.ctrace");
  const ct = CTrace.create(path, "research", "openai", "", "2026-07-01T00:00:00Z");
  const sys: Row = ["system", "message", "You are a helpful research assistant. Be concise and cite sources."];
  const toolA: Row = ["system", "tool_schema", JSON.stringify({ type: "function", function: { name: "web_search" } })];
  const toolB: Row = ["system", "tool_schema", JSON.stringify({ type: "function", function: { name: "calculator" } })];

  ct.recordCall({
    seq: 1, params: { model: "gpt-4o", temperature: 0.2 },
    usage: { prompt_tokens: 40, completion_tokens: 12, total_tokens: 52 },
    latencyMs: 120, error: null,
    callBlocks: blocks([sys, toolA, toolB, ["user", "message", "What is the capital of France?"]]),
    provider: "openai",
  });
  ct.recordCall({
    seq: 2, params: { model: "gpt-4o", temperature: 0.2 },
    usage: { prompt_tokens: 70, completion_tokens: 20, total_tokens: 90 },
    latencyMs: 140, error: null,
    callBlocks: blocks([
      sys, toolA, toolB,
      ["user", "message", "What is the capital of France?"],
      ["assistant", "message", "The capital of France is Paris."],
      ["user", "message", "And its population?"],
    ]),
    provider: "openai",
  });
  ct.recordCall({
    seq: 3, params: { model: "gpt-4o", temperature: 0.2 },
    usage: { prompt_tokens: 95, completion_tokens: 25, total_tokens: 120 },
    latencyMs: 150, error: null,
    callBlocks: blocks([
      ["system", "message", "You are a helpful research assistant. Be concise and cite sources. Always answer in English."],
      toolA, toolB,
      ["user", "message", "What is the capital of France?"],
      ["assistant", "message", "The capital of France is Paris."],
      ["user", "message", "And its population, roughly?"],
    ]),
    provider: "openai",
  });
  ct.close();
  return path;
}

/** A two-agent trace: researcher + writer interleaved, so cache grouping must
 * not count the cross-agent hand-off as a break. */
export function writeMultiagent(dir: string): string {
  const path = join(dir, "multiagent.ctrace");
  const ct = CTrace.create(path, "pipeline", "openai", "", "2026-07-02T00:00:00Z");
  const rSys: Row = ["system", "message", "You are the RESEARCHER. Gather facts."];
  const wSys: Row = ["system", "message", "You are the WRITER. Compose prose."];
  ct.recordCall({ seq: 1, params: { model: "gpt-4o" }, usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }, latencyMs: 100, error: null, callBlocks: blocks([rSys, ["user", "message", "Find facts about Mars."]]), agent: "researcher", step: "gather", provider: "openai" });
  ct.recordCall({ seq: 2, params: { model: "gpt-4o" }, usage: { prompt_tokens: 12, completion_tokens: 6, total_tokens: 18 }, latencyMs: 110, error: null, callBlocks: blocks([wSys, ["user", "message", "Write an intro about Mars."]]), agent: "writer", step: "compose", provider: "openai" });
  ct.recordCall({ seq: 3, params: { model: "gpt-4o" }, usage: { prompt_tokens: 20, completion_tokens: 8, total_tokens: 28 }, latencyMs: 120, error: null, callBlocks: blocks([rSys, ["user", "message", "Find facts about Mars."], ["assistant", "message", "Mars is the fourth planet."], ["user", "message", "More detail?"]]), agent: "researcher", step: "gather", provider: "openai" });
  ct.recordCall({ seq: 4, params: { model: "gpt-4o" }, usage: { prompt_tokens: 22, completion_tokens: 9, total_tokens: 31 }, latencyMs: 130, error: null, callBlocks: blocks([wSys, ["user", "message", "Write an intro about Mars."], ["assistant", "message", "Mars, the red planet, ..."], ["user", "message", "Expand it."]]), agent: "writer", step: "compose", provider: "openai" });
  ct.close();
  return path;
}

/** A dynamic-timestamp trace: an early system block carries a changing timestamp,
 * breaking the prefix identically every turn (triggers the fix hint). */
export function writeDynamic(dir: string): string {
  const path = join(dir, "dynamic.ctrace");
  const ct = CTrace.create(path, "dynamic", "openai", "", "2026-07-03T00:00:00Z");
  for (let t = 1; t <= 3; t++) {
    ct.recordCall({
      seq: t, params: { model: "gpt-4o" },
      usage: { prompt_tokens: 30 + t, completion_tokens: 5, total_tokens: 35 + t },
      latencyMs: 100, error: null,
      callBlocks: blocks([
        ["system", "message", `Current time is 2026-07-03T10:0${t}:00Z. You are an assistant.`],
        ["user", "message", `Question number ${t}?`],
      ]),
      provider: "openai",
    });
  }
  ct.close();
  return path;
}

/** A trace whose second turn ADDS a user block carrying zero-width and bidi
 * format characters (ZWSP, LRM/RLM), so the diff renders that block's snippet
 * through `pyRepr` — exercising non-printable escaping against Python. Built
 * from code points (never literal control chars). */
export function writeBidi(dir: string): string {
  const cp = String.fromCodePoint;
  const bidiText = `Reply${cp(0x200b)}now ${cp(0x200e)}left${cp(0x200f)}right ok`;
  const path = join(dir, "bidi.ctrace");
  const ct = CTrace.create(path, "bidi", "openai", "", "2026-07-05T00:00:00Z");
  const sys: Row = ["system", "message", "You are an assistant."];
  const user1: Row = ["user", "message", "First question?"];
  ct.recordCall({ seq: 1, params: { model: "gpt-4o" }, usage: { prompt_tokens: 8, completion_tokens: 2, total_tokens: 10 }, latencyMs: 100, error: null, callBlocks: blocks([sys, user1]), provider: "openai" });
  ct.recordCall({ seq: 2, params: { model: "gpt-4o" }, usage: { prompt_tokens: 14, completion_tokens: 3, total_tokens: 17 }, latencyMs: 110, error: null, callBlocks: blocks([sys, user1, ["assistant", "message", "First answer."], ["user", "message", bidiText]]), provider: "openai" });
  ct.close();
  return path;
}

/**
 * A PROJECT trace: one file, TWO sessions ("good" then "bad"), each with the
 * same two agents on the same turn numbers, and one block whose text differs
 * between the runs. This is the regression shape the session CLI exists for —
 * `diff --session GOOD:3 --session BAD:3 --agent researcher` must show exactly
 * that one changed block — and it doubles as the ambiguity fixture, since a
 * two-session project is precisely what makes a bare `--session`-less command
 * refuse to guess.
 *
 * Timestamps are fixed and UTC so the local-time column is a pure function of
 * `TZ`, which is what lets the cross-language conformance suite pin a timezone
 * and compare bytes.
 */
export function writeProject(dir: string): string {
  const path = join(dir, "project.ctrace");
  const rSys: Row = ["system", "message", "You are the RESEARCHER. Gather facts."];
  const wSys: Row = ["system", "message", "You are the WRITER. Compose prose."];
  for (const [startedAt, tail, bump] of [
    ["2026-07-20T09:15:00+00:00", "good", 0],
    ["2026-07-21T18:42:30+00:00", "bad", 1],
  ] as [string, string, number][]) {
    const ct = CTrace.openOrCreateSession(path, "pipeline", "openai", "", startedAt);
    ct.recordCall({
      seq: 1, params: { model: "gpt-4o" },
      usage: { prompt_tokens: 100 + bump, completion_tokens: 20, total_tokens: 120 + bump },
      latencyMs: 100, error: null,
      callBlocks: blocks([rSys, ["user", "message", "Find facts about Mars."]]),
      agent: "researcher", step: "gather", provider: "openai",
    });
    ct.recordCall({
      seq: 2, params: { model: "gpt-4o" },
      usage: { prompt_tokens: 40, completion_tokens: 8, total_tokens: 48 },
      latencyMs: 110, error: null,
      callBlocks: blocks([wSys, ["user", "message", "Write an intro about Mars."]]),
      agent: "writer", step: "compose", provider: "openai",
    });
    ct.recordCall({
      seq: 3, params: { model: "gpt-4o" }, usage: null, latencyMs: 120, error: null,
      callBlocks: blocks([
        rSys,
        ["user", "message", "Find facts about Mars."],
        ["assistant", "message", "Mars is the fourth planet."],
        ["user", "message", `More detail (${tail})?`],
      ]),
      agent: "researcher", step: "gather", provider: "openai",
    });
    ct.close();
  }
  return path;
}

/**
 * A trace whose sessions carry AWKWARD `started_at` values — every spelling and
 * edge the local-time column has to survive, in one file:
 *
 * - the two `datetime` boundaries (year 9999 and year 1), which OVERFLOW when a
 *   local offset shifts them past MINYEAR/MAXYEAR and so must be echoed raw by
 *   both CLIs rather than rendered as a year one of them cannot represent;
 * - an 1800 instant, when most zones still ran on LOCAL MEAN TIME with SECONDS
 *   in the offset (`America/New_York` was -04:56:02) — the case where flooring
 *   and truncating whole minutes disagree;
 * - an hour-only offset (`+05`) and the ISO BASIC form (`20260704T100000Z`),
 *   both of which Python's `fromisoformat` accepts;
 * - a value neither side can parse, which must be echoed unchanged.
 *
 * Nothing ctxdiff writes looks like these — a foreign or hand-edited database
 * does, and that is exactly when the two SDKs must not disagree.
 */
export function writeEdgeTimestamps(dir: string): string {
  const path = join(dir, "edge.ctrace");
  for (const startedAt of [
    "9999-12-31T23:59:59+00:00",
    "0001-01-01T00:00:00+00:00",
    "1800-06-01T12:00:00Z",
    "2026-07-04T10:00:00+05",
    "20260704T100000Z",
    "not a timestamp",
  ]) {
    const ct = CTrace.openOrCreateSession(path, "edge", "openai", "", startedAt);
    ct.recordCall({
      seq: 1, params: { model: "gpt-4o" }, usage: null, latencyMs: 10, error: null,
      callBlocks: blocks([["user", "message", "hi"]]), agent: "solo", provider: "openai",
    });
    ct.close();
  }
  return path;
}

/**
 * A trace whose every turn carries an image of UNKNOWABLE cost — a remote URL,
 * which ctxdiff must never fetch — so each call's stored block total is a FLOOR
 * rather than a measurement.
 *
 * This is the false-pass shape in the flesh: the provider bills 800 then 1,600
 * prompt tokens (recorded verbatim in `usage`), while the blocks sum to 4 and 8,
 * because the two pictures are stored as zero-token estimates. A `check
 * --max-context 500` that compared those sums to the budget would print PASS and
 * exit 0 over a run that used three times the budget. The block is built through
 * the REAL `imageRawBlock`/`buildBlock` path, so the fixture is evidence about
 * capture rather than about a hand-written zero.
 */
export function writeUnmeasured(dir: string): string {
  const path = join(dir, "unmeasured.ctrace");
  const ct = CTrace.create(path, "vision", "openai", "", "2026-07-06T00:00:00Z");
  const remoteImage = (n: number): CallBlock => {
    const rb = imageRawBlock(
      "user",
      { type: "image_url", image_url: { url: `https://cdn.example.com/shot-${n}.png` } },
      "openai",
    )!;
    const block = buildBlock(rb, "openai");
    return { block, position: 0, label: "user", labelSource: "heuristic" };
  };
  const reposition = (list: CallBlock[]): CallBlock[] =>
    list.map((cb, position) => ({ ...cb, position }));
  const turn1 = reposition([...blocks([["user", "message", "look"]]), remoteImage(1)]);
  const turn2 = reposition([
    ...turn1,
    ...blocks([["assistant", "message", "ok"]]),
    remoteImage(2),
  ]);
  ct.recordCall({
    seq: 1, params: { model: "gpt-4o" },
    usage: { prompt_tokens: 800, completion_tokens: 10, total_tokens: 810 },
    latencyMs: 100, error: null, callBlocks: turn1, provider: "openai",
  });
  ct.recordCall({
    seq: 2, params: { model: "gpt-4o" },
    usage: { prompt_tokens: 1600, completion_tokens: 12, total_tokens: 1612 },
    latencyMs: 110, error: null, callBlocks: turn2, provider: "openai",
  });
  ct.close();
  return path;
}

/**
 * A FAN-OUT trace: four turns, four agents, one turn each — a supervisor
 * dispatching four workers, which is an ordinary agent topology and not an
 * edge case.
 *
 * Nothing here can be paired: pairing is per-agent by design (a hand-off is not
 * growth and not a cache break), so every adjacent pair is cross-agent and both
 * `--require-stable-prefix` and `--max-growth` have nothing to measure. The
 * fixture exists to pin what they SAY about that — reporting "fewer than 2
 * turns" over a four-turn run is how an assertion that verified nothing goes
 * unnoticed.
 */
export function writeFanout(dir: string): string {
  const path = join(dir, "fanout.ctrace");
  const ct = CTrace.create(path, "fanout", "openai", "", "2026-07-07T00:00:00Z");
  const agents = ["scout", "planner", "coder", "critic"];
  agents.forEach((agent, i) => {
    ct.recordCall({
      seq: i + 1, params: { model: "gpt-4o" },
      usage: { prompt_tokens: 20 + i, completion_tokens: 4, total_tokens: 24 + i },
      latencyMs: 100, error: null,
      callBlocks: blocks([
        ["system", "message", `You are the ${agent.toUpperCase()}.`],
        ["user", "message", `Task for the ${agent}.`],
      ]),
      agent, step: "work", provider: "openai",
    });
  });
  ct.close();
  return path;
}

/**
 * A TAGGED trace: the fixture behind the eviction detector and its
 * counter-examples, in one interleaved multi-agent file.
 *
 * The researcher tags a retrieved passage at turn 1; the writer's turn 2 does
 * not carry it (a HAND-OFF, which must never read as an eviction); the
 * researcher still has it at turn 3; and at turn 5 it is gone for good — the one
 * real eviction, and the only line the report may print. The writer's own
 * timeline has nothing tagged at all, so `--agent writer` exercises the
 * "structurally vacuous pass" branch that must not tick as if it had measured
 * something.
 */
export function writeTagged(dir: string): string {
  const path = join(dir, "tagged.ctrace");
  const ct = CTrace.create(path, "tagged", "openai", "", "2026-07-08T00:00:00Z");
  const rSys: Row = ["system", "message", "You are the RESEARCHER. Gather facts."];
  const wSys: Row = ["system", "message", "You are the WRITER. Compose prose."];
  const passage: Row = [
    "user",
    "message",
    "Retrieved passage: Mars has two moons, Phobos and Deimos.",
  ];
  // The (label, needle) pair `tracer.tag("rag", ...)` registers: a needle found
  // inside a block overrides its role-based label and marks it `tagged`.
  const tags: [string, string][] = [["rag", "Retrieved passage"]];
  ct.recordCall({
    seq: 1, params: { model: "gpt-4o" },
    usage: { prompt_tokens: 40, completion_tokens: 8, total_tokens: 48 },
    latencyMs: 100, error: null,
    callBlocks: blocks([rSys, passage], "openai", tags),
    agent: "researcher", step: "gather", provider: "openai",
  });
  ct.recordCall({
    seq: 2, params: { model: "gpt-4o" },
    usage: { prompt_tokens: 22, completion_tokens: 6, total_tokens: 28 },
    latencyMs: 110, error: null,
    callBlocks: blocks([wSys, ["user", "message", "Write an intro about Mars."]]),
    agent: "writer", step: "compose", provider: "openai",
  });
  ct.recordCall({
    seq: 3, params: { model: "gpt-4o" },
    usage: { prompt_tokens: 58, completion_tokens: 9, total_tokens: 67 },
    latencyMs: 120, error: null,
    callBlocks: blocks([
      rSys, passage,
      ["assistant", "message", "Mars is the fourth planet from the Sun."],
      ["user", "message", "Anything about its moons?"],
    ], "openai", tags),
    agent: "researcher", step: "gather", provider: "openai",
  });
  ct.recordCall({
    seq: 4, params: { model: "gpt-4o" },
    usage: { prompt_tokens: 34, completion_tokens: 7, total_tokens: 41 },
    latencyMs: 130, error: null,
    callBlocks: blocks([
      wSys,
      ["user", "message", "Write an intro about Mars."],
      ["assistant", "message", "Mars, the red planet, has fascinated observers."],
    ]),
    agent: "writer", step: "compose", provider: "openai",
  });
  ct.recordCall({
    seq: 5, params: { model: "gpt-4o" },
    usage: { prompt_tokens: 52, completion_tokens: 10, total_tokens: 62 },
    latencyMs: 140, error: null,
    callBlocks: blocks([
      rSys,
      ["assistant", "message", "Mars is the fourth planet from the Sun."],
      ["user", "message", "Anything about its moons?"],
      ["assistant", "message", "Phobos and Deimos, both small and irregular."],
    ]),
    agent: "researcher", step: "gather", provider: "openai",
  });
  ct.close();
  return path;
}

export function makeFixtures(dir: string): {
  multiturn: string;
  multiagent: string;
  dynamic: string;
  bidi: string;
  project: string;
  edge: string;
  unmeasured: string;
  fanout: string;
  tagged: string;
} {
  return {
    multiturn: writeMultiturn(dir),
    multiagent: writeMultiagent(dir),
    dynamic: writeDynamic(dir),
    bidi: writeBidi(dir),
    project: writeProject(dir),
    edge: writeEdgeTimestamps(dir),
    unmeasured: writeUnmeasured(dir),
    fanout: writeFanout(dir),
    tagged: writeTagged(dir),
  };
}
