/**
 * Unit tests for `src/selectors.ts` — the session/agent resolution layer behind
 * `--project`/`--session`/`--agent`/`--turn`.
 *
 * Four things are pinned here, all of which the CLI's byte-identity with the
 * Python SDK depends on: the `NAME:TURN` selector grammar, LOCAL-timezone
 * timestamp rendering (driven by `TZ`, including a DST-sensitive zone), the
 * defaulting/ambiguity rules and the exact error text they produce, and the
 * cross-diff scope header. Everything here is pure — no store is opened.
 */
import { describe, it, expect, afterEach } from "vitest";
import {
  agentPicker,
  agentsText,
  chooseSession,
  diffScopeLine,
  distinctAgentNames,
  formatLocal,
  parseSelector,
  requireAgent,
  requireSingleAgent,
  SelectionError,
  sessionLine,
  shortId,
} from "../src/selectors.js";
import type { Call, Session } from "../src/models.js";

/** A `Session` summary with sensible defaults, so each test states only the
 * fields it actually cares about. */
function session(over: Partial<Session> = {}): Session {
  return {
    id: "0123456789abcdef0123456789abcdef",
    project: "demo",
    startedAt: "2026-07-20T09:15:00+00:00",
    provider: "openai",
    models: ["gpt-4o"],
    agents: [],
    turnCount: 3,
    ...over,
  };
}

/** A `Call` carrying only what the selectors read: seq and agent. */
function call(seq: number, agent: string | null): Call {
  return {
    id: `c${seq}`, runId: "r", seq, params: {}, usage: null,
    latencyMs: null, error: null, agent, step: null, provider: null,
  };
}

describe("parseSelector", () => {
  it("splits a trailing :N into name and turn", () => {
    expect(parseSelector("researcher:8")).toEqual({
      name: "researcher", turn: { value: 8, text: "8" },
    });
    expect(parseSelector("4f3a2b1c9d8e:12")).toEqual({
      name: "4f3a2b1c9d8e", turn: { value: 12, text: "12" },
    });
  });

  it("carries the turn's text the way Python's int() would echo it", () => {
    // Leading zeros normalize away, and a turn larger than a double can hold
    // keeps every digit — `Number` would echo 1e+21, which nobody typed.
    expect(parseSelector("researcher:008").turn).toEqual({ value: 8, text: "8" });
    expect(parseSelector("researcher:1000000000000000000000").turn?.text).toBe(
      "1000000000000000000000",
    );
  });

  it("leaves a bare name alone", () => {
    expect(parseSelector("researcher")).toEqual({ name: "researcher", turn: null });
  });

  it("splits on the LAST colon, so a name may contain one", () => {
    expect(parseSelector("tools:web:3")).toEqual({
      name: "tools:web", turn: { value: 3, text: "3" },
    });
    expect(parseSelector("tools:web")).toEqual({ name: "tools:web", turn: null });
  });

  it("rejects a non-ASCII-digit suffix rather than guessing", () => {
    // Python's str.isdigit() accepts these; the selector grammar deliberately
    // does not, so both SDKs parse the same strings the same way.
    expect(parseSelector("agent:٨")).toEqual({ name: "agent:٨", turn: null });
    expect(parseSelector("agent:²")).toEqual({ name: "agent:²", turn: null });
    expect(parseSelector("agent:-1")).toEqual({ name: "agent:-1", turn: null });
    expect(parseSelector("agent:")).toEqual({ name: "agent:", turn: null });
    expect(parseSelector(":8")).toEqual({ name: ":8", turn: null });
  });
});

describe("shortId / agentsText", () => {
  it("shortens a session id to 12 chars", () => {
    expect(shortId("0123456789abcdef0123456789abcdef")).toBe("0123456789ab");
  });

  it("renders an empty agent list as a dash", () => {
    expect(agentsText([])).toBe("-");
    expect(agentsText(["a", "b"])).toBe("a, b");
  });
});

describe("formatLocal", () => {
  const originalTz = process.env.TZ;
  afterEach(() => {
    if (originalTz === undefined) delete process.env.TZ;
    else process.env.TZ = originalTz;
  });

  it("renders a stored UTC timestamp in the local zone with its offset", () => {
    process.env.TZ = "Asia/Dubai";
    expect(formatLocal("2026-07-20T09:15:00+00:00")).toBe("2026-07-20 13:15:00 +04:00");
    process.env.TZ = "UTC";
    expect(formatLocal("2026-07-20T09:15:00+00:00")).toBe("2026-07-20 09:15:00 +00:00");
  });

  it("uses the offset in effect AT THAT INSTANT, not today's", () => {
    // America/New_York is UTC-4 in July and UTC-5 in January: a fixed-offset
    // renderer would get one of these two wrong.
    process.env.TZ = "America/New_York";
    expect(formatLocal("2026-07-20T09:15:00Z")).toBe("2026-07-20 05:15:00 -04:00");
    expect(formatLocal("2026-01-20T09:15:00Z")).toBe("2026-01-20 04:15:00 -05:00");
  });

  it("handles a half-hour zone", () => {
    process.env.TZ = "Asia/Kolkata";
    expect(formatLocal("2026-07-20T09:15:00Z")).toBe("2026-07-20 14:45:00 +05:30");
  });

  it("truncates a sub-minute (LMT) offset toward zero", () => {
    // Pre-standard-time zones carry offsets with SECONDS in them:
    // America/New_York was -04:56:02 until 1883, America/Los_Angeles
    // -07:52:58. Python's twin of this line has to truncate toward zero to
    // agree with `Date.getTimezoneOffset()`; flooring would say -04:57.
    process.env.TZ = "America/New_York";
    expect(formatLocal("1800-06-01T12:00:00Z")).toBe("1800-06-01 07:03:58 -04:56");
    process.env.TZ = "America/Los_Angeles";
    expect(formatLocal("1800-06-01T12:00:00Z")).toBe("1800-06-01 04:07:02 -07:52");
  });

  it("echoes a timestamp local time cannot represent in BOTH SDKs", () => {
    // Python's datetime tops out at year 9999 and bottoms out at year 1, so the
    // local shift past either edge raises OverflowError there and the raw text
    // is echoed. `Date` would happily render `10000-01-01`/`0000-12-31` — a
    // year the other CLI can never print — so the same edge is enforced here.
    process.env.TZ = "Asia/Dubai"; // east of UTC: 9999-12-31T23:59Z -> 10000
    expect(formatLocal("9999-12-31T23:59:59+00:00")).toBe("9999-12-31T23:59:59+00:00");
    process.env.TZ = "America/Los_Angeles"; // west: 0001-01-01T00:00Z -> 0000
    expect(formatLocal("0001-01-01T00:00:00+00:00")).toBe("0001-01-01T00:00:00+00:00");
    // Still rendered when the shift stays inside the range.
    process.env.TZ = "America/Los_Angeles";
    expect(formatLocal("9999-12-31T23:59:59+00:00")).toBe("9999-12-31 15:59:59 -08:00");
  });

  it("renders the ISO spellings Python's fromisoformat accepts", () => {
    // Reachable from a foreign or hand-edited database: a two-digit offset and
    // the ISO BASIC form both parse in Python, so echoing them raw here would
    // put a different string in one CLI's listing than the other's.
    process.env.TZ = "Asia/Dubai";
    expect(formatLocal("2026-07-04T10:00:00+05")).toBe("2026-07-04 09:00:00 +04:00");
    expect(formatLocal("20260704T100000Z")).toBe("2026-07-04 14:00:00 +04:00");
    expect(formatLocal("20260704T100000+0530")).toBe("2026-07-04 08:30:00 +04:00");
  });

  it("treats a naive stored value as UTC", () => {
    process.env.TZ = "UTC";
    expect(formatLocal("2026-07-20T09:15:00")).toBe("2026-07-20 09:15:00 +00:00");
  });

  it("degrades instead of throwing on empty or unparseable input", () => {
    expect(formatLocal("")).toBe("-");
    expect(formatLocal("   ")).toBe("-");
    expect(formatLocal("not a timestamp")).toBe("not a timestamp");
  });
});

describe("chooseSession", () => {
  const a = session({ id: "aaaa1111bbbb2222", turnCount: 3, agents: ["researcher"] });
  const b = session({ id: "cccc3333dddd4444", turnCount: 5 });

  it("needs no flag when the project holds one session", () => {
    expect(chooseSession([a], null, a.id)).toBe(a.id);
  });

  it("refuses to guess between several sessions, listing them all", () => {
    let err: unknown;
    try {
      chooseSession([a, b], null, b.id);
    } catch (e) {
      err = e;
    }
    expect(err).toBeInstanceOf(SelectionError);
    const msg = (err as Error).message;
    expect(msg.split("\n")[0]).toBe(
      "ctxdiff: this project holds 2 sessions — pass --session to pick one:",
    );
    expect(msg).toContain("  " + sessionLine(a));
    expect(msg).toContain("  " + sessionLine(b));
  });

  it("accepts an exact id and any unambiguous prefix", () => {
    expect(chooseSession([a, b], a.id, b.id)).toBe(a.id);
    expect(chooseSession([a, b], "aaaa", b.id)).toBe(a.id);
  });

  it("reports an unknown id with the listing", () => {
    expect(() => chooseSession([a, b], "zzzz", b.id)).toThrow(
      /no session 'zzzz' in this project — available sessions:/,
    );
  });

  it("reports an ambiguous prefix with only the matches", () => {
    const c = session({ id: "aaaa9999eeee0000" });
    let msg = "";
    try {
      chooseSession([a, b, c], "aaaa", b.id);
    } catch (e) {
      msg = (e as Error).message;
    }
    expect(msg.split("\n")[0]).toBe("ctxdiff: session 'aaaa' is ambiguous — 2 sessions match:");
    expect(msg).not.toContain(shortId(b.id));
  });
});

describe("agent selection", () => {
  const calls = [call(1, "researcher"), call(2, "writer"), call(3, "researcher"), call(4, null)];

  it("collects distinct NAMED agents in first-appearance order", () => {
    expect(distinctAgentNames(calls)).toEqual(["researcher", "writer"]);
  });

  it("rejects an unknown agent with the real names listed", () => {
    let msg = "";
    try {
      requireAgent("nope", ["researcher", "writer"], "session abc");
    } catch (e) {
      msg = (e as Error).message;
    }
    expect(msg).toBe(
      "ctxdiff: no agent 'nope' in session abc — available agents:\n  researcher\n  writer",
    );
  });

  it("explains rather than listing nothing when a session has no agents", () => {
    expect(agentPicker([])).toBe("  (none — this session's calls carry no agent label)");
    expect(() => requireAgent("x", [], "session abc")).toThrow(
      /carry no agent label/,
    );
  });

  it("requires --agent only when a cross-session diff spans several agents", () => {
    expect(() => requireSingleAgent([], "these sessions")).not.toThrow();
    expect(() => requireSingleAgent(["solo"], "these sessions")).not.toThrow();
    expect(() => requireSingleAgent(["a", "b"], "these sessions")).toThrow(
      "ctxdiff: these sessions hold 2 agents — pass --agent to pick one:\n  a\n  b",
    );
  });
});

describe("diffScopeLine", () => {
  /** A diff side's turn, spelled as the user typed it — see `TurnArg`. */
  const t = (n: number) => ({ value: n, text: String(n) });

  it("is null for an ordinary same-session, same-agent diff", () => {
    expect(
      diffScopeLine(
        { sessionId: "s1", agent: "r", turn: t(7) },
        { sessionId: "s1", agent: "r", turn: t(8) },
      ),
    ).toBeNull();
  });

  it("names both sessions when the sides come from different runs", () => {
    expect(
      diffScopeLine(
        { sessionId: "aaaa1111bbbb2222", agent: "researcher", turn: t(8) },
        { sessionId: "cccc3333dddd4444", agent: "researcher", turn: t(8) },
      ),
    ).toBe("── aaaa1111bbbb · researcher · turn 8  →  cccc3333dddd · researcher · turn 8 ──");
  });

  it("drops the identical session on a cross-agent diff", () => {
    expect(
      diffScopeLine(
        { sessionId: "s1", agent: "researcher", turn: t(1) },
        { sessionId: "s1", agent: "writer", turn: t(2) },
      ),
    ).toBe("── researcher · turn 1  →  writer · turn 2 ──");
  });

  it("omits the agent segment when no agent was selected", () => {
    expect(
      diffScopeLine(
        { sessionId: "aaaa1111bbbb2222", agent: null, turn: t(8) },
        { sessionId: "cccc3333dddd4444", agent: null, turn: t(8) },
      ),
    ).toBe("── aaaa1111bbbb · turn 8  →  cccc3333dddd · turn 8 ──");
  });

  it("echoes a turn too large for a double with the digits typed", () => {
    expect(
      diffScopeLine(
        { sessionId: "s1", agent: "r", turn: { value: 1e21, text: "1000000000000000000000" } },
        { sessionId: "s2", agent: "r", turn: t(8) },
      ),
    ).toContain("turn 1000000000000000000000");
  });
});
