/**
 * Concurrency-correctness tests for the capture path — the JS parity for the
 * Python `test_concurrency.py` tag/step-isolation suite.
 *
 * The bug these guard against: under a `Promise.all` fan-out (the normal shape
 * of a modern agent making parallel tool/model calls) the OLD instance-level
 * tag/step state was read AFTER the call's `await`, by which point every
 * concurrent branch had already run its own `tag()`/`mark()` — so one branch's
 * attribution bled into another and MISLABELED it (confidently wrong, the worst
 * failure mode for a debugger).
 *
 * The fix: tag/step live in an `AsyncLocalStorage` store and the interceptor
 * SNAPSHOTS them synchronously on the calling side, before the async record —
 * so each concurrent call records ITS OWN tag/step. The scoped `tracer.step()`
 * additionally isolates branches that must `await` between labeling and calling.
 *
 * Node is single-threaded (`node:sqlite` writes stay synchronous on the main
 * thread), so — unlike the Python port — there is no cross-thread DB race and no
 * background writer thread; this file only exercises the tag/step interleaving.
 */
import { describe, it, expect, afterEach, vi } from "vitest";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { rmSync } from "node:fs";
import OpenAI from "openai";
import { init } from "../src/trace.js";
import { CTrace } from "../src/store/ctrace.js";

const created: string[] = [];
function tmpTrace(): string {
  const p = join(tmpdir(), `ctxdiff-conc-${randomUUID()}.ctrace`);
  created.push(p);
  return p;
}
afterEach(() => {
  vi.restoreAllMocks();
  for (const p of created.splice(0)) {
    try {
      rmSync(p, { force: true });
    } catch {
      /* ignore */
    }
  }
});

/** The unique text carried in call i's user message; also the tag needle, so the
 * recorded block for call i is labeled with call i's own tag. */
function needle(i: number): string {
  return `needle-${String(i).padStart(3, "0")}`;
}

/** Recover call index i from a stored block's text (see needle()). */
function indexOf(text: string): number | null {
  const m = text.match(/needle-(\d+)/);
  return m ? Number(m[1]) : null;
}

/**
 * A real OpenAI client whose HTTP layer is a canned stub that resolves after a
 * STAGGERED delay, so concurrent branches genuinely interleave between each
 * branch's tag()/mark() and its record — exactly the window the old
 * instance-state bug corrupted. Echoes the request's first user message back so
 * the recorded block text carries the call's needle.
 */
function staggeredClient(): OpenAI {
  const fetchFn = async (_url: unknown, init?: { body?: string }) => {
    const body = init?.body ? JSON.parse(init.body) : {};
    // Jittered delay forces the event loop to interleave the branches.
    await new Promise((r) => setTimeout(r, Math.floor(Math.random() * 12)));
    const model = (body.model as string) ?? "gpt-4o";
    return new Response(
      JSON.stringify({
        id: "cmpl",
        object: "chat.completion",
        model,
        choices: [
          { index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      }),
      { headers: { "content-type": "application/json" } },
    );
  };
  return new OpenAI({ apiKey: "test", fetch: fetchFn as unknown as typeof fetch });
}

/** Map each recorded call to { needleIndex, step, firstBlockLabel } for asserting
 * per-branch attribution. */
function attributionByNeedle(
  ct: CTrace,
): Map<number, { step: string | null; label: string; labelSource: string }> {
  const out = new Map<number, { step: string | null; label: string; labelSource: string }>();
  for (const c of ct.getCalls()) {
    const blocks = ct.getCallBlocks(c.id);
    const i = indexOf(blocks[0].block.text);
    if (i !== null) {
      out.set(i, { step: c.step, label: blocks[0].label, labelSource: blocks[0].labelSource });
    }
  }
  return out;
}

describe("concurrency: Promise.all tag/step isolation", () => {
  // (a) The core regression test. Bare tag()/mark() immediately before each
  // concurrent create(). Under the OLD read-after-await code every branch's
  // attribution bled together (all recorded the LAST branch's step/tag); the
  // snapshot-before-await fix makes each record its own. This FAILS against the
  // old instance-state implementation and passes against the fixed one.
  it("each concurrent call records its OWN tag and step (no bleed)", async () => {
    const n = 25;
    const path = tmpTrace();
    const tracer = init("async-fan", { path });
    const wrapped = tracer.wrap(staggeredClient()) as OpenAI;

    await Promise.all(
      [...Array(n).keys()].map(async (i) => {
        tracer.mark(`step-${String(i).padStart(3, "0")}`);
        tracer.tag(`tag-${String(i).padStart(3, "0")}`, [needle(i)]);
        await wrapped.chat.completions.create({
          model: "gpt-4o",
          messages: [{ role: "user", content: `hello ${needle(i)} world` }],
        });
      }),
    );
    tracer.close();

    const ct = CTrace.open(path);
    expect(ct.getCalls()).toHaveLength(n);
    const attr = attributionByNeedle(ct);
    expect(attr.size).toBe(n);
    for (let i = 0; i < n; i++) {
      const a = attr.get(i)!;
      const suffix = String(i).padStart(3, "0");
      expect(a.step, `call ${i} step`).toBe(`step-${suffix}`);
      expect(a.label, `call ${i} label`).toBe(`tag-${suffix}`);
      expect(a.labelSource).toBe("tagged");
    }
    ct.close();
  });

  // Documents the residual hazard the scoped form exists for: a bare mark()/tag()
  // with an `await` BETWEEN it and the call shares the root store, so a sibling
  // branch's mark() runs in the gap and relabels this branch. This is the JS
  // analog of Python's raw-ThreadPoolExecutor leak caveat.
  it("bare mark() with an await before the call CAN bleed (why step() exists)", async () => {
    const n = 12;
    const path = tmpTrace();
    const tracer = init("bleed-caveat", { path });
    const wrapped = tracer.wrap(staggeredClient()) as OpenAI;

    await Promise.all(
      [...Array(n).keys()].map(async (i) => {
        tracer.mark(`step-${String(i).padStart(3, "0")}`);
        // An await between labeling and calling: a sibling branch's mark() lands
        // in this gap and overwrites the shared root store's step.
        await new Promise((r) => setTimeout(r, 3));
        await wrapped.chat.completions.create({
          model: "gpt-4o",
          messages: [{ role: "user", content: `hi ${needle(i)}` }],
        });
      }),
    );
    tracer.close();

    const ct = CTrace.open(path);
    const attr = attributionByNeedle(ct);
    // At least one branch is mislabeled — the documented reason to use step().
    const mislabeled = [...attr.entries()].filter(
      ([i, a]) => a.step !== `step-${String(i).padStart(3, "0")}`,
    );
    expect(mislabeled.length).toBeGreaterThan(0);
    ct.close();
  });
});

describe("concurrency: scoped tracer.step()", () => {
  // (b) The scoped form fixes the await-in-the-gap hazard, and does not leak into
  // concurrent branches that opt out of it. Even branches wrap their whole call
  // in a step() scope (with an await inside); odd branches use no step at all and
  // must record step=null — never a sibling scope's label.
  it("isolates each branch and does not leak into unscoped concurrent branches", async () => {
    const n = 24;
    const path = tmpTrace();
    const tracer = init("step-scope", { path });
    const wrapped = tracer.wrap(staggeredClient()) as OpenAI;

    await Promise.all(
      [...Array(n).keys()].map((i) => {
        const suffix = String(i).padStart(3, "0");
        const call = async (): Promise<void> => {
          // await BEFORE the call — the exact gap that bleeds without a scope.
          await new Promise((r) => setTimeout(r, 3));
          await wrapped.chat.completions.create({
            model: "gpt-4o",
            messages: [{ role: "user", content: `x ${needle(i)}` }],
          });
        };
        // Even branches scope their phase; odd branches deliberately don't.
        return i % 2 === 0 ? tracer.step(`phase-${suffix}`, call) : call();
      }),
    );
    tracer.close();

    const ct = CTrace.open(path);
    const attr = attributionByNeedle(ct);
    expect(attr.size).toBe(n);
    for (let i = 0; i < n; i++) {
      const a = attr.get(i)!;
      if (i % 2 === 0) {
        expect(a.step, `scoped call ${i}`).toBe(`phase-${String(i).padStart(3, "0")}`);
      } else {
        // No leak from a sibling scope onto an unscoped branch.
        expect(a.step, `unscoped call ${i}`).toBeNull();
      }
    }
    ct.close();
  });

  it("callback form restores the previous step on exit (reentrant/nested)", async () => {
    const path = tmpTrace();
    const tracer = init("step-nest", { path });
    const wrapped = tracer.wrap(staggeredClient()) as OpenAI;

    const call = (i: number): Promise<unknown> =>
      wrapped.chat.completions.create({
        model: "gpt-4o",
        messages: [{ role: "user", content: `n ${needle(i)}` }],
      });

    tracer.mark("outer"); // sticky on the root store
    await call(0); // records "outer"
    await tracer.step("inner", async () => {
      await call(1); // records "inner"
      await tracer.step("deep", async () => {
        await call(2); // records "deep"
      });
      await call(3); // back to "inner"
    });
    await call(4); // root store restored: "outer"
    tracer.close();

    const ct = CTrace.open(path);
    const attr = attributionByNeedle(ct);
    expect(attr.get(0)!.step).toBe("outer");
    expect(attr.get(1)!.step).toBe("inner");
    expect(attr.get(2)!.step).toBe("deep");
    expect(attr.get(3)!.step).toBe("inner");
    expect(attr.get(4)!.step).toBe("outer");
    ct.close();
  });

  it("Disposable form scopes the step and restores it on dispose", async () => {
    const path = tmpTrace();
    const tracer = init("step-dispose", { path });
    const wrapped = tracer.wrap(staggeredClient()) as OpenAI;

    const call = (i: number): Promise<unknown> =>
      wrapped.chat.completions.create({
        model: "gpt-4o",
        messages: [{ role: "user", content: `d ${needle(i)}` }],
      });

    await call(0); // no scope -> null
    {
      const scope = tracer.step("scoped");
      await call(1); // "scoped"
      scope[Symbol.dispose]();
    }
    await call(2); // restored -> null
    tracer.close();

    const ct = CTrace.open(path);
    const attr = attributionByNeedle(ct);
    expect(attr.get(0)!.step).toBeNull();
    expect(attr.get(1)!.step).toBe("scoped");
    expect(attr.get(2)!.step).toBeNull();
    ct.close();
  });
});

describe("concurrency: scoped step() side effects", () => {
  // Regression: the callback form used to wrap `als.run(child, fn)` in a
  // try/catch that re-ran `fn()` on failure. But als.run only throws when `fn`
  // ITSELF throws synchronously, so the catch double-invoked `fn` — running its
  // side effects twice. The catch is gone; als.run's throw now propagates
  // unchanged. FAILS pre-fix (sideEffect runs twice), passes now.
  it("runs fn exactly once on a synchronous throw and propagates its error", async () => {
    const path = tmpTrace();
    const tracer = init("step-throw", { path });
    const wrapped = tracer.wrap(staggeredClient()) as OpenAI;

    let sideEffects = 0;
    const boom = new Error("kaboom");
    expect(() =>
      tracer.step("p", () => {
        sideEffects++;
        throw boom;
      }),
    ).toThrow(boom);
    // The single most important assertion: the side effect ran once, not twice.
    expect(sideEffects).toBe(1);

    // The step is restored after the throw (ALS unwinding): a subsequent call
    // outside any scope records step=null, not the leaked "p".
    await wrapped.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: `after ${needle(0)}` }],
    });
    tracer.close();

    const ct = CTrace.open(path);
    const attr = attributionByNeedle(ct);
    expect(attr.get(0)!.step).toBeNull();
    ct.close();
  });
});

describe("concurrency: sequential behavior is unchanged", () => {
  it("mark() is sticky and tag() is one-shot, exactly as before", async () => {
    const path = tmpTrace();
    const tracer = init("seq", { path });
    const wrapped = tracer.wrap(staggeredClient()) as OpenAI;

    const call = (i: number): Promise<unknown> =>
      wrapped.chat.completions.create({
        model: "gpt-4o",
        messages: [{ role: "user", content: `s ${needle(i)}` }],
      });

    tracer.mark("phase-1");
    tracer.tag("special", [needle(0)]);
    await call(0); // step=phase-1, block label=special (tagged)
    await call(1); // step=phase-1 (sticky), tag consumed -> heuristic label
    tracer.mark(null);
    await call(2); // step=null
    tracer.close();

    const ct = CTrace.open(path);
    const attr = attributionByNeedle(ct);
    expect(attr.get(0)!.step).toBe("phase-1");
    expect(attr.get(0)!.label).toBe("special");
    expect(attr.get(0)!.labelSource).toBe("tagged");
    expect(attr.get(1)!.step).toBe("phase-1");
    expect(attr.get(1)!.labelSource).toBe("heuristic"); // tag was one-shot
    expect(attr.get(2)!.step).toBeNull();
    ct.close();
  });
});

describe("concurrency: fail-open in the ALS plumbing", () => {
  it("a throwing tag()/mark() never breaks the host call", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const path = tmpTrace();
    const tracer = init("failopen-als", { path });
    const wrapped = tracer.wrap(staggeredClient()) as OpenAI;

    // Sabotage the ALS so any store access throws deep in the plumbing.
    const als = (tracer as unknown as { als: { getStore: () => unknown } }).als;
    als.getStore = () => {
      throw new Error("als boom");
    };

    // tag()/mark() swallow the error; the host call still returns intact.
    expect(() => tracer.mark("x")).not.toThrow();
    expect(() => tracer.tag("y", ["z"])).not.toThrow();
    const res = await wrapped.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(res.choices[0].message.content).toBe("ok");
    tracer.close();
  });
});
