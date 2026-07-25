/**
 * A minimal DOM good enough to BOOT the dashboard page and drive its three
 * levels — the only way to test that the exported HTML actually works, since the
 * page's whole behavior lives in an inline `<script>` that no payload assertion
 * can exercise.
 *
 * Deliberately tiny and deliberately strict: it implements exactly the surface
 * the template uses (createElement, getElementById, textContent, appendChild,
 * innerHTML on set, classList, setAttribute, style, hidden/disabled, click
 * listeners, keydown on document) and nothing else, so a template that starts
 * relying on some other DOM API fails loudly here rather than silently in a real
 * browser.
 *
 * `textContent` and `innerHTML` are tracked SEPARATELY, which is what lets the
 * XSS tests assert the real property: trace-derived text must arrive as
 * textContent, never as markup. `text()` walks the tree and returns only what a
 * user would read; `markup()` returns only what was ever assigned as HTML.
 */

/** One element. Attributes, children and the two text channels, nothing more. */
export class El {
  tag: string;
  className = "";
  children: El[] = [];
  attrs: Record<string, string> = {};
  style: Record<string, string> = {};
  listeners: Record<string, ((ev: unknown) => void)[]> = {};
  parent: El | null = null;
  disabled = false;
  hidden = false;
  title = "";
  /** Text assigned via `.textContent` — inert by construction. */
  private ownText = "";
  /** Markup assigned via `.innerHTML` — the channel that COULD be live. */
  htmlAssigned = "";

  constructor(tag: string) {
    this.tag = tag;
  }

  set textContent(value: string) {
    this.ownText = value == null ? "" : String(value);
    this.children = [];
    this.htmlAssigned = "";
  }
  get textContent(): string {
    return this.ownText + this.children.map((c) => c.textContent).join("");
  }

  /** Setting innerHTML clears children, exactly as a browser does. Only `""`
   * (the page's "wipe this host" idiom) and static/numeric markup ever reach
   * this in the template; the tests assert as much. */
  set innerHTML(value: string) {
    this.htmlAssigned = value == null ? "" : String(value);
    this.children = [];
    this.ownText = "";
  }
  get innerHTML(): string {
    return this.htmlAssigned;
  }

  appendChild(child: El): El {
    child.parent = this;
    this.children.push(child);
    return child;
  }
  setAttribute(name: string, value: string): void {
    this.attrs[name] = String(value);
  }
  getAttribute(name: string): string | null {
    return this.attrs[name] ?? null;
  }
  addEventListener(type: string, fn: (ev: unknown) => void): void {
    (this.listeners[type] ??= []).push(fn);
  }
  /** Fire this element's click handlers, as a user clicking it would. */
  click(): void {
    for (const fn of this.listeners.click ?? []) fn({ preventDefault() {} });
  }
  querySelector(sel: string): El | null {
    return this.find((e) => e.tag === sel)[0] ?? null;
  }
  get classList(): { add(c: string): void; contains(c: string): boolean } {
    const self = this;
    return {
      add(c: string) {
        self.className = self.className ? self.className + " " + c : c;
      },
      contains(c: string) {
        return self.className.split(/\s+/).includes(c);
      },
    };
  }

  /** Every descendant (and self) matching `pred`, in document order. */
  find(pred: (e: El) => boolean): El[] {
    const out: El[] = [];
    const walk = (e: El) => {
      if (pred(e)) out.push(e);
      for (const c of e.children) walk(c);
    };
    walk(this);
    return out;
  }
  /** Everything a user would READ under this element — text only, no markup. */
  text(): string {
    return this.textContent;
  }
  /** Every string that was ever assigned as HTML under this element. This is the
   * only channel through which markup can become live, so an XSS test asserts
   * against exactly this. */
  markup(): string {
    return this.find(() => true)
      .map((e) => e.htmlAssigned)
      .join("\n");
  }
}

/** The page's globals: `document`, `window`, and the handful of things the
 * template's boot path touches. */
export interface PageWindow {
  document: {
    documentElement: El;
    title: string;
    getElementById(id: string): El | null;
    createElement(tag: string): El;
    createTextNode(text: string): El;
    addEventListener(type: string, fn: (ev: unknown) => void): void;
    /** Fire a document-level event — how the tests drive the arrow keys. */
    dispatch(type: string, ev: unknown): void;
  };
  window: { matchMedia(q: string): { matches: boolean } };
  /** Every element created with an id, by id. */
  byId: Map<string, El>;
}

/** Build a fresh page environment whose `#`-ids are the ones the template's
 * static HTML declares. */
export function makeWindow(ids: string[]): PageWindow {
  const byId = new Map<string, El>();
  for (const id of ids) {
    const e = new El("div");
    e.attrs.id = id;
    byId.set(id, e);
  }
  const docListeners: Record<string, ((ev: unknown) => void)[]> = {};
  return {
    byId,
    window: { matchMedia: () => ({ matches: true }) },
    document: {
      documentElement: new El("html"),
      title: "",
      getElementById: (id) => byId.get(id) ?? null,
      createElement: (tag) => new El(tag),
      createTextNode: (t) => {
        const e = new El("#text");
        e.textContent = t;
        return e;
      },
      addEventListener: (type, fn) => {
        (docListeners[type] ??= []).push(fn);
      },
      dispatch: (type, ev) => {
        for (const fn of docListeners[type] ?? []) fn(ev);
      },
    },
  };
}
