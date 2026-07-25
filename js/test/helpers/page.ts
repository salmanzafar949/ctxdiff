/**
 * Boot an EXPORTED dashboard's inline script against the minimal DOM in
 * `minidom.ts`, and hand the test a small driver for the three levels.
 *
 * Why run the real page rather than assert on the payload: the payload says what
 * the dashboard COULD show; only executing the page says what it does. Every
 * behavior the three-level feature actually promises — landing on the right
 * level, drilling agent -> session -> turns, the breadcrumb walking back up,
 * timestamps rendered in the viewer's local zone, and hostile agent names
 * arriving as text rather than markup — lives in that script.
 *
 * The script is lifted from the rendered HTML, so this exercises the SHIPPED
 * bytes (identical in both SDKs), not a copy.
 */
import { runInNewContext } from "node:vm";
import { makeWindow, type El, type PageWindow } from "./minidom.js";

/** Every id the template's static HTML declares. */
const PAGE_IDS = [
  "ctxdiff-data", "h-project", "h-meta", "h-agents", "theme-btn", "crumbs",
  "l1", "l2", "l3", "app", "scrubber", "changed", "alloc", "cache", "blocks",
  "growth",
];

/** A booted page plus the handles a test drives it with. */
export interface Page extends PageWindow {
  /** `#l1`/`#l2`/`#l3` by number — the level containers. */
  level(n: 1 | 2 | 3): El;
  /** Which level is visible (exactly one always is). */
  visibleLevel(): number;
  /** The clickable row controls in the visible listing, in order. */
  rows(): El[];
  /** The breadcrumb's clickable ancestors, in order. */
  crumbs(): El[];
  /** The turn bars in the L3 scrubber. */
  bars(): El[];
  /** The L3 agent scope chips. */
  chips(): El[];
  /** Fire an arrow key at the document, as the page's keydown handler sees it. */
  key(name: string): void;
}

/** Split the two inline pieces out of a rendered dashboard: the JSON island's
 * raw text and the page script's source. Both are located by the exact markers
 * the template emits. */
function split(html: string): { island: string; script: string } {
  const open = '<script id="ctxdiff-data" type="application/json">';
  const start = html.indexOf(open) + open.length;
  const end = html.indexOf("</script>", start);
  const island = html.slice(start, end);
  const sOpen = html.indexOf("<script>", end);
  const sEnd = html.indexOf("</script>", sOpen);
  return { island, script: html.slice(sOpen + "<script>".length, sEnd) };
}

/**
 * Run `html`'s page script and return a driver for it. Throws whatever the page
 * throws, so a runtime error in the dashboard fails the test that booted it.
 */
export function bootPage(html: string): Page {
  const { island, script } = split(html);
  const env = makeWindow(PAGE_IDS);
  env.byId.get("ctxdiff-data")!.textContent = island;

  const sandbox: Record<string, unknown> = {
    document: env.document,
    window: env.window,
    JSON,
    Math,
    Object,
    Array,
    String,
    Number,
    Date,
    isNaN,
    console,
  };
  runInNewContext(script, sandbox);

  const level = (n: 1 | 2 | 3) => env.byId.get("l" + n)!;
  return {
    ...env,
    level,
    visibleLevel: () => [1, 2, 3].find((n) => !level(n as 1 | 2 | 3).hidden) ?? 0,
    rows: () =>
      [1, 2]
        .map((n) => level(n as 1 | 2))
        .filter((e) => !e.hidden)
        .flatMap((e) => e.find((x) => x.className === "rowlink")),
    crumbs: () => env.byId.get("crumbs")!.find((e) => e.className === "crumb"),
    bars: () => env.byId.get("scrubber")!.children,
    chips: () =>
      env.byId.get("h-agents")!.find((e) => e.className.startsWith("agent-chip")),
    key: (name: string) => env.document.dispatch("keydown", { key: name, preventDefault() {} }),
  };
}
