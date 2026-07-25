"""The dashboard page as Python string constants — HTML shell, inline CSS, and
inline JS, all in one module so packaging needs no data files. Everything is
self-contained: no external stylesheet, script, font, or image, and no URL of
any kind (an SVG namespace URL would trip the "no http(s)" self-containment
guarantee, so the growth chart is built as an HTML string and handed to
`innerHTML`, where the HTML parser namespaces `<svg>` automatically — no
`createElementNS` and no xmlns needed).

THE PAGE IS AGENT-FIRST AND THREE-LEVEL:

- LEVEL 1 lists every agent in the project across every session, with its
  aggregate footprint. It is the landing view, because "which agent" is the
  question someone opening a multi-agent project actually has.
- LEVEL 2 lists the sessions ONE agent appeared in, newest first, with each
  session's local start time and that agent's turns and spend in it.
- LEVEL 3 is the turn-by-turn detail — scrubber, block diff, token heatmap,
  cache breaks, block inspector, growth chart — scoped to the chosen agent
  within the chosen session.

A breadcrumb walks back up, and a project with one session and one agent skips
straight to level 3, so the single-session dashboard is unchanged for the case
it was designed for.

TIMESTAMPS ARE CONVERTED IN THE BROWSER. Everything is stored in UTC and
rendered in the VIEWER's local zone by `localTime()` at render time, never baked
in at export — the file is meant to be shared, and the reader may well be in a
different zone than the machine that captured the run. The same bytes therefore
show different clock times to different viewers, which is correct.

`render_page` fills two markers: `__CTXDIFF_TITLE__` (the already-escaped
`<title>` text) and `__CTXDIFF_DATA__` (the JSON island). BOTH are substituted
in one pass, so neither value can be mistaken for the other's marker — a
project name is user text and may spell either marker out verbatim.

The runtime contract with export.py: the JSON island is read back with
`.textContent` and parsed once; all TRACE-DERIVED TEXT — block text, and the
AGENT NAMES and session labels levels 1/2 and the breadcrumb render — reaches
the DOM with `.textContent` (never `.innerHTML`), so untrusted trace data can
never become live markup. Only static chrome and numeric-derived SVG use
`.innerHTML`."""
from __future__ import annotations

import re

# The full page as one ordinary triple-quoted string. The only backslashes are
# deliberate JS-level escapes — `\\uXXXX` (emits a JS `\uXXXX` glyph escape, so
# no non-ASCII bytes ever appear in the source) and `\\"` (an escaped quote
# inside a double-quoted JS string) — each written doubled so Python collapses
# it to the single backslash the emitted JS needs. CSS/JS braces are literal
# (this module never uses str.format), and the page uses string concatenation
# rather than JS template literals, so there are no `${...}` placeholders.
_PAGE = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__CTXDIFF_TITLE__</title>
<style>
:root{
  --bg:#0c100e; --surface:#121815; --panel:#0e1210; --ink:#e7ece8;
  --secondary:#9ba69f; --hairline:rgba(231,236,232,.13);
  --c-system:#3987e5; --c-tool_schema:#d95926; --c-rag:#199e70;
  --c-history:#c98500; --c-user:#d55181; --c-tool_output:#008300;
  --c-unknown:#9085e9;
  --added:#3fb950; --added-bg:rgba(63,185,80,.14);
  --evicted:#f85149; --evicted-bg:rgba(248,81,73,.13);
  --modified:#d29922; --modified-bg:rgba(210,153,34,.15);
  --warn:#fab219; --good:#0ca30c;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
}
:root[data-theme="dark"]{
  --bg:#0c100e; --surface:#121815; --panel:#0e1210; --ink:#e7ece8;
  --secondary:#9ba69f; --hairline:rgba(231,236,232,.13);
  --c-system:#3987e5; --c-tool_schema:#d95926; --c-rag:#199e70;
  --c-history:#c98500; --c-user:#d55181; --c-tool_output:#008300;
  --c-unknown:#9085e9;
}
@media (prefers-color-scheme: light){
  :root{
    --bg:#f4f6f3; --surface:#fdfefc; --panel:#fdfefc; --ink:#171d19;
    --secondary:#57605a; --hairline:rgba(23,29,25,.13);
    --c-system:#2a78d6; --c-tool_schema:#eb6834; --c-rag:#1baf7a;
    --c-history:#eda100; --c-user:#e87ba4; --c-tool_output:#008300;
    --c-unknown:#4a3aa7;
  }
}
:root[data-theme="light"]{
  --bg:#f4f6f3; --surface:#fdfefc; --panel:#fdfefc; --ink:#171d19;
  --secondary:#57605a; --hairline:rgba(23,29,25,.13);
  --c-system:#2a78d6; --c-tool_schema:#eb6834; --c-rag:#1baf7a;
  --c-history:#eda100; --c-user:#e87ba4; --c-tool_output:#008300;
  --c-unknown:#4a3aa7;
}
*{box-sizing:border-box}
html,body{margin:0}
body{
  background:var(--bg); color:var(--ink); font-family:var(--sans);
  font-size:14px; line-height:1.5; -webkit-font-smoothing:antialiased;
}
.num,.mono,h1,h2,.wordmark,.meta,.delta,.seg-pct,.leg-val,
th,td.num,.snippet,.text-prev,.text-full,.inline,.cache-snip{
  font-variant-numeric:tabular-nums;
}
a{color:inherit}

.topbar{
  display:flex; align-items:center; justify-content:space-between;
  gap:16px; flex-wrap:wrap; padding:16px 22px;
  border-bottom:1px solid var(--hairline); background:var(--surface);
  position:sticky; top:0; z-index:5;
}
.brand{display:flex; align-items:baseline; gap:12px; min-width:0}
.wordmark{font-family:var(--mono); font-weight:600; font-size:17px; letter-spacing:-.02em}
.wordmark::before{content:"◆ "; color:var(--c-rag)}
.project{font-family:var(--mono); color:var(--secondary); font-size:14px;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap}
.right{display:flex; align-items:center; gap:14px; min-width:0}
.meta{font-family:var(--mono); font-size:12px; color:var(--secondary);
  display:flex; flex-wrap:wrap; gap:8px; align-items:center; overflow:hidden}
.meta .sep{opacity:.5}
.theme-btn{
  background:transparent; color:var(--secondary); border:1px solid var(--hairline);
  border-radius:8px; width:32px; height:32px; cursor:pointer; font-size:15px;
  flex:none;
}
.theme-btn:hover{color:var(--ink)}
.theme-btn:focus-visible,.bar:focus-visible,summary:focus-visible{
  outline:2px solid var(--c-system); outline-offset:2px;
}

.noscript{margin:18px 22px; padding:14px 16px; border:1px solid var(--warn);
  border-radius:10px; color:var(--warn)}

/* minmax(0,1fr) is load-bearing: grid children default to min-width:auto, so a
   panel holding a long unbreakable line (a single-line diff row) would refuse
   to shrink, blow past max-width, and force the PAGE to scroll horizontally —
   text-overflow ellipsis only engages once the track is genuinely constrained. */
main{max-width:1100px; margin:0 auto; padding:22px; display:grid;
  grid-template-columns:minmax(0,1fr); gap:18px}
main > *{min-width:0}

.panel{
  background:var(--panel); border:1px solid var(--hairline); border-radius:14px;
  padding:18px 20px;
}
.panel-head{display:flex; align-items:baseline; justify-content:space-between;
  gap:12px; margin-bottom:14px}
.panel-head h2{font-family:var(--mono); font-size:13px; font-weight:600;
  letter-spacing:.02em; text-transform:uppercase; margin:0; color:var(--ink)}
.panel-meta{font-family:var(--mono); font-size:12px; color:var(--secondary)}
.empty{color:var(--secondary); font-style:italic; margin:2px 0}
.dim{color:var(--secondary); font-size:12.5px; margin:8px 0 0}
.dim.hint{font-family:var(--mono)}

/* --- scrubber (the spine) --- */
.spine{padding-bottom:20px}
.scrubber{display:flex; align-items:flex-end; gap:5px; min-height:130px;
  overflow-x:auto; padding:6px 2px 2px}
.bar{
  flex:0 0 auto; width:34px; min-height:12px; border:none; cursor:pointer;
  border-radius:5px 5px 2px 2px;
  background:linear-gradient(180deg,var(--c-system),color-mix(in srgb,var(--c-system) 55%, transparent));
  opacity:.62; transition:opacity .12s, transform .12s; position:relative;
}
.bar:hover{opacity:.85}
.bar.sel{opacity:1; outline:2px solid var(--ink); outline-offset:2px}
.bar.err{
  background:linear-gradient(180deg,var(--evicted),color-mix(in srgb,var(--evicted) 45%, transparent));
}
/* Agent-colored underline strip on each turn bar, so an unscoped multi-agent
   timeline still reads as one agent handing off to another. */
.bar-underline{position:absolute; left:2px; right:2px; bottom:0; height:3px;
  border-radius:2px}

/* --- agent chips (header) --- */
.agents{display:flex; flex-wrap:wrap; gap:6px; align-items:center; min-width:0}
.agent-chip{display:inline-flex; align-items:center; gap:6px; cursor:pointer;
  font-family:var(--mono); font-size:12px; color:var(--ink); flex:none;
  background:transparent; border:1px solid var(--hairline); border-radius:999px;
  padding:3px 10px}
.agent-chip:hover{border-color:var(--secondary)}
.agent-chip.active{border-color:var(--ink); background:var(--surface)}
.agent-dot{width:9px; height:9px; border-radius:50%; flex:none; display:inline-block}
.agent-count{color:var(--secondary); font-size:11px}
.agent-chip:focus-visible{outline:2px solid var(--c-system); outline-offset:2px}

/* --- agent hand-off marker (what-changed panel) --- */
.handoff{font-family:var(--mono); font-size:12px; color:var(--modified);
  background:color-mix(in srgb,var(--modified) 12%, transparent);
  border:1px solid color-mix(in srgb,var(--modified) 32%, transparent);
  border-radius:8px; padding:8px 11px; margin-bottom:10px}

/* --- diff / what changed --- */
.diff-item,.diff-row{margin:3px 0}
.diff-row{display:flex; align-items:baseline; gap:10px; padding:5px 9px;
  border-radius:8px; border:1px solid transparent}
details.diff-item{border-radius:8px}
details.diff-item>summary{list-style:none; cursor:pointer}
details.diff-item>summary::-webkit-details-marker{display:none}
.diff-row.added{background:var(--added-bg); border-color:color-mix(in srgb,var(--added) 30%, transparent)}
.diff-row.evicted{background:var(--evicted-bg); border-color:color-mix(in srgb,var(--evicted) 30%, transparent)}
.diff-row.modified{background:var(--modified-bg); border-color:color-mix(in srgb,var(--modified) 30%, transparent)}
.glyph{font-family:var(--mono); font-weight:700; width:1ch; flex:none}
.diff-row.added .glyph{color:var(--added)}
.diff-row.evicted .glyph{color:var(--evicted)}
.diff-row.modified .glyph{color:var(--modified)}
.tag{display:inline-flex; align-items:center; gap:6px; flex:none;
  font-family:var(--mono); font-size:12px; color:var(--secondary)}
.dot{width:8px; height:8px; border-radius:50%; flex:none; display:inline-block}
.snippet{flex:1 1 auto; min-width:0; overflow:hidden; text-overflow:ellipsis;
  white-space:nowrap; font-family:var(--mono); font-size:12.5px; color:var(--ink)}
.delta{flex:none; font-family:var(--mono); font-size:12px}
.delta.added{color:var(--added)} .delta.evicted{color:var(--evicted)}
.delta.modified{color:var(--modified)}
.inline{font-family:var(--mono); font-size:12.5px; white-space:pre-wrap;
  word-break:break-word; padding:8px 11px 10px 30px; line-height:1.7}
.seg-equal{color:var(--secondary)}
.seg-delete{color:var(--evicted); text-decoration:line-through;
  background:var(--evicted-bg); border-radius:3px}
.seg-insert{color:var(--added); background:var(--added-bg); border-radius:3px}
.diff-summary{font-family:var(--mono); font-size:12px; color:var(--secondary);
  margin-bottom:10px}
.diff-summary .up{color:var(--added)} .diff-summary .dn{color:var(--evicted)}

/* --- token allocation --- */
.badge{display:inline-block; font-family:var(--mono); font-size:11px;
  color:var(--warn); border:1px solid var(--warn); border-radius:6px;
  padding:1px 6px; margin-left:8px; vertical-align:middle}
.stack{display:flex; gap:2px; height:26px; margin:4px 0 14px}
.stack .seg{min-width:2px; border-radius:1px; display:flex; align-items:center;
  justify-content:center; overflow:hidden}
.stack .seg:first-child{border-radius:4px 1px 1px 4px}
.stack .seg:last-child{border-radius:1px 4px 4px 1px}
.stack .seg:only-child{border-radius:4px}
.seg-pct{font-family:var(--mono); font-size:11px; font-weight:600;
  color:#fff; mix-blend-mode:normal; text-shadow:0 0 2px rgba(0,0,0,.45)}
.legend{display:flex; flex-wrap:wrap; gap:6px 20px}
.leg-item{display:flex; align-items:center; gap:8px; font-size:12.5px}
.chip{width:10px; height:10px; border-radius:3px; flex:none}
.leg-label{font-family:var(--mono); color:var(--ink)}
.leg-val{color:var(--secondary); font-family:var(--mono); font-size:12px}
.bloat{margin-top:14px; padding:10px 13px; border-radius:9px;
  background:color-mix(in srgb,var(--warn) 14%, transparent);
  border:1px solid color-mix(in srgb,var(--warn) 40%, transparent);
  color:var(--warn); font-size:12.5px}

/* --- cache --- */
.cache-ok{color:var(--good); font-family:var(--mono); font-size:13px;
  padding:8px 0}
.cache-warn{padding:10px 13px; margin:8px 0; border-radius:9px;
  background:color-mix(in srgb,var(--modified) 12%, transparent);
  border:1px solid color-mix(in srgb,var(--modified) 35%, transparent)}
.cache-warn-head{font-family:var(--mono); font-size:12.5px; color:var(--modified)}
.cache-snip{font-family:var(--mono); font-size:12px; color:var(--ink);
  margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
.cache-detail{font-size:12px; color:var(--secondary); margin-top:4px}
.cache-sum{font-family:var(--mono); font-size:12.5px; margin-top:12px;
  color:var(--ink)}

/* --- blocks table --- */
.table-wrap{overflow-x:auto}
table.blocks-table{border-collapse:collapse; width:100%; font-size:12.5px}
.blocks-table th{text-align:left; font-family:var(--mono); font-size:11px;
  text-transform:uppercase; letter-spacing:.04em; color:var(--secondary);
  font-weight:600; padding:6px 12px; border-bottom:1px solid var(--hairline)}
.blocks-table td{padding:7px 12px; border-bottom:1px solid var(--hairline);
  vertical-align:top}
.blocks-table td.num{font-family:var(--mono); color:var(--secondary); text-align:right;
  white-space:nowrap}
.label-chip{display:inline-flex; align-items:center; gap:6px; font-family:var(--mono);
  font-size:12px}
.label-chip .tagged{font-size:10px; color:var(--c-rag); border:1px solid var(--c-rag);
  border-radius:5px; padding:0 4px; margin-left:2px}
sup.est{color:var(--warn); font-size:9px; margin-left:2px; font-family:var(--mono)}
.text-cell details>summary{list-style:none; cursor:pointer}
.text-cell details>summary::-webkit-details-marker{display:none}
.text-prev{font-family:var(--mono); color:var(--ink); white-space:pre-wrap;
  word-break:break-word}
.text-prev:hover{color:var(--c-system)}
.text-full{font-family:var(--mono); font-size:12px; white-space:pre-wrap;
  word-break:break-word; margin:8px 0 0; padding:10px 12px;
  background:var(--surface); border:1px solid var(--hairline); border-radius:8px;
  color:var(--ink); max-height:340px; overflow:auto}

/* --- growth chart --- */
.chart-wrap{width:100%; overflow-x:auto}
.chart-wrap svg{display:block; max-width:100%; height:auto}
.chart-wrap .area{fill:color-mix(in srgb,var(--c-rag) 16%, transparent); stroke:none}
.chart-wrap .line{stroke:var(--c-rag); stroke-width:2; fill:none;
  stroke-linejoin:round; stroke-linecap:round}
.chart-wrap .dot{fill:var(--panel); stroke:var(--c-rag); stroke-width:1.5}
.chart-wrap .dot-sel{fill:var(--c-rag); stroke:var(--ink); stroke-width:1.5}

/* --- three-level navigation --- */
/* `hidden` must beat the display:grid the level containers carry, or every
   level would render at once stacked down the page. */
[hidden]{display:none !important}
#l3{display:grid; grid-template-columns:minmax(0,1fr); gap:18px; min-width:0}
#l3 > *{min-width:0}
.crumbs{display:flex; flex-wrap:wrap; align-items:center; gap:8px;
  font-family:var(--mono); font-size:12.5px; color:var(--secondary);
  padding:2px 2px 0}
.crumb{background:transparent; border:none; padding:0; cursor:pointer;
  font:inherit; color:var(--c-rag); text-decoration:none;
  border-bottom:1px solid color-mix(in srgb,var(--c-rag) 45%, transparent)}
.crumb:hover{color:var(--ink); border-bottom-color:var(--ink)}
.crumb:focus-visible{outline:2px solid var(--c-system); outline-offset:2px}
.crumb-here{color:var(--ink); max-width:100%; overflow:hidden;
  text-overflow:ellipsis; white-space:nowrap}
.crumb-sep{opacity:.5}

/* --- level 1 / level 2 listings --- */
table.list-table{border-collapse:collapse; width:100%; font-size:12.5px}
.list-table th{text-align:left; font-family:var(--mono); font-size:11px;
  text-transform:uppercase; letter-spacing:.04em; color:var(--secondary);
  font-weight:600; padding:6px 12px; border-bottom:1px solid var(--hairline);
  white-space:nowrap}
.list-table td{padding:8px 12px; border-bottom:1px solid var(--hairline);
  vertical-align:middle}
.list-table td.num{font-family:var(--mono); color:var(--ink); text-align:right;
  white-space:nowrap}
.list-table td.when{font-family:var(--mono); color:var(--secondary);
  white-space:nowrap}
.list-table tr.row-open:hover td{background:var(--surface)}
/* The whole first cell is the control, so the click target is the row's name
   rather than a bare chevron — and it stays a real <button>, so keyboard and
   screen-reader users get the same affordance a mouse user does. */
.rowlink{background:transparent; border:none; padding:0; cursor:pointer;
  font:inherit; color:var(--ink); font-family:var(--mono); font-size:12.5px;
  display:inline-flex; align-items:center; gap:8px; text-align:left;
  max-width:100%; min-width:0}
.rowlink:hover .rowlink-name{color:var(--c-rag)}
.rowlink:focus-visible{outline:2px solid var(--c-system); outline-offset:2px}
.rowlink-name{overflow:hidden; text-overflow:ellipsis; white-space:nowrap}
.rowlink[disabled]{cursor:default; color:var(--secondary)}
.rowlink[disabled]:hover .rowlink-name{color:var(--secondary)}
.sub{font-family:var(--mono); font-size:11.5px; color:var(--secondary)}
.nodetail{font-family:var(--mono); font-size:11px; color:var(--secondary);
  border:1px solid var(--hairline); border-radius:6px; padding:1px 6px;
  white-space:nowrap}
.cap-note{color:var(--secondary); font-size:12px; margin:12px 0 0;
  font-family:var(--mono)}
</style>
</head>
<body>
<header class="topbar">
  <div class="brand">
    <span class="wordmark">ctxdiff</span>
    <span id="h-project" class="project"></span>
  </div>
  <div class="right">
    <div id="h-agents" class="agents"></div>
    <div id="h-meta" class="meta"></div>
    <button id="theme-btn" class="theme-btn" title="toggle light/dark" aria-label="toggle light/dark theme">&#9680;</button>
  </div>
</header>

<noscript>
  <p class="noscript">This dashboard renders an embedded trace with JavaScript.
  Nothing is fetched &mdash; every byte, including the trace data, is inline in
  this file &mdash; but you need to enable JavaScript to view it.</p>
</noscript>

<main id="app">
  <nav id="crumbs" class="crumbs" aria-label="breadcrumb" hidden></nav>
  <section id="l1" class="panel" hidden></section>
  <section id="l2" class="panel" hidden></section>
  <div id="l3" hidden>
    <section class="panel spine">
      <div class="panel-head"><h2>Turns</h2>
        <span class="panel-meta">click a bar &middot; &larr; &rarr; to navigate</span></div>
      <div id="scrubber" class="scrubber" role="tablist" aria-label="turn scrubber"></div>
    </section>
    <section id="changed" class="panel"></section>
    <section id="alloc" class="panel"></section>
    <section id="cache" class="panel"></section>
    <section id="blocks" class="panel"></section>
    <section id="growth" class="panel"></section>
  </div>
</main>

<script id="ctxdiff-data" type="application/json">__CTXDIFF_DATA__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("ctxdiff-data").textContent);

// --- the three levels --------------------------------------------------------
// L1 ALL AGENTS -> L2 that agent's SESSIONS -> L3 one session's turn-by-turn
// detail. The focus session's detail is the payload's TOP LEVEL (run/calls/
// diffs/tokens/cache/stats); every other embedded session's detail lives under
// project.details, keyed by session id. A payload with no `project` key at all
// is a plain single-session export, which is simply L3 with nothing above it.
const PROJECT = DATA.project || null;
const FOCUS_DETAIL = {run: DATA.run, calls: DATA.calls, diffs: DATA.diffs,
                      tokens: DATA.tokens, cache: DATA.cache, stats: DATA.stats};
const P_AGENTS = PROJECT ? (PROJECT.agents || []) : [];
const P_SESSIONS = PROJECT ? (PROJECT.sessions || []) : [];
// Whether there is anything ABOVE the detail view to navigate to. A project with
// one session and one agent has no L1/L2 worth showing, so the breadcrumb stays
// hidden and the page behaves exactly like the single-session dashboard always
// did.
const MULTI_LEVEL = P_SESSIONS.length > 1 || P_AGENTS.length > 1;

let level = 3;          // which of the three levels is on screen
let curAgent = null;    // the agent scope: L2's subject, and L3's turn filter
let curSession = PROJECT ? PROJECT.focus : null;
let D = FOCUS_DETAIL;   // the ACTIVE session's detail (what every L3 panel reads)
let CALLS = D.calls || [];
let VIEW = [];          // indices into CALLS visible under the current scope
let sel = 0;            // the selected turn, as an index into CALLS

/** One session's detail, or null when it was not embedded (see the exporter's
 * detail cap). The focus session is answered from the top level rather than from
 * project.details, which is why it is never serialized twice. */
function detailFor(sid){
  if(!PROJECT || sid === PROJECT.focus) return FOCUS_DETAIL;
  return (PROJECT.details || {})[sid] || null;
}
/** Point the L3 panels at `sid`. Returns false (changing nothing) when that
 * session has no embedded detail, so a caller can leave the row inert instead of
 * navigating into an empty view. */
function openSession(sid){
  const d = detailFor(sid);
  if(!d) return false;
  curSession = sid;
  D = d;
  CALLS = D.calls || [];
  sel = 0;
  return true;
}
/** The L2 row for `sid`, which is where a session's start time, provider and
 * turn count live once the page has navigated away from the focus session. */
function sessionRow(sid){
  for(const s of P_SESSIONS) if(s.id === sid) return s;
  return null;
}
/** The 12-character session prefix the CLI prints and accepts — what is shown
 * wherever a session needs a stable name to paste back into `--session`. */
function shortId(id){ return (id || "").slice(0, 12); }

// --- local-time rendering ----------------------------------------------------
// Timestamps are STORED in UTC and rendered in the VIEWER's local zone, here, at
// render time — not baked in at export. The same file is meant to be shared, and
// the person opening it is trying to match a run against "the one I did after
// lunch" in THEIR day, not in the capturing machine's. Two viewers in two zones
// therefore see different text from byte-identical HTML, which is the point.
/** Normalize the timestamp spellings a store may hold into something `Date` can
 * parse: the ISO BASIC form (`20260704T100000Z`), an hour-only offset (`+05`),
 * and a naive value — which is UTC by ctxdiff's storage contract, so it is
 * stamped as such rather than being read in the viewer's own zone. */
function isoNormalize(text){
  let t = text.trim();
  const basic = t.match(/^([0-9]{4})([0-9]{2})([0-9]{2})T([0-9]{2})([0-9]{2})([0-9]{2})(.*)$/);
  if(basic) t = basic[1] + "-" + basic[2] + "-" + basic[3] + "T" +
                basic[4] + ":" + basic[5] + ":" + basic[6] + basic[7];
  if(/[+-][0-9]{2}$/.test(t)) t += ":00";
  if(!/(Z|z|[+-][0-9]{2}:[0-9]{2})$/.test(t)) t += "Z";
  return t;
}
/** A stored UTC timestamp as `2026-07-24 16:03:11 +04:00` in the viewer's local
 * zone — the same columns and spelling `ctxdiff sessions` prints, so the CLI and
 * the dashboard describe a run identically.
 *
 * Degrades rather than throwing: an empty value renders "-", and anything the
 * parser cannot make sense of is echoed back unchanged, so one odd row never
 * blanks a whole listing. */
function localTime(value){
  const raw = (value == null ? "" : String(value)).trim();
  if(!raw) return "-";
  const d = new Date(isoNormalize(raw));
  if(isNaN(d.getTime())) return raw;
  const p = n => (n < 10 ? "0" : "") + n;
  // getTimezoneOffset() is minutes WEST of UTC, so the sign is inverted to read
  // as the offset a timestamp is normally written with.
  const off = -d.getTimezoneOffset();
  const sign = off < 0 ? "-" : "+";
  const ab = Math.abs(off);
  return d.getFullYear() + "-" + p(d.getMonth() + 1) + "-" + p(d.getDate()) +
         " " + p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds()) +
         " " + sign + p(Math.floor(ab / 60)) + ":" + p(ab % 60);
}

// --- small DOM helpers -------------------------------------------------------
function el(tag, cls, txt){
  const e = document.createElement(tag);
  if(cls) e.className = cls;
  if(txt != null) e.textContent = txt;   // textContent, never innerHTML, for data
  return e;
}
function fmt(n){ return (n == null ? 0 : n).toLocaleString("en-US"); }
// Color follows the entity: a fixed CSS var per label, violet for anything
// unknown/custom, so the same label is the same hue in every panel.
// The membership test asks for an OWN property: labels are arbitrary user text
// (tracer.tag() takes any string), and a plain object INHERITS truthy values for
// `__proto__`/`constructor`/`toString` — those read as known and interpolated
// into `var(--c-...)`, naming a custom property the stylesheet never declares,
// so the swatch lost its color entirely. Only the six names below may reach CSS.
const KNOWN_LABELS = {system:1, tool_schema:1, rag:1, history:1, user:1, tool_output:1};
function labelColor(label){
  const known = Object.prototype.hasOwnProperty.call(KNOWN_LABELS, label);
  return "var(--c-" + (known ? label : "unknown") + ")";
}
function head(title, meta){
  const h = el("div", "panel-head");
  h.appendChild(el("h2", null, title));
  if(meta != null) h.appendChild(el("span", "panel-meta", meta));
  return h;
}
function dot(label){ const d = el("span", "dot"); d.style.background = labelColor(label); return d; }

// --- agents ------------------------------------------------------------------
// Per-agent color is assigned by order of first appearance from a fixed
// categorical palette, cycled when a project has more agents than colors. The
// assignment is PROJECT-wide (not per session) so one agent keeps one hue across
// all three levels — a color that changed between two sessions of the same run
// would make the levels read as unrelated pages. Agent NAMES are NEVER
// interpolated into CSS — only these fixed hex values reach a style property —
// so a hostile agent name cannot inject styles. Every agent name that reaches
// the DOM does so via el()/textContent.
const AGENT_PALETTE = ["#3987e5","#d95926","#199e70","#c98500","#d55181",
                       "#9085e9","#008300","#c0498a"];
// A NULL-PROTOTYPE map, because agent names are arbitrary user text and a plain
// object inherits keys it was never given: assigning a string to `__proto__` on
// one is a silent no-op (the setter runs, no own property appears), so that
// agent lost its color and `agentColor` handed back Object.prototype, which the
// browser drops as a style value. `constructor`/`toString` would have returned
// inherited functions the same way. With no prototype, every name is data.
const AGENT_COLOR = Object.create(null);
(P_AGENTS.length ? P_AGENTS : ((DATA.stats && DATA.stats.agents) || []))
  .forEach((a, i) => { AGENT_COLOR[a.name] = AGENT_PALETTE[i % AGENT_PALETTE.length]; });
function agentKey(call){ return call && call.agent != null ? call.agent : "(unlabeled)"; }
function agentColor(name){ return AGENT_COLOR[name] || "var(--c-unknown)"; }
/** The active session's own agents (the L3 header chips) — a session shows only
 * the agents that actually ran in it, not every agent in the project. */
function sessionAgents(){ return (D.stats && D.stats.agents) || []; }
/** Which of CALLS the current agent scope admits. An agent scope carried in from
 * L1/L2 that turns out to be absent from this session falls back to showing
 * everything, because an empty scrubber is a worse answer than an unscoped one.
 * `sel` is pulled back into range whenever the scope moves it out. */
function computeView(){
  VIEW = [];
  for(let i = 0; i < CALLS.length; i++){
    if(curAgent === null || agentKey(CALLS[i]) === curAgent) VIEW.push(i);
  }
  if(!VIEW.length) for(let i = 0; i < CALLS.length; i++) VIEW.push(i);
  if(VIEW.indexOf(sel) < 0) sel = VIEW.length ? VIEW[0] : 0;
}
// A trailing " \\u00b7 agent \\u00b7 step" fragment for a panel header, or "" when
// neither is set. Built as plain text handed to el()/textContent by the caller.
function agentStep(call){
  const parts = [];
  if(call.agent != null) parts.push(call.agent);
  if(call.step != null) parts.push(call.step);
  return parts.length ? " \\u00b7 " + parts.join(" \\u00b7 ") : "";
}
function renderAgents(){
  const host = document.getElementById("h-agents");
  host.innerHTML = "";
  const agents = sessionAgents();
  // Chips are the L3 agent SCOPE control. One agent means there is nothing to
  // scope to, and levels 1/2 have their own subject, so neither shows chips.
  if(level !== 3 || agents.length <= 1) return;
  agents.forEach(a => {
    const chip = el("button", "agent-chip" + (curAgent === a.name ? " active" : ""));
    const d = el("span", "agent-dot"); d.style.background = agentColor(a.name);
    chip.appendChild(d);
    chip.appendChild(el("span", "agent-name", a.name));      // textContent — safe
    chip.appendChild(el("span", "agent-count", "\\u00b7 " + a.calls));
    // Provider in/out on the tooltip (title attribute — value, never parsed as
    // markup); present only when this agent reported usage.
    const uba = (D.stats.usage || {}).by_agent || {};
    // OWN property only: `uba` is a plain object, so an agent named __proto__
    // or toString would otherwise inherit a truthy value and render a tooltip
    // built from a prototype rather than from this run's numbers.
    const io = Object.prototype.hasOwnProperty.call(uba, a.name)
      ? uba[a.name] : null;
    if(io) chip.setAttribute("title", a.name + " \\u00b7 in " + fmt(io[0]) +
                             " \\u00b7 out " + fmt(io[1]));
    chip.setAttribute("aria-pressed", curAgent === a.name ? "true" : "false");
    chip.addEventListener("click", () => {
      curAgent = (curAgent === a.name) ? null : a.name;  // toggle the scope
      render();
    });
    host.appendChild(chip);
  });
}

// --- breadcrumb + level switching --------------------------------------------
/** Jump to level 1 (all agents). */
function goAgents(){ level = 1; curAgent = null; render(); }
/** Jump to level 2: the sessions `name` appeared in. */
function goSessions(name){ level = 2; curAgent = name; render(); }
/** Jump to level 3: one session's detail, optionally scoped to one agent.
 * Silently does nothing when that session's detail was not embedded — the row
 * that would have called this is rendered inert, so this is belt and braces. */
function goDetail(sid, name){
  if(!openSession(sid)) return;
  if(name !== undefined) curAgent = name;
  level = 3;
  render();
}
/** The trail back up: `all agents › <agent> › <session>`, with every ancestor a
 * real button and the current position plain text. Hidden entirely for a project
 * with nothing above the detail view. */
function renderCrumbs(){
  const host = document.getElementById("crumbs");
  host.innerHTML = "";
  host.hidden = !MULTI_LEVEL;
  if(!MULTI_LEVEL) return;
  const parts = [];
  if(level === 1){
    parts.push(el("span", "crumb-here", "all agents"));
  } else {
    const b = el("button", "crumb", "all agents");
    b.addEventListener("click", goAgents);
    parts.push(b);
  }
  if(curAgent !== null){
    if(level === 2){
      parts.push(el("span", "crumb-here", curAgent));      // textContent — safe
    } else {
      const b = el("button", "crumb", curAgent);
      const name = curAgent;
      b.addEventListener("click", () => goSessions(name));
      parts.push(b);
    }
  }
  if(level === 3){
    const row = sessionRow(curSession);
    const when = localTime(row ? row.started_at : (D.run || {}).started_at);
    parts.push(el("span", "crumb-here",
                  when + " \\u00b7 " + shortId(curSession || (D.run || {}).id)));
  }
  parts.forEach((p, i) => {
    if(i) host.appendChild(el("span", "crumb-sep", "\\u203a"));
    host.appendChild(p);
  });
}
/** Show exactly one level's container. */
function showLevel(){
  document.getElementById("l1").hidden = level !== 1;
  document.getElementById("l2").hidden = level !== 2;
  document.getElementById("l3").hidden = level !== 3;
}

// --- level 1: all agents -----------------------------------------------------
/** A token cell: provider in/out summed, or "-" when NO call of this row
 * reported usage — 0 would read as "this was free", which is a lie about a
 * provider that simply returned no usage block. */
function tokenCell(row){
  const td = el("td", "num");
  td.textContent = row.reported ? fmt(row.input + row.output) : "-";
  if(row.reported) td.title = "in " + fmt(row.input) + " \\u00b7 out " + fmt(row.output) +
                              " \\u00b7 " + row.reported + " calls reported usage";
  return td;
}
/** A clickable first cell: the row's name as a real <button>. `onOpen` null
 * leaves it inert (used for a session whose detail was not embedded). */
function rowLink(name, color, onOpen){
  const b = el("button", "rowlink");
  if(color){ const d = el("span", "agent-dot"); d.style.background = color; b.appendChild(d); }
  b.appendChild(el("span", "rowlink-name", name));          // textContent — safe
  if(onOpen) b.addEventListener("click", onOpen);
  else b.disabled = true;
  return b;
}
/** Build a table with `cols` headers and return its <tbody> to fill. */
function listTable(host, cols){
  const wrap = el("div", "table-wrap");
  const tbl = el("table", "list-table");
  const thead = el("thead"); const hr = el("tr");
  cols.forEach(c => hr.appendChild(el("th", null, c)));
  thead.appendChild(hr); tbl.appendChild(thead);
  const tb = el("tbody");
  tbl.appendChild(tb); wrap.appendChild(tbl); host.appendChild(wrap);
  return tb;
}
/** LEVEL 1 — every agent in the project, aggregated across every session. The
 * landing view: pick who you want to look at before picking which run. */
function renderLevel1(){
  const host = document.getElementById("l1");
  host.innerHTML = "";
  host.appendChild(head("Agents", P_AGENTS.length + " in " +
                        (PROJECT ? PROJECT.sessions_total : P_SESSIONS.length) + " sessions"));
  if(!P_AGENTS.length){
    host.appendChild(el("p", "empty", "no agents in this project"));
    return;
  }
  const tb = listTable(host, ["agent", "sessions", "calls", "tokens", "first seen", "last seen"]);
  P_AGENTS.forEach(a => {
    const tr = el("tr", "row-open");
    const td = el("td");
    const name = a.name;
    td.appendChild(rowLink(name, agentColor(name), () => goSessions(name)));
    tr.appendChild(td);
    tr.appendChild(el("td", "num", fmt(a.sessions)));
    tr.appendChild(el("td", "num", fmt(a.calls)));
    tr.appendChild(tokenCell(a));
    tr.appendChild(el("td", "when", localTime(a.first_seen)));
    tr.appendChild(el("td", "when", localTime(a.last_seen)));
    tb.appendChild(tr);
  });
}

// --- level 2: one agent's sessions -------------------------------------------
/** LEVEL 2 — every session the selected agent appeared in, newest first, with
 * ITS turns and ITS spend in that session (not the session's totals). */
function renderLevel2(){
  const host = document.getElementById("l2");
  host.innerHTML = "";
  const rows = [];
  P_SESSIONS.forEach(s => {
    for(const a of (s.agents || [])) if(a.name === curAgent) rows.push([s, a]);
  });
  host.appendChild(head("Sessions", (curAgent || "") + " \\u00b7 " + rows.length +
                        (rows.length === 1 ? " session" : " sessions")));
  if(!rows.length){
    host.appendChild(el("p", "empty", "this agent has no recorded sessions"));
    return;
  }
  const tb = listTable(host, ["started", "session", "turns", "tokens", "model", ""]);
  let capped = 0;
  rows.forEach(([s, a]) => {
    const tr = el("tr", "row-open");
    const td = el("td");
    const label = localTime(s.started_at);
    const sid = s.id, name = curAgent;
    td.appendChild(rowLink(label, null, s.detail ? () => goDetail(sid, name) : null));
    tr.appendChild(td);
    tr.appendChild(el("td", "sub", shortId(s.id)));
    tr.appendChild(el("td", "num", fmt(a.turns) + " / " + fmt(s.turn_count)));
    tr.appendChild(tokenCell(a));
    tr.appendChild(el("td", "sub", (s.models || []).join(", ") || s.provider || "-"));
    const last = el("td");
    if(!s.detail){ capped += 1; last.appendChild(el("span", "nodetail", "detail not embedded")); }
    tr.appendChild(last);
    tb.appendChild(tr);
  });
  if(capped){
    host.appendChild(el("p", "cap-note",
      "turn-by-turn detail is embedded for the " + PROJECT.detail_cap +
      " most recent sessions to keep this file self-contained \\u2014 " + capped +
      " older " + (capped === 1 ? "session is" : "sessions are") +
      " listed with totals only. Re-export with --session <id> to drill into one."));
  }
}

// --- header ------------------------------------------------------------------
/** The meta strip is LEVEL-AWARE: levels 1 and 2 describe the PROJECT (how many
 * sessions and agents it holds), because a per-session rollup would be
 * describing a session the user has not chosen yet; level 3 describes the
 * session on screen, exactly as the single-session dashboard always did. */
function projectMeta(){
  const total = PROJECT ? PROJECT.sessions_total : P_SESSIONS.length;
  const turns = P_SESSIONS.reduce((a, s) => a + (s.turn_count || 0), 0);
  return [total + (total === 1 ? " session" : " sessions"),
          P_AGENTS.length + (P_AGENTS.length === 1 ? " agent" : " agents"),
          fmt(turns) + " turns"];
}
function sessionMeta(){
  const r = D.run || {};
  const total = (D.stats.context_growth || []).reduce((a,b)=>a+b, 0);
  const dedup = D.stats.distinct_blocks + " distinct blocks / " +
                D.stats.total_block_refs + " references";
  // Provider-usage rollup, shown only when at least one call reported usage —
  // never fabricate an "in 0 / out 0" from a run with no provider numbers.
  const u = D.stats.usage || {};
  const cov = u.coverage || [0, 0];
  // The model segment is OMITTED entirely (not rendered as "?") when
  // r.models is empty — a run whose calls never reported a model (or, pre-
  // capture, one with no calls yet) should show "openai · <started> · ..."
  // rather than a dangling " · ? · " placeholder for a field that simply
  // has no value to show.
  const modelsStr = (r.models || []).join(", ");
  const items = [ r.provider || "?" ];
  if(modelsStr) items.push(modelsStr);
  // The start time is rendered in the VIEWER's local zone, not echoed as the
  // stored UTC string — see localTime().
  items.push(localTime(r.started_at), CALLS.length + " turns",
             fmt(total) + " tokens");
  if(cov[0] > 0){
    items.push("in " + fmt(u.input) + " \\u00b7 out " + fmt(u.output) +
               " (" + cov[0] + "/" + cov[1] + " reported)");
  }
  items.push(dedup);
  return items;
}
function renderHeader(){
  const name = PROJECT ? PROJECT.name : ((DATA.run || {}).project || "");
  document.getElementById("h-project").textContent = name;
  document.title = "ctxdiff \\u2014 " + name;
  const items = level === 3 ? sessionMeta() : projectMeta();
  const box = document.getElementById("h-meta");
  box.innerHTML = "";
  items.forEach((m, i) => {
    if(i) box.appendChild(el("span", "sep", "\\u00b7"));
    box.appendChild(el("span", "meta-item", m));
  });
}

// --- scrubber ----------------------------------------------------------------
// The scrubber shows the turns the current agent SCOPE admits (VIEW), not every
// turn of the session — an agent's own timeline is what "this agent's runs with
// traces" means. Bar heights are still scaled against the whole session's peak
// so scoping never makes a turn look bigger than it was.
function renderScrubber(){
  const strip = document.getElementById("scrubber");
  strip.innerHTML = "";
  const growth = D.stats.context_growth || [];
  const max = Math.max(1, ...growth);
  const multi = sessionAgents().length > 1;
  VIEW.forEach(i => {
    const c = CALLS[i];
    const tok = growth[i] || 0;
    const b = document.createElement("button");
    b.className = "bar" + (i === sel ? " sel" : "") + (c.error ? " err" : "");
    b.style.height = (12 + (tok / max) * 104).toFixed(1) + "px";
    b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", i === sel ? "true" : "false");
    b.setAttribute("aria-label", "turn " + c.seq + " \\u2014 " + fmt(tok) +
                   " tokens" + (c.error ? " (error)" : ""));
    b.addEventListener("click", () => { sel = i; render(); });
    if(multi){
      const u = el("span", "bar-underline"); u.style.background = agentColor(agentKey(c));
      b.appendChild(u);
    }
    strip.appendChild(b);
  });
}

// --- what changed ------------------------------------------------------------
function fillDiffRow(parent, e){
  parent.appendChild(el("span", "glyph",
    e.kind === "added" ? "+" : e.kind === "evicted" ? "\\u2212" :
    e.kind === "modified" ? "~" : "="));
  const tag = el("span", "tag");
  tag.appendChild(dot(e.label));
  tag.appendChild(el("span", "tag-txt", "[" + e.label + "\\u00b7" + e.role + "]"));
  parent.appendChild(tag);
  parent.appendChild(el("span", "snippet", e.snippet));
  const sign = e.kind === "evicted" ? "\\u2212" : "+";
  parent.appendChild(el("span", "delta " + e.kind, sign + fmt(e.token_count) + " tok"));
}
function diffRow(e){
  if(e.kind === "modified" && e.inline_diff){
    const det = document.createElement("details");
    det.className = "diff-item";
    const sum = document.createElement("summary");
    sum.className = "diff-row modified";
    fillDiffRow(sum, e);
    det.appendChild(sum);
    const body = el("div", "inline");
    e.inline_diff.forEach(([op, txt]) => body.appendChild(el("span", "seg-" + op, txt)));
    det.appendChild(body);
    return det;
  }
  const row = el("div", "diff-row " + e.kind);
  fillDiffRow(row, e);
  return row;
}
function renderChanged(){
  const host = document.getElementById("changed");
  host.innerHTML = "";
  const c = CALLS[sel];
  const label = sel > 0 ? ("turn " + CALLS[sel-1].seq + " \\u2192 " + c.seq)
                        : ("turn " + c.seq);
  host.appendChild(head("What changed", label));
  if(sel === 0){
    host.appendChild(el("p", "empty", "first turn \\u2014 everything is new"));
    return;
  }
  let d = D.diffs[sel-1];
  // Agent hand-off: when the previous GLOBAL turn belongs to a different
  // agent, mark it and (when available) show the diff against THIS agent's own
  // previous turn instead of the cross-agent one.
  if(d.cross_agent){
    const prevName = agentKey(CALLS[sel-1]);
    const curName = agentKey(c);
    host.appendChild(el("div", "handoff",
      "agent hand-off \\u2014 previous turn was " + prevName +
      (d.same_agent_diff
        ? "; diff vs " + curName + "'s own previous turn shown instead"
        : "; no earlier turn for " + curName + " to diff against")));
    if(d.same_agent_diff) d = d.same_agent_diff;
  }
  const sum = el("div", "diff-summary");
  const up = el("span", "up", "+" + fmt(d.tokens_added) + " added");
  const dn = el("span", "dn", "\\u2212" + fmt(d.tokens_evicted) + " evicted");
  sum.appendChild(up); sum.appendChild(document.createTextNode("  \\u00b7  ")); sum.appendChild(dn);
  host.appendChild(sum);
  const changed = d.entries.filter(e => e.kind !== "unchanged");
  if(!changed.length) host.appendChild(el("p", "empty", "no block-level changes this turn"));
  changed.forEach(e => host.appendChild(diffRow(e)));
  const unchanged = d.entries.length - changed.length;
  if(unchanged) host.appendChild(el("p", "dim", "= " + unchanged + " unchanged blocks"));
}

// --- token allocation --------------------------------------------------------
function renderTokens(){
  const host = document.getElementById("alloc");
  host.innerHTML = "";
  const t = D.tokens.calls[sel];
  const h = head("Token allocation", fmt(t.total) + " tokens" + agentStep(CALLS[sel]));
  if(t.approximate){ const b = el("span", "badge", "~approx"); h.querySelector("h2").appendChild(b); }
  host.appendChild(h);

  const stack = el("div", "stack");
  t.slices.forEach(s => {
    const seg = el("div", "seg");
    seg.style.width = s.pct + "%";
    seg.style.background = labelColor(s.label);
    seg.title = s.label + " \\u00b7 " + fmt(s.tokens) + " tok \\u00b7 " + s.pct + "%";
    if(s.pct >= 8) seg.appendChild(el("span", "seg-pct", Math.round(s.pct) + "%"));
    stack.appendChild(seg);
  });
  host.appendChild(stack);

  const legend = el("div", "legend");
  t.slices.forEach(s => {
    const it = el("div", "leg-item");
    const chip = el("span", "chip"); chip.style.background = labelColor(s.label);
    it.appendChild(chip);
    it.appendChild(el("span", "leg-label", s.label));
    it.appendChild(el("span", "leg-val", fmt(s.tokens) + " \\u00b7 " + s.pct.toFixed(1) + "%"));
    legend.appendChild(it);
  });
  host.appendChild(legend);

  const usage = CALLS[sel].usage;
  const parts = [];
  if(usage) for(const k in usage) parts.push(k + " " + fmt(usage[k]));
  if(t.reconciliation_delta != null)
    parts.push("\\u0394 vs measured " + (t.reconciliation_delta >= 0 ? "+" : "") + t.reconciliation_delta);
  if(parts.length) host.appendChild(el("p", "dim", "provider usage: " + parts.join("  \\u00b7  ")));

  const bloat = D.tokens.bloat;
  if(bloat && bloat.unused_tools && bloat.unused_tools.length){
    host.appendChild(el("div", "bloat",
      "\\u26a0 schema bloat: " + bloat.unused_tools.join(", ") + " \\u2014 " +
      fmt(bloat.unused_tokens_per_call) + " tok/call (" +
      bloat.pct_of_avg_context + "% of avg context) spent on dead schemas every turn"));
  }
}

// --- cache alignment ---------------------------------------------------------
function renderCache(){
  const host = document.getElementById("cache");
  host.innerHTML = "";
  const cc = D.cache;
  host.appendChild(head("Cache alignment", cc.pairs_analyzed + " turn pairs"));
  if(cc.pairs_analyzed === 0){
    host.appendChild(el("p", "empty", cc.waste_note || "nothing to analyze"));
    return;
  }
  if(!cc.breaks.length){
    host.appendChild(el("div", "cache-ok",
      "\\u2713 prefix stable across all turns \\u2014 minimum stable prefix " +
      fmt(cc.stable_prefix_tokens_min) + " tokens"));
  } else {
    const groups = {};
    cc.breaks.forEach(b => {
      const k = b.culprit_kind + "|" + b.culprit_label + "|" + b.divergent_position;
      (groups[k] = groups[k] || []).push(b);
    });
    Object.values(groups).forEach(g => {
      const rep = g[0];
      const w = el("div", "cache-warn");
      w.appendChild(el("div", "cache-warn-head",
        "\\u26a0 [" + rep.culprit_label + "\\u00b7" + rep.culprit_kind + "] breaks the prefix on " +
        g.length + "/" + cc.pairs_analyzed + " pairs \\u2014 at position " + rep.divergent_position));
      w.appendChild(el("div", "cache-snip", rep.culprit_snippet));
      w.appendChild(el("div", "cache-detail", rep.detail));
      host.appendChild(w);
    });
    host.appendChild(el("div", "cache-sum",
      "stable prefix (min) " + fmt(cc.stable_prefix_tokens_min) +
      " tokens \\u00b7 re-billed " + fmt(cc.rebilled_tokens_total) + " tokens"));
  }
  if(cc.waste_note) host.appendChild(el("p", "dim", cc.waste_note));
  if(cc.fix_hint) host.appendChild(el("p", "dim hint", "hint: " + cc.fix_hint));
}

// --- blocks table ------------------------------------------------------------
function renderBlocks(){
  const host = document.getElementById("blocks");
  host.innerHTML = "";
  const c = CALLS[sel];
  host.appendChild(head("Blocks \\u00b7 turn " + c.seq,
                        c.blocks.length + " blocks" + agentStep(c)));
  const wrap = el("div", "table-wrap");
  const tbl = el("table", "blocks-table");
  const thead = el("thead"); const hr = el("tr");
  ["#", "label", "role", "kind", "tokens", "text"].forEach(x => hr.appendChild(el("th", null, x)));
  thead.appendChild(hr); tbl.appendChild(thead);
  const tb = el("tbody");
  c.blocks.forEach(b => {
    const tr = el("tr");
    tr.appendChild(el("td", "num", String(b.position)));

    const lc = el("td");
    const chip = el("span", "label-chip");
    chip.appendChild(dot(b.label));
    chip.appendChild(el("span", null, b.label));
    if(b.label_source === "tagged") chip.appendChild(el("span", "tagged", "tagged"));
    lc.appendChild(chip); tr.appendChild(lc);

    tr.appendChild(el("td", null, b.role));
    tr.appendChild(el("td", null, b.kind));

    const tk = el("td", "num"); tk.appendChild(document.createTextNode(fmt(b.token_count)));
    if(b.token_method === "estimate") tk.appendChild(el("sup", "est", "~est"));
    tr.appendChild(tk);

    const tc = el("td", "text-cell");
    const det = document.createElement("details");
    const sm = document.createElement("summary");
    sm.className = "text-prev";
    sm.textContent = b.text.slice(0, 120) + (b.text.length > 120 ? "\\u2026" : "");
    det.appendChild(sm);
    const full = el("pre", "text-full"); full.textContent = b.text;   // full text via textContent
    det.appendChild(full);
    tc.appendChild(det); tr.appendChild(tc);

    tb.appendChild(tr);
  });
  tbl.appendChild(tb); wrap.appendChild(tbl); host.appendChild(wrap);
}

// --- growth chart (inline SVG built from numbers only) -----------------------
function renderGrowth(){
  const host = document.getElementById("growth");
  host.innerHTML = "";
  host.appendChild(head("Context growth", null));
  const g = VIEW.map(i => (D.stats.context_growth || [])[i] || 0);
  const n = g.length;
  const W = Math.max(320, n * 64), H = 150, pad = 26;
  const max = Math.max(1, ...g);
  const xat = i => n <= 1 ? W / 2 : pad + i * (W - 2 * pad) / (n - 1);
  const yat = v => H - pad - (v / max) * (H - 2 * pad);
  const pts = g.map((v, i) => xat(i).toFixed(1) + "," + yat(v).toFixed(1)).join(" ");
  // The chart plots the SCOPED timeline (VIEW), so `sel` — an index into CALLS —
  // has to be translated into this series' own index before it can mark a point.
  const at = VIEW.indexOf(sel);
  let svg = "<svg viewBox=\\"0 0 " + W + " " + H + "\\" preserveAspectRatio=\\"xMidYMid meet\\" role=\\"img\\" aria-label=\\"context tokens per turn\\">";
  if(n > 0){
    const area = pad.toFixed(1) + "," + (H - pad) + " " + pts + " " +
                 xat(n - 1).toFixed(1) + "," + (H - pad);
    svg += "<polygon class=\\"area\\" points=\\"" + area + "\\"></polygon>";
    svg += "<polyline class=\\"line\\" points=\\"" + pts + "\\"></polyline>";
    g.forEach((v, i) => {
      svg += "<circle class=\\"" + (i === at ? "dot-sel" : "dot") + "\\" cx=\\"" +
             xat(i).toFixed(1) + "\\" cy=\\"" + yat(v).toFixed(1) + "\\" r=\\"" +
             (i === at ? 4.5 : 3) + "\\"></circle>";
    });
  }
  svg += "</svg>";
  const box = el("div", "chart-wrap");
  box.innerHTML = svg;   // numeric-derived markup only; no trace text here
  host.appendChild(box);
}

// --- orchestration -----------------------------------------------------------
/** One entry point for every state change: recompute the scoped turn list, then
 * repaint the header, the breadcrumb, and whichever level is on screen. Only the
 * visible level is built, so a project with thousands of listed sessions never
 * pays to render turn panels nobody is looking at. */
function render(){
  computeView();
  renderHeader();
  renderCrumbs();
  showLevel();
  if(level === 1){ renderLevel1(); return; }
  if(level === 2){ renderLevel2(); return; }
  if(!CALLS.length){ renderEmptyDetail(); return; }
  renderAgents();
  renderScrubber();
  renderChanged();
  renderTokens();
  renderCache();
  renderBlocks();
  renderGrowth();
}

/** A session with no captured calls still has to render SOMETHING at level 3 —
 * blanking the page would look like a broken file, and the breadcrumb above it
 * is the way back to a session that does have turns. */
function renderEmptyDetail(){
  ["scrubber", "changed", "alloc", "cache", "blocks", "growth"]
    .forEach(id => { document.getElementById(id).innerHTML = ""; });
  document.getElementById("h-agents").innerHTML = "";
  const host = document.getElementById("changed");
  host.appendChild(head("Turns", null));
  host.appendChild(el("p", "empty", "this session has no captured calls"));
}

/** Move the selection one turn along the SCOPED timeline. Stepping through VIEW
 * rather than through CALLS is what makes the arrow keys walk one agent's own
 * turns when a scope is active, instead of jumping to another agent's. */
function step(delta){
  const at = VIEW.indexOf(sel);
  const next = at + delta;
  if(at < 0 || next < 0 || next >= VIEW.length) return false;
  sel = VIEW[next];
  render();
  return true;
}

function boot(){
  // Where to open: the exporter decides from the project's shape and the
  // selectors the user passed (see `_start_level`), so a single-agent
  // single-session project lands on its detail view rather than on a one-row
  // listing, and `--session`/`--agent` preselect a level.
  const start = (PROJECT && PROJECT.start) || {level: 3, agent: null, session: null};
  level = start.level || 3;
  curAgent = start.agent != null ? start.agent : null;
  if(start.session != null) openSession(start.session);
  if(level === 3 && !CALLS.length && !MULTI_LEVEL){
    renderHeader();
    document.getElementById("app").innerHTML =
      "<section class=\\"panel\\"><p class=\\"empty\\">This run has no captured calls.</p></section>";
    return;
  }
  render();
  document.addEventListener("keydown", ev => {
    if(level !== 3) return;   // the arrows scrub turns; the listings have none
    if(ev.key === "ArrowLeft"){ if(step(-1)) ev.preventDefault(); }
    else if(ev.key === "ArrowRight"){ if(step(1)) ev.preventDefault(); }
  });
  const root = document.documentElement;
  document.getElementById("theme-btn").addEventListener("click", () => {
    const cur = root.getAttribute("data-theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const next = cur === "light" ? "dark" : cur === "dark" ? "light" : (prefersDark ? "light" : "dark");
    root.setAttribute("data-theme", next);
  });
}
boot();
</script>
</body>
</html>
'''


# The two placeholders `render_page` fills, matched together so ONE pass over
# the page consumes both. Two sequential `str.replace` calls could not: the
# title is substituted first, and `str.replace` replaces EVERY occurrence, so a
# project named `__CTXDIFF_DATA__` re-introduced the data marker inside
# `<title>` and the second pass filled that too — the title became the whole
# payload. One pass never revisits what it just wrote.
_MARKERS = re.compile(r"__CTXDIFF_(TITLE|DATA)__")


def render_page(project_title: str, data_json: str) -> str:
    """Fill the page's two markers and return the complete HTML document.
    `project_title` must already be HTML-escaped (it lands in `<title>`);
    `data_json` must already be `</`-escaped for the JSON island.

    Both markers are filled in a SINGLE pass (`count=2`, the number the page
    declares), so neither replacement can be re-scanned as part of the other —
    a project name is user text and may contain either marker verbatim. The
    replacement is a FUNCTION, so no backslash/group syntax in the title or the
    payload is expanded; the JS twin uses a function replacer for the same
    reason (there it is `$`-expansion) and stays byte-identical."""
    return _MARKERS.sub(
        lambda m: project_title if m.group(1) == "TITLE" else data_json,
        _PAGE, count=2)
