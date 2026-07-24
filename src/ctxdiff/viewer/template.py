"""The dashboard page as Python string constants — HTML shell, inline CSS, and
inline JS, all in one module so packaging needs no data files. Everything is
self-contained: no external stylesheet, script, font, or image, and no URL of
any kind (an SVG namespace URL would trip the "no http(s)" self-containment
guarantee, so the growth chart is built as an HTML string and handed to
`innerHTML`, where the HTML parser namespaces `<svg>` automatically — no
`createElementNS` and no xmlns needed).

`render_page` fills two markers: `__CTXDIFF_TITLE__` (the already-escaped
`<title>` text) and `__CTXDIFF_DATA__` (the JSON island). Title is substituted
first so a value in the data can never be mistaken for the title marker.

The runtime contract with export.py: the JSON island is read back with
`.textContent` and parsed once; all BLOCK TEXT is rendered with `.textContent`
(never `.innerHTML`), so untrusted trace data can never become live markup.
Only static chrome and numeric-derived SVG use `.innerHTML`."""
from __future__ import annotations

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
/* Agent-colored underline strip on each turn bar, and dimming for bars whose
   agent isn't the active filter. Placed AFTER .bar.sel so a dim wins on equal
   specificity. */
.bar-underline{position:absolute; left:2px; right:2px; bottom:0; height:3px;
  border-radius:2px}
.bar.dim{opacity:.2}

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
</main>

<script id="ctxdiff-data" type="application/json">__CTXDIFF_DATA__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("ctxdiff-data").textContent);
const CALLS = DATA.calls || [];
let sel = 0;

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
const KNOWN_LABELS = {system:1, tool_schema:1, rag:1, history:1, user:1, tool_output:1};
function labelColor(label){ return "var(--c-" + (KNOWN_LABELS[label] ? label : "unknown") + ")"; }
function head(title, meta){
  const h = el("div", "panel-head");
  h.appendChild(el("h2", null, title));
  if(meta != null) h.appendChild(el("span", "panel-meta", meta));
  return h;
}
function dot(label){ const d = el("span", "dot"); d.style.background = labelColor(label); return d; }

// --- agents ------------------------------------------------------------------
// Per-agent color is assigned by order of first appearance from a fixed
// categorical palette, cycled when a run has more agents than colors. Agent
// NAMES are NEVER interpolated into CSS — only these fixed hex values reach a
// style property — so a hostile agent name cannot inject styles. Every agent
// name that reaches the DOM does so via el()/textContent.
const AGENTS = (DATA.stats && DATA.stats.agents) || [];
const AGENT_MULTI = AGENTS.length > 1;
const AGENT_PALETTE = ["#3987e5","#d95926","#199e70","#c98500","#d55181",
                       "#9085e9","#008300","#c0498a"];
const AGENT_COLOR = {};
AGENTS.forEach((a, i) => { AGENT_COLOR[a.name] = AGENT_PALETTE[i % AGENT_PALETTE.length]; });
let agentFilter = null;   // active agent-chip filter (dims other agents' bars)
function agentKey(call){ return call && call.agent != null ? call.agent : "(unlabeled)"; }
function agentColor(name){ return AGENT_COLOR[name] || "var(--c-unknown)"; }
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
  if(!AGENT_MULTI) return;   // one (or zero) agent: no chips, nothing to filter
  AGENTS.forEach(a => {
    const chip = el("button", "agent-chip" + (agentFilter === a.name ? " active" : ""));
    const d = el("span", "agent-dot"); d.style.background = agentColor(a.name);
    chip.appendChild(d);
    chip.appendChild(el("span", "agent-name", a.name));      // textContent — safe
    chip.appendChild(el("span", "agent-count", "\\u00b7 " + a.calls));
    // Provider in/out on the tooltip (title attribute — value, never parsed as
    // markup); present only when this agent reported usage.
    const uba = (DATA.stats.usage || {}).by_agent || {};
    const io = uba[a.name];
    if(io) chip.setAttribute("title", a.name + " \\u00b7 in " + fmt(io[0]) +
                             " \\u00b7 out " + fmt(io[1]));
    chip.setAttribute("aria-pressed", agentFilter === a.name ? "true" : "false");
    chip.addEventListener("click", () => {
      agentFilter = (agentFilter === a.name) ? null : a.name;  // toggle
      render();
    });
    host.appendChild(chip);
  });
}

// --- header ------------------------------------------------------------------
function renderHeader(){
  const r = DATA.run || {};
  document.getElementById("h-project").textContent = r.project || "";
  document.title = "ctxdiff \\u2014 " + (r.project || "");
  const total = (DATA.stats.context_growth || []).reduce((a,b)=>a+b, 0);
  const dedup = DATA.stats.distinct_blocks + " distinct blocks / " +
                DATA.stats.total_block_refs + " references";
  // Provider-usage rollup, shown only when at least one call reported usage —
  // never fabricate an "in 0 / out 0" from a run with no provider numbers.
  const u = DATA.stats.usage || {};
  const cov = u.coverage || [0, 0];
  const items = [ r.provider || "?", (r.models || []).join(", ") || "?",
                  r.started_at || "?", CALLS.length + " turns",
                  fmt(total) + " tokens" ];
  if(cov[0] > 0){
    items.push("in " + fmt(u.input) + " \\u00b7 out " + fmt(u.output) +
               " (" + cov[0] + "/" + cov[1] + " reported)");
  }
  items.push(dedup);
  const box = document.getElementById("h-meta");
  box.innerHTML = "";
  items.forEach((m, i) => {
    if(i) box.appendChild(el("span", "sep", "\\u00b7"));
    box.appendChild(el("span", "meta-item", m));
  });
}

// --- scrubber ----------------------------------------------------------------
function renderScrubber(){
  const strip = document.getElementById("scrubber");
  strip.innerHTML = "";
  const growth = DATA.stats.context_growth || [];
  const max = Math.max(1, ...growth);
  CALLS.forEach((c, i) => {
    const tok = growth[i] || 0;
    const b = document.createElement("button");
    b.className = "bar" + (i === sel ? " sel" : "") + (c.error ? " err" : "");
    b.style.height = (12 + (tok / max) * 104).toFixed(1) + "px";
    b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", i === sel ? "true" : "false");
    b.setAttribute("aria-label", "turn " + c.seq + " \\u2014 " + fmt(tok) +
                   " tokens" + (c.error ? " (error)" : ""));
    b.addEventListener("click", () => { sel = i; render(); });
    if(AGENT_MULTI){
      const key = agentKey(c);
      const u = el("span", "bar-underline"); u.style.background = agentColor(key);
      b.appendChild(u);
      if(agentFilter && agentFilter !== key) b.classList.add("dim");
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
  let d = DATA.diffs[sel-1];
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
  const t = DATA.tokens.calls[sel];
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

  const bloat = DATA.tokens.bloat;
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
  const cc = DATA.cache;
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
  const g = DATA.stats.context_growth || [];
  const n = g.length;
  const W = Math.max(320, n * 64), H = 150, pad = 26;
  const max = Math.max(1, ...g);
  const xat = i => n <= 1 ? W / 2 : pad + i * (W - 2 * pad) / (n - 1);
  const yat = v => H - pad - (v / max) * (H - 2 * pad);
  const pts = g.map((v, i) => xat(i).toFixed(1) + "," + yat(v).toFixed(1)).join(" ");
  let svg = "<svg viewBox=\\"0 0 " + W + " " + H + "\\" preserveAspectRatio=\\"xMidYMid meet\\" role=\\"img\\" aria-label=\\"context tokens per turn\\">";
  if(n > 0){
    const area = pad.toFixed(1) + "," + (H - pad) + " " + pts + " " +
                 xat(n - 1).toFixed(1) + "," + (H - pad);
    svg += "<polygon class=\\"area\\" points=\\"" + area + "\\"></polygon>";
    svg += "<polyline class=\\"line\\" points=\\"" + pts + "\\"></polyline>";
    g.forEach((v, i) => {
      svg += "<circle class=\\"" + (i === sel ? "dot-sel" : "dot") + "\\" cx=\\"" +
             xat(i).toFixed(1) + "\\" cy=\\"" + yat(v).toFixed(1) + "\\" r=\\"" +
             (i === sel ? 4.5 : 3) + "\\"></circle>";
    });
  }
  svg += "</svg>";
  const box = el("div", "chart-wrap");
  box.innerHTML = svg;   // numeric-derived markup only; no trace text here
  host.appendChild(box);
}

// --- orchestration -----------------------------------------------------------
function render(){
  renderAgents();
  renderScrubber();
  renderChanged();
  renderTokens();
  renderCache();
  renderBlocks();
  renderGrowth();
}

function boot(){
  renderHeader();
  if(!CALLS.length){
    document.getElementById("app").innerHTML =
      "<section class=\\"panel\\"><p class=\\"empty\\">This run has no captured calls.</p></section>";
    return;
  }
  render();
  document.addEventListener("keydown", ev => {
    if(ev.key === "ArrowLeft" && sel > 0){ sel--; render(); ev.preventDefault(); }
    else if(ev.key === "ArrowRight" && sel < CALLS.length - 1){ sel++; render(); ev.preventDefault(); }
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


def render_page(project_title: str, data_json: str) -> str:
    """Fill the page's two markers and return the complete HTML document.
    `project_title` must already be HTML-escaped (it lands in `<title>`);
    `data_json` must already be `</`-escaped for the JSON island. Title is
    substituted before data so nothing in the data can shadow the title
    marker."""
    return (_PAGE
            .replace("__CTXDIFF_TITLE__", project_title)
            .replace("__CTXDIFF_DATA__", data_json))
