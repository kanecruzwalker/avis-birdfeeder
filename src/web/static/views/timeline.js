// Timeline view — horizontal SVG scrub with one marker per observation.
//
// X-axis = time, single row of markers. Click a marker → detail view.
// Filter bar shares the same field names as gallery (window, species,
// suppressed) so the two views feel like the same surface.

const $ = (sel) => document.querySelector(sel);

let api = null;
let toast = null;

const WINDOWS = {
  "1h": 60 * 60 * 1000,
  "6h": 6 * 60 * 60 * 1000,
  "24h": 24 * 60 * 60 * 1000,
  "7d": 7 * 24 * 60 * 60 * 1000,
  "30d": 30 * 24 * 60 * 60 * 1000,
};

const state = {
  window: "24h",
  species: "",
  showSuppressed: false,
};

const SVG_NS = "http://www.w3.org/2000/svg";
// Pad above + below so labels don't clip.
const PAD_X = 14;
const PAD_TOP = 28;
const PAD_BOTTOM = 36;
const ROW_Y = 56;

function confidenceBand(value) {
  if (typeof value !== "number") return "low";
  if (value >= 0.7) return "high";
  if (value >= 0.4) return "mid";
  return "low";
}

function fmtTime(date) {
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function buildSvg(observations, fromMs, toMs) {
  const svg = $("#timeline-svg");
  if (!svg) return;
  svg.innerHTML = "";

  const width = svg.clientWidth || 800;
  const height = ROW_Y + PAD_BOTTOM + 20;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("height", String(height));

  const innerLeft = PAD_X;
  const innerRight = width - PAD_X;
  const innerWidth = innerRight - innerLeft;
  const range = Math.max(1, toMs - fromMs);

  const xFor = (ms) => innerLeft + ((ms - fromMs) / range) * innerWidth;

  // Axis line
  const axis = document.createElementNS(SVG_NS, "line");
  axis.setAttribute("x1", innerLeft);
  axis.setAttribute("x2", innerRight);
  axis.setAttribute("y1", ROW_Y);
  axis.setAttribute("y2", ROW_Y);
  axis.setAttribute("class", "timeline-axis");
  svg.appendChild(axis);

  // Endpoint labels
  const fromLabel = document.createElementNS(SVG_NS, "text");
  fromLabel.setAttribute("x", innerLeft);
  fromLabel.setAttribute("y", ROW_Y + 22);
  fromLabel.setAttribute("class", "timeline-label");
  fromLabel.textContent = fmtTime(new Date(fromMs));
  svg.appendChild(fromLabel);

  const toLabel = document.createElementNS(SVG_NS, "text");
  toLabel.setAttribute("x", innerRight);
  toLabel.setAttribute("y", ROW_Y + 22);
  toLabel.setAttribute("text-anchor", "end");
  toLabel.setAttribute("class", "timeline-label");
  toLabel.textContent = fmtTime(new Date(toMs));
  svg.appendChild(toLabel);

  // Markers
  for (const obs of observations) {
    const ts = new Date(obs.timestamp).getTime();
    if (isNaN(ts)) continue;
    const x = xFor(ts);

    const a = document.createElementNS(SVG_NS, "a");
    a.setAttribute("href", `#/detail/${obs.id}`);
    a.setAttribute("class", "timeline-marker-link");

    const circle = document.createElementNS(SVG_NS, "circle");
    circle.setAttribute("cx", x);
    circle.setAttribute("cy", ROW_Y);
    circle.setAttribute("r", obs.dispatched ? 6 : 4);
    circle.setAttribute("class", "timeline-marker");
    circle.dataset.band = confidenceBand(obs.fused_confidence);
    circle.dataset.suppressed = String(!obs.dispatched);

    const title = document.createElementNS(SVG_NS, "title");
    const conf = `${(obs.fused_confidence * 100).toFixed(0)}%`;
    const sup = obs.dispatched ? "" : " · suppressed";
    title.textContent = `${obs.species_code} ${conf} · ${fmtTime(new Date(ts))}${sup}`;
    circle.appendChild(title);

    a.appendChild(circle);
    svg.appendChild(a);
  }
}

async function fetchAndRender() {
  const stage = $("#timeline-stage");
  const empty = $("#timeline-empty");
  const count = $("#timeline-count");
  if (!stage) return;

  const now = Date.now();
  const fromMs = now - (WINDOWS[state.window] || WINDOWS["24h"]);
  const toMs = now;

  const params = new URLSearchParams();
  params.set("from", new Date(fromMs).toISOString());
  params.set("to", new Date(toMs).toISOString());
  params.set("limit", "500"); // route clamps; we want everything in window
  if (state.species) params.set("species", state.species.toUpperCase());
  if (state.showSuppressed) params.set("dispatched", "all");

  let body;
  try {
    body = await api.fetch(`/api/observations?${params.toString()}`);
  } catch (err) {
    if (err.message === "no-token" || err.message === "unauthorized") return;
    toast?.(`Failed to load timeline: ${err.message}`, { tone: "error" });
    return;
  }

  if (count) count.textContent = `${body.count} detection${body.count === 1 ? "" : "s"}`;
  if (empty) empty.hidden = body.count > 0;
  buildSvg(body.items, fromMs, toMs);
}

function bindFilters() {
  const bar = document.querySelector('[data-filter-bar="timeline"]');
  if (!bar) return;
  bar.querySelectorAll("[data-filter]").forEach((el) => {
    const field = el.dataset.filter;
    el.addEventListener("change", () => {
      if (field === "window") state.window = el.value;
      else if (field === "species") state.species = el.value.trim();
      else if (field === "suppressed") state.showSuppressed = el.checked;
      fetchAndRender();
    });
  });
}

export function mountTimeline(ctx) {
  api = ctx.api;
  toast = ctx.toast;
  bindFilters();
}

export function refreshTimeline() {
  if (api) fetchAndRender();
}
