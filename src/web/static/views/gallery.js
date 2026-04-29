// Gallery view — grid of cropped thumbnails.
//
// Same filter bar as timeline (window / species / suppressed). Pagination
// via /api/observations cursor: "Load more" appends another page.

const $ = (sel) => document.querySelector(sel);

let api = null;
let toast = null;

const PAGE_LIMIT = 60;

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
  cursor: null,
  rendered: 0,
};

function confidenceBand(value) {
  if (typeof value !== "number") return "low";
  if (value >= 0.7) return "high";
  if (value >= 0.4) return "mid";
  return "low";
}

function fmtTimestamp(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function renderTile(obs) {
  const tile = document.createElement("a");
  tile.className = "gallery-tile";
  tile.dataset.suppressed = String(!obs.dispatched);
  tile.dataset.band = confidenceBand(obs.fused_confidence);
  tile.href = `#/detail/${obs.id}`;

  const thumb = document.createElement("div");
  thumb.className = "gallery-thumb";
  if (obs.image_path) {
    const img = document.createElement("img");
    img.alt = `${obs.species_code} cropped`;
    img.loading = "lazy";
    img.src = api.streamUrl(`/api/observations/${obs.id}/image/cropped`);
    img.addEventListener("error", () => {
      img.remove();
      thumb.textContent = "no image";
    }, { once: true });
    thumb.appendChild(img);
  } else {
    thumb.textContent = "no image";
  }

  const overlay = document.createElement("div");
  overlay.className = "gallery-overlay";

  const code = document.createElement("span");
  code.className = "gallery-species";
  code.textContent = obs.species_code;
  overlay.appendChild(code);

  const conf = document.createElement("span");
  conf.className = "gallery-confidence";
  conf.textContent = `${(obs.fused_confidence * 100).toFixed(0)}%`;
  overlay.appendChild(conf);

  const ts = document.createElement("span");
  ts.className = "gallery-timestamp";
  ts.textContent = fmtTimestamp(obs.timestamp);
  overlay.appendChild(ts);

  tile.append(thumb, overlay);
  return tile;
}

async function fetchPage(append = false) {
  const grid = $("#gallery-grid");
  const empty = $("#gallery-empty");
  const loadMore = $("#gallery-load-more");
  const count = $("#gallery-count");
  if (!grid) return;

  const now = Date.now();
  const fromMs = now - (WINDOWS[state.window] || WINDOWS["24h"]);

  const params = new URLSearchParams();
  params.set("from", new Date(fromMs).toISOString());
  params.set("to", new Date(now).toISOString());
  params.set("limit", String(PAGE_LIMIT));
  if (state.species) params.set("species", state.species.toUpperCase());
  if (state.showSuppressed) params.set("dispatched", "all");
  if (state.cursor && append) params.set("cursor", state.cursor);

  let body;
  try {
    body = await api.fetch(`/api/observations?${params.toString()}`);
  } catch (err) {
    if (err.message === "no-token" || err.message === "unauthorized") return;
    toast?.(`Failed to load gallery: ${err.message}`, { tone: "error" });
    return;
  }

  if (!append) {
    grid.innerHTML = "";
    state.rendered = 0;
  }

  for (const obs of body.items) {
    grid.appendChild(renderTile(obs));
    state.rendered += 1;
  }

  if (state.rendered === 0) {
    if (empty) {
      empty.textContent = "No detections in this window.";
      grid.appendChild(empty);
    }
    if (count) count.textContent = "0";
  } else if (count) {
    count.textContent = `${state.rendered}${body.next_cursor ? "+" : ""} detection${state.rendered === 1 ? "" : "s"}`;
  }

  state.cursor = body.next_cursor;
  if (loadMore) loadMore.hidden = !state.cursor;
}

function bindFilters() {
  const bar = document.querySelector('[data-filter-bar="gallery"]');
  if (!bar) return;
  bar.querySelectorAll("[data-filter]").forEach((el) => {
    const field = el.dataset.filter;
    el.addEventListener("change", () => {
      if (field === "window") state.window = el.value;
      else if (field === "species") state.species = el.value.trim();
      else if (field === "suppressed") state.showSuppressed = el.checked;
      state.cursor = null;
      fetchPage(false);
    });
  });
  const loadMore = $("#gallery-load-more");
  if (loadMore) {
    loadMore.addEventListener("click", () => fetchPage(true));
  }
}

export function mountGallery(ctx) {
  api = ctx.api;
  toast = ctx.toast;
  bindFilters();
}

export function refreshGallery() {
  if (api) {
    state.cursor = null;
    fetchPage(false);
  }
}
