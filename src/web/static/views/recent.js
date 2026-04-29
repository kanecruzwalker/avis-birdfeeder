// Recent view — paginated list of observations, newest first.
//
// Cards distinguish dispatched vs suppressed visually. The "Show suppressed"
// toggle re-fetches with dispatched=null (both) instead of the default
// dispatched=true. Pagination is cursor-based via /api/observations.

const $ = (sel) => document.querySelector(sel);

let api = null;
let toast = null;

const PAGE_LIMIT = 25;

const state = {
  showSuppressed: false,
  cursor: null,
  rendered: 0,
};

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

function confidenceBand(value) {
  if (typeof value !== "number") return "low";
  if (value >= 0.7) return "high";
  if (value >= 0.4) return "mid";
  return "low";
}

function obsId(obs) {
  // ID convention from src/web/observation_store.py: timestamp formatted
  // as YYYYMMDDTHHMMSSffffff (UTC). The list endpoint already includes it.
  return obs.id;
}

function renderCard(obs) {
  const card = document.createElement("a");
  card.className = "obs-card";
  card.dataset.suppressed = String(!obs.dispatched);
  card.dataset.id = obsId(obs);
  // No detail view yet (PR 7) — anchor to itself, will hook up later.
  card.href = "#/recent";

  const thumb = document.createElement("div");
  thumb.className = "obs-thumb";
  if (obs.image_path) {
    const img = document.createElement("img");
    img.alt = `${obs.species_code} cropped`;
    img.loading = "lazy";
    img.src = api.streamUrl(`/api/observations/${obsId(obs)}/image/cropped`);
    img.addEventListener("error", () => {
      // Image missing on disk → fall back to placeholder text.
      img.remove();
      thumb.textContent = "no image";
    }, { once: true });
    thumb.appendChild(img);
  } else {
    thumb.textContent = "no image";
  }

  const body = document.createElement("div");
  body.className = "obs-body";

  const species = document.createElement("div");
  species.className = "obs-species";
  const code = document.createElement("span");
  code.className = "obs-species-code";
  code.textContent = obs.species_code;
  const name = document.createElement("span");
  name.className = "obs-species-name";
  name.textContent = obs.common_name || "";
  species.append(code, name);

  const meta = document.createElement("div");
  meta.className = "obs-meta";

  const ts = document.createElement("span");
  ts.textContent = fmtTimestamp(obs.timestamp);
  meta.appendChild(ts);

  if (obs.detection_mode) {
    const mode = document.createElement("span");
    mode.className = "obs-meta-pill";
    mode.textContent = obs.detection_mode;
    meta.appendChild(mode);
  }
  if (!obs.dispatched) {
    const sup = document.createElement("span");
    sup.className = "obs-meta-pill";
    sup.dataset.suppressed = "true";
    sup.textContent = obs.gate_reason || "suppressed";
    meta.appendChild(sup);
  }

  body.append(species, meta);

  const conf = document.createElement("div");
  conf.className = "obs-confidence";
  conf.dataset.band = confidenceBand(obs.fused_confidence);
  const value = document.createElement("span");
  value.className = "obs-confidence-value";
  value.textContent = `${(obs.fused_confidence * 100).toFixed(0)}%`;
  const bar = document.createElement("div");
  bar.className = "obs-confidence-bar";
  const fill = document.createElement("div");
  fill.className = "obs-confidence-bar-fill";
  fill.style.width = `${Math.max(0, Math.min(100, obs.fused_confidence * 100)).toFixed(0)}%`;
  bar.appendChild(fill);
  conf.append(value, bar);

  card.append(thumb, body, conf);
  return card;
}

async function fetchPage(append = false) {
  const params = new URLSearchParams();
  params.set("limit", String(PAGE_LIMIT));
  if (state.showSuppressed) {
    params.set("dispatched", "all");
  }
  if (state.cursor) {
    params.set("cursor", state.cursor);
  }

  let body;
  try {
    body = await api.fetch(`/api/observations?${params.toString()}`);
  } catch (err) {
    if (err.message === "no-token" || err.message === "unauthorized") return;
    toast?.(`Failed to load observations: ${err.message}`, { tone: "error" });
    return;
  }

  const list = $("#recent-list");
  const empty = $("#recent-empty");
  const loadMore = $("#recent-load-more");
  if (!list) return;

  if (!append) {
    list.innerHTML = "";
    state.rendered = 0;
  }

  for (const obs of body.items) {
    list.appendChild(renderCard(obs));
    state.rendered += 1;
  }

  if (state.rendered === 0) {
    if (empty) {
      empty.textContent = state.showSuppressed
        ? "No observations yet."
        : "No dispatched observations yet. Toggle 'Show suppressed' to see all.";
      list.appendChild(empty);
    }
  }

  state.cursor = body.next_cursor;
  if (loadMore) loadMore.hidden = !state.cursor;
}

function bindControls() {
  const toggle = $("#recent-show-suppressed");
  const loadMore = $("#recent-load-more");

  if (toggle) {
    toggle.addEventListener("change", () => {
      state.showSuppressed = toggle.checked;
      state.cursor = null;
      fetchPage(false);
    });
  }
  if (loadMore) {
    loadMore.addEventListener("click", () => {
      fetchPage(true);
    });
  }
}

export function mountRecent(ctx) {
  api = ctx.api;
  toast = ctx.toast;
  bindControls();
  fetchPage(false);
}
