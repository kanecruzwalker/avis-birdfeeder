// Detail view — single observation with three image tabs + metadata.
//
// Reached via openDetail(id) from the router when the hash is
// #/detail/<id>. Image tabs (cropped/annotated/full) load lazily and
// fall back to a "not available" message on 404 (the agent only saves
// the annotated variant when YOLO produced a box, and full + annotated
// are dispatched-only).

const $ = (sel) => document.querySelector(sel);

let api = null;
let toast = null;

const state = {
  observationId: null,
  observation: null,
  variant: "cropped",
};

function fmtTimestamp(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "medium",
  });
}

function fmtFraction(value) {
  if (typeof value !== "number") return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function renderMetaRow(label, value, { mono = false } = {}) {
  const row = document.createElement("div");
  row.className = "detail-meta-row";
  const k = document.createElement("span");
  k.className = "muted";
  k.textContent = label;
  const v = document.createElement("strong");
  if (mono) v.classList.add("mono");
  v.textContent = value;
  row.append(k, v);
  return row;
}

function renderMetadata(obs) {
  const meta = $("#detail-meta");
  if (!meta) return;
  meta.innerHTML = "";

  const heading = document.createElement("div");
  heading.className = "detail-meta-heading";
  const code = document.createElement("h2");
  code.textContent = obs.species_code;
  const name = document.createElement("p");
  name.className = "muted";
  name.textContent = `${obs.common_name || ""}${obs.scientific_name ? ` · ${obs.scientific_name}` : ""}`;
  heading.append(code, name);
  meta.appendChild(heading);

  const status = document.createElement("div");
  status.className = "detail-status-row";
  const dispatchedPill = document.createElement("span");
  dispatchedPill.className = "obs-meta-pill";
  if (!obs.dispatched) dispatchedPill.dataset.suppressed = "true";
  dispatchedPill.textContent = obs.dispatched ? "dispatched" : (obs.gate_reason || "suppressed");
  status.appendChild(dispatchedPill);
  if (obs.detection_mode) {
    const modePill = document.createElement("span");
    modePill.className = "obs-meta-pill";
    modePill.textContent = obs.detection_mode;
    status.appendChild(modePill);
  }
  meta.appendChild(status);

  const rows = [
    ["Confidence", fmtFraction(obs.fused_confidence)],
    ["Timestamp", fmtTimestamp(obs.timestamp)],
  ];
  if (obs.audio_result) {
    rows.push([
      "Audio",
      `${obs.audio_result.species_code} ${fmtFraction(obs.audio_result.confidence)}`,
    ]);
  }
  if (obs.visual_result) {
    rows.push([
      "Visual cam0",
      `${obs.visual_result.species_code} ${fmtFraction(obs.visual_result.confidence)}`,
    ]);
  }
  if (obs.visual_result_2) {
    rows.push([
      "Visual cam1",
      `${obs.visual_result_2.species_code} ${fmtFraction(obs.visual_result_2.confidence)}`,
    ]);
  }
  if (obs.detection_box) {
    rows.push(["Box (x1,y1,x2,y2)", obs.detection_box.join(", ")]);
  }
  if (obs.estimated_depth_cm != null) {
    rows.push(["Depth (cm)", obs.estimated_depth_cm.toFixed(1)]);
  }
  if (obs.estimated_size_cm != null) {
    rows.push(["Size (cm)", obs.estimated_size_cm.toFixed(1)]);
  }

  for (const [label, value] of rows) {
    meta.appendChild(renderMetaRow(label, value));
  }
}

function setActiveTab(variant) {
  document.querySelectorAll(".image-tab").forEach((btn) => {
    btn.setAttribute("aria-selected", String(btn.dataset.variant === variant));
  });
}

function loadImage(variant) {
  state.variant = variant;
  setActiveTab(variant);
  const img = $("#detail-image");
  const fallback = $("#detail-image-fallback");
  if (!img || !state.observationId) return;

  if (fallback) fallback.hidden = true;
  img.hidden = false;
  img.alt = `${state.observation?.species_code || ""} ${variant}`;
  img.src = api.streamUrl(`/api/observations/${state.observationId}/image/${variant}`);
  img.onerror = () => {
    img.hidden = true;
    if (fallback) fallback.hidden = false;
  };
  img.onload = () => {
    if (fallback) fallback.hidden = true;
  };
}

function bindTabs() {
  document.querySelectorAll(".image-tab").forEach((btn) => {
    btn.addEventListener("click", () => loadImage(btn.dataset.variant));
  });
}

async function fetchObservation(id) {
  const empty = $("#detail-empty");
  const idLabel = $("#detail-id");
  if (idLabel) idLabel.textContent = id;
  if (empty) {
    empty.textContent = "Loading…";
    empty.hidden = false;
  }

  let obs;
  try {
    obs = await api.fetch(`/api/observations/${encodeURIComponent(id)}`);
  } catch (err) {
    if (err.message === "no-token" || err.message === "unauthorized") return;
    if (empty) {
      empty.textContent = "Observation not found.";
      empty.hidden = false;
    }
    toast?.(`Could not load observation: ${err.message}`, { tone: "error" });
    return;
  }

  if (empty) empty.hidden = true;
  state.observation = obs;
  renderMetadata(obs);
  loadImage("cropped");
}

export function mountDetail(ctx) {
  api = ctx.api;
  toast = ctx.toast;
  bindTabs();
}

export function openDetail(id) {
  state.observationId = id;
  state.variant = "cropped";
  fetchObservation(id);
}
