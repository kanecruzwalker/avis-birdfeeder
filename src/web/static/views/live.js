// Live view — MJPEG <img> + meta sidebar.
//
// The MJPEG stream is wired via the <img>'s src attribute so the browser's
// native multipart/x-mixed-replace handling does the work. Refreshing
// just means re-fetching /api/status for the meta sidebar; the stream
// itself self-heals as long as the connection stays open.

const $ = (sel) => document.querySelector(sel);

let api = null;
let toast = null;

function fmtUptime(seconds) {
  if (typeof seconds !== "number" || !isFinite(seconds) || seconds < 0) return "—";
  const s = Math.floor(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
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

function fmtConfidence(value) {
  if (typeof value !== "number") return "";
  return ` ${(value * 100).toFixed(0)}%`;
}

async function pullStatus() {
  try {
    const status = await api.fetch("/api/status");
    const last = status.last_dispatched_at
      ? `${fmtTimestamp(status.last_dispatched_at)}`
      : "—";
    setText("#live-last-detection", last);
    setText("#live-total", String(status.total_observations ?? "—"));
    setText("#live-dispatched", String(status.total_dispatched ?? "—"));
    setText("#live-mode", status.current_mode || "—");
    setText("#live-uptime", fmtUptime(status.uptime_seconds));
  } catch (err) {
    if (err.message === "no-token" || err.message === "unauthorized") return;
    // /api/status is fail-soft; one outage shouldn't yell at the user.
  }
}

function setText(sel, value) {
  const el = $(sel);
  if (el) el.textContent = value;
}

function bindStream() {
  const img = $("#live-stream");
  const overlay = $("#live-overlay");
  const overlayText = $("#live-overlay-text");
  if (!img) return;

  img.src = api.streamUrl("/api/stream");
  if (overlay) overlay.hidden = false;
  if (overlayText) overlayText.textContent = "Waiting for first frame…";

  img.addEventListener("load", () => {
    if (overlay) overlay.hidden = true;
  }, { once: true });

  img.addEventListener("error", () => {
    if (overlay) overlay.hidden = false;
    if (overlayText) overlayText.textContent = "Stream unavailable. The agent may not be publishing yet.";
  });
}

export function mountLive(ctx) {
  api = ctx.api;
  toast = ctx.toast;
  bindStream();
  pullStatus();
}

export function refreshLive() {
  // Re-trigger the stream src in case the connection died while the user
  // was on another tab. Cheap if the stream is already alive — the browser
  // dedupes identical srcs.
  if (api) bindStream();
  if (api) pullStatus();
}
