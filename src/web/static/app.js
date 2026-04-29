// Avis web dashboard — vanilla JS SPA entry point.
//
// Responsibilities:
//   1. Token bootstrap (URL ?token=… → localStorage → headers/MJPEG src)
//   2. Hash-based view router (#/live, #/recent)
//   3. Theme switcher (six themes, persisted)
//   4. Status polling for the agent-chip in the topbar
//   5. Mounts the live + recent view modules
//
// No build step. ES modules, native browser support.

import { mountLive, refreshLive } from "/static/views/live.js";
import { mountRecent } from "/static/views/recent.js";
import { mountTimeline, refreshTimeline } from "/static/views/timeline.js";
import { mountGallery, refreshGallery } from "/static/views/gallery.js";
import { mountDetail, openDetail } from "/static/views/detail.js";

// ── Token bootstrap ───────────────────────────────────────────────────────────

const TOKEN_KEY = "avis.token";

function readTokenFromUrl() {
  const params = new URLSearchParams(window.location.search);
  return params.get("token");
}

function persistToken(token) {
  try {
    localStorage.setItem(TOKEN_KEY, token);
  } catch {
    // Storage unavailable (private mode, quota) — fall back to in-memory only.
  }
}

function loadToken() {
  // Order: URL takes precedence (operator can rotate by reopening with a new
  // token), then cached token. Once loaded from URL, we strip the param so
  // the token doesn't linger in the address bar where it can be screen-shared
  // or accidentally bookmarked.
  const fromUrl = readTokenFromUrl();
  if (fromUrl) {
    persistToken(fromUrl);
    const url = new URL(window.location.href);
    url.searchParams.delete("token");
    window.history.replaceState({}, "", url.pathname + url.hash);
    return fromUrl;
  }
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

const token = loadToken();

// ── API client ────────────────────────────────────────────────────────────────

async function apiFetch(path, options = {}) {
  if (!token) {
    throw new Error("no-token");
  }
  const headers = new Headers(options.headers || {});
  headers.set("X-Avis-Token", token);
  headers.set("Accept", "application/json");
  const response = await fetch(path, { ...options, headers });
  if (response.status === 401) {
    // Token rejected — clear and force the user back to the gate.
    try { localStorage.removeItem(TOKEN_KEY); } catch {}
    throw new Error("unauthorized");
  }
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(`HTTP ${response.status}: ${detail || response.statusText}`);
  }
  return response.json();
}

function streamUrl(path) {
  // MJPEG / image endpoints can't set custom headers, so the token must
  // ride the query string. Caller invokes this for <img src=…>.
  return `${path}?token=${encodeURIComponent(token || "")}`;
}

// Expose to view modules via the API context object.
const api = {
  fetch: apiFetch,
  streamUrl,
};

// ── DOM helpers ───────────────────────────────────────────────────────────────

const $ = (sel) => document.querySelector(sel);

const els = {
  body: document.body,
  agentChipDot: $(".agent-chip-dot"),
  agentChipLabel: $("#agent-chip-label"),
  themeIcon: $("#theme-mode-icon"),
  toast: $("#toast"),
};

// ── Theme switcher ────────────────────────────────────────────────────────────

const THEME_PALETTES = ["warm", "pollen", "mono"];
const STORAGE_PALETTE = "avis.theme.palette";
const STORAGE_MODE = "avis.theme.mode";

const ICON_SUN = '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/>';
const ICON_MOON = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';

const themeState = {
  palette: localStorage.getItem(STORAGE_PALETTE) || "warm",
  mode: localStorage.getItem(STORAGE_MODE) || "light",
};
if (!THEME_PALETTES.includes(themeState.palette)) themeState.palette = "warm";
if (themeState.mode !== "light" && themeState.mode !== "dark") themeState.mode = "light";

function applyTheme() {
  els.body.dataset.theme = `${themeState.palette}-${themeState.mode}`;
  document.querySelectorAll("[data-theme-palette]").forEach((btn) => {
    btn.setAttribute("aria-pressed", String(btn.dataset.themePalette === themeState.palette));
  });
  if (els.themeIcon) {
    els.themeIcon.innerHTML = themeState.mode === "dark" ? ICON_MOON : ICON_SUN;
  }
}

function bindThemeSwitcher() {
  document.querySelectorAll("[data-theme-palette]").forEach((btn) => {
    btn.addEventListener("click", () => {
      themeState.palette = btn.dataset.themePalette;
      localStorage.setItem(STORAGE_PALETTE, themeState.palette);
      applyTheme();
    });
  });
  const modeBtn = document.querySelector("[data-theme-mode-toggle]");
  if (modeBtn) {
    modeBtn.addEventListener("click", () => {
      themeState.mode = themeState.mode === "dark" ? "light" : "dark";
      localStorage.setItem(STORAGE_MODE, themeState.mode);
      applyTheme();
    });
  }
}

// ── Toast ─────────────────────────────────────────────────────────────────────

let toastTimer = null;

function showToast(message, { tone = "info", durationMs = 3000 } = {}) {
  if (!els.toast) return;
  els.toast.textContent = message;
  els.toast.dataset.tone = tone;
  els.toast.dataset.visible = "true";
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    els.toast.dataset.visible = "false";
  }, durationMs);
}

// ── Router ────────────────────────────────────────────────────────────────────
//
// Hash routes:
//   #/live           — live preview + status sidebar
//   #/recent         — paginated card list
//   #/timeline       — horizontal SVG scrub
//   #/gallery        — grid of cropped thumbnails
//   #/detail/<id>    — single observation + image tabs

const VIEWS = ["live", "recent", "timeline", "gallery", "detail"];
const DEFAULT_VIEW = "live";

function parseHash() {
  const raw = window.location.hash.replace(/^#\//, "");
  if (!raw) return { view: DEFAULT_VIEW, id: null };
  const [view, id] = raw.split("/");
  if (!VIEWS.includes(view)) return { view: DEFAULT_VIEW, id: null };
  return { view, id: id || null };
}

function setRoute({ view, id }) {
  els.body.dataset.view = view;
  if (view === "live") refreshLive();
  if (view === "timeline") refreshTimeline();
  if (view === "gallery") refreshGallery();
  if (view === "detail" && id) openDetail(id);
}

function bindRouter() {
  window.addEventListener("hashchange", () => setRoute(parseHash()));
  document.querySelectorAll("[data-nav]").forEach((el) => {
    el.addEventListener("click", (event) => {
      const target = el.dataset.nav;
      if (VIEWS.includes(target) && target !== "detail") {
        event.preventDefault();
        const targetHash = `#/${target}`;
        if (window.location.hash !== targetHash) {
          window.location.hash = targetHash;
        } else {
          // Clicking the active link refreshes its data.
          setRoute({ view: target, id: null });
        }
      }
    });
  });
}

// ── Status polling for the agent chip ─────────────────────────────────────────

const AGENT_LABELS = { live: "Live", idle: "Idle", stale: "Stale", unknown: "—" };

async function pollStatus() {
  try {
    const status = await apiFetch("/api/status");
    const state = status.agent_status || "unknown";
    if (els.agentChipDot) els.agentChipDot.dataset.state = state;
    if (els.agentChipLabel) els.agentChipLabel.textContent = AGENT_LABELS[state] || "—";
    return status;
  } catch (err) {
    if (els.agentChipDot) els.agentChipDot.dataset.state = "unknown";
    if (els.agentChipLabel) els.agentChipLabel.textContent = "offline";
    return null;
  }
}

// ── Boot ──────────────────────────────────────────────────────────────────────

function showTokenGate() {
  els.body.dataset.view = "token-gate";
  showToast("No access token found. Open with ?token=… once.", { tone: "error", durationMs: 8000 });
}

function boot() {
  bindThemeSwitcher();
  applyTheme();

  if (!token) {
    showTokenGate();
    return;
  }

  bindRouter();

  // Mount view modules. They take an api object + the toast function so they
  // don't have to know how auth works.
  const ctx = { api, toast: showToast };
  mountLive(ctx);
  mountRecent(ctx);
  mountTimeline(ctx);
  mountGallery(ctx);
  mountDetail(ctx);

  setRoute(parseHash());

  // Initial agent-chip update; then poll every 30 s. Cheap call (cached store).
  pollStatus();
  setInterval(pollStatus, 30000);
}

document.addEventListener("DOMContentLoaded", boot);
