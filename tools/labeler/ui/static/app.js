// Avis labeling-assistant review UI — vanilla JS SPA.
//
// Three concerns:
//   1. API client wrappers (fetch + auth token + JSON + error handling)
//   2. View renderers (summary, review, verified)
//   3. Input handling (keyboard, touch, click, theme switcher)
//
// Single file because the surface area is small. Splitting into modules
// would add a build step for zero benefit.

(function () {
  "use strict";

  // ── Config & state ──────────────────────────────────────────────────

  const TOKEN = window.AVIS.token;
  const INITIAL_VIEW = window.AVIS.initialView;
  const SPECIES_FILTER = window.AVIS.speciesFilter;

  const CALT_SUSPECT_BUCKETS = new Set(["MOCH", "WREN"]);
  const THEME_PALETTES = ["warm", "pollen", "mono"];
  const THEME_MODES = ["light", "dark"];
  const STORAGE_KEY_PALETTE = "avis.theme.palette";
  const STORAGE_KEY_MODE = "avis.theme.mode";

  const state = {
    view: INITIAL_VIEW,
    speciesFilter: SPECIES_FILTER,
    speciesList: { known: [], sentinels: [], all: [] },
    summary: { species: [], coverage: null },
    current: null,
    pendingPayload: null,
    themePalette: "warm",
    themeMode: "light",
  };

  // ── DOM lookups ─────────────────────────────────────────────────────

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const els = {
    body: document.body,
    coverage: $("#topbar-coverage"),
    themeIcon: $("#theme-mode-icon"),
    // summary
    summaryOverall: $("#summary-overall"),
    bucketList: $("#bucket-list"),
    bucketEmpty: $("#bucket-empty"),
    btnReviewAll: $("#btn-review-all"),
    // review
    reviewBack: $("#review-back"),
    reviewSkip: $("#review-skip"),
    reviewProgress: $("#review-progress"),
    reviewImageWrap: $("#review-image-wrap"),
    reviewImage: $("#review-image"),
    reviewImageLoading: $("#review-image-loading"),
    prelabelSpecies: $("#prelabel-species"),
    prelabelConfidence: $("#prelabel-confidence"),
    prelabelReasoning: $("#prelabel-reasoning"),
    audioHint: $("#audio-hint"),
    alreadyVerified: $("#already-verified"),
    btnConfirm: $("#btn-confirm"),
    btnConfirmSpecies: $("#btn-confirm-species"),
    btnSkip: $("#btn-skip"),
    quickCorrectGrid: $("#quick-correct-grid"),
    otherInput: $("#other-input"),
    otherCodeInput: $("#other-code-input"),
    otherNotesInput: $("#other-notes-input"),
    btnConfirmOther: $("#btn-confirm-other"),
    btnCancelOther: $("#btn-cancel-other"),
    reviewEmpty: $("#review-empty"),
    // verified
    verifiedFilter: $("#verified-filter"),
    verifiedCount: $("#verified-count"),
    verifiedTable: $("#verified-table"),
    verifiedTbody: $("#verified-tbody"),
    verifiedEmpty: $("#verified-empty"),
    // modal
    conflictModal: $("#conflict-modal"),
    conflictExistingSpecies: $("#conflict-existing-species"),
    conflictExistingWhen: $("#conflict-existing-when"),
    conflictNewSpecies: $("#conflict-new-species"),
    conflictCancel: $("#conflict-cancel"),
    conflictOverwrite: $("#conflict-overwrite"),
    // toast
    toast: $("#toast"),
  };

  // ── Theme switcher ──────────────────────────────────────────────────

  // SVG paths for sun (light) and moon (dark) icons. Swapped on toggle.
  const ICON_SUN = '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/>';
  const ICON_MOON = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';

  function applyTheme() {
    const themeName = `${state.themePalette}-${state.themeMode}`;
    els.body.dataset.theme = themeName;

    // Update palette buttons
    $$('[data-theme-palette]').forEach((btn) => {
      const isActive = btn.dataset.themePalette === state.themePalette;
      btn.setAttribute("aria-pressed", isActive ? "true" : "false");
    });

    // Update mode toggle icon
    els.themeIcon.innerHTML = state.themeMode === "light" ? ICON_SUN : ICON_MOON;

    // Persist
    try {
      localStorage.setItem(STORAGE_KEY_PALETTE, state.themePalette);
      localStorage.setItem(STORAGE_KEY_MODE, state.themeMode);
    } catch (_e) { /* localStorage may be disabled in private browsing */ }
  }

  function loadTheme() {
    try {
      const palette = localStorage.getItem(STORAGE_KEY_PALETTE);
      const mode = localStorage.getItem(STORAGE_KEY_MODE);
      if (palette && THEME_PALETTES.includes(palette)) state.themePalette = palette;
      if (mode && THEME_MODES.includes(mode)) state.themeMode = mode;
      else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
        state.themeMode = "dark";
      }
    } catch (_e) { /* fall through to defaults */ }
    applyTheme();
  }

  function bindThemeSwitcher() {
    $$('[data-theme-palette]').forEach((btn) => {
      btn.addEventListener("click", () => {
        state.themePalette = btn.dataset.themePalette;
        applyTheme();
      });
    });
    $('[data-theme-mode-toggle]').addEventListener("click", () => {
      state.themeMode = state.themeMode === "light" ? "dark" : "light";
      applyTheme();
    });
  }

  // ── API client ──────────────────────────────────────────────────────

  function urlWithToken(path) {
    const sep = path.includes("?") ? "&" : "?";
    return `${path}${sep}token=${encodeURIComponent(TOKEN)}`;
  }

  async function api(path, options = {}) {
    const init = {
      method: "GET",
      headers: { "X-Avis-Token": TOKEN, ...(options.headers || {}) },
      ...options,
    };
    if (init.body && typeof init.body !== "string") {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(init.body);
    }
    const resp = await fetch(path, init);
    let data = null;
    try {
      data = await resp.json();
    } catch (_e) { /* not all endpoints return JSON */ }
    if (!resp.ok) {
      const err = new Error((data && data.detail && (data.detail.message || data.detail)) || resp.statusText);
      err.status = resp.status;
      err.body = data;
      throw err;
    }
    return data;
  }
  const apiGet = (path) => api(path);
  const apiPost = (path, body) => api(path, { method: "POST", body });

  // ── Toast ───────────────────────────────────────────────────────────

  let toastTimer = null;
  function toast(message, kind = "info") {
    if (toastTimer) clearTimeout(toastTimer);
    els.toast.textContent = message;
    els.toast.className = "toast" + (kind === "error" ? " toast-error" : "");
    // Force reflow before adding the show class so the slide-up animates
    void els.toast.offsetHeight;
    els.toast.classList.add("toast-show");
    toastTimer = setTimeout(() => {
      els.toast.classList.remove("toast-show");
    }, 2400);
  }

  // ── Image flash micro-interaction ───────────────────────────────────

  function flashImage(kind) {
    const cls = kind === "warning" ? "flash-warning" : "flash-success";
    els.reviewImageWrap.classList.add(cls);
    setTimeout(() => {
      els.reviewImageWrap.classList.remove(cls);
    }, 280);
  }

  // ── Navigation ──────────────────────────────────────────────────────

  function navigate(view, options = {}) {
    state.view = view;
    if (view === "review" && options.species !== undefined) {
      state.speciesFilter = options.species;
    } else if (view !== "review") {
      state.speciesFilter = "";
    }
    els.body.dataset.view = view;
    closeOtherInput();
    hideConflictModal();

    let url = "/" + (view === "summary" ? "" : view);
    if (view === "review" && state.speciesFilter) {
      url += `?species=${encodeURIComponent(state.speciesFilter)}`;
    }
    history.pushState({ view, species: state.speciesFilter }, "", urlWithToken(url));

    if (view === "summary") loadSummary();
    else if (view === "review") loadNext();
    else if (view === "verified") loadVerified();

    window.scrollTo(0, 0);
  }

  window.addEventListener("popstate", (e) => {
    const view = (e.state && e.state.view) || "summary";
    const species = (e.state && e.state.species) || "";
    state.view = view;
    state.speciesFilter = species;
    els.body.dataset.view = view;
    if (view === "summary") loadSummary();
    else if (view === "review") loadNext();
    else if (view === "verified") loadVerified();
  });

  // ── Coverage badge ──────────────────────────────────────────────────

  function updateCoverageBadge(coverage) {
    if (!coverage) return;
    const pct = coverage.total_pre_labels === 0
      ? "—"
      : Math.round((coverage.total_verified / coverage.total_pre_labels) * 100) + "%";
    els.coverage.textContent = `${coverage.total_verified} / ${coverage.total_pre_labels} · ${pct}`;
  }

  // ── Summary view ────────────────────────────────────────────────────

  async function loadSummary() {
    try {
      const data = await apiGet("/api/summary");
      state.summary = data;
      updateCoverageBadge(data.coverage);
      renderSummary();
    } catch (err) {
      toast(`Failed to load summary: ${err.message}`, "error");
    }
  }

  function renderSummary() {
    const { species, coverage } = state.summary;
    els.summaryOverall.textContent = coverage
      ? `${coverage.total_verified} of ${coverage.total_pre_labels} verified · ${coverage.remaining} remaining`
      : "";

    // Drop existing bucket nodes — keep the empty state
    $$("button.bucket", els.bucketList).forEach((n) => n.remove());

    if (!species.length) {
      els.bucketEmpty.hidden = false;
      els.bucketEmpty.textContent = "No pre-labels found. Run python -m tools.labeler first.";
      return;
    }
    els.bucketEmpty.hidden = true;

    for (const row of species) {
      const btn = document.createElement("button");
      btn.className = "bucket";
      btn.type = "button";

      // Top row: code + counts
      const topRow = document.createElement("div");
      topRow.className = "bucket-row";

      const code = document.createElement("div");
      code.className = "bucket-code";
      code.textContent = row.species_code;
      if (CALT_SUSPECT_BUCKETS.has(row.species_code)) {
        const flag = document.createElement("span");
        flag.className = "bucket-flag";
        flag.textContent = "calt?";
        code.appendChild(flag);
      }
      topRow.appendChild(code);

      const counts = document.createElement("div");
      counts.className = "bucket-counts";
      counts.textContent = `${row.verified} / ${row.total}`;
      topRow.appendChild(counts);

      btn.appendChild(topRow);

      // Progress bar
      const bar = document.createElement("div");
      bar.className = "bucket-bar";
      const fill = document.createElement("div");
      fill.className = "bucket-bar-fill";
      const pct = row.total === 0 ? 0 : (row.verified / row.total) * 100;
      // Defer the width set so the transition runs from 0% on render
      requestAnimationFrame(() => { fill.style.width = `${pct}%`; });
      bar.appendChild(fill);
      btn.appendChild(bar);

      // Footer: percentage + remaining label
      const meta = document.createElement("div");
      meta.className = "bucket-meta";
      meta.innerHTML = `<span>${Math.round(pct)}% reviewed</span><span>${row.remaining} left</span>`;
      btn.appendChild(meta);

      btn.addEventListener("click", () => navigate("review", { species: row.species_code }));
      els.bucketList.appendChild(btn);
    }
  }

  els.btnReviewAll.addEventListener("click", () => navigate("review", { species: "" }));

  // ── Review view ─────────────────────────────────────────────────────

  function fmtConfidence(c) {
    return (c == null ? "—" : c.toFixed(2));
  }

  async function loadNext() {
    showLoading(true);
    closeOtherInput();
    try {
      const path = state.speciesFilter
        ? `/api/next?species=${encodeURIComponent(state.speciesFilter)}`
        : "/api/next";
      const data = await apiGet(path);
      state.current = data;
      renderReview(data);
      els.reviewEmpty.hidden = true;
    } catch (err) {
      if (err.status === 404 && err.body && err.body.detail && err.body.detail.code === "queue_empty") {
        state.current = null;
        showQueueEmpty();
      } else {
        toast(`Failed to load image: ${err.message}`, "error");
      }
    } finally {
      showLoading(false);
    }
  }

  async function loadSpecific(filename) {
    showLoading(true);
    closeOtherInput();
    try {
      const data = await apiGet(`/api/review/${encodeURIComponent(filename)}`);
      state.current = data;
      renderReview(data);
      els.reviewEmpty.hidden = true;
    } catch (err) {
      toast(`Failed to load image: ${err.message}`, "error");
    } finally {
      showLoading(false);
    }
  }

  function showLoading(loading) {
    els.reviewImageLoading.style.display = loading ? "flex" : "none";
  }

  function renderReview(item) {
    els.reviewImage.src = urlWithToken(item.image_url);
    els.reviewImage.alt = item.image_filename;

    els.prelabelSpecies.textContent = item.pre_label_species;
    els.prelabelConfidence.textContent = fmtConfidence(item.pre_label_confidence);
    els.prelabelReasoning.textContent = item.pre_label_reasoning || "";

    if (item.audio_hint) {
      els.audioHint.textContent = `Audio hint: ${item.audio_hint} (${fmtConfidence(item.audio_confidence)})`;
      els.audioHint.hidden = false;
    } else {
      els.audioHint.hidden = true;
    }

    if (item.already_verified_species) {
      const when = item.already_verified_at
        ? new Date(item.already_verified_at).toLocaleString()
        : "";
      let label = item.already_verified_species;
      if (item.already_verified_other_species) {
        label += ` · ${item.already_verified_other_species}`;
      }
      els.alreadyVerified.textContent = `Already verified as ${label}${when ? ` (${when})` : ""}.`;
      els.alreadyVerified.hidden = false;
    } else {
      els.alreadyVerified.hidden = true;
    }

    els.btnConfirmSpecies.textContent = item.pre_label_species;

    if (state.speciesFilter && state.summary.species.length) {
      const bucket = state.summary.species.find((b) => b.species_code === state.speciesFilter);
      if (bucket) {
        const remaining = bucket.total - bucket.verified;
        els.reviewProgress.textContent = `${state.speciesFilter} · ${remaining} of ${bucket.total} left`;
      } else {
        els.reviewProgress.textContent = state.speciesFilter;
      }
    } else if (state.summary.coverage) {
      const c = state.summary.coverage;
      els.reviewProgress.textContent = `${c.remaining} left`;
    } else {
      els.reviewProgress.textContent = "";
    }

    renderQuickCorrectButtons(item);
  }

  function renderQuickCorrectButtons(item) {
    els.quickCorrectGrid.innerHTML = "";
    const candidates = [];

    const top = state.summary.species
      .filter((b) => b.species_code !== item.pre_label_species && state.speciesList.known.includes(b.species_code))
      .slice(0, 6)
      .map((b) => b.species_code);
    candidates.push(...top);

    candidates.push("NONE", "UNKNOWN");

    candidates.forEach((code, i) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn btn-quick";
      const num = i + 1;
      const numEl = document.createElement("span");
      numEl.className = "keynum";
      numEl.textContent = num <= 9 ? String(num) : (code === "NONE" ? "N" : "U");
      btn.appendChild(numEl);
      btn.appendChild(document.createTextNode(code));
      btn.dataset.species = code;
      btn.dataset.shortcut = num <= 9 ? String(num) : (code === "NONE" ? "n" : "u");
      btn.addEventListener("click", () => submitVerify({ species_code: code, agreed_with_pre_label: false }));
      els.quickCorrectGrid.appendChild(btn);
    });

    const otherBtn = document.createElement("button");
    otherBtn.type = "button";
    otherBtn.className = "btn btn-quick btn-quick-other";
    const numEl = document.createElement("span");
    numEl.className = "keynum";
    numEl.textContent = "O";
    otherBtn.appendChild(numEl);
    otherBtn.appendChild(document.createTextNode("OTHER"));
    otherBtn.dataset.shortcut = "o";
    otherBtn.addEventListener("click", openOtherInput);
    els.quickCorrectGrid.appendChild(otherBtn);
  }

  function showQueueEmpty() {
    els.reviewImage.removeAttribute("src");
    els.reviewEmpty.hidden = false;
    const filterLabel = state.speciesFilter ? `for ${state.speciesFilter}` : "across all species";
    els.reviewEmpty.querySelector("p").textContent = `Nothing left to review ${filterLabel}.`;
  }

  // ── Verify submission ───────────────────────────────────────────────

  async function submitVerify(extra) {
    if (!state.current) return;
    const payload = {
      image_filename: state.current.image_filename,
      species_code: extra.species_code,
      other_species_code: extra.other_species_code || null,
      reviewer_notes: extra.reviewer_notes || null,
      agreed_with_pre_label:
        extra.agreed_with_pre_label != null
          ? extra.agreed_with_pre_label
          : extra.species_code === state.current.pre_label_species,
      client_load_time: state.current.client_load_time,
      force_overwrite: !!extra.force_overwrite,
    };
    state.pendingPayload = payload;
    try {
      await apiPost("/api/verify", payload);
      // Visual feedback: green halo if agreed, amber if corrected
      flashImage(payload.agreed_with_pre_label ? "success" : "warning");
      const shown = payload.species_code +
        (payload.other_species_code ? `·${payload.other_species_code}` : "");
      toast(`Saved as ${shown}`);
      state.pendingPayload = null;
      try { state.summary = await apiGet("/api/summary"); updateCoverageBadge(state.summary.coverage); } catch (_) {}
      // Slight delay so the user sees the halo before the next image swaps in
      setTimeout(loadNext, 180);
    } catch (err) {
      if (err.status === 409 && err.body && err.body.detail && err.body.detail.code === "conflict") {
        showConflictModal(err.body.detail.existing, payload);
      } else if (err.status === 422) {
        toast(`Validation: ${err.message}`, "error");
        state.pendingPayload = null;
      } else {
        toast(`Failed to save: ${err.message}`, "error");
        state.pendingPayload = null;
      }
    }
  }

  function showConflictModal(existing, attempted) {
    els.conflictExistingSpecies.textContent = existing.species_code +
      (existing.other_species_code ? ` · ${existing.other_species_code}` : "");
    els.conflictExistingWhen.textContent = existing.verified_at
      ? `(${new Date(existing.verified_at).toLocaleString()})`
      : "";
    els.conflictNewSpecies.textContent = attempted.species_code +
      (attempted.other_species_code ? ` · ${attempted.other_species_code}` : "");
    els.conflictModal.hidden = false;
  }
  function hideConflictModal() { els.conflictModal.hidden = true; }
  els.conflictCancel.addEventListener("click", () => {
    hideConflictModal();
    state.pendingPayload = null;
  });
  els.conflictOverwrite.addEventListener("click", () => {
    hideConflictModal();
    if (state.pendingPayload) {
      submitVerify({ ...state.pendingPayload, force_overwrite: true });
    }
  });

  // ── Buttons & navigation links ──────────────────────────────────────

  els.btnConfirm.addEventListener("click", () => {
    if (!state.current) return;
    submitVerify({
      species_code: state.current.pre_label_species,
      agreed_with_pre_label: true,
    });
  });
  els.btnSkip.addEventListener("click", loadNext);
  els.reviewSkip.addEventListener("click", loadNext);
  els.reviewBack.addEventListener("click", () => navigate("summary"));

  function openOtherInput() {
    els.otherInput.hidden = false;
    els.otherCodeInput.value = "";
    els.otherNotesInput.value = "";
    setTimeout(() => els.otherCodeInput.focus(), 50);
  }
  function closeOtherInput() { els.otherInput.hidden = true; }
  els.btnCancelOther.addEventListener("click", closeOtherInput);
  els.btnConfirmOther.addEventListener("click", () => {
    const code = (els.otherCodeInput.value || "").trim().toUpperCase();
    if (!/^[A-Z]{4}$/.test(code)) {
      toast("Code must be 4 letters (e.g. CALT)", "error");
      return;
    }
    submitVerify({
      species_code: "OTHER",
      other_species_code: code,
      reviewer_notes: (els.otherNotesInput.value || "").trim() || null,
      agreed_with_pre_label: false,
    });
  });

  $$(".nav-link, [data-nav]").forEach((link) => {
    link.addEventListener("click", (e) => {
      const target = link.dataset.nav;
      if (!target) return;
      e.preventDefault();
      navigate(target);
    });
  });

  // ── Verified view ───────────────────────────────────────────────────

  async function loadVerified() {
    try {
      const speciesParam = els.verifiedFilter.value
        ? `?species=${encodeURIComponent(els.verifiedFilter.value)}`
        : "";
      const data = await apiGet(`/api/verified${speciesParam}&limit=500`.replace("?&", "?"));
      renderVerified(data);
    } catch (err) {
      toast(`Failed to load verified: ${err.message}`, "error");
    }
    if (els.verifiedFilter.options.length <= 1 && state.speciesList.all.length) {
      for (const code of state.speciesList.all) {
        const opt = document.createElement("option");
        opt.value = code;
        opt.textContent = code;
        els.verifiedFilter.appendChild(opt);
      }
    }
  }

  function renderVerified(data) {
    els.verifiedCount.textContent = `${data.total} verified${data.total > data.returned ? ` (showing ${data.returned})` : ""}`;
    els.verifiedTbody.innerHTML = "";
    if (!data.records.length) {
      els.verifiedEmpty.hidden = false;
      els.verifiedTable.style.display = "none";
      return;
    }
    els.verifiedEmpty.hidden = true;
    els.verifiedTable.style.display = "";

    for (const r of data.records) {
      const tr = document.createElement("tr");
      tr.dataset.filename = r.image_filename;
      tr.title = r.reviewer_notes || "";

      const tdThumb = document.createElement("td");
      tdThumb.className = "col-thumb";
      const img = document.createElement("img");
      img.className = "verified-thumb";
      img.src = urlWithToken(r.image_url);
      img.alt = "";
      img.loading = "lazy";
      tdThumb.appendChild(img);
      tr.appendChild(tdThumb);

      const tdName = document.createElement("td");
      tdName.className = "verified-filename";
      tdName.textContent = r.image_filename;
      tr.appendChild(tdName);

      const tdSpecies = document.createElement("td");
      tdSpecies.className = "col-species";
      const pill = document.createElement("span");
      pill.className = pillClass(r.species_code);
      let label = r.species_code;
      if (r.other_species_code) label += ` · ${r.other_species_code}`;
      pill.textContent = label;
      tdSpecies.appendChild(pill);
      tr.appendChild(tdSpecies);

      const tdAgreed = document.createElement("td");
      tdAgreed.className = "col-agreed";
      if (r.agreed_with_pre_label === true) {
        tdAgreed.innerHTML = '<span class="agreed-yes">✓</span>';
      } else if (r.agreed_with_pre_label === false) {
        tdAgreed.innerHTML = '<span class="agreed-no">✗</span>';
      } else {
        tdAgreed.textContent = "—";
      }
      tr.appendChild(tdAgreed);

      const tdWhen = document.createElement("td");
      tdWhen.className = "col-when";
      tdWhen.style.color = "var(--text-muted)";
      tdWhen.style.fontSize = "12px";
      tdWhen.textContent = formatRelativeTime(r.verified_at);
      tr.appendChild(tdWhen);

      tr.addEventListener("click", () => {
        navigate("review");
        setTimeout(() => loadSpecific(r.image_filename), 50);
      });

      els.verifiedTbody.appendChild(tr);
    }
  }

  function pillClass(code) {
    if (code === "NONE") return "species-pill species-pill-none";
    if (code === "UNKNOWN") return "species-pill species-pill-unknown";
    if (code === "OTHER") return "species-pill species-pill-other";
    return "species-pill species-pill-known";
  }

  function formatRelativeTime(iso) {
    if (!iso) return "";
    const then = new Date(iso);
    const seconds = (Date.now() - then.getTime()) / 1000;
    if (seconds < 60) return "just now";
    if (seconds < 3600) return Math.round(seconds / 60) + "m ago";
    if (seconds < 86400) return Math.round(seconds / 3600) + "h ago";
    return Math.round(seconds / 86400) + "d ago";
  }

  els.verifiedFilter.addEventListener("change", loadVerified);

  // ── Keyboard shortcuts ──────────────────────────────────────────────

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    if (!els.conflictModal.hidden) return;
    if (state.view !== "review" || !state.current) return;
    if (!els.otherInput.hidden) return;

    if (e.key === "Enter") { e.preventDefault(); els.btnConfirm.click(); return; }
    if (e.key === "Escape") { e.preventDefault(); navigate("summary"); return; }
    if (e.key === "s" || e.key === "S" || e.key === "ArrowRight") {
      e.preventDefault(); loadNext(); return;
    }
    if (e.key === "ArrowLeft") { e.preventDefault(); window.history.back(); return; }

    const matchingButton = $$('button[data-shortcut]').find(
      (b) => b.dataset.shortcut === e.key.toLowerCase()
    );
    if (matchingButton) { e.preventDefault(); matchingButton.click(); }
  });

  // ── Touch gestures ──────────────────────────────────────────────────

  let touchStartX = null;
  let touchStartY = null;
  els.reviewImageWrap.addEventListener("touchstart", (e) => {
    if (e.touches.length !== 1) return;
    touchStartX = e.touches[0].clientX;
    touchStartY = e.touches[0].clientY;
  }, { passive: true });
  els.reviewImageWrap.addEventListener("touchend", (e) => {
    if (touchStartX == null || e.changedTouches.length !== 1) {
      touchStartX = touchStartY = null;
      return;
    }
    const dx = e.changedTouches[0].clientX - touchStartX;
    const dy = e.changedTouches[0].clientY - touchStartY;
    touchStartX = touchStartY = null;
    if (Math.abs(dx) < 60 || Math.abs(dx) < Math.abs(dy)) return;
    if (dx > 0) els.btnConfirm.click();
    else loadNext();
  }, { passive: true });

  // ── Boot ─────────────────────────────────────────────────────────────

  async function boot() {
    loadTheme();
    bindThemeSwitcher();

    try {
      const [species, summary] = await Promise.all([
        apiGet("/api/species"),
        apiGet("/api/summary"),
      ]);
      state.speciesList = species;
      state.summary = summary;
      updateCoverageBadge(summary.coverage);
    } catch (err) {
      toast(`Boot failed: ${err.message}`, "error");
    }

    if (state.view === "summary") renderSummary();
    else if (state.view === "review") loadNext();
    else if (state.view === "verified") loadVerified();
  }

  boot();
})();