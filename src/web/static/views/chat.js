// Chat view — bubble thread over POST /api/ask.
//
// One question, one answer. No SSE / no streaming for this PR; the
// analyst's wait-for-full response (a few seconds in production) is
// fine for the dashboard's expected use ("how many cardinals today?")
// and keeps the wire shape and frontend simple. History lives only in
// the DOM — refresh wipes it. The analyst itself maintains no
// per-user thread; conversational context comes from the question
// text alone.

const $ = (sel) => document.querySelector(sel);

let api = null;
let toast = null;

const state = {
  pending: false,
  available: null, // null = unknown, true/false once we've tried once
};

// ── Rendering ────────────────────────────────────────────────────────────────

function dropEmpty() {
  const empty = $("#chat-empty");
  if (empty && empty.parentElement) empty.remove();
}

function appendBubble({ role, text, toolsCalled = null, error = null }) {
  dropEmpty();
  const thread = $("#chat-thread");
  if (!thread) return;

  const bubble = document.createElement("div");
  bubble.className = "chat-bubble";
  bubble.dataset.role = role;
  if (error) bubble.dataset.tone = "error";

  const body = document.createElement("div");
  body.className = "chat-bubble-body";
  body.textContent = text;
  bubble.appendChild(body);

  if (toolsCalled && toolsCalled.length) {
    // The analyst reports which tools it called (read_recent_observations,
    // get_feeder_health, etc.) — surfacing that builds trust in the answer
    // without flooding the bubble. Collapsed by default; clicking expands.
    const details = document.createElement("details");
    details.className = "chat-tools";
    const summary = document.createElement("summary");
    summary.textContent = `tools (${toolsCalled.length})`;
    details.appendChild(summary);
    const list = document.createElement("ul");
    for (const name of toolsCalled) {
      const li = document.createElement("li");
      li.textContent = name;
      list.appendChild(li);
    }
    details.appendChild(list);
    bubble.appendChild(details);
  }

  thread.appendChild(bubble);
  thread.scrollTop = thread.scrollHeight;
  return bubble;
}

function appendPending() {
  // Returns a node we can swap with the real answer once it arrives.
  // Two reasons over a global spinner: it stays anchored to the
  // question, and it survives quick re-asks (the user can fire a
  // second question before the first returns).
  const bubble = appendBubble({ role: "assistant", text: "Thinking…" });
  if (bubble) bubble.dataset.pending = "true";
  return bubble;
}

function setStatus(msg) {
  const el = $("#chat-status");
  if (el) el.textContent = msg;
}

function setSubmitEnabled(enabled) {
  const btn = $("#chat-submit");
  const input = $("#chat-input");
  if (btn) btn.disabled = !enabled;
  if (input) input.disabled = !enabled;
}

// ── Wire ────────────────────────────────────────────────────────────────────

async function ask(question) {
  state.pending = true;
  setSubmitEnabled(false);
  appendBubble({ role: "user", text: question });
  const placeholder = appendPending();

  try {
    const body = await api.fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    state.available = true;
    if (placeholder) placeholder.remove();
    appendBubble({
      role: "assistant",
      text: body.answer || "(no answer)",
      toolsCalled: body.tools_called || [],
      error: body.error ? body.error : null,
    });
    if (!body.llm_available) {
      setStatus("LLM unavailable — fallback response.");
    } else {
      setStatus(body.tools_called?.length ? `${body.tools_called.length} tool(s) used` : "ready");
    }
  } catch (err) {
    if (placeholder) placeholder.remove();
    if (err.message === "no-token" || err.message === "unauthorized") return;
    // 503 from the route surfaces as "HTTP 503: …" — special-case so the
    // user sees a clear "not configured" state rather than a raw error.
    if (err.message?.startsWith("HTTP 503")) {
      state.available = false;
      setStatus("Analyst not configured (server has no GEMINI_API_KEY).");
      appendBubble({
        role: "assistant",
        text:
          "The analyst is not configured on this server. Set GEMINI_API_KEY in the dashboard's environment and restart.",
        error: true,
      });
    } else {
      appendBubble({
        role: "assistant",
        text: `Request failed: ${err.message}`,
        error: true,
      });
      toast?.(err.message, { tone: "error" });
    }
  } finally {
    state.pending = false;
    setSubmitEnabled(true);
  }
}

function bindForm() {
  const form = $("#chat-form");
  const input = $("#chat-input");
  if (!form || !input) return;

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (state.pending) return;
    const q = input.value.trim();
    if (!q) return;
    input.value = "";
    ask(q);
  });

  // Enter to send, Shift+Enter for newline — matches the convention
  // every chat surface uses.
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
}

export function mountChat(ctx) {
  api = ctx.api;
  toast = ctx.toast;
  bindForm();
  setStatus("ready");
}
