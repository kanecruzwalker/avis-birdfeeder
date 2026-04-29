"""Avis web dashboard (Phase 8C).

FastAPI application that serves a single-page dashboard over Tailscale
(or ngrok for ephemeral demos) with:

- Live MJPEG preview of the feeder
- Classification box overlay during active detections
- Detection history with timeline, recent-list, and gallery views
- Per-observation detail with cropped / annotated / full-frame variants
- Conversational ``/api/ask`` proxy to ``BirdAnalystAgent``

Auth is a single shared bearer token (``AVIS_WEB_TOKEN``) on every route
except ``/health``.

Module layout (filled in across PRs 1–9 of ``feat/web-dashboard``):

    src/web/__init__.py
    src/web/auth.py            — token middleware (mirrors tools/labeler/ui/auth.py)
    src/web/app.py             — FastAPI app factory
    src/web/__main__.py        — ``python -m src.web`` CLI entry point
    src/web/routes/            — status, stream, observations, chat (PR 2+)
    src/web/stream_buffer.py   — thread-safe MJPEG ring buffer (PR 3)
    src/web/box_cache.py       — last-known YOLO box with TTL (PR 5)
    src/web/frame_annotator.py — PIL box-drawing helper (PR 5)
    src/web/observation_store.py — read-only adapter over observations.jsonl (PR 2)
    src/web/templates/         — Jinja2 root layout (PR 6)
    src/web/static/            — vanilla JS SPA (PR 6+)

The dashboard doesn't write to anything the agent reads.
``VisionCapture.publish()`` and ``BirdAgent.box_cache.update()`` are
no-ops when the dependencies aren't injected, so stopping
``avis-web.service`` leaves the agent untouched.

See ``docs/investigations/web-dashboard-2026-04-28.md`` for the full
design rationale.
"""
