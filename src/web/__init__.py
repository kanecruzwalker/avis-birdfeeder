"""Avis web dashboard.

FastAPI application that serves a single-page dashboard over Tailscale
(or ngrok for ephemeral demos) with:

- Live MJPEG preview of the feeder
- Classification box overlay during active detections
- Detection history with timeline, recent-list, and gallery views
- Per-observation detail with cropped / annotated / full-frame variants
- Conversational ``/api/ask`` proxy to ``BirdAnalystAgent``

Auth is a single shared bearer token (``AVIS_WEB_TOKEN``) on every route
except ``/health``.

The dashboard doesn't write to anything the agent reads.
``VisionCapture.publish()`` and ``BirdAgent.box_cache.update()`` are
no-ops when the dependencies aren't injected, so stopping
``avis-web.service`` leaves the agent untouched.

See ``docs/investigations/web-dashboard-2026-04-28.md`` for the full
design rationale.
"""
