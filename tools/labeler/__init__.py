"""Avis labeling assistant.

Agent-assisted labeling tool that uses Gemini 2.5 Flash vision to
pre-label deployment-captured images with suggested species. Humans
review and confirm pre-labels to produce verified training data for
classifier retraining.

Pipeline (this module handles step 1; other PRs add 2-4):
    1. pre_labeler.py: batch pre-label images with Gemini
    2. server.py: interactive human-review UI (separate PR)
    3. export.py: JSONL -> folder-structured training set (separate PR)
    4. scripts/retrain_classifier.py: train new LogReg head (separate PR)

See docs/investigations/labeling-assistant-2026-04-24.md for full context.
"""
