# 🐦 Avis — Agentic AI Birdfeeder

> An autonomous, multi-modal bird species identification system running on Raspberry Pi 5 + Hailo AI HAT+.

[![CI](https://github.com/kanecruzwalker/avis-birdfeeder/actions/workflows/ci.yml/badge.svg)](https://github.com/kanecruzwalker/avis-birdfeeder/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: Custom](https://img.shields.io/badge/License-Custom-orange.svg)](LICENSE)


## What This Is

Avis turns a standard bird feeder into an intelligent wildlife monitoring station. A continuously running AI agent monitors camera and microphone input, detects bird presence, identifies species using both audio and visual classifiers, notifies the owner, and logs all observations — autonomously, with no human intervention required.

**Target users:**
- Backyard hobbyists who want to learn about local birds
- Ornithologists and researchers needing autonomous field data collection

## System Overview

```
┌──────────────────────────────────────────────────────┐
│                   Bird Agent (Orchestrator)           │
│                                                      │
│   ┌──────────────┐         ┌──────────────────────┐  │
│   │ Audio Pipeline│         │  Visual Pipeline     │  │
│   │  Mic → WAV   │         │  Camera → Frames     │  │
│   │  → Spectrogram│        │  → Normalized Image  │  │
│   │  → BirdNET   │         │  → EfficientNet      │  │
│   └──────┬───────┘         └──────────┬───────────┘  │
│          │                            │               │
│          └────────────┬───────────────┘               │
│                       ▼                               │
│               ┌───────────────┐                       │
│               │  Score Fusion │                       │
│               └──────┬────────┘                       │
│                      ▼                                │
│          ┌───────────────────────┐                    │
│          │  Notify + Log Result  │                    │
│          └───────────────────────┘                    │
└──────────────────────────────────────────────────────┘
```

## Hardware

| Component | Model |
|-----------|-------|
| Compute | Raspberry Pi 5 (8GB) |
| AI Accelerator | Hailo AI HAT+ (26 TOPS) |
| Primary Camera | Raspberry Pi Camera Module 3 |
| Secondary Camera | Raspberry Pi Camera Module 3 |
| Microphone | Fifine USB Microphone |
| Storage | 128GB SSD |
| Power | 27W USB-C |

## Quick Start

See [docs/SETUP.md](docs/SETUP.md) for full installation and hardware setup instructions.

```bash
git clone https://github.com/kanecruzwalker/avis-birdfeeder.git
cd avis-birdfeeder
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then fill in your values
python -m src.agent.bird_agent   # start the agent
```

## Web dashboard

A separate FastAPI app under `src/web/` exposes a single-page dashboard
with a live MJPEG preview, observation history (recent / timeline /
gallery / detail), and a chat proxy to the LLM analyst. Token-auth,
served over Tailscale by default; an ngrok helper script is included
for ad-hoc class demos.

See [docs/WEB_DASHBOARD.md](docs/WEB_DASHBOARD.md) for install,
Tailscale invite flow, ngrok demo flow, token rotation, and
troubleshooting.

## Project Structure

```
avis-birdfeeder/
├── src/
│   ├── agent/      # Orchestration: the main agentic loop
│   ├── audio/      # Audio capture, preprocessing, classification
│   ├── vision/     # Camera capture, preprocessing, classification
│   ├── fusion/     # Combines audio + visual confidence scores
│   ├── notify/     # Notification dispatch (push, email, etc.)
│   └── data/       # Shared data models and dataset utilities
├── configs/        # YAML configuration (thresholds, paths, species)
├── scripts/        # One-off utilities (dataset download, model export)
├── notebooks/      # EDA and training experimentation
├── tests/          # Mirrors src/ — unit tests per module
└── docs/           # Architecture, setup, and context docs
```

## Contributing

Both team members contribute across all modules. See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for branch naming, PR, and commit conventions, and [docs/ONBOARDING.md](docs/ONBOARDING.md) for full environment setup and onboarding instructions.

## Team

| Name | GitHub |
|------|--------|
| Kane Cruz-Walker | kanecruzwalker |
| Daniel Wen | @DANIEL_HANDLE <- TODO |


## Academic context

Developed as a capstone project for CS 450 - Introduction to AI
at San Diego State University, Spring 2026.
Kane Cruz-Walker and Daniel Wen. All work is original.

## License and commercial use

This project is publicly available for academic reference and non-commercial research. Commercial use requires written permission from the authors. See LICENSE for details


