# Setup Guide

## Prerequisites

- Python 3.11 or higher
- Git
- (On Pi) Raspberry Pi OS Bookworm 64-bit, Hailo drivers installed

## Clone and Install

```bash
git clone https://github.com/YOUR_USERNAME/avis-birdfeeder.git
cd avis-birdfeeder

# Create virtual environment
python -m venv .venv

# Activate (Mac/Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install runtime dependencies
pip install -r requirements.txt

# Install dev dependencies (linting, testing)
pip install -r requirements-dev.txt
```

## Environment Variables

```bash
cp .env.example .env
```

Open `.env` and fill in values. See `.env.example` for all required variables with descriptions.

## Running the Agent

```bash
python -m src.agent.bird_agent
```

## Running Tests

```bash
pytest tests/ -v
```

## Running the Linter

```bash
ruff check src/ tests/
ruff format --check src/ tests/
```

## Raspberry Pi — Hailo Setup

Before running inference on the Pi, ensure the Hailo runtime is installed:

```bash
# Follow the official Hailo RPi5 setup guide:
# https://github.com/hailo-ai/hailo-rpi5-examples
pip install hailort  # version must match your HailoRT runtime
```

Verify the Hailo HAT+ is detected:

```bash
hailortcli fw-control identify
```

## Dataset Download

Datasets are not committed to the repo. Use the download script after setup:

```bash
python scripts/download_datasets.py --config configs/paths.yaml
```

See `docs/DATASETS.md` (coming in Phase 2) for full dataset documentation.
