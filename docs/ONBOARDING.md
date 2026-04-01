# Avis — Onboarding Cheat Sheet
### For new contributors (hi Dan 👋)

---

## 1. Prerequisites — install these first

| Tool | Version | How to install |
|------|---------|----------------|
| Python | **3.11.9** | `winget install Python.Python.3.11` (Windows) or `brew install python@3.11` (Mac) |
| Git | Any recent | https://git-scm.com/downloads |
| VS Code | Any recent | https://code.visualstudio.com (recommended) |

> ⚠️ Do NOT use Python 3.12+ or 3.14 — our ML dependencies require 3.11.
> Verify with: `py -3.11 --version` → should show `Python 3.11.9`

---

## 2. Clone and set up

```bash
git clone https://github.com/kanecruzwalker/avis-birdfeeder.git
cd avis-birdfeeder

# Create virtual environment with Python 3.11 explicitly
py -3.11 -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate

# You should see (.venv) in your prompt — confirm Python version
python --version   # must show 3.11.9

# Install all dependencies
pip install -r requirements-dev.txt
```

---

## 3. Verify everything works

```bash
# Run all tests — should show 287 passed, 0 warnings
python -m pytest tests/ -v

# Check lint is clean — should show "All checks passed!"
ruff check src/ tests/

# Check format is clean
ruff format --check src/ tests/
```

If all three are clean you're fully set up.

---

## 4. Environment file

```bash
# Create your local env file from the template
cp .env.example .env
```

Leave `.env` values blank for now — you'll fill in `AUDIO_DEVICE_INDEX`
once you have the Pi mic connected. Never commit `.env` — it's in `.gitignore`.

---

## 5. Daily workflow — every time you work

```bash
# 1. Activate venv (every session)
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Mac/Linux

# 2. Pull latest from main before starting
git pull origin main

# 3. Create a branch for your work
git checkout -b feature/your-feature-name

# 4. Write code in src/, write tests in tests/
# 5. Run tests frequently as you go
python -m pytest tests/ -v

# 6. Before committing — format and lint
ruff format src/ tests/
ruff check src/ tests/

# 7. Commit with a meaningful message
git add src/ tests/
git commit -m "feat(audio): add mel spectrogram generation"

# 8. Push and open a PR on GitHub
git push origin feature/your-feature-name
```

---

## 6. Branch naming convention

```
feature/short-description    # new functionality
fix/short-description        # bug fix
docs/short-description       # documentation only
refactor/short-description   # cleanup, no behavior change
test/short-description       # adding or fixing tests
```

---

## 7. Commit message format

```
type(scope): short description

Types:  feat, fix, docs, test, refactor, chore
Scopes: agent, audio, vision, fusion, notify, data, ci, config
```

Examples:
```
feat(audio): add mel spectrogram generation from WAV
fix(vision): handle missing frame gracefully in capture loop
test(fusion): add confidence score weighting tests
docs(setup): add Pi deployment instructions
```

---

## 8. Rules — read these once, follow them always

- **Never push directly to `main`** — always use a branch + PR
- **Never hardcode paths or numbers in src/** — they go in `configs/*.yaml`
- **Every function you write needs at least one test** in the matching `tests/` file
- **Tests that need the Pi** get marked `@pytest.mark.hardware` — CI skips them automatically
- **Run `ruff format` and `ruff check` before every commit** — CI enforces both

---

## 9. Project structure — where things live

```
src/
├── agent/      # Main loop — orchestrates everything
├── audio/      # Mic capture → spectrogram → BirdNET classification
├── vision/     # Camera capture → normalize → EfficientNet classification
├── fusion/     # Combines audio + visual confidence scores
├── notify/     # Sends results to user (log, print, push)
└── data/       # Shared data models (schema.py) — read this first

configs/        # All tunable values — confidence, paths, species list
tests/          # Mirrors src/ exactly — one test file per source file
docs/           # Architecture, setup, and contributing guides
```

---

## 10. Key commands at a glance

| What | Command |
|------|---------|
| Run all tests | `python -m pytest tests/ -v` |
| Run one module's tests | `python -m pytest tests/audio/ -v` |
| Run tests matching a name | `python -m pytest tests/ -v -k "schema"` |
| Lint check | `ruff check src/ tests/` |
| Auto-fix lint | `ruff check src/ tests/ --fix` |
| Format code | `ruff format src/ tests/` |
| Run the agent | `python -m src.agent.bird_agent` |
| Run hardware tests (Pi only) | `python -m pytest tests/ -m hardware -v` |
| Download audio datasets    | `python scripts/download_datasets.py` |
| Generate train/val/test splits | `python scripts/generate_splits.py` |
| Launch Jupyter             | `jupyter notebook` |

---

## 11. If something is broken

1. Make sure venv is active — check for `(.venv)` in your prompt
2. Run `pip install -r requirements-dev.txt` — a new dep may have been added
3. Run `git pull origin main` — you may be behind
4. Check the Actions tab on GitHub — if CI is red, main may have a known issue
5. Ask Kane

---

*See `docs/ARCHITECTURE.md` for full system design.*
*See `docs/CONTRIBUTING.md` for detailed PR and commit conventions.*
