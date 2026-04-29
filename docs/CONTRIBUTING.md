# Contributing

Both Kane and Daniel contribute across all modules. This document defines our conventions so the codebase stays consistent regardless of who wrote what.

## Branch Naming

```
feature/short-description    # new functionality
fix/short-description        # bug fix
docs/short-description       # documentation only
refactor/short-description   # no behavior change, code cleanup
test/short-description       # adding or fixing tests
```

Examples:
- `feature/audio-mel-spectrogram`
- `fix/camera-capture-timeout`
- `docs/update-architecture`

## Commit Messages

Format: `type(scope): short description`

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`  
Scopes: `agent`, `audio`, `vision`, `fusion`, `notify`, `data`, `ci`, `config`

Examples:
```
feat(audio): add mel spectrogram generation from WAV
fix(vision): handle missing frame gracefully in capture loop
test(fusion): add unit tests for confidence score weighting
docs(setup): add Hailo driver install instructions
chore(ci): pin ruff version in workflow
```

## Pull Request Rules

1. **No direct pushes to `main`** — every change goes through a PR, even solo work
2. **PR must pass CI** before merging (lint + format + tests)
3. **Fill out the PR template** — describe what changed and why
4. **Self-review before requesting review** — read your own diff first
5. **One logical change per PR** — don't bundle unrelated changes

## Code Style

- Formatter: `ruff format` (replaces black)
- Linter: `ruff check`
- Line length: 100 characters
- Type hints on all function signatures
- Docstrings on all public functions and classes (Google style)

Run before every commit:
```bash
ruff format src/ tests/
ruff check src/ tests/
```

## Testing

- Every function in `src/` should have at least one test in the mirrored `tests/` path
- Tests that require hardware (camera, mic, Hailo) must be marked `@pytest.mark.hardware` and are skipped in CI
- Tests that require model weights must be marked `@pytest.mark.requires_model` and are skipped in CI

## Config Rule

If it's a number, a path, or a threshold — it belongs in `configs/`, not in source code. Source code reads from config. Source code never defines magic values.

## Adding a New Dependency

1. Add to `requirements.txt` (runtime) or `requirements-dev.txt` (dev/test only)
2. Pin to a specific version: `librosa==0.10.2`
3. Add a comment above it explaining why it's needed
4. Mention it in your PR description



## LLM / Agentic Components

The agentic layer (src/agent/tools/, BirdAnalystAgent, LangChainAnalyst,
ExperimentOrchestrator) requires a GEMINI_API_KEY in .env to run the LLM paths.

- All LLM calls degrade gracefully — no key = fallback to fixed schedule, no crash
- Tests mock all LLM calls — no API key required to run the test suite
- Cost: ~$0.004 per full notebook run, <$0.10/month Pi deployment at 30min intervals
- Model: gemini-2.5-flash (set in configs/hardware.yaml llm.model)