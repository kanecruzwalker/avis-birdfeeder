# Pi Deployment Guide

> Full documentation for deploying, updating, and managing the Avis system
> on the Raspberry Pi 5. Covers first-time setup, the laptop↔Pi workflow,
> the config override model, and troubleshooting for the issues we've
> actually hit during Phase 5–8 deployment.

---

## Mental model

The Pi runs the Avis system. The laptop is the dev machine and the source of
truth for git. You push commits from the laptop; the Pi pulls them down.
Anything that needs to change on the Pi should flow through git — never edit
production code directly on the Pi.

Config values are different. Three kinds of values live in `configs/*.yaml`:

1. **Committed defaults** — safe, production-conservative. These are in git.
2. **Per-deployment tuning** — push-token enabled, NPU enabled, detection
   mode set to `yolo`, per-camera crop zones calibrated to the physical
   feeder mount, testing threshold for live tuning. These should never be
   committed — they'd break other people's deployments and expose safety
   defaults. They live only on the Pi and are re-applied after every
   `git pull` via `scripts/dev_config.py`.
3. **Secrets** — Pushover tokens, Gemini API key. These live in the Pi's
   `.env` file, which is gitignored.

The workflow is designed so that `git pull origin main && python scripts/dev_config.py`
on the Pi always produces a running, correctly-configured deployment.

---

## First-time Pi setup

Assumes you have a Pi 5 with Debian Trixie, 8GB SSD storage at `/mnt/data`,
the Hailo AI HAT+, dual IMX708 cameras, and the Fifine USB mic physically
installed. If you're setting up the hardware from scratch, see
`docs/SETUP.md`.

### 1. Clone the repo on the Pi

```bash
cd /mnt/data
git clone https://github.com/kanecruzwalker/avis-birdfeeder.git
cd avis-birdfeeder
```

### 2. Create the main Python 3.13 venv

```bash
python3 -m venv /mnt/data/avis-venv
source /mnt/data/avis-venv/bin/activate
pip install -r requirements-pi.txt
```

Expose the system-installed `picamera2` C extension to the venv:

```bash
echo "/usr/lib/python3/dist-packages" > \
  /mnt/data/avis-venv/lib/python3.13/site-packages/system-dist-packages.pth
```

### 3. Set up the Python 3.11 subprocess bridge for BirdNET

BirdNET depends on `tflite_runtime`, which has no Python 3.13 aarch64 wheel.
We run BirdNET in a Python 3.11 subprocess. Install pyenv and Python 3.11:

```bash
curl https://pyenv.run | bash
# Follow the prompt to add pyenv to your shell init, then reload:
exec $SHELL
pyenv install 3.11.9
/home/birdfeeder01/.pyenv/versions/3.11.9/bin/pip install \
  pyyaml birdnetlib==0.9.0 tflite-runtime resampy librosa "numpy<2"
```

### 4. Copy model artifacts from the laptop

Trained model weights are gitignored. From the laptop:

```powershell
scp models/visual/frozen_extractor.pt     birdfeeder01@birdfeeder.local:/mnt/data/avis-birdfeeder/models/visual/
scp models/visual/sklearn_pipeline.pkl    birdfeeder01@birdfeeder.local:/mnt/data/avis-birdfeeder/models/visual/
scp models/visual/efficientnet_b0_avis_v2.hef  birdfeeder01@birdfeeder.local:/mnt/data/avis-birdfeeder/models/visual/
```

Confirm on the Pi:

```bash
ls -lh models/visual/
```

### 5. Create the `.env` file

```bash
cp .env.example .env
nano .env
```

Fill in:
PUSHOVER_USER_KEY=...
PUSHOVER_APP_TOKEN=...
GEMINI_API_KEY=...

### 6. Apply Pi-local config overrides

```bash
source /mnt/data/avis-venv/bin/activate
python scripts/dev_config.py
```

See the "Config override model" section below for what this does.

### 7. Install and start the systemd service

```bash
bash scripts/install_service.sh
```

That installs `scripts/avis.service` to `/etc/systemd/system/`, enables it so
it starts on every boot, and starts it now. The Pi will boot autonomously
into the Avis system going forward — no SSH or manual commands needed.

### 8. Verify from the laptop

In PowerShell on the laptop:

```powershell
. .\pi.ps1
pi-status
```

You should see `active (running)`. Done.

---

## Laptop-side setup

### `pi.ps1` shortcuts

`pi.ps1` at repo root is a dot-source script that loads a set of PowerShell
functions for managing the Pi. Load it into your session:

```powershell
. .\pi.ps1
```

To have it loaded automatically in every new PowerShell session, add this to
your `$PROFILE` (run `notepad $PROFILE` to edit):

```powershell
if (Test-Path "$HOME\Desktop\Kane\GithubRepositories\avis-birdfeeder-pr1-scaffold\avis-birdfeeder\pi.ps1") {
    . "$HOME\Desktop\Kane\GithubRepositories\avis-birdfeeder-pr1-scaffold\avis-birdfeeder\pi.ps1"
}
```

### Host override

`pi.ps1` defaults to `birdfeeder01@birdfeeder.local` (mDNS hostname). If mDNS
doesn't resolve on your network, or if you're using a different Pi (Dan has
his own), override the host in your `$PROFILE`:

```powershell
$env:AVIS_PI_HOST = "birdfeeder01@192.168.4.76"
```

### SSH key auth (strongly recommended)

Without an SSH key, every `pi-*` command prompts for the Pi password. Set
up a key once and you're done:

```powershell
# On the laptop, if you don't already have one:
ssh-keygen -t ed25519

# Copy the key to the Pi:
type $HOME\.ssh\id_ed25519.pub | ssh birdfeeder01@birdfeeder.local "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

Test — this should connect without a password prompt:

```powershell
pi-status
```

---

## Daily workflow

### Push new work from laptop

Normal git flow:

```powershell
# Make changes on a feature branch, commit, push, open PR, merge to main.
git checkout main
git pull origin main
```

### Deploy to Pi

Once your changes are in `main` on GitHub, deploy from the laptop:

```powershell
pi-deploy
```

This runs on the Pi, in order:

1. `git pull origin main`
2. `python scripts/dev_config.py` to re-apply Pi-local overrides
3. `sudo systemctl restart avis`
4. `sudo systemctl status avis`

If you want finer control:

```powershell
pi-pull        # git pull + dev_config.py, but don't restart yet
pi-config-check   # validate all three YAMLs parse cleanly
pi-restart     # apply the changes
pi-status      # verify
```

### Watching the live system

```powershell
pi-logs        # stream live logs (Ctrl+C to stop)
pi-logs-since "1 hour ago"       # non-streaming, bounded window
pi-logs-since "2026-04-19 15:00"
pi-logs-since "today"
```

### Manual run for debugging

When you want to see stdout directly in your terminal without the systemd
wrapper:

```powershell
pi-run        # stops service, runs orchestrator for 60s, then exits
pi-run 180    # 3-minute smoke test
pi-start      # bring service back up when done
```

---

## Config override model

`scripts/dev_config.py` applies a known set of Pi-local overrides to the
three YAML configs. Every override is declared at the top of the script in
the `PI_OVERRIDES` list with its exact key path — there are no sed-style
regexes and no silent fallbacks. If the script prints `old -> new` for a
key, it changed that key. If it prints `unchanged`, it didn't.

### What gets overridden

| Config file | Key | Override value |
|-------------|-----|----------------|
| `hardware.yaml` | `hailo.enabled` | `true` |
| `hardware.yaml` | `hailo.detection_mode` | `yolo` |
| `hardware.yaml` | `cameras.motion_threshold` | `0.005` |
| `hardware.yaml` | `cameras.feeder_crop_cam0` | `{x: 630, y: 130, width: 700, height: 580}` |
| `hardware.yaml` | `cameras.feeder_crop_cam1` | `{x: 420, y: 130, width: 700, height: 580}` |
| `notify.yaml` | `channels.push` | `true` |
| `thresholds.yaml` | `agent.confidence_threshold` | `0.20` |

### Changing an override

Edit `PI_OVERRIDES` at the top of `scripts/dev_config.py`. That's the only
place anyone should edit. Re-run the script after every change.

### Backup and recovery

Every run backs up the pre-override YAML to `configs/*.yaml.bak` before
modifying. If something goes wrong:

```bash
cp configs/hardware.yaml.bak configs/hardware.yaml
```

The `.bak` files are gitignored so they never end up in a commit.

### Validation

After applying overrides, the script parse-checks every config file. If any
come out corrupted (e.g. from a hand-edit prior to override application),
the script exits non-zero and tells you which file and why. Systemd-friendly
exit codes: `0` success, `1` validation failure, `2` unexpected error.

You can re-validate at any time without re-applying:

```powershell
pi-config-check
```

---

## Troubleshooting

### Service won't start

First check for YAML corruption:

```powershell
pi-config-check
```

If that fails, the error message will name the file. Restore from backup:

```bash
ssh birdfeeder01@birdfeeder.local
cd /mnt/data/avis-birdfeeder
cp configs/hardware.yaml.bak configs/hardware.yaml   # or whichever file is broken
python scripts/dev_config.py
sudo systemctl restart avis
```

### Microphone not capturing (audio_result: null for hours)

The Fifine USB mic's sounddevice index is non-deterministic across reboots.
If audio cycles are silently failing, the device index may have shifted.
Check:

```bash
ssh birdfeeder01@birdfeeder.local
source /mnt/data/avis-venv/bin/activate
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Find the "USB PnP Audio Device" or "Fifine" entry and note its index. If it
doesn't match what's in `configs/hardware.yaml` under `microphone.device_index`,
edit the config and restart the service.

Fix-by-name is scheduled for `fix/audio-device-lookup-by-name` (PR-B).

### Hailo errors on startup

`HAILO_STREAM_NOT_ACTIVATED(72)` warnings are harmless when sharing a
VDevice between YOLO and EfficientNet — the ROUND_ROBIN scheduler emits
these but inference completes correctly. If you see continuous errors even
with `hailo.enabled: false`, you likely have stale code instantiating Hailo
objects anyway. Check the code path in `src/vision/classify.py` for
`_load_hailo()` calls that bypass the enable flag.

To fall back to CPU-only for debugging:

```bash
ssh birdfeeder01@birdfeeder.local
cd /mnt/data/avis-birdfeeder
# Edit the hailo.enabled key in dev_config.py temporarily to False
# then re-run:
python scripts/dev_config.py
sudo systemctl restart avis
```

### "birdfeeder.local" doesn't resolve on laptop

mDNS isn't working on your LAN. Either:

- Set `$env:AVIS_PI_HOST = "birdfeeder01@<static-IP>"` in your `$PROFILE`
- Or install Apple Bonjour Services on Windows (enables mDNS)

### Password prompt on every `pi-*` command

Set up SSH key auth — see "Laptop-side setup" above.

### Config drift after `git pull`

If you pulled on the Pi and forgot to run `dev_config.py`, you'll be running
with committed defaults (hailo disabled, push off, threshold 0.7, no per-
camera crops). The service will still start but behavior changes visibly
within a minute. `pi-deploy` does both in one shot — prefer it over manual
`pi-pull` + `pi-restart`.

---

## Files touched by this workflow

| File | Purpose | Who edits |
|------|---------|-----------|
| `pi.ps1` | Laptop-side Pi shortcuts | Committed — version for everyone |
| `scripts/dev_config.py` | Pi-local config overrides | Committed — change `PI_OVERRIDES` only |
| `scripts/install_service.sh` | One-shot systemd installer | Committed — rarely edited |
| `scripts/avis.service` | systemd unit | Committed |
| `configs/*.yaml` | Committed defaults | Committed — never put Pi-local values here |
| `configs/*.yaml.bak` | Auto-generated backups | Gitignored |
| `.env` on Pi | Secrets | Gitignored, lives only on Pi |