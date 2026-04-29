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


### 7. Install and start the systemd service

```bash
bash scripts/install_service.sh
```

That installs `scripts/avis.service` to `/etc/systemd/system/`, enables it so
it starts on every boot, and starts it now. The Pi will boot autonomously
into the Avis system going forward — no SSH or manual commands needed.

### 8. Arm the systemd watchdog

The watchdog prevents silent multi-hour freezes by restarting the service if
it stops heartbeating for 5 minutes. Setup is one-time per Pi.

```bash
sudo systemctl edit avis.service
```

In the editor, paste:

```ini
[Service]
Restart=always
RestartSec=30
WatchdogSec=300
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl restart avis.service
```

See the "Systemd watchdog" section below for details and verification.

### 9. Verify from the laptop






### 9. Verify from the laptop

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


---

## Systemd watchdog

The Avis service runs under a watchdog that restarts the process if it stops heartbeating for 5 minutes. This prevents silent freezes in which the service appears `active (running)` but the Python loop has stalled — a failure mode we hit on 2026-04-23 when two multi-hour freezes produced zero observations.

### How it works

Two cooperating pieces:

1. **Application side** — `ExperimentOrchestrator.run()` sends `WATCHDOG=1` to systemd after every cycle via the `sdnotify` package. If the main loop deadlocks or blocks on a network call, the signals stop.
2. **Systemd side** — the service unit override specifies `WatchdogSec=300`. If systemd doesn't receive a `WATCHDOG=1` within 5 minutes, it kills the process and restarts it.

The result: any silent freeze self-corrects within 5 minutes.

The orchestrator sends three signal types:

| Signal | When | Purpose |
|---|---|---|
| `READY=1` | After boot sequence completes | Service is up and ready |
| `WATCHDOG=1` | After each cycle | Heartbeat — prevents timeout |
| `STOPPING=1` | In the `finally` block on shutdown | Graceful exit |

The `sdnotify` import is wrapped in try/except so the orchestrator runs cleanly in dev/test environments without sdnotify installed (laptop, CI).

### First-time setup on the Pi

This is done once per Pi deployment. The systemd override is Pi-local infrastructure — it is not committed to the repo.

First, install `sdnotify` in the service venv:

```bash
source /mnt/data/avis-venv/bin/activate
pip install sdnotify
deactivate
```

Then create the systemd override:

```bash
sudo systemctl edit avis.service
```

In the editor that opens, paste:

```ini
[Service]
Restart=always
RestartSec=30
WatchdogSec=300
```

Save and exit (in nano: `Ctrl+O`, `Enter`, `Ctrl+X`). Then reload systemd and restart the service:

```bash
sudo systemctl daemon-reload
sudo systemctl restart avis.service
```

### Override values explained

- `Restart=always` — restart under any failure condition, not just explicit failure exit codes
- `RestartSec=30` — wait 30 seconds between exit and restart, giving network/hardware time to recover from transient issues
- `WatchdogSec=300` — consider the service dead if no `WATCHDOG=1` received within 5 minutes

The 5-minute timeout is deliberately generous. Normal cycle time is 1-5 seconds, so 5 minutes allows ~60 cycles of slack. This avoids false positives during legitimate slow cycles (e.g. Pushover retries on flaky networks) while still catching true deadlocks.

### Verification

After applying the override, confirm the watchdog is armed:

```bash
sudo journalctl -u avis.service -n 30 --no-pager | grep -i watchdog
```

You should see: Systemd watchdog notifier armed (heartbeat per cycle).

If you see `sdnotify unavailable — running without systemd watchdog` instead, sdnotify is not installed in the service venv. Fix with the `pip install sdnotify` step above, then restart.

### Testing the watchdog

To prove the watchdog will catch a freeze, simulate one by pausing the process:

```bash
# Find the Python PID
ps aux | grep experiment_orchestrator | grep -v grep

# Suspend it (simulates deadlock)
sudo kill -STOP <PID>

# Watch systemd handle it
sudo journalctl -u avis.service -f
```

Within 5 minutes:

avis.service: Watchdog timeout (limit 5min)!
avis.service: Killing process ...
avis.service: Main process exited, code=killed, status=6/ABRT
avis.service: Scheduled restart job, restart counter is at N.
avis.service: Starting avis.service...

If you see the automatic restart, the watchdog is working correctly.

### History

- **2026-04-23** — Watchdog added in response to two observed silent freezes during Branch 3 adaptive YOLO crop deployment (gaps of 5.3 hours at 06:30-11:46 UTC and 5.9 hours at 14:11-20:07 UTC). Root cause not fully diagnosed; suspected culprits include synchronous Gemini LLM calls, stacked Pushover retries on flaky network, and `ReportBuilder` parsing corrupted JSONL. The watchdog serves as a defense-in-depth measure regardless of the specific cause.

---

## Web dashboard

The web dashboard runs as its own systemd unit (`avis-web.service`),
independent of `avis.service`. Stopping or restarting one never
affects the other. Full operator guide — including Tailscale invite
flow and the ngrok demo helper — is in
[`WEB_DASHBOARD.md`](WEB_DASHBOARD.md). Quick install on the Pi:

### 1. Add the token to `.env`

Generate a random token (≥16 chars) on the laptop:

```bash
python -c "import secrets; print(secrets.token_urlsafe(24))"
```

Append to `/mnt/data/avis-birdfeeder/.env`:

```
AVIS_WEB_TOKEN=<paste it>
```

The dashboard refuses to start if the token is missing or shorter
than 16 chars.

### 2. Install the systemd unit

```bash
sudo cp scripts/avis-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable avis-web
sudo systemctl start avis-web
sudo systemctl status avis-web
```

### 3. Verify

```bash
curl http://127.0.0.1:8000/health
```

Expected: `{"status":"ok","service":"avis-web","version":"0.1.0"}`.

Open from the laptop (Tailscale connected):

```
http://birdfeeder01:8000/?token=<your token>
```

The SPA strips `?token=` from the URL after first load. Bookmark the
bare URL.

### 4. Class demo via ngrok

When Tailscale isn't an option (audience laptops aren't on the
tailnet), run:

```bash
bash scripts/avis-web-ngrok.sh
```

The script prints a one-shot demo URL with `?token=` filled in. Ctrl-C
tears it down. See `WEB_DASHBOARD.md` for the one-time ngrok install
+ authtoken steps.

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

### Microphone not capturing

Audio capture resolves the sounddevice index by name (`microphone.device_name`
in `hardware.yaml`) with fallback to `microphone.device_index`. On startup,
the Avis service logs which resolution path was taken:

or, if the name didn't match any device:

**If audio is silently failing** (`audio_result: null` on every observation),
check the startup logs first:

```powershell
pi-logs-since "5 min ago" | findstr /i "audio device"
```

From the output, decide:

- **No "Audio device resolved" line at all** → `AudioCapture` never initialized.
  Check for an earlier exception in the service logs.
- **"Falling back to device_index"** → the configured `device_name` substring
  didn't match any connected device. List what sounddevice actually sees:

```powershell
  pi-ssh
```
  Then on the Pi:
```bash
  source /mnt/data/avis-venv/bin/activate
  python -c "import sounddevice as sd; [print(i, d['name']) for i, d in enumerate(sd.query_devices())]"
```

  Update `microphone.device_name` in `hardware.yaml` to a distinctive
  substring of the actual device name. Avoid matching on the volatile
  `hw:X,Y` ALSA address — prefer the product description.

- **"Resolved by name → index N" but no audio in logs afterwards** → sounddevice
  found the device but recording is failing at runtime. Likely a USB disconnect
  or permission issue. Check `dmesg` for USB errors and confirm the Pi user
  has audio group membership:

```bash
  groups birdfeeder01
  # should include: audio
```

After any config change, restart the service via `pi-restart` and re-check
logs with `pi-logs`.

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

---


### Service restarts every ~5 minutes

If the service keeps getting killed and restarted by the watchdog, the Python loop is genuinely stalling within the `WatchdogSec` window. Check logs for the pattern that repeats just before each kill:

```bash
sudo journalctl -u avis.service --since "30 minutes ago" --no-pager | grep -iE "error|traceback|exception|timeout"
```

Common culprits during our deployment:

- Long Pushover retry chains (77s timeouts × 3 attempts) when network drops
- Synchronous Gemini LLM calls hanging with no explicit timeout
- `ReportBuilder` stuck parsing corrupted JSONL in observations.jsonl

As a temporary mitigation while diagnosing, you can extend the watchdog window:

```bash
sudo systemctl edit avis.service
# Change WatchdogSec=300 to WatchdogSec=900 (15 min)
sudo systemctl daemon-reload
sudo systemctl restart avis.service
```

### Watchdog not triggering on freeze

If `sudo kill -STOP <PID>` doesn't lead to an automatic restart within 5 minutes, the watchdog is not armed correctly. Check for these common issues:

1. `sdnotify` is not installed in the venv:
```bash
   /mnt/data/avis-venv/bin/python -c "import sdnotify; print(sdnotify.__file__)"
```
   If this errors, install it: `source /mnt/data/avis-venv/bin/activate && pip install sdnotify`.

2. The systemd override wasn't reloaded:
```bash
   sudo systemctl cat avis.service | grep WatchdogSec
```
   If no `WatchdogSec` appears, the override didn't take. Re-run `sudo systemctl edit avis.service`, re-add the override, then `sudo systemctl daemon-reload`.

3. The orchestrator log shows `sdnotify unavailable`. Restart the service after installing sdnotify: `sudo systemctl restart avis.service`.

---


---

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