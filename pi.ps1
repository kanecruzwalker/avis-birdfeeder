# pi.ps1 — Avis Pi management shortcuts
#
# Laptop-side PowerShell functions for managing the deployed Avis system on
# the Raspberry Pi 5 via SSH. Dot-source this file to load the functions
# into the current PowerShell session:
#
#     . .\pi.ps1
#
# Functions will be available for the rest of the session. Add the dot-source
# line to your PowerShell $PROFILE to have them loaded in every session.
#
# ── Configuration ────────────────────────────────────────────────────────────
# Set $env:AVIS_PI_HOST in your PowerShell profile to override the default
# host. Useful for per-user IPs  and for
# switching between LAN hostname and static IP if mDNS doesn't resolve.
#
#     $env:AVIS_PI_HOST = "birdfeeder01@192.168.4.76"   # static IP
#     $env:AVIS_PI_HOST = "birdfeeder01@birdfeeder.local"  # mDNS
#
# Default falls back to the LAN mDNS hostname. No hardware IPs in the repo.

$PI = if ($env:AVIS_PI_HOST) { $env:AVIS_PI_HOST } else { "birdfeeder01@birdfeeder.local" }
$PI_REPO = "/mnt/data/avis-birdfeeder"
$PI_VENV = "/mnt/data/avis-venv"

# ── Connection ───────────────────────────────────────────────────────────────

function pi-ssh {
    <#
    .SYNOPSIS
      Open an interactive SSH session to the Pi.
    #>
    ssh $PI
}

# ── Service lifecycle ────────────────────────────────────────────────────────

function pi-status {
    <#
    .SYNOPSIS
      Check the systemd status of the avis service.
    #>
    ssh $PI "sudo systemctl status avis"
}

function pi-logs {
    <#
    .SYNOPSIS
      Stream live journalctl output from the avis service.
    .NOTES
      Press Ctrl+C to stop streaming.
    #>
    ssh $PI "sudo journalctl -u avis -f"
}

function pi-logs-since {
    <#
    .SYNOPSIS
      Print avis service logs since a given time, non-streaming.
    .PARAMETER Since
      systemd-recognized time spec. Examples: "today", "1 hour ago",
      "2026-04-19 15:00", "30 min ago". Default: "today".
    .EXAMPLE
      pi-logs-since
      pi-logs-since "1 hour ago"
      pi-logs-since "2026-04-19 15:00"
    #>
    param(
        [string]$Since = "today"
    )
    ssh $PI "sudo journalctl -u avis --since `"$Since`" --no-pager"
}

function pi-stop {
    <#
    .SYNOPSIS
      Stop the avis service.
    #>
    ssh $PI "sudo systemctl stop avis"
    Write-Host "Avis service stopped." -ForegroundColor Yellow
}

function pi-start {
    <#
    .SYNOPSIS
      Start the avis service.
    #>
    ssh $PI "sudo systemctl start avis"
    Write-Host "Avis service started." -ForegroundColor Green
}

function pi-restart {
    <#
    .SYNOPSIS
      Restart the avis service.
    #>
    ssh $PI "sudo systemctl restart avis"
    Write-Host "Avis service restarted." -ForegroundColor Green
}

# ── Manual execution ─────────────────────────────────────────────────────────

function pi-run {
    <#
    .SYNOPSIS
      Stop the service and run the orchestrator manually for N seconds.
    .PARAMETER Seconds
      Timeout in seconds. Default: 60.
    .EXAMPLE
      pi-run        # 60-second smoke test
      pi-run 180    # 3-minute smoke test
    .NOTES
      Use this for interactive debugging where you want to see live stdout
      from the orchestrator. Exits automatically after the timeout. The
      systemd service is left stopped — run pi-start after you're done.
    #>
    param(
        [int]$Seconds = 60
    )
    ssh $PI "sudo systemctl stop avis && cd $PI_REPO && source $PI_VENV/bin/activate && timeout $Seconds python -m src.agent.experiment_orchestrator || true"
}

# ── Config management ────────────────────────────────────────────────────────

function pi-config-check {
    <#
    .SYNOPSIS
      Validate that all three YAML config files on the Pi parse cleanly.
    .NOTES
      Catches YAML corruption (e.g. from manual nano edits) before a service
      restart hits an unparseable file and crash-loops. Run after any manual
      edit to configs/ on the Pi.
    #>
    ssh $PI "cd $PI_REPO && source $PI_VENV/bin/activate && python -c 'import yaml; [yaml.safe_load(open(p)) for p in [\"configs/hardware.yaml\", \"configs/notify.yaml\", \"configs/thresholds.yaml\"]]; print(\"All three configs parse cleanly.\")'"
}

# ── Deployment ───────────────────────────────────────────────────────────────

function pi-pull {
    <#
    .SYNOPSIS
      Pull the latest code on the Pi and apply Pi-local config overrides.
    .NOTES
      Runs git pull origin main, then scripts/dev_config.py to re-apply
      Pi-local overrides (push: true, hailo: enabled, per-camera crops,
      testing threshold, etc.). Does not restart the service — use
      pi-deploy to pull and restart in one command.
    #>
    ssh $PI "cd $PI_REPO && source $PI_VENV/bin/activate && git pull origin main && python scripts/dev_config.py"
}

function pi-deploy {
    <#
    .SYNOPSIS
      Full deploy: pull latest code, apply overrides, restart service, show status.
    #>
    pi-pull
    pi-restart
    pi-status
}

# ── Help banner ──────────────────────────────────────────────────────────────

Write-Host "Pi shortcuts loaded. Target host: $PI" -ForegroundColor Cyan
Write-Host ""
Write-Host "Connection:"         -ForegroundColor Cyan
Write-Host "  pi-ssh                     open SSH session"          -ForegroundColor White
Write-Host ""
Write-Host "Service lifecycle:"  -ForegroundColor Cyan
Write-Host "  pi-status                  check service status"      -ForegroundColor White
Write-Host "  pi-logs                    stream live logs"          -ForegroundColor White
Write-Host "  pi-logs-since [time]       logs since a time window"  -ForegroundColor White
Write-Host "  pi-stop / pi-start         stop or start service"     -ForegroundColor White
Write-Host "  pi-restart                 restart service"           -ForegroundColor White
Write-Host ""
Write-Host "Manual execution:"  -ForegroundColor Cyan
Write-Host "  pi-run [seconds]           run orchestrator manually, default 60s" -ForegroundColor White
Write-Host ""
Write-Host "Config management:"  -ForegroundColor Cyan
Write-Host "  pi-config-check            validate all YAMLs parse" -ForegroundColor White
Write-Host ""
Write-Host "Deployment:"         -ForegroundColor Cyan
Write-Host "  pi-pull                    git pull + apply overrides" -ForegroundColor White
Write-Host "  pi-deploy                  pull + restart + status"    -ForegroundColor White
Write-Host ""
Write-Host "Override host with `$env:AVIS_PI_HOST" -ForegroundColor DarkGray