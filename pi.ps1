# pi.ps1 — Avis Pi management shortcuts
# Usage: . .\pi.ps1    (dot-source to load functions into your session)

$PI = "birdfeeder01@192.168.4.76"
$PI_REPO = "/mnt/data/avis-birdfeeder"

function pi-ssh {
    ssh $PI
}

function pi-status {
    ssh $PI "sudo systemctl status avis"
}

function pi-logs {
    ssh $PI "sudo journalctl -u avis -f"
}

function pi-stop {
    ssh $PI "sudo systemctl stop avis"
    Write-Host "Avis service stopped." -ForegroundColor Yellow
}

function pi-start {
    ssh $PI "sudo systemctl start avis"
    Write-Host "Avis service started." -ForegroundColor Green
}

function pi-restart {
    ssh $PI "sudo systemctl restart avis"
    Write-Host "Avis service restarted." -ForegroundColor Green
}

function pi-run {
    # Stop service and run orchestrator manually (60s smoke test)
    ssh $PI "sudo systemctl stop avis && cd $PI_REPO && source /mnt/data/avis-venv/bin/activate && timeout 60 python -m src.agent.experiment_orchestrator || true"
}

function pi-pull {
    # Pull latest code and restore Pi local configs
    ssh $PI "cd $PI_REPO && source /mnt/data/avis-venv/bin/activate && git pull origin main && bash scripts/dev_config.sh"
}

function pi-deploy {
    # Full deploy: pull, restore configs, restart service
    pi-pull
    pi-restart
    pi-status
}

Write-Host "Pi shortcuts loaded:" -ForegroundColor Cyan
Write-Host "  pi-ssh      — open SSH session" -ForegroundColor White
Write-Host "  pi-status   — check service status" -ForegroundColor White
Write-Host "  pi-logs     — stream live logs" -ForegroundColor White
Write-Host "  pi-stop     — stop service" -ForegroundColor White
Write-Host "  pi-start    — start service" -ForegroundColor White
Write-Host "  pi-restart  — restart service" -ForegroundColor White
Write-Host "  pi-run      — stop service and run 60s manual smoke test" -ForegroundColor White
Write-Host "  pi-pull     — pull latest code + restore Pi configs" -ForegroundColor White
Write-Host "  pi-deploy   — pull + restart service" -ForegroundColor White