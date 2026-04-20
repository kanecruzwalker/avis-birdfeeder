#!/bin/bash
#
# install_service.sh — one-command Pi systemd setup for Avis
#
# Copies scripts/avis.service to /etc/systemd/system/, enables the service
# so it starts on every boot, starts it now, and prints its current status.
#
# Run once on the Pi after first-time setup, or any time the systemd unit
# file in scripts/avis.service is edited. Re-running is safe — systemd
# reload and enable are idempotent.
#
# Usage (from project root on the Pi):
#
#     bash scripts/install_service.sh
#
# Requires sudo (prompts for password). The service runs as the birdfeeder01
# user, not as root — see scripts/avis.service for user / permissions detail.

set -e
echo "Installing Avis systemd service..."
sudo cp scripts/avis.service /etc/systemd/system/avis.service
sudo systemctl daemon-reload
sudo systemctl enable avis
sudo systemctl start avis
sudo systemctl status avis
echo "Done. Avis will now start automatically on boot."