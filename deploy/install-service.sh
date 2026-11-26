#!/usr/bin/env bash
# Install the cane as a systemd service that starts at boot and restarts on failure.
#
#   bash deploy/install-service.sh --print   # show the unit without installing it
#   bash deploy/install-service.sh           # install, enable and start it
#
# The unit is generated for the invoking user and this checkout, so it works
# whatever the home directory is. No network is needed at boot.
set -euo pipefail

PRINT=0
if [[ "${1:-}" == "--print" ]]; then
  PRINT=1
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--print]" >&2
  exit 2
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-${USER}}"
if [[ -z "${RUN_USER}" || "${RUN_USER}" == "root" ]]; then
  echo "run this as the user that owns the cane checkout (not root); sudo is used internally" >&2
  exit 1
fi
RUN_HOME="$(getent passwd "${RUN_USER}" | cut -d: -f6)"
RUN_GROUP="$(id -gn "${RUN_USER}")"
PYTHON="${REPO}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "no ${PYTHON}; create the venv first (docs/pi-terminal-setup-and-run.md)" >&2
  exit 1
fi

UNIT="$(cat <<UNIT
[Unit]
Description=Smart assistive cane (sensors, vision, haptics)
After=multi-user.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=${RUN_USER}
Group=${RUN_GROUP}
SupplementaryGroups=i2c gpio video
WorkingDirectory=${REPO}
Environment=PYTHONPATH=${REPO}/src
Environment=PYTHONUNBUFFERED=1
ExecStart=${PYTHON} -m yolo_pi.cane --config configs/cane-hardware.json run
Restart=always
RestartSec=2
TimeoutStopSec=20

[Install]
WantedBy=multi-user.target
UNIT
)"

if [[ "${PRINT}" == "1" ]]; then
  printf '%s\n' "${UNIT}"
  exit 0
fi

missing=()
for group in i2c gpio video; do
  if ! id -nG "${RUN_USER}" | tr ' ' '\n' | grep -qx "${group}"; then
    missing+=("${group}")
  fi
done
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "warning: ${RUN_USER} is not in: ${missing[*]}" >&2
  echo "         run: sudo usermod -aG ${missing[*]// /,} ${RUN_USER}  and reboot" >&2
fi

printf '%s\n' "${UNIT}" | sudo tee /etc/systemd/system/yolo-pi-cane.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now yolo-pi-cane

echo
echo "installed /etc/systemd/system/yolo-pi-cane.service for ${RUN_USER} in ${REPO}"
echo "  systemctl status yolo-pi-cane --no-pager"
echo "  journalctl -u yolo-pi-cane -f"
echo "  sudo systemctl stop yolo-pi-cane    # before running the cane by hand"
