#!/usr/bin/env bash
# One-time setup of the cane runtime on a Raspberry Pi 5 that already has the
# repository and the benchmark environment (docs/pi-terminal-setup-and-run.md).
# Safe to run twice. Run from the repository root:  bash deploy/install-cane-on-pi.sh
set -euo pipefail

if [[ ! -x .venv/bin/python ]]; then
  echo "No .venv here. Create it first (docs/pi-terminal-setup-and-run.md, section 'Python environment')." >&2
  exit 1
fi

echo "== apt packages =="
sudo apt update
sudo apt install -y python3-libgpiod python3-lgpio i2c-tools

echo "== enable I2C and set 400 kHz =="
sudo raspi-config nonint do_i2c 0
CONFIG=/boot/firmware/config.txt
[[ -f "$CONFIG" ]] || CONFIG=/boot/config.txt
if ! grep -q '^dtparam=i2c_arm_baudrate=' "$CONFIG"; then
  echo 'dtparam=i2c_arm_baudrate=400000' | sudo tee -a "$CONFIG" >/dev/null
  echo "added dtparam=i2c_arm_baudrate=400000 to $CONFIG (takes effect after a reboot)"
fi

echo "== groups =="
sudo usermod -aG i2c,gpio,video "$USER"

echo "== python drivers =="
uv pip install --python .venv/bin/python -r requirements-cane-hardware.txt

echo "== checks =="
PYTHONPATH=src .venv/bin/python -c "import board, busio, adafruit_vl53l0x, adafruit_tca9548a, lgpio; print('driver imports OK')"
i2cdetect -y 1 || true

cat <<'MSG'

Done. If this was the first run, reboot once (I2C speed, group membership), then:

  PYTHONPATH=src .venv/bin/python -m yolo_pi.cane scan       # what is on each mux channel
  PYTHONPATH=src .venv/bin/python -m yolo_pi.cane selftest   # read every sensor, buzz every motor
  PYTHONPATH=src .venv/bin/python -m yolo_pi.cane run        # the real cane

Then install the service: see deploy/yolo-pi-cane.service.
MSG
