# Raspberry Pi Connect remote-shell setup and benchmark runbook

This is the standalone remote-shell procedure for the formal Raspberry Pi 5
benchmark. Use Raspberry Pi Connect Remote Shell for every command run on the
Pi. It works from a browser without finding the Pi's IP address, opening router
ports, or exposing SSH to the internet. If the Wi-Fi blocks device-to-device
connections, use Tailscale only as the private `rsync` transport for the ignored
artifacts and returned evidence. The browser shell itself cannot receive files
with `rsync` or `scp`. Use `tmux` so a Connect session closing does not stop a
benchmark. Do not use laptop timings as Pi evidence and do not add
`--allow-non-pi` to a formal run.

The fixed saved-image matrix must run first. The IMX219 camera matrix must use a
different output directory because it measures capture and frame-age overhead
in addition to model execution.

## Remote-access choice and first-time Connect setup

For this project, Raspberry Pi Connect is the control path. The guide uses:

- **Raspberry Pi Connect Remote Shell:** installation, camera checks, registry
  creation, benchmarks, troubleshooting, and monitoring;
- **Tailscale SSH/rsync:** one payload upload before the run and one evidence
  download afterward when the Wi-Fi blocks local SSH. No router port is opened.

Raspberry Pi Connect requires Raspberry Pi OS Bookworm or later. On the first
boot, either link Connect in Raspberry Pi Imager or run the following once from
a local Pi terminal:

```bash
sudo apt update
command -v rpi-connect || sudo apt install -y rpi-connect-lite
rpi-connect on
rpi-connect shell on
rpi-connect signin
loginctl enable-linger
rpi-connect status
```

Open the verification URL printed by `rpi-connect signin`, sign in with the
same Raspberry Pi ID, and name the device. `loginctl enable-linger` keeps the
user-level Connect service reachable after a headless reboot.

For every later Pi terminal session:

1. Open <https://connect.raspberrypi.com/> in a browser.
2. Select the Pi under **Devices**.
3. Select **Connect via → Remote shell**.
4. Start or reattach `tmux` before a long-running benchmark.

### If the Connect Remote Shell is already open

Do not reinstall or sign in to Connect again. Remote access is solved, but the
Pi still needs the repository, ignored artifacts, runtime environment, camera
gate, OpenVINO smoke, and production registry before formal timing.

If this command prints `READY_FOR_PRE_RUN`, the setup was already completed and
you may skip directly to section 10, then run the image and camera matrices:

```bash
cd ~/capstone-pi-benchmarking

test -x .venv/bin/python && \
test -f configs/pi-model-registry.json && \
test -f artifacts/models/cane-v1/best.pt && \
test -f artifacts/benchmarks/pi-runtime-inputs/manifest.json && \
rpicam-hello --list-cameras 2>&1 | grep -qi imx219 && \
echo READY_FOR_PRE_RUN
```

If it does not print the marker, continue from section 1 if the camera still
needs fitting, or section 2 if it is already connected. Having a remote shell
does not by itself mean the benchmark dependencies and model artifacts exist.

## 0. What “ready” means

Do not start formal timing until every item below passes:

- the Pi reports a 64-bit `aarch64` Raspberry Pi OS environment;
- the active cooler and final power supply are fitted;
- the repository code, production exports, registry inputs, and 30-image fixed
  pack have been transferred;
- `.venv/bin/python` can import Ultralytics, ONNX Runtime, OpenVINO, MNN,
  NCNN, LiteRT, `psutil`, and Picamera2;
- all 30 saved images pass their manifest hashes;
- the IMX219 appears as camera 0 and a headless test image can be captured;
- OpenVINO passes a functional smoke on the Pi and that checksum-bound evidence
  is merged into the export manifest;
- `configs/pi-model-registry.json` is generated successfully;
- `vcgencmd get_throttled` is `0x0` before measurement;
- the provisional top-handle mount has been physically checked and its actual
  geometry recorded.

If any item fails, fix it before running the full matrix. A saved failed row is
useful diagnostic evidence, but it is not a completed benchmark.

## 1. Power down before connecting the IMX219

From the Raspberry Pi Connect Remote Shell:

```bash
sudo poweroff
```

Wait until the Pi has completely stopped before connecting or reseating the CSI
cable. Fit the Arducam 8MP IMX219 near the top handle, with the lens facing
forward. The provisional profile is 40 mm below the grip reference, 12 degrees
downward pitch, and 0 degrees roll. Do not connect the ribbon cable while the Pi
is powered.

Power the Pi back on, return to the Connect dashboard, and open **Connect via →
Remote shell** again. If the device remains offline after boot, see the Connect
row in the common-mistakes table.

## 2. Transfer the exact Pi payload from the source machine

Git intentionally ignores trained weights, runtime exports, benchmark inputs,
and generated evidence. A clone alone is therefore not enough. From a terminal
on the source Mac, after local finalization has produced
`artifacts/models/cane-v1/model-comparison-640.json`, run:

First, in the Connect Remote Shell, clone or update the tracked repository:

```bash
cd ~
sudo apt update
sudo apt install -y git curl

if test -d capstone-pi-benchmarking/.git; then
  git -C capstone-pi-benchmarking pull --ff-only
elif ! test -d capstone-pi-benchmarking; then
  git clone https://github.com/ctxnn/capstone-pi-benchmarking.git
fi
```

The model weights, exports, comparison report, fixed inputs, and training summary
are intentionally ignored by Git. If LAN SSH does not work, create a private
Tailscale transport from the same Connect shell:

```bash
command -v tailscale || curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
sudo systemctl enable --now ssh
tailscale ip -4
```

Open the authentication URL printed by `tailscale up`. Install Tailscale on the
source Mac, sign in to the same tailnet, and replace `TAILSCALE_IP` below with
the Pi's `100.x.y.z` address. Then run from the source Mac terminal:

```bash
cd /Users/chiragtaneja/Codes/capstone-everything/yolo-pi

rsync -a --info=progress2 --relative \
  ./README.md \
  ./pyproject.toml \
  ./uv.lock \
  ./configs \
  ./docs \
  ./scripts \
  ./src \
  ./artifacts/models/yolo26n.pt \
  ./artifacts/models/bus.jpg \
  ./artifacts/models/cane-v1 \
  ./artifacts/benchmarks/pi-runtime-inputs \
  ./artifacts/training/lightning-final/training-summary.json \
  PI_USER@TAILSCALE_IP:~/capstone-pi-benchmarking/
```

After transfer, return to Raspberry Pi Connect Remote Shell for every setup and
benchmark command. Tailscale is only the file-transfer path. Do not configure
router port forwarding or expose port 22 publicly.

Do not transfer `.venv`, raw datasets, processed training datasets, provider
checkpoints, or old benchmark runs. Python environments are platform-specific;
the Mac `.venv` cannot run on ARM Linux.

If these ignored artifacts are already present and section 6 succeeds, skip the
transfer. Otherwise the `rsync` step is required even after cloning GitHub. If
Tailscale cannot be used, copy the same paths with a USB drive; the browser
Remote Shell has no direct file-upload channel.

## 3. Record the Pi and OS identity

In the Connect Remote Shell:

```bash
cd ~/capstone-pi-benchmarking

cat /proc/device-tree/model; echo
uname -a
uname -m
getconf LONG_BIT
cat /etc/os-release
python3 --version
free -h
df -h .
date -u
timedatectl status
```

Required results:

- `/proc/device-tree/model` identifies the intended Raspberry Pi;
- `uname -m` is `aarch64`;
- `getconf LONG_BIT` is `64`;
- system time is synchronized;
- there is enough free storage for the environment, models, raw rows, and two
  complete run directories.

Do not proceed on a 32-bit OS. Do not change OS packages, the cooler, swap,
governor, enclosure, or power supply halfway through the matrix.

## 4. Install OS packages

Use Raspberry Pi OS Bookworm or later. Install Picamera2 from `apt`, not from
PyPI, so it stays compatible with the OS `libcamera` stack:

```bash
sudo apt update
sudo apt install -y \
  git curl build-essential cmake pkg-config \
  python3-dev python3-venv \
  python3-picamera2 rpicam-apps \
  libgl1 libglib2.0-0 libgomp1 tmux jq
```

Install `uv` as the normal Pi user, never with `sudo`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv --version
```

Log out and reconnect if `uv` is still not on `PATH`.

## 5. Create the Pi Python environment correctly

Picamera2 is supplied by the OS. The project virtual environment must therefore
use `/usr/bin/python3` and expose system site-packages:

```bash
cd ~/capstone-pi-benchmarking

uv venv --python /usr/bin/python3 --system-site-packages

uv sync --frozen \
  --extra pi \
  --extra edge \
  --extra runtime-onnx \
  --extra runtime-openvino \
  --extra runtime-mnn \
  --extra runtime-litert
```

On Linux ARM64, the project pins `torch` and `torchvision` to PyTorch's official
CPU wheel index. If the progress display mentions `nvidia-cuda`, `nvidia-cudnn`,
`nvidia-nccl`, or `triton`, press `Ctrl+C`; that is the wrong dependency plan
for a Raspberry Pi. Pull the latest `pyproject.toml` and `uv.lock`, clean the
partial cache, recreate `.venv`, and run the command again:

```bash
uv cache clean
mv .venv ".venv.cuda-plan.$(date -u +%Y%m%dT%H%M%SZ)"
uv venv --python /usr/bin/python3 --system-site-packages
uv sync --frozen \
  --extra pi --extra edge --extra runtime-onnx \
  --extra runtime-openvino --extra runtime-mnn \
  --extra runtime-litert
```

Do not use the `export` extra on the Pi. ONNX, OpenVINO, MNN, NCNN, and LiteRT
models were already converted off-device. The Pi only needs their runtime
packages.

Confirm the actual interpreter and imports:

```bash
.venv/bin/python -c "import platform,sys; print(sys.executable); print(platform.machine()); print(platform.python_version())"

.venv/bin/python -c "import torch,ultralytics,psutil; print('torch',torch.__version__); print('ultralytics',ultralytics.__version__); print('psutil',psutil.__version__)"

.venv/bin/python -c "import onnxruntime,openvino,MNN,ncnn; print('ONNX Runtime',onnxruntime.__version__); print('OpenVINO',openvino.__version__); print('MNN OK'); print('NCNN OK')"

.venv/bin/python -c "from ai_edge_litert.interpreter import Interpreter; print('LiteRT OK')"

.venv/bin/python -c "from picamera2 import Picamera2; print(Picamera2.global_camera_info())"
```

If the last command says Picamera2 is missing, do not `pip install picamera2`.
The usual cause is a virtual environment created without
`--system-site-packages` or with a uv-downloaded Python instead of
`/usr/bin/python3`.

For a bad environment, preserve it for diagnosis and rebuild:

```bash
mv .venv ".venv.bad.$(date -u +%Y%m%dT%H%M%SZ)"
uv venv --python /usr/bin/python3 --system-site-packages
uv sync --frozen \
  --extra pi --extra edge --extra runtime-onnx \
  --extra runtime-openvino --extra runtime-mnn \
  --extra runtime-litert
```

## 6. Verify the transferred artifacts and fixed inputs

Check that the required files exist:

```bash
test -f configs/pi-benchmark-matrix.json
test -f artifacts/models/yolo26n.pt
test -f artifacts/models/cane-v1/best.pt
test -f artifacts/models/cane-v1/best.onnx
test -d artifacts/models/cane-v1/best_openvino_model
test -f artifacts/models/cane-v1/best.mnn
test -d artifacts/models/cane-v1/best_ncnn_model
test -f artifacts/models/cane-v1/best.tflite
test -f artifacts/models/cane-v1/export-manifest.json
test -f artifacts/models/cane-v1/model-comparison-640.json
test -f artifacts/training/lightning-final/training-summary.json
test -f artifacts/benchmarks/pi-runtime-inputs/manifest.json
echo "REQUIRED_FILES=present"
```

Verify all 30 fixed images against their manifest:

```bash
PYTHONPATH=src .venv/bin/python -c \
  "from pathlib import Path; from yolo_pi.pi_benchmark import load_fixed_inputs; items=load_fixed_inputs(Path('artifacts/benchmarks/pi-runtime-inputs/manifest.json'), Path.cwd()); print('FIXED_INPUTS',len(items)); assert len(items)==30"
```

Expected result:

```text
FIXED_INPUTS 30
```

A missing or changed image must be retransferred. Do not replace it with a new
picture because that breaks the fixed-input comparison.

## 7. Verify the IMX219 entirely from the remote shell

List detected cameras and sensor modes:

```bash
rpicam-hello --list-cameras
```

Camera 0 should contain `imx219`. Then perform a headless capture:

```bash
rpicam-still \
  --nopreview \
  --timeout 2000 \
  --width 640 \
  --height 480 \
  --output /tmp/imx219-terminal-check.jpg

file /tmp/imx219-terminal-check.jpg
ls -lh /tmp/imx219-terminal-check.jpg
```

Finally test Picamera2 through the project interpreter:

```bash
.venv/bin/python -c \
  "from picamera2 import Picamera2; camera=Picamera2(0); config=camera.create_preview_configuration(main={'size':(640,480),'format':'RGB888'}); camera.configure(config); camera.start(); frame=camera.capture_array(); camera.stop(); camera.close(); print('PICAMERA2_FRAME',frame.shape,frame.dtype)"
```

Expected shape is normally `(480, 640, 3)`. Do not leave `rpicam-*`, another
Picamera2 process, or a camera preview running; only one process can own the
camera during the benchmark.

## 8. Complete the Pi-side OpenVINO functional gate

The OpenVINO export is checksum-complete but must execute on Linux/ARM before the
production registry is created. Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/smoke_export_artifact.py \
  --artifact artifacts/models/cane-v1/best_openvino_model \
  --sample artifacts/models/bus.jpg \
  --output artifacts/models/cane-v1/openvino-pi-smoke.json \
  --format openvino \
  --imgsz 640 \
  --precision FP32
```

The command must exit zero and the JSON must contain `"status": "complete"`.
Merge it only through the checksum gate:

```bash
.venv/bin/python scripts/merge_export_smoke.py \
  --manifest artifacts/models/cane-v1/export-manifest.json \
  --format openvino \
  --evidence artifacts/models/cane-v1/openvino-pi-smoke.json
```

Now build the production registry:

```bash
.venv/bin/python scripts/build_pi_model_registry.py \
  --training-summary artifacts/training/lightning-final/training-summary.json \
  --quality-comparison artifacts/models/cane-v1/model-comparison-640.json \
  --export-manifest artifacts/models/cane-v1/export-manifest.json \
  --baseline artifacts/models/yolo26n.pt \
  --output configs/pi-model-registry.json
```

The registry builder verifies the selected `best.pt`, baseline, quality report,
all export tree hashes, 640-pixel resolution, and complete held-out test split.
If the manifest records the Mac's old absolute path, the builder safely rebases
it by artifact name under `artifacts/models/cane-v1/` and recomputes the tree
hash before writing a Pi-relative path.
Never copy `pi-model-registry.example.json` and call it production evidence.

## 9. Validate the mount contract

```bash
PYTHONPATH=src .venv/bin/python -m yolo_pi.cli validate-mount \
  --profile configs/camera-mount-top-handle.json
```

The JSON mount profile remains `hardware_verified=false` until the actual camera
offset, pitch, roll, hand/cable occlusion, and calibration requirements have
been checked. Passing schema validation is not proof that the physical mount was
measured.

## 10. Stabilize and record the Pi before each matrix

Run these commands and retain their output with the experiment notes:

```bash
cat /proc/device-tree/model; echo
free -h
df -h .
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
vcgencmd measure_temp
vcgencmd get_throttled
ps -eo pid,comm,%cpu,%mem --sort=-%cpu | head -n 15
```

`vcgencmd get_throttled` should be `throttled=0x0`. A non-zero result means
undervoltage, throttling, or a historical throttle condition; fix power/cooling
and reboot before the formal series. Stop browsers, desktops, package updates,
camera previews, and unrelated workloads.

The runner performs 20 excluded warmups, 100 measured repetitions, and waits for
the CPU to cool to 55 C before every row, with a ten-minute timeout. Do not alter
these values midway through the comparison.

## 11. Start a disconnect-safe terminal session

```bash
cd ~/capstone-pi-benchmarking
tmux new -s pi-bench
```

Inside `tmux`, run the commands below. Detach with `Ctrl+B`, then `D`. Reattach
after reopening Connect Remote Shell with:

```bash
tmux attach -t pi-bench
```

A Connect browser/session disconnect does not stop a process running in `tmux`.

## 12. Run the formal fixed-image matrix first

Inside `tmux`:

```bash
cd ~/capstone-pi-benchmarking

IMAGE_RUN_DIR="artifacts/pi-benchmarks/images-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$IMAGE_RUN_DIR"
printf '%s\n' "$IMAGE_RUN_DIR" > artifacts/pi-benchmarks/LATEST_IMAGE_RUN.txt

PYTHONPATH=src .venv/bin/python -u scripts/run_pi_benchmarks.py \
  --matrix configs/pi-benchmark-matrix.json \
  --registry configs/pi-model-registry.json \
  --output "$IMAGE_RUN_DIR" \
  --source images \
  --threads 4 \
  2>&1 | tee "$IMAGE_RUN_DIR/console.log"
```

This runs exactly:

- M1: pretrained YOLO26n, PyTorch, 640, FP32;
- M2: fine-tuned YOLO26n-Cane V1, PyTorch, 640, FP32;
- R1: PyTorch;
- R2: ONNX Runtime;
- R3: OpenVINO;
- R4: MNN with the explicit high-precision/four-thread guard;
- R5: NCNN;
- R6: LiteRT.

The command is resumable. Every completed row is saved immediately under
`rows/` and skipped on rerun.

After an interruption:

```bash
IMAGE_RUN_DIR="$(cat artifacts/pi-benchmarks/LATEST_IMAGE_RUN.txt)"

PYTHONPATH=src .venv/bin/python -u scripts/run_pi_benchmarks.py \
  --matrix configs/pi-benchmark-matrix.json \
  --registry configs/pi-model-registry.json \
  --output "$IMAGE_RUN_DIR" \
  --source images \
  --threads 4 \
  2>&1 | tee -a "$IMAGE_RUN_DIR/console.log"
```

Retry only one failed row after fixing its cause:

```bash
IMAGE_RUN_DIR="$(cat artifacts/pi-benchmarks/LATEST_IMAGE_RUN.txt)"

PYTHONPATH=src .venv/bin/python -u scripts/run_pi_benchmarks.py \
  --matrix configs/pi-benchmark-matrix.json \
  --registry configs/pi-model-registry.json \
  --output "$IMAGE_RUN_DIR" \
  --source images \
  --threads 4 \
  --row R4 \
  --force \
  2>&1 | tee -a "$IMAGE_RUN_DIR/console.log"
```

Never use `--force` on the whole matrix unless the entire series is intentionally
being invalidated and rerun.

## 13. Check the fixed-image outputs

```bash
IMAGE_RUN_DIR="$(cat artifacts/pi-benchmarks/LATEST_IMAGE_RUN.txt)"

find "$IMAGE_RUN_DIR/rows" -maxdepth 1 -type f -name '*.json' -print | sort
jq -r '[.row.id,.status,.artifact.resolved_backend] | @tsv' "$IMAGE_RUN_DIR"/rows/*.json

sed -n '1,220p' "$IMAGE_RUN_DIR/tables/model-architecture-benchmark.md"
sed -n '1,260p' "$IMAGE_RUN_DIR/tables/runtime-backend-benchmark.md"

jq . "$IMAGE_RUN_DIR/tables/table-manifest.json"
vcgencmd get_throttled
```

All eight row files, M1, M2, and R1–R6, must say `complete`. Check that R4's
resolved backend begins with `mnn-high-precision-`. If the console says MNN used
low precision, discard and rerun R4 after fixing the runtime; never relabel it
FP32.

## 14. Run the IMX219 camera matrix separately

Make sure no camera test process is still running, then create a new directory:

```bash
CAMERA_RUN_DIR="artifacts/pi-benchmarks/camera-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$CAMERA_RUN_DIR"
printf '%s\n' "$CAMERA_RUN_DIR" > artifacts/pi-benchmarks/LATEST_CAMERA_RUN.txt

PYTHONPATH=src .venv/bin/python -u scripts/run_pi_benchmarks.py \
  --matrix configs/pi-benchmark-matrix.json \
  --registry configs/pi-model-registry.json \
  --output "$CAMERA_RUN_DIR" \
  --source camera \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --camera-num 0 \
  --camera-pixel-format RGB888 \
  --threads 4 \
  2>&1 | tee "$CAMERA_RUN_DIR/console.log"
```

The camera producer continually overwrites a one-slot latest-frame buffer.
Inference receives the newest available frame instead of draining a stale FIFO.
The rows record capture latency, frame age when available, frame age at model
start, sensor-to-model-finish latency, capture-start-to-finish latency, and
overwritten frames.

Resume the same camera directory after interruption:

```bash
CAMERA_RUN_DIR="$(cat artifacts/pi-benchmarks/LATEST_CAMERA_RUN.txt)"

PYTHONPATH=src .venv/bin/python -u scripts/run_pi_benchmarks.py \
  --matrix configs/pi-benchmark-matrix.json \
  --registry configs/pi-model-registry.json \
  --output "$CAMERA_RUN_DIR" \
  --source camera \
  --camera-width 640 --camera-height 480 --camera-fps 30 \
  --camera-num 0 --camera-pixel-format RGB888 \
  --threads 4 \
  2>&1 | tee -a "$CAMERA_RUN_DIR/console.log"
```

Do not use the fixed-image directory for camera rows. The runner skips completed
row IDs, so mixing source types would silently preserve the first source's rows.

## 15. Optional measured power

### Live annotated preview after benchmarking

Use Raspberry Pi Connect **Screen Sharing**, open a terminal on the Pi desktop,
and run the qualitative demo after the formal matrices finish:

```bash
cd ~/capstone-pi-benchmarking
PYTHONPATH=src .venv/bin/python scripts/live_camera_demo.py
```

The preview uses the validated fine-tuned OpenVINO export by default and draws
boxes, labels, confidence values, inference time, and display FPS. Press `Q` or
`Esc` in the preview window to stop. The camera is released in a `finally`
block. Do not run the preview alongside a benchmark or another camera process.
This visual check is qualitative evidence and does not replace the held-out
quality evaluation or formal benchmark tables.

If M1 completes but later camera rows fail with `Camera in Configured state`
or `Camera __init__ sequence did not complete`, update the runner: the camera
adapter must call Picamera2 `close()` as well as `stop()` between rows. Exit the
old runner and resume the same camera output directory using section 14's
resume command. Completed rows are skipped; failed rows are retried without
`--force`. Copy the old run folder first if retaining the original failure
reports is required.

Power must come from an external meter/logger. The runner does not estimate it
from CPU load. The logger must append newline-delimited JSON with Unix epoch
nanoseconds:

```json
{"timestamp_ns": 1787976000000000000, "power_w": 6.42}
```

Check the live stream before running:

```bash
tail -n 5 artifacts/power/power-samples.jsonl | jq -c .
date -u
timedatectl status
```

Then add this argument to either matrix command:

```text
--power-jsonl artifacts/power/power-samples.jsonl
```

Start the logger before the benchmark and stop it afterward. If there is no
external meter, omit the argument and report Power as `NA`; do not invent a
value.

## 16. Recompile tables without rerunning inference

For the image run:

```bash
IMAGE_RUN_DIR="$(cat artifacts/pi-benchmarks/LATEST_IMAGE_RUN.txt)"

PYTHONPATH=src .venv/bin/python scripts/compile_pi_tables.py \
  --matrix configs/pi-benchmark-matrix.json \
  --registry configs/pi-model-registry.json \
  --runs "$IMAGE_RUN_DIR" \
  --output "$IMAGE_RUN_DIR/tables"
```

For the camera run:

```bash
CAMERA_RUN_DIR="$(cat artifacts/pi-benchmarks/LATEST_CAMERA_RUN.txt)"

PYTHONPATH=src .venv/bin/python scripts/compile_pi_tables.py \
  --matrix configs/pi-benchmark-matrix.json \
  --registry configs/pi-model-registry.json \
  --runs "$CAMERA_RUN_DIR" \
  --output "$CAMERA_RUN_DIR/tables"
```

Each directory retains raw observations, failures, environment identity, model
hashes, cooldown state, throttling state, CSV tables, Markdown tables, and the
table manifest.

## 17. Copy the complete evidence back with Tailscale rsync

From the source Mac terminal:

```bash
cd /Users/chiragtaneja/Codes/capstone-everything/yolo-pi

rsync -a --info=progress2 \
  PI_USER@TAILSCALE_IP:~/capstone-pi-benchmarking/artifacts/pi-benchmarks/ \
  artifacts/pi-benchmarks/

rsync -a --info=progress2 \
  PI_USER@TAILSCALE_IP:~/capstone-pi-benchmarking/artifacts/models/cane-v1/openvino-pi-smoke.json \
  artifacts/models/cane-v1/

rsync -a --info=progress2 \
  PI_USER@TAILSCALE_IP:~/capstone-pi-benchmarking/artifacts/models/cane-v1/export-manifest.json \
  artifacts/models/cane-v1/export-manifest.json

rsync -a --info=progress2 \
  PI_USER@TAILSCALE_IP:~/capstone-pi-benchmarking/configs/pi-model-registry.json \
  configs/pi-model-registry.json
```

Do not copy only screenshots or final CSV files. The row JSON and environment
files are required to audit the tables.

## 18. Common mistakes and exact fixes

| Mistake or symptom | Cause | Fix |
|---|---|---|
| `pi-model-registry.json` missing | Git clone contained only tracked scaffold | Transfer the ignored artifacts, run the Pi OpenVINO smoke, merge it, then build the production registry |
| `Picamera2 is required` or import fails | `.venv` cannot see OS packages | Rebuild with `/usr/bin/python3` and `--system-site-packages`; keep Picamera2 installed through `apt` |
| `No cameras available` | Pi cannot detect the sensor | Power down, reseat/orient the CSI cable, boot, and rerun `rpicam-hello --list-cameras` |
| Camera is busy | Another `rpicam-*` or Python process owns it | Stop the other process; do not run the headless test and benchmark simultaneously |
| Trying `raspistill`, `raspivid`, or legacy `picamera` | Legacy camera stack instructions | Use Bookworm `rpicam-*` and Picamera2 |
| Preview fails in the remote shell | Browser remote shells do not provide a normal camera preview window | Use `--nopreview`; the benchmark itself is headless |
| Formal run accepts laptop | `--allow-non-pi` was added | Remove it; that option is diagnostic only and never paper evidence |
| Fixed-input count is not 30 | Partial transfer or changed file | Retransfer the complete `pi-runtime-inputs` directory; do not substitute images |
| Row says checksum mismatch | Artifact differs from the registry | Retransfer the exact artifact; never edit the registry hash to match an unknown file |
| R4 reports low precision | Native MNN did not use the guarded FP32 path | Discard R4, fix the MNN install/adapter, and rerun only R4 with `--force` |
| OpenVINO registry build refuses | Linux smoke was not merged or tree hash differs | Rerun the Pi smoke against the transferred directory and merge with `merge_export_smoke.py` |
| Connect device stays offline after reboot | Connect is a user-level service and no user session is active | Run `loginctl enable-linger` as the signed-in Pi user, then check `rpi-connect status` |
| Connect shell works but LAN SSH/rsync fails | The Wi-Fi isolates clients or blocks port 22 | Keep using Connect for commands; install Tailscale on the Pi and Mac and use the Pi's `100.x.y.z` address only for `rsync` |
| Git clone succeeded but model files are missing | Large artifacts are intentionally ignored by Git | Run the Tailscale `rsync` payload step or copy the same artifact paths from USB |
| `Killed` or Connect session disconnect | 2 GB memory pressure or browser/session loss | Use `tmux`; keep swap policy fixed; stop unrelated workloads; resume the same run directory |
| Old rows are unexpectedly skipped | Output directory already contains complete row IDs | Use the saved run directory only for resuming; use a new timestamped directory for a new series |
| Camera run shows image-source rows | Image and camera runs used the same directory | Create a new camera directory; never mix source types |
| Power is `NA` | No timestamp-aligned external JSONL samples | This is correct without a meter; do not estimate power |
| Non-zero `get_throttled` | Undervoltage or thermal history | Fix the supply/cooling, reboot, and recheck before formal timing |
| Tables contain `NA` after a failed row | The compiler preserves missing evidence honestly | Inspect `rows/<ID>.json`, fix the cause, rerun only that row, then recompile |
| `uv sync` tries to build export tools | The `export` extra was selected | Use `runtime-onnx` and the other runtime extras shown in this guide |
| `uv sync` downloads CUDA, cuDNN, NCCL, or Triton | PyTorch was resolved from the GPU-enabled PyPI Linux wheel set | Stop with `Ctrl+C`, update `pyproject.toml` and `uv.lock`, run `uv cache clean`, recreate `.venv`, and sync from the CPU-bound lock |
| `sudo uv` or system Python becomes inconsistent | Project packages were installed into the OS interpreter | Install/run uv as the normal user and keep project packages inside `.venv` |

## 19. Final evidence checklist

Before calling the Pi benchmark complete, retain and return:

- the Pi environment identity and `get_throttled` outputs;
- `configs/pi-model-registry.json`;
- the merged export manifest and `openvino-pi-smoke.json`;
- the complete fixed-image timestamped directory;
- the complete camera timestamped directory;
- all `rows/*.json`, including failures and reruns;
- both CSV and Markdown tables plus `table-manifest.json`;
- the external power JSONL if one was used;
- the measured top-handle camera geometry and calibration/occlusion evidence.

The model-quality values in M1/M2 are already bound to the full held-out test
split. Pi latency, FPS, CPU, RAM, temperature, camera timing, dropped frames, and
power become valid only after this terminal procedure runs on the physical Pi.

## Official command references

- [Raspberry Pi Connect and Remote Shell](https://www.raspberrypi.com/documentation/services/connect.html)
- [Tailscale installation on Raspberry Pi](https://tailscale.com/kb/1076/dogcam)
- [Raspberry Pi camera software and `rpicam-*`](https://www.raspberrypi.com/documentation/computers/camera_software.html)
- [uv installation](https://docs.astral.sh/uv/getting-started/installation/)
- [uv project synchronization](https://docs.astral.sh/uv/concepts/projects/sync/)
