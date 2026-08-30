# Lightning AI training handoff: Cane V1 YOLO26n

> Status on 2026-08-30: training completed epoch 30 and the returned package was
> accepted locally. This file remains the reproducible training/recovery record;
> the active device procedure is `docs/pi-terminal-setup-and-run.md`.

## Short answer

Upload the prepared `lightning-ai-upload` folder, **not the entire repository**.
The prepared folder contains only the four things Lightning needs:

1. the immutable 640-pixel training dataset archive;
2. the resumable epoch-22 checkpoint with optimizer state;
3. the checksum-gated Lightning training runner;
4. start, status, and recovery instructions.

The rest of this repository contains Pi code, old Colab transfers, test assets,
multiple historical checkpoints, backend exports, and caches. Uploading it would
waste time and make it easier to select the wrong model.

Prepared local folder:

```text
/Users/chiragtaneja/Codes/capstone-everything/yolo-pi/lightning-ai-upload/
```

Recommended Studio destination:

```text
/teamspace/studios/this_studio/cane-v1-lightning/
```

## Why Lightning Studio is suitable

Lightning documents that files and installed packages under the Studio home,
including `/teamspace/studios/this_studio`, persist when the machine is stopped
or switched. It also recommends setting up on CPU and switching to a GPU only
when the run is ready. See the official documentation for
[Studio persistence](https://lightning.ai/docs/overview/ai-studio/environment-persistence),
[starting and stopping a Studio](https://lightning.ai/docs/overview/ai-studio/start-and-stop-studio),
and [switching GPUs](https://lightning.ai/docs/platform/build/ai-studio/add-gpus).

Persistence protects the files, but a sleeping or stopped machine does not keep
executing Python. This handoff therefore writes a verified recovery checkpoint
after every completed epoch. If compute stops, start the Studio and run the same
start command again.

## Contents and integrity

The upload folder must contain these files:

| File | Purpose | Expected integrity |
|---|---|---|
| `cane-v1-training-640.tar` | Final decontaminated 640-pixel dataset | 2,013,204,992 bytes; SHA-256 `f469a971a1c0e167b01661f4d2296e4b5e38501f0d6c88fa26f72e34842f8b91` |
| `resume-last-epoch22.pt` | Resumable YOLO checkpoint with optimizer state | SHA-256 `0da75726f7a058e63d7b030601447ddeb4bd98b420d1387482761aa5799a0eea` |
| `lightning_train_resume.py` | Training, recovery, validation, packaging | Runs only after both hashes pass |
| `start-training.sh` | Idempotent background launcher | Refuses to start a duplicate live PID |
| `status-training.sh` | Read-only status and recent log viewer | Shows latest protected epoch and completion receipt |
| `resume-manifest.json` | Human-readable input contract | Target is 30 total epochs at image size 416 |
| `README-FIRST.md` | Short on-platform reminder | No secrets or credentials |
| `PACKAGE-RECEIPT.json` | Local handoff build receipt | Generated locally; records source and scaffold hashes |

The checkpoint has 22 completed epochs. Ultralytics stores the zero-based value
`epoch=21`, which means the next displayed training epoch must be **23/30**. It
also contains optimizer state and the exact 15-class taxonomy.

## Rebuild or verify the local upload folder

Git tracks the handoff scaffold but deliberately ignores the 1.9 GB dataset
archive and optimizer checkpoint. A fresh clone therefore still needs those
exact two binaries restored from the project artifact handoff. By default they
must exist at:

```text
artifacts/training/cane-v1-training-640.tar
artifacts/training/checkpoint-guard/checkpoints/snapshots/last-epoch-022-0da75726f7a0.pt
```

From the repository root, reconstruct the upload folder and write a package
receipt with:

```bash
.venv/bin/python scripts/prepare_lightning_handoff.py
```

Verify an already prepared folder without changing it with:

```bash
.venv/bin/python scripts/prepare_lightning_handoff.py --check
```

The command checks the fixed archive byte count and SHA-256, checkpoint SHA-256,
completed epoch, optimizer state, 15-class taxonomy, and tracked scaffold. It
refuses to replace a mismatched destination. On the same filesystem it uses
hard links to avoid an unnecessary 1.9 GB duplicate; treat the two large files
inside `lightning-ai-upload/` as immutable and never edit them in place.

If the exact binaries were restored elsewhere, pass their paths explicitly:

```bash
.venv/bin/python scripts/prepare_lightning_handoff.py \
  --archive /absolute/path/to/cane-v1-training-640.tar \
  --checkpoint /absolute/path/to/last-epoch-022.pt
```

## Step 1: create the Studio on CPU

1. Open [Lightning AI Studio](https://studio.lightning.ai/).
2. Create a new Studio. A descriptive name such as `cane-v1-training` is fine.
3. Start with the free/default CPU machine. Do not spend GPU time during upload.
4. Open the Studio's VS Code or terminal view.
5. Confirm the persistent home path:

```bash
pwd
ls -ld /teamspace/studios/this_studio
```

Lightning recommends one Studio per project/environment. Do not create a Python
virtual environment inside the Studio; use its existing environment.

## Step 2: upload the prepared folder

### Option A: browser upload

In the Studio file explorer, create or open:

```text
/teamspace/studios/this_studio/cane-v1-lightning
```

Drag all contents of the local `lightning-ai-upload` folder into that directory.
If the browser supports folder upload, dragging the whole prepared folder is
correct. Do not upload the surrounding `yolo-pi` folder.

Lightning also supports browser uploads through its Drive. The official
[Drive documentation](https://lightning.ai/docs/overview/drive) says folders can
be dragged into Drive and files can later be downloaded from the browser or VS
Code. Keep the training copy inside the persistent Studio home shown above.

### Option B: Lightning CLI

This is usually more reliable for the 1.9 GB archive. On the Mac, from the
repository root:

```bash
python3 -m pip install --upgrade lightning-sdk
lightning login
lightning studio cp -r lightning-ai-upload/ \
  lit://OWNER/TEAMSPACE/studios/STUDIO/cane-v1-lightning/
```

Replace `OWNER`, `TEAMSPACE`, and `STUDIO` with the values shown in Lightning.
The current official syntax is documented under
[Lightning Studio CLI file copy](https://lightning.ai/docs/platform/build/ai-studio/cli).

Do not run both upload methods at once. Wait for the 1.9 GB archive to finish.

## Step 3: verify the upload before enabling a GPU

In the Lightning terminal:

```bash
cd /teamspace/studios/this_studio/cane-v1-lightning
ls -lh
sha256sum cane-v1-training-640.tar resume-last-epoch22.pt
python -m json.tool resume-manifest.json
```

The hashes must be exactly:

```text
f469a971a1c0e167b01661f4d2296e4b5e38501f0d6c88fa26f72e34842f8b91  cane-v1-training-640.tar
0da75726f7a058e63d7b030601447ddeb4bd98b420d1387482761aa5799a0eea  resume-last-epoch22.pt
```

Do not train if either value differs. A mismatched large archive usually means
the upload was incomplete; upload that file again.

The script independently repeats both checks before extracting or training, so
copying a command cannot bypass this gate.

## Step 4: prepare the environment while still on CPU

Run:

```bash
cd /teamspace/studios/this_studio/cane-v1-lightning
python -m pip install --upgrade "ultralytics==8.4.132"
python -c "import ultralytics; print(ultralytics.__version__)"
```

The printed version must be `8.4.132`. This is the version used for the previous
22 epochs. Do not upgrade to an unpinned future release mid-training.

The runner deliberately does not reinstall PyTorch. Lightning's machine image
must supply the CUDA-compatible PyTorch build.

## Step 5: switch to a single GPU

Use the machine selector at the top-right of the Studio and switch to one CUDA
GPU. A T4 with 16 GB was used for the previous epochs; an L4, A10G, A100, or
another single GPU with at least 16 GB is also acceptable. A multi-GPU machine is
unnecessary for the remaining eight epochs and changes the execution setup.

Prefer a non-interruptible machine if Lightning offers the choice. Interruptible
machines may be reclaimed, but the per-epoch recovery mechanism still protects
completed epochs.

After the switch finishes:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

The first command must show a GPU, and Python must print `True`.

## Step 6: start the protected training process

Run:

```bash
cd /teamspace/studios/this_studio/cane-v1-lightning
bash start-training.sh
```

The launcher:

- installs only the pinned Ultralytics version;
- refuses to launch if the recorded PID is still alive;
- verifies CUDA;
- starts Python with `nohup` in the background;
- writes the PID and all console output under `lightning-output/`;
- creates a download-ready epoch-22 recovery ZIP at startup and atomically
  replaces it after each newer completed epoch.

The terminal should immediately print a PID. Then monitor:

```bash
bash status-training.sh
```

For a continuously updating view:

```bash
tail -f lightning-output/training-console.log
```

Press `Ctrl+C` only to stop following `tail`; it does not stop training.

## Step 7: enforce the resume gate

The log must show that training resumes, and the first new progress row must be:

```text
23/30
```

It must **not** show `1/30`. If it starts at epoch 1, stop immediately:

```bash
kill "$(cat lightning-output/training.pid)"
```

Then send back:

```text
lightning-output/training-console.log
lightning-output/resume-selection.json
```

Do not allow a fresh 30-epoch run to overwrite the intended continuation.

The expected configuration is:

| Setting | Value |
|---|---:|
| Architecture | YOLO26n |
| Total target epochs | 30 |
| Already completed | 22 |
| Remaining | 8 |
| Image size | 416 |
| Batch | 32 |
| Workers | 4 |
| Patience | 8 |
| Seed | 42 |
| Deterministic | true |
| Precision during training | AMP enabled from original checkpoint args |
| Dataset classes | 15 |

## What is saved after every epoch

After each successful epoch, the runner copies the resumable `last.pt` to:

```text
lightning-output/checkpoint-snapshots/last-epoch-NNN.pt
```

It then verifies the saved epoch, optimizer state, taxonomy, byte count, and
SHA-256 and atomically updates:

```text
lightning-output/LATEST_RESUMABLE.json
```

It also creates or atomically replaces:

```text
lightning-output/cane-v1-latest-recovery.zip
lightning-output/LATEST_RECOVERY_BUNDLE.json
```

The ZIP contains `checkpoint-snapshots/last-epoch-NNN.pt` and
`LATEST_RESUMABLE.json`. It is written to a temporary archive, CRC-checked, and
only then moved over the previous ZIP, so a termination during ZIP creation
cannot destroy the last valid download. `bash status-training.sh` displays the
epoch and archive SHA-256. This is stronger than relying only on one mutable
`last.pt` or waiting until the final training package exists.

## If the Studio or GPU stops

1. Start the same Studio again; do not create a fresh Studio.
2. Switch back to one CUDA GPU.
3. Run `bash start-training.sh` again.
4. Run `bash status-training.sh`.
5. Confirm the next epoch follows the `completed_epoch` in
   `LATEST_RESUMABLE.json`.

The runner scans the immutable snapshots and the current run checkpoint, rejects
anything without optimizer state or with the wrong 15 classes, and selects the
highest completed epoch. The original epoch-22 checkpoint remains unchanged as
the final fallback.

If the usage limit ends and the run must move to another persistent Studio,
download `lightning-output/cane-v1-latest-recovery.zip` before deleting the old
Studio. Upload the original prepared handoff folder plus that ZIP to the new
Studio, then restore it from the handoff directory with:

```bash
mkdir -p lightning-output
unzip -o cane-v1-latest-recovery.zip -d lightning-output
bash start-training.sh
```

The unchanged epoch-22 input checkpoint is still required as the trust anchor;
the runner then validates and selects the higher restored snapshot. Confirm the
next displayed epoch follows the restored `completed_epoch`.

## If CUDA runs out of memory

The proven batch is 32 on a 16 GB T4. If the selected GPU cannot hold it, stop
the process and relaunch at batch 16:

```bash
kill "$(cat lightning-output/training.pid)"
BATCH=16 bash start-training.sh
```

Batch size is one of the settings Ultralytics permits overriding on resume. Do
not change image size, target epochs, model architecture, dataset, or class list.
Tell the team that batch 16 was used so the training record remains accurate.

## Completion gate

Training is complete only when:

```bash
bash status-training.sh
```

prints:

```text
COMPLETION_RECEIPT=present
```

Also run:

```bash
python -m json.tool lightning-output/LIGHTNING_COMPLETE.json
ls -lh lightning-output/deliverables
sha256sum lightning-output/deliverables/best.pt
```

The runner performs held-out `val` and `test` evaluation at image size 416 before
writing the completion receipt. It produces:

```text
lightning-output/deliverables/
├── best.pt
├── last.pt
├── latest-resumable.pt
├── LATEST_RESUMABLE.json
├── results.csv
├── args.yaml
└── validation-summary.json

lightning-output/cane-v1-lightning-deliverable.zip
```

`best.pt` is the model used for deployment and the final quality comparison.
`latest-resumable.pt` is retained for audit/recovery. `last.pt` is also retained,
but final Ultralytics processing may strip optimizer state after training; do not
assume it is resumable without checking the receipt.

## Download and local handoff

Download this one file from the Lightning file explorer:

```text
lightning-output/cane-v1-lightning-deliverable.zip
```

Lightning documents downloads through Drive, VS Code, SDK, and CLI in its
[Drive documentation](https://lightning.ai/docs/overview/drive). Keep the Studio
and its output until the local SHA and model taxonomy have been verified.

Place the downloaded ZIP locally at:

```text
/Users/chiragtaneja/Codes/capstone-everything/yolo-pi/artifacts/training/lightning-inbox/cane-v1-lightning-deliverable.zip
```

Then tell Codex that the file is present. The remaining local pipeline will:

1. verify the archive and every model hash;
2. confirm the exact 15-class taxonomy;
3. consolidate the training history through the final epoch;
4. export ONNX, OpenVINO, MNN, NCNN, and LiteRT artifacts;
5. smoke-test each exported backend, with Linux evidence where required;
6. compare pretrained YOLO26n and Cane V1 on the shared held-out test split;
7. build the production Pi model registry;
8. compile the M1/M2 architecture and R1-R6 runtime table templates;
9. run the complete automated test and notebook validation suite.

The complete local finalization command is already prepared:

```bash
.venv/bin/python scripts/finalize_lightning_return.py --device cpu
```

It first extracts into `artifacts/training/lightning-final/` only after the ZIP
CRC, paths, required files, summary hashes, val/test metrics, final-model
taxonomy, and optimizer-bearing latest recovery model pass. It then consolidates
the training history, exports the five Pi backends, runs the full 1,515-image
M1/M2 test comparison at 640, and builds the production registry only when all
runtime smoke evidence is complete. Every stage is recorded in
`artifacts/training/finalization-state.json`, so the command is safe to resume.

On this Mac, OpenVINO conversion may succeed while its x86-oriented local smoke
cannot execute. In that case the finalizer returns a documented pending state
and prepares `artifacts/models/cane-v1/openvino-linux-smoke/`; it does not
mislabel an untested export as complete. The checksum-bound smoke can be run in
the same Lightning Linux Studio and merged before rerunning the finalizer.

Lightning GPU timing is training evidence only. It is **not** Raspberry Pi
latency, FPS, RAM, CPU, temperature, or power evidence. Those values stay blank
until the prepared benchmark is run on the physical Pi.

## What to send if something fails

Do not send only a screenshot. Download or copy these text files:

```text
lightning-output/training-console.log
lightning-output/resume-selection.json
lightning-output/LATEST_RESUMABLE.json          # if present
lightning-output/LATEST_RECOVERY_BUNDLE.json     # if present
lightning-output/LIGHTNING_COMPLETE.json        # if present
```

Also provide the output of:

```bash
nvidia-smi
python -c "import torch, ultralytics; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), ultralytics.__version__)"
```

These are sufficient to diagnose GPU, environment, resume, and checkpoint
problems without guessing from the UI.
