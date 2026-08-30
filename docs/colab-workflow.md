# Colab GPU workflow

The command-line script is the authoritative automated smoke test. It provisions
a named T4 session, trains for one epoch on COCO8, validates the best checkpoint,
exports ONNX and FP16 NCNN, validates both exports, and packages all results.
The explicit `end2end=False` NCNN argument is intentional: NCNN uses YOLO26's
one-to-many head and performs NMS in the deployment runtime.

```bash
colab run --gpu T4 --keep -s yolo-pi-gpu-smoke \
  scripts/colab_train_export.py \
  --data coco8.yaml --epochs 1 --imgsz 320 --batch 8

colab download -s yolo-pi-gpu-smoke \
  /content/yolo-pi-colab-artifacts.zip \
  artifacts/training/yolo-pi-colab-artifacts.zip

colab stop -s yolo-pi-gpu-smoke
```

Always confirm `colab sessions` no longer lists the session after cleanup.

For the complete V1 dataset, create a named session, upload the processed archive,
extract it remotely, execute the same script with the remote `data.yaml`, download
the result archive, and stop the session. Avoid `colab run` without `--keep` when
artifacts must be retrieved because the ephemeral VM is released immediately.

The notebook is the reader-facing companion:

```bash
colab new -s yolo-pi-notebook --gpu T4
colab exec -s yolo-pi-notebook -f notebooks/cane_v1_training_export.ipynb
colab download -s yolo-pi-notebook \
  /content/yolo-pi-notebook-artifacts.zip \
  artifacts/training/yolo-pi-notebook-artifacts.zip
colab stop -s yolo-pi-notebook
```

Notebook execution writes `notebooks/cane_v1_training_export_output.ipynb` next
to the input notebook. Keep that generated file out of version control; retain it
with the research run artifacts when its outputs support a paper result.

## Cane V1 durable fine-tune and resume

The production run uses the pinned 21,269 / 2,768 / 1,515 training/validation/
test dataset and Ultralytics 8.4.132. It has survived Colab VM reclamation by
resuming from optimizer-bearing `last.pt` checkpoints; a VM disappearing is not
treated as early stopping.

The transfer archive is split into bounded chunks so uploads can resume without
retransmitting completed parts:

```bash
uv run python scripts/colab_chunk_upload.py \
  --session "$SESSION" \
  --chunks artifacts/training/cane-v1-training-640-chunks \
  --remote-dir /content \
  --state "artifacts/training/${SESSION}-upload-state.json" \
  --workers 3 --retries 3

colab exec -s "$SESSION" \
  -f scripts/colab_reassemble_training_archive.py --timeout 900
```

Do not launch training unless the remote verifier reports 39 parts,
2,013,204,992 bytes, and SHA-256
`f469a971a1c0e167b01661f4d2296e4b5e38501f0d6c88fa26f72e34842f8b91`.
The resume launcher separately verifies the checkpoint hash before installing it
as the run's `weights/last.pt`.

### Usage-window-independent checkpoint guard

Immediately after training starts, launch the local guard:

```bash
uv run python scripts/colab_checkpoint_guard.py \
  --session "$SESSION" \
  --output artifacts/training/checkpoint-guard \
  --interval-seconds 120 \
  --max-consecutive-failures 10 \
  --daemon
```

This process is independent of a Codex response window. Each successful poll:

- atomically refreshes `latest-best.pt` and resumable `latest-last.pt`;
- validates the exact 15-class taxonomy and records optimizer presence;
- creates immutable epoch/hash checkpoint snapshots using hard links;
- refreshes the training log, results CSV and exact args YAML; and
- writes a machine-readable `state.json` with hashes and remote errors.

If the VM disappears, the guard stops after ten consecutive failed current
polls and leaves the newest verified local checkpoint untouched. A checkpoint
saved on an older poll never resets that counter.

When `/content/yolo-pi-colab-artifacts.zip` appears, the guard downloads it,
runs a complete ZIP integrity test, requires the validation summary and both
weights, and atomically promotes:

```text
artifacts/training/checkpoint-guard/final/best.pt
artifacts/training/checkpoint-guard/final/last.pt
artifacts/training/checkpoint-guard/validation-summary.json
artifacts/training/checkpoint-guard/yolo-pi-colab-artifacts.zip
```

The promoted `best.pt` must match `best_weights.sha256` in the embedded summary.
The summary is extracted from the verified local ZIP, so finalization does not
depend on another remote request after the archive is safe.

### Accepted history across replacement VMs

Fresh VMs may start a new `results.csv`. Reconstruct the paper-facing trajectory
from immutable checkpoints instead of concatenating CSV files by hand:

```bash
uv run python scripts/consolidate_training_history.py \
  --checkpoint artifacts/training/checkpoints/cane-v1-best-epoch6.pt \
  --checkpoint artifacts/training/checkpoint-guard/checkpoints/snapshots/last-epoch-015-cfeb7c9a252b.pt \
  --checkpoint artifacts/training/checkpoint-guard/final/last.pt \
  --output artifacts/training/cane-v1-training-history.csv \
  --manifest artifacts/training/cane-v1-training-history.json
```

The consolidator validates one class taxonomy, records checkpoint hashes,
requires contiguous epochs, rejects divergent replayed epochs, and keeps only
the last occurrence when replayed metrics are identical. Inspect `state.json`
and the consolidation manifest before stopping the final Colab session.
