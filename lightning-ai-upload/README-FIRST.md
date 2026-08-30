# Cane V1 Lightning AI handoff

This is the complete upload folder. Do **not** upload the full `yolo-pi`
repository. Follow the detailed instructions in `LIGHTNING-INSTRUCTIONS.md`.

The important integrity values are:

- `cane-v1-training-640.tar`: 2,013,204,992 bytes; SHA-256
  `f469a971a1c0e167b01661f4d2296e4b5e38501f0d6c88fa26f72e34842f8b91`
- `resume-last-epoch22.pt`: SHA-256
  `0da75726f7a058e63d7b030601447ddeb4bd98b420d1387482761aa5799a0eea`
- Resume point: 22 completed epochs; the first new epoch must display `23/30`.

Quick start after the folder is uploaded and the Studio is switched to a GPU:

```bash
cd /teamspace/studios/this_studio/cane-v1-lightning
bash start-training.sh
bash status-training.sh
```

If the Studio is interrupted, run `bash start-training.sh` again. The runner
selects the newest checksum-validated per-epoch snapshot automatically.

Even before training finishes, the newest completed epoch is downloadable as:

```text
lightning-output/cane-v1-latest-recovery.zip
```

That recovery ZIP is created at startup and atomically replaced after every
completed epoch. It contains optimizer state and is for resuming an unfinished
run; it is not the final deployment deliverable.

When `COMPLETION_RECEIPT=present`, download:

```text
lightning-output/cane-v1-lightning-deliverable.zip
```
