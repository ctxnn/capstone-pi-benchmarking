#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p lightning-output

if [[ -f lightning-output/training.pid ]]; then
  previous_pid="$(tr -dc '0-9' < lightning-output/training.pid)"
  if [[ -n "$previous_pid" ]] && kill -0 "$previous_pid" 2>/dev/null; then
    echo "Training is already running with PID $previous_pid"
    echo "Use: bash status-training.sh"
    exit 0
  fi
fi

python -m pip install --upgrade "ultralytics==8.4.132"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable: switch this Studio to a GPU machine'; print('GPU:', torch.cuda.get_device_name(0)); print('CUDA:', torch.version.cuda)"

batch="${BATCH:-32}"
workers="${WORKERS:-4}"
nohup python -u lightning_train_resume.py \
  --workspace "$PWD" \
  --dataset-archive cane-v1-training-640.tar \
  --checkpoint resume-last-epoch22.pt \
  --output lightning-output \
  --device 0 \
  --batch "$batch" \
  --workers "$workers" \
  >> lightning-output/training-console.log 2>&1 &
training_pid=$!
echo "$training_pid" > lightning-output/training.pid

echo "Started protected training with PID $training_pid"
echo "Monitor with: bash status-training.sh"
echo "The first resumed epoch must be 23/30."
echo "A download-ready cane-v1-latest-recovery.zip is replaced atomically after every completed epoch."
