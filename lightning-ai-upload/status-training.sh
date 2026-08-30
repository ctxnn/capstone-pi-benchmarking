#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
if [[ -f lightning-output/training.pid ]]; then
  training_pid="$(tr -dc '0-9' < lightning-output/training.pid)"
  if [[ -n "$training_pid" ]] && kill -0 "$training_pid" 2>/dev/null; then
    echo "STATUS=running PID=$training_pid"
  else
    echo "STATUS=not-running last-PID=${training_pid:-unknown}"
  fi
else
  echo "STATUS=not-started"
fi

if [[ -f lightning-output/LIGHTNING_COMPLETE.json ]]; then
  echo "COMPLETION_RECEIPT=present"
else
  echo "COMPLETION_RECEIPT=absent"
fi

if [[ -f lightning-output/LATEST_RESUMABLE.json ]]; then
  echo "LATEST_RESUMABLE:"
  python -m json.tool lightning-output/LATEST_RESUMABLE.json
fi

if [[ -f lightning-output/LATEST_RECOVERY_BUNDLE.json ]] && \
   [[ -f lightning-output/cane-v1-latest-recovery.zip ]]; then
  echo "LATEST_RECOVERY_BUNDLE:"
  python -m json.tool lightning-output/LATEST_RECOVERY_BUNDLE.json
  sha256sum lightning-output/cane-v1-latest-recovery.zip
fi

if [[ -f lightning-output/training-console.log ]]; then
  echo "RECENT_LOG:"
  tail -n 80 lightning-output/training-console.log
fi
