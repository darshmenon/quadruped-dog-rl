#!/bin/bash
# Staged flat → rough → stairs MuJoCo training
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python3 "$ROOT/training/train_curriculum.py" "$@"
