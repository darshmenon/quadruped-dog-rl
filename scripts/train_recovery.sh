#!/bin/bash
# Fall recovery training (optional --rough for rough terrain scene)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python3 "$ROOT/training/train_recovery.py" "$@"
