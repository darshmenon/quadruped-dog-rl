#!/usr/bin/env bash
# Train Go2 parkour / jump-landing policy in MuJoCo.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec python3 training/train_parkour.py "$@"
