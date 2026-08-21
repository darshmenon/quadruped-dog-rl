#!/usr/bin/env bash
# Train Go2 agility policy (stance / gait styles / jump) in MuJoCo.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec python3 training/train_agility.py "$@"
