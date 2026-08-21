#!/usr/bin/env python3
"""Download public Go2 model files into training/pretrained/.

Weights are gitignored (*.pt). Safe to re-run; skips files that already exist
unless --force is set.

Examples:
  python3 scripts/download_pretrained.py
  python3 scripts/download_pretrained.py --only locomotion
  python3 scripts/download_pretrained.py --only parkour --force
  python3 scripts/download_pretrained.py --list
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training" / "pretrained"

# name -> (relpath under training/pretrained, url, approx bytes for progress)
ASSETS: dict[str, tuple[str, str, int]] = {
    "flat": (
        "go2_locomotion/flat_model_6800.pt",
        "https://raw.githubusercontent.com/sallu-786/Go2_Isaac_ros2/main/ckpts/unitree_go2/flat_model_6800.pt",
        983_394,
    ),
    "rough": (
        "go2_locomotion/rough_model_7850.pt",
        "https://raw.githubusercontent.com/sallu-786/Go2_Isaac_ros2/main/ckpts/unitree_go2/rough_model_7850.pt",
        6_881_323,
    ),
    "rpl_rough": (
        "go2_parkour/rpl_rough_go2_model_2000.pt",
        "https://huggingface.co/real-jiashu-yu/parkour-drl-checkpoints/resolve/main/final_project_release_20260620/checkpoints/rpl_rough_go2_model_2000.pt",
        17_653_809,
    ),
    "rpl_field": (
        "go2_parkour/rpl_field_go2_model_40000.pt",
        "https://huggingface.co/real-jiashu-yu/parkour-drl-checkpoints/resolve/main/final_project_release_20260620/checkpoints/rpl_field_go2_model_40000.pt",
        17_653_992,
    ),
    "rpl_visual": (
        "go2_parkour/rpl_visual_distill_go2_model_100000.pt",
        "https://huggingface.co/real-jiashu-yu/parkour-drl-checkpoints/resolve/main/final_project_release_20260620/checkpoints/rpl_visual_distill_go2_model_100000.pt",
        16_448_932,
    ),
    # Genesis sim2real — walk + stairs (JIT actor; not SB3)
    "sim2real_walk": (
        "go2_stairs/sim2real_walk.pt",
        "https://raw.githubusercontent.com/saifahmadgit/go2-sim2real-deploy/master/example/go2/low_level/final/walk.pt",
        4_941_375,
    ),
    "sim2real_stairs": (
        "go2_stairs/sim2real_stairs.pt",
        "https://raw.githubusercontent.com/saifahmadgit/go2-sim2real-deploy/master/example/go2/low_level/final/stairs.pt",
        5_420_681,
    ),
    "sim2real_stairs_39cm": (
        "go2_stairs/sim2real_stairs_39cm_104000.pt",
        "https://raw.githubusercontent.com/saifahmadgit/go2-sim2real-deploy/master/example/go2/low_level/stair_39cm_104000.pt",
        5_420_681,
    ),
    # CTS MoE — strong stairs/slope MuJoCo deploy policy (HF LFS)
    "cts_stairs": (
        "go2_stairs/cts_moe_policy.pt",
        "https://huggingface.co/wty-yy/go2_rl_gym_data/resolve/main/go2_moe_cts_137000_0.6713/policy.pt",
        5_206_984,
    ),
}

GROUPS = {
    "locomotion": ("flat", "rough"),
    "parkour": ("rpl_rough", "rpl_field", "rpl_visual"),
    "stairs": ("sim2real_walk", "sim2real_stairs", "sim2real_stairs_39cm", "cts_stairs"),
    "all": tuple(ASSETS.keys()),
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    print(f"  downloading {dest.name} …")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    print(f"  wrote {dest} ({dest.stat().st_size:,} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=tuple(GROUPS.keys()),
        default="all",
        help="Which group to fetch (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    parser.add_argument("--list", action="store_true", help="List assets and exit")
    args = parser.parse_args()

    if args.list:
        for key in GROUPS[args.only]:
            rel, url, size = ASSETS[key]
            path = OUT / rel
            status = "present" if path.exists() else "missing"
            print(f"{key:12} {status:8} {size/1e6:5.1f} MB  {rel}")
            print(f"             {url}")
        return 0

    keys = GROUPS[args.only]
    print(f"Fetching {args.only} → {OUT}")
    ok = 0
    for key in keys:
        rel, url, _ = ASSETS[key]
        dest = OUT / rel
        if dest.exists() and not args.force:
            print(f"  skip {rel} (exists; use --force to replace)")
            ok += 1
            continue
        try:
            _download(url, dest)
            ok += 1
        except Exception as exc:
            print(f"  FAIL {rel}: {exc}", file=sys.stderr)
    print(f"Done: {ok}/{len(keys)} files under {OUT}")
    print("See docs/PRETRAINED.md for model names and usage.")
    return 0 if ok == len(keys) else 1


if __name__ == "__main__":
    raise SystemExit(main())
