#!/usr/bin/env python3
"""Download Gazebo Harmonic worlds + Fuel models into training/envs/worlds/.

Fuel assets land in ~/.gz/fuel/ (gz cache) and are also copied into
training/envs/worlds/fuel/ when possible. Upstream Go2 packs are cloned
shallow into training/envs/worlds/_src/ (gitignored).

Examples:
  python3 scripts/download_worlds.py
  python3 scripts/download_worlds.py --only fuel
  python3 scripts/download_worlds.py --only models
  python3 scripts/download_worlds.py --list
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORLDS = ROOT / "training" / "envs" / "worlds"
FUEL_DST = WORLDS / "fuel"
SRC_DST = WORLDS / "_src"
FUEL_CACHE = Path.home() / ".gz" / "fuel" / "fuel.gazebosim.org" / "openrobotics"

FUEL_WORLDS = {
    "industrial_warehouse": "https://fuel.gazebosim.org/1.0/OpenRobotics/worlds/industrial-warehouse",
    "cave": "https://fuel.gazebosim.org/1.0/OpenRobotics/worlds/Cave%20World",
    "tugbot_warehouse": "https://fuel.gazebosim.org/1.0/OpenRobotics/worlds/Tugbot%20in%20Warehouse",
    "island": "https://fuel.gazebosim.org/1.0/OpenRobotics/worlds/Null%20Island",
    "garden": "https://fuel.gazebosim.org/1.0/OpenRobotics/worlds/garden%20demo",
    "sonoma": "https://fuel.gazebosim.org/1.0/OpenRobotics/worlds/Prius%20on%20Sonoma%20Raceway",
    "fortress": "https://fuel.gazebosim.org/1.0/OpenRobotics/worlds/Fortress%20demo",
    "jetty": "https://fuel.gazebosim.org/1.0/OpenRobotics/worlds/Jetty%20World",
}

FUEL_MODELS = {
    "construction_cone": "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Construction%20Cone",
    "jersey_barrier": "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Jersey%20Barrier",
    "standing_person": "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Standing%20Person",
    "dumpster": "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Dumpster",
    "oak_tree": "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Oak%20Tree",
    "pine_tree": "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Pine%20Tree",
}

UPSTREAM_REPOS = {
    "go2-quadruped-sim": "https://github.com/AOShei/go2-quadruped-sim.git",
    "unitree_go2_ros2": "https://github.com/khaledgabr77/unitree_go2_ros2.git",
    "unitree-go2-ros2": "https://github.com/anujjain-dev/unitree-go2-ros2.git",
}


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    return subprocess.call(cmd)


def download_fuel_worlds() -> None:
    FUEL_DST.mkdir(parents=True, exist_ok=True)
    for name, url in FUEL_WORLDS.items():
        print(f"\n== Fuel world: {name}")
        rc = _run(["gz", "fuel", "download", "-t", "world", "-u", url, "-v", "2"])
        if rc != 0:
            print(f"  WARN: download failed for {name} (rc={rc})")
            continue
        # Copy newest matching *.sdf from cache into worlds/fuel/
        matches = sorted(FUEL_CACHE.glob(f"worlds/**/*.sdf"))
        # Prefer files whose stem relates to the key
        picked = None
        for p in reversed(matches):
            stem = p.stem.lower().replace(" ", "_").replace("-", "_")
            key = name.lower()
            if key.split("_")[0] in stem or stem in key or key in stem:
                picked = p
                break
        if picked is None and matches:
            # fall back: most recently modified under worlds/
            world_sdfs = sorted(
                (FUEL_CACHE / "worlds").rglob("*.sdf"),
                key=lambda p: p.stat().st_mtime, reverse=True)
            picked = world_sdfs[0] if world_sdfs else None
        if picked and picked.is_file():
            dst = FUEL_DST / f"{name}.sdf"
            shutil.copy2(picked, dst)
            print(f"  copied → {dst.relative_to(ROOT)}")


def download_fuel_models() -> None:
    for name, url in FUEL_MODELS.items():
        print(f"\n== Fuel model: {name}")
        rc = _run(["gz", "fuel", "download", "-t", "model", "-u", url, "-v", "2"])
        if rc != 0:
            print(f"  WARN: model download failed for {name}")


def clone_upstream() -> None:
    SRC_DST.mkdir(parents=True, exist_ok=True)
    for name, url in UPSTREAM_REPOS.items():
        dest = SRC_DST / name
        if dest.exists():
            print(f"EXISTS {dest}")
            continue
        print(f"\n== Clone {name}")
        _run([
            "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
            url, str(dest),
        ])
        _run(["git", "-C", str(dest), "sparse-checkout", "set",
              "**/worlds", "src/**/worlds", "unitree_go2_description",
              "unitree_go2_nav2", "robots"])


def list_assets() -> None:
    print("Fuel worlds:")
    for k, u in FUEL_WORLDS.items():
        print(f"  {k:22s} {u}")
    print("Fuel models:")
    for k, u in FUEL_MODELS.items():
        print(f"  {k:22s} {u}")
    print("Upstream repos:")
    for k, u in UPSTREAM_REPOS.items():
        print(f"  {k:22s} {u}")
    print("\nOn disk:")
    if WORLDS.exists():
        for p in sorted(WORLDS.rglob("*")):
            if p.is_file() and p.suffix in {".sdf", ".world", ".md"}:
                print(f"  {p.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", choices=("fuel", "models", "upstream", "all"), default="all")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        list_assets()
        return 0

    if shutil.which("gz") is None and args.only in ("fuel", "models", "all"):
        print("ERROR: `gz` not on PATH (install Gazebo Harmonic).", file=sys.stderr)
        return 1

    WORLDS.mkdir(parents=True, exist_ok=True)
    if args.only in ("fuel", "all"):
        download_fuel_worlds()
    if args.only in ("models", "all"):
        download_fuel_models()
    if args.only in ("upstream", "all"):
        clone_upstream()

    print("\nDone. Launch Fuel SDF with:")
    print("  gz sim training/envs/worlds/fuel/industrial_warehouse.sdf")
    print("Go2-compatible course worlds:")
    print("  ros2 launch launch/champ_go2_gazebo.launch.py \\")
    print("    world:=$(pwd)/training/envs/go2_gz_world_arena.sdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
