#!/usr/bin/env python3
"""Generate Go2 stair / ledge worlds for Gazebo Harmonic and MuJoCo.

Creates:
  training/envs/go2_gz_world_stairs.sdf   — straight stairs + return flight
  training/envs/go2_gz_world_ledges.sdf   — raised platforms, gaps, hollow steps
  training/envs/go2_stairs_scene.xml      — MuJoCo scene (robot + stairs)

Dimensions follow common Go2 stair curricula (rise ~5–12 cm, tread ~25–39 cm)
from blind stair climbing / Isaac Lab Go2 stairs work. Flat spawn at origin
so fall-recovery resets stay safe.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENVS = ROOT / "training" / "envs"
GO2_SCENE = ENVS / "go2_scene.xml"

WORLD_HEADER = """\
<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="{name}">

    <physics name="1ms" type="ode">
      <max_step_size>0.005</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Flat spawn / fall-recovery pad at origin -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
          <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
          <material><ambient>0.8 0.8 0.8 1</ambient><diffuse>0.8 0.8 0.8 1</diffuse></material>
        </visual>
      </link>
    </model>
"""

WORLD_FOOTER = """
  </world>
</sdf>
"""


def _box_link(
    name: str,
    x: float,
    y: float,
    z: float,
    sx: float,
    sy: float,
    sz: float,
    rgba: str = "0.55 0.55 0.55 1",
) -> str:
    # Gazebo box size is full extents; pose z is center.
    return f"""\
      <link name="{name}">
        <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 0</pose>
        <collision name="collision">
          <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
          <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface>
        </collision>
        <visual name="visual">
          <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
          <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material>
        </visual>
      </link>
"""


def _stairs_model(
    model_name: str,
    x0: float,
    rise: float,
    run: float,
    width: float,
    n_steps: int,
    y: float = 0.0,
    rgba: str = "0.55 0.55 0.55 1",
) -> str:
    """Solid staircase: each step is a filled box from ground up (stable contacts)."""
    links = []
    for i in range(n_steps):
        # step i (0-based): top surface at (i+1)*rise, center x along +X
        sx = run
        sy = width
        sz = (i + 1) * rise
        cx = x0 + (i + 0.5) * run
        cy = y
        cz = sz / 2.0
        links.append(_box_link(f"step_{i+1}", cx, cy, cz, sx, sy, sz, rgba))
    return (
        f'    <model name="{model_name}">\n'
        "      <static>true</static>\n"
        '      <pose>0 0 0 0 0 0</pose>\n'
        + "".join(links)
        + "    </model>\n"
    )


def _platform(
    name: str,
    x: float,
    y: float,
    height: float,
    length: float,
    width: float,
    rgba: str = "0.45 0.55 0.7 1",
) -> str:
    return (
        f'    <model name="{name}">\n'
        "      <static>true</static>\n"
        '      <pose>0 0 0 0 0 0</pose>\n'
        + _box_link("deck", x, y, height / 2.0, length, width, height, rgba)
        + "    </model>\n"
    )


def _hollow_steps(
    model_name: str,
    x0: float,
    rise: float,
    run: float,
    width: float,
    n_steps: int,
    tread_thickness: float = 0.04,
    rgba: str = "0.7 0.45 0.35 1",
) -> str:
    """Hollow / floating treads (StairMaster-style open risers)."""
    links = []
    for i in range(n_steps):
        top_z = (i + 1) * rise
        cx = x0 + (i + 0.5) * run
        cz = top_z - tread_thickness / 2.0
        links.append(
            _box_link(f"tread_{i+1}", cx, 0.0, cz, run * 0.95, width, tread_thickness, rgba)
        )
    return (
        f'    <model name="{model_name}">\n'
        "      <static>true</static>\n"
        '      <pose>0 0 0 0 0 0</pose>\n'
        + "".join(links)
        + "    </model>\n"
    )


def write_stairs_sdf(path: Path) -> None:
    # Curriculum-friendly: easy then harder flights along +X
    # World name must stay go2_rl so gazebo_rl / CHAMP bridges & stand/odom
    # scripts (hardcoded /world/go2_rl/...) work without remapping.
    body = WORLD_HEADER.format(name="go2_rl")
    body += "\n    <!-- Easy stairs: 6 cm rise x 30 cm run (x ~ 2.5–4.3 m) -->\n"
    body += _stairs_model("stairs_easy", x0=2.5, rise=0.06, run=0.30, width=1.6, n_steps=6)
    body += "\n    <!-- Mid stairs: 8 cm rise x 28 cm run (x ~ 6–7.7 m) -->\n"
    body += _stairs_model(
        "stairs_mid", x0=6.0, rise=0.08, run=0.28, width=1.6, n_steps=6, rgba="0.5 0.5 0.58 1"
    )
    body += "\n    <!-- Hard stairs: 12 cm rise x 25 cm run (x ~ 10–11.5 m) -->\n"
    body += _stairs_model(
        "stairs_hard", x0=10.0, rise=0.12, run=0.25, width=1.6, n_steps=6, rgba="0.45 0.45 0.5 1"
    )
    # Landing + descent so the dog can practice down-stairs too
    body += "\n    <!-- Top landing after hard stairs -->\n"
    body += _platform("landing_hard", x=12.0, y=0.0, height=0.72, length=1.2, width=1.8)
    body += "\n    <!-- Descent (facing -X): 8 cm steps starting at x~14 -->\n"
    # Build descent as ascending boxes placed with decreasing height toward +X
    # by using a forward staircase then a down staircase via mirrored solid steps.
    links = []
    rise, run, width = 0.08, 0.28, 1.6
    n = 6
    x0 = 13.5
    for i in range(n):
        # height decreases along +X
        sz = (n - i) * rise
        cx = x0 + (i + 0.5) * run
        cz = sz / 2.0
        links.append(_box_link(f"down_{i+1}", cx, 0.0, cz, run, width, sz, "0.5 0.55 0.5 1"))
    body += (
        '    <model name="stairs_down">\n'
        "      <static>true</static>\n"
        '      <pose>0 0 0 0 0 0</pose>\n'
        + "".join(links)
        + "    </model>\n"
    )
    body += WORLD_FOOTER
    path.write_text(body)
    print(f"wrote {path}")


def write_ledges_sdf(path: Path) -> None:
    # Same go2_rl world name as other training worlds (see write_stairs_sdf).
    body = WORLD_HEADER.format(name="go2_rl")
    body += "\n    <!-- Raised ledges / platforms (parkour-style climbs) -->\n"
    body += _platform("ledge_low", x=3.0, y=0.0, height=0.12, length=1.5, width=1.4)
    body += _platform("ledge_mid", x=5.5, y=0.0, height=0.22, length=1.5, width=1.4, rgba="0.4 0.55 0.65 1")
    body += _platform("ledge_high", x=8.0, y=0.0, height=0.35, length=1.5, width=1.4, rgba="0.35 0.5 0.6 1")

    body += "\n    <!-- Gap crossings: two platforms with open air between -->\n"
    body += _platform("gap_a", x=11.0, y=0.0, height=0.15, length=1.0, width=1.2, rgba="0.6 0.5 0.4 1")
    body += _platform("gap_b", x=12.4, y=0.0, height=0.15, length=1.0, width=1.2, rgba="0.6 0.5 0.4 1")
    # ~0.4 m clear gap between decks (centers 1.4 m apart, each length 1.0 → edge gap 0.4)

    body += "\n    <!-- Wider gap (~0.55 m clear) -->\n"
    body += _platform("gap2_a", x=15.0, y=0.0, height=0.18, length=1.0, width=1.2, rgba="0.65 0.45 0.4 1")
    body += _platform("gap2_b", x=16.55, y=0.0, height=0.18, length=1.0, width=1.2, rgba="0.65 0.45 0.4 1")

    body += "\n    <!-- Hollow stairs (open risers) — harder footholds -->\n"
    body += _hollow_steps(
        "hollow_stairs", x0=19.0, rise=0.10, run=0.28, width=1.5, n_steps=5, tread_thickness=0.045
    )

    body += "\n    <!-- Side curb / sidewalk ledge -->\n"
    body += _platform("curb", x=4.0, y=2.2, height=0.15, length=6.0, width=0.35, rgba="0.5 0.5 0.5 1")
    body += WORLD_FOOTER
    path.write_text(body)
    print(f"wrote {path}")


def _mj_box(name: str, x: float, y: float, z: float, hx: float, hy: float, hz: float, rgba: str) -> str:
    # MuJoCo size is half-extents
    return (
        f'    <geom name="{name}" type="box" pos="{x:.4f} {y:.4f} {z:.4f}" '
        f'size="{hx:.4f} {hy:.4f} {hz:.4f}" rgba="{rgba}" '
        f'condim="3" friction="1.0 0.005 0.0001" group="1"/>\n'
    )


def write_mujoco_stairs_scene(path: Path, source_scene: Path = GO2_SCENE) -> None:
    """Clone go2_scene.xml and inject stair/ledge geoms after the floor plane."""
    text = source_scene.read_text()
    if 'name="floor"' not in text:
        raise SystemExit(f"expected floor geom in {source_scene}")

    # Mark floor as group 1 for height rays (match rough scene convention)
    text = text.replace(
        '<geom name="floor" type="plane" size="100 100 0.1" rgba="0.8 0.9 0.8 1" condim="3" friction="1.0 0.005 0.0001"/>',
        '<geom name="floor" type="plane" size="100 100 0.1" rgba="0.8 0.9 0.8 1" condim="3" friction="1.0 0.005 0.0001" group="1"/>',
        1,
    )
    text = text.replace('model="go2_scene"', 'model="go2_stairs_scene"', 1)

    geoms = ["\n    <!-- Stair / ledge course (+X). Flat spawn at origin. -->\n"]
    # Easy solid stairs
    rise, run, width, n = 0.06, 0.30, 1.6, 6
    x0 = 2.5
    for i in range(n):
        hz = (i + 1) * rise / 2.0
        hx, hy = run / 2.0, width / 2.0
        cx = x0 + (i + 0.5) * run
        geoms.append(_mj_box(f"stair_e_{i+1}", cx, 0.0, hz, hx, hy, hz, "0.55 0.55 0.55 1"))

    # Mid stairs
    rise, run, n = 0.08, 0.28, 6
    x0 = 6.0
    for i in range(n):
        hz = (i + 1) * rise / 2.0
        hx, hy = run / 2.0, width / 2.0
        cx = x0 + (i + 0.5) * run
        geoms.append(_mj_box(f"stair_m_{i+1}", cx, 0.0, hz, hx, hy, hz, "0.5 0.5 0.58 1"))

    # Hard stairs
    rise, run, n = 0.12, 0.25, 6
    x0 = 10.0
    for i in range(n):
        hz = (i + 1) * rise / 2.0
        hx, hy = run / 2.0, width / 2.0
        cx = x0 + (i + 0.5) * run
        geoms.append(_mj_box(f"stair_h_{i+1}", cx, 0.0, hz, hx, hy, hz, "0.45 0.45 0.5 1"))

    # Ledge platforms
    for name, x, h, length in [
        ("ledge_low", 14.0, 0.12, 1.5),
        ("ledge_mid", 16.5, 0.22, 1.5),
        ("ledge_high", 19.0, 0.35, 1.5),
    ]:
        geoms.append(
            _mj_box(name, x, 0.0, h / 2.0, length / 2.0, 0.7, h / 2.0, "0.4 0.55 0.65 1")
        )

    # Gap pair
    geoms.append(_mj_box("gap_a", 22.0, 0.0, 0.075, 0.5, 0.6, 0.075, "0.6 0.5 0.4 1"))
    geoms.append(_mj_box("gap_b", 23.4, 0.0, 0.075, 0.5, 0.6, 0.075, "0.6 0.5 0.4 1"))

    # Hollow treads
    rise, run, n = 0.10, 0.28, 5
    x0 = 25.0
    thick = 0.045
    for i in range(n):
        top_z = (i + 1) * rise
        cx = x0 + (i + 0.5) * run
        cz = top_z - thick / 2.0
        geoms.append(
            _mj_box(
                f"hollow_{i+1}",
                cx,
                0.0,
                cz,
                run * 0.475,
                0.75,
                thick / 2.0,
                "0.7 0.45 0.35 1",
            )
        )

    insert = "".join(geoms)
    needle = 'group="1"/>\n'
    # Insert after the floor geom line we just tagged with group="1"
    idx = text.find(needle)
    if idx < 0:
        raise SystemExit("could not find floor geom to insert stairs after")
    idx += len(needle)
    text = text[:idx] + insert + text[idx:]
    path.write_text(text)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("all", "gazebo", "mujoco"),
        default="all",
        help="Which assets to regenerate",
    )
    args = parser.parse_args()
    ENVS.mkdir(parents=True, exist_ok=True)
    if args.only in ("all", "gazebo"):
        write_stairs_sdf(ENVS / "go2_gz_world_stairs.sdf")
        write_ledges_sdf(ENVS / "go2_gz_world_ledges.sdf")
    if args.only in ("all", "mujoco"):
        write_mujoco_stairs_scene(ENVS / "go2_stairs_scene.xml")


if __name__ == "__main__":
    main()
