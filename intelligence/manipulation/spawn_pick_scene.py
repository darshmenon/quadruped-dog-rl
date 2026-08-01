"""
Spawns a low pedestal ("table") with a graspable cylinder ("bottle") on top,
positioned within the 5-DOF arm's reach envelope, into an already-running
Gazebo world -- for use with pick_demo.py.

The arm's base_mount frame origin (arm_base link) sits at world (0.08, 0,
0.057) when the robot is standing at the spawn pose used by
training/launch/gazebo_rl.launch.py (x=0, y=0, z=0.32, zero yaw) -- verified
by querying /world/<world>/pose/info live. Since the robot spawns with zero
yaw, world coordinates equal arm-frame coordinates offset by that fixed
translation, no rotation needed.

The cylinder is centered at arm-frame (0.42, 0, 0.0) -- reachable per
arm_ik.py's grid (see test_arm_ik.py), matching the "grasp" waypoint in
pick_demo.py.

Usage (with Gazebo already running, e.g. via
`ros2 launch training/launch/gazebo_rl.launch.py enable_arm_reach:=true`):

    python3 intelligence/manipulation/spawn_pick_scene.py
    python3 intelligence/manipulation/spawn_pick_scene.py --world go2_rl
"""

import argparse
import subprocess
import sys

# Arm base_mount frame origin in world coordinates, for this session's robot
# spawn pose (x=0, y=0, z=0.32, zero yaw) -- see module docstring.
ARM_BASE_WORLD_OFFSET = (0.08, 0.0, 0.057)

# Grasp target in the arm's base_mount frame -- must match pick_demo.py's
# GRASP waypoint so the cylinder is actually where the gripper closes.
CYLINDER_ARM_FRAME = (0.42, 0.0, 0.0)

TABLE_SIZE = (0.35, 0.35, 0.034)  # x, y, z (m)
CYLINDER_RADIUS = 0.018  # m -- fits within the gripper's 0.05m max opening
CYLINDER_LENGTH = 0.08   # m
CYLINDER_MASS = 0.05     # kg


def arm_to_world(x, y, z):
    ox, oy, oz = ARM_BASE_WORLD_OFFSET
    return (x + ox, y + oy, z + oz)


def table_sdf():
    sx, sy, sz = TABLE_SIZE
    return f"""<sdf version="1.9">
<model name="pick_table">
  <static>true</static>
  <link name="link">
    <collision name="collision">
      <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
    </collision>
    <visual name="visual">
      <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
      <material>
        <ambient>0.55 0.38 0.20 1</ambient>
        <diffuse>0.55 0.38 0.20 1</diffuse>
      </material>
    </visual>
  </link>
</model>
</sdf>"""


def cylinder_sdf():
    r, l, m = CYLINDER_RADIUS, CYLINDER_LENGTH, CYLINDER_MASS
    # Solid-cylinder inertia: Ixx=Iyy=m(3r^2+l^2)/12, Izz=m*r^2/2
    ixx = m * (3 * r * r + l * l) / 12.0
    izz = m * r * r / 2.0
    return f"""<sdf version="1.9">
<model name="pick_cylinder">
  <link name="link">
    <inertial>
      <mass>{m}</mass>
      <inertia>
        <ixx>{ixx}</ixx><iyy>{ixx}</iyy><izz>{izz}</izz>
        <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
      </inertia>
    </inertial>
    <collision name="collision">
      <geometry><cylinder><radius>{r}</radius><length>{l}</length></cylinder></geometry>
      <surface>
        <friction><ode><mu>1.2</mu><mu2>1.2</mu2></ode></friction>
      </surface>
    </collision>
    <visual name="visual">
      <geometry><cylinder><radius>{r}</radius><length>{l}</length></cylinder></geometry>
      <material>
        <ambient>0.10 0.30 0.80 1</ambient>
        <diffuse>0.10 0.30 0.80 1</diffuse>
      </material>
    </visual>
  </link>
</model>
</sdf>"""


def spawn(world, name, sdf, pose):
    x, y, z = pose
    # gz's protobuf text-format parser rejects string literals that cross
    # line boundaries, so the (human-readable, multi-line) SDF has to be
    # flattened onto one line before going into the --req argument.
    sdf_oneline = " ".join(sdf.split())
    req = (
        f"sdf: '{sdf_oneline}' "
        f"name: '{name}' "
        f"pose: {{position: {{x: {x}, y: {y}, z: {z}}}}}"
    )
    print(f"[spawn_pick_scene] spawning '{name}' at world pose ({x:.3f}, {y:.3f}, {z:.3f})")
    result = subprocess.run(
        [
            "gz", "service",
            "-s", f"/world/{world}/create",
            "--reqtype", "gz.msgs.EntityFactory",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "5000",
            "--req", req,
        ],
        capture_output=True, text=True,
    )
    print(f"[spawn_pick_scene] '{name}' spawn result: stdout={result.stdout.strip()!r} "
          f"stderr={result.stderr.strip()!r} returncode={result.returncode}")
    if result.returncode != 0 or "data: true" not in result.stdout:
        print(f"[spawn_pick_scene] WARNING: '{name}' spawn may have failed -- "
              "check that Gazebo is running and the world name is correct.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default="go2_rl", help="Gazebo world name")
    args = parser.parse_args()

    cylinder_world_pos = arm_to_world(*CYLINDER_ARM_FRAME)
    # Table top must sit CYLINDER_LENGTH/2 below the cylinder's center so the
    # cylinder rests on top of it (bottom flush with the table surface),
    # not table-top == cylinder-center (which buries half the cylinder
    # inside the table and lets contact resolution slowly eject it --
    # exactly the "floating" drift seen before this was fixed).
    table_top_z = cylinder_world_pos[2] - CYLINDER_LENGTH / 2.0
    table_center_z = table_top_z - TABLE_SIZE[2] / 2.0
    table_world_pos = (cylinder_world_pos[0], cylinder_world_pos[1], table_center_z)

    print(f"[spawn_pick_scene] cylinder target arm-frame={CYLINDER_ARM_FRAME}, "
          f"table_world={table_world_pos}, cylinder_world={cylinder_world_pos}")

    ok_table = spawn(args.world, "pick_table", table_sdf(), table_world_pos)
    ok_cyl = spawn(args.world, "pick_cylinder", cylinder_sdf(), cylinder_world_pos)

    if not (ok_table and ok_cyl):
        print("[spawn_pick_scene] one or more spawns failed, see warnings above")
        sys.exit(1)
    print("[spawn_pick_scene] scene ready")


if __name__ == "__main__":
    main()
