"""
Jump curriculum: upward → forward → diagonal → over-obstacle.

Stages unlock with training progress. Landing targets are relative to the
episode start pose (world XY). Obstacle clearance is handled by the parkour
env placing a box between start and landing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class JumpStage(Enum):
    UPWARD = "upward"
    FORWARD = "forward"
    DIAGONAL = "diagonal"
    OVER_OBSTACLE = "over_obstacle"


@dataclass(frozen=True)
class JumpTarget:
    stage: JumpStage
    landing_dx: float   # meters forward from start
    landing_dy: float   # meters left from start
    jump_height: float  # peak height above nominal stand
    obstacle_height: float = 0.0  # 0 = no obstacle
    obstacle_width: float = 0.08  # half-depth along jump axis
    obstacle_span: float = 0.45   # half-width across jump axis


def stage_for_level(curriculum_level: float) -> JumpStage:
    level = float(np.clip(curriculum_level, 0.0, 1.0))
    if level < 0.25:
        return JumpStage.UPWARD
    if level < 0.50:
        return JumpStage.FORWARD
    if level < 0.75:
        return JumpStage.DIAGONAL
    return JumpStage.OVER_OBSTACLE


def sample_jump_target(
    rng: np.random.Generator,
    *,
    curriculum_level: float = 0.0,
    stage: Optional[JumpStage] = None,
) -> JumpTarget:
    """Sample a jump landing / obstacle target for the current curriculum."""
    level = float(np.clip(curriculum_level, 0.0, 1.0))
    st = stage if stage is not None else stage_for_level(level)

    # Peak height grows slowly; distance grows after upward is unlocked
    h_lo, h_hi = 0.08, 0.10 + 0.16 * level
    jump_h = float(rng.uniform(h_lo, h_hi))

    if st == JumpStage.UPWARD:
        return JumpTarget(
            stage=st,
            landing_dx=float(rng.uniform(-0.05, 0.05)),
            landing_dy=float(rng.uniform(-0.05, 0.05)),
            jump_height=jump_h,
        )

    if st == JumpStage.FORWARD:
        dist = float(rng.uniform(0.15, 0.25 + 0.55 * level))
        return JumpTarget(
            stage=st,
            landing_dx=dist,
            landing_dy=float(rng.uniform(-0.08, 0.08)),
            jump_height=max(jump_h, 0.10 + 0.05 * dist),
        )

    if st == JumpStage.DIAGONAL:
        dist = float(rng.uniform(0.20, 0.30 + 0.50 * level))
        lat = float(rng.uniform(0.10, 0.15 + 0.25 * level)) * float(rng.choice([-1.0, 1.0]))
        return JumpTarget(
            stage=st,
            landing_dx=dist,
            landing_dy=lat,
            jump_height=max(jump_h, 0.12 + 0.04 * dist),
        )

    # OVER_OBSTACLE: landing past a hurdle whose height scales with curriculum
    dist = float(rng.uniform(0.35, 0.45 + 0.45 * level))
    obst_h = float(rng.uniform(0.06, 0.08 + 0.14 * level))
    return JumpTarget(
        stage=st,
        landing_dx=dist,
        landing_dy=float(rng.uniform(-0.10, 0.10)),
        jump_height=max(jump_h, obst_h + 0.12),
        obstacle_height=obst_h,
        obstacle_width=float(rng.uniform(0.05, 0.10)),
        obstacle_span=float(rng.uniform(0.35, 0.55)),
    )


def landing_error(
    start_xy: np.ndarray,
    current_xy: np.ndarray,
    target: JumpTarget,
) -> Tuple[float, float]:
    """Return (dx_err, dy_err) of current pose vs desired landing in world XY."""
    desired = start_xy + np.array([target.landing_dx, target.landing_dy], dtype=np.float32)
    err = current_xy.astype(np.float32) - desired
    return float(err[0]), float(err[1])
