"""
Agility skill presets for Go2 — stance height, gait style, and jump targets.

Command vector (11 floats):
    vx, vy, wz, height_offset, jump_height,
    gait_freq, gait_phase, gait_offset, gait_bound,
    landing_dx, landing_dy
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Sequence

import numpy as np

from intelligence.skills.jump_curriculum import JumpStage, sample_jump_target


class Skill(Enum):
    STAND = "stand"
    CROUCH = "crouch"
    SIT = "sit"
    WALK = "walk"
    TROT = "trot"
    PACE = "pace"
    BOUND = "bound"
    PRONK = "pronk"
    JUMP = "jump"
    JUMP_FORWARD = "jump_forward"
    JUMP_DIAGONAL = "jump_diagonal"


@dataclass(frozen=True)
class AgilityCommand:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    height_offset: float = 0.0   # relative to nominal stand height (m)
    jump_height: float = 0.0     # extra peak height target above stand (m)
    gait_freq: float = 0.0       # Hz; 0 = hold stance
    gait_phase: float = 0.0
    gait_offset: float = 0.0
    gait_bound: float = 0.0
    landing_dx: float = 0.0      # desired landing offset from episode start (m)
    landing_dy: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.vx, self.vy, self.wz,
                self.height_offset, self.jump_height,
                self.gait_freq, self.gait_phase, self.gait_offset, self.gait_bound,
                self.landing_dx, self.landing_dy,
            ],
            dtype=np.float32,
        )

    @classmethod
    def from_array(cls, arr: Sequence[float]) -> "AgilityCommand":
        a = list(arr) + [0.0] * 11
        return cls(*[float(x) for x in a[:11]])

    @property
    def is_jump(self) -> bool:
        return self.jump_height > 0.05 or abs(self.landing_dx) > 0.08 or abs(self.landing_dy) > 0.08


SKILL_PRESETS: Dict[Skill, AgilityCommand] = {
    Skill.STAND: AgilityCommand(height_offset=0.0, gait_freq=0.0),
    Skill.CROUCH: AgilityCommand(height_offset=-0.10, gait_freq=0.0),
    Skill.SIT: AgilityCommand(height_offset=-0.16, gait_freq=0.0),
    Skill.WALK: AgilityCommand(
        vx=0.35, gait_freq=1.4, gait_phase=0.25,
    ),
    Skill.TROT: AgilityCommand(
        vx=0.7, gait_freq=2.2, gait_phase=0.5,
    ),
    Skill.PACE: AgilityCommand(
        vx=0.6, gait_freq=2.0, gait_offset=0.5,
    ),
    Skill.BOUND: AgilityCommand(
        vx=1.0, gait_freq=2.8, gait_bound=0.5,
    ),
    Skill.PRONK: AgilityCommand(
        vx=0.4, gait_freq=2.5,
    ),
    Skill.JUMP: AgilityCommand(jump_height=0.18),
    Skill.JUMP_FORWARD: AgilityCommand(
        jump_height=0.16, landing_dx=0.45,
    ),
    Skill.JUMP_DIAGONAL: AgilityCommand(
        jump_height=0.16, landing_dx=0.40, landing_dy=0.20,
    ),
}


def command_from_skill(
    skill: Skill,
    *,
    speed_scale: float = 1.0,
    jump_scale: float = 1.0,
) -> AgilityCommand:
    p = SKILL_PRESETS[skill]
    return AgilityCommand(
        vx=p.vx * speed_scale,
        vy=p.vy * speed_scale,
        wz=p.wz * speed_scale,
        height_offset=p.height_offset,
        jump_height=p.jump_height * jump_scale,
        gait_freq=p.gait_freq,
        gait_phase=p.gait_phase,
        gait_offset=p.gait_offset,
        gait_bound=p.gait_bound,
        landing_dx=p.landing_dx * jump_scale,
        landing_dy=p.landing_dy * jump_scale,
    )


def command_from_jump_target(target) -> AgilityCommand:
    """Build an agility command from a JumpTarget."""
    return AgilityCommand(
        jump_height=target.jump_height,
        landing_dx=target.landing_dx,
        landing_dy=target.landing_dy,
    )


def sample_command(
    rng: np.random.Generator,
    *,
    curriculum_level: float = 0.0,
    skill: Optional[Skill] = None,
    jump_bias: float = 0.0,
) -> AgilityCommand:
    """
    Sample a training command. Early curriculum prefers stand/walk/trot;
    later levels mix gaits and jump landings. jump_bias > 0 forces more
    jump samples (used by the parkour trainer).
    """
    level = float(np.clip(curriculum_level, 0.0, 1.0))

    if skill is not None:
        return command_from_skill(
            skill,
            speed_scale=0.5 + 0.5 * level,
            jump_scale=0.4 + 0.6 * level,
        )

    # Chance to sample a curriculum jump instead of a gait/stance skill
    jump_p = min(0.55, 0.12 + 0.35 * level + jump_bias)
    if rng.random() < jump_p:
        target = sample_jump_target(rng, curriculum_level=level)
        return command_from_jump_target(target)

    if level < 0.25:
        choices = [Skill.STAND, Skill.CROUCH, Skill.WALK, Skill.TROT]
        weights = [0.25, 0.15, 0.30, 0.30]
    elif level < 0.55:
        choices = [
            Skill.STAND, Skill.CROUCH, Skill.SIT, Skill.WALK,
            Skill.TROT, Skill.PACE, Skill.JUMP, Skill.JUMP_FORWARD,
        ]
        weights = [0.10, 0.08, 0.06, 0.18, 0.22, 0.10, 0.12, 0.14]
    else:
        choices = list(Skill)
        weights = [
            0.06, 0.06, 0.05, 0.12, 0.14, 0.08, 0.10, 0.07,
            0.10, 0.12, 0.10,
        ]

    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()
    picked = choices[int(rng.choice(len(choices), p=weights))]
    cmd = command_from_skill(
        picked,
        speed_scale=0.4 + 0.6 * level,
        jump_scale=0.35 + 0.65 * level,
    )
    return AgilityCommand(
        vx=float(np.clip(cmd.vx + rng.uniform(-0.1, 0.1) * level, -0.3, 1.4)),
        vy=float(np.clip(cmd.vy + rng.uniform(-0.15, 0.15) * level, -0.4, 0.4)),
        wz=float(np.clip(cmd.wz + rng.uniform(-0.3, 0.3) * level, -0.8, 0.8)),
        height_offset=float(np.clip(
            cmd.height_offset + rng.uniform(-0.03, 0.03), -0.18, 0.08)),
        jump_height=float(np.clip(cmd.jump_height, 0.0, 0.28)),
        gait_freq=float(np.clip(
            cmd.gait_freq + rng.uniform(-0.3, 0.3) * (cmd.gait_freq > 0),
            0.0, 3.5)),
        gait_phase=cmd.gait_phase,
        gait_offset=cmd.gait_offset,
        gait_bound=cmd.gait_bound,
        landing_dx=float(np.clip(cmd.landing_dx, -0.2, 1.0)),
        landing_dy=float(np.clip(cmd.landing_dy, -0.5, 0.5)),
    )


__all__ = [
    "AgilityCommand",
    "Skill",
    "SKILL_PRESETS",
    "command_from_skill",
    "command_from_jump_target",
    "sample_command",
    "JumpStage",
]
