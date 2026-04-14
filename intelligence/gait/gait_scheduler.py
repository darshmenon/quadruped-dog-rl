"""
Gait Scheduler — switches between locomotion gaits based on speed or command.

Gaits supported:
    - stand:    all feet on ground, body held up
    - trot:     diagonal pairs move together (FL+RR, FR+RL)
    - walk:     one foot at a time, stable at low speed
    - canter:   three-beat gait, medium-high speed
    - bound:    front pair then rear pair, high speed
    - pronk:    all four feet leave ground simultaneously

Usage:
    from intelligence.gait.gait_scheduler import GaitScheduler
    scheduler = GaitScheduler()
    cmd = scheduler.get_gait_command(speed=0.8)
"""

from dataclasses import dataclass
from enum import Enum
from typing import List


class Gait(Enum):
    STAND  = "stand"
    WALK   = "walk"
    TROT   = "trot"
    CANTER = "canter"
    BOUND  = "bound"
    PRONK  = "pronk"


@dataclass
class GaitParams:
    name: str
    frequency: float       # step frequency (Hz)
    duty_factor: float     # fraction of cycle each foot is on ground
    phase_offsets: List[float]  # [FL, FR, RL, RR] phase offsets (0-1)
    speed_range: tuple     # (min, max) m/s


GAITS = {
    Gait.STAND:  GaitParams("stand",  0.0, 1.00, [0.0, 0.0, 0.0, 0.0],  (0.0, 0.05)),
    Gait.WALK:   GaitParams("walk",   1.2, 0.75, [0.0, 0.5, 0.25, 0.75],(0.05, 0.4)),
    Gait.TROT:   GaitParams("trot",   2.0, 0.60, [0.0, 0.5, 0.5, 0.0],  (0.4, 1.5)),
    Gait.CANTER: GaitParams("canter", 2.8, 0.55, [0.0, 0.33, 0.66, 0.2],(1.5, 2.5)),
    Gait.BOUND:  GaitParams("bound",  3.5, 0.40, [0.0, 0.0, 0.5, 0.5],  (2.5, 4.0)),
    Gait.PRONK:  GaitParams("pronk",  3.0, 0.30, [0.0, 0.0, 0.0, 0.0],  (4.0, 6.0)),
}


class GaitScheduler:
    def __init__(self):
        self.current_gait = Gait.STAND

    def select_gait(self, speed: float) -> Gait:
        if speed < 0:
            return Gait.STAND
        for gait, params in GAITS.items():
            if params.speed_range[0] <= speed < params.speed_range[1]:
                return gait
        # Speed exceeds all defined ranges — use the fastest gait
        return Gait.PRONK

    def get_gait_params(self, speed: float) -> GaitParams:
        gait = self.select_gait(speed)
        if gait != self.current_gait:
            print(f"Gait switch: {self.current_gait.value} -> {gait.value} at {speed:.2f} m/s")
            self.current_gait = gait
        return GAITS[gait]

    def get_phase(self, leg_index: int, t: float, speed: float) -> float:
        """Returns current swing/stance phase (0=stance, 1=swing) for a leg."""
        params = self.get_gait_params(speed)
        if params.frequency == 0:
            return 0.0
        cycle_pos = (t * params.frequency + params.phase_offsets[leg_index]) % 1.0
        return 1.0 if cycle_pos > params.duty_factor else 0.0
