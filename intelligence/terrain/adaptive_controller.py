"""
Adaptive Controller — combines terrain estimation and gait scheduling
to produce safe velocity commands adapted to the current surface.

Sits between high-level navigation commands and the low-level gait controller.
Reduces speed on slopes, increases foot clearance on rough terrain, etc.

Usage:
    from intelligence.terrain.adaptive_controller import AdaptiveController
    ctrl = AdaptiveController()
    safe_cmd = ctrl.adapt(desired_speed=1.2, imu_pitch=0.12, contacts=[110,115,108,120])
"""

from dataclasses import dataclass
from intelligence.gait.gait_scheduler import GaitScheduler, GaitParams
from intelligence.perception.terrain_estimator import TerrainEstimator, TerrainEstimate, TerrainType
from typing import List


@dataclass
class AdaptedCommand:
    linear_x: float
    angular_z: float
    gait: str
    foot_clearance: float
    terrain: str
    slope_deg: float


class AdaptiveController:
    def __init__(self):
        self.gait_scheduler = GaitScheduler()
        self.terrain_estimator = TerrainEstimator()

    def adapt(
        self,
        desired_speed: float,
        desired_angular: float = 0.0,
        imu_roll: float = 0.0,
        imu_pitch: float = 0.0,
        contacts: List[float] = None,
    ) -> AdaptedCommand:

        if contacts is None:
            contacts = [100.0, 100.0, 100.0, 100.0]

        terrain: TerrainEstimate = self.terrain_estimator.estimate(
            imu_roll=imu_roll,
            imu_pitch=imu_pitch,
            contacts=contacts,
        )

        # Clamp speed to terrain limit
        safe_speed = min(desired_speed, terrain.recommended_speed_limit)

        # On steep slopes, reduce angular too
        if terrain.terrain_type == TerrainType.SLOPE:
            desired_angular *= 0.5

        # On stairs, near-zero speed
        if terrain.terrain_type == TerrainType.STAIRS:
            safe_speed = min(safe_speed, 0.3)

        gait_params: GaitParams = self.gait_scheduler.get_gait_params(safe_speed)

        return AdaptedCommand(
            linear_x=safe_speed,
            angular_z=desired_angular,
            gait=gait_params.name,
            foot_clearance=terrain.recommended_foot_clearance,
            terrain=terrain.terrain_type.value,
            slope_deg=terrain.slope_deg,
        )
