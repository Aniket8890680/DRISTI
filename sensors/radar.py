"""
Radar Sensor Simulation:
Simulates a millimeter-wave automotive radar unit:
- Measures range, azimuth, and direct relative Doppler velocity
- High resilience in fog, heavy rain, and night conditions
- Lower angular resolution compared to LiDAR/Camera
- Radar Cross Section (RCS) sensitivity: metal vehicles produce strong reflections,
  whereas pedestrians and animals have lower reflection probabilities.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from simulation.actors import Actor, ActorType
from simulation.vehicle import Vehicle, normalize_angle
from simulation.environment import WeatherCondition


@dataclass
class RadarTarget:
    """Radar reflection target detection."""
    actor_id: int
    range_m: float                # Target distance (m)
    azimuth_rad: float            # Target angle relative to radar boresight (rad)
    radial_velocity_mps: float    # Range-rate / closing speed (positive = moving away, negative = closing)
    est_world_x: float            # Estimated world X position
    est_world_y: float            # Estimated world Y position
    confidence: float             # Target detection confidence based on RCS


class RadarSensor:
    """
    Automotive Doppler Radar model.
    """

    def __init__(
        self,
        max_range: float = 90.0,
        fov_deg: float = 40.0,         # Long-range forward radar cone
        range_noise_std: float = 0.3,   # meters
        angle_noise_std_deg: float = 1.0, # degrees
        velocity_noise_std: float = 0.2, # m/s
        random_seed: Optional[int] = None,
    ):
        self.max_range = max_range
        self.fov_rad = np.deg2rad(fov_deg)
        self.range_noise_std = range_noise_std
        self.angle_noise_std_rad = np.deg2rad(angle_noise_std_deg)
        self.velocity_noise_std = velocity_noise_std
        self.rng = np.random.RandomState(random_seed)

    def sense(
        self,
        ego: Vehicle,
        actors: List[Actor],
        weather: WeatherCondition = WeatherCondition.NORMAL
    ) -> List[RadarTarget]:
        """
        Scans forward cone and returns radar targets with Doppler velocity measurements.
        """
        targets: List[RadarTarget] = []
        ego_x, ego_y = ego.center_position
        ego_psi = ego.state.psi
        ego_vx = ego.state.v * np.cos(ego_psi)
        ego_vy = ego.state.v * np.sin(ego_psi)

        for actor in actors:
            if not actor.is_active:
                continue

            dx = actor.x - ego_x
            dy = actor.y - ego_y
            true_range = float(np.hypot(dx, dy))

            if true_range > self.max_range or true_range < 0.5:
                continue

            # Target angle relative to ego heading
            bearing = np.arctan2(dy, dx)
            rel_azimuth = normalize_angle(bearing - ego_psi)

            # Check if within radar FOV
            if abs(rel_azimuth) <= (self.fov_rad / 2.0):
                # Calculate relative velocity vector
                rel_vx = actor.vx - ego_vx
                rel_vy = actor.vy - ego_vy

                # Radial Doppler velocity: projection of relative velocity onto line of sight
                unit_los_x = dx / true_range
                unit_los_y = dy / true_range
                true_radial_v = rel_vx * unit_los_x + rel_vy * unit_los_y

                # Radar Cross Section (RCS) factor
                rcs_prob = 0.95
                if actor.type in [ActorType.PEDESTRIAN, ActorType.ANIMAL]:
                    rcs_prob = 0.70  # Lower radar reflectance
                elif actor.type in [ActorType.MOTORCYCLE, ActorType.BICYCLE]:
                    rcs_prob = 0.85

                if self.rng.uniform(0.0, 1.0) > rcs_prob:
                    continue  # Missed detection due to low RCS

                # Add sensor noise
                noisy_range = max(0.1, true_range + self.rng.normal(0, self.range_noise_std))
                noisy_azimuth = rel_azimuth + self.rng.normal(0, self.angle_noise_std_rad)
                noisy_radial_v = true_radial_v + self.rng.normal(0, self.velocity_noise_std)

                # Reconstruct estimated world position
                est_bearing = ego_psi + noisy_azimuth
                est_x = ego_x + noisy_range * np.cos(est_bearing)
                est_y = ego_y + noisy_range * np.sin(est_bearing)

                # Radar maintains high confidence even in rain/fog
                conf = 0.90 if weather in [WeatherCondition.FOG, WeatherCondition.RAIN] else 0.95

                targets.append(RadarTarget(
                    actor_id=actor.id,
                    range_m=float(noisy_range),
                    azimuth_rad=float(noisy_azimuth),
                    radial_velocity_mps=float(noisy_radial_v),
                    est_world_x=float(est_x),
                    est_world_y=float(est_y),
                    confidence=float(conf)
                ))

        return targets
