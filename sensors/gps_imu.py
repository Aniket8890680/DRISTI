"""
GPS and IMU Sensor Simulation:
Simulates high-rate onboard localization package:
- GNSS / GPS: Global position (x, y) coordinates with realistic dilution of precision noise
- IMU: 6-DOF gyroscope and accelerometer providing yaw/heading, speed, and linear acceleration
- CRITICAL ARCHITECTURAL RULE:
  GPS is used strictly for global localization and road waypoint guidance.
  GPS is NEVER used for obstacle detection or collision avoidance!
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from simulation.vehicle import Vehicle, normalize_angle


@dataclass
class EgoPoseMeasurement:
    """Localized state of the ego autonomous vehicle."""
    x: float             # estimated global X (m)
    y: float             # estimated global Y (m)
    psi: float           # estimated heading (rad)
    v: float             # estimated longitudinal velocity (m/s)
    a: float             # estimated longitudinal acceleration (m/s^2)
    timestamp: float     # simulation time


class GpsImuSensor:
    """
    Simulated GPS + IMU dead-reckoning fusion sensor.
    """

    def __init__(
        self,
        gps_pos_noise_std: float = 0.25,   # meters (standard automotive RTK/DGPS assisted)
        heading_noise_std: float = 0.02,   # radians (~1.1 degrees IMU gyro drift)
        speed_noise_std: float = 0.10,     # m/s (wheel speed sensor / IMU)
        accel_noise_std: float = 0.15,     # m/s^2 (accelerometer noise)
        random_seed: Optional[int] = None,
    ):
        self.gps_pos_noise_std = gps_pos_noise_std
        self.heading_noise_std = heading_noise_std
        self.speed_noise_std = speed_noise_std
        self.accel_noise_std = accel_noise_std
        self.rng = np.random.RandomState(random_seed)

    def sense(self, ego: Vehicle, current_time: float) -> EgoPoseMeasurement:
        """
        Samples ego state and applies sensor noise characteristics.
        """
        noisy_x = ego.state.x + self.rng.normal(0, self.gps_pos_noise_std)
        noisy_y = ego.state.y + self.rng.normal(0, self.gps_pos_noise_std)
        noisy_psi = normalize_angle(ego.state.psi + self.rng.normal(0, self.heading_noise_std))
        noisy_v = max(0.0, ego.state.v + self.rng.normal(0, self.speed_noise_std))
        noisy_a = ego.state.a + self.rng.normal(0, self.accel_noise_std)

        return EgoPoseMeasurement(
            x=float(noisy_x),
            y=float(noisy_y),
            psi=float(noisy_psi),
            v=float(noisy_v),
            a=float(noisy_a),
            timestamp=current_time
        )
