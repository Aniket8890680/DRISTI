"""
Camera Sensor Simulation:
Simulates a forward-facing camera system with:
- Configurable Field of View (FOV) and maximum perception range
- Object type classification (Car, Motorcycle, Rickshaw, Animal, Pedestrian, etc.)
- Visual bounding box detection
- Environmental sensitivity (fog, heavy rain, darkness reduce confidence and detection range)
- Gaussian noise on bounding box centers
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np

from simulation.actors import Actor, ActorType
from simulation.vehicle import Vehicle, normalize_angle
from simulation.environment import WeatherCondition


@dataclass
class CameraDetection:
    """Individual object detected by the simulated camera."""
    actor_id: int
    actor_type: ActorType
    x: float             # estimated world X (m)
    y: float             # estimated world Y (m)
    box_corners: np.ndarray # 4x2 world corners
    confidence: float    # 0.0 to 1.0
    relative_distance: float
    relative_angle: float # rad from camera boresight


class CameraSensor:
    """
    Forward-facing monocular / stereo camera model.
    """

    def __init__(
        self,
        fov_deg: float = 90.0,
        max_range: float = 65.0,
        pos_noise_std: float = 0.25, # meters standard deviation
        random_seed: Optional[int] = None,
    ):
        self.fov_rad = np.deg2rad(fov_deg)
        self.max_range = max_range
        self.pos_noise_std = pos_noise_std
        self.rng = np.random.RandomState(random_seed)

    def sense(
        self,
        ego: Vehicle,
        actors: List[Actor],
        weather: WeatherCondition = WeatherCondition.NORMAL
    ) -> List[CameraDetection]:
        """
        Processes surrounding actors and returns camera detections within FOV and range.
        Weather modulates effective range and detection confidence.
        """
        detections: List[CameraDetection] = []
        
        # Weather degradation multipliers
        range_mult = 1.0
        conf_mult = 1.0
        noise_mult = 1.0

        if weather == WeatherCondition.RAIN:
            range_mult = 0.85
            conf_mult = 0.85
            noise_mult = 1.3
        elif weather == WeatherCondition.FOG:
            range_mult = 0.50
            conf_mult = 0.60
            noise_mult = 1.8
        elif weather == WeatherCondition.NIGHT:
            range_mult = 0.70
            conf_mult = 0.75
            noise_mult = 1.4

        effective_range = self.max_range * range_mult
        ego_x, ego_y = ego.center_position
        ego_psi = ego.state.psi

        for actor in actors:
            if not actor.is_active:
                continue

            dx = actor.x - ego_x
            dy = actor.y - ego_y
            dist = float(np.hypot(dx, dy))

            if dist > effective_range or dist < 0.1:
                continue

            # Angle relative to ego heading
            abs_angle = np.arctan2(dy, dx)
            rel_angle = normalize_angle(abs_angle - ego_psi)

            # Check if within FOV
            if abs(rel_angle) <= (self.fov_rad / 2.0):
                # Calculate confidence: drops with distance and weather
                base_conf = max(0.2, 1.0 - (dist / effective_range) * 0.75)
                # Animals and pedestrians have slightly lower visual confidence in clutter
                type_conf_factor = 0.9 if actor.type in [ActorType.ANIMAL, ActorType.PEDESTRIAN] else 1.0
                final_conf = float(np.clip(base_conf * conf_mult * type_conf_factor, 0.1, 0.99))

                # Add measurement noise
                noise_std = self.pos_noise_std * noise_mult * (1.0 + dist / self.max_range)
                noisy_x = actor.x + self.rng.normal(0, noise_std)
                noisy_y = actor.y + self.rng.normal(0, noise_std)

                # Generate noisy bounding box
                box = actor.get_bounding_box()
                noisy_box = box + self.rng.normal(0, noise_std * 0.5, size=box.shape)

                detections.append(CameraDetection(
                    actor_id=actor.id,
                    actor_type=actor.type,
                    x=float(noisy_x),
                    y=float(noisy_y),
                    box_corners=noisy_box,
                    confidence=final_conf,
                    relative_distance=dist,
                    relative_angle=rel_angle
                ))

        return detections
