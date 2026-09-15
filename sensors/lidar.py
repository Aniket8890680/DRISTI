"""
LiDAR Sensor Simulation:
Simulates a multi-beam 2D/3D LiDAR sensor:
- Accurate spatial range measurements and obstacle boundary point extraction
- High angular and range accuracy (standard deviation ~ 0.05m)
- Detects obstacle geometry regardless of visual texture or color
- Atmospheric backscatter attenuation in heavy fog and rain
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from simulation.actors import Actor
from simulation.vehicle import Vehicle, normalize_angle
from simulation.environment import WeatherCondition


@dataclass
class LidarObstacleCluster:
    """Detected geometric point cluster representing an obstacle boundary."""
    actor_id: int
    centroid_x: float
    centroid_y: float
    estimated_length: float
    estimated_width: float
    heading: float
    point_cloud: np.ndarray    # Nx2 array of boundary points
    distance_to_ego: float
    confidence: float


class LidarSensor:
    """
    Simulated 360-degree range-finding LiDAR sensor.
    """

    def __init__(
        self,
        max_range: float = 60.0,
        num_rays: int = 180,
        range_noise_std: float = 0.05, # high spatial accuracy
        random_seed: Optional[int] = None,
    ):
        self.max_range = max_range
        self.num_rays = num_rays
        self.range_noise_std = range_noise_std
        self.rng = np.random.RandomState(random_seed)

    def sense(
        self,
        ego: Vehicle,
        actors: List[Actor],
        weather: WeatherCondition = WeatherCondition.NORMAL
    ) -> List[LidarObstacleCluster]:
        """
        Scans surroundings and produces geometric clusters for detected obstacles.
        """
        clusters: List[LidarObstacleCluster] = []
        
        # Weather attenuation
        range_mult = 1.0
        conf_mult = 1.0
        if weather == WeatherCondition.FOG:
            range_mult = 0.70
            conf_mult = 0.80
        elif weather == WeatherCondition.RAIN:
            range_mult = 0.85
            conf_mult = 0.90

        effective_range = self.max_range * range_mult
        ego_x, ego_y = ego.center_position

        for actor in actors:
            if not actor.is_active:
                continue

            dist = float(np.hypot(actor.x - ego_x, actor.y - ego_y))
            if dist > effective_range:
                continue

            # Sample points along the obstacle's oriented bounding box
            box = actor.get_bounding_box()
            sampled_points = []
            
            # Interpolate points along edges to simulate laser beam hits
            for i in range(4):
                p1 = box[i]
                p2 = box[(i + 1) % 4]
                for alpha in [0.2, 0.5, 0.8]:
                    pt = p1 + alpha * (p2 - p1)
                    # Add laser range noise
                    noisy_pt = pt + self.rng.normal(0, self.range_noise_std, size=2)
                    sampled_points.append(noisy_pt)

            pts_arr = np.array(sampled_points)
            
            # Estimate centroid from points
            centroid = np.mean(pts_arr, axis=0)
            
            # Geometry estimates with slight measurement noise
            est_l = actor.length + self.rng.normal(0, self.range_noise_std)
            est_w = actor.width + self.rng.normal(0, self.range_noise_std)
            est_heading = actor.heading + self.rng.normal(0, 0.03)

            clusters.append(LidarObstacleCluster(
                actor_id=actor.id,
                centroid_x=float(centroid[0]),
                centroid_y=float(centroid[1]),
                estimated_length=float(est_l),
                estimated_width=float(est_w),
                heading=float(est_heading),
                point_cloud=pts_arr,
                distance_to_ego=dist,
                confidence=float(np.clip(0.95 * conf_mult, 0.1, 0.99))
            ))

        return clusters
