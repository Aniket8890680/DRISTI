"""
Environment Module: Manages the closed-loop simulation world state.

Handles:
- Time progression and synchronization
- Road network and surface hazards
- Dynamic traffic participants and behavioral updates
- Ego vehicle dynamics integration
- Ground-truth collision, near-miss, and off-road detection
- Environmental / weather conditions (NORMAL, RAIN, FOG, NIGHT)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict
import numpy as np

from simulation.road import Road, Pothole
from simulation.vehicle import Vehicle
from simulation.actors import Actor, ActorType, check_sat_collision, compute_polygon_clearance


class WeatherCondition(Enum):
    NORMAL = "normal"
    RAIN = "rain"
    FOG = "fog"
    NIGHT = "night"


@dataclass
class CollisionEvent:
    """Record of a collision during simulation."""
    time: float
    ego_x: float
    ego_y: float
    obstacle_id: int
    obstacle_type: str
    relative_speed: float


@dataclass
class SimulationStats:
    """Live runtime simulation statistics."""
    collision_count: int = 0
    near_miss_count: int = 0
    off_road_count: int = 0
    pothole_hit_count: int = 0
    min_obstacle_clearance: float = float('inf')
    collisions: List[CollisionEvent] = field(default_factory=list)


class Environment:
    """
    Top-level simulation environment coordinating the road, ego vehicle,
    surrounding actors, and environmental conditions.
    """

    def __init__(
        self,
        road: Road,
        ego_vehicle: Vehicle,
        weather: WeatherCondition = WeatherCondition.NORMAL,
        dt: float = 0.05,
        near_miss_threshold: float = 1.0,  # meters
    ):
        self.road = road
        self.ego = ego_vehicle
        self.weather = weather
        self.dt = dt
        self.near_miss_threshold = near_miss_threshold
        
        self.actors: List[Actor] = []
        self.time: float = 0.0
        self.step_count: int = 0
        self.stats = SimulationStats()
        self.is_done: bool = False

    def add_actor(self, actor: Actor) -> None:
        """Adds a dynamic or static traffic participant to the environment."""
        self.actors.append(actor)

    def step(self, target_accel: float, target_steer: float) -> Tuple[bool, float]:
        """
        Advances the simulation by dt seconds.
        Returns: (has_collision, min_clearance_this_frame)
        """
        if self.is_done:
            return True, 0.0

        # 1. Step ego vehicle physically
        self.ego.step(target_accel, target_steer, self.dt)

        # 2. Update surrounding dynamic actors
        for actor in self.actors:
            actor.update(self.dt, self.time, self.road)

        # 3. Collision and proximity checking
        has_collision = False
        min_clearance = float('inf')
        ego_box = self.ego.get_bounding_box()

        # Check against dynamic/static actors
        for actor in self.actors:
            if not actor.is_active:
                continue

            actor_box = actor.get_bounding_box()
            collides = check_sat_collision(ego_box, actor_box)
            clearance = compute_polygon_clearance(ego_box, actor_box)

            if clearance < min_clearance:
                min_clearance = clearance

            if collides:
                has_collision = True
                self.stats.collision_count += 1
                rel_v = float(np.hypot(self.ego.state.v * np.cos(self.ego.state.psi) - actor.vx,
                                       self.ego.state.v * np.sin(self.ego.state.psi) - actor.vy))
                self.stats.collisions.append(CollisionEvent(
                    time=self.time,
                    ego_x=self.ego.state.x,
                    ego_y=self.ego.state.y,
                    obstacle_id=actor.id,
                    obstacle_type=actor.type.value,
                    relative_speed=rel_v
                ))
            elif clearance < self.near_miss_threshold:
                self.stats.near_miss_count += 1

        # Check against potholes (represented as circles)
        cx, cy = self.ego.center_position
        for pothole in self.road.potholes:
            dist = float(np.hypot(cx - pothole.x, cy - pothole.y))
            # If vehicle footprint covers the pothole
            effective_radius = max(self.ego.limits.length, self.ego.limits.width) / 2.0
            if dist < (effective_radius + pothole.radius * 0.5):
                self.stats.pothole_hit_count += 1

        # Check road boundary compliance
        if not self.road.is_box_in_road(ego_box, margin=0.0):
            self.stats.off_road_count += 1

        # Update global minimum clearance
        if min_clearance < self.stats.min_obstacle_clearance:
            self.stats.min_obstacle_clearance = min_clearance

        self.time += self.dt
        self.step_count += 1

        return has_collision, min_clearance

    def get_ground_truth_state(self) -> Dict:
        """Provides ground truth data for evaluation and sensor simulation."""
        return {
            "time": self.time,
            "ego": {
                "x": self.ego.state.x,
                "y": self.ego.state.y,
                "v": self.ego.state.v,
                "a": self.ego.state.a,
                "psi": self.ego.state.psi,
                "delta": self.ego.state.delta,
                "box": self.ego.get_bounding_box()
            },
            "actors": [
                {
                    "id": a.id,
                    "type": a.type.value,
                    "x": a.x,
                    "y": a.y,
                    "v": a.v,
                    "vx": a.vx,
                    "vy": a.vy,
                    "heading": a.heading,
                    "length": a.length,
                    "width": a.width,
                    "box": a.get_bounding_box(),
                    "behavior": a.behavior.value
                }
                for a in self.actors if a.is_active
            ],
            "potholes": [
                {"id": p.id, "x": p.x, "y": p.y, "radius": p.radius, "severity": p.severity}
                for p in self.road.potholes
            ]
        }
