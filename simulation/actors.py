"""
Actors Module: Models heterogeneous dynamic traffic participants and static obstacles
characteristic of unstructured Indian roadways.

Actors:
- Car, Bus, Truck, Auto-Rickshaw, Motorcycle, Bicycle, Pedestrian, Pushcart, Cattle/Animal,
  Pothole, Construction Barrier, Static Obstacle.
- Includes realistic physical dimensions, kinematic update rules, and behavioral modes:
  (sudden cut-in, motorcycle filtering, animal wander/freeze, pedestrian crossing,
   wrong-side movement, informal merging).
- Separating Axis Theorem (SAT) oriented bounding box collision detection and clearance computation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict
import numpy as np


class ActorType(Enum):
    CAR = "car"
    BUS = "bus"
    TRUCK = "truck"
    AUTO_RICKSHAW = "auto_rickshaw"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    PEDESTRIAN = "pedestrian"
    PUSHCART = "pushcart"
    ANIMAL = "animal"
    POTHOLE = "pothole"
    CONSTRUCTION_BARRIER = "construction_barrier"
    STATIC_OBSTACLE = "static_obstacle"


class BehaviorMode(Enum):
    CONSTANT_VELOCITY = "constant_velocity"
    LANE_FOLLOWING = "lane_following"
    SUDDEN_CUT_IN = "sudden_cut_in"
    MOTORCYCLE_FILTERING = "motorcycle_filtering"
    PEDESTRIAN_CROSSING = "pedestrian_crossing"
    ANIMAL_WANDERING = "animal_wandering"
    INFORMAL_MERGE = "informal_merge"
    WRONG_SIDE = "wrong_side"
    STOPPED = "stopped"


# Standard dimensions (length, width) and uncertainty ratings for Indian road participants
ACTOR_PROPERTIES: Dict[ActorType, Dict[str, float]] = {
    ActorType.CAR: {"length": 4.5, "width": 1.8, "nominal_speed": 12.0, "uncertainty": 0.2},
    ActorType.BUS: {"length": 10.5, "width": 2.6, "nominal_speed": 9.0, "uncertainty": 0.25},
    ActorType.TRUCK: {"length": 9.0, "width": 2.5, "nominal_speed": 8.0, "uncertainty": 0.25},
    ActorType.AUTO_RICKSHAW: {"length": 2.8, "width": 1.35, "nominal_speed": 7.5, "uncertainty": 0.45},
    ActorType.MOTORCYCLE: {"length": 2.0, "width": 0.8, "nominal_speed": 11.0, "uncertainty": 0.6},
    ActorType.BICYCLE: {"length": 1.8, "width": 0.6, "nominal_speed": 4.0, "uncertainty": 0.4},
    ActorType.PEDESTRIAN: {"length": 0.6, "width": 0.6, "nominal_speed": 1.4, "uncertainty": 0.75},
    ActorType.PUSHCART: {"length": 2.2, "width": 1.2, "nominal_speed": 1.8, "uncertainty": 0.4},
    ActorType.ANIMAL: {"length": 2.3, "width": 1.1, "nominal_speed": 1.2, "uncertainty": 0.95},
    ActorType.POTHOLE: {"length": 0.8, "width": 0.8, "nominal_speed": 0.0, "uncertainty": 0.05},
    ActorType.CONSTRUCTION_BARRIER: {"length": 3.0, "width": 1.0, "nominal_speed": 0.0, "uncertainty": 0.05},
    ActorType.STATIC_OBSTACLE: {"length": 2.0, "width": 1.5, "nominal_speed": 0.0, "uncertainty": 0.05},
}


class Actor:
    """
    Simulated dynamic or static traffic participant on the road.
    """

    def __init__(
        self,
        actor_id: int,
        actor_type: ActorType,
        x: float,
        y: float,
        v: float,
        heading: float,
        behavior: BehaviorMode = BehaviorMode.CONSTANT_VELOCITY,
        length: Optional[float] = None,
        width: Optional[float] = None,
        behavior_params: Optional[dict] = None,
    ):
        self.id = actor_id
        self.type = actor_type
        self.x = float(x)
        self.y = float(y)
        self.v = float(v)
        self.heading = float(heading)
        self.vx = self.v * np.cos(self.heading)
        self.vy = self.v * np.sin(self.heading)
        self.behavior = behavior
        self.behavior_params = behavior_params if behavior_params is not None else {}

        # Default physical dimensions
        props = ACTOR_PROPERTIES.get(actor_type, {"length": 2.0, "width": 1.0, "uncertainty": 0.5})
        self.length = float(length if length is not None else props["length"])
        self.width = float(width if width is not None else props["width"])
        self.uncertainty_factor = props.get("uncertainty", 0.5)

        # Behavioral state memory
        self.behavior_time = 0.0
        self.is_active = True
        self.history: List[Tuple[float, float, float, float]] = [(0.0, self.x, self.y, self.v)]

    def get_bounding_box(self) -> np.ndarray:
        """
        Returns the 4 corner coordinates in 2D world space:
        [Front-Left, Front-Right, Rear-Right, Rear-Left]
        """
        half_l = self.length / 2.0
        half_w = self.width / 2.0

        local_corners = np.array([
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w]
        ])

        cos_h = np.cos(self.heading)
        sin_h = np.sin(self.heading)
        R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

        world_corners = (R @ local_corners.T).T + np.array([self.x, self.y])
        return world_corners

    def update(self, dt: float, current_time: float, road: Optional[object] = None) -> None:
        """
        Updates the actor's position, velocity, and orientation based on its behavior mode.
        """
        if not self.is_active or self.behavior == BehaviorMode.STOPPED:
            return

        self.behavior_time += dt

        # Behavioral controller
        if self.behavior == BehaviorMode.CONSTANT_VELOCITY:
            self.x += self.vx * dt
            self.y += self.vy * dt

        elif self.behavior == BehaviorMode.SUDDEN_CUT_IN:
            # Starts moving alongside, then abruptly changes yaw/heading towards the target lateral position
            cut_trigger_time = self.behavior_params.get("trigger_time", 1.5)
            target_heading = self.behavior_params.get("target_heading", self.heading - np.deg2rad(30))
            if self.behavior_time >= cut_trigger_time:
                # Steer sharply across path
                yaw_rate = self.behavior_params.get("yaw_rate", -0.3)
                self.heading = float(self.heading + yaw_rate * dt)
                self.vx = self.v * np.cos(self.heading)
                self.vy = self.v * np.sin(self.heading)
            self.x += self.vx * dt
            self.y += self.vy * dt

        elif self.behavior == BehaviorMode.MOTORCYCLE_FILTERING:
            # Weaves through gaps with lateral sinusoidal motion
            freq = self.behavior_params.get("frequency", 1.2)
            amp = self.behavior_params.get("amplitude", 0.6)
            base_heading = self.behavior_params.get("base_heading", 0.0)
            self.heading = float(base_heading + amp * np.cos(freq * self.behavior_time) * 0.4)
            self.vx = self.v * np.cos(self.heading)
            self.vy = self.v * np.sin(self.heading)
            self.x += self.vx * dt
            self.y += self.vy * dt

        elif self.behavior == BehaviorMode.PEDESTRIAN_CROSSING:
            # Crosses road laterally, may pause or suddenly accelerate
            pause_start = self.behavior_params.get("pause_start", -1.0)
            pause_duration = self.behavior_params.get("pause_duration", 1.0)
            if pause_start <= self.behavior_time < (pause_start + pause_duration):
                current_v = 0.0
            else:
                current_v = self.v
            self.x += current_v * np.cos(self.heading) * dt
            self.y += current_v * np.sin(self.heading) * dt

        elif self.behavior == BehaviorMode.ANIMAL_WANDERING:
            # Animal (cattle/dog): moves slowly across road, then might freeze or turn randomly
            freeze_start = self.behavior_params.get("freeze_start", 2.5)
            freeze_duration = self.behavior_params.get("freeze_duration", 3.0)
            if freeze_start <= self.behavior_time < (freeze_start + freeze_duration):
                current_v = 0.0  # Cattle freezes in middle of road
            else:
                # Slight random walk drift
                wander = np.sin(1.5 * self.behavior_time) * 0.05
                self.heading += wander
                current_v = self.v
            self.vx = current_v * np.cos(self.heading)
            self.vy = current_v * np.sin(self.heading)
            self.x += self.vx * dt
            self.y += self.vy * dt

        elif self.behavior == BehaviorMode.WRONG_SIDE:
            # Traveling against nominal traffic flow
            self.x += self.vx * dt
            self.y += self.vy * dt

        elif self.behavior == BehaviorMode.INFORMAL_MERGE:
            # Shallow angle informal entry from shoulder
            merge_rate = self.behavior_params.get("merge_rate", 0.1)
            self.heading += merge_rate * dt
            self.vx = self.v * np.cos(self.heading)
            self.vy = self.v * np.sin(self.heading)
            self.x += self.vx * dt
            self.y += self.vy * dt

        else:
            self.x += self.vx * dt
            self.y += self.vy * dt

        self.history.append((current_time, self.x, self.y, self.v))


# ==============================================================================
# GEOMETRIC COLLISION DETECTION: Separating Axis Theorem (SAT) & Distance Checks
# ==============================================================================

def check_sat_collision(poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
    """
    Checks if two convex 2D polygons (such as oriented bounding boxes) intersect
    using the Separating Axis Theorem (SAT).
    poly_a, poly_b: Nx2 arrays of polygon vertices.
    """
    for polygon in (poly_a, poly_b):
        num_vertices = len(polygon)
        for i in range(num_vertices):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % num_vertices]

            # Edge vector and normal (perpendicular) axis
            edge = p2 - p1
            axis = np.array([-edge[1], edge[0]])
            norm = np.hypot(axis[0], axis[1])
            if norm < 1e-8:
                continue
            axis = axis / norm

            # Project poly_a onto axis
            proj_a = np.dot(poly_a, axis)
            min_a, max_a = np.min(proj_a), np.max(proj_a)

            # Project poly_b onto axis
            proj_b = np.dot(poly_b, axis)
            min_b, max_b = np.min(proj_b), np.max(proj_b)

            # Check for separating gap
            if max_a < min_b or max_b < min_a:
                return False  # Separating axis found -> No collision

    return True  # No separating axis -> Polygons intersect


def compute_polygon_clearance(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """
    Computes the minimum Euclidean distance (clearance) between two 2D convex polygons.
    Returns 0.0 if they overlap.
    """
    if check_sat_collision(poly_a, poly_b):
        return 0.0

    # Minimum distance between any vertex of poly_a to edges of poly_b and vice-versa
    min_dist = float('inf')

    for poly1, poly2 in [(poly_a, poly_b), (poly_b, poly_a)]:
        for p in poly1:
            for i in range(len(poly2)):
                e1 = poly2[i]
                e2 = poly2[(i + 1) % len(poly2)]
                dist = point_to_segment_distance(p, e1, e2)
                if dist < min_dist:
                    min_dist = dist

    return float(min_dist)


def point_to_segment_distance(point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> float:
    """Computes Euclidean distance from a point to a 2D line segment."""
    seg = seg_end - seg_start
    seg_len_sq = np.dot(seg, seg)
    if seg_len_sq < 1e-8:
        return float(np.linalg.norm(point - seg_start))

    t = np.dot(point - seg_start, seg) / seg_len_sq
    t_clamped = np.clip(t, 0.0, 1.0)
    projection = seg_start + t_clamped * seg
    return float(np.linalg.norm(point - projection))
