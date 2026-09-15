"""
Drivable Area Estimation Module:
Estimates the dynamically traversable road corridor without relying on lane markings.

Inputs:
- Road boundaries and varying width profile
- Static hazards (potholes, construction zones, road-edge ditches)
- Dynamic traffic participants and their safety clearance margins
- Environmental uncertainty factors

Outputs:
- Longitudinal-lateral boundary intervals: [d_min(s), d_max(s)]
- Free-space polygons / drivable corridors
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np

from simulation.road import Road
from simulation.actors import ActorType
from tracking.tracker import TrackedObject


@dataclass
class DrivableCorridorSegment:
    """Represents the safe lateral driving bounds at longitudinal station s."""
    s: float
    d_min: float  # Rightmost drivable lateral offset (m)
    d_max: float  # Leftmost drivable lateral offset (m)
    has_pothole: bool = False
    has_construction: bool = False


class DrivableAreaEstimator:
    """
    Computes lane-independent drivable free space for unstructured Indian roads.
    """

    def __init__(
        self,
        nominal_safety_margin: float = 0.5, # meters from road edge
        actor_safety_buffer: float = 0.8,   # meters around dynamic obstacles
    ):
        self.edge_margin = nominal_safety_margin
        self.actor_buffer = actor_safety_buffer

    def estimate_corridor(
        self,
        road: Road,
        ego_s: float,
        tracked_objects: List[TrackedObject],
        horizon_m: float = 50.0,
        ds: float = 2.0
    ) -> List[DrivableCorridorSegment]:
        """
        Generates lateral drivable intervals [d_min, d_max] from ego_s to ego_s + horizon_m.
        """
        s_stations = np.arange(ego_s, min(road.total_length, ego_s + horizon_m) + ds, ds)
        corridor: List[DrivableCorridorSegment] = []

        # Project tracked objects into Frenet frame (s_obj, d_obj)
        frenet_objects = []
        for obj in tracked_objects:
            s_obj, d_obj = road.cartesian_to_frenet(obj.x, obj.y)
            # Effective radius considering length, width, and safety buffer
            r_eff = max(obj.length, obj.width) / 2.0 + self.actor_buffer
            frenet_objects.append((s_obj, d_obj, r_eff, obj))

        # Project potholes
        frenet_potholes = []
        for ph in road.potholes:
            s_ph, d_ph = road.cartesian_to_frenet(ph.x, ph.y)
            frenet_potholes.append((s_ph, d_ph, ph.radius + 0.4))

        for s in s_stations:
            half_w = road.get_width(s) / 2.0
            d_min = -half_w + self.edge_margin  # Right boundary
            d_max = half_w - self.edge_margin   # Left boundary

            has_pothole = False
            has_construction = False

            # Check for nearby potholes restricting lateral corridor
            for s_ph, d_ph, r_ph in frenet_potholes:
                if abs(s - s_ph) < r_ph:
                    has_pothole = True
                    # If pothole is on right side (d < 0), push d_min to avoid it
                    if d_ph < 0:
                        d_min = max(d_min, d_ph + r_ph)
                    else:
                        d_max = min(d_max, d_ph - r_ph)

            # Check for nearby static/slow obstacles
            for s_obj, d_obj, r_eff, obj in frenet_objects:
                if abs(s - s_obj) < r_eff:
                    if obj.actor_type == ActorType.CONSTRUCTION_BARRIER:
                        has_construction = True
                    
                    # Constrain corridor around the obstacle
                    if d_obj < 0:
                        d_min = max(d_min, d_obj + r_eff)
                    else:
                        d_max = min(d_max, d_obj - r_eff)

            # Ensure valid interval ordering
            if d_min > d_max:
                # Completely blocked at this station
                d_min_adj = (d_min + d_max) / 2.0
                d_max_adj = d_min_adj
            else:
                d_min_adj = d_min
                d_max_adj = d_max

            corridor.append(DrivableCorridorSegment(
                s=float(s),
                d_min=float(d_min_adj),
                d_max=float(d_max_adj),
                has_pothole=has_pothole,
                has_construction=has_construction
            ))

        return corridor

    def is_trajectory_within_drivable_area(
        self,
        road: Road,
        trajectory_points: List[Tuple[float, float]], # List of (x, y)
        margin: float = 0.2
    ) -> bool:
        """
        Checks if every waypoint along a candidate trajectory lies strictly
        within the drivable road boundaries with safety margin.
        """
        for x, y in trajectory_points:
            if not road.is_point_in_road(x, y, margin=margin):
                return False
        return True
