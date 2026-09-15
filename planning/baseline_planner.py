"""
Baseline Planner Module:
Implements a standard lane/path-following planner with simple reactive braking,
used for quantitative SIH benchmarking against our Adaptive Planner.

Baseline Characteristics:
- Naive centerline / lane tracking without lateral evasion
- Simple distance-to-lead-obstacle stopping rule
- Ignorant of actor trajectory predictions and uncertainty envelopes
- Vulnerable to:
  * Missing lane markings
  * Abrupt motorcycle cut-ins
  * Potholes on the road
  * Unpredictable animal crossings
  * Narrow village road bottlenecks
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np

from simulation.road import Road
from simulation.vehicle import Vehicle
from tracking.tracker import TrackedObject
from planning.candidate_paths import CandidateTrajectory, TrajectoryWaypoint


class BaselinePlanner:
    """
    Standard naive lane-follower with basic reactive emergency stop.
    """

    def __init__(
        self,
        target_cruise_speed: float = 12.0,
        safe_following_distance: float = 18.0,
        emergency_stop_distance: float = 8.0,
    ):
        self.target_speed = target_cruise_speed
        self.safe_dist = safe_following_distance
        self.emergency_dist = emergency_stop_distance
        self.current_trajectory: Optional[CandidateTrajectory] = None
        self.replanning_count: int = 0

    def plan(
        self,
        ego: Vehicle,
        road: Road,
        tracked_objects: List[TrackedObject],
    ) -> CandidateTrajectory:
        """
        Plans a fixed centerline-following path with naive reactive braking.
        """
        ego_s, ego_d = road.cartesian_to_frenet(ego.state.x, ego.state.y)
        v0 = max(0.0, ego.state.v)
        
        # Check closest obstacle directly ahead in the path (d close to 0)
        closest_ahead_dist = float('inf')
        for obj in tracked_objects:
            s_obj, d_obj = road.cartesian_to_frenet(obj.x, obj.y)
            # Check if obstacle is ahead and within the same narrow corridor (abs(d) < 1.5m)
            if s_obj > ego_s and abs(d_obj) < 1.5:
                dist = s_obj - ego_s
                if dist < closest_ahead_dist:
                    closest_ahead_dist = dist

        # Naive reactive speed decision
        if closest_ahead_dist < self.emergency_dist:
            target_v = 0.0
            maneuver = "Baseline Emergency Stop"
        elif closest_ahead_dist < self.safe_dist:
            target_v = self.target_speed * 0.4
            maneuver = "Baseline Cautious Slowdown"
        else:
            target_v = self.target_speed
            maneuver = "Baseline Centerline Follow"

        # Plan strictly along centerline (d = 0.0)
        horizon = 3.0
        dt = 0.1
        time_steps = np.arange(0.0, horizon + dt, dt)

        waypoints: List[TrajectoryWaypoint] = []
        curr_s = ego_s
        curr_v = v0
        accel = (target_v - v0) / horizon

        for t in time_steps:
            curr_v = np.clip(curr_v + accel * dt, 0.0, self.target_speed)
            curr_s += curr_v * dt

            x_t, y_t, psi_road = road.frenet_to_cartesian(curr_s, 0.0) # Always d = 0.0
            waypoints.append(TrajectoryWaypoint(
                t=float(t),
                x=float(x_t),
                y=float(y_t),
                s=float(curr_s),
                d=0.0,
                v=float(curr_v),
                a=float(accel),
                psi=float(psi_road),
                kappa=0.0
            ))

        self.replanning_count += 1
        traj = CandidateTrajectory(
            id=1000,
            waypoints=waypoints,
            target_d=0.0,
            target_v=target_v,
            duration=horizon,
            maneuver_name=maneuver,
            total_cost=0.0,
            is_collision_free=True,
            is_within_boundaries=True
        )
        self.current_trajectory = traj
        return traj
