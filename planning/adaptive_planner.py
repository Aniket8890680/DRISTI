"""
Adaptive Path Planner Module:
The core decision-making and trajectory optimization engine for unstructured Indian roads.

Features:
- Multi-objective cost evaluation across collision risk, boundaries, smoothness, comfort, progress, and actor uncertainty
- Continuous replanning upon environmental change or obstacle trajectory deviation
- Controlled emergency braking fallback when no safe lateral evasion exists
- State machine tracking:
  NORMAL_DRIVING -> CAUTION -> OBSTACLE_DETECTED -> PREDICTING -> REPLANNING -> EMERGENCY_BRAKING -> STOPPED
- Measurement of replanning latency and event counters for metrics evaluation
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict
import numpy as np

from simulation.road import Road
from simulation.vehicle import Vehicle, normalize_angle
from simulation.actors import ActorType
from tracking.tracker import TrackedObject
from prediction.trajectory_predictor import PredictedTrajectory
from planning.drivable_area import DrivableAreaEstimator
from planning.candidate_paths import CandidateGenerator, CandidateTrajectory, TrajectoryWaypoint
from planning.risk import RiskAssessor, EnvironmentRiskSummary, RiskLevel


class PlannerState(Enum):
    NORMAL_DRIVING = "NORMAL_DRIVING"
    CAUTION = "CAUTION"
    OBSTACLE_DETECTED = "OBSTACLE_DETECTED"
    PREDICTING = "PREDICTING"
    REPLANNING = "REPLANNING"
    EMERGENCY_BRAKING = "EMERGENCY_BRAKING"
    STOPPED = "STOPPED"


@dataclass
class PlannerConfig:
    """Weight tuning parameters for adaptive trajectory selection."""
    w_collision: float = 10000.0     # Critical penalty for potential collision
    w_boundary: float = 5000.0       # Penalty for exceeding road boundaries
    w_pothole: float = 300.0         # Penalty for running over potholes
    w_uncertainty: float = 150.0     # Penalty for passing close to erratic actors (animals/pedestrians)
    w_smoothness: float = 15.0       # Penalty for excessive curvature/lateral jerk
    w_speed: float = 2.0             # Penalty for deviating from target cruise speed
    w_progress: float = 25.0         # Reward for forward longitudinal progress (-progress)
    w_centerline: float = 1.0        # Minor bias towards road center when open
    target_cruise_speed: float = 12.0 # m/s (~43 km/h)
    min_clearance_hard_limit: float = 0.2 # meters


class AdaptivePlanner:
    """
    Closed-loop Adaptive Path Planner for Autonomous Navigation on Unstructured Roads.
    """

    def __init__(
        self,
        config: Optional[PlannerConfig] = None,
        candidate_generator: Optional[CandidateGenerator] = None,
        drivable_area_estimator: Optional[DrivableAreaEstimator] = None,
        risk_assessor: Optional[RiskAssessor] = None,
    ):
        self.config = config if config is not None else PlannerConfig()
        self.generator = candidate_generator if candidate_generator is not None else CandidateGenerator()
        self.drivable_area = drivable_area_estimator if drivable_area_estimator is not None else DrivableAreaEstimator()
        self.risk_assessor = risk_assessor if risk_assessor is not None else RiskAssessor()

        self.current_state = PlannerState.NORMAL_DRIVING
        self.current_trajectory: Optional[CandidateTrajectory] = None
        self.replanning_count: int = 0
        self.replanning_latencies_ms: List[float] = []

    def plan(
        self,
        ego: Vehicle,
        road: Road,
        tracked_objects: List[TrackedObject],
        predicted_trajectories: Dict[int, PredictedTrajectory],
        risk_summary: EnvironmentRiskSummary,
    ) -> CandidateTrajectory:
        """
        Executes a closed-loop planning cycle and returns the safest feasible trajectory.
        """
        start_time = time.perf_counter()

        # Update planner state based on risk
        if ego.state.v < 0.1 and risk_summary.highest_risk_level == RiskLevel.CRITICAL:
            self.current_state = PlannerState.STOPPED
        elif risk_summary.should_emergency_brake:
            self.current_state = PlannerState.EMERGENCY_BRAKING
        elif len(tracked_objects) > 0:
            if risk_summary.highest_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                self.current_state = PlannerState.OBSTACLE_DETECTED
            else:
                self.current_state = PlannerState.CAUTION
        else:
            self.current_state = PlannerState.NORMAL_DRIVING

        # Handle Immediate Emergency Braking if demanded by severe critical risk
        if self.current_state == PlannerState.EMERGENCY_BRAKING:
            emergency_traj = self._generate_emergency_braking_trajectory(ego, road)
            self.current_trajectory = emergency_traj
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.replanning_latencies_ms.append(elapsed_ms)
            return emergency_traj

        # 1. Generate candidate trajectories
        target_speed = min(self.config.target_cruise_speed, risk_summary.recommended_max_speed)
        candidates = self.generator.generate_candidates(ego, road, target_cruise_speed=target_speed)

        # 2. Evaluate cost for each candidate trajectory
        best_traj = None
        min_cost = float('inf')

        for candidate in candidates:
            cost, is_safe, within_bounds = self._evaluate_candidate_cost(
                candidate, ego, road, tracked_objects, predicted_trajectories, target_speed
            )
            candidate.total_cost = cost
            candidate.is_collision_free = is_safe
            candidate.is_within_boundaries = within_bounds

            if is_safe and within_bounds and cost < min_cost:
                min_cost = cost
                best_traj = candidate

        # 3. If no safe candidate was found (e.g. road blocked or evasions exceed road limits),
        # fallback safely to emergency controlled braking rather than swerving into a ditch or obstacle
        if best_traj is None:
            self.current_state = PlannerState.EMERGENCY_BRAKING
            best_traj = self._generate_emergency_braking_trajectory(ego, road)

        # Check if replanning occurred
        if self.current_trajectory is None or (best_traj.id != self.current_trajectory.id):
            self.replanning_count += 1
            if self.current_state != PlannerState.EMERGENCY_BRAKING:
                self.current_state = PlannerState.REPLANNING

        self.current_trajectory = best_traj
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.replanning_latencies_ms.append(elapsed_ms)

        return best_traj

    def _evaluate_candidate_cost(
        self,
        candidate: CandidateTrajectory,
        ego: Vehicle,
        road: Road,
        tracked_objects: List[TrackedObject],
        predicted_trajectories: Dict[int, PredictedTrajectory],
        target_speed: float,
    ) -> Tuple[float, bool, bool]:
        """
        Evaluates the multi-objective cost for a single candidate trajectory.
        Returns: (cost, is_collision_free, is_within_boundaries)
        """
        cost = 0.0
        is_collision_free = True
        is_within_bounds = True

        ego_half_w = ego.limits.width / 2.0
        ego_half_l = ego.limits.length / 2.0

        # Cost components
        col_cost = 0.0
        bound_cost = 0.0
        pothole_cost = 0.0
        unc_cost = 0.0
        smooth_cost = 0.0
        speed_cost = 0.0

        # Precompute obstacle Frenet states for this candidate evaluation
        # to avoid repeated coordinate transforms
        road_potholes = road.potholes

        for wp in candidate.waypoints:
            # 1. Fast Road boundary check in Frenet frame (O(1))
            half_w = road.get_width(wp.s) / 2.0
            # Allow realistic shoulder tolerance (0.35m) for narrow unstructured roads
            max_d_allowed = half_w + 0.35 - ego_half_w
            if abs(wp.d) > max_d_allowed:
                bound_cost += (abs(wp.d) - max_d_allowed) * self.config.w_boundary
                is_within_bounds = False
            else:
                edge_clearance = (half_w - ego_half_w) - abs(wp.d)
                if edge_clearance < 0.6:
                    bound_cost += np.exp(-max(0.0, edge_clearance) / 0.4) * 40.0

            # 2. Pothole check
            for pothole in road_potholes:
                d_pothole = np.hypot(wp.x - pothole.x, wp.y - pothole.y)
                if d_pothole < (pothole.radius + ego_half_w):
                    pothole_cost += self.config.w_pothole * pothole.severity

            # 3. Dynamic obstacle and predicted trajectory collision check
            for obj in tracked_objects:
                # Find obstacle position at time wp.t
                pred_traj = predicted_trajectories.get(obj.track_id)
                if pred_traj and len(pred_traj.points) > 0:
                    idx = min(int(round(wp.t / pred_traj.points[0].t)), len(pred_traj.points) - 1)
                    idx = max(0, idx)
                    pt_pred = pred_traj.points[idx]
                    # Pre-cached Frenet coords if available
                    if not hasattr(pt_pred, '_s_cached'):
                        pt_pred._s_cached, pt_pred._d_cached = road.cartesian_to_frenet(pt_pred.x, pt_pred.y)
                    s_obj, d_obj = pt_pred._s_cached, pt_pred._d_cached
                    unc_radius = pt_pred.uncertainty_radius
                else:
                    if not hasattr(obj, '_s_cached'):
                        obj._s_cached, obj._d_cached = road.cartesian_to_frenet(obj.x, obj.y)
                    s_obj = obj._s_cached + (obj.vx * np.cos(obj.heading) + obj.vy * np.sin(obj.heading)) * wp.t
                    d_obj = obj._d_cached
                    unc_radius = obj.uncertainty * 1.5

                delta_s = abs(wp.s - s_obj)
                delta_d = abs(wp.d - d_obj)

                # Physical geometry bounding box limits
                physical_lon = ego_half_l + (obj.length / 2.0) + self.config.min_clearance_hard_limit
                physical_lat = ego_half_w + (obj.width / 2.0) + self.config.min_clearance_hard_limit

                # Collision check: collision occurs only when BOTH longitudinal AND lateral bounds physically overlap
                if delta_s < physical_lon and delta_d < physical_lat:
                    is_collision_free = False
                    col_cost += self.config.w_collision
                elif delta_s < (physical_lon + 3.0) and delta_d < (physical_lat + 1.5):
                    proximity_penalty = ((physical_lon + 3.0 - delta_s) / 3.0)**2 + ((physical_lat + 1.5 - delta_d) / 1.5)**2
                    col_cost += 100.0 * proximity_penalty

                # Actor uncertainty penalty (higher for cattle, pedestrians, filtering motorcycles)
                euclid_dist = np.hypot(delta_s, delta_d)
                if euclid_dist < (physical_lon + unc_radius + 2.0):
                    unc_cost += self.config.w_uncertainty * obj.uncertainty / max(euclid_dist, 0.5)

                # Oncoming traffic corridor protection
                if obj.vx < -1.0 and (s_obj > wp.s) and (s_obj - wp.s) < 55.0 and abs(wp.d - d_obj) < 1.8:
                    oncoming_factor = 1.0 - (s_obj - wp.s) / 55.0
                    col_cost += self.config.w_collision * 0.4 * oncoming_factor

            # Smoothness and curvature penalty
            smooth_cost += (wp.kappa ** 2) * self.config.w_smoothness

            # Speed tracking penalty
            speed_cost += ((wp.v - target_speed) ** 2) * self.config.w_speed

        # Longitudinal progress reward
        final_s = candidate.waypoints[-1].s
        start_s = candidate.waypoints[0].s
        progress = max(0.0, final_s - start_s)
        progress_reward = -progress * self.config.w_progress

        # Centerline deviation penalty (minor bias)
        centerline_cost = abs(candidate.target_d) * self.config.w_centerline

        total_cost = (
            col_cost + bound_cost + pothole_cost + unc_cost +
            smooth_cost + speed_cost + progress_reward + centerline_cost
        )

        return float(total_cost), is_collision_free, is_within_bounds

    def _generate_emergency_braking_trajectory(self, ego: Vehicle, road: Road) -> CandidateTrajectory:
        """
        Generates an emergency stopping trajectory that decelerates maximally in-lane
        without erratic lateral maneuvers.
        """
        ego_s, ego_d = road.cartesian_to_frenet(ego.state.x, ego.state.y)
        v0 = max(0.0, ego.state.v)
        max_decel = ego.limits.max_decel

        # Stopping distance and time: v^2 = 2*a*d -> d = v^2 / (2*a), t = v / a
        t_stop = max(0.5, v0 / max_decel)
        dt = self.generator.dt_sample
        time_steps = np.arange(0.0, max(2.0, t_stop + 0.5) + dt, dt)

        waypoints: List[TrajectoryWaypoint] = []
        curr_s = ego_s
        curr_v = v0

        for t in time_steps:
            if curr_v > 0.0:
                curr_v = max(0.0, v0 - max_decel * t)
                curr_s += curr_v * dt

            x_t, y_t, psi_road = road.frenet_to_cartesian(curr_s, ego_d)
            waypoints.append(TrajectoryWaypoint(
                t=float(t),
                x=float(x_t),
                y=float(y_t),
                s=float(curr_s),
                d=float(ego_d),
                v=float(curr_v),
                a=-max_decel if curr_v > 0.0 else 0.0,
                psi=float(psi_road),
                kappa=0.0
            ))

        return CandidateTrajectory(
            id=9999,
            waypoints=waypoints,
            target_d=ego_d,
            target_v=0.0,
            duration=float(time_steps[-1]),
            maneuver_name="EMERGENCY CONTROLLED BRAKING",
            total_cost=0.0,
            is_collision_free=True,
            is_within_boundaries=True
        )
