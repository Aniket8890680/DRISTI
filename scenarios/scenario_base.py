"""
Base Scenario Module:
Defines the standard interface, closed-loop execution loop, and telemetry collection
for autonomous driving scenarios.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from simulation.road import Road
from simulation.vehicle import Vehicle, VehicleState
from simulation.actors import Actor
from simulation.environment import Environment, WeatherCondition
from sensors.camera import CameraSensor
from sensors.radar import RadarSensor
from sensors.lidar import LidarSensor
from sensors.gps_imu import GpsImuSensor
from perception.sensor_fusion import SensorFusion
from tracking.tracker import MultiObjectTracker, TrackedObject
from prediction.trajectory_predictor import TrajectoryPredictor, PredictedTrajectory
from planning.risk import RiskAssessor, EnvironmentRiskSummary
from planning.adaptive_planner import AdaptivePlanner, PlannerState
from planning.baseline_planner import BaselinePlanner
from planning.candidate_paths import CandidateTrajectory
from control.vehicle_controller import VehicleController, ControlCommand


@dataclass
class TelemetrySnapshot:
    """Recorded state of the entire system at one simulation timestep."""
    time: float
    ego_x: float
    ego_y: float
    ego_v: float
    ego_a: float
    ego_psi: float
    ego_delta: float
    command_accel: float
    command_steer: float
    planner_state: str
    risk_level: str
    min_ttc: float
    min_clearance: float
    has_collision: bool
    is_off_road: bool
    active_trajectory: Optional[CandidateTrajectory]
    candidate_trajectories: List[CandidateTrajectory]
    tracked_objects: List[TrackedObject]
    predicted_trajectories: Dict[int, PredictedTrajectory]


class ScenarioBase(ABC):
    """
    Abstract base class providing the standardized closed-loop simulation pipeline.
    """

    def __init__(
        self,
        name: str,
        description: str,
        weather: WeatherCondition = WeatherCondition.NORMAL,
        use_baseline_planner: bool = False,
        max_duration: float = 25.0,
        target_distance: float = 80.0,
        random_seed: Optional[int] = 42,
    ):
        self.name = name
        self.description = description
        self.weather = weather
        self.use_baseline = use_baseline_planner
        self.max_duration = max_duration
        self.target_distance = target_distance
        self.random_seed = random_seed

        # Pipeline components
        self.road: Optional[Road] = None
        self.ego: Optional[Vehicle] = None
        self.env: Optional[Environment] = None
        
        self.camera = CameraSensor(random_seed=random_seed)
        self.radar = RadarSensor(random_seed=random_seed)
        self.lidar = LidarSensor(random_seed=random_seed)
        self.gps_imu = GpsImuSensor(random_seed=random_seed)
        
        self.fusion = SensorFusion()
        self.tracker = MultiObjectTracker()
        self.predictor = TrajectoryPredictor()
        self.risk_assessor = RiskAssessor()
        
        if self.use_baseline:
            self.planner = BaselinePlanner()
        else:
            self.planner = AdaptivePlanner()

        self.controller = VehicleController()

        # Telemetry & Status
        self.telemetry: List[TelemetrySnapshot] = []
        self.is_completed: bool = False
        self.is_success: bool = False
        self.termination_reason: str = "In Progress"

    @abstractmethod
    def build_environment(self) -> None:
        """Constructs the specific road geometry, ego starting pose, and traffic actors."""
        pass

    def reset(self) -> None:
        """Resets the scenario to its initial conditions."""
        self.tracker = MultiObjectTracker()
        self.active_trajectory = None
        self.telemetry.clear()
        self.is_completed = False
        self.is_success = False
        self.termination_reason = "In Progress"
        self.build_environment()

    def step(self) -> bool:
        """
        Executes one closed-loop cycle of the autonomous driving pipeline:
        Sensors -> Perception/Fusion -> Tracking -> Prediction -> Risk -> Planning -> Control -> Actuation -> Environment
        Returns: True if scenario has finished, False otherwise.
        """
        if self.is_completed:
            return True

        current_time = self.env.time

        # 1. Simulated Sensors
        cam_dets = self.camera.sense(self.env.ego, self.env.actors, self.weather)
        rad_tgts = self.radar.sense(self.env.ego, self.env.actors, self.weather)
        lid_cl = self.lidar.sense(self.env.ego, self.env.actors, self.weather)
        ego_pose = self.gps_imu.sense(self.env.ego, current_time)

        # 2. Perception & Multi-Sensor Fusion
        fused_objects = self.fusion.fuse(cam_dets, rad_tgts, lid_cl, ego_pose)

        # 3. Object Tracking across frames
        tracked_objects = self.tracker.update(fused_objects, self.env.dt)

        # 4. Short-Term Motion Prediction
        predicted_trajectories_list = self.predictor.predict(tracked_objects, current_time)
        pred_dict = {p.track_id: p for p in predicted_trajectories_list}

        # 5. Risk Assessment
        risk_summary = self.risk_assessor.assess_risk(
            self.env.ego, tracked_objects, pred_dict, road_width=self.road.nominal_width
        )

        # 6. Adaptive Path Planning (or Baseline)
        should_replan = (
            getattr(self, 'active_trajectory', None) is None or
            (self.env.step_count % 2 == 0) or
            risk_summary.highest_risk_level.value in ["HIGH", "CRITICAL"]
        )

        if should_replan:
            if self.use_baseline:
                self.active_trajectory = self.planner.plan(self.env.ego, self.road, tracked_objects)
                candidate_trajs = [self.active_trajectory]
                planner_state = "BASELINE_CRUISE"
            else:
                self.active_trajectory = self.planner.plan(
                    self.env.ego, self.road, tracked_objects, pred_dict, risk_summary
                )
                candidate_trajs = getattr(self.planner, 'current_candidates', [self.active_trajectory])
                planner_state = self.planner.current_state.value
        else:
            candidate_trajs = [self.active_trajectory]
            planner_state = "BASELINE_CRUISE" if self.use_baseline else self.planner.current_state.value

        active_trajectory = self.active_trajectory

        # 7. Vehicle Control
        cmd = self.controller.compute_control(self.env.ego, active_trajectory, self.env.dt)

        # 8. Physical Environment Step
        collided, min_clearance = self.env.step(cmd.target_accel, cmd.target_steer)

        # Check off-road (with realistic shoulder tolerance for narrow Indian roads)
        ego_box = self.env.ego.get_bounding_box()
        is_off_road = not self.road.is_box_in_road(ego_box, margin=-0.4)

        # 9. Record Telemetry
        self.telemetry.append(TelemetrySnapshot(
            time=current_time,
            ego_x=self.env.ego.state.x,
            ego_y=self.env.ego.state.y,
            ego_v=self.env.ego.state.v,
            ego_a=self.env.ego.state.a,
            ego_psi=self.env.ego.state.psi,
            ego_delta=self.env.ego.state.delta,
            command_accel=cmd.target_accel,
            command_steer=cmd.target_steer,
            planner_state=planner_state,
            risk_level=risk_summary.highest_risk_level.value,
            min_ttc=risk_summary.min_ttc,
            min_clearance=min_clearance,
            has_collision=collided,
            is_off_road=is_off_road,
            active_trajectory=active_trajectory,
            candidate_trajectories=candidate_trajs,
            tracked_objects=tracked_objects,
            predicted_trajectories=pred_dict
        ))

        # 10. Check Termination Conditions
        ego_s, _ = self.road.cartesian_to_frenet(self.env.ego.state.x, self.env.ego.state.y)
        
        if collided:
            self.is_completed = True
            self.is_success = False
            self.termination_reason = "COLLISION"
        elif is_off_road:
            self.is_completed = True
            self.is_success = False
            self.termination_reason = "OFF_ROAD"
        elif ego_s >= self.target_distance:
            self.is_completed = True
            self.is_success = True
            self.termination_reason = "SUCCESS_REACHED_GOAL"
        elif current_time >= self.max_duration:
            self.is_completed = True
            # If vehicle stopped safely before collision, marked as completed
            if ego_s > self.target_distance * 0.7:
                self.is_success = True
                self.termination_reason = "SUCCESS_PROGRESS"
            else:
                self.is_success = False
                self.termination_reason = "TIMEOUT"

        return self.is_completed

    def run(self) -> bool:
        """Executes the entire scenario to completion."""
        self.reset()
        while not self.step():
            pass
        return self.is_success
