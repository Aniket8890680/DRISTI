"""
Motion Prediction Module:
Forecasts the short-term future trajectory of surrounding road participants
over a configurable time horizon (1.0 to 5.0 seconds).

Models heterogeneous Indian roadway behaviors:
- Non-lane-bound movements
- Motorcycle filtering and abrupt cutting across lanes
- Pedestrian crossing dynamics
- Animal wandering / sudden freeze with expanding uncertainty ellipses
- Informal merging and wrong-side traffic
- Extensible architecture to swap in ML / neural prediction models (e.g. Trajectron, TNT)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np

from simulation.actors import ActorType, ACTOR_PROPERTIES
from tracking.tracker import TrackedObject


@dataclass
class TrajectoryPoint:
    """A single predicted state waypoint along a future trajectory."""
    t: float          # future time offset (s)
    x: float          # predicted world X (m)
    y: float          # predicted world Y (m)
    vx: float         # predicted X velocity (m/s)
    vy: float         # predicted Y velocity (m/s)
    heading: float    # predicted heading (rad)
    uncertainty_radius: float  # expanding standard deviation envelope (m)


@dataclass
class PredictedTrajectory:
    """Multi-step predicted trajectory for a tracked actor."""
    track_id: int
    actor_type: ActorType
    points: List[TrajectoryPoint]
    nominal_speed: float
    intent_description: str


class TrajectoryPredictor:
    """
    Kinematic and behavioral multi-hypothesis trajectory predictor.
    """

    def __init__(
        self,
        horizon_seconds: float = 3.0,
        dt_pred: float = 0.1,
    ):
        self.horizon = horizon_seconds
        self.dt = dt_pred
        self.num_steps = int(np.ceil(self.horizon / self.dt))

    def predict(
        self,
        tracked_objects: List[TrackedObject],
        current_time: float = 0.0
    ) -> List[PredictedTrajectory]:
        """
        Generates predicted trajectories for all tracked road participants.
        """
        predictions: List[PredictedTrajectory] = []

        for obj in tracked_objects:
            pred_points: List[TrajectoryPoint] = []
            
            # Base kinematic state
            x = obj.x
            y = obj.y
            vx = obj.vx
            vy = obj.vy
            speed = obj.speed
            heading = obj.heading

            # Actor-specific uncertainty expansion rate (sigma_dot in m/s)
            base_uncertainty = obj.uncertainty
            # Animals and pedestrians expand uncertainty rapidly
            if obj.actor_type == ActorType.ANIMAL:
                uncertainty_expansion = 0.8
                intent = "animal_wander_freeze_risk"
            elif obj.actor_type == ActorType.PEDESTRIAN:
                uncertainty_expansion = 0.5
                intent = "pedestrian_crossing"
            elif obj.actor_type in [ActorType.MOTORCYCLE, ActorType.AUTO_RICKSHAW]:
                uncertainty_expansion = 0.4
                intent = "informal_filter_cut"
            elif obj.actor_type in [ActorType.CAR, ActorType.BUS, ActorType.TRUCK]:
                uncertainty_expansion = 0.2
                intent = "forward_corridor"
            else:
                # Pothole, construction, static
                uncertainty_expansion = 0.02
                intent = "stationary"

            # Rollout future trajectory
            for step in range(1, self.num_steps + 1):
                tau = step * self.dt
                
                # Behavioral motion models
                if obj.actor_type == ActorType.ANIMAL:
                    # Cattle model: may decelerate or freeze in center of road, or drift laterally
                    # Slight deceleration factor simulating pausing
                    decay = max(0.2, 1.0 - 0.25 * tau)
                    curr_vx = vx * decay
                    curr_vy = vy * decay
                    x += curr_vx * self.dt
                    y += curr_vy * self.dt

                elif obj.actor_type == ActorType.MOTORCYCLE:
                    # Motorcycle model: lateral agility, possible slight sinusoidal swerve
                    lateral_drift = 0.2 * np.sin(2.0 * tau)
                    norm_x = -np.sin(heading)
                    norm_y = np.cos(heading)
                    x += (vx * self.dt + norm_x * lateral_drift * self.dt)
                    y += (vy * self.dt + norm_y * lateral_drift * self.dt)

                elif obj.actor_type == ActorType.PEDESTRIAN:
                    # Pedestrian model: constant velocity crossing
                    x += vx * self.dt
                    y += vy * self.dt

                else:
                    # Constant Velocity extrapolation for standard vehicles & static objects
                    x += vx * self.dt
                    y += vy * self.dt

                # Expanding spatial uncertainty envelope
                sigma = base_uncertainty + uncertainty_expansion * tau

                pred_points.append(TrajectoryPoint(
                    t=tau,
                    x=float(x),
                    y=float(y),
                    vx=float(vx),
                    vy=float(vy),
                    heading=float(heading),
                    uncertainty_radius=float(sigma)
                ))

            predictions.append(PredictedTrajectory(
                track_id=obj.track_id,
                actor_type=obj.actor_type,
                points=pred_points,
                nominal_speed=speed,
                intent_description=intent
            ))

        return predictions
