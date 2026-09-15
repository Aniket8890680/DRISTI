"""
Risk Assessment Module:
Calculates multi-factor collision hazard indicators:
- Time-to-Collision (TTC) based on closing speed and relative geometry
- Spatial Closest Point of Approach (CPA) using predicted trajectories
- Actor-specific uncertainty weighting (animals and pedestrians incur higher risk weights)
- Categorical risk states: LOW, MEDIUM, HIGH, CRITICAL
- Available escape space evaluation
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict
import numpy as np

from simulation.actors import ActorType
from simulation.vehicle import Vehicle
from tracking.tracker import TrackedObject
from prediction.trajectory_predictor import PredictedTrajectory


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class ObjectRisk:
    """Calculated risk metrics for an individual tracked object."""
    track_id: int
    actor_type: ActorType
    distance: float                 # Current Euclidean distance (m)
    closing_speed: float            # Relative closing speed towards ego (m/s)
    ttc: float                      # Time-to-Collision (seconds, inf if diverging)
    cpa_distance: float             # Closest Point of Approach distance (m)
    cpa_time: float                 # Time until CPA (s)
    risk_score: float               # Normalized composite risk score [0.0, 1.0]
    risk_level: RiskLevel           # Categorical level
    uncertainty_weight: float       # Actor uncertainty multiplier


@dataclass
class EnvironmentRiskSummary:
    """Overall environmental risk state evaluated for the ego vehicle."""
    highest_risk_level: RiskLevel
    max_risk_score: float
    min_ttc: float
    most_critical_object_id: Optional[int]
    object_risks: List[ObjectRisk]
    should_emergency_brake: bool
    recommended_max_speed: float    # m/s


class RiskAssessor:
    """
    Evaluates multi-faceted risk metrics without relying exclusively on scalar TTC.
    """

    def __init__(
        self,
        critical_ttc_threshold: float = 1.8, # seconds
        high_ttc_threshold: float = 3.0,     # seconds
        critical_dist_threshold: float = 6.0, # meters
        near_miss_distance: float = 1.0,     # meters
    ):
        self.critical_ttc = critical_ttc_threshold
        self.high_ttc = high_ttc_threshold
        self.critical_dist = critical_dist_threshold
        self.near_miss_dist = near_miss_distance

    def assess_risk(
        self,
        ego: Vehicle,
        tracked_objects: List[TrackedObject],
        predicted_trajectories: Dict[int, PredictedTrajectory],
        road_width: float = 7.0
    ) -> EnvironmentRiskSummary:
        """
        Computes composite risk across all active obstacles.
        """
        ego_x, ego_y = ego.center_position
        ego_psi = ego.state.psi
        ego_vx = ego.state.v * np.cos(ego_psi)
        ego_vy = ego.state.v * np.sin(ego_psi)
        ego_speed = ego.state.v

        obj_risks: List[ObjectRisk] = []
        max_score = 0.0
        min_ttc = float('inf')
        min_dist = float('inf')
        min_in_corridor_dist = float('inf')
        critical_obj_id = None

        for obj in tracked_objects:
            dx = obj.x - ego_x
            dy = obj.y - ego_y
            dist = float(np.hypot(dx, dy))

            # Relative velocity vector
            rel_vx = obj.vx - ego_vx
            rel_vy = obj.vy - ego_vy

            # Closing speed: component of relative velocity directed towards ego
            if dist > 1e-3:
                closing_speed = -(dx * rel_vx + dy * rel_vy) / dist
            else:
                closing_speed = 0.0

            # Time to collision calculation
            if closing_speed > 0.3:
                ttc = dist / closing_speed
            else:
                ttc = float('inf')

            # Trajectory Closest Point of Approach (CPA) calculation
            cpa_dist = dist
            cpa_time = 0.0
            if obj.track_id in predicted_trajectories:
                pred = predicted_trajectories[obj.track_id]
                for pt in pred.points:
                    # Extrapolate ego forward along current heading
                    ego_pred_x = ego_x + ego_vx * pt.t
                    ego_pred_y = ego_y + ego_vy * pt.t
                    d_pred = float(np.hypot(pt.x - ego_pred_x, pt.y - ego_pred_y))
                    if d_pred < cpa_dist:
                        cpa_dist = d_pred
                        cpa_time = pt.t

            # Actor uncertainty multiplier: animals, pedestrians, and motorcycles carry higher hazard
            unc_weight = 1.0 + 0.4 * obj.uncertainty

            # Proximity risk factor: e.g. < 5m is very high
            r_prox = np.exp(-max(0.0, dist - 2.0) / 6.0)

            # TTC risk factor: e.g. < 2.0s is extremely urgent
            if ttc < 10.0:
                r_ttc = np.exp(-ttc / 2.5)
            else:
                r_ttc = 0.0

            # CPA risk factor: predicted near miss along trajectory
            r_cpa = np.exp(-max(0.0, cpa_dist - 1.0) / 4.0)

            # Combined risk score [0, 1]
            base_score = max(r_prox * 0.7 + r_ttc * 0.8, r_cpa)
            score = float(np.clip(base_score * unc_weight, 0.0, 1.0))

            # Categorical classification
            if score >= 0.75 or (ttc < self.critical_ttc and dist < 20.0) or dist < self.critical_dist:
                level = RiskLevel.CRITICAL
            elif score >= 0.45 or ttc < self.high_ttc:
                level = RiskLevel.HIGH
            elif score >= 0.20 or dist < 25.0:
                level = RiskLevel.MEDIUM
            else:
                level = RiskLevel.LOW

            if score > max_score:
                max_score = score
                critical_obj_id = obj.track_id

            if ttc < min_ttc:
                min_ttc = ttc

            if dist < min_dist:
                min_dist = dist

            # Longitudinal and lateral distance in ego vehicle frame
            cos_psi = np.cos(ego_psi)
            sin_psi = np.sin(ego_psi)
            long_dist = dx * cos_psi + dy * sin_psi
            lat_dist = -dx * sin_psi + dy * cos_psi

            # Precise lateral half-width based on obstacle orientation
            cos_h = abs(np.cos(obj.heading))
            sin_h = abs(np.sin(obj.heading))
            obj_lat_hw = cos_h * (obj.width / 2.0) + sin_h * (obj.length / 2.0)

            corridor_hw = (ego.limits.width / 2.0) + obj_lat_hw + 0.35
            if long_dist > -0.5 and abs(lat_dist) < corridor_hw:
                if long_dist < min_in_corridor_dist:
                    min_in_corridor_dist = long_dist

            obj_risks.append(ObjectRisk(
                track_id=obj.track_id,
                actor_type=obj.actor_type,
                distance=dist,
                closing_speed=float(closing_speed),
                ttc=float(ttc),
                cpa_distance=float(cpa_dist),
                cpa_time=float(cpa_time),
                risk_score=score,
                risk_level=level,
                uncertainty_weight=unc_weight
            ))

        # Determine overall environmental risk level
        # Emergency braking is only mandated when time/distance is too short to evade an obstacle in our path
        emergency_brake = (min_ttc < self.critical_ttc and min_in_corridor_dist < 15.0) or (min_in_corridor_dist < self.critical_dist)

        if max_score >= 0.75 or min_ttc < self.critical_ttc:
            highest_level = RiskLevel.CRITICAL
            rec_speed = 0.0 if emergency_brake else max(6.0, ego_speed * 0.7)
        elif max_score >= 0.45 or min_ttc < self.high_ttc:
            highest_level = RiskLevel.HIGH
            rec_speed = max(7.0, ego_speed * 0.8)
        elif max_score >= 0.20:
            highest_level = RiskLevel.MEDIUM
            rec_speed = max(9.0, ego_speed * 0.9)
        else:
            highest_level = RiskLevel.LOW
            rec_speed = 15.0

        return EnvironmentRiskSummary(
            highest_risk_level=highest_level,
            max_risk_score=max_score,
            min_ttc=min_ttc,
            most_critical_object_id=critical_obj_id,
            object_risks=obj_risks,
            should_emergency_brake=emergency_brake,
            recommended_max_speed=rec_speed
        )
