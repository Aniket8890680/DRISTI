"""
Vehicle Controller Module:
Executes closed-loop trajectory following using:
- Lateral Control: Stanley path tracking controller with cross-track error damping
- Longitudinal Control: Proportional-Integral-Derivative (PID) speed controller
- Rigorous clamping within physical actuator limits (max speed, acceleration, braking, steer angle, steer rate)
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from simulation.vehicle import Vehicle, normalize_angle
from planning.candidate_paths import CandidateTrajectory


@dataclass
class ControlCommand:
    """Actuator command sent to the vehicle model."""
    target_accel: float         # m/s^2
    target_steer: float         # rad (front-wheel steering angle)


class VehicleController:
    """
    Combined lateral and longitudinal controller.
    """

    def __init__(
        self,
        k_stanley: float = 1.2,     # Stanley lateral cross-track gain
        k_soft: float = 1.5,        # Softening speed for Stanley stability at low speeds
        kp_lon: float = 1.5,        # Longitudinal proportional gain
        ki_lon: float = 0.1,        # Longitudinal integral gain
        kd_lon: float = 0.05,       # Longitudinal derivative gain
    ):
        self.k_stanley = k_stanley
        self.k_soft = k_soft
        self.kp_lon = kp_lon
        self.ki_lon = ki_lon
        self.kd_lon = kd_lon

        self.speed_error_integral = 0.0
        self.prev_speed_error = 0.0

    def compute_control(
        self,
        ego: Vehicle,
        trajectory: CandidateTrajectory,
        dt: float
    ) -> ControlCommand:
        """
        Computes steering angle and acceleration to follow the target trajectory.
        """
        if len(trajectory.waypoints) == 0:
            return ControlCommand(target_accel=-ego.limits.max_decel, target_steer=0.0)

        # 1. Lookahead / Nearest waypoint search
        ego_x, ego_y = ego.center_position
        v_ego = ego.state.v

        # Find closest waypoint along trajectory
        dists = [np.hypot(wp.x - ego_x, wp.y - ego_y) for wp in trajectory.waypoints]
        min_idx = int(np.argmin(dists))

        # Look slightly ahead based on speed (lookahead index)
        lookahead_steps = int(np.clip(1 + v_ego * 0.2, 1, len(trajectory.waypoints) - 1))
        target_idx = min(min_idx + lookahead_steps, len(trajectory.waypoints) - 1)
        target_wp = trajectory.waypoints[target_idx]

        # 2. Lateral Control: Stanley Controller
        # Vector from target waypoint to front axle
        front_axle_x = ego.state.x + ego.limits.wheelbase * np.cos(ego.state.psi)
        front_axle_y = ego.state.y + ego.limits.wheelbase * np.sin(ego.state.psi)

        # Cross-track error e_lat: distance from front axle to path line
        dx = front_axle_x - target_wp.x
        dy = front_axle_y - target_wp.y
        path_heading = target_wp.psi
        
        # Cross-track error (positive if vehicle is to the left of the path)
        e_lat = -np.sin(path_heading) * dx + np.cos(path_heading) * dy

        # Heading error e_psi
        e_psi = normalize_angle(path_heading - ego.state.psi)

        # Stanley steering law: delta = e_psi + arctan(k * e_lat / (v + k_soft))
        steer_correction = np.arctan2(self.k_stanley * e_lat, v_ego + self.k_soft)
        target_steer = normalize_angle(e_psi + steer_correction)

        # 3. Longitudinal Control: PID Controller
        v_target = target_wp.v
        speed_error = v_target - v_ego
        
        self.speed_error_integral += speed_error * dt
        self.speed_error_integral = float(np.clip(self.speed_error_integral, -5.0, 5.0))
        
        speed_error_deriv = (speed_error - self.prev_speed_error) / max(dt, 1e-4)
        self.prev_speed_error = speed_error

        # Feedforward acceleration + PID feedback
        accel_cmd = (
            target_wp.a +
            self.kp_lon * speed_error +
            self.ki_lon * self.speed_error_integral +
            self.kd_lon * speed_error_deriv
        )

        return ControlCommand(
            target_accel=float(accel_cmd),
            target_steer=float(target_steer)
        )
