"""
Vehicle Model: Kinematic Bicycle Model for Autonomous Driving Simulation.

Simulates the ego vehicle dynamics without teleportation:
- State: [x, y, velocity, acceleration, heading (psi), steering_angle (delta)]
- Actuation inputs: target acceleration and target steering angle
- Physical limits: max velocity, max acceleration, max braking, max steering angle, max steering rate
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class VehicleLimits:
    """Physical dynamic and kinematic constraints for the vehicle."""
    max_speed: float = 25.0         # m/s (~90 km/h)
    min_speed: float = -3.0         # m/s (reverse limit)
    max_accel: float = 3.0          # m/s^2 (comfortable acceleration)
    max_decel: float = 6.0          # m/s^2 (emergency deceleration limit: 8.0)
    max_steer: float = np.deg2rad(36.0)     # rad (~0.628 rad)
    max_steer_rate: float = np.deg2rad(45.0)  # rad/s (~0.785 rad/s)
    wheelbase: float = 2.7          # m (L: distance between front & rear axles)
    width: float = 1.85             # m (total vehicle width)
    length: float = 4.6             # m (total vehicle length)
    rear_overhang: float = 0.95     # m (distance from rear axle to rear bumper)


@dataclass
class VehicleState:
    """Current state representation of the vehicle."""
    x: float = 0.0                  # m (rear-axle center or center of mass)
    y: float = 0.0                  # m
    v: float = 0.0                  # m/s (longitudinal speed)
    a: float = 0.0                  # m/s^2 (current acceleration)
    psi: float = 0.0                # rad (yaw / heading angle, 0 = East / +X)
    delta: float = 0.0              # rad (steering angle of front wheel)


class Vehicle:
    """
    Ego vehicle model using a continuous-time Kinematic Bicycle Model
    integrated with forward Euler / RK4 integration.
    """

    def __init__(self, initial_state: Optional[VehicleState] = None, limits: Optional[VehicleLimits] = None):
        self.state = initial_state if initial_state is not None else VehicleState()
        self.limits = limits if limits is not None else VehicleLimits()
        
        # History for metrics and path visualization
        self.trajectory_history: List[Tuple[float, float, float, float]] = []  # (t, x, y, v)
        self.acceleration_history: List[float] = []
        self.steering_history: List[float] = []
        self.jerk_history: List[float] = []
        self.time: float = 0.0
        self._record_history()

    def _record_history(self) -> None:
        self.trajectory_history.append((self.time, self.state.x, self.state.y, self.state.v))
        self.acceleration_history.append(self.state.a)
        self.steering_history.append(self.state.delta)

    def step(self, target_accel: float, target_steer: float, dt: float) -> None:
        """
        Advances the kinematic bicycle model by dt seconds.
        
        Kinematic equations of motion (referenced to rear-axle center):
            dx/dt = v * cos(psi)
            dy/dt = v * sin(psi)
            dpsi/dt = (v / L) * tan(delta)
            dv/dt = a
            ddelta/dt = clamp((target_delta - delta)/dt_filter, -max_rate, max_rate)
        """
        if dt <= 0.0:
            return

        # 1. Apply acceleration limits
        target_accel = np.clip(target_accel, -self.limits.max_decel, self.limits.max_accel)
        
        # Acceleration rate smoothing (jerk constraint)
        prev_a = self.state.a
        self.state.a = float(target_accel)
        jerk = (self.state.a - prev_a) / dt
        self.jerk_history.append(jerk)

        # 2. Update steering angle with rate limit
        target_steer = np.clip(target_steer, -self.limits.max_steer, self.limits.max_steer)
        steer_diff = target_steer - self.state.delta
        max_delta_change = self.limits.max_steer_rate * dt
        delta_change = np.clip(steer_diff, -max_delta_change, max_delta_change)
        self.state.delta += delta_change

        # 3. Kinematic bicycle state integration
        x = self.state.x
        y = self.state.y
        v = self.state.v
        psi = self.state.psi
        delta = self.state.delta
        L = self.limits.wheelbase

        # State derivatives
        dx = v * np.cos(psi)
        dy = v * np.sin(psi)
        dpsi = (v / L) * np.tan(delta)
        dv = self.state.a

        # Integration
        self.state.x = float(x + dx * dt)
        self.state.y = float(y + dy * dt)
        self.state.psi = float(normalize_angle(psi + dpsi * dt))
        self.state.v = float(np.clip(v + dv * dt, self.limits.min_speed, self.limits.max_speed))

        # Time update
        self.time += dt
        self._record_history()

    @property
    def center_position(self) -> Tuple[float, float]:
        """Calculates the physical geometric center of the vehicle."""
        # Rear axle to geometric center offset
        center_offset = (self.limits.length / 2.0) - self.limits.rear_overhang
        cx = self.state.x + center_offset * np.cos(self.state.psi)
        cy = self.state.y + center_offset * np.sin(self.state.psi)
        return float(cx), float(cy)

    def get_bounding_box(self) -> np.ndarray:
        """
        Returns the 4 corner points of the oriented bounding box in 2D world coordinates.
        Points returned in order: [Front-Left, Front-Right, Rear-Right, Rear-Left] (shape: 4x2)
        """
        cx, cy = self.center_position
        half_l = self.limits.length / 2.0
        half_w = self.limits.width / 2.0
        
        # Local corners relative to center
        local_corners = np.array([
            [half_l, half_w],    # Front-Left
            [half_l, -half_w],   # Front-Right
            [-half_l, -half_w],  # Rear-Right
            [-half_l, half_w]    # Rear-Left
        ])

        # Rotation matrix for vehicle heading psi
        cos_p = np.cos(self.state.psi)
        sin_p = np.sin(self.state.psi)
        R = np.array([
            [cos_p, -sin_p],
            [sin_p, cos_p]
        ])

        world_corners = (R @ local_corners.T).T + np.array([cx, cy])
        return world_corners


def normalize_angle(angle: float) -> float:
    """Normalizes an angle to the range [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi
