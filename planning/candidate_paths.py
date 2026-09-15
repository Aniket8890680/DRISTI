"""
Candidate Trajectory Generation Module:
Generates a diverse bundle of kinematically feasible candidate trajectories
using quintic polynomial splines in Frenet coordinate space.

Covers essential adaptive maneuvers for Indian road conditions:
- Continue forward (center)
- Moderate / evasive lateral shifts (left / right)
- Speed maintenance, slowing down, crawling, stopping, and emergency braking
- Smooth boundary condition satisfaction (position, velocity, acceleration continuity)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

from simulation.road import Road
from simulation.vehicle import Vehicle, normalize_angle


@dataclass
class TrajectoryWaypoint:
    """Waypoint along a candidate trajectory."""
    t: float       # time offset (s)
    x: float       # world X (m)
    y: float       # world Y (m)
    s: float       # Frenet longitudinal distance (m)
    d: float       # Frenet lateral offset (m)
    v: float       # longitudinal speed (m/s)
    a: float       # longitudinal acceleration (m/s^2)
    psi: float     # heading angle (rad)
    kappa: float   # road curvature (1/m)


@dataclass
class CandidateTrajectory:
    """Full candidate trajectory with physical profile and cost metrics."""
    id: int
    waypoints: List[TrajectoryWaypoint]
    target_d: float
    target_v: float
    duration: float
    maneuver_name: str
    total_cost: float = float('inf')
    is_collision_free: bool = True
    is_within_boundaries: bool = True


class QuinticPolynomial:
    """
    1D Quintic Polynomial:
    x(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    Satisfies boundary conditions:
    at t=0: x0, v0, a0
    at t=T: xf, vf, af
    """

    def __init__(self, x0: float, v0: float, a0: float, xf: float, vf: float, af: float, T: float):
        self.a0 = x0
        self.a1 = v0
        self.a2 = 0.5 * a0

        A = np.array([
            [T**3,       T**4,        T**5],
            [3.0 * T**2, 4.0 * T**3,  5.0 * T**4],
            [6.0 * T,    12.0 * T**2, 20.0 * T**3]
        ])

        b = np.array([
            xf - self.a0 - self.a1 * T - self.a2 * T**2,
            vf - self.a1 - 2.0 * self.a2 * T,
            af - 2.0 * self.a2
        ])

        try:
            x = np.linalg.solve(A, b)
            self.a3 = x[0]
            self.a4 = x[1]
            self.a5 = x[2]
        except np.linalg.LinAlgError:
            self.a3 = self.a4 = self.a5 = 0.0

    def calc_point(self, t: float) -> float:
        return self.a0 + self.a1 * t + self.a2 * t**2 + self.a3 * t**3 + self.a4 * t**4 + self.a5 * t**5

    def calc_first_derivative(self, t: float) -> float:
        return self.a1 + 2.0 * self.a2 * t + 3.0 * self.a3 * t**2 + 4.0 * self.a4 * t**3 + 5.0 * self.a5 * t**4

    def calc_second_derivative(self, t: float) -> float:
        return 2.0 * self.a2 + 6.0 * self.a3 * t + 12.0 * self.a4 * t**2 + 20.0 * self.a5 * t**3

    def calc_third_derivative(self, t: float) -> float:
        return 6.0 * self.a3 + 24.0 * self.a4 * t + 60.0 * self.a5 * t**2


class CandidateGenerator:
    """
    Generates a structured set of candidate trajectories in Frenet space.
    """

    def __init__(
        self,
        lateral_offsets: Optional[List[float]] = None,
        speed_fractions: Optional[List[float]] = None,
        planning_horizon: float = 3.0,
        dt_sample: float = 0.1,
    ):
        # Diverse lateral target offsets (meters from centerline)
        self.lateral_offsets = lateral_offsets if lateral_offsets is not None else [
            0.0, 0.8, -0.8, 1.6, -1.6, 2.4, -2.4
        ]
        # Speed fractions relative to target speed
        self.speed_fractions = speed_fractions if speed_fractions is not None else [
            1.0, 0.70, 0.35, 0.0
        ]
        self.horizon = planning_horizon
        self.dt_sample = dt_sample

    def generate_candidates(
        self,
        ego: Vehicle,
        road: Road,
        target_cruise_speed: float = 12.0
    ) -> List[CandidateTrajectory]:
        """
        Generates candidate trajectories from current ego state.
        """
        candidates: List[CandidateTrajectory] = []
        traj_id = 1

        # Current ego Frenet state
        ego_s, ego_d = road.cartesian_to_frenet(ego.state.x, ego.state.y)
        v0 = max(0.1, ego.state.v)
        a0 = ego.state.a
        
        # Heading error relative to road tangent
        _, _, road_psi = road.evaluate_centerline(ego_s)
        delta_psi = normalize_angle(ego.state.psi - road_psi)
        d_dot0 = v0 * np.sin(delta_psi)
        d_ddot0 = a0 * np.sin(delta_psi)

        T = self.horizon
        time_steps = np.arange(0.0, T + self.dt_sample, self.dt_sample)

        # Iterate over lateral offsets and speed targets
        for d_target in self.lateral_offsets:
            # Lateral polynomial: d(t) from current ego_d to d_target
            lat_poly = QuinticPolynomial(
                x0=ego_d, v0=d_dot0, a0=d_ddot0,
                xf=d_target, vf=0.0, af=0.0, T=T
            )

            for s_frac in self.speed_fractions:
                vf = target_cruise_speed * s_frac
                
                # Longitudinal motion: smooth acceleration profile
                # Target longitudinal distance sf
                avg_v = (v0 + vf) / 2.0
                sf = ego_s + avg_v * T
                lon_poly = QuinticPolynomial(
                    x0=ego_s, v0=v0, a0=a0,
                    xf=sf, vf=vf, af=0.0, T=T
                )

                # Determine maneuver description
                if s_frac == 0.0:
                    maneuver = f"Stop (d={d_target:+.1f}m)"
                elif abs(d_target) < 0.2:
                    maneuver = f"Center (v={vf:.1f}m/s)"
                elif d_target > 0:
                    maneuver = f"Swerve Left (d=+{d_target:.1f}m, v={vf:.1f}m/s)"
                else:
                    maneuver = f"Swerve Right (d={d_target:.1f}m, v={vf:.1f}m/s)"

                # Sample trajectory waypoints
                waypoints: List[TrajectoryWaypoint] = []
                for t in time_steps:
                    s_t = lon_poly.calc_point(t)
                    d_t = lat_poly.calc_point(t)
                    v_t = max(0.0, lon_poly.calc_first_derivative(t))
                    a_t = lon_poly.calc_second_derivative(t)

                    # Convert to Cartesian world frame
                    x_t, y_t, psi_road = road.frenet_to_cartesian(s_t, d_t)
                    
                    # Heading along candidate path
                    d_dot_t = lat_poly.calc_first_derivative(t)
                    psi_t = normalize_angle(psi_road + np.arctan2(d_dot_t, max(v_t, 1e-3)))

                    # Approximate curvature
                    d_ddot_t = lat_poly.calc_second_derivative(t)
                    kappa_t = d_ddot_t / ((1.0 + d_dot_t**2)**1.5 + 1e-6)

                    waypoints.append(TrajectoryWaypoint(
                        t=float(t),
                        x=float(x_t),
                        y=float(y_t),
                        s=float(s_t),
                        d=float(d_t),
                        v=float(v_t),
                        a=float(a_t),
                        psi=float(psi_t),
                        kappa=float(kappa_t)
                    ))

                candidates.append(CandidateTrajectory(
                    id=traj_id,
                    waypoints=waypoints,
                    target_d=d_target,
                    target_v=vf,
                    duration=T,
                    maneuver_name=maneuver
                ))
                traj_id += 1

        return candidates
