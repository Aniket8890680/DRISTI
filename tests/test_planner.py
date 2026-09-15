"""Unit tests for Stage 8 (Candidates), Stage 9 (Adaptive Planner), and Stage 10 (Control)."""

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

from simulation.road import Road
from simulation.actors import Actor, ActorType, BehaviorMode
from simulation.vehicle import Vehicle, VehicleState
from tracking.tracker import TrackedObject
from prediction.trajectory_predictor import TrajectoryPredictor
from planning.drivable_area import DrivableAreaEstimator
from planning.candidate_paths import CandidateGenerator, QuinticPolynomial
from planning.adaptive_planner import AdaptivePlanner, PlannerConfig, PlannerState
from planning.risk import RiskAssessor
from control.vehicle_controller import VehicleController


class TestPlannerAndControl(unittest.TestCase):

    def setUp(self):
        # 100m straight road, width 8m
        waypoints = np.array([[0, 0], [100, 0]], dtype=float)
        self.road = Road(waypoints=waypoints, width=8.0)
        self.ego = Vehicle(initial_state=VehicleState(x=0.0, y=0.0, v=10.0, psi=0.0))

    def test_quintic_polynomial_boundary_conditions(self):
        # Test boundary satisfaction: x0=0, xf=5, v0=1, vf=0, a0=0, af=0, T=2.0
        poly = QuinticPolynomial(x0=0.0, v0=1.0, a0=0.0, xf=5.0, vf=0.0, af=0.0, T=2.0)
        self.assertAlmostEqual(poly.calc_point(0.0), 0.0)
        self.assertAlmostEqual(poly.calc_first_derivative(0.0), 1.0)
        self.assertAlmostEqual(poly.calc_point(2.0), 5.0)
        self.assertAlmostEqual(poly.calc_first_derivative(2.0), 0.0)

    def test_candidate_generation(self):
        gen = CandidateGenerator(planning_horizon=2.5, dt_sample=0.1)
        candidates = gen.generate_candidates(self.ego, self.road, target_cruise_speed=10.0)
        self.assertGreater(len(candidates), 10)
        # Verify first waypoint matches ego
        first_wp = candidates[0].waypoints[0]
        self.assertAlmostEqual(first_wp.x, self.ego.state.x, delta=0.5)

    def test_drivable_corridor_estimation(self):
        estimator = DrivableAreaEstimator(nominal_safety_margin=0.5)
        # Add pothole at (30, -1.0) with radius 0.8
        self.road.add_pothole(x=30.0, y=-1.0, radius=0.8)

        corridor = estimator.estimate_corridor(self.road, ego_s=0.0, tracked_objects=[])
        self.assertGreater(len(corridor), 5)
        
        # Pothole on right side should push d_min inward
        pothole_seg = [seg for seg in corridor if seg.has_pothole]
        self.assertGreater(len(pothole_seg), 0)
        self.assertGreater(pothole_seg[0].d_min, -3.5)

    def test_adaptive_evasion_planning(self):
        # Obstacle placed directly ahead at x=25m, y=0.0m (centerline)
        obstacle = TrackedObject(
            track_id=1, actor_type=ActorType.CAR,
            x=25.0, y=0.0, vx=0.0, vy=0.0, speed=0.0, heading=0.0,
            length=4.5, width=1.8, confidence=0.95, uncertainty=0.2, sources=['camera']
        )
        
        predictor = TrajectoryPredictor(horizon_seconds=3.0, dt_pred=0.1)
        preds = {p.track_id: p for p in predictor.predict([obstacle])}

        risk_assessor = RiskAssessor()
        risk_summary = risk_assessor.assess_risk(self.ego, [obstacle], preds, road_width=8.0)

        planner = AdaptivePlanner()
        best_traj = planner.plan(self.ego, self.road, [obstacle], preds, risk_summary)

        self.assertIsNotNone(best_traj)
        # Planner should not choose staying at y=0! It must swerve left or right around the obstacle
        self.assertNotAlmostEqual(best_traj.target_d, 0.0, delta=0.3)
        self.assertTrue(best_traj.is_collision_free)

    def test_controller_tracking(self):
        controller = VehicleController()
        # Create a straight trajectory
        gen = CandidateGenerator(planning_horizon=2.0)
        candidates = gen.generate_candidates(self.ego, self.road, target_cruise_speed=10.0)
        straight_traj = [c for c in candidates if abs(c.target_d) < 0.1 and c.target_v > 8.0][0]

        cmd = controller.compute_control(self.ego, straight_traj, dt=0.05)
        self.assertIsInstance(cmd.target_accel, float)
        self.assertIsInstance(cmd.target_steer, float)
        self.assertLessEqual(abs(cmd.target_steer), self.ego.limits.max_steer)


if __name__ == '__main__':
    unittest.main()
