"""Unit tests for Stage 6 (Motion Prediction) and Stage 7 (Risk Assessment)."""

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

from simulation.actors import ActorType
from simulation.vehicle import Vehicle, VehicleState
from tracking.tracker import TrackedObject
from prediction.trajectory_predictor import TrajectoryPredictor
from planning.risk import RiskAssessor, RiskLevel


class TestPredictionAndRisk(unittest.TestCase):

    def test_trajectory_prediction_horizon_and_uncertainty(self):
        predictor = TrajectoryPredictor(horizon_seconds=2.0, dt_pred=0.1)
        
        # Car object moving forward at 10 m/s
        car_obj = TrackedObject(
            track_id=1, actor_type=ActorType.CAR,
            x=10.0, y=0.0, vx=10.0, vy=0.0, speed=10.0, heading=0.0,
            length=4.5, width=1.8, confidence=0.9, uncertainty=0.2, sources=['camera']
        )
        # Animal object moving across at 1 m/s
        animal_obj = TrackedObject(
            track_id=2, actor_type=ActorType.ANIMAL,
            x=20.0, y=-2.0, vx=0.0, vy=1.0, speed=1.0, heading=np.pi/2,
            length=2.2, width=1.0, confidence=0.85, uncertainty=0.9, sources=['camera']
        )

        preds = predictor.predict([car_obj, animal_obj])
        self.assertEqual(len(preds), 2)
        
        # 2.0s / 0.1s = 20 points
        car_pred = preds[0]
        self.assertEqual(len(car_pred.points), 20)
        # Car position at t=2.0 should be roughly 10 + 10*2 = 30m
        self.assertAlmostEqual(car_pred.points[-1].x, 30.0, delta=0.5)

        # Animal uncertainty expansion should be significantly larger than car
        animal_pred = preds[1]
        self.assertGreater(animal_pred.points[-1].uncertainty_radius, car_pred.points[-1].uncertainty_radius)

    def test_risk_assessment_ttc_and_critical_level(self):
        # Ego moving forward at 15 m/s
        ego = Vehicle(initial_state=VehicleState(x=0.0, y=0.0, v=15.0, psi=0.0))
        
        # Obstacle 15m ahead, stationary (closing speed = 15 m/s, TTC = 1.0s -> CRITICAL)
        obstacle = TrackedObject(
            track_id=10, actor_type=ActorType.CAR,
            x=15.0, y=0.0, vx=0.0, vy=0.0, speed=0.0, heading=0.0,
            length=4.5, width=1.8, confidence=0.95, uncertainty=0.2, sources=['camera', 'radar']
        )

        predictor = TrajectoryPredictor(horizon_seconds=2.0, dt_pred=0.1)
        preds = {p.track_id: p for p in predictor.predict([obstacle])}

        risk_assessor = RiskAssessor()
        risk_summary = risk_assessor.assess_risk(ego, [obstacle], preds)

        self.assertEqual(risk_summary.highest_risk_level, RiskLevel.CRITICAL)
        self.assertTrue(risk_summary.should_emergency_brake)
        self.assertAlmostEqual(risk_summary.min_ttc, 1.0, delta=0.1)

    def test_risk_assessment_safe_distance(self):
        # Ego moving forward at 10 m/s
        ego = Vehicle(initial_state=VehicleState(x=0.0, y=0.0, v=10.0, psi=0.0))
        
        # Obstacle far ahead at 70m moving at same speed (closing speed = 0, TTC = inf -> LOW)
        obstacle = TrackedObject(
            track_id=11, actor_type=ActorType.CAR,
            x=70.0, y=0.0, vx=10.0, vy=0.0, speed=10.0, heading=0.0,
            length=4.5, width=1.8, confidence=0.9, uncertainty=0.2, sources=['radar']
        )

        predictor = TrajectoryPredictor(horizon_seconds=2.0, dt_pred=0.1)
        preds = {p.track_id: p for p in predictor.predict([obstacle])}

        risk_assessor = RiskAssessor()
        risk_summary = risk_assessor.assess_risk(ego, [obstacle], preds)

        self.assertEqual(risk_summary.highest_risk_level, RiskLevel.LOW)
        self.assertFalse(risk_summary.should_emergency_brake)


if __name__ == '__main__':
    unittest.main()
