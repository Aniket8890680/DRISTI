"""Unit tests for Stage 2 (Actors) and Stage 3 (Collision Detection)."""

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

from simulation.actors import (
    Actor, ActorType, BehaviorMode, check_sat_collision,
    compute_polygon_clearance, ACTOR_PROPERTIES
)
from simulation.vehicle import Vehicle, VehicleState
from simulation.road import Road
from simulation.environment import Environment, WeatherCondition


class TestActorsAndCollision(unittest.TestCase):

    def test_actor_dimensions_and_defaults(self):
        auto = Actor(actor_id=1, actor_type=ActorType.AUTO_RICKSHAW, x=10.0, y=2.0, v=6.0, heading=0.0)
        self.assertEqual(auto.type, ActorType.AUTO_RICKSHAW)
        self.assertEqual(auto.length, ACTOR_PROPERTIES[ActorType.AUTO_RICKSHAW]["length"])
        self.assertEqual(auto.width, ACTOR_PROPERTIES[ActorType.AUTO_RICKSHAW]["width"])
        self.assertAlmostEqual(auto.vx, 6.0)
        self.assertAlmostEqual(auto.vy, 0.0)

    def test_sudden_cut_in_behavior(self):
        # Motorcycle initially on right shoulder (heading 0.0), then cuts in at t=1.0s
        moto = Actor(
            actor_id=2,
            actor_type=ActorType.MOTORCYCLE,
            x=15.0,
            y=-2.0,
            v=10.0,
            heading=0.0,
            behavior=BehaviorMode.SUDDEN_CUT_IN,
            behavior_params={"trigger_time": 0.5, "yaw_rate": 0.5}
        )
        
        # Step before trigger
        moto.update(dt=0.2, current_time=0.2)
        self.assertAlmostEqual(moto.heading, 0.0)
        self.assertAlmostEqual(moto.y, -2.0)

        # Step past trigger time (t = 0.6)
        moto.update(dt=0.4, current_time=0.6)
        # Heading should turn left (positive yaw rate) and y should increase toward road center
        self.assertGreater(moto.heading, 0.0)
        self.assertGreater(moto.y, -2.0)

    def test_sat_collision_detection(self):
        # Two boxes overlapping
        box_a = np.array([[0, 0], [4, 0], [4, 2], [0, 2]], dtype=float)
        box_b = np.array([[2, 1], [6, 1], [6, 3], [2, 3]], dtype=float)
        self.assertTrue(check_sat_collision(box_a, box_b))

        # Two boxes separated along X axis
        box_c = np.array([[10, 0], [14, 0], [14, 2], [10, 2]], dtype=float)
        self.assertFalse(check_sat_collision(box_a, box_c))

        # Test clearance between separated boxes
        clearance = compute_polygon_clearance(box_a, box_c)
        self.assertAlmostEqual(clearance, 6.0, places=2)

    def test_environment_closed_loop_collision(self):
        waypoints = np.array([[0, 0], [100, 0]], dtype=float)
        road = Road(waypoints=waypoints, width=7.0)
        
        # Ego moving forward at 10 m/s from x=0
        ego = Vehicle(initial_state=VehicleState(x=0.0, y=0.0, v=10.0, psi=0.0))
        env = Environment(road=road, ego_vehicle=ego, dt=0.1)

        # Obstacle placed at x=15.0 (stationary car)
        stationary_car = Actor(actor_id=1, actor_type=ActorType.CAR, x=15.0, y=0.0, v=0.0, heading=0.0, behavior=BehaviorMode.STOPPED)
        env.add_actor(stationary_car)

        # Before stepping near obstacle: no collision
        has_collided, clearance = env.step(target_accel=0.0, target_steer=0.0)
        self.assertFalse(has_collided)
        self.assertGreater(clearance, 5.0)

        # Continue driving straight towards obstacle
        collided = False
        for _ in range(25):
            c, _ = env.step(target_accel=0.0, target_steer=0.0)
            if c:
                collided = True
                break

        self.assertTrue(collided, "Ego should collide if driving straight into obstacle without avoidance")
        self.assertGreaterEqual(env.stats.collision_count, 1)


if __name__ == '__main__':
    unittest.main()
