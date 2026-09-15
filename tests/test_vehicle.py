"""Unit tests for Vehicle model and Road geometry (Stage 1)."""

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from simulation.vehicle import Vehicle, VehicleState, VehicleLimits
from simulation.road import Road


class TestVehicleAndRoad(unittest.TestCase):

    def test_vehicle_straight_motion(self):
        state = VehicleState(x=0.0, y=0.0, v=10.0, a=0.0, psi=0.0, delta=0.0)
        veh = Vehicle(initial_state=state)
        
        # Step forward for 1 second at constant 10 m/s
        dt = 0.1
        for _ in range(10):
            veh.step(target_accel=0.0, target_steer=0.0, dt=dt)

        self.assertTrue(np.isclose(veh.state.x, 10.0, atol=0.1))
        self.assertTrue(np.isclose(veh.state.y, 0.0, atol=1e-3))
        self.assertTrue(np.isclose(veh.state.psi, 0.0, atol=1e-3))
        self.assertTrue(np.isclose(veh.state.v, 10.0, atol=0.01))

    def test_vehicle_acceleration_and_deceleration_limits(self):
        limits = VehicleLimits(max_accel=3.0, max_decel=6.0, max_speed=20.0)
        state = VehicleState(x=0.0, y=0.0, v=5.0, a=0.0, psi=0.0, delta=0.0)
        veh = Vehicle(initial_state=state, limits=limits)

        # Request excessive acceleration (10 m/s^2)
        veh.step(target_accel=10.0, target_steer=0.0, dt=0.5)
        self.assertLessEqual(veh.state.a, limits.max_accel + 1e-4)

        # Request excessive braking (-15 m/s^2)
        veh.step(target_accel=-15.0, target_steer=0.0, dt=0.5)
        self.assertGreaterEqual(veh.state.a, -limits.max_decel - 1e-4)

    def test_vehicle_steering_kinematics(self):
        state = VehicleState(x=0.0, y=0.0, v=5.0, a=0.0, psi=0.0, delta=0.0)
        veh = Vehicle(initial_state=state)

        # Turn left with 0.2 rad steering angle
        for _ in range(10):
            veh.step(target_accel=0.0, target_steer=0.2, dt=0.1)

        # Heading must have increased (turned left, positive yaw rate)
        self.assertGreater(veh.state.psi, 0.0)
        self.assertGreater(veh.state.y, 0.0)
        self.assertGreater(veh.state.x, 0.0)

    def test_vehicle_bounding_box(self):
        state = VehicleState(x=10.0, y=5.0, v=0.0, a=0.0, psi=0.0, delta=0.0)
        limits = VehicleLimits(length=4.0, width=2.0, rear_overhang=1.0)
        veh = Vehicle(initial_state=state, limits=limits)

        corners = veh.get_bounding_box()
        self.assertEqual(corners.shape, (4, 2))
        
        # Center is at x = 10 + (4/2 - 1) = 11, y = 5
        cx, cy = veh.center_position
        self.assertTrue(np.isclose(cx, 11.0))
        self.assertTrue(np.isclose(cy, 5.0))

    def test_road_frenet_conversion(self):
        # Straight road along X axis
        waypoints = np.array([
            [0.0, 0.0],
            [50.0, 0.0],
            [100.0, 0.0],
            [150.0, 0.0]
        ])
        road = Road(waypoints=waypoints, width=7.0)

        # Point at (30.0, 2.0) should be s=30.0, d=2.0
        s, d = road.cartesian_to_frenet(30.0, 2.0)
        self.assertTrue(np.isclose(s, 30.0, atol=0.2))
        self.assertTrue(np.isclose(d, 2.0, atol=0.2))

        # Convert back
        x, y, psi = road.frenet_to_cartesian(30.0, 2.0)
        self.assertTrue(np.isclose(x, 30.0, atol=0.2))
        self.assertTrue(np.isclose(y, 2.0, atol=0.2))
        self.assertTrue(np.isclose(psi, 0.0, atol=0.1))

        # Inside road check: width is 7.0 (boundaries at +3.5 and -3.5)
        self.assertTrue(road.is_point_in_road(30.0, 2.0))
        self.assertFalse(road.is_point_in_road(30.0, 4.0))


if __name__ == '__main__':
    unittest.main()
