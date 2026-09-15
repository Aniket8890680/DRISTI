"""Unit tests for Stage 4 (Sensors) and Stage 5 (Fusion and Tracking)."""

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np

from simulation.actors import Actor, ActorType, BehaviorMode
from simulation.vehicle import Vehicle, VehicleState
from simulation.environment import WeatherCondition
from sensors.camera import CameraSensor
from sensors.radar import RadarSensor
from sensors.lidar import LidarSensor
from sensors.gps_imu import GpsImuSensor
from perception.sensor_fusion import SensorFusion
from tracking.tracker import MultiObjectTracker, TrackState


class TestSensorsAndTracking(unittest.TestCase):

    def setUp(self):
        self.ego = Vehicle(initial_state=VehicleState(x=0.0, y=0.0, v=10.0, psi=0.0))
        self.actors = [
            Actor(actor_id=1, actor_type=ActorType.CAR, x=20.0, y=0.0, v=8.0, heading=0.0),
            Actor(actor_id=2, actor_type=ActorType.MOTORCYCLE, x=15.0, y=2.0, v=12.0, heading=0.0),
            Actor(actor_id=3, actor_type=ActorType.ANIMAL, x=10.0, y=-1.5, v=1.0, heading=np.pi/2),
        ]

    def test_camera_detection_and_weather(self):
        camera = CameraSensor(fov_deg=90.0, max_range=50.0, random_seed=42)
        
        # Test normal weather
        normal_dets = camera.sense(self.ego, self.actors, WeatherCondition.NORMAL)
        self.assertGreaterEqual(len(normal_dets), 2)
        
        # Test fog weather: confidence should decrease
        fog_dets = camera.sense(self.ego, self.actors, WeatherCondition.FOG)
        if len(fog_dets) > 0 and len(normal_dets) > 0:
            self.assertLess(fog_dets[0].confidence, normal_dets[0].confidence)

    def test_radar_doppler_velocity(self):
        radar = RadarSensor(max_range=60.0, random_seed=42)
        targets = radar.sense(self.ego, self.actors, WeatherCondition.NORMAL)
        self.assertGreaterEqual(len(targets), 1)

        # Actor 1 is moving at 8 m/s, ego at 10 m/s -> relative radial velocity should be negative (~ -2.0 m/s)
        car_target = [t for t in targets if t.actor_id == 1]
        if car_target:
            self.assertAlmostEqual(car_target[0].radial_velocity_mps, -2.0, delta=1.0)

    def test_lidar_point_clusters(self):
        lidar = LidarSensor(max_range=50.0, random_seed=42)
        clusters = lidar.sense(self.ego, self.actors, WeatherCondition.NORMAL)
        self.assertGreaterEqual(len(clusters), 2)
        for cl in clusters:
            self.assertGreater(len(cl.point_cloud), 0)
            self.assertGreater(cl.confidence, 0.7)

    def test_gps_imu_localization(self):
        gps_imu = GpsImuSensor(random_seed=42)
        pose = gps_imu.sense(self.ego, current_time=1.0)
        self.assertAlmostEqual(pose.x, self.ego.state.x, delta=1.0)
        self.assertAlmostEqual(pose.y, self.ego.state.y, delta=1.0)
        self.assertAlmostEqual(pose.psi, self.ego.state.psi, delta=0.1)

    def test_sensor_fusion_integration(self):
        camera = CameraSensor(random_seed=42)
        radar = RadarSensor(random_seed=42)
        lidar = LidarSensor(random_seed=42)
        gps_imu = GpsImuSensor(random_seed=42)
        fusion = SensorFusion(spatial_gate_distance=3.0)

        cam_dets = camera.sense(self.ego, self.actors)
        rad_tgts = radar.sense(self.ego, self.actors)
        lid_cl = lidar.sense(self.ego, self.actors)
        pose = gps_imu.sense(self.ego, current_time=0.0)

        fused = fusion.fuse(cam_dets, rad_tgts, lid_cl, pose)
        self.assertGreaterEqual(len(fused), 1)

        # Verify multi-modal sources
        found_multi = any(len(f.sources) >= 2 for f in fused)
        self.assertTrue(found_multi, "At least one object should be detected by multiple sensors")

    def test_tracker_kalman_continuity(self):
        fusion = SensorFusion()
        tracker = MultiObjectTracker()
        camera = CameraSensor(random_seed=42)
        radar = RadarSensor(random_seed=42)
        lidar = LidarSensor(random_seed=42)
        gps_imu = GpsImuSensor(random_seed=42)

        dt = 0.1
        # Step for 5 frames
        for frame in range(5):
            t = frame * dt
            # Step actors
            for a in self.actors:
                a.update(dt, t)

            cam_dets = camera.sense(self.ego, self.actors)
            rad_tgts = radar.sense(self.ego, self.actors)
            lid_cl = lidar.sense(self.ego, self.actors)
            pose = gps_imu.sense(self.ego, t)

            fused = fusion.fuse(cam_dets, rad_tgts, lid_cl, pose)
            tracks = tracker.update(fused, dt)

        # Tracker should have confirmed tracks
        confirmed = [tr for tr in tracks if tr.state == TrackState.CONFIRMED]
        self.assertGreaterEqual(len(confirmed), 1)
        
        # Verify velocity estimation is positive and within reasonable bounds
        for tr in confirmed:
            self.assertGreater(tr.speed, 0.0)


if __name__ == '__main__':
    unittest.main()
