"""
Sensor Fusion Module:
Combines multi-modal observations from Camera, Radar, LiDAR, and GPS/IMU
into a unified, robust environmental world model.

Key Principles:
- Do not blindly trust one sensor.
- Camera excels in semantic classification (e.g., distinguishing cattle vs motorcycle vs rickshaw).
- Radar provides direct, highly accurate range-rate (Doppler velocity) and all-weather capability.
- LiDAR provides high-precision metric boundaries, obstacle contours, and spatial centroids.
- Spatial gating and weighted least-squares sensor fusion generate consistent tracked inputs.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np

from simulation.actors import ActorType, ACTOR_PROPERTIES
from sensors.camera import CameraDetection
from sensors.radar import RadarTarget
from sensors.lidar import LidarObstacleCluster
from sensors.gps_imu import EgoPoseMeasurement
from perception.detector import DetectionPacket


@dataclass
class FusedObjectObservation:
    """Consolidated environmental object observation from multi-sensor fusion."""
    fusion_id: int
    actor_type: ActorType
    x: float
    y: float
    vx: float
    vy: float
    heading: float
    length: float
    width: float
    confidence: float
    sources: List[str]
    uncertainty: float = 0.5


class SensorFusion:
    """
    Multi-sensor fusion engine fusing Camera, Radar, LiDAR, and GPS/IMU measurements.
    """

    def __init__(self, spatial_gate_distance: float = 2.5):
        self.spatial_gate_dist = spatial_gate_distance
        self.next_fusion_id = 1

    def fuse(
        self,
        camera_detections: List[CameraDetection],
        radar_targets: List[RadarTarget],
        lidar_clusters: List[LidarObstacleCluster],
        ego_pose: EgoPoseMeasurement,
    ) -> List[FusedObjectObservation]:
        """
        Fuses asynchronous sensor packets into a common spatial observation list.
        """
        # 1. Convert all measurements into normalized DetectionPackets
        all_packets: List[DetectionPacket] = []

        # From Camera
        for cd in camera_detections:
            all_packets.append(DetectionPacket(
                source_sensor="camera",
                actor_id_hint=cd.actor_id,
                actor_type=cd.actor_type,
                x=cd.x,
                y=cd.y,
                confidence=cd.confidence,
                range_to_ego=cd.relative_distance
            ))

        # From LiDAR
        for lc in lidar_clusters:
            all_packets.append(DetectionPacket(
                source_sensor="lidar",
                actor_id_hint=lc.actor_id,
                actor_type=ActorType.STATIC_OBSTACLE,  # LiDAR geometry without semantic class
                x=lc.centroid_x,
                y=lc.centroid_y,
                heading=lc.heading,
                length=lc.estimated_length,
                width=lc.estimated_width,
                confidence=lc.confidence,
                range_to_ego=lc.distance_to_ego
            ))

        # From Radar
        for rt in radar_targets:
            # Estimate absolute velocity using ego motion
            ego_vx = ego_pose.v * np.cos(ego_pose.psi)
            ego_vy = ego_pose.v * np.sin(ego_pose.psi)
            
            # Unit vector to target
            bearing = ego_pose.psi + rt.azimuth_rad
            ux = np.cos(bearing)
            uy = np.sin(bearing)
            
            # Radial velocity contribution
            est_abs_vx = ego_vx + rt.radial_velocity_mps * ux
            est_abs_vy = ego_vy + rt.radial_velocity_mps * uy

            all_packets.append(DetectionPacket(
                source_sensor="radar",
                actor_id_hint=rt.actor_id,
                actor_type=ActorType.CAR,  # Default radar hypothesis
                x=rt.est_world_x,
                y=rt.est_world_y,
                vx=est_abs_vx,
                vy=est_abs_vy,
                confidence=rt.confidence,
                range_to_ego=rt.range_m
            ))

        # 2. Cluster observations based on spatial proximity
        clusters: List[List[DetectionPacket]] = []
        for packet in all_packets:
            matched_cluster = None
            for cl in clusters:
                # Distance to cluster centroid
                cx = np.mean([p.x for p in cl])
                cy = np.mean([p.y for p in cl])
                if np.hypot(packet.x - cx, packet.y - cy) < self.spatial_gate_dist:
                    matched_cluster = cl
                    break
            if matched_cluster is not None:
                matched_cluster.append(packet)
            else:
                clusters.append([packet])

        # 3. Fuse properties for each cluster
        fused_observations: List[FusedObjectObservation] = []
        for cl in clusters:
            sources = list(set(p.source_sensor for p in cl))
            
            # Semantic classification preference: Camera > LiDAR/Radar
            cam_packets = [p for p in cl if p.source_sensor == "camera"]
            if cam_packets:
                actor_type = cam_packets[0].actor_type
            else:
                # Default to static obstacle or car
                actor_type = cl[0].actor_type

            # Spatial position fusion (weighted by sensor confidence and variance)
            weights = []
            xs = []
            ys = []
            for p in cl:
                # LiDAR has highest spatial weight, then Camera, then Radar
                sensor_weight = 3.0 if p.source_sensor == "lidar" else (1.5 if p.source_sensor == "camera" else 1.0)
                w = p.confidence * sensor_weight
                weights.append(w)
                xs.append(p.x)
                ys.append(p.y)
            
            w_sum = sum(weights)
            fused_x = float(np.sum(np.array(xs) * np.array(weights)) / max(w_sum, 1e-6))
            fused_y = float(np.sum(np.array(ys) * np.array(weights)) / max(w_sum, 1e-6))

            # Velocity fusion: prioritize Radar Doppler velocity
            radar_packets = [p for p in cl if p.source_sensor == "radar" and p.vx is not None]
            if radar_packets:
                fused_vx = float(np.mean([p.vx for p in radar_packets]))
                fused_vy = float(np.mean([p.vy for p in radar_packets]))
            else:
                fused_vx = 0.0
                fused_vy = 0.0

            # Heading and dimension fusion: prioritize LiDAR
            lidar_packets = [p for p in cl if p.source_sensor == "lidar"]
            if lidar_packets:
                fused_length = float(lidar_packets[0].length)
                fused_width = float(lidar_packets[0].width)
                fused_heading = float(lidar_packets[0].heading)
            else:
                props = ACTOR_PROPERTIES.get(actor_type, {"length": 2.0, "width": 1.0})
                fused_length = props["length"]
                fused_width = props["width"]
                fused_heading = np.arctan2(fused_vy, fused_vx) if np.hypot(fused_vx, fused_vy) > 0.5 else 0.0

            # Overall confidence: 1 - prod(1 - c_i)
            fused_conf = 1.0 - np.prod([1.0 - p.confidence for p in cl])
            fused_conf = float(np.clip(fused_conf, 0.1, 0.99))

            # Uncertainty factor: intrinsic to actor type + inverse of sensor agreement
            intrinsic_uncertainty = ACTOR_PROPERTIES.get(actor_type, {}).get("uncertainty", 0.5)
            sensor_bonus = 0.1 * (len(sources) - 1)  # Higher confidence if detected by multiple sensors
            final_uncertainty = float(np.clip(intrinsic_uncertainty - sensor_bonus, 0.05, 1.0))

            fused_id = cl[0].actor_id_hint if (cl[0].actor_id_hint is not None) else self.next_fusion_id
            self.next_fusion_id += 1

            fused_observations.append(FusedObjectObservation(
                fusion_id=fused_id,
                actor_type=actor_type,
                x=fused_x,
                y=fused_y,
                vx=fused_vx,
                vy=fused_vy,
                heading=fused_heading,
                length=fused_length,
                width=fused_width,
                confidence=fused_conf,
                sources=sources,
                uncertainty=final_uncertainty
            ))

        return fused_observations
