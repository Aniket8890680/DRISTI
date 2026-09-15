"""
Telemetry and Scenario Serializer:
Converts simulation execution traces into structured JSON for high-performance
real-time web rendering, HUD gauges, and telemetry analytics.
"""

from typing import Dict, Any, List
import numpy as np

from scenarios.scenario_base import ScenarioBase, TelemetrySnapshot
from metrics.evaluator import MetricsEvaluator, ScenarioMetrics


def serialize_scenario_run(scenario: ScenarioBase) -> Dict[str, Any]:
    """
    Serializes a completed scenario run and its telemetry timeline into a JSON-compatible dict.
    """
    metrics = MetricsEvaluator.evaluate(scenario)

    # 1. Road Geometry
    road_data: Dict[str, Any] = {}
    if scenario.road is not None:
        r = scenario.road
        # Downsample boundaries if needed for lightweight transmission
        step = max(1, len(r.dense_centerline) // 200)
        road_data = {
            "nominal_width": float(r.nominal_width),
            "has_lane_markings": bool(r.has_lane_markings),
            "waypoints": r.raw_waypoints.tolist(),
            "centerline": r.dense_centerline[::step].round(3).tolist(),
            "left_boundary": r.left_boundary[::step].round(3).tolist(),
            "right_boundary": r.right_boundary[::step].round(3).tolist(),
            "potholes": [
                {
                    "x": round(float(ph.x), 2),
                    "y": round(float(ph.y), 2),
                    "radius": round(float(ph.radius), 2),
                    "severity": round(float(ph.severity), 2)
                }
                for ph in r.potholes
            ]
        }

    # 2. Metrics Summary
    metrics_data = {
        "scenario_name": metrics.scenario_name,
        "planner_mode": metrics.planner_type,
        "completion_status": metrics.completion_status,
        "is_success": metrics.is_success,
        "collision_count": int(metrics.collision_count),
        "near_miss_count": int(metrics.near_miss_count),
        "minimum_clearance_m": round(float(metrics.minimum_clearance_m), 2),
        "minimum_ttc_s": round(float(metrics.min_ttc_observed_s), 2),
        "avg_speed_kmh": round(float(metrics.average_speed_kmh), 1),
        "max_speed_kmh": round(float(metrics.max_speed_kmh), 1),
        "path_length_m": round(float(metrics.total_path_length_m), 1),
        "path_smoothness_jerk": round(float(metrics.path_smoothness_jerk_metric), 1),
        "max_lateral_accel": round(float(metrics.max_lateral_accel_mps2), 2),
        "replanning_count": int(metrics.replanning_events),
        "avg_replanning_latency_ms": round(float(metrics.avg_replanning_latency_ms), 1),
    }

    # 3. Frames Timeline
    frames: List[Dict[str, Any]] = []
    # Sample every frame (dt=0.05 is 20 FPS, ideal for smooth web playback)
    for snap in scenario.telemetry:
        # Active trajectory
        active_path = []
        maneuver_name = "Cruise"
        if snap.active_trajectory and snap.active_trajectory.waypoints:
            maneuver_name = snap.active_trajectory.maneuver_name
            active_path = [
                [round(float(wp.x), 2), round(float(wp.y), 2), round(float(wp.v), 2)]
                for wp in snap.active_trajectory.waypoints[::2]
            ]

        # Candidate trajectories (sampled fan)
        candidates = []
        if snap.candidate_trajectories and not scenario.use_baseline:
            for cand in snap.candidate_trajectories[:12]:
                if cand.waypoints:
                    candidates.append([
                        [round(float(wp.x), 2), round(float(wp.y), 2)]
                        for wp in cand.waypoints[::3]
                    ])

        # Tracked objects
        tracked = []
        for obj in snap.tracked_objects:
            box = obj.get_bounding_box().round(2).tolist()
            tracked.append({
                "id": int(obj.track_id),
                "type": obj.actor_type.value,
                "x": round(float(obj.x), 2),
                "y": round(float(obj.y), 2),
                "vx": round(float(obj.vx), 2),
                "vy": round(float(obj.vy), 2),
                "speed": round(float(obj.speed), 2),
                "heading": round(float(obj.heading), 3),
                "length": round(float(obj.length), 2),
                "width": round(float(obj.width), 2),
                "uncertainty": round(float(obj.uncertainty), 2),
                "bbox": box,
                "sources": obj.sources
            })

        # Predicted trajectories
        preds = {}
        for trk_id, p_traj in snap.predicted_trajectories.items():
            preds[str(trk_id)] = [
                [round(float(pt.x), 2), round(float(pt.y), 2), round(float(pt.uncertainty_radius), 2)]
                for pt in p_traj.points[::2]
            ]

        frame_data = {
            "time": round(float(snap.time), 2),
            "ego": {
                "x": round(float(snap.ego_x), 2),
                "y": round(float(snap.ego_y), 2),
                "v": round(float(snap.ego_v), 2),
                "a": round(float(snap.ego_a), 2),
                "psi": round(float(snap.ego_psi), 3),
                "delta": round(float(snap.ego_delta), 3),
                "cmd_a": round(float(snap.command_accel), 2),
                "cmd_steer": round(float(snap.command_steer), 3),
                "length": 4.6,
                "width": 1.85,
            },
            "planner_state": snap.planner_state,
            "maneuver": maneuver_name,
            "risk_level": snap.risk_level,
            "min_ttc": round(float(snap.min_ttc), 2) if snap.min_ttc < 99.0 else 99.9,
            "min_clearance": round(float(snap.min_clearance), 2),
            "has_collision": bool(snap.has_collision),
            "is_off_road": bool(snap.is_off_road),
            "active_trajectory": active_path,
            "candidates": candidates,
            "tracked_objects": tracked,
            "predictions": preds,
        }
        frames.append(frame_data)

    return {
        "scenario_name": scenario.name,
        "description": scenario.description,
        "road": road_data,
        "metrics": metrics_data,
        "frames": frames,
        "total_frames": len(frames),
        "dt": scenario.env.dt if scenario.env else 0.05
    }
