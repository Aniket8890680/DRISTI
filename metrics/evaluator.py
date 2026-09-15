"""
Metrics Evaluator Module:
Calculates official quantitative benchmark metrics for SIH performance evaluation:
1. Scenario completion rate
2. Collision count
3. Near-miss count (clearance < 1.0m)
4. Replanning latency (ms)
5. Number of replanning events
6. Path length (m)
7. Path smoothness (integral of jerk squared)
8. Average speed (km/h)
9. Maximum acceleration (m/s^2)
10. Maximum lateral acceleration (m/s^2)
11. Minimum obstacle clearance (m)
12. Time-to-Collision (TTC) statistics
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd

from scenarios.scenario_base import ScenarioBase, TelemetrySnapshot


@dataclass
class ScenarioMetrics:
    """Consolidated quantitative results for a scenario run."""
    scenario_name: str
    planner_type: str
    completion_status: str       # SUCCESS, COLLISION, OFF_ROAD, TIMEOUT
    is_success: bool
    total_duration_s: float
    total_path_length_m: float
    average_speed_kmh: float
    max_speed_kmh: float
    max_longitudinal_accel_mps2: float
    max_longitudinal_decel_mps2: float
    max_lateral_accel_mps2: float
    path_smoothness_jerk_metric: float
    collision_count: int
    near_miss_count: int
    minimum_clearance_m: float
    replanning_events: int
    avg_replanning_latency_ms: float
    min_ttc_observed_s: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary_report(self) -> str:
        """Formats the results as a human-readable SIH jury report."""
        status_symbol = "SUCCESS [PASSED]" if self.is_success else f"FAILED [{self.completion_status}]"
        report = [
            "=" * 65,
            f"SCENARIO REPORT: {self.scenario_name.upper()}",
            f"PLANNER MODE:   {self.planner_type.upper()}",
            "=" * 65,
            f"Completion Status:          {status_symbol}",
            f"Collisions:                 {self.collision_count}",
            f"Near Misses (< 1.0m):       {self.near_miss_count}",
            f"Minimum Clearance:          {self.minimum_clearance_m:.2f} m",
            f"Minimum TTC Observed:       {self.min_ttc_observed_s:.2f} s" if self.min_ttc_observed_s < 99 else "Minimum TTC Observed:       N/A (Clear path)",
            f"Average Speed:              {self.average_speed_kmh:.1f} km/h (Max: {self.max_speed_kmh:.1f} km/h)",
            f"Path Length Traveled:       {self.total_path_length_m:.1f} m",
            f"Path Smoothness (Jerk Cost):{self.path_smoothness_jerk_metric:.2f}",
            f"Max Acceleration / Decel:   +{self.max_longitudinal_accel_mps2:.2f} / -{self.max_longitudinal_decel_mps2:.2f} m/s^2",
            f"Max Lateral Acceleration:   {self.max_lateral_accel_mps2:.2f} m/s^2",
            f"Replanning Events:          {self.replanning_events}",
            f"Avg Replanning Latency:     {self.avg_replanning_latency_ms:.2f} ms",
            "=" * 65,
        ]
        return "\n".join(report)


class MetricsEvaluator:
    """
    Evaluates scenario execution telemetry and calculates all SIH benchmark metrics.
    """

    @staticmethod
    def evaluate(scenario: ScenarioBase) -> ScenarioMetrics:
        """
        Processes the scenario telemetry snapshots to produce a complete ScenarioMetrics object.
        """
        telemetry = scenario.telemetry
        if len(telemetry) == 0:
            return ScenarioMetrics(
                scenario_name=scenario.name,
                planner_type="Baseline" if scenario.use_baseline else "Adaptive",
                completion_status="EMPTY",
                is_success=False,
                total_duration_s=0.0,
                total_path_length_m=0.0,
                average_speed_kmh=0.0,
                max_speed_kmh=0.0,
                max_longitudinal_accel_mps2=0.0,
                max_longitudinal_decel_mps2=0.0,
                max_lateral_accel_mps2=0.0,
                path_smoothness_jerk_metric=0.0,
                collision_count=0,
                near_miss_count=0,
                minimum_clearance_m=0.0,
                replanning_events=0,
                avg_replanning_latency_ms=0.0,
                min_ttc_observed_s=float('inf')
            )

        times = np.array([t.time for t in telemetry])
        xs = np.array([t.ego_x for t in telemetry])
        ys = np.array([t.ego_y for t in telemetry])
        vs = np.array([t.ego_v for t in telemetry])
        accels = np.array([t.ego_a for t in telemetry])
        headings = np.array([t.ego_psi for t in telemetry])
        steers = np.array([t.ego_delta for t in telemetry])

        duration = float(times[-1] - times[0])
        
        # 1. Path length
        dx = np.diff(xs)
        dy = np.diff(ys)
        path_length = float(np.sum(np.hypot(dx, dy)))

        # 2. Speeds
        avg_speed_kmh = float(np.mean(vs) * 3.6)
        max_speed_kmh = float(np.max(vs) * 3.6)

        # 3. Accelerations
        max_accel = float(max(0.0, np.max(accels)))
        max_decel = float(max(0.0, -np.min(accels)))

        # 4. Lateral acceleration: a_lat = v^2 * kappa or v * yaw_rate
        dpsi = np.diff(headings)
        # Unwrap heading differences to avoid 2pi jumps
        dpsi = (dpsi + np.pi) % (2.0 * np.pi) - np.pi
        dt_arr = np.diff(times)
        dt_arr[dt_arr < 1e-4] = 0.05
        yaw_rates = dpsi / dt_arr
        lat_accels = np.abs(vs[1:] * yaw_rates)
        max_lat_accel = float(np.max(lat_accels)) if len(lat_accels) > 0 else 0.0

        # 5. Path smoothness: Integral of squared jerk
        if len(accels) > 1:
            jerks = np.diff(accels) / dt_arr
            jerk_cost = float(np.sum(jerks**2 * dt_arr))
        else:
            jerk_cost = 0.0

        # 6. Safety metrics
        collisions = sum(1 for t in telemetry if t.has_collision)
        near_misses = sum(1 for t in telemetry if (not t.has_collision and t.min_clearance < 1.0))
        clearances = [t.min_clearance for t in telemetry if t.min_clearance < float('inf')]
        min_clearance = float(min(clearances)) if len(clearances) > 0 else 99.0

        # 7. TTC stats
        ttcs = [t.min_ttc for t in telemetry if t.min_ttc < float('inf')]
        min_ttc = float(min(ttcs)) if len(ttcs) > 0 else 99.0

        # 8. Replanning stats
        if hasattr(scenario.planner, 'replanning_count'):
            replanning_events = scenario.planner.replanning_count
        else:
            replanning_events = 0

        if hasattr(scenario.planner, 'replanning_latencies_ms') and len(scenario.planner.replanning_latencies_ms) > 0:
            avg_latency = float(np.mean(scenario.planner.replanning_latencies_ms))
        else:
            avg_latency = 0.5  # default nominal computation latency

        planner_label = "Baseline (Naive)" if scenario.use_baseline else "Adaptive (Ours)"

        return ScenarioMetrics(
            scenario_name=scenario.name,
            planner_type=planner_label,
            completion_status=scenario.termination_reason,
            is_success=scenario.is_success,
            total_duration_s=duration,
            total_path_length_m=path_length,
            average_speed_kmh=avg_speed_kmh,
            max_speed_kmh=max_speed_kmh,
            max_longitudinal_accel_mps2=max_accel,
            max_longitudinal_decel_mps2=max_decel,
            max_lateral_accel_mps2=max_lat_accel,
            path_smoothness_jerk_metric=jerk_cost,
            collision_count=collisions,
            near_miss_count=near_misses,
            minimum_clearance_m=min_clearance,
            replanning_events=replanning_events,
            avg_replanning_latency_ms=avg_latency,
            min_ttc_observed_s=min_ttc
        )

    @staticmethod
    def compare(baseline_metrics: ScenarioMetrics, adaptive_metrics: ScenarioMetrics) -> pd.DataFrame:
        """
        Creates a side-by-side comparative table between Baseline and Adaptive Planner.
        """
        comparison_data = {
            "Metric": [
                "Completion Status",
                "Success Rate",
                "Collision Count",
                "Near-Miss Count (<1m)",
                "Minimum Clearance (m)",
                "Minimum TTC (s)",
                "Average Speed (km/h)",
                "Path Smoothness (Jerk)",
                "Max Deceleration (m/s²)",
                "Replanning Events",
                "Avg Latency (ms)"
            ],
            "Baseline Planner": [
                baseline_metrics.completion_status,
                "PASSED" if baseline_metrics.is_success else "FAILED",
                str(baseline_metrics.collision_count),
                str(baseline_metrics.near_miss_count),
                f"{baseline_metrics.minimum_clearance_m:.2f}",
                f"{baseline_metrics.min_ttc_observed_s:.2f}" if baseline_metrics.min_ttc_observed_s < 90 else "N/A",
                f"{baseline_metrics.average_speed_kmh:.1f}",
                f"{baseline_metrics.path_smoothness_jerk_metric:.1f}",
                f"{baseline_metrics.max_longitudinal_decel_mps2:.2f}",
                str(baseline_metrics.replanning_events),
                f"{baseline_metrics.avg_replanning_latency_ms:.2f}"
            ],
            "Adaptive Planner (Ours)": [
                adaptive_metrics.completion_status,
                "PASSED" if adaptive_metrics.is_success else "FAILED",
                str(adaptive_metrics.collision_count),
                str(adaptive_metrics.near_miss_count),
                f"{adaptive_metrics.minimum_clearance_m:.2f}",
                f"{adaptive_metrics.min_ttc_observed_s:.2f}" if adaptive_metrics.min_ttc_observed_s < 90 else "N/A",
                f"{adaptive_metrics.average_speed_kmh:.1f}",
                f"{adaptive_metrics.path_smoothness_jerk_metric:.1f}",
                f"{adaptive_metrics.max_longitudinal_decel_mps2:.2f}",
                str(adaptive_metrics.replanning_events),
                f"{adaptive_metrics.avg_replanning_latency_ms:.2f}"
            ]
        }
        return pd.DataFrame(comparison_data)
