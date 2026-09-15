
"""
Main Entry Point: Autonomous Driving Simulation Prototype for Unstructured Indian Roads
Smart India Hackathon (SIH) Prototype

Usage Examples:
    python main.py --scenario village
    python main.py --scenario intersection
    python main.py --scenario highway_merge
    python main.py --scenario market
    python main.py --scenario cattle_crossing
    python main.py --scenario all
    python main.py --compare
    python main.py --scenario village --weather fog --save-plots
"""

import os
import sys
import argparse
from typing import Dict, List, Type
import pandas as pd

from simulation.environment import WeatherCondition
from scenarios.scenario_base import ScenarioBase
from scenarios.village import VillageScenario
from scenarios.intersection import IntersectionScenario
from scenarios.highway_merge import HighwayMergeScenario
from scenarios.market import MarketScenario
from scenarios.cattle_crossing import CattleCrossingScenario
from metrics.evaluator import MetricsEvaluator, ScenarioMetrics
from visualization.renderer import TopDownRenderer

SCENARIO_REGISTRY: Dict[str, Type[ScenarioBase]] = {
    "village": VillageScenario,
    "intersection": IntersectionScenario,
    "highway_merge": HighwayMergeScenario,
    "market": MarketScenario,
    "cattle_crossing": CattleCrossingScenario,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SIH Autonomous Driving: Adaptive Path Planning on Unstructured Indian Roads"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="village",
        choices=["village", "intersection", "highway_merge", "market", "cattle_crossing", "all"],
        help="Scenario to simulate or 'all' to run full suite."
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run baseline comparison (Baseline vs Adaptive Planner) across scenarios."
    )
    parser.add_argument(
        "--weather",
        type=str,
        default="normal",
        choices=["normal", "rain", "fog", "night"],
        help="Environmental weather condition modulating sensor uncertainty."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sensor noise and actor behaviors."
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save top-down trajectory snapshots and metrics to output/ directory."
    )
    parser.add_argument(
        "--save-gif",
        action="store_true",
        help="Save animated replay GIF to output/ directory."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save generated plots and reports."
    )
    return parser.parse_args()


def run_single_scenario(
    scenario_key: str,
    weather: WeatherCondition,
    use_baseline: bool,
    seed: int,
    save_plots: bool,
    save_gif: bool,
    output_dir: str,
) -> ScenarioMetrics:
    """Instantiates, executes, evaluates, and renders a single scenario."""
    cls = SCENARIO_REGISTRY[scenario_key]
    scenario = cls(weather=weather, use_baseline_planner=use_baseline, random_seed=seed)
    
    planner_mode = "Baseline" if use_baseline else "Adaptive"
    print(f"\n>>> Running Scenario: {scenario.name} [{planner_mode} Planner] | Weather: {weather.value.upper()} ...")
    
    # Run closed loop
    scenario.run()

    # Evaluate metrics
    metrics = MetricsEvaluator.evaluate(scenario)
    print(metrics.summary_report())

    # Visualizations
    if save_plots or save_gif:
        os.makedirs(output_dir, exist_ok=True)
        renderer = TopDownRenderer()
        prefix = f"{scenario_key}_{planner_mode.lower()}_{weather.value}"

        if save_plots:
            # Save final state snapshot
            snap_path = os.path.join(output_dir, f"{prefix}_snapshot.png")
            renderer.render_snapshot(scenario, frame_idx=-1, save_path=snap_path)
            print(f"[Plot Saved] Top-down snapshot saved to: {snap_path}")

        if save_gif:
            gif_path = os.path.join(output_dir, f"{prefix}_animation.gif")
            renderer.save_animation_gif(scenario, output_path=gif_path)
            print(f"[Animation Saved] Replay GIF saved to: {gif_path}")

    return metrics


def run_comparison(weather: WeatherCondition, seed: int, output_dir: str, save_plots: bool) -> None:
    """Runs all 5 scenarios under both Baseline and Adaptive Planners and prints comparison."""
    print("\n" + "=" * 75)
    print("      SIH BENCHMARK: BASELINE PLANNER VS OUR ADAPTIVE PLANNER")
    print("=" * 75)

    summary_rows = []

    for key, cls in SCENARIO_REGISTRY.items():
        print(f"\n{'='*30} BENCHMARK: {key.upper()} {'='*30}")
        # Run Baseline
        base_m = run_single_scenario(
            scenario_key=key, weather=weather, use_baseline=True,
            seed=seed, save_plots=save_plots, save_gif=False, output_dir=output_dir
        )
        # Run Adaptive
        adapt_m = run_single_scenario(
            scenario_key=key, weather=weather, use_baseline=False,
            seed=seed, save_plots=save_plots, save_gif=False, output_dir=output_dir
        )

        # Print comparison table
        comp_df = MetricsEvaluator.compare(base_m, adapt_m)
        print(f"\n--- SIDE-BY-SIDE COMPARISON: {base_m.scenario_name} ---")
        print(comp_df.to_string(index=False))

        summary_rows.append({
            "Scenario": base_m.scenario_name,
            "Baseline Collisions": base_m.collision_count,
            "Adaptive Collisions": adapt_m.collision_count,
            "Baseline Near-Misses": base_m.near_miss_count,
            "Adaptive Near-Misses": adapt_m.near_miss_count,
            "Baseline Min Clearance (m)": f"{base_m.minimum_clearance_m:.2f}",
            "Adaptive Min Clearance (m)": f"{adapt_m.minimum_clearance_m:.2f}",
            "Baseline Status": base_m.completion_status,
            "Adaptive Status": adapt_m.completion_status,
        })

    # Overall Summary Table
    print("\n" + "=" * 75)
    print("                     OVERALL BENCHMARK SUMMARY")
    print("=" * 75)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    os.makedirs(output_dir, exist_ok=True)
    summary_csv = os.path.join(output_dir, "benchmark_comparison_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[Report Exported] Comparative CSV summary saved to: {summary_csv}\n")


def main() -> None:
    args = parse_arguments()
    weather = WeatherCondition(args.weather)

    if args.compare:
        run_comparison(weather=weather, seed=args.seed, output_dir=args.output_dir, save_plots=args.save_plots)
    elif args.scenario == "all":
        print("\n========================================================")
        print("          EXECUTING FULL SUITE: ALL 5 SCENARIOS")
        print("========================================================")
        for key in SCENARIO_REGISTRY.keys():
            run_single_scenario(
                scenario_key=key,
                weather=weather,
                use_baseline=False,
                seed=args.seed,
                save_plots=args.save_plots,
                save_gif=args.save_gif,
                output_dir=args.output_dir
            )
    else:
        run_single_scenario(
            scenario_key=args.scenario,
            weather=weather,
            use_baseline=False,
            seed=args.seed,
            save_plots=args.save_plots,
            save_gif=args.save_gif,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
