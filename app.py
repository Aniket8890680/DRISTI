"""
Streamlit Web Application:
Interactive Autonomous Driving Simulation & Demonstration Dashboard
Smart India Hackathon (SIH) Prototype
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from simulation.environment import WeatherCondition
from scenarios.village import VillageScenario
from scenarios.intersection import IntersectionScenario
from scenarios.highway_merge import HighwayMergeScenario
from scenarios.market import MarketScenario
from scenarios.cattle_crossing import CattleCrossingScenario
from metrics.evaluator import MetricsEvaluator
from visualization.renderer import TopDownRenderer
from planning.adaptive_planner import PlannerConfig

# Page Configuration
st.set_page_config(
    page_title="Autonomous Driving on Unstructured Indian Roads (SIH)",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark-mode modern dashboard styling
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e222d; padding: 12px; border-radius: 8px; border: 1px solid #2e3440; }
    .status-badge { padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 14px; }
    .risk-low { background-color: #2ecc71; color: white; }
    .risk-med { background-color: #f1c40f; color: black; }
    .risk-high { background-color: #e67e22; color: white; }
    .risk-crit { background-color: #e74c3c; color: white; }
</style>
""", unsafe_allow_html=True)

# Scenario registry mapping
SCENARIO_MAP = {
    "1. Unmarked Village Road": ("village", VillageScenario, "Narrow rural bottleneck (5.6m -> 4.8m), no lane markings, surface potholes, slow pushcart, and oncoming motorcycle."),
    "2. Busy Unsignalized Urban Intersection": ("intersection", IntersectionScenario, "Uncontrolled 4-way intersection with crossing auto-rickshaws, turning bike, and crossing pedestrians."),
    "3. Highway Merge": ("highway_merge", HighwayMergeScenario, "High-speed expressway cruising with slow commercial truck ahead and merging passenger car."),
    "4. Dense Market Area": ("market", MarketScenario, "Crowded bazaar with parked delivery van, vendors, weaving auto-rickshaw, and filtering motorcycles."),
    "5. Sudden Cattle Crossing": ("cattle_crossing", CattleCrossingScenario, "Cattle on shoulder abruptly steps into the driving path and freezes in the lane."),
}

# Sidebar Controls
st.sidebar.title("🚗 Navigation & Controls")
st.sidebar.markdown("**Smart India Hackathon Prototype**")

selected_scenario_name = st.sidebar.selectbox(
    "Select Scenario",
    list(SCENARIO_MAP.keys()),
    index=0
)
scenario_key, scenario_class, scenario_desc = SCENARIO_MAP[selected_scenario_name]

planner_choice = st.sidebar.radio(
    "Planner Algorithm",
    ["🚀 Adaptive Planner (Ours)", "⚠️ Baseline Planner (Naive)"],
    index=0
)
use_baseline = "Baseline" in planner_choice

weather_choice = st.sidebar.selectbox(
    "Environmental Weather",
    ["normal", "rain", "fog", "night"],
    index=0
)
weather_cond = WeatherCondition(weather_choice)

seed = st.sidebar.number_input("Deterministic Random Seed", min_value=1, max_value=9999, value=42)

# Advanced Planner Tuning
with st.sidebar.expander("⚙️ Advanced Planner Tuning", expanded=False):
    st.caption("Fine-tune multi-objective cost optimization weights in real time:")
    cruise_speed = st.slider("Target Cruise Speed (m/s)", min_value=5.0, max_value=20.0, value=12.0, step=0.5)
    w_collision = st.slider("Collision Weight (w_col)", min_value=1000.0, max_value=20000.0, value=10000.0, step=1000.0)
    w_boundary = st.slider("Boundary Weight (w_bound)", min_value=1000.0, max_value=10000.0, value=5000.0, step=500.0)
    w_uncertainty = st.slider("Actor Uncertainty Weight (w_unc)", min_value=20.0, max_value=500.0, value=150.0, step=10.0)
    w_pothole = st.slider("Pothole Penalty (w_pothole)", min_value=50.0, max_value=1000.0, value=300.0, step=50.0)

st.sidebar.markdown("---")
st.sidebar.info(f"**Scenario Challenge:**\n{scenario_desc}")

# Caching simulation runs for snappy interactivity
@st.cache_resource(show_spinner=False)
def get_simulation_run(
    scen_key: str,
    is_baseline: bool,
    weather_val: str,
    seed_val: int,
    v_cruise: float = 12.0,
    w_col: float = 10000.0,
    w_bound: float = 5000.0,
    w_unc: float = 150.0,
    w_pot: float = 300.0
):
    cls = SCENARIO_MAP[
        [k for k, v in SCENARIO_MAP.items() if v[0] == scen_key][0]
    ][1]
    scenario = cls(
        weather=WeatherCondition(weather_val),
        use_baseline_planner=is_baseline,
        random_seed=seed_val
    )
    if not is_baseline:
        scenario.planner.config = PlannerConfig(
            target_cruise_speed=v_cruise,
            w_collision=w_col,
            w_boundary=w_bound,
            w_uncertainty=w_unc,
            w_pothole=w_pot
        )
    scenario.run()
    metrics = MetricsEvaluator.evaluate(scenario)
    return scenario, metrics

# Main Header
st.title("Autonomous Driving on Unstructured Indian Roads")
st.markdown("##### Adaptive Path Planning & Collision Avoidance for Heterogeneous, Lane-Free Traffic Environments")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🛰️ Top-Down Simulation & Telemetry",
    "📊 Quantitative Benchmark Comparison",
    "🧠 Multi-Sensor Perception & Fusion",
    "📐 Mathematical Formulation & Defense"
])

# Run or retrieve scenario
with st.spinner("Running closed-loop simulation pipeline..."):
    scenario, metrics = get_simulation_run(
        scenario_key, use_baseline, weather_choice, seed,
        cruise_speed, w_collision, w_boundary, w_uncertainty, w_pothole
    )

planner_prefix = "baseline" if use_baseline else "adaptive"
gif_path = os.path.join("output", f"{scenario_key}_{planner_prefix}_{weather_choice}_animation.gif")

# ==============================================================================
# TAB 1: TOP-DOWN SIMULATION & TELEMETRY
# ==============================================================================
with tab1:
    col_sim, col_metrics = st.columns([3, 1])

    with col_metrics:
        st.subheader("Live Status")
        
        status_color = "green" if scenario.is_success else "red"
        st.markdown(f"**Outcome:** :{status_color}[**{scenario.termination_reason}**]")
        
        # Frame slider for interactive replay
        total_frames = len(scenario.telemetry)
        if total_frames > 0:
            frame_idx = st.slider(
                "Simulation Step Replay",
                min_value=0,
                max_value=total_frames - 1,
                value=total_frames - 1,
                step=1,
                help="Scrub through the simulation timeline to observe real-time replanning and actor interactions."
            )
            current_snap = scenario.telemetry[frame_idx]
        else:
            frame_idx = 0
            current_snap = None

        if current_snap:
            st.metric("Ego Speed", f"{current_snap.ego_v * 3.6:.1f} km/h")
            st.metric("Min Obstacle Clearance", f"{current_snap.min_clearance:.2f} m")
            
            ttc_display = f"{current_snap.min_ttc:.2f} s" if current_snap.min_ttc < 90 else "CLEAR"
            st.metric("Time-to-Collision (TTC)", ttc_display)

            # Risk badge
            risk_class = {
                "LOW": "risk-low",
                "MEDIUM": "risk-med",
                "HIGH": "risk-high",
                "CRITICAL": "risk-crit"
            }.get(current_snap.risk_level, "risk-low")
            st.markdown(f"**Risk Level:** <span class='status-badge {risk_class}'>{current_snap.risk_level}</span>", unsafe_allow_html=True)
            
            st.markdown(f"**Planner State:** `{current_snap.planner_state}`")
            if current_snap.active_trajectory:
                st.markdown(f"**Maneuver:** `{current_snap.active_trajectory.maneuver_name}`")
            st.markdown(f"**Simulation Time:** `{current_snap.time:.2f} s`")
            st.markdown(f"**Total Replans:** `{getattr(scenario.planner, 'replanning_count', 0)}`")

    with col_sim:
        st.subheader(f"{scenario.name} — Top-Down View")
        
        view_mode = st.radio(
            "Visualization Mode",
            ["🖼️ Step-by-Step Frame Scrubbing", "🎬 Full Motion Video / Animated Replay (GIF)"],
            horizontal=True
        )

        if view_mode == "🖼️ Step-by-Step Frame Scrubbing":
            if current_snap:
                renderer = TopDownRenderer(figsize=(12, 6.5))
                fig = renderer.render_snapshot(scenario, frame_idx=frame_idx, show_plot=False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        else:
            if os.path.exists(gif_path):
                st.image(gif_path, caption=f"Continuous 15 FPS Simulation Replay — {scenario.name} ({planner_choice})", use_container_width=True)
                with open(gif_path, "rb") as f:
                    st.download_button(
                        "📥 Download Animated Replay (GIF)",
                        data=f,
                        file_name=f"{scenario_key}_{planner_prefix}_{weather_choice}_animation.gif",
                        mime="image/gif"
                    )
            else:
                st.info("Animated replay GIF is not yet rendered for this scenario/planner configuration.")
                if st.button("🎬 Render High-Resolution Replay GIF Now"):
                    with st.spinner("Rendering scenario frames into smooth GIF (takes ~20s)..."):
                        os.makedirs("output", exist_ok=True)
                        renderer = TopDownRenderer(figsize=(12, 6.5))
                        renderer.save_animation_gif(scenario, output_path=gif_path, fps=15)
                        st.success("Replay generated successfully!")
                        st.rerun()

    # Telemetry Line Chart & CSV Download
    if len(scenario.telemetry) > 0:
        st.subheader("Vehicle Dynamics & Trajectory Profiles")
        t_data = pd.DataFrame({
            "Time (s)": [t.time for t in scenario.telemetry],
            "Speed (km/h)": [t.ego_v * 3.6 for t in scenario.telemetry],
            "Acceleration (m/s²)": [t.ego_a for t in scenario.telemetry],
            "Clearance (m)": [t.min_clearance for t in scenario.telemetry],
            "Steer Angle (deg)": [np.rad2deg(t.ego_delta) for t in scenario.telemetry]
        }).set_index("Time (s)")

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.line_chart(t_data[["Speed (km/h)", "Acceleration (m/s²)"]])
        with col_c2:
            st.line_chart(t_data[["Clearance (m)", "Steer Angle (deg)"]])

        st.download_button(
            "📥 Download Telemetry Profiles (CSV)",
            data=t_data.to_csv().encode('utf-8'),
            file_name=f"{scenario_key}_{planner_prefix}_telemetry.csv",
            mime="text/csv"
        )


# ==============================================================================
# TAB 2: QUANTITATIVE BENCHMARK COMPARISON
# ==============================================================================
with tab2:
    st.subheader("Quantitative Benchmark: Baseline Planner vs Our Adaptive Planner")
    st.markdown("""
    To rigorously demonstrate value for the Smart India Hackathon jury, each scenario is evaluated
    against a standard **Baseline Planner** (traditional lane-follower with naive reactive stop).
    """)

    # Run both planners for the current scenario
    scen_base, m_base = get_simulation_run(scenario_key, True, weather_choice, seed)
    scen_adapt, m_adapt = get_simulation_run(
        scenario_key, False, weather_choice, seed,
        cruise_speed, w_collision, w_boundary, w_uncertainty, w_pothole
    )

    comp_df = MetricsEvaluator.compare(m_base, m_adapt)
    st.dataframe(comp_df, use_container_width=True)

    # KPI Comparisons
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.metric("Collisions", f"{m_adapt.collision_count}", delta=f"{m_adapt.collision_count - m_base.collision_count} vs Baseline", delta_color="inverse")
    with col_k2:
        st.metric("Near Misses (<1m)", f"{m_adapt.near_miss_count}", delta=f"{m_adapt.near_miss_count - m_base.near_miss_count} vs Baseline", delta_color="inverse")
    with col_k3:
        st.metric("Min Clearance", f"{m_adapt.minimum_clearance_m:.2f} m", delta=f"{m_adapt.minimum_clearance_m - m_base.minimum_clearance_m:+.2f} m")
    with col_k4:
        st.metric("Avg Replanning Latency", f"{m_adapt.avg_replanning_latency_ms:.1f} ms", help="Automotive real-time requirement is < 100ms for a 10 Hz loop.")

    # Side-by-side Visual Comparison Charts
    st.markdown("#### 📊 Comparative Visual Metrics")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("**Minimum Clearance to Obstacles (m)** *(Higher is safer)*")
        chart_df1 = pd.DataFrame({
            "Planner": ["Baseline (Naive)", "Adaptive (Ours)"],
            "Min Clearance (m)": [m_base.minimum_clearance_m, m_adapt.minimum_clearance_m]
        }).set_index("Planner")
        st.bar_chart(chart_df1)
    with col_g2:
        st.markdown("**Safety Incidents & Conflicts** *(Lower is safer)*")
        chart_df2 = pd.DataFrame({
            "Baseline": [m_base.collision_count, m_base.near_miss_count],
            "Adaptive": [m_adapt.collision_count, m_adapt.near_miss_count]
        }, index=["Collisions", "Near Misses (<1m)"])
        st.bar_chart(chart_df2)

    # All-Scenario Comparative Summary Table
    st.markdown("### 🏆 Overall Multi-Scenario Performance Summary")
    summary_csv_path = "output/benchmark_comparison_summary.csv"
    if os.path.exists(summary_csv_path):
        all_summary_df = pd.read_csv(summary_csv_path)
        st.dataframe(all_summary_df, use_container_width=True)
        
        with open(summary_csv_path, "rb") as f:
            st.download_button(
                "📥 Download Official SIH Benchmark CSV",
                data=f,
                file_name="sih_benchmark_comparison_summary.csv",
                mime="text/csv"
            )


# ==============================================================================
# TAB 3: MULTI-SENSOR PERCEPTION & FUSION
# ==============================================================================
with tab3:
    st.subheader("Multi-Modal Sensor Suite & State Estimation")
    st.markdown("""
    The prototype simulates four distinct physical sensor packages in software with realistic noise,
    FOV constraints, and weather degradation:
    """)

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.markdown("#### 📷 Forward Camera")
        st.markdown("- **FOV:** 90° | **Range:** 65m\n- **Role:** High-accuracy semantic classification (Animal vs Rickshaw vs Car)\n- **Sensitivity:** Rain/Fog reduces confidence.")
    with col_s2:
        st.markdown("#### 📡 Doppler Radar")
        st.markdown("- **FOV:** 40° | **Range:** 90m\n- **Role:** Direct relative radial velocity (range-rate)\n- **Sensitivity:** All-weather resilience.")
    with col_s3:
        st.markdown("#### 🔘 360° LiDAR")
        st.markdown(r"- **Rays:** 180 | **Range:** 60m" + "\n" + r"- **Role:** High-precision metric obstacle geometry & road edges" + "\n" + r"- **Accuracy:** $\pm 0.05$m spatial variance.")
    with col_s4:
        st.markdown("#### 📍 GNSS / IMU")
        st.markdown(r"- **Role:** Ego vehicle global pose $(x, y, \psi, v, a)$" + "\n" + r"- **Rule:** Used solely for ego localization; **NEVER** for obstacle detection.")

    # Tracked objects table at current frame
    if current_snap and len(current_snap.tracked_objects) > 0:
        st.subheader("Live Kalman Filter Tracked Objects (Current Step)")
        track_records = []
        for trk in current_snap.tracked_objects:
            track_records.append({
                "Track ID": trk.track_id,
                "Type": trk.actor_type.value.upper(),
                "World X (m)": f"{trk.x:.2f}",
                "World Y (m)": f"{trk.y:.2f}",
                "Speed (km/h)": f"{trk.speed * 3.6:.1f}",
                "Heading (deg)": f"{np.rad2deg(trk.heading):.1f}°",
                "Dimensions (L x W)": f"{trk.length:.1f}m x {trk.width:.1f}m",
                "Uncertainty": f"{trk.uncertainty:.2f}",
                "Status": trk.state.value.upper(),
                "Sensors": ", ".join(trk.sources)
            })
        st.dataframe(pd.DataFrame(track_records), use_container_width=True)


# ==============================================================================
# TAB 4: MATHEMATICAL FORMULATION & DEFENSE
# ==============================================================================
with tab4:
    st.subheader("Technical & Mathematical Architecture for SIH Evaluation")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("### 1. Kinematic Bicycle Model")
        st.latex(r"\dot{x} = v \cos\psi, \quad \dot{y} = v \sin\psi")
        st.latex(r"\dot{\psi} = \frac{v}{L} \tan\delta, \quad \dot{v} = a")
        st.markdown("Guarantees that the vehicle follows physically realizable non-holonomic trajectories with acceleration and steering rate limits.")

        st.markdown("### 2. Frenet Quintic Polynomial Lattice")
        st.latex(r"d(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5")
        st.markdown(r"Satisfies initial boundary conditions $[d_0, \dot{d}_0, \ddot{d}_0]$ and terminal constraints $[d_f, \dot{d}_f = 0, \ddot{d}_f = 0]$ for continuous jerk and curvature.")

    with col_m2:
        st.markdown("### 3. Multi-Factor Risk & CPA")
        st.latex(r"R_{\text{composite}} = \max\left(e^{-\frac{d - d_0}{\lambda_d}}, e^{-\frac{\text{TTC}}{\tau_0}}, e^{-\frac{d_{\text{CPA}}}{\lambda_{\text{cpa}}}}\right) \times (1 + \alpha U_{\text{actor}})")
        st.markdown("Incorporates spatial closest point of approach (CPA) along predicted trajectories and intrinsic behavioral uncertainty (Cattle: $U=0.95$, Pedestrians: $U=0.75$).")

        st.markdown("### 4. Multi-Objective Cost Optimization")
        st.latex(r"J(\tau) = w_{\text{col}} J_{\text{col}} + w_{\text{bound}} J_{\text{bound}} + w_{\text{pothole}} J_{\text{pothole}} + w_{\text{unc}} J_{\text{unc}} + w_{\text{smooth}} J_{\text{smooth}} + w_{\text{prog}} J_{\text{prog}}")
        st.markdown("Selects the safest feasible candidate trajectory, automatically engaging controlled emergency deceleration if road boundaries or obstacles block all lateral evasions.")

    st.markdown("---")
    st.markdown("### 5. Production Automotive Roadmap: Mapping Python Prototype to Industry Standards")
    roadmap_df = pd.DataFrame([
        {"Python Prototype Module": "simulation/vehicle.py", "MATLAB / Simulink Equivalent": "Vehicle Dynamics Blockset (Kinematic/Dynamic Bicycle)", "RoadRunner / CARLA Equivalent": "Ego Vehicle Actor Asset"},
        {"Python Prototype Module": "simulation/road.py", "MATLAB / Simulink Equivalent": "Automated Driving Toolbox (drivingScenario, roadNetwork)", "RoadRunner / CARLA Equivalent": "RoadRunner Road Asset (.rrdata / OpenDRIVE)"},
        {"Python Prototype Module": "sensors/ (camera, radar, lidar)", "MATLAB / Simulink Equivalent": "Sensor Simulation Toolbox (visionDetectionGenerator, radarDataGenerator)", "RoadRunner / CARLA Equivalent": "RoadRunner / Unreal Engine Sensor Bridges"},
        {"Python Prototype Module": "tracking/tracker.py", "MATLAB / Simulink Equivalent": "Automated Driving Toolbox (trackerJPDA, trackerGNN)", "RoadRunner / CARLA Equivalent": "ROS 2 / Apollo Tracking Node"},
        {"Python Prototype Module": "planning/adaptive_planner.py", "MATLAB / Simulink Equivalent": "Navigation Toolbox (trajectoryOptimalFrenet, Stateflow)", "RoadRunner / CARLA Equivalent": "Apollo Lattice Planner / Autoware Planner"},
        {"Python Prototype Module": "control/vehicle_controller.py", "MATLAB / Simulink Equivalent": "Simulink Control Design (Stanley / Pure Pursuit / MPC)", "RoadRunner / CARLA Equivalent": "Low-Level CAN / Drive-by-Wire Bridge"},
    ])
    st.dataframe(roadmap_df, use_container_width=True)
