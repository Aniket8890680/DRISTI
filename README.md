# Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles on Unstructured Indian Roads
**Smart India Hackathon (SIH) — Software & Simulation Prototype**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)
![Tests](https://img.shields.io/badge/tests-23%2F23%20passing-success.svg)
![Status](https://img.shields.io/badge/SIH-Ready-orange.svg)

---

## 1. Executive Summary & Problem Context

Most traditional autonomous-driving systems (e.g., Autoware, Apollo) are designed around structured Western roadways featuring crisp lane markings, standardized signage, uniform vehicle dynamics, and predictable driver behavior. 

**Indian road environments present radically different challenges:**
* **Absence of Predefined Lanes:** Lane markings are frequently faded, missing, or disregarded.
* **Extreme Heterogeneity of Road Participants:** Mixed traffic includes passenger cars, heavy trucks, buses, three-wheeled auto-rickshaws, high-agility motorcycles, pushcart vendors, crossing pedestrians, and free-roaming cattle/animals.
* **Informal & Dynamic Driving Behaviors:** Two-wheeler filtering between lanes, sudden lateral cut-ins, informal merging from dirt shoulders, wrong-side driving, and stationary road surface defects (deep potholes, construction zones).

### The Core Solution
This project implements a **100% software-based, closed-loop simulation prototype** capable of navigating unstructured roads without relying on lane markers. At each timestep ($dt = 0.05\text{s}$), the system executes:

$$\text{Simulated Environment} \longrightarrow \text{Sensors} \longrightarrow \text{Sensor Fusion} \longrightarrow \text{EKF Tracking} \longrightarrow \text{Motion Prediction} \longrightarrow \text{Risk Assessment} \longrightarrow \text{Drivable Corridor} \longrightarrow \text{Frenet Lattice Planning} \longrightarrow \text{Vehicle Control} \longrightarrow \text{Closed Loop Feedback}$$

---

## 2. End-to-End System Architecture

```
                                  CLOSED-LOOP PIPELINE
                                  
+-----------------------------------------------------------------------------------+
|                            SIMULATED ENVIRONMENT (2D)                             |
|  - Road Geometry (Frenet Frame)        - Mixed Traffic Actors (Bicycle, Cattle..) |
|  - Road Boundaries & Dirt Shoulders    - Surface Defects (Potholes, Debris)       |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                        SIMULATED MULTI-MODAL SENSORS                              |
|  - Forward Camera (FOV 90°, Classification, Visual Noise, Weather Attenuation)    |
|  - Long-Range Doppler Radar (FOV 40°, Direct Relative Radial Velocity, RCS)      |
|  - 360° LiDAR (180 Rays, High-Precision Metric Obstacle Clusters & Bounds)        |
|  - GNSS/IMU (Ego Pose x, y, Heading psi, Speed v, Acceleration a)                 |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                     PERCEPTION & MULTI-SENSOR FUSION                              |
|  - Spatial Gating & Multi-Sensor Clustering                                       |
|  - Weighted Covariance Fusion (Camera Classification + Radar Doppler + LiDAR Size)|
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                     MULTI-TARGET TRACKING (KALMAN FILTER)                         |
|  - Constant-Velocity (CV) State Estimation: x = [pos_x, pos_y, vel_x, vel_y]^T     |
|  - Greedy Nearest-Neighbor Association & Track Lifecycles (Tentative -> Confirmed)|
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                     SHORT-TERM MOTION PREDICTION (1.0s - 3.0s)                    |
|  - Kinematic & Behavioral Modes (Cut-In, Motorcycle Filtering, Animal Wander)     |
|  - Expanding Spatial Uncertainty Envelopes: sigma(t) = sigma_0 + alpha * t        |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                         MULTI-FACTOR RISK ASSESSMENT                              |
|  - Time-to-Collision: TTC = dist / closing_speed                                  |
|  - Spatial Closest Point of Approach (CPA) along predicted trajectories           |
|  - Actor Uncertainty Multiplier (Cattle/Pedestrian > Rickshaw > Car > Static)     |
|  - Risk Classification: LOW | MEDIUM | HIGH | CRITICAL                            |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                  LANE-INDEPENDENT DRIVABLE CORRIDOR ESTIMATION                    |
|  - Free-space boundary extraction: [d_min(s), d_max(s)]                           |
|  - Pothole & Construction Avoidance Envelopes                                     |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                     ADAPTIVE FRENET PATH PLANNER (CORE)                           |
|  - Candidate Generation: Quintic Polynomial Splines over Diverse Lateral Offsets  |
|  - Multi-Objective Cost Function: Collision + Boundary + Smoothness + Uncertainty |
|  - Automatic Emergency Controlled Braking Fallback (prevents ditch swerving)      |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                         VEHICLE ACTUATION CONTROLLER                              |
|  - Lateral Stanley Path Tracking: delta = e_psi + arctan(k * e_lat / (v + k_soft)) |
|  - Longitudinal PID Speed Tracking: a_cmd = FF + Kp * e_v + Ki * int(e_v) + Kd * de|
+------------------------------------------+----------------------------------------+
                                           |
                                           v
                                [Simulated Ego Vehicle]
                                           |
                                           +---> Closed-loop update to Environment
```

---

## 3. Mathematical Formulations & Algorithms

### 3.1 Kinematic Bicycle Model
The autonomous vehicle dynamics are governed by a continuous-time rear-axle Kinematic Bicycle Model:

$$\dot{x} = v \cos(\psi), \quad \dot{y} = v \sin(\psi)$$

$$\dot{\psi} = \frac{v}{L} \tan(\delta), \quad \dot{v} = a$$

$$\dot{\delta} = \text{clip}\left(\frac{\delta_{\text{target}} - \delta}{\tau}, -\dot{\delta}_{\max}, \dot{\delta}_{\max}\right)$$

Where $L = 2.7\text{m}$ is the wheelbase, $\psi$ is the heading angle, $\delta$ is front-wheel steer angle, and $a$ is longitudinal acceleration constrained by physical limits:
* $v \in [0.0, 25.0]\text{ m/s}$ ($90\text{ km/h}$)
* $a \in [-6.0, +3.0]\text{ m/s}^2$
* $\delta \in [-36^\circ, +36^\circ]$, $\dot{\delta} \le 45^\circ/\text{s}$

### 3.2 Oriented Bounding Box Collision via Separating Axis Theorem (SAT)
Collision detection does not make naive circular approximations. Two oriented bounding boxes $A$ and $B$ intersect if and only if their 1D projections overlap across all 4 edge-normal axes:

$$\text{Project}(A, \mathbf{u}_i) = \left[ \min_{p \in A} (p \cdot \mathbf{u}_i), \max_{p \in A} (p \cdot \mathbf{u}_i) \right]$$

If $\max(\text{Proj}_A) < \min(\text{Proj}_B)$ or $\max(\text{Proj}_B) < \min(\text{Proj}_A)$ for ANY normal axis $\mathbf{u}_i$, the polygons are completely separated.

### 3.3 Multi-Target Tracking via 2D Kalman Filter
Each traffic participant is tracked using a Constant Velocity state model:

$$\mathbf{x}_k = \begin{bmatrix} x \\ y \\ v_x \\ v_y \end{bmatrix}, \quad \mathbf{F} = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

$$\mathbf{P}_{k|k-1} = \mathbf{F}\mathbf{P}_{k-1|k-1}\mathbf{F}^T + \mathbf{Q}$$

$$\mathbf{K}_k = \mathbf{P}_{k|k-1}\mathbf{H}^T \left( \mathbf{H}\mathbf{P}_{k|k-1}\mathbf{H}^T + \mathbf{R} \right)^{-1}$$

$$\mathbf{x}_{k|k} = \mathbf{x}_{k|k-1} + \mathbf{K}_k \left( \mathbf{z}_k - \mathbf{H}\mathbf{x}_{k|k-1} \right)$$

### 3.4 Multi-Factor Risk Assessment
Rather than relying solely on scalar TTC (which fails when trajectories diverge or cross at angles), our risk engine computes a composite hazard index:

$$v_{\text{closing}} = -\frac{(\mathbf{p}_{\text{actor}} - \mathbf{p}_{\text{ego}}) \cdot (\mathbf{v}_{\text{actor}} - \mathbf{v}_{\text{ego}})}{\|\mathbf{p}_{\text{actor}} - \mathbf{p}_{\text{ego}}\|}$$

$$\text{TTC} = \frac{d}{v_{\text{closing}}} \quad (\text{if } v_{\text{closing}} > 0.3\text{ m/s})$$

$$d_{\text{CPA}} = \min_{t \in [0, T]} \|\mathbf{p}_{\text{ego}}(t) - \mathbf{p}_{\text{actor}}(t)\|$$

$$R_{\text{composite}} = \max\left(e^{-\frac{d - d_0}{\lambda_d}}, e^{-\frac{\text{TTC}}{\tau_0}}, e^{-\frac{d_{\text{CPA}}}{\lambda_{\text{cpa}}}}\right) \times (1 + \alpha \cdot U_{\text{actor}})$$

Where $U_{\text{actor}}$ represents the intrinsic behavioral unpredictability rating:
* **Cattle / Animals:** $U = 0.95$ (High freeze/random walk probability)
* **Pedestrians:** $U = 0.75$
* **Motorcycles:** $U = 0.60$
* **Auto-Rickshaws:** $U = 0.45$
* **Cars / Trucks:** $U = 0.25$
* **Static Obstacles:** $U = 0.05$

### 3.5 Candidate Trajectory Generation in Frenet Frame
Trajectories are parameterized using 1D Quintic Polynomials in Frenet longitudinal $s(t)$ and lateral $d(t)$ coordinates:

$$d(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 + a_4 t^4 + a_5 t^5$$

Boundary conditions at $t=0$: $[d_0, \dot{d}_0, \ddot{d}_0]$ (current state).  
Boundary conditions at $t=T$: $[d_f, \dot{d}_f = 0, \ddot{d}_f = 0]$ (target lateral offset).

### 3.6 Multi-Objective Trajectory Cost Function
Each candidate trajectory $\tau$ is scored against 8 competing objectives:

$$J(\tau) = w_{\text{col}} J_{\text{col}} + w_{\text{bound}} J_{\text{bound}} + w_{\text{pothole}} J_{\text{pothole}} + w_{\text{unc}} J_{\text{unc}} + w_{\text{smooth}} J_{\text{smooth}} + w_{\text{speed}} J_{\text{speed}} + w_{\text{progress}} J_{\text{progress}} + w_{\text{center}} J_{\text{center}}$$

* $J_{\text{col}}$: Hard collision penalty ($+\infty$ / $10,000$) if vehicle bounding box intersects any dynamic actor or its expanding uncertainty ellipse.
* $J_{\text{bound}}$: Heavy penalty for exceeding drivable road boundaries beyond shoulder tolerance.
* $J_{\text{pothole}}$: Penalty for running tires over potholes weighted by pothole severity.
* $J_{\text{unc}}$: Proximity penalty scaled by actor uncertainty (repels the planner away from cattle and darting pedestrians).
* $J_{\text{smooth}}$: Integral of curvature squared $\int \kappa(t)^2 dt$ (penalizes erratic swerves).
* $J_{\text{progress}}$: Negative reward for forward travel along road arc-length ($-w \cdot \Delta s$).
* $J_{\text{speed}}$: Deviation from target speed $(v(t) - v_{\text{target}})^2$.

---

## 4. Benchmark Scenarios & Empirical Results

The simulation includes the **five mandatory SIH scenarios** tested under identical environmental conditions with real simulation runs:

### Scenario 1: Unmarked Village Road
* **Road Conditions:** Narrow 5.6m road narrowing to a 4.8m bottleneck, no painted lines, dirt shoulders, surface potholes.
* **Traffic:** Slow pushcart along right edge, oncoming motorcycle at 7.0 m/s in left lane.
* **Adaptive Behavior:** Planner detects oncoming motorcycle corridor, yields in-lane behind pothole/pushcart until the motorcycle passes, then executes a smooth overtaking maneuver without ditch departure.

### Scenario 2: Busy Unsignalized Urban Intersection
* **Road Conditions:** Uncontrolled 4-way urban crossing without traffic signals or stop signs.
* **Traffic:** Perpendicular crossing auto-rickshaw (5.5 m/s), turning motorcycle cutting across, and an informal pedestrian crossing.
* **Adaptive Behavior:** Planner forecasts crossing trajectories, computes spatial CPA, yields for the crossing auto-rickshaw, and accelerates smoothly through the safe opening.

### Scenario 3: Highway Merge
* **Road Conditions:** Wide expressway corridor with fast 16.0 m/s cruise speed.
* **Traffic:** Slow overloaded commercial truck (7.0 m/s) ahead in lane, passenger car merging informally from left on-ramp.
* **Adaptive Behavior:** Planner processes Doppler velocity, identifies closing speed on the truck, evaluates merging car trajectory, and executes a smooth lane change into the open corridor.

### Scenario 4: Dense Market Area
* **Road Conditions:** Crowded 6.4m market street with encroaching parked delivery vehicle.
* **Traffic:** Pushcart vendor, weaving auto-rickshaw, filtering motorcycle, and mid-block crossing pedestrian.
* **Adaptive Behavior:** Planner continuously replans at 10 Hz, crawls through narrow 0.84m clearance gaps, avoids the parked car without emergency deadlock, and safely reaches the goal.

### Scenario 5: Sudden Cattle Crossing
* **Road Conditions:** Rural roadway with high-speed 11.0 m/s cruising.
* **Traffic:** Cattle grazing on dirt shoulder suddenly enters the road at $t=1.0\text{s}$ and freezes in the center of the lane.
* **Adaptive Behavior:** Perception tags cattle with $U=0.95$ uncertainty; risk engine spikes to CRITICAL; planner assesses oncoming lane clearance and executes a safe evasive swerve with 1.92m clearance.

---

## 5. Quantitative Baseline Comparison (Actual Run Data)

Below is the side-by-side performance comparison between a standard **Baseline Planner** (centerline follower with naive reactive stop) and our **Adaptive Path Planner**:

| Scenario | Baseline Collisions | Adaptive Collisions | Baseline Status | Adaptive Status | Baseline Min Clearance | Adaptive Min Clearance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Unmarked Village Road** | 0 | **0** | `FAILED [OFF_ROAD]` | **`SUCCESS [PASSED]`** | 1.97 m | **0.39 m** (Bottleneck) |
| **2. Unsignalized Intersection** | 0 | **0** | `SUCCESS [PASSED]` | **`SUCCESS [PASSED]`** | 2.51 m | **5.54 m** |
| **3. Highway Merge** | 0 | **0** | `FAILED [OFF_ROAD]` | **`SUCCESS [PASSED]`** | 16.95 m | **1.84 m** (Safe Pass) |
| **4. Dense Market Area** | **1** | **0** | `FAILED [COLLISION]` | **`SUCCESS [PASSED]`** | 0.00 m (Crash) | **0.84 m** (Navigated) |
| **5. Sudden Cattle Crossing** | 0 | **0** | `FAILED [OFF_ROAD]` | **`SUCCESS [PASSED]`** | 19.22 m | **1.92 m** (Safe Swerve) |

### Key Benchmark Findings:
1. **Collision Elimination:** Baseline crashed into an obstacle in the Dense Market scenario (1 collision); our Adaptive Planner achieved **0 collisions (100% collision-free)** across all scenarios.
2. **Off-Road Prevention:** In 3 out of 5 scenarios, the Baseline planner departed the road into ditches (`OFF_ROAD`) because it lacked drivable-area estimation and Frenet boundary constraints. Our planner completed **5 out of 5 scenarios successfully**.
3. **Replanning Latency:** Average replanning cycle latency is **~50–110 ms**, fully compatible with a real-time 10 Hz automotive loop.

---

## 6. Project Structure

```
autonomous/
├── README.md                          # Comprehensive documentation & mathematical specification
├── requirements.txt                   # Dependency manifest (numpy, scipy, matplotlib, pandas, pyyaml)
├── main.py                            # CLI entry point for execution, benchmarks, and rendering
│
├── config/
│   └── scenarios.yaml                 # Scenario dimensions, vehicle limits, and planner weights
│
├── simulation/
│   ├── vehicle.py                     # Kinematic bicycle model & state tracking
│   ├── road.py                        # Parametric Frenet road geometry, boundaries, and potholes
│   ├── actors.py                      # Heterogeneous Indian traffic participants & SAT collision
│   └── environment.py                 # Time-synchronous closed-loop simulation environment
│
├── sensors/
│   ├── camera.py                      # Simulated forward camera with FOV & weather attenuation
│   ├── radar.py                       # Automotive Doppler radar with direct radial velocity & RCS
│   ├── lidar.py                       # Multi-beam range-finding LiDAR with geometric clustering
│   └── gps_imu.py                     # Localization package (ego state only; never used for obstacles)
│
├── perception/
│   ├── detector.py                    # Standardized DetectionPacket interface
│   └── sensor_fusion.py               # Weighted multi-modal sensor fusion engine
│
├── tracking/
│   └── tracker.py                     # Multi-target 2D Kalman Filter & track lifecycle manager
│
├── prediction/
│   └── trajectory_predictor.py        # Behavioral trajectory predictor with expanding uncertainty
│
├── planning/
│   ├── risk.py                        # Multi-factor risk assessor (TTC, CPA, uncertainty weight)
│   ├── drivable_area.py               # Lane-independent free-space drivable corridor estimator
│   ├── candidate_paths.py             # Frenet quintic polynomial candidate trajectory generator
│   ├── adaptive_planner.py            # Core multi-objective cost optimization planner
│   └── baseline_planner.py            # Naive benchmark comparison planner
│
├── control/
│   └── vehicle_controller.py          # Stanley path tracking + longitudinal PID controller
│
├── scenarios/
│   ├── scenario_base.py               # Abstract scenario pipeline & telemetry logger
│   ├── village.py                     # Scenario 1: Unmarked Village Road
│   ├── intersection.py                # Scenario 2: Busy Unsignalized Urban Intersection
│   ├── highway_merge.py               # Scenario 3: Highway Merge
│   ├── market.py                      # Scenario 4: Dense Market Area
│   └── cattle_crossing.py             # Scenario 5: Sudden Cattle Crossing
│
├── metrics/
│   └── evaluator.py                   # Quantitative SIH metric calculations & report generator
│
├── visualization/
│   └── renderer.py                    # High-resolution top-down renderer, HUD, and GIF exporter
│
├── tests/
│   ├── test_vehicle.py                # Kinematic vehicle model unit tests
│   ├── test_actors_and_collision.py   # SAT collision & actor behavioral tests
│   ├── test_sensors_and_tracking.py   # Multi-sensor fusion & EKF tracking tests
│   ├── test_prediction_and_risk.py    # Motion prediction & TTC risk tests
│   └── test_planner.py                # Trajectory generation & evasion tests
│
└── output/                            # Saved snapshots, comparative CSV tables, and animated GIFs
    ├── benchmark_comparison_summary.csv
    ├── cattle_crossing_adaptive_normal_animation.gif
    └── *.png snapshots
```

---

## 7. Quickstart & How to Run

### Installation
Ensure Python 3.10+ is installed:
```bash
cd autonomous
pip install -r requirements.txt
```

### Run Any Individual Scenario
```bash
# Scenario 1: Unmarked Village Road
python main.py --scenario village --save-plots

# Scenario 2: Busy Unsignalized Urban Intersection
python main.py --scenario intersection --save-plots

# Scenario 3: Highway Merge
python main.py --scenario highway_merge --save-plots

# Scenario 4: Dense Market Area
python main.py --scenario market --save-plots

# Scenario 5: Sudden Cattle Crossing
python main.py --scenario cattle_crossing --save-plots --save-gif
```

### Run All 5 Scenarios in Sequence
```bash
python main.py --scenario all
```

### Run Full Baseline vs Adaptive Benchmark Comparison
```bash
python main.py --compare --save-plots
```
This runs all 5 scenarios under both planners, outputs comparative metrics in the terminal, and saves `output/benchmark_comparison_summary.csv`.

### Test Environmental Weather Conditions
Simulate perception degradation under adverse weather (rain, fog, night):
```bash
python main.py --scenario village --weather fog --save-plots
python main.py --scenario cattle_crossing --weather rain --save-plots
```

### Run Automated Unit Test Suite
```bash
python -m unittest discover -s tests -p "test_*.py"
```
*Expected output: `Ran 23 tests in ~0.2s - OK`.*

---

## 8. Recording a Demonstration Video for SIH

To create a compelling demonstration video for the Smart India Hackathon jury:

1. **Option A — Animated Replay Export (Recommended):**
   Run the scenario with `--save-gif`:
   ```bash
   python main.py --scenario cattle_crossing --save-gif
   ```
   The high-resolution animated replay is saved to `output/cattle_crossing_adaptive_normal_animation.gif`.
2. **Option B — Screen Recording:**
   * Use OBS Studio, Windows Game Bar (`Win + G`), or ShareX.
   * Run `python main.py --compare` to display the real-time terminal output and comparative benchmark metrics.
   * Open the generated snapshots in `output/` showing the top-down road layout, vehicle heading arrows, candidate trajectory fans, and risk badges.

---

## 9. Roadmap: Mapping Python Prototype to MATLAB / Simulink / RoadRunner

This software architecture was explicitly designed to map directly into production automotive toolchains:

| Python Prototype Module | MATLAB / Simulink Equivalent Block | RoadRunner / CARLA Equivalent |
| :--- | :--- | :--- |
| `simulation/vehicle.py` | *Vehicle Dynamics Blockset* (Kinematic / Dynamic Bicycle Model) | Ego Vehicle Actor Asset |
| `simulation/road.py` | *Automated Driving Toolbox* (`drivingScenario`, `roadNetwork`) | RoadRunner Road Asset (.rrdata / OpenDRIVE) |
| `sensors/` (`camera`, `radar`, `lidar`) | *Sensor Simulation Toolbox* (`visionDetectionGenerator`, `radarDataGenerator`) | RoadRunner / Unreal Engine Sensor Bridges |
| `tracking/tracker.py` | *Automated Driving Toolbox* (`trackerJPDA`, `trackerGNN`) | ROS2 / Apollo Tracking Node |
| `planning/adaptive_planner.py` | *Navigation Toolbox* (`trajectoryOptimalFrenet`, Stateflow) | Apollo Lattice Planner / Autoware Planner |
| `control/vehicle_controller.py` | *Simulink Control Design* (Stanley / Pure Pursuit / Model Predictive Control) | Low-Level CAN / Drive-by-Wire Bridge |

---

## 10. Technical Integrity Declaration
* **Closed-Loop Execution:** The vehicle does not follow a predetermined path; every steering and throttle actuation is generated in real time from live sensor detections.
* **Deterministic Verification:** With `--seed 42`, all metrics and behaviors are 100% reproducible.
* **Zero Fabricated Results:** All tabular comparison numbers are dynamically computed by the `MetricsEvaluator` from actual simulation logs.
