"""
High-Performance Simulation & Telemetry API Server:
Built with Starlette + Uvicorn to power the Real-Time Cockpit & 60 FPS Visualizer.
"""

import os
import sys
import json
import socket
import threading
from typing import Dict, Any, Tuple
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, FileResponse, HTMLResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles

from simulation.environment import WeatherCondition
from scenarios.village import VillageScenario
from scenarios.intersection import IntersectionScenario
from scenarios.highway_merge import HighwayMergeScenario
from scenarios.market import MarketScenario
from scenarios.cattle_crossing import CattleCrossingScenario
from simulation.serializer import serialize_scenario_run
from metrics.evaluator import MetricsEvaluator

# Scenario Registry
SCENARIO_MAP = {
    "village": {
        "name": "Unmarked Village Road",
        "class": VillageScenario,
        "description": "Narrow 5.6m road with 4.8m bottleneck, no lane markings, surface potholes, slow pushcart, and oncoming motorcycle.",
        "icon": "🏡"
    },
    "intersection": {
        "name": "Busy Unsignalized Urban Intersection",
        "class": IntersectionScenario,
        "description": "Uncontrolled 4-way intersection with crossing auto-rickshaws, turning bike, and crossing pedestrians.",
        "icon": "🚦"
    },
    "highway_merge": {
        "name": "Highway Merge",
        "class": HighwayMergeScenario,
        "description": "High-speed expressway cruising (16.0 m/s) with slow commercial truck ahead and merging passenger car.",
        "icon": "🛣️"
    },
    "market": {
        "name": "Dense Market Area",
        "class": MarketScenario,
        "description": "Crowded 6.4m bazaar with parked delivery van, pushcart vendor, weaving auto-rickshaw, and filtering motorcycles.",
        "icon": "🏪"
    },
    "cattle_crossing": {
        "name": "Sudden Cattle Crossing",
        "class": CattleCrossingScenario,
        "description": "High-speed rural cruising (11.0 m/s) with cattle suddenly entering the lane from the dirt shoulder and freezing.",
        "icon": "🐄"
    },
}

# In-Memory Cache for instantaneous responses
SIM_CACHE: Dict[Tuple[str, bool, str, int], Dict[str, Any]] = {}


def get_or_run_simulation(scenario_key: str, is_baseline: bool, weather_str: str, seed: int) -> Dict[str, Any]:
    cache_key = (scenario_key, is_baseline, weather_str, seed)
    if cache_key in SIM_CACHE:
        return SIM_CACHE[cache_key]

    scen_info = SCENARIO_MAP.get(scenario_key, SCENARIO_MAP["village"])
    cls = scen_info["class"]

    try:
        weather = WeatherCondition(weather_str)
    except ValueError:
        weather = WeatherCondition.NORMAL

    scenario = cls(
        weather=weather,
        use_baseline_planner=is_baseline,
        random_seed=seed
    )
    scenario.run()
    data = serialize_scenario_run(scenario)
    data["scenario_key"] = scenario_key
    data["planner_mode"] = "Baseline" if is_baseline else "Adaptive"
    data["weather"] = weather.value

    SIM_CACHE[cache_key] = data
    return data


def api_scenarios(request):
    """Returns available scenarios with metadata."""
    scenarios_list = [
        {
            "key": k,
            "name": v["name"],
            "description": v["description"],
            "icon": v["icon"]
        }
        for k, v in SCENARIO_MAP.items()
    ]
    return JSONResponse({"scenarios": scenarios_list})


def api_simulation(request):
    """Executes or retrieves cached simulation run telemetry (threaded)."""
    scen_key = request.query_params.get("scenario", "village")
    is_baseline = request.query_params.get("planner", "adaptive").lower() == "baseline"
    weather = request.query_params.get("weather", "normal").lower()
    seed = int(request.query_params.get("seed", 42))

    data = get_or_run_simulation(scen_key, is_baseline, weather, seed)
    return JSONResponse(data)


def api_benchmark(request):
    """Returns baseline vs adaptive benchmark comparison metrics (threaded)."""
    weather = request.query_params.get("weather", "normal").lower()
    seed = int(request.query_params.get("seed", 42))

    comparison_results = []
    for key in SCENARIO_MAP.keys():
        base_data = get_or_run_simulation(key, True, weather, seed)
        adapt_data = get_or_run_simulation(key, False, weather, seed)
        comparison_results.append({
            "scenario_key": key,
            "scenario_name": SCENARIO_MAP[key]["name"],
            "baseline": base_data["metrics"],
            "adaptive": adapt_data["metrics"],
        })

    return JSONResponse({"benchmarks": comparison_results})


def index(request):
    """Serves the main HTML application."""
    index_path = os.path.join(os.path.dirname(__file__), "web", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>Loading Autonomous Driving Cockpit...</h1>")


routes = [
    Route("/", endpoint=index),
    Route("/api/scenarios", endpoint=api_scenarios),
    Route("/api/simulation", endpoint=api_simulation),
    Route("/api/benchmark", endpoint=api_benchmark),
]

# Mount web directory for static assets
web_dir = os.path.join(os.path.dirname(__file__), "web")
os.makedirs(web_dir, exist_ok=True)
routes.append(Mount("/static", app=StaticFiles(directory=web_dir), name="static"))

app = Starlette(debug=True, routes=routes)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    
    # Auto-detect local network IP address
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"

    print(f"\n=======================================================")
    print(f"  AUTONOMOUS VEHICLE REAL-TIME COCKPIT & VISUALIZER")
    print(f"  > Local URL:   http://localhost:{port}")
    print(f"  > Network URL: http://{local_ip}:{port}")
    print(f"=======================================================\n")
    # Pre-seed village default run in background thread so server accepts connections immediately
    threading.Thread(target=get_or_run_simulation, args=("village", False, "normal", 42), daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
