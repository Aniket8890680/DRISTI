"""
Scenario 1: Unmarked Village Road
=================================
Requirements:
- No reliable lane markings (completely unmarked surface)
- Narrow road with width narrowing (5.5m narrowing to 4.8m bottleneck)
- Mixed traffic: slow pushcart along shoulder, oncoming motorcycle, road potholes
- Drivable area estimation is essential because traditional lane trackers fail.
"""

from typing import Optional
import numpy as np

from simulation.road import Road
from simulation.vehicle import Vehicle, VehicleState
from simulation.actors import Actor, ActorType, BehaviorMode
from simulation.environment import Environment, WeatherCondition
from scenarios.scenario_base import ScenarioBase


def village_width(s: float) -> float:
    """Narrows from 5.6m down to 4.8m bottleneck between s=30 and s=70."""
    if 30.0 <= s <= 70.0:
        return 5.6 - 0.8 * np.sin(np.pi * (s - 30.0) / 40.0)
    return 5.6


class VillageScenario(ScenarioBase):
    """
    Unmarked Village Road Scenario.
    """

    def __init__(
        self,
        weather: WeatherCondition = WeatherCondition.NORMAL,
        use_baseline_planner: bool = False,
        random_seed: Optional[int] = 42
    ):
        super().__init__(
            name="Unmarked Village Road",
            description="Narrow Indian village road with no lane markings, surface potholes, slow pushcart, and oncoming motorcycle.",
            weather=weather,
            use_baseline_planner=use_baseline_planner,
            max_duration=26.0,
            target_distance=85.0,
            random_seed=random_seed
        )

    def build_environment(self) -> None:
        # 1. Narrow road with variable bottleneck width
        waypoints = np.array([
            [0.0, 0.0],
            [30.0, 0.0],
            [60.0, 1.0],   # slight gentle bend
            [100.0, 1.0],
            [130.0, 0.0]
        ], dtype=float)

        self.road = Road(
            waypoints=waypoints,
            width=5.6,
            has_lane_markings=False,  # UNSTRUCTURED: No markings!
            variable_width_fn=village_width
        )

        # Add road surface potholes
        self.road.add_pothole(x=25.0, y=-0.5, radius=0.6, severity=0.9)
        self.road.add_pothole(x=55.0, y=0.8, radius=0.5, severity=0.7)

        # 2. Ego vehicle starts at (0, 0) cruising at 8.0 m/s
        self.ego = Vehicle(initial_state=VehicleState(x=0.0, y=0.0, v=8.0, psi=0.0))
        self.env = Environment(
            road=self.road,
            ego_vehicle=self.ego,
            weather=self.weather,
            dt=0.05
        )

        # 3. Traffic actors:
        # Actor 1: Slow pushcart along right edge of road (v = 1.5 m/s)
        pushcart = Actor(
            actor_id=1,
            actor_type=ActorType.PUSHCART,
            x=35.0,
            y=-1.5,
            v=1.5,
            heading=0.0,
            behavior=BehaviorMode.CONSTANT_VELOCITY
        )
        self.env.add_actor(pushcart)

        # Actor 2: Oncoming motorcycle on left side traveling towards ego (v = -7.0 m/s)
        oncoming_moto = Actor(
            actor_id=2,
            actor_type=ActorType.MOTORCYCLE,
            x=85.0,
            y=1.5,
            v=7.0,
            heading=np.pi, # facing towards -X
            behavior=BehaviorMode.WRONG_SIDE
        )
        self.env.add_actor(oncoming_moto)
