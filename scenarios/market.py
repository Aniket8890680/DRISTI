"""
Scenario 4: Dense Market Area
=============================
Requirements:
- Narrow, highly crowded urban bazaar / market road
- Mixed chaotic traffic:
  * Dense pedestrians crossing irregularly
  * Filtering motorcycles weaving between gaps
  * Auto-rickshaws making abrupt stops
  * Pushcart vendors on road shoulders
  * Illegally parked vehicles encroaching on road space
- Demands continuous adaptive replanning and fine drivable area estimation.
"""

from typing import Optional
import numpy as np

from simulation.road import Road
from simulation.vehicle import Vehicle, VehicleState
from simulation.actors import Actor, ActorType, BehaviorMode
from simulation.environment import Environment, WeatherCondition
from scenarios.scenario_base import ScenarioBase


class MarketScenario(ScenarioBase):
    """
    Dense Congested Market Street Scenario.
    """

    def __init__(
        self,
        weather: WeatherCondition = WeatherCondition.NORMAL,
        use_baseline_planner: bool = False,
        random_seed: Optional[int] = None
    ):
        super().__init__(
            name="Dense Market Area",
            description="Crowded Indian bazaar with pedestrians, weaving auto-rickshaws, filtering two-wheelers, and vendors encroaching on road.",
            weather=weather,
            use_baseline_planner=use_baseline_planner,
            max_duration=25.0,
            target_distance=80.0,
            random_seed=random_seed
        )

    def build_environment(self) -> None:
        # Market street (width 6.4m)
        waypoints = np.array([
            [0.0, 0.0],
            [30.0, 0.0],
            [60.0, 0.5],
            [95.0, 0.0]
        ], dtype=float)

        self.road = Road(
            waypoints=waypoints,
            width=6.4,
            has_lane_markings=False # No lanes in market bazaar
        )

        # Add road surface defect / pothole
        self.road.add_pothole(x=42.0, y=0.4, radius=0.6, severity=0.8)

        # Ego vehicle driving at cautious market speed: 6.0 m/s (~22 km/h)
        self.ego = Vehicle(initial_state=VehicleState(x=0.0, y=0.0, v=6.0, psi=0.0))
        self.env = Environment(
            road=self.road,
            ego_vehicle=self.ego,
            weather=self.weather,
            dt=0.05
        )

        # 1. Parked delivery vehicle partially blocking left shoulder (x=24m, y=2.0m)
        parked_car = Actor(
            actor_id=1,
            actor_type=ActorType.CAR,
            x=24.0,
            y=2.0,
            v=0.0,
            heading=0.0,
            behavior=BehaviorMode.STOPPED
        )
        self.env.add_actor(parked_car)

        # 2. Pushcart fruit vendor on right edge (x=35m, y=-2.2m, v=1.2 m/s)
        pushcart = Actor(
            actor_id=2,
            actor_type=ActorType.PUSHCART,
            x=35.0,
            y=-2.2,
            v=1.2,
            heading=0.0,
            behavior=BehaviorMode.CONSTANT_VELOCITY
        )
        self.env.add_actor(pushcart)

        # 3. Weaving auto-rickshaw ahead in center (starts at x=20m, y=-0.5m, v=4.5 m/s)
        rickshaw = Actor(
            actor_id=3,
            actor_type=ActorType.AUTO_RICKSHAW,
            x=20.0,
            y=-0.5,
            v=4.5,
            heading=0.0,
            behavior=BehaviorMode.CONSTANT_VELOCITY
        )
        self.env.add_actor(rickshaw)

        # 4. Filtering motorcycle weaving through traffic
        motorcycle = Actor(
            actor_id=4,
            actor_type=ActorType.MOTORCYCLE,
            x=15.0,
            y=1.0,
            v=7.0,
            heading=0.0,
            behavior=BehaviorMode.MOTORCYCLE_FILTERING,
            behavior_params={"frequency": 1.5, "amplitude": 0.5}
        )
        self.env.add_actor(motorcycle)

        # 5. Pedestrian crossing path mid-block (x=50m, starts at y=-3.0m, walks across to +2.5m)
        pedestrian = Actor(
            actor_id=5,
            actor_type=ActorType.PEDESTRIAN,
            x=50.0,
            y=-2.8,
            v=1.4,
            heading=np.pi / 2.0,
            behavior=BehaviorMode.PEDESTRIAN_CROSSING
        )
        self.env.add_actor(pedestrian)
