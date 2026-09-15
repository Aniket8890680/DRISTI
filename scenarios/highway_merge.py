"""
Scenario 3: Highway Merge
=========================
Requirements:
- Higher speed expressway traffic
- Slow-moving overloaded truck traveling ahead in lane
- Merging passenger car entering informally from shoulder/ramp
- Planner must evaluate relative Doppler velocity, recognize closing speeds,
  and execute adaptive gap-acceptance or lane change.
"""

from typing import Optional
import numpy as np

from simulation.road import Road
from simulation.vehicle import Vehicle, VehicleState
from simulation.actors import Actor, ActorType, BehaviorMode
from simulation.environment import Environment, WeatherCondition
from scenarios.scenario_base import ScenarioBase


class HighwayMergeScenario(ScenarioBase):
    """
    Highway Merge with Mixed Speed Traffic Scenario.
    """

    def __init__(
        self,
        weather: WeatherCondition = WeatherCondition.NORMAL,
        use_baseline_planner: bool = False,
        random_seed: Optional[int] = None
    ):
        super().__init__(
            name="Highway Merge",
            description="Expressway cruising with slow-moving commercial truck ahead and a vehicle merging from the shoulder.",
            weather=weather,
            use_baseline_planner=use_baseline_planner,
            max_duration=22.0,
            target_distance=140.0,
            random_seed=random_seed
        )

    def build_environment(self) -> None:
        # Multi-lane highway road (width 10.5m)
        waypoints = np.array([
            [0.0, 0.0],
            [50.0, 0.0],
            [100.0, 0.0],
            [160.0, 0.0]
        ], dtype=float)

        self.road = Road(
            waypoints=waypoints,
            width=10.5,
            has_lane_markings=True
        )

        # Ego vehicle traveling at fast highway cruise speed: 16.0 m/s (~58 km/h)
        self.ego = Vehicle(initial_state=VehicleState(x=0.0, y=-1.5, v=16.0, psi=0.0))
        self.env = Environment(
            road=self.road,
            ego_vehicle=self.ego,
            weather=self.weather,
            dt=0.05
        )

        # 1. Slow overloaded truck directly ahead in ego corridor (starts at x=40m, y=-1.5m, v=7.0 m/s)
        truck = Actor(
            actor_id=1,
            actor_type=ActorType.TRUCK,
            x=40.0,
            y=-1.5,
            v=7.0,
            heading=0.0,
            behavior=BehaviorMode.CONSTANT_VELOCITY
        )
        self.env.add_actor(truck)

        # 2. Car merging from left on-ramp / shoulder at shallow angle (starts at x=25m, y=4.5m, v=11.0 m/s, merging inward)
        merging_car = Actor(
            actor_id=2,
            actor_type=ActorType.CAR,
            x=25.0,
            y=4.5,
            v=11.0,
            heading=-np.deg2rad(8.0), # merging inward toward y=1.5
            behavior=BehaviorMode.INFORMAL_MERGE,
            behavior_params={"merge_rate": -0.05}
        )
        self.env.add_actor(merging_car)
