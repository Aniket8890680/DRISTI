"""
Scenario 5: Sudden Cattle Crossing
==================================
Requirements:
- Animal (cattle/cow) initially stationary or grazing outside the vehicle path on the shoulder
- Suddenly enters the roadway at an unpredictable moment and freezes in the lane
- High trajectory uncertainty envelope
- Autonomous vehicle must:
  * Detect animal presence via multi-sensor fusion
  * Track its trajectory and assign high behavioral uncertainty
  * Predict potential wandering/freeze in middle of path
  * Calculate elevated composite risk
  * Adaptively generate a collision-free evasive corridor or execute controlled stopping
"""

from typing import Optional
import numpy as np

from simulation.road import Road
from simulation.vehicle import Vehicle, VehicleState
from simulation.actors import Actor, ActorType, BehaviorMode
from simulation.environment import Environment, WeatherCondition
from scenarios.scenario_base import ScenarioBase


class CattleCrossingScenario(ScenarioBase):
    """
    Sudden Cattle / Animal Crossing Scenario.
    """

    def __init__(
        self,
        weather: WeatherCondition = WeatherCondition.NORMAL,
        use_baseline_planner: bool = False,
        random_seed: Optional[int] = None
    ):
        super().__init__(
            name="Sudden Cattle Crossing",
            description="Cattle grazing on shoulder suddenly steps into road corridor and freezes, requiring rapid risk assessment and adaptive evasion/stopping.",
            weather=weather,
            use_baseline_planner=use_baseline_planner,
            max_duration=18.0,
            target_distance=75.0,
            random_seed=random_seed
        )

    def build_environment(self) -> None:
        # Standard two-lane rural road (width 7.5m)
        waypoints = np.array([
            [0.0, 0.0],
            [40.0, 0.0],
            [80.0, 0.0],
            [110.0, 0.0]
        ], dtype=float)

        self.road = Road(
            waypoints=waypoints,
            width=7.5,
            has_lane_markings=False
        )

        # Ego vehicle traveling at rural cruise speed: 11.0 m/s (~40 km/h)
        self.ego = Vehicle(initial_state=VehicleState(x=0.0, y=-1.0, v=11.0, psi=0.0))
        self.env = Environment(
            road=self.road,
            ego_vehicle=self.ego,
            weather=self.weather,
            dt=0.05
        )

        # 1. Cattle initially on left dirt shoulder (x=36.0m, y=3.2m),
        # Steps into road at t=1.0s and freezes in center (y=0.3m)
        cattle = Actor(
            actor_id=1,
            actor_type=ActorType.ANIMAL,
            x=36.0,
            y=3.2,
            v=1.2,
            heading=-np.pi / 2.0, # Moving downward across road
            behavior=BehaviorMode.ANIMAL_WANDERING,
            behavior_params={"freeze_start": 2.2, "freeze_duration": 8.0}
        )
        self.env.add_actor(cattle)

        # 2. Distant motorcycle approaching in far oncoming lane to test safe margin during swerve
        oncoming_bike = Actor(
            actor_id=2,
            actor_type=ActorType.MOTORCYCLE,
            x=90.0,
            y=2.5,
            v=6.0,
            heading=np.pi,
            behavior=BehaviorMode.CONSTANT_VELOCITY
        )
        self.env.add_actor(oncoming_bike)
