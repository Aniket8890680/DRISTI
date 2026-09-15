"""
Scenario 2: Busy Unsignalized Urban Intersection
================================================
Requirements:
- No traffic signal or structured stop signs
- Mixed multi-directional traffic:
  * Perpendicular crossing auto-rickshaw
  * Turning motorcycle informally merging into path
  * Crossing pedestrians
- Vehicle must predict cross-traffic intersections, calculate TTC, and yield/replan.
"""

from typing import Optional
import numpy as np

from simulation.road import Road
from simulation.vehicle import Vehicle, VehicleState
from simulation.actors import Actor, ActorType, BehaviorMode
from simulation.environment import Environment, WeatherCondition
from scenarios.scenario_base import ScenarioBase


class IntersectionScenario(ScenarioBase):
    """
    Busy Unsignalized Urban Intersection Scenario.
    """

    def __init__(
        self,
        weather: WeatherCondition = WeatherCondition.NORMAL,
        use_baseline_planner: bool = False,
        random_seed: Optional[int] = None
    ):
        super().__init__(
            name="Busy Unsignalized Urban Intersection",
            description="Chaotic unsignalized 4-way intersection with crossing auto-rickshaws, turning motorcycle, and darting pedestrians.",
            weather=weather,
            use_baseline_planner=use_baseline_planner,
            max_duration=20.0,
            target_distance=85.0,
            random_seed=random_seed
        )

    def build_environment(self) -> None:
        # Straight main road through intersection (intersection center at x=45.0m)
        waypoints = np.array([
            [0.0, 0.0],
            [30.0, 0.0],
            [60.0, 0.0],
            [100.0, 0.0]
        ], dtype=float)

        self.road = Road(
            waypoints=waypoints,
            width=8.0,
            has_lane_markings=False  # Faded / missing markings typical of urban India
        )

        # Ego vehicle starts at (0, 0) with speed 9.0 m/s
        self.ego = Vehicle(initial_state=VehicleState(x=0.0, y=0.0, v=9.0, psi=0.0))
        self.env = Environment(
            road=self.road,
            ego_vehicle=self.ego,
            weather=self.weather,
            dt=0.05
        )

        # Traffic participants:
        # 1. Auto-rickshaw crossing perpendicularly from right (starts at x=44, y=-14, heading north pi/2, v=5.5 m/s)
        # Reaches intersection center (y=0) at t = 14 / 5.5 = ~2.5s
        rickshaw = Actor(
            actor_id=1,
            actor_type=ActorType.AUTO_RICKSHAW,
            x=44.0,
            y=-14.0,
            v=5.5,
            heading=np.pi / 2.0,
            behavior=BehaviorMode.CONSTANT_VELOCITY
        )
        self.env.add_actor(rickshaw)

        # 2. Motorcycle turning right across ego path (starts ahead at x=55, y=3.0, cuts across)
        moto = Actor(
            actor_id=2,
            actor_type=ActorType.MOTORCYCLE,
            x=55.0,
            y=3.0,
            v=6.0,
            heading=np.pi,
            behavior=BehaviorMode.SUDDEN_CUT_IN,
            behavior_params={"trigger_time": 1.0, "yaw_rate": -0.4}
        )
        self.env.add_actor(moto)

        # 3. Pedestrian crossing informally at far edge of intersection (x=68.0, crossing from y=4.5 to y=-4.5 at 1.3 m/s)
        pedestrian = Actor(
            actor_id=3,
            actor_type=ActorType.PEDESTRIAN,
            x=68.0,
            y=4.2,
            v=1.3,
            heading=-np.pi / 2.0,
            behavior=BehaviorMode.PEDESTRIAN_CROSSING
        )
        self.env.add_actor(pedestrian)
