"""
Perception Detector Module:
Translates sensor outputs into standardized intermediate detection packets,
allowing real neural networks or object detectors to be plugged in interchangeably.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

from simulation.actors import ActorType


@dataclass
class DetectionPacket:
    """Standardized representation of an object observation from any perception modality."""
    source_sensor: str          # 'camera', 'radar', 'lidar'
    actor_id_hint: Optional[int] # Sensor-level ID if available
    actor_type: ActorType       # Estimated classification
    x: float                    # World X (m)
    y: float                    # World Y (m)
    vx: Optional[float] = None  # World velocity X if measured (e.g. Radar)
    vy: Optional[float] = None  # World velocity Y if measured
    heading: float = 0.0        # Estimated orientation
    length: float = 2.0         # Estimated length (m)
    width: float = 1.0          # Estimated width (m)
    confidence: float = 0.8     # Sensor confidence [0.0, 1.0]
    range_to_ego: float = 0.0   # Distance to ego vehicle (m)
