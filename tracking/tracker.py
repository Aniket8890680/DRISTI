"""
Object Tracker Module:
Maintains object identities and filters state estimates across simulation frames
using 2D Kalman Filters and track lifecycle management.

Solves the requirement:
- Do not treat every frame as a completely new observation.
- Track position, velocity, and orientation over time.
- Filter measurement noise from sensors.
- Maintain tracks through brief dropouts (coasting).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict
import numpy as np

from simulation.actors import ActorType
from simulation.vehicle import normalize_angle
from perception.sensor_fusion import FusedObjectObservation


class TrackState(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTING = "coasting"
    DELETED = "deleted"


class KalmanBoxTracker:
    """
    Constant-Velocity Kalman Filter for an individual traffic participant:
    State: x = [pos_x, pos_y, vel_x, vel_y]^T
    """

    def __init__(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        # State vector: [x, y, vx, vy]
        self.x = np.array([x, y, vx, vy], dtype=np.float64)
        
        # State covariance matrix P
        self.P = np.diag([0.5, 0.5, 2.0, 2.0])

        # Process noise covariance Q base
        self.q_var = 1.0

        # Measurement noise covariance R (default position only)
        self.R_pos = np.diag([0.15, 0.15])
        self.R_full = np.diag([0.15, 0.15, 0.25, 0.25])

    def predict(self, dt: float) -> np.ndarray:
        """Propagates state and covariance forward by dt."""
        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        # Discrete process noise matrix Q
        dt2 = dt * dt
        dt3 = dt2 * dt / 2.0
        dt4 = dt2 * dt2 / 4.0
        q = self.q_var
        Q = np.array([
            [dt4 * q, 0.0,     dt3 * q, 0.0    ],
            [0.0,     dt4 * q, 0.0,     dt3 * q],
            [dt3 * q, 0.0,     dt2 * q, 0.0    ],
            [0.0,     dt3 * q, 0.0,     dt2 * q]
        ])

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x

    def update(self, z_x: float, z_y: float, z_vx: Optional[float] = None, z_vy: Optional[float] = None) -> None:
        """Incorporates new sensor observation."""
        if z_vx is not None and z_vy is not None and (abs(z_vx) > 0.1 or abs(z_vy) > 0.1):
            # Full state measurement (position + Doppler velocity)
            H = np.eye(4)
            z = np.array([z_x, z_y, z_vx, z_vy])
            R = self.R_full
        else:
            # Position-only measurement
            H = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0]
            ])
            z = np.array([z_x, z_y])
            R = self.R_pos

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P


@dataclass
class TrackedObject:
    """Consolidated state of a tracked road participant."""
    track_id: int
    actor_type: ActorType
    x: float
    y: float
    vx: float
    vy: float
    speed: float
    heading: float
    length: float
    width: float
    confidence: float
    uncertainty: float
    sources: List[str]
    time_since_update: int = 0
    hits: int = 1
    state: TrackState = TrackState.TENTATIVE

    def get_bounding_box(self) -> np.ndarray:
        """Returns the 4 corner coordinates in 2D world space."""
        half_l = self.length / 2.0
        half_w = self.width / 2.0

        local_corners = np.array([
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w]
        ])

        cos_h = np.cos(self.heading)
        sin_h = np.sin(self.heading)
        R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

        world_corners = (R @ local_corners.T).T + np.array([self.x, self.y])
        return world_corners


class MultiObjectTracker:
    """
    Multi-target tracker managing tracks, Kalman filters, data association, and lifecycle.
    """

    def __init__(
        self,
        max_dist_threshold: float = 3.5, # Association gating distance in meters
        max_missed_frames: int = 4,     # Coasting threshold before track deletion
        min_hits_to_confirm: int = 2,   # Hits before track is considered CONFIRMED
    ):
        self.max_dist_threshold = max_dist_threshold
        self.max_missed_frames = max_missed_frames
        self.min_hits_to_confirm = min_hits_to_confirm
        
        self.tracks: Dict[int, TrackedObject] = {}
        self.filters: Dict[int, KalmanBoxTracker] = {}
        self.next_track_id = 1

    def update(self, fused_detections: List[FusedObjectObservation], dt: float) -> List[TrackedObject]:
        """
        Advances tracker state by dt, associates detections, updates Kalman filters,
        and manages track lifecycles.
        """
        # 1. Prediction step for all existing tracks
        for tid, kf in self.filters.items():
            pred_state = kf.predict(dt)
            track = self.tracks[tid]
            track.x = float(pred_state[0])
            track.y = float(pred_state[1])
            track.vx = float(pred_state[2])
            track.vy = float(pred_state[3])
            track.speed = float(np.hypot(track.vx, track.vy))
            if track.speed > 0.3:
                track.heading = float(np.arctan2(track.vy, track.vx))
            track.time_since_update += 1

        # 2. Data association (Greedy nearest neighbor matching)
        unmatched_detections = set(range(len(fused_detections)))
        unmatched_tracks = set(self.tracks.keys())
        matched_pairs: List[Tuple[int, int]] = []

        if len(self.tracks) > 0 and len(fused_detections) > 0:
            track_ids = list(self.tracks.keys())
            cost_matrix = np.zeros((len(track_ids), len(fused_detections)))

            for i, tid in enumerate(track_ids):
                trk = self.tracks[tid]
                for j, det in enumerate(fused_detections):
                    dist = np.hypot(trk.x - det.x, trk.y - det.y)
                    cost_matrix[i, j] = dist

            # Greedy matching in order of lowest distance
            sorted_indices = np.argsort(cost_matrix, axis=None)
            used_t = set()
            used_d = set()

            for flat_idx in sorted_indices:
                i, j = np.unravel_index(flat_idx, cost_matrix.shape)
                if i in used_t or j in used_d:
                    continue
                if cost_matrix[i, j] <= self.max_dist_threshold:
                    matched_pairs.append((track_ids[i], j))
                    used_t.add(i)
                    used_d.add(j)
                    unmatched_tracks.discard(track_ids[i])
                    unmatched_detections.discard(j)

        # 3. Update matched tracks with Kalman measurement update
        for tid, det_idx in matched_pairs:
            det = fused_detections[det_idx]
            kf = self.filters[tid]
            kf.update(det.x, det.y, det.vx, det.vy)

            trk = self.tracks[tid]
            trk.x = float(kf.x[0])
            trk.y = float(kf.x[1])
            trk.vx = float(kf.x[2])
            trk.vy = float(kf.x[3])
            trk.speed = float(np.hypot(trk.vx, trk.vy))
            if trk.speed > 0.3:
                trk.heading = float(np.arctan2(trk.vy, trk.vx))
            elif abs(det.heading) > 1e-3:
                trk.heading = det.heading
            
            trk.length = det.length
            trk.width = det.width
            trk.actor_type = det.actor_type
            trk.confidence = det.confidence
            trk.uncertainty = det.uncertainty
            trk.sources = det.sources
            trk.time_since_update = 0
            trk.hits += 1

            if trk.hits >= self.min_hits_to_confirm:
                trk.state = TrackState.CONFIRMED

        # 4. Handle coasting for unmatched tracks
        for tid in unmatched_tracks:
            trk = self.tracks[tid]
            trk.state = TrackState.COASTING
            trk.confidence *= 0.85  # Confidence decays during coasting

        # 5. Initialize new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det = fused_detections[det_idx]
            new_id = self.next_track_id
            self.next_track_id += 1

            kf = KalmanBoxTracker(det.x, det.y, det.vx, det.vy)
            self.filters[new_id] = kf

            speed = float(np.hypot(det.vx, det.vy))
            heading = float(det.heading if abs(det.heading) > 1e-3 else (np.arctan2(det.vy, det.vx) if speed > 0.3 else 0.0))

            self.tracks[new_id] = TrackedObject(
                track_id=new_id,
                actor_type=det.actor_type,
                x=det.x,
                y=det.y,
                vx=det.vx,
                vy=det.vy,
                speed=speed,
                heading=heading,
                length=det.length,
                width=det.width,
                confidence=det.confidence,
                uncertainty=det.uncertainty,
                sources=det.sources,
                time_since_update=0,
                hits=1,
                state=TrackState.TENTATIVE
            )

        # 6. Delete dead tracks exceeding max_missed_frames
        dead_ids = [tid for tid, trk in self.tracks.items() if trk.time_since_update > self.max_missed_frames]
        for tid in dead_ids:
            del self.tracks[tid]
            del self.filters[tid]

        # Return active confirmed and tentative tracks
        return [trk for trk in self.tracks.values() if trk.state in [TrackState.CONFIRMED, TrackState.TENTATIVE]]
