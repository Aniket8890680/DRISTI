"""
Road Model: Represents structured, semi-structured, and completely unstructured roads.

Provides:
- Parametric centerline with continuous curvature computation (spline/arc-length)
- Varying road width (e.g. narrow village roads, highway expansion, bottlenecks)
- Left and right boundary definitions
- Optional lane markings (solid, dashed, or missing)
- Unstructured hazards: Potholes, construction zones, irregular dirt shoulders
- Coordinate transformation between Cartesian (x, y) and Frenet (s, d) frames
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
from scipy.interpolate import CubicSpline


@dataclass
class Pothole:
    """Represents a road surface defect (pothole)."""
    id: int
    x: float
    y: float
    radius: float = 0.6       # meters
    severity: float = 0.8     # 0.0 (minor bump) to 1.0 (severe wheel damage)


@dataclass
class ConstructionZone:
    """Represents a temporary road obstruction or construction barrier."""
    id: int
    x: float
    y: float
    length: float             # along road
    width: float              # across road
    heading: float = 0.0      # orientation in radians


@dataclass
class LaneMarking:
    """Optional lane dividing line definition."""
    center_d: float           # lateral offset from road centerline (meters)
    line_type: str = "dashed" # 'dashed', 'solid', 'none'
    color: str = "white"


class Road:
    """
    Continuous road representation supporting unstructured Indian roadway geometry.
    """

    def __init__(
        self,
        waypoints: np.ndarray,
        width: float = 7.0,
        lane_markings: Optional[List[LaneMarking]] = None,
        has_lane_markings: bool = False,
        variable_width_fn: Optional[callable] = None,
    ):
        """
        waypoints: Nx2 array of (x, y) coordinates defining the centerline.
        width: Nominal road width in meters.
        lane_markings: List of lane markings if road is structured.
        has_lane_markings: Flag indicating if lanes are painted and visible.
        variable_width_fn: Function s -> width(s) for varying road width.
        """
        if len(waypoints) < 2:
            raise ValueError("Road waypoints must contain at least 2 points.")

        self.raw_waypoints = np.array(waypoints, dtype=np.float64)
        self.nominal_width = width
        self.has_lane_markings = has_lane_markings
        self.lane_markings = lane_markings if lane_markings is not None else []
        self.variable_width_fn = variable_width_fn

        # Road surface hazards
        self.potholes: List[Pothole] = []
        self.construction_zones: List[ConstructionZone] = []

        # Parameterize centerline by cumulative arc-length s
        diffs = np.diff(self.raw_waypoints, axis=0)
        segment_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        self.s_points = np.insert(np.cumsum(segment_lengths), 0, 0.0)
        self.total_length = float(self.s_points[-1])

        # Fit cubic splines for smooth reference trajectory (x(s), y(s))
        # If too few points, linear interpolation is preserved
        if len(self.raw_waypoints) >= 4:
            self.spline_x = CubicSpline(self.s_points, self.raw_waypoints[:, 0], bc_type='natural')
            self.spline_y = CubicSpline(self.s_points, self.raw_waypoints[:, 1], bc_type='natural')
        else:
            self.spline_x = None
            self.spline_y = None

        # Precompute dense polyline for fast geometric queries
        self._sample_dense_centerline()

    def _sample_dense_centerline(self, ds: float = 0.5) -> None:
        """Samples the road centerline and boundaries at uniform intervals ds."""
        s_vals = np.arange(0.0, self.total_length + ds, ds)
        if s_vals[-1] > self.total_length:
            s_vals[-1] = self.total_length
        self.dense_s = s_vals

        cx, cy, heading = [], [], []
        left_x, left_y, right_x, right_y = [], [], [], []

        for s in s_vals:
            x, y, psi = self.evaluate_centerline(s)
            w = self.get_width(s)
            half_w = w / 2.0

            # Normal vector: 90 degrees counter-clockwise from heading
            nx = -np.sin(psi)
            ny = np.cos(psi)

            cx.append(x)
            cy.append(y)
            heading.append(psi)

            # Left (+d) and Right (-d) road boundaries
            left_x.append(x + nx * half_w)
            left_y.append(y + ny * half_w)
            right_x.append(x - nx * half_w)
            right_y.append(y - ny * half_w)

        self.dense_centerline = np.column_stack((cx, cy))
        self.dense_headings = np.array(heading)
        self.left_boundary = np.column_stack((left_x, left_y))
        self.right_boundary = np.column_stack((right_x, right_y))

    def evaluate_centerline(self, s: float) -> Tuple[float, float, float]:
        """
        Evaluates road centerline position (x, y) and tangent heading psi at arc-length s.
        """
        s_clamped = np.clip(s, 0.0, self.total_length)
        if self.spline_x is not None and self.spline_y is not None:
            x = float(self.spline_x(s_clamped))
            y = float(self.spline_y(s_clamped))
            dx = float(self.spline_x.derivative(1)(s_clamped))
            dy = float(self.spline_y.derivative(1)(s_clamped))
            heading = np.arctan2(dy, dx)
        else:
            # Linear interpolation fallback
            idx = np.searchsorted(self.s_points, s_clamped) - 1
            idx = np.clip(idx, 0, len(self.s_points) - 2)
            ds = self.s_points[idx + 1] - self.s_points[idx]
            ratio = (s_clamped - self.s_points[idx]) / max(ds, 1e-6)
            p0 = self.raw_waypoints[idx]
            p1 = self.raw_waypoints[idx + 1]
            pt = p0 + ratio * (p1 - p0)
            x, y = float(pt[0]), float(pt[1])
            heading = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])

        return x, y, float(heading)

    def get_width(self, s: float) -> float:
        """Returns the road width at arc-length s."""
        if self.variable_width_fn is not None:
            return float(self.variable_width_fn(s))
        return self.nominal_width

    def cartesian_to_frenet(self, x: float, y: float) -> Tuple[float, float]:
        """
        Converts world (x, y) coordinates to Frenet coordinates (s, d).
        s: longitudinal distance along centerline (m)
        d: lateral deviation from centerline (+: left, -: right) (m)
        """
        # Find nearest point on dense centerline
        dists_sq = np.sum((self.dense_centerline - np.array([x, y])) ** 2, axis=1)
        min_idx = np.argmin(dists_sq)
        s_approx = self.dense_s[min_idx]

        # Refine with local projection
        cx, cy, heading = self.evaluate_centerline(s_approx)
        dx = x - cx
        dy = y - cy

        # Tangent and normal vectors
        tx = np.cos(heading)
        ty = np.sin(heading)
        nx = -np.sin(heading)
        ny = np.cos(heading)

        # Projection
        s = s_approx + (dx * tx + dy * ty)
        d = dx * nx + dy * ny
        return float(np.clip(s, 0.0, self.total_length)), float(d)

    def frenet_to_cartesian(self, s: float, d: float) -> Tuple[float, float, float]:
        """
        Converts Frenet coordinates (s, d) to world coordinates (x, y, heading).
        """
        cx, cy, heading = self.evaluate_centerline(s)
        nx = -np.sin(heading)
        ny = np.cos(heading)
        x = cx + d * nx
        y = cy + d * ny
        return float(x), float(y), heading

    def is_point_in_road(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Checks if a point (x, y) is inside the drivable road boundaries with a safety margin."""
        s, d = self.cartesian_to_frenet(x, y)
        half_w = self.get_width(s) / 2.0
        return abs(d) <= (half_w - margin)

    def distance_to_boundaries(self, x: float, y: float) -> Tuple[float, float]:
        """
        Returns distances to (left_boundary, right_boundary).
        Positive when inside the boundary, negative if outside.
        """
        s, d = self.cartesian_to_frenet(x, y)
        half_w = self.get_width(s) / 2.0
        dist_left = half_w - d
        dist_right = half_w + d
        return float(dist_left), float(dist_right)

    def is_box_in_road(self, corners: np.ndarray, margin: float = 0.0) -> bool:
        """Checks if all 4 corners of an oriented bounding box are within the road boundaries."""
        for pt in corners:
            if not self.is_point_in_road(pt[0], pt[1], margin=margin):
                return False
        return True

    def add_pothole(self, x: float, y: float, radius: float = 0.6, severity: float = 0.8) -> Pothole:
        """Adds a pothole to the road surface."""
        pothole = Pothole(id=len(self.potholes) + 1, x=x, y=y, radius=radius, severity=severity)
        self.potholes.append(pothole)
        return pothole

    def add_construction_zone(self, x: float, y: float, length: float, width: float, heading: float = 0.0) -> ConstructionZone:
        """Adds a construction barrier zone to the road."""
        cz = ConstructionZone(id=len(self.construction_zones) + 1, x=x, y=y, length=length, width=width, heading=heading)
        self.construction_zones.append(cz)
        return cz
