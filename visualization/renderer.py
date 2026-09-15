"""
Top-Down Visualization Module:
Provides comprehensive real-time rendering and replay visualization
designed for SIH judges and technical evaluation.

Renders:
- Scrolling top-down view centered on the ego vehicle
- Unstructured road edges and varying road boundaries
- Surface hazards (potholes, debris)
- Ego vehicle bounding box, heading orientation, and velocity
- Surrounding heterogeneous actors with distinctive color palettes
- Velocity vector arrows
- Predicted multi-second future trajectories with uncertainty envelopes
- Candidate trajectory fan (faint) and chosen safe trajectory (bold green)
- Live Telemetry HUD: Speed, Mode, TTC, Clearance, Replanning counter, Risk Badge
- Export capability to save simulation frame sequences or animated MP4/GIFs
"""

from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

from simulation.actors import ActorType
from simulation.environment import WeatherCondition
from scenarios.scenario_base import ScenarioBase, TelemetrySnapshot


# Palette for heterogeneous Indian traffic
ACTOR_COLORS = {
    ActorType.CAR: "#2980b9",            # Blue
    ActorType.BUS: "#d35400",            # Rust Orange
    ActorType.TRUCK: "#b9770e",          # Heavy Truck Ochre
    ActorType.AUTO_RICKSHAW: "#f39c12",  # Auto Yellow
    ActorType.MOTORCYCLE: "#8e44ad",     # Purple
    ActorType.BICYCLE: "#16a085",        # Teal
    ActorType.PEDESTRIAN: "#e91e63",     # Magenta
    ActorType.PUSHCART: "#27ae60",       # Green
    ActorType.ANIMAL: "#c0392b",         # Crimson
    ActorType.POTHOLE: "#7f8c8d",        # Grey
    ActorType.CONSTRUCTION_BARRIER: "#e74c3c", # Hazard Red
    ActorType.STATIC_OBSTACLE: "#7f8c8d" # Dark Grey
}

RISK_COLORS = {
    "LOW": "#2ecc71",      # Bright Green
    "MEDIUM": "#f1c40f",   # Amber
    "HIGH": "#e67e22",     # Orange
    "CRITICAL": "#e74c3c"  # Bright Red
}


class TopDownRenderer:
    """
    Renders top-down 2D simulation views and animations.
    """

    def __init__(self, figsize: Tuple[int, int] = (14, 8)):
        self.figsize = figsize

    def render_snapshot(
        self,
        scenario: ScenarioBase,
        frame_idx: int = -1,
        save_path: Optional[str] = None,
        show_plot: bool = False
    ) -> plt.Figure:
        """
        Renders a single high-resolution snapshot at frame_idx.
        """
        if len(scenario.telemetry) == 0:
            raise ValueError("Scenario telemetry is empty. Run scenario first.")

        if frame_idx < 0:
            frame_idx = len(scenario.telemetry) + frame_idx
        frame_idx = int(np.clip(frame_idx, 0, len(scenario.telemetry) - 1))
        snap = scenario.telemetry[frame_idx]

        fig, ax = plt.subplots(figsize=self.figsize, facecolor="#1a1a1a")
        ax.set_facecolor("#222222")

        self._draw_frame(ax, scenario, snap)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def _draw_frame(self, ax: plt.Axes, scenario: ScenarioBase, snap: TelemetrySnapshot) -> None:
        """Draws all environmental, perception, planning, and HUD layers."""
        road = scenario.road
        ego_x, ego_y = snap.ego_x, snap.ego_y

        # Dynamic camera window centered on ego
        view_x_range = 45.0
        view_y_range = 16.0
        ax.set_xlim(ego_x - 15.0, ego_x + view_x_range)
        ax.set_ylim(-view_y_range / 2.0, view_y_range / 2.0)
        ax.set_aspect('equal')
        ax.axis('off')

        # 1. Road Surface and Boundaries
        if road is not None:
            # Left and right road boundaries
            ax.plot(road.left_boundary[:, 0], road.left_boundary[:, 1], color="#ffffff", lw=2.5, alpha=0.9)
            ax.plot(road.right_boundary[:, 0], road.right_boundary[:, 1], color="#ffffff", lw=2.5, alpha=0.9)

            # Centerline (dashed if structured, faint guideline if unstructured)
            if road.has_lane_markings:
                ax.plot(road.dense_centerline[:, 0], road.dense_centerline[:, 1], color="#f1c40f", lw=1.5, ls="--", alpha=0.7)
            else:
                ax.plot(road.dense_centerline[:, 0], road.dense_centerline[:, 1], color="#555555", lw=0.8, ls=":", alpha=0.5)

            # Road surface fill between boundaries
            poly_x = np.concatenate([road.left_boundary[:, 0], road.right_boundary[::-1, 0]])
            poly_y = np.concatenate([road.left_boundary[:, 1], road.right_boundary[::-1, 1]])
            ax.fill(poly_x, poly_y, color="#2c3e50", alpha=0.6)

            # Draw Potholes
            for ph in road.potholes:
                circ = patches.Circle((ph.x, ph.y), ph.radius, color="#e74c3c", alpha=0.5)
                ax.add_patch(circ)
                rim = patches.Circle((ph.x, ph.y), ph.radius, fill=False, edgecolor="#c0392b", lw=1.5, ls="--")
                ax.add_patch(rim)
                ax.text(ph.x, ph.y - ph.radius - 0.4, "POTHOLE", color="#e74c3c", fontsize=7, ha='center', weight='bold')

        # 2. Candidate Trajectories (Faint Fan)
        if snap.candidate_trajectories and not scenario.use_baseline:
            for cand in snap.candidate_trajectories[:18]:
                cx = [wp.x for wp in cand.waypoints]
                cy = [wp.y for wp in cand.waypoints]
                ax.plot(cx, cy, color="#00bcd4", lw=0.8, alpha=0.25)

        # 3. Active Planned Trajectory (Bold Green)
        if snap.active_trajectory and len(snap.active_trajectory.waypoints) > 0:
            tx = [wp.x for wp in snap.active_trajectory.waypoints]
            ty = [wp.y for wp in snap.active_trajectory.waypoints]
            ax.plot(tx, ty, color="#00ff88", lw=3.0, alpha=0.95, label="Planned Path", zorder=6)
            ax.scatter(tx[::3], ty[::3], color="#00ff88", s=15, zorder=7)

        # 4. Surrounding Tracked Objects & Predicted Trajectories
        for obj in snap.tracked_objects:
            color = ACTOR_COLORS.get(obj.actor_type, "#95a5a6")
            
            # Oriented Bounding Box
            box = obj.get_bounding_box()
            poly = patches.Polygon(box, closed=True, facecolor=color, edgecolor="#ffffff", lw=1.5, alpha=0.85, zorder=8)
            ax.add_patch(poly)

            # Object Label & Speed
            label_text = f"{obj.actor_type.value.upper()} ({obj.speed*3.6:.0f} km/h)"
            ax.text(obj.x, obj.y + max(obj.length, obj.width)/2.0 + 0.5, label_text,
                    color="#ffffff", fontsize=7, ha='center', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="none", alpha=0.7), zorder=9)

            # Velocity Vector Arrow
            if obj.speed > 0.3:
                ax.arrow(obj.x, obj.y, obj.vx * 0.8, obj.vy * 0.8,
                         head_width=0.4, head_length=0.5, fc="#ffffff", ec="#ffffff", alpha=0.8, zorder=9)

            # Predicted Trajectory (1-3s future)
            if obj.track_id in snap.predicted_trajectories:
                pred = snap.predicted_trajectories[obj.track_id]
                px = [pt.x for pt in pred.points]
                py = [pt.y for pt in pred.points]
                ax.plot(px, py, color=color, lw=1.8, ls=":", alpha=0.8, zorder=7)
                
                # Draw uncertainty halo at end of prediction
                if len(pred.points) > 0:
                    end_pt = pred.points[-1]
                    halo = patches.Circle((end_pt.x, end_pt.y), end_pt.uncertainty_radius,
                                          facecolor=color, edgecolor=color, alpha=0.15, ls="--", zorder=5)
                    ax.add_patch(halo)

        # 5. Ego Vehicle
        ego_box = scenario.env.ego.get_bounding_box()
        ego_poly = patches.Polygon(ego_box, closed=True, facecolor="#00d2ff", edgecolor="#ffffff", lw=2.0, alpha=0.95, zorder=10)
        ax.add_patch(ego_poly)

        # Ego Heading & Center Point
        ax.plot(ego_x, ego_y, marker="o", markersize=4, color="#ffffff", zorder=11)
        ax.arrow(ego_x, ego_y, 2.5 * np.cos(snap.ego_psi), 2.5 * np.sin(snap.ego_psi),
                 head_width=0.5, head_length=0.6, fc="#ffffff", ec="#ffffff", zorder=11)
        ax.text(ego_x, ego_y - 1.8, "EGO VEHICLE", color="#00d2ff", fontsize=8, ha='center', weight='bold', zorder=11)

        # 6. Telemetry HUD Overlay
        risk_color = RISK_COLORS.get(snap.risk_level, "#2ecc71")
        maneuver = snap.active_trajectory.maneuver_name if snap.active_trajectory else "Cruise"
        ttc_str = f"{snap.min_ttc:.1f} s" if snap.min_ttc < 99.0 else "CLEAR"

        hud_text = (
            f"SCENARIO: {scenario.name.upper()} | TIME: {snap.time:.2f}s\n"
            f"SPEED: {snap.ego_v * 3.6:4.1f} km/h  |  ACCEL: {snap.ego_a:+4.2f} m/s²  |  STEER: {np.rad2deg(snap.ego_delta):+4.1f}°\n"
            f"MODE: {snap.planner_state}  |  MANEUVER: {maneuver}\n"
            f"MIN TTC: {ttc_str}  |  CLEARANCE: {snap.min_clearance:.2f} m  |  REPLANS: {getattr(scenario.planner, 'replanning_count', 0)}"
        )

        ax.text(ego_x - 14.0, 6.8, hud_text, color="#ffffff", fontsize=8, family='monospace',
                bbox=dict(boxstyle="square,pad=0.5", fc="#111111", ec="#444444", alpha=0.85), zorder=15)

        # Risk Indicator Badge in Top Right
        ax.text(ego_x + view_x_range - 2.0, 6.8, f"RISK: {snap.risk_level}",
                color="#ffffff", fontsize=10, weight='bold', ha='right',
                bbox=dict(boxstyle="round,pad=0.4", fc=risk_color, ec="#ffffff", alpha=0.9), zorder=15)

    def save_animation_gif(
        self,
        scenario: ScenarioBase,
        output_path: str,
        fps: int = 15,
        max_frames: int = 150
    ) -> None:
        """
        Creates and saves an animated GIF / MP4 of the scenario execution.
        """
        if len(scenario.telemetry) == 0:
            return

        # Downsample frames to achieve smooth animation within limit
        total_snaps = len(scenario.telemetry)
        step_size = max(1, total_snaps // max_frames)
        sampled_indices = list(range(0, total_snaps, step_size))

        fig, ax = plt.subplots(figsize=self.figsize, facecolor="#1a1a1a")

        def update(frame_num):
            ax.clear()
            snap = scenario.telemetry[sampled_indices[frame_num]]
            self._draw_frame(ax, scenario, snap)

        anim = FuncAnimation(fig, update, frames=len(sampled_indices), interval=1000 // fps)
        anim.save(output_path, writer='pillow', fps=fps)
        plt.close(fig)
