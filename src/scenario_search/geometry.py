"""Small geometry helpers (no nuscenes-devkit dependency)."""

from __future__ import annotations

import numpy as np


def quat_to_yaw(q) -> float:
    """Yaw (rad, CCW from +x) of a [w, x, y, z] quaternion."""
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def wrap_angle(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def unwrap_yaw(yaw: np.ndarray) -> np.ndarray:
    return np.unwrap(np.asarray(yaw, dtype=float))


def to_ego_frame(points_xy: np.ndarray, ego_xy: np.ndarray, ego_yaw: float) -> np.ndarray:
    """Express world xy points in the ego frame (x forward, y left)."""
    d = np.asarray(points_xy, dtype=float) - np.asarray(ego_xy, dtype=float)
    c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
    return np.stack([c * d[..., 0] - s * d[..., 1], s * d[..., 0] + c * d[..., 1]], axis=-1)
