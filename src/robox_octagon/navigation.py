# Author: Samuel Lozano
from __future__ import annotations

from typing import Optional

import numpy as np


def clip_to_octagon(pos: np.ndarray, inradius: float = 1.0) -> np.ndarray:
    """Project a point onto/inside a regular octagon defined by inradius."""
    p = np.asarray(pos, dtype=float).copy()
    r = float(inradius)

    angles = np.deg2rad(np.arange(0.0, 360.0, 45.0))
    normals = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    for _ in range(16):
        signed = normals @ p - r
        max_violation_idx = int(np.argmax(signed))
        max_violation = float(signed[max_violation_idx])
        if max_violation <= 0.0:
            break

        n = normals[max_violation_idx]
        p = p - max_violation * n

    return p.astype(np.float32)


def estimate_travel_time(start_pos: np.ndarray, target_pos: np.ndarray, max_speed: float) -> float:
    """Travel time estimate in seconds from Euclidean distance and max speed."""
    distance = float(np.linalg.norm(np.asarray(target_pos) - np.asarray(start_pos)))
    speed = float(max(1e-8, max_speed))
    return distance / speed


class NavigationController:
    """Deterministic low-level controller that moves toward a selected patch."""

    def __init__(self, inradius: float = 1.0, max_speed: float = 0.5) -> None:
        self.inradius = float(inradius)
        self.max_speed = float(max_speed)

    def step(self, agent_pos: np.ndarray, target_patch_pos: Optional[np.ndarray], dt: float) -> np.ndarray:
        pos = np.asarray(agent_pos, dtype=float)
        if target_patch_pos is None:
            return pos.astype(np.float32)

        target = np.asarray(target_patch_pos, dtype=float)
        vec = target - pos
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            return pos.astype(np.float32)

        direction = vec / norm
        movement = direction * self.max_speed * float(dt)
        candidate = pos + movement
        return clip_to_octagon(candidate, inradius=self.inradius)
