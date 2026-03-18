from __future__ import annotations

import importlib

import numpy as np


def make_box(low: np.ndarray, high: np.ndarray):
    try:
        box_cls = importlib.import_module("gym.spaces").Box
    except Exception:
        box_cls = importlib.import_module("gymnasium.spaces").Box
    return box_cls(low=low, high=high, dtype=np.float32)
