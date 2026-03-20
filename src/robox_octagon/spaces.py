# Author: Samuel Lozano
from __future__ import annotations

import importlib

import numpy as np

def make_discrete(n: int):
    try:
        discrete_cls = importlib.import_module("gymnasium.spaces").Discrete
    except Exception:
        discrete_cls = importlib.import_module("gym.spaces").Discrete
    return discrete_cls(n)

def make_box(low: np.ndarray, high: np.ndarray):
    try:
        box_cls = importlib.import_module("gymnasium.spaces").Box
    except Exception:
        box_cls = importlib.import_module("gym.spaces").Box
    return box_cls(low=low, high=high, dtype=np.float32)
