# Author: Samuel Lozano
from pathlib import Path
import importlib
import sys

_repo_src = Path(__file__).resolve().parent / "src"
if _repo_src.exists() and str(_repo_src) not in sys.path:
    sys.path.insert(0, str(_repo_src))

_pkg = importlib.import_module("robox_octagon")

OctagonEnv = _pkg.OctagonEnv
RewardShaper = _pkg.RewardShaper
TrialResult = _pkg.TrialResult
build_observation = _pkg.build_observation
build_global_state = _pkg.build_global_state

__all__ = [
    "OctagonEnv",
    "RewardShaper",
    "TrialResult",
    "build_observation",
    "build_global_state",
]
