# Author: Samuel Lozano
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class TrialResult:
    trial_index: int
    trial_type: str
    patch_indices: Tuple[int, int]
    patch_roles: Tuple[str, str]
    r_high: int
    p_low: float
    stimulus_onset_positions: List[List[float]]
    stimulus_onset_speeds: List[float]
    response_positions: List[List[float]]
    choice_patch: Optional[int]
    choice_role: Optional[str]
    response_time: Optional[float]
    winner_agent: Optional[int]
    travel_distance: Optional[float]
    rewards: List[float]
    raw_rewards: List[float]
    shaped_rewards: List[float]
    arrival_times: List[List[float]]
    reward_details: List[Dict[str, object]]
