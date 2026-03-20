# Author: Samuel Lozano
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .env import OctagonEnv


def _sin_cos_from_vector(vec: np.ndarray) -> Tuple[float, float]:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return 0.0, 1.0
    return float(vec[1] / norm), float(vec[0] / norm)


def _last_k_rewards(env: "OctagonEnv", agent_id: int, k: int = 5) -> np.ndarray:
    values = list(env.reward_history[agent_id])
    if len(values) < k:
        values = [0.0] * (k - len(values)) + values
    return np.asarray(values[-k:], dtype=np.float32)


def build_observation(env: "OctagonEnv", agent_id: int, t_stimulus_onset: Optional[float]):
    """Build observation vector for one agent with stimulus-anchored patch features."""
    if agent_id < 0 or agent_id >= env.n_agents:
        raise ValueError("agent_id out of bounds for current context")

    pos = env.positions[agent_id] / env.inradius
    heading = env.agent_headings[agent_id] if hasattr(env, 'agent_headings') else 0.0
    n_bins = 8  # 120º vision, split into 8 bins
    vision_angle = np.deg2rad(120)

    def vision_field_vector(type: str):
        vec = np.zeros(n_bins, dtype=np.float32)
        for i in range(n_bins):
            angle = heading - vision_angle/2 + i * vision_angle / n_bins
            direction = np.array([np.cos(angle), np.sin(angle)])
            if type == 'wall':
                # Raycast to wall, get distance
                # Placeholder: use min distance to wall
                vec[i] = env._distance_to_wall(pos, direction)
            elif type == 'high':
                high_idx, _ = env._get_high_low_patch_indices()
                if high_idx is not None:
                    patch_angle = env.patch_angles_rad[high_idx]
                    bin_idx = int(((patch_angle - heading + vision_angle/2) % vision_angle) / (vision_angle / n_bins))
                    if bin_idx == i:
                        vec[i] = 1.0
            elif type == 'low':
                _, low_idx = env._get_high_low_patch_indices()
                if low_idx is not None:
                    patch_angle = env.patch_angles_rad[low_idx]
                    bin_idx = int(((patch_angle - heading + vision_angle/2) % vision_angle) / (vision_angle / n_bins))
                    if bin_idx == i:
                        vec[i] = 1.0
            elif type == 'agent':
                if env.context == 'social':
                    opp_id = 1 - agent_id
                    opp_pos = env.positions[opp_id] / env.inradius
                    rel_vec = opp_pos - pos
                    rel_angle = np.arctan2(rel_vec[1], rel_vec[0])
                    bin_idx = int(((rel_angle - heading + vision_angle/2) % vision_angle) / (vision_angle / n_bins))
                    if bin_idx == i:
                        vec[i] = 1.0
        return vec

    obs_wall = vision_field_vector('wall')
    obs_high = vision_field_vector('high')
    obs_low = vision_field_vector('low')
    obs_agent = vision_field_vector('agent')

    obs = np.concatenate([obs_wall, obs_high, obs_low, obs_agent]).astype(np.float32)
    return obs


def build_global_state(env: "OctagonEnv", t_stimulus_onset: Optional[float]):
    """Centralized critic state: concat(obs_0, obs_1, [trial_type, context_flag, timestep_in_trial])."""
    obs0 = build_observation(env, agent_id=0, t_stimulus_onset=t_stimulus_onset)
    if env.n_agents > 1:
        obs1 = build_observation(env, agent_id=1, t_stimulus_onset=t_stimulus_onset)
    else:
        obs1 = np.zeros_like(obs0)

    trial_type_flag = 1.0 if env.trial_type == "choice" else 0.0
    context_flag = 1.0 if env.context == "social" else 0.0
    timestep_in_trial = float(env.trial_elapsed)

    return np.concatenate(
        [
            obs0,
            obs1,
            np.array([trial_type_flag, context_flag, timestep_in_trial], dtype=np.float32),
        ],
        axis=0,
    )
