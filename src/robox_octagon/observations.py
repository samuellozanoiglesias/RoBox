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
    stim_flag = 1.0 if env.phase == "choice" and t_stimulus_onset is not None else 0.0
    trial_type_flag = 1.0 if env.trial_type == "choice" else 0.0
    context_flag = 1.0 if env.context == "social" else 0.0

    dist2high = 0.0
    dist2low = 0.0
    angle2high_sin = 0.0
    angle2high_cos = 1.0
    angle2low_sin = 0.0
    angle2low_cos = 1.0
    high_patch_angle_sin = 0.0
    high_patch_angle_cos = 1.0
    low_patch_angle_sin = 0.0
    low_patch_angle_cos = 1.0
    patch_separation = 0.0

    opp_dist2high = 0.0
    opp_dist2low = 0.0
    opp_pos_x = 0.0
    opp_pos_y = 0.0
    opp_speed = 0.0

    if env.context == "social":
        opp_id = 1 - agent_id
        opp_pos = env.positions[opp_id] / env.inradius
        opp_pos_x = float(opp_pos[0])
        opp_pos_y = float(opp_pos[1])
        opp_speed = float(np.mean(env.speed_history[opp_id]))

    high_idx: Optional[int]
    low_idx: Optional[int]
    high_idx, low_idx = env._get_high_low_patch_indices()

    if stim_flag > 0.0 and high_idx is not None and low_idx is not None:
        anchor_positions = env.stimulus_onset_positions
        self_anchor = anchor_positions[agent_id]
        high_patch = env.patch_coords[high_idx]
        low_patch = env.patch_coords[low_idx]

        vec_high = high_patch - self_anchor
        vec_low = low_patch - self_anchor
        dist2high = float(np.linalg.norm(vec_high) / env.inradius)
        dist2low = float(np.linalg.norm(vec_low) / env.inradius)
        angle2high_sin, angle2high_cos = _sin_cos_from_vector(vec_high)
        angle2low_sin, angle2low_cos = _sin_cos_from_vector(vec_low)

        high_patch_angle = env.patch_angles_rad[high_idx]
        low_patch_angle = env.patch_angles_rad[low_idx]
        high_patch_angle_sin = float(np.sin(high_patch_angle))
        high_patch_angle_cos = float(np.cos(high_patch_angle))
        low_patch_angle_sin = float(np.sin(low_patch_angle))
        low_patch_angle_cos = float(np.cos(low_patch_angle))

        diff = abs(high_idx - low_idx)
        patch_separation = float(min(diff, 8 - diff))

        if env.context == "social":
            opp_id = 1 - agent_id
            opp_anchor = anchor_positions[opp_id]
            opp_dist2high = float(np.linalg.norm(high_patch - opp_anchor) / env.inradius)
            opp_dist2low = float(np.linalg.norm(low_patch - opp_anchor) / env.inradius)

    last5 = _last_k_rewards(env, agent_id=agent_id, k=5)

    obs = np.array(
        [
            dist2high,
            dist2low,
            angle2high_sin,
            angle2high_cos,
            angle2low_sin,
            angle2low_cos,
            float(pos[0]),
            float(pos[1]),
            opp_dist2high,
            opp_dist2low,
            opp_pos_x,
            opp_pos_y,
            opp_speed,
            high_patch_angle_sin,
            high_patch_angle_cos,
            low_patch_angle_sin,
            low_patch_angle_cos,
            patch_separation,
            trial_type_flag,
            context_flag,
            stim_flag,
            float(last5[0]),
            float(last5[1]),
            float(last5[2]),
            float(last5[3]),
            float(last5[4]),
        ],
        dtype=np.float32,
    )
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
