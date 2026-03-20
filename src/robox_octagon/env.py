# Author: Samuel Lozano
from __future__ import annotations

from collections import deque
import importlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .navigation import NavigationController, estimate_travel_time
from .observations import build_global_state, build_observation
from .rewards import RewardShaper
from .spaces import make_box
from .trial_types import TrialResult


class OctagonEnv:
    """2D multi-agent competitive foraging environment in a regular octagon."""

    def __init__(
        self,
        dt: float = 0.05,
        inradius: float = 1.0,
        max_speed: float = 0.5,
        max_trials: int = 150,
        seed: Optional[int] = None,
    ) -> None:
        print(f"[DEBUG - env.py] Initializing OctagonEnv with dt={dt}, inradius={inradius}, max_speed={max_speed}, max_trials={max_trials}, seed={seed}")
        self.dt = float(dt)
        self.inradius = float(inradius)
        self.max_speed = float(max_speed)
        self.max_step = self.max_speed * self.dt
        self.max_trials = int(max_trials)
        self.choice_timeout = 30.0
        self.choice_radius = 0.05
        self.use_shaped_rewards = True
        self.shaping_toward_bonus = 0.01
        self.shaping_inaction_penalty = -0.005
        self.reward_shaper = RewardShaper(use_normalized=True)
        self.navigation_controller = NavigationController(
            inradius=self.inradius,
            max_speed=self.max_speed,
        )

        self.rng = np.random.default_rng(seed)

        self.patch_angles_deg = np.arange(0, 360, 45, dtype=float)
        self.patch_angles_rad = np.deg2rad(self.patch_angles_deg)
        self.patch_coords = np.stack(
            [np.cos(self.patch_angles_rad), np.sin(self.patch_angles_rad)], axis=1
        ) * self.inradius

        self.wall_normals = self.patch_coords / self.inradius
        self.valid_patch_pairs = self._build_valid_patch_pairs()

        self.context = "solo"
        self.n_agents = 1

        self.positions = np.zeros((1, 2), dtype=float)
        self.background_flag = 0
        self.session_time = 0.0
        self.session_done = False

        self.trial_count = 0
        self.phase = "iti"
        self.phase_elapsed = 0.0
        self.iti_duration = 0.0
        self.pre_stim_delay = 0.0

        self.trial_type = "choice"
        self.active_patch_indices: Tuple[int, int] = (-1, -1)
        self.active_patch_roles: Tuple[str, str] = ("low", "high")
        self.r_high = 3
        self.p_low = 0.3

        self.current_trial_response_time: Optional[float] = None
        self.current_trial_winner: Optional[int] = None
        self.current_trial_choice_patch: Optional[int] = None
        self.current_trial_choice_role: Optional[str] = None
        self.current_trial_travel_distance: Optional[float] = None
        self.current_trial_rewards = np.zeros(1, dtype=float)
        self.current_trial_raw_rewards = np.zeros(1, dtype=float)
        self.current_trial_shaped_rewards = np.zeros(1, dtype=float)
        self.current_trial_travel_times = np.zeros(1, dtype=float)
        self.current_trial_arrival_times = np.full((1, 2), np.inf, dtype=float)
        self.current_agent_arrival_patch = np.full(1, -1, dtype=int)
        self.current_agent_travel_distance = np.zeros(1, dtype=float)
        self.current_trial_reward_details: List[Dict[str, object]] = []
        self.stimulus_onset_positions = np.zeros((1, 2), dtype=float)
        self.stimulus_onset_speeds = np.zeros(1, dtype=float)
        self.response_positions = np.zeros((1, 2), dtype=float)
        self.t_stimulus_onset: Optional[float] = None
        self.trial_elapsed = 0.0
        self.current_target_patch = np.full(1, -1, dtype=int)
        self.target_selection_done = False
        self.waiting_state = np.zeros(1, dtype=bool)

        self.agent_value_maps: List[Dict[str, str]] = []
        self.trial_log: List[Dict[str, object]] = []
        self.trial_schedule: List[str] = []
        self.speed_history: List[deque] = []
        self.reward_history: List[deque] = []

        self.obs_dim = 32  # 4 vision vectors x 8 bins
        low = np.zeros(self.obs_dim, dtype=np.float32)
        high = np.ones(self.obs_dim, dtype=np.float32) * 10.0
        self.observation_space = make_box(low=low, high=high)
        global_low = np.concatenate([low, low, np.array([0.0, 0.0, 0.0], dtype=np.float32)])
        global_high = np.concatenate(
            [
                high,
                high,
                np.array([1.0, 1.0, float(self.choice_timeout + 12.0 + 1.5)], dtype=np.float32),
            ]
        )
        self.global_state_space = make_box(low=global_low, high=global_high)

        # Action space: 8 directions (walls), turn left, turn right
        from .spaces import make_discrete
        self.action_dim = 10
        self.action_space = make_discrete(self.action_dim)

        # Agent state tracking
        self.agent_states = ["waiting"] * self.n_agents  # "moving", "waiting"
        self.agent_paths = [[] for _ in range(self.n_agents)]
        self.agent_prev_vision = [None] * self.n_agents
        self.agent_headings = np.zeros(self.n_agents, dtype=np.float32)

    def _distance_to_wall(self, pos: np.ndarray, direction: np.ndarray) -> float:
        """Compute minimum distance from pos in direction to octagon wall."""
        min_dist = np.inf
        for i in range(len(self.patch_coords)):
            wall_point = self.patch_coords[i]
            normal = self.wall_normals[i]
            denom = np.dot(direction, normal)
            if abs(denom) < 1e-8:
                continue  # Ray is parallel to wall
            t = np.dot(wall_point - pos, normal) / denom
            if t > 0:
                min_dist = min(min_dist, t)
        return float(min_dist) if np.isfinite(min_dist) else 0.0
    
    def _get_agent_vision(self, agent_id: int):
        from .observations import build_observation
        return build_observation(self, agent_id, self.t_stimulus_onset)

    def set_context(self, context: str) -> None:
        print(f"[DEBUG - env.py] Setting context: {context}")
        context = context.lower().strip()
        if context not in {"solo", "social"}:
            raise ValueError("context must be 'solo' or 'social'")
        self.context = context
        self.n_agents = 1 if context == "solo" else 2

    def reset(self, context: str = "solo"):
        print(f"[DEBUG - env.py] Resetting environment with context: {context}")
        self.set_context(context)

        self.positions = np.zeros((self.n_agents, 2), dtype=float)
        for i in range(self.n_agents):
            self.positions[i] = self._sample_point_in_octagon()
            print(f"[DEBUG - env.py] Agent {i} initial position: {self.positions[i]}")

        self.agent_value_maps = [self._sample_value_map() for _ in range(self.n_agents)]
        print(f"[DEBUG - env.py] Agent value maps: {self.agent_value_maps}")

        self.session_time = 0.0
        self.session_done = False
        self.trial_count = 0
        self.trial_log = []
        self.trial_schedule = self._build_trial_schedule()
        print(f"[DEBUG - env.py] Trial schedule: {self.trial_schedule}")
        self.speed_history = [deque([0.0] * 10, maxlen=10) for _ in range(self.n_agents)]
        self.reward_history = [deque(maxlen=5) for _ in range(self.n_agents)]
        self.current_trial_arrival_times = np.full((self.n_agents, 2), np.inf, dtype=float)
        self.current_agent_arrival_patch = np.full(self.n_agents, -1, dtype=int)
        self.current_agent_travel_distance = np.zeros(self.n_agents, dtype=float)
        self.current_target_patch = np.full(self.n_agents, -1, dtype=int)
        self.target_selection_done = False
        self.waiting_state = np.zeros(self.n_agents, dtype=bool)

        self.agent_states = ["waiting"] * self.n_agents
        self.agent_paths = [[] for _ in range(self.n_agents)]
        self.agent_prev_vision = [None] * self.n_agents
        self.agent_headings = np.zeros(self.n_agents, dtype=np.float32)

        self._start_new_trial()

        return self._get_observation()

    def step(self, actions: Sequence[Sequence[float]]):
        print(f"[DEBUG - env.py] step() called with actions: {actions}")
        if self.session_done:
            print(f"[DEBUG - env.py] Session done, returning zeros.")
            rewards = np.zeros(self.n_agents, dtype=float)
            dones = {
                "agents": np.ones(self.n_agents, dtype=bool),
                "__all__": True,
            }
            info = {
                "session_done": True,
                "message": "Session has ended. Call reset() to start a new session.",
            }
            return self._get_observation(), rewards, dones, info

        prev_positions = self.positions.copy()
        speeds = np.zeros(self.n_agents, dtype=float)

        # Ensure actions is always a sequence
        if isinstance(actions, int):
            actions = [actions]
        # Action handling
        for i in range(self.n_agents):
            ai = actions[i]
            if isinstance(ai, int):
                action = ai
            elif isinstance(ai, (list, tuple, np.ndarray)):
                action = int(ai[0])
            else:
                raise TypeError(f"Unsupported action type for agent {i}: {type(ai)}")
            print(f"[DEBUG - env.py] Agent {i} action: {action}, state: {self.agent_states[i]}")
            if self.agent_states[i] == "waiting":
                if action < 8:  # Choose wall direction
                    # Check if blocked by other agent
                    blocked = False
                    if self.context == "social":
                        opp_pos = self.positions[1-i]
                        # If opp is in the way, block
                        # (simple check: opp is between agent and wall)
                        # TODO: improve with vision field
                        pass
                    if not blocked:
                        # Plan path using A*
                        from .astar import AStarPathfinder
                        grid = np.zeros((8,8), dtype=int)  # Placeholder grid
                        start = tuple(np.round(self.positions[i]).astype(int))
                        goal = (action, action)  # Placeholder: wall index
                        pathfinder = AStarPathfinder(grid)
                        path = pathfinder.find_path(start, goal)
                        self.agent_paths[i] = path
                        self.agent_states[i] = "moving"
                        print(f"[DEBUG - env.py] Agent {i} path: {path}")
                elif action == 8:  # Turn left
                    self.agent_headings[i] -= np.deg2rad(120)
                    print(f"[DEBUG - env.py] Agent {i} turned left. Heading: {self.agent_headings[i]}")
                elif action == 9:  # Turn right
                    self.agent_headings[i] += np.deg2rad(120)
                    print(f"[DEBUG - env.py] Agent {i} turned right. Heading: {self.agent_headings[i]}")
                # After rotation, agent is prompted for new action next tick
            elif self.agent_states[i] == "moving":
                # Move along path
                if self.agent_paths[i]:
                    next_pos = np.array(self.agent_paths[i].pop(0), dtype=float)
                    self.positions[i] = next_pos
                    print(f"[DEBUG - env.py] Agent {i} moved to: {self.positions[i]}")
                    # Check for collision
                    for j in range(self.n_agents):
                        if i != j and np.allclose(self.positions[i], self.positions[j]):
                            self.agent_states[i] = "waiting"
                            self.agent_states[j] = "waiting"
                            print(f"[DEBUG - env.py] Collision detected between agent {i} and agent {j}")
                    # If reached wall, stop
                    if not self.agent_paths[i]:
                        self.agent_states[i] = "waiting"
                        print(f"[DEBUG - env.py] Agent {i} reached wall and is now waiting.")
                else:
                    self.agent_states[i] = "waiting"
                    print(f"[DEBUG - env.py] Agent {i} has no path, now waiting.")

        # Detect if other agent appears in vision field and was not there before
        for i in range(self.n_agents):
            vision = self._get_agent_vision(i)
            if self.agent_prev_vision[i] is not None:
                prev = self.agent_prev_vision[i]
                if vision[3].sum() > 0 and prev[3].sum() == 0:
                    self.agent_states[i] = "waiting"
                    print(f"[DEBUG - env.py] Agent {i} sees other agent in vision field.")
            self.agent_prev_vision[i] = vision

        self._update_speed_history(speeds)

        self.session_time += self.dt
        self.phase_elapsed += self.dt
        self.trial_elapsed += self.dt

        rewards = np.zeros(self.n_agents, dtype=float)
        info: Dict[str, object] = {
            "phase": self.phase,
            "trial_index": self.trial_count,
            "context": self.context,
            "trial_type": self.trial_type,
        }

        print(f"[DEBUG - env.py] Phase: {self.phase}, phase_elapsed: {self.phase_elapsed}")

        if self.phase == "iti":
            if self.phase_elapsed >= self.iti_duration:
                self.phase = "pre_stim"
                self.phase_elapsed = 0.0
                self.background_flag = 1
                info["event"] = "trial_start"
                print(f"[DEBUG - env.py] Transition to pre_stim phase.")

        elif self.phase == "pre_stim":
            if self.phase_elapsed >= self.pre_stim_delay:
                self.phase = "choice"
                self.phase_elapsed = 0.0
                self.stimulus_onset_positions = self.positions.copy()
                self.stimulus_onset_speeds = np.array(
                    [float(np.mean(self.speed_history[i])) for i in range(self.n_agents)],
                    dtype=float,
                )
                self.t_stimulus_onset = self.session_time
                info["event"] = "stimulus_onset"
                print(f"[DEBUG - env.py] Transition to choice phase.")

        elif self.phase == "choice":
            shaping_step = self._compute_step_shaping(prev_positions=prev_positions)
            print(f"[DEBUG - env.py] Shaping step: {shaping_step}")
            if self.use_shaped_rewards:
                rewards += shaping_step
            outcome = self._resolve_choice_phase(shaping_step=shaping_step)
            print(f"[DEBUG - env.py] Outcome: {outcome}")
            if outcome is not None:
                rewards += outcome
                info["event"] = "trial_end"
                info["winner_agent"] = self.current_trial_winner
                info["choice_patch"] = self.current_trial_choice_patch
                info["response_time"] = self.current_trial_response_time
                info["travel_distance"] = self.current_trial_travel_distance
                info["patch_chosen_role"] = self.current_trial_choice_role
                info["trial_result"] = self.trial_log[-1]
                print(f"[DEBUG - env.py] Trial ended. Winner: {self.current_trial_winner}, Choice patch: {self.current_trial_choice_patch}")
                self._start_new_trial()

        dones = {
            "agents": np.full(self.n_agents, self.session_done, dtype=bool),
            "__all__": bool(self.session_done),
        }
        print(f"[DEBUG - env.py] step() finished. Rewards: {rewards}, Dones: {dones}, Info: {info}")
        return self._get_observation(), rewards, dones, info

    def render(self):
        """Return a matplotlib figure showing geometry, patches, and agent positions."""
        plt = importlib.import_module("matplotlib.pyplot")

        fig, ax = plt.subplots(figsize=(6, 6))

        circumradius = self.inradius / np.cos(np.pi / 8.0)
        vertex_angles = np.deg2rad(np.arange(22.5, 360.0 + 22.5, 45.0))
        vertices = np.stack(
            [circumradius * np.cos(vertex_angles), circumradius * np.sin(vertex_angles)],
            axis=1,
        )
        ax.plot(vertices[:, 0], vertices[:, 1], color="black", linewidth=2)

        ax.scatter(self.patch_coords[:, 0], self.patch_coords[:, 1], s=40, color="gray")
        for idx, (x, y) in enumerate(self.patch_coords):
            ax.text(x + 0.02, y + 0.02, str(idx), fontsize=9)

        if self.phase == "choice":
            active = np.array(self.active_patch_indices, dtype=int)
            active_coords = self.patch_coords[active]
            ax.scatter(
                active_coords[:, 0],
                active_coords[:, 1],
                s=140,
                facecolors="none",
                edgecolors="orange",
                linewidths=2,
            )

        colors = ["tab:blue", "tab:red"]
        for i in range(self.n_agents):
            ax.scatter(
                self.positions[i, 0],
                self.positions[i, 1],
                s=90,
                color=colors[i],
                label=f"agent_{i}",
            )

        ax.set_title(
            f"OctagonEnv | context={self.context} | trial={self.trial_count}/{self.max_trials} | phase={self.phase}"
        )
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")

        return fig

    def _build_valid_patch_pairs(self) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int]] = []
        for i in range(8):
            for j in range(i + 1, 8):
                sep = min((j - i) % 8, (i - j) % 8)
                if sep in {1, 2, 3}:
                    pairs.append((i, j))
        return pairs

    def _build_trial_schedule(self) -> List[str]:
        n_choice = int(round(0.8 * self.max_trials))
        n_forced = self.max_trials - n_choice
        schedule = ["choice"] * n_choice + ["forced"] * n_forced
        self.rng.shuffle(schedule)
        return schedule

    def _build_observation_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        low = np.array(
            [
                0.0,
                0.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                0.0,
                0.0,
                -1.0,
                -1.0,
                0.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        high = np.array(
            [
                3.0,
                3.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                3.0,
                3.0,
                1.0,
                1.0,
                self.max_speed,
                1.0,
                1.0,
                1.0,
                1.0,
                3.0,
                1.0,
                1.0,
                1.0,
                5.0,
                5.0,
                5.0,
                5.0,
                5.0,
            ],
            dtype=np.float32,
        )
        return low, high

    def _format_free_movement_actions(self, actions: Sequence[Sequence[float]]) -> Optional[np.ndarray]:
        arr = np.asarray(actions)
        if arr.ndim == 0:
            return None

        try:
            arr = arr.astype(float)
        except Exception:
            return None

        if self.n_agents == 1:
            if arr.shape == (2,):
                arr = arr[None, :]
            elif arr.shape != (1, 2):
                return None
        else:
            if arr.shape != (2, 2):
                return None

        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        scale = np.ones_like(norms)
        over = norms > self.max_step
        scale[over] = self.max_step / norms[over]
        return arr * scale

    def _parse_patch_selection_actions(self, actions) -> np.ndarray:
        arr = np.asarray(actions)

        if self.n_agents == 1:
            if arr.ndim == 0:
                parsed = np.asarray([arr], dtype=int)
            elif arr.shape == (1,):
                parsed = arr.astype(int)
            elif arr.shape == (1, 1):
                parsed = arr.reshape(1).astype(int)
            else:
                parsed = np.asarray([8], dtype=int)
        else:
            if arr.shape == (2,):
                parsed = arr.astype(int)
            elif arr.shape == (2, 1):
                parsed = arr.reshape(2).astype(int)
            else:
                parsed = np.asarray([8, 8], dtype=int)

        return np.clip(parsed, 0, 8)

    def _set_trial_targets(self, actions) -> None:
        selections = self._parse_patch_selection_actions(actions)
        self.current_target_patch = np.full(self.n_agents, -1, dtype=int)
        self.waiting_state = np.zeros(self.n_agents, dtype=bool)

        for i in range(self.n_agents):
            choice = int(selections[i])
            if choice == 8:
                self.current_target_patch[i] = -1
                self.waiting_state[i] = True
            else:
                self.current_target_patch[i] = choice

    def _apply_navigation_step(self) -> np.ndarray:
        speeds = np.zeros(self.n_agents, dtype=float)
        for i in range(self.n_agents):
            if self.waiting_state[i]:
                continue

            target_idx = int(self.current_target_patch[i])
            target_pos = None
            if 0 <= target_idx < 8:
                target_pos = self.patch_coords[target_idx]

            prev = self.positions[i].copy()
            self.positions[i] = self.navigation_controller.step(
                agent_pos=self.positions[i],
                target_patch_pos=target_pos,
                dt=self.dt,
            )
            speeds[i] = float(np.linalg.norm(self.positions[i] - prev) / self.dt)
        return speeds

    def _sample_value_map(self) -> Dict[str, str]:
        if self.rng.uniform() < 0.5:
            return {"high": "grating", "low": "light"}
        return {"high": "light", "low": "grating"}

    def _sample_point_in_octagon(self) -> np.ndarray:
        while True:
            candidate = self.rng.uniform(-self.inradius, self.inradius, size=2)
            if self._point_in_octagon(candidate):
                return candidate

    def _point_in_octagon(self, point: np.ndarray) -> bool:
        return bool(np.all(self.wall_normals @ point <= self.inradius + 1e-12))

    def _format_actions(self, actions: Sequence[Sequence[float]]) -> np.ndarray:
        arr = np.asarray(actions, dtype=float)
        if self.n_agents == 1:
            if arr.shape == (2,):
                arr = arr[None, :]
            elif arr.shape != (1, 2):
                raise ValueError("solo mode expects action shape (2,) or (1, 2)")
        else:
            if arr.shape != (2, 2):
                raise ValueError("social mode expects action shape (2, 2)")

        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        scale = np.ones_like(norms)
        over = norms > self.max_step
        scale[over] = self.max_step / norms[over]
        return arr * scale

    def _move_agents(self, deltas: np.ndarray) -> np.ndarray:
        speeds = np.zeros(self.n_agents, dtype=float)
        for i in range(self.n_agents):
            current = self.positions[i]
            proposed = current + deltas[i]
            if self._point_in_octagon(proposed):
                self.positions[i] = proposed
                displacement = self.positions[i] - current
                speeds[i] = float(np.linalg.norm(displacement) / self.dt)
                continue

            lo = 0.0
            hi = 1.0
            for _ in range(25):
                mid = 0.5 * (lo + hi)
                candidate = current + mid * deltas[i]
                if self._point_in_octagon(candidate):
                    lo = mid
                else:
                    hi = mid
            self.positions[i] = current + lo * deltas[i]
            displacement = self.positions[i] - current
            speeds[i] = float(np.linalg.norm(displacement) / self.dt)
        return speeds

    def _update_speed_history(self, speeds: np.ndarray) -> None:
        for i in range(self.n_agents):
            self.speed_history[i].append(float(speeds[i]))

    def _update_reward_history(self) -> None:
        for i in range(self.n_agents):
            self.reward_history[i].append(float(self.current_trial_rewards[i]))

    def _get_high_low_patch_indices(self) -> Tuple[Optional[int], Optional[int]]:
        if self.phase != "choice" or self.active_patch_indices[0] < 0:
            return None, None

        p0, p1 = self.active_patch_indices
        r0, r1 = self.active_patch_roles
        if r0 == "high" and r1 == "low":
            return p0, p1
        if r0 == "low" and r1 == "high":
            return p1, p0
        return p0, p1

    def _compute_step_shaping(self, prev_positions: np.ndarray) -> np.ndarray:
        shaping = np.zeros(self.n_agents, dtype=float)
        if self.phase != "choice":
            return shaping
        if not self.use_shaped_rewards:
            return shaping

        active_coords = self.patch_coords[np.array(self.active_patch_indices, dtype=int)]
        for i in range(self.n_agents):
            prev_pos = prev_positions[i]
            curr_pos = self.positions[i]
            step_distance = float(np.linalg.norm(curr_pos - prev_pos))

            if step_distance < 1e-8:
                shaping[i] += self.shaping_inaction_penalty
                continue

            prev_nearest = float(np.min(np.linalg.norm(active_coords - prev_pos, axis=1)))
            curr_nearest = float(np.min(np.linalg.norm(active_coords - curr_pos, axis=1)))
            if curr_nearest < prev_nearest:
                shaping[i] += self.shaping_toward_bonus

        return shaping

    def _update_arrival_times(self) -> None:
        active_coords = self.patch_coords[np.array(self.active_patch_indices, dtype=int)]
        dists = np.linalg.norm(self.positions[:, None, :] - active_coords[None, :, :], axis=2)
        within = dists <= self.choice_radius

        for i in range(self.n_agents):
            for p in range(2):
                if within[i, p] and np.isinf(self.current_trial_arrival_times[i, p]):
                    self.current_trial_arrival_times[i, p] = float(self.phase_elapsed)
                    patch_global = int(self.active_patch_indices[p])
                    self.current_agent_arrival_patch[i] = patch_global
                    self.current_agent_travel_distance[i] = float(
                        np.linalg.norm(self.patch_coords[patch_global] - self.stimulus_onset_positions[i])
                        / self.inradius
                    )
                    self.current_trial_travel_times[i] = float(
                        estimate_travel_time(
                            start_pos=self.stimulus_onset_positions[i],
                            target_pos=self.patch_coords[patch_global],
                            max_speed=self.max_speed,
                        )
                    )
                    self.waiting_state[i] = True

    def _determine_winner_from_arrivals(self) -> Tuple[Optional[int], Optional[int], Optional[float]]:
        best: Optional[Tuple[float, int, int]] = None
        for i in range(self.n_agents):
            for p in range(2):
                t = float(self.current_trial_arrival_times[i, p])
                if not np.isfinite(t):
                    continue
                candidate = (t, i, p)
                if best is None or candidate < best:
                    best = candidate

        if best is None:
            return None, None, None
        return int(best[1]), int(best[2]), float(best[0])

    def _start_new_trial(self) -> None:
        if self.trial_count >= self.max_trials:
            self.session_done = True
            self.phase = "done"
            return

        schedule_index = self.trial_count
        self.trial_count += 1
        self.phase = "iti"
        self.phase_elapsed = 0.0
        self.trial_elapsed = 0.0
        self.background_flag = 0
        self.iti_duration = float(self.rng.uniform(6.0, 12.0))
        self.pre_stim_delay = float(self.rng.uniform(0.5, 1.5))
        self.t_stimulus_onset = None

        self.current_trial_response_time = None
        self.current_trial_winner = None
        self.current_trial_choice_patch = None
        self.current_trial_choice_role = None
        self.current_trial_travel_distance = None
        self.current_trial_rewards = np.zeros(self.n_agents, dtype=float)
        self.current_trial_raw_rewards = np.zeros(self.n_agents, dtype=float)
        self.current_trial_shaped_rewards = np.zeros(self.n_agents, dtype=float)
        self.current_trial_travel_times = np.zeros(self.n_agents, dtype=float)
        self.current_trial_arrival_times = np.full((self.n_agents, 2), np.inf, dtype=float)
        self.current_agent_arrival_patch = np.full(self.n_agents, -1, dtype=int)
        self.current_agent_travel_distance = np.zeros(self.n_agents, dtype=float)
        self.current_trial_reward_details = []
        self.stimulus_onset_positions = np.zeros((self.n_agents, 2), dtype=float)
        self.stimulus_onset_speeds = np.zeros(self.n_agents, dtype=float)
        self.response_positions = np.zeros((self.n_agents, 2), dtype=float)
        self.current_target_patch = np.full(self.n_agents, -1, dtype=int)
        self.target_selection_done = False
        self.waiting_state = np.zeros(self.n_agents, dtype=bool)

        self.trial_type = self.trial_schedule[schedule_index]
        self.active_patch_indices = self.valid_patch_pairs[
            int(self.rng.integers(0, len(self.valid_patch_pairs)))
        ]

        if self.trial_type == "choice":
            self.r_high = int(self.rng.integers(3, 6))
            self.p_low = float(self.rng.uniform(0.3, 0.6))
            high_pos = int(self.rng.integers(0, 2))
            if high_pos == 0:
                self.active_patch_roles = ("high", "low")
            else:
                self.active_patch_roles = ("low", "high")
        else:
            forced_value = "high" if self.rng.uniform() < 0.5 else "low"
            if forced_value == "high":
                self.r_high = int(self.rng.integers(3, 6))
                self.p_low = float(self.rng.uniform(0.3, 0.6))
                self.active_patch_roles = ("high", "high")
            else:
                self.r_high = int(self.rng.integers(3, 6))
                self.p_low = float(self.rng.uniform(0.3, 0.6))
                self.active_patch_roles = ("low", "low")

    def _resolve_choice_phase(self, shaping_step: np.ndarray) -> Optional[np.ndarray]:
        if self.use_shaped_rewards:
            self.current_trial_shaped_rewards += shaping_step

        self._update_arrival_times()
        winner, winner_local_patch, winner_time = self._determine_winner_from_arrivals()

        if winner is not None and winner_local_patch is not None and winner_time is not None:
            chosen_patch_global = int(self.active_patch_indices[winner_local_patch])
            chosen_role = self.active_patch_roles[winner_local_patch]
            travel_distance = float(
                np.linalg.norm(
                    self.patch_coords[chosen_patch_global] - self.stimulus_onset_positions[winner]
                )
                / self.inradius
            )
            travel_time = estimate_travel_time(
                start_pos=self.stimulus_onset_positions[winner],
                target_pos=self.patch_coords[chosen_patch_global],
                max_speed=self.max_speed,
            )
            self.current_trial_travel_times[winner] = float(travel_time)

            raw_rewards = np.zeros(self.n_agents, dtype=float)
            for agent_id in range(self.n_agents):
                raw_rewards[agent_id] = self.reward_shaper.compute_reward(
                    agent_id=agent_id,
                    winner_id=winner if self.context == "social" else 0,
                    choice=chosen_role,
                    travel_dist=travel_distance,
                    context=self.context,
                )

            self.current_trial_raw_rewards = raw_rewards
            self.current_trial_rewards = self.current_trial_raw_rewards + self.current_trial_shaped_rewards
            self.current_trial_winner = winner if self.context == "social" else None
            self.current_trial_choice_patch = chosen_patch_global
            self.current_trial_choice_role = chosen_role
            self.current_trial_response_time = float(winner_time)
            self.current_trial_travel_distance = travel_distance
            self.response_positions = self.positions.copy()

            self.current_trial_reward_details = []
            for agent_id in range(self.n_agents):
                self.current_trial_reward_details.append(
                    {
                        "agent_id": agent_id,
                        "winner_id": self.current_trial_winner,
                        "patch_chosen": chosen_role,
                        "chosen_patch_id": int(self.current_agent_arrival_patch[agent_id]),
                        "RT": self.current_trial_response_time,
                        "travel_distance": float(self.current_agent_travel_distance[agent_id]),
                        "travel_time": float(self.current_trial_travel_times[agent_id]),
                        "raw_reward": float(self.current_trial_raw_rewards[agent_id]),
                        "shaped_reward": float(self.current_trial_rewards[agent_id]),
                    }
                )

            self._update_reward_history()
            self._append_trial_log()
            return raw_rewards

        if self.phase_elapsed >= self.choice_timeout:
            raw_rewards = np.zeros(self.n_agents, dtype=float)
            self.current_trial_raw_rewards = raw_rewards
            self.current_trial_rewards = self.current_trial_raw_rewards + self.current_trial_shaped_rewards
            self.current_trial_winner = None
            self.current_trial_choice_patch = None
            self.current_trial_choice_role = None
            self.current_trial_response_time = None
            self.current_trial_travel_distance = None
            self.response_positions = self.positions.copy()

            self.current_trial_reward_details = []
            for agent_id in range(self.n_agents):
                self.current_trial_reward_details.append(
                    {
                        "agent_id": agent_id,
                        "winner_id": None,
                        "patch_chosen": None,
                        "chosen_patch_id": int(self.current_agent_arrival_patch[agent_id]),
                        "RT": None,
                        "travel_distance": None,
                        "travel_time": None,
                        "raw_reward": 0.0,
                        "shaped_reward": float(self.current_trial_rewards[agent_id]),
                    }
                )

            self._update_reward_history()
            self._append_trial_log()
            return raw_rewards

        return None

    def _append_trial_log(self) -> None:
        record = TrialResult(
            trial_index=self.trial_count,
            trial_type=self.trial_type,
            patch_indices=self.active_patch_indices,
            patch_roles=self.active_patch_roles,
            r_high=self.r_high,
            p_low=self.p_low,
            stimulus_onset_positions=self.stimulus_onset_positions.tolist(),
            stimulus_onset_speeds=self.stimulus_onset_speeds.tolist(),
            response_positions=self.response_positions.tolist(),
            choice_patch=self.current_trial_choice_patch,
            choice_role=self.current_trial_choice_role,
            response_time=self.current_trial_response_time,
            winner_agent=self.current_trial_winner if self.context == "social" else None,
            travel_distance=self.current_trial_travel_distance,
            rewards=self.current_trial_rewards.tolist(),
            raw_rewards=self.current_trial_raw_rewards.tolist(),
            shaped_rewards=self.current_trial_rewards.tolist(),
            arrival_times=self.current_trial_arrival_times.tolist(),
            reward_details=[dict(item) for item in self.current_trial_reward_details],
        )
        self.trial_log.append(record.__dict__)

    def _get_observation(self) -> Dict[str, object]:
        active_mask = np.zeros(8, dtype=np.int8)
        active_coords = np.zeros((2, 2), dtype=float)
        active_indices = np.array([-1, -1], dtype=np.int8)

        if self.phase == "choice":
            idx = np.array(self.active_patch_indices, dtype=np.int8)
            active_mask[idx] = 1
            active_coords = self.patch_coords[idx].copy()
            active_indices = idx

        per_agent_obs = np.stack(
            [
                build_observation(self, agent_id=i, t_stimulus_onset=self.t_stimulus_onset)
                for i in range(self.n_agents)
            ],
            axis=0,
        )
        global_state = build_global_state(self, t_stimulus_onset=self.t_stimulus_onset)

        return {
            "positions": self.positions.copy(),
            "background_flag": int(self.background_flag),
            "context": self.context,
            "phase": self.phase,
            "trial_index": int(self.trial_count),
            "trial_type": self.trial_type,
            "patch_coords": self.patch_coords.copy(),
            "active_patch_mask": active_mask,
            "active_patch_indices": active_indices,
            "active_patch_coords": active_coords,
            "value_maps": [dict(m) for m in self.agent_value_maps],
            "stimulus_present": int(self.phase == "choice"),
            "agent_obs": per_agent_obs,
            "global_state": global_state,
        }
