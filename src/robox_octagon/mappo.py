from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class MAPPOActor(nn.Module):
    """Actor with discrete patch-choice head and continuous movement-direction head."""

    def __init__(self, obs_dim: int, action_dim: int = 9) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        self.backbone = nn.Sequential(
            nn.Linear(self.obs_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
        )
        self.discrete_head = nn.Linear(64, self.action_dim)
        self.continuous_head = nn.Sequential(
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        logits = self.discrete_head(h)
        delta = self.continuous_head(h)
        return logits, delta


class MAPPOCritic(nn.Module):
    """Centralized critic V(global_state)."""

    def __init__(self, global_state_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state)


@dataclass
class RolloutStats:
    steps_collected: int
    trials_completed: int
    trial_high_count: int
    trial_choice_count: int
    reward_sum: float
    trial_results: List[Dict[str, object]]


class MAPPOBuffer:
    """Rollout storage for MAPPO with per-local-agent trajectories."""

    def __init__(self, actor_ids: List[int], obs_dim: int, global_state_dim: int) -> None:
        self.actor_ids = list(actor_ids)
        self.n_local_agents = len(actor_ids)
        self.obs_dim = int(obs_dim)
        self.global_state_dim = int(global_state_dim)

        self.obs: List[List[np.ndarray]] = [[] for _ in range(self.n_local_agents)]
        self.global_state: List[List[np.ndarray]] = [[] for _ in range(self.n_local_agents)]
        self.action_discrete: List[List[int]] = [[] for _ in range(self.n_local_agents)]
        self.action_continuous: List[List[np.ndarray]] = [[] for _ in range(self.n_local_agents)]
        self.log_prob: List[List[float]] = [[] for _ in range(self.n_local_agents)]
        self.reward: List[List[float]] = [[] for _ in range(self.n_local_agents)]
        self.done: List[List[float]] = [[] for _ in range(self.n_local_agents)]
        self.value: List[List[float]] = [[] for _ in range(self.n_local_agents)]
        self.policy_active: List[List[float]] = [[] for _ in range(self.n_local_agents)]
        self.last_value = np.zeros(self.n_local_agents, dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        global_state: np.ndarray,
        action_discrete: np.ndarray,
        action_continuous: np.ndarray,
        log_prob: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        policy_active: np.ndarray,
    ) -> None:
        for i in range(self.n_local_agents):
            self.obs[i].append(np.asarray(obs[i], dtype=np.float32))
            self.global_state[i].append(np.asarray(global_state, dtype=np.float32))
            self.action_discrete[i].append(int(action_discrete[i]))
            self.action_continuous[i].append(np.asarray(action_continuous[i], dtype=np.float32))
            self.log_prob[i].append(float(log_prob[i]))
            self.reward[i].append(float(reward[i]))
            self.done[i].append(float(done[i]))
            self.value[i].append(float(value[i]))
            self.policy_active[i].append(float(policy_active[i]))

    def set_last_value(self, values: np.ndarray) -> None:
        self.last_value = np.asarray(values, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.reward[0]) if self.n_local_agents > 0 else 0


class MAPPOTrainer:
    """MAPPO trainer with solo pretraining and interleaved solo/social curriculum."""

    def __init__(
        self,
        obs_dim: int,
        global_state_dim: int,
        action_dim: int = 9,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        clip_eps: float = 0.2,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        n_epochs: int = 10,
        batch_size: int = 64,
        rollout_length: int = 2048,
        entropy_coeff: float = 0.01,
        value_loss_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        checkpoint_dir: str = "checkpoints",
        device: Optional[str] = None,
        actor_seeds: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.global_state_dim = int(global_state_dim)
        self.action_dim = int(action_dim)

        self.gamma = float(gamma)
        self.lambda_gae = float(lambda_gae)
        self.clip_eps = float(clip_eps)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.rollout_length = int(rollout_length)
        self.entropy_coeff = float(entropy_coeff)
        self.value_loss_coeff = float(value_loss_coeff)
        self.max_grad_norm = float(max_grad_norm)

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if actor_seeds is not None:
            torch.manual_seed(int(actor_seeds[0]))
        actor0 = MAPPOActor(obs_dim=self.obs_dim, action_dim=self.action_dim).to(self.device)
        if actor_seeds is not None:
            torch.manual_seed(int(actor_seeds[1]))
        actor1 = MAPPOActor(obs_dim=self.obs_dim, action_dim=self.action_dim).to(self.device)

        self.actors = [actor0, actor1]
        self.critic = MAPPOCritic(global_state_dim=self.global_state_dim).to(self.device)

        self.actor_opts = [
            torch.optim.Adam(self.actors[0].parameters(), lr=lr_actor),
            torch.optim.Adam(self.actors[1].parameters(), lr=lr_actor),
        ]
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log: Dict[str, List] = {
            "trial": [],
            "phase": [],
            "p_high": [],
            "actor_loss": [],
            "critic_loss": [],
            "entropy": [],
            "mean_reward": [],
        }

        self.total_trials = 0
        self.current_obs: Optional[Dict[str, object]] = None
        self.current_context = "solo"
        self.current_solo_agent = 0

    def _select_action(
        self,
        actor: MAPPOActor,
        obs_vec: np.ndarray,
    ) -> Tuple[int, np.ndarray, float]:
        obs_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, delta = actor(obs_t)
            dist = Categorical(logits=logits)
            discrete_action = int(dist.sample().item())
            log_prob = float(dist.log_prob(torch.tensor([discrete_action], device=self.device)).item())
            continuous = delta.squeeze(0).cpu().numpy()
        return discrete_action, continuous.astype(np.float32), log_prob

    def collect_rollout(self, env, n_steps: int) -> Tuple[MAPPOBuffer, RolloutStats]:
        n_steps = int(n_steps)
        if self.current_obs is None:
            self.current_obs = env.reset(context=self.current_context)

        if self.current_context == "social":
            actor_ids = [0, 1]
        else:
            actor_ids = [self.current_solo_agent]

        buffer = MAPPOBuffer(actor_ids=actor_ids, obs_dim=self.obs_dim, global_state_dim=self.global_state_dim)

        trials_completed = 0
        high_count = 0
        choice_count = 0
        reward_sum = 0.0
        trial_results: List[Dict[str, object]] = []

        selected_trial_idx = np.full(len(actor_ids), -1, dtype=np.int64)
        selected_discrete = np.full(len(actor_ids), 8, dtype=np.int64)
        selected_continuous = np.zeros((len(actor_ids), 2), dtype=np.float32)
        selected_logprob = np.zeros(len(actor_ids), dtype=np.float32)

        for _ in range(n_steps):
            obs_dict = self.current_obs
            agent_obs = np.asarray(obs_dict["agent_obs"], dtype=np.float32)
            global_state = np.asarray(obs_dict["global_state"], dtype=np.float32)
            phase = str(obs_dict.get("phase", "iti"))
            trial_index = int(obs_dict.get("trial_index", 0))

            with torch.no_grad():
                gs_t = torch.as_tensor(global_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                v_pred = float(self.critic(gs_t).squeeze(0).item())

            local_n = len(actor_ids)
            discrete_actions = np.zeros(local_n, dtype=np.int64)
            continuous_actions = np.zeros((local_n, 2), dtype=np.float32)
            log_probs = np.zeros(local_n, dtype=np.float32)
            values = np.full(local_n, v_pred, dtype=np.float32)
            policy_active = np.zeros(local_n, dtype=np.float32)

            for local_idx, actor_id in enumerate(actor_ids):
                if phase == "choice":
                    if selected_trial_idx[local_idx] != trial_index:
                        act, cont, lp = self._select_action(
                            actor=self.actors[actor_id],
                            obs_vec=agent_obs[local_idx],
                        )
                        selected_trial_idx[local_idx] = trial_index
                        selected_discrete[local_idx] = int(act)
                        selected_continuous[local_idx] = cont
                        selected_logprob[local_idx] = float(lp)

                    discrete_actions[local_idx] = int(selected_discrete[local_idx])
                    continuous_actions[local_idx] = np.asarray(
                        selected_continuous[local_idx], dtype=np.float32
                    )
                    log_probs[local_idx] = float(selected_logprob[local_idx])
                    policy_active[local_idx] = 1.0
                else:
                    selected_trial_idx[local_idx] = -1
                    selected_discrete[local_idx] = 8
                    selected_continuous[local_idx] = np.zeros(2, dtype=np.float32)
                    selected_logprob[local_idx] = 0.0

                    discrete_actions[local_idx] = 8
                    continuous_actions[local_idx] = np.zeros(2, dtype=np.float32)
                    log_probs[local_idx] = 0.0

            if self.current_context == "social":
                env_action_arr = np.asarray(discrete_actions, dtype=np.int64)
            else:
                env_action_arr = int(discrete_actions[0])

            next_obs, rewards, dones, info = env.step(env_action_arr)
            rewards = np.asarray(rewards, dtype=np.float32)
            local_done = np.asarray(dones["agents"], dtype=np.float32)

            buffer.add(
                obs=agent_obs,
                global_state=global_state,
                action_discrete=discrete_actions,
                action_continuous=continuous_actions,
                log_prob=log_probs,
                reward=rewards,
                done=local_done,
                value=values,
                policy_active=policy_active,
            )

            reward_sum += float(np.mean(rewards))

            if info.get("event") == "trial_end":
                trials_completed += 1
                trial_result = info.get("trial_result", {})
                if isinstance(trial_result, dict):
                    trial_results.append(dict(trial_result))
                    if trial_result.get("trial_type") == "choice":
                        choice_count += 1
                        if trial_result.get("choice_role") == "high":
                            high_count += 1
                self.total_trials += 1
                if self.total_trials % 1000 == 0:
                    self._save_checkpoint(trial=self.total_trials)

            if bool(dones["__all__"]):
                next_obs = env.reset(context=self.current_context)

            self.current_obs = next_obs

        # Bootstrap value at final state for GAE.
        obs_dict = self.current_obs
        global_state = np.asarray(obs_dict["global_state"], dtype=np.float32)
        with torch.no_grad():
            gs_t = torch.as_tensor(global_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            v_last = float(self.critic(gs_t).squeeze(0).item())
        buffer.set_last_value(np.full(len(actor_ids), v_last, dtype=np.float32))

        stats = RolloutStats(
            steps_collected=n_steps,
            trials_completed=trials_completed,
            trial_high_count=high_count,
            trial_choice_count=choice_count,
            reward_sum=reward_sum,
            trial_results=trial_results,
        )
        return buffer, stats

    def state_dict(self) -> Dict[str, object]:
        return {
            "actor_0": self.actors[0].state_dict(),
            "actor_1": self.actors[1].state_dict(),
            "critic": self.critic.state_dict(),
            "actor_opt_0": self.actor_opts[0].state_dict(),
            "actor_opt_1": self.actor_opts[1].state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "log": self.log,
            "total_trials": int(self.total_trials),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.actors[0].load_state_dict(state["actor_0"])
        self.actors[1].load_state_dict(state["actor_1"])
        self.critic.load_state_dict(state["critic"])
        self.actor_opts[0].load_state_dict(state["actor_opt_0"])
        self.actor_opts[1].load_state_dict(state["actor_opt_1"])
        self.critic_opt.load_state_dict(state["critic_opt"])
        if "log" in state:
            self.log = state["log"]
        self.total_trials = int(state.get("total_trials", self.total_trials))

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.asarray(rewards, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        next_value = float(last_value)

        for t in range(len(rewards) - 1, -1, -1):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * non_terminal - values[t]
            gae = delta + self.gamma * self.lambda_gae * non_terminal * gae
            advantages[t] = gae
            next_value = values[t]

        returns = advantages + values
        return advantages, returns

    def update(self, buffer: MAPPOBuffer) -> Tuple[float, float, float]:
        actor_losses: List[float] = []
        critic_losses: List[float] = []
        entropies: List[float] = []

        critic_state_batches: List[torch.Tensor] = []
        critic_target_batches: List[torch.Tensor] = []

        for local_idx, actor_id in enumerate(buffer.actor_ids):
            obs = np.asarray(buffer.obs[local_idx], dtype=np.float32)
            gstate = np.asarray(buffer.global_state[local_idx], dtype=np.float32)
            actions = np.asarray(buffer.action_discrete[local_idx], dtype=np.int64)
            old_log_probs = np.asarray(buffer.log_prob[local_idx], dtype=np.float32)
            rewards = np.asarray(buffer.reward[local_idx], dtype=np.float32)
            dones = np.asarray(buffer.done[local_idx], dtype=np.float32)
            values = np.asarray(buffer.value[local_idx], dtype=np.float32)
            active = np.asarray(buffer.policy_active[local_idx], dtype=np.float32)
            last_value = float(buffer.last_value[local_idx])

            advantages, returns = self.compute_gae(
                rewards=rewards,
                values=values,
                dones=dones,
                last_value=last_value,
            )
            adv_std = float(np.std(advantages) + 1e-8)
            advantages = (advantages - float(np.mean(advantages))) / adv_std

            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
            old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
            adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
            returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
            gstate_t = torch.as_tensor(gstate, dtype=torch.float32, device=self.device)
            active_t = torch.as_tensor(active > 0.5, dtype=torch.bool, device=self.device)

            critic_state_batches.append(gstate_t)
            critic_target_batches.append(returns_t)

            active_idx = torch.nonzero(active_t, as_tuple=False).squeeze(-1)
            n = int(active_idx.numel())
            if n == 0:
                continue

            for _ in range(self.n_epochs):
                idx = active_idx[torch.randperm(n, device=self.device)]
                for start in range(0, n, self.batch_size):
                    mb = idx[start : start + self.batch_size]

                    logits, _ = self.actors[actor_id](obs_t[mb])
                    dist = Categorical(logits=logits)
                    new_log_prob = dist.log_prob(actions_t[mb])
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_log_prob - old_log_probs_t[mb])
                    surr1 = ratio * adv_t[mb]
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t[mb]
                    actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy

                    self.actor_opts[actor_id].zero_grad(set_to_none=True)
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.actors[actor_id].parameters(), self.max_grad_norm)
                    self.actor_opts[actor_id].step()

                    actor_losses.append(float(actor_loss.item()))
                    entropies.append(float(entropy.item()))

        critic_states = torch.cat(critic_state_batches, dim=0)
        critic_targets = torch.cat(critic_target_batches, dim=0)

        n_critic = critic_states.shape[0]
        for _ in range(self.n_epochs):
            idx = torch.randperm(n_critic, device=self.device)
            for start in range(0, n_critic, self.batch_size):
                mb = idx[start : start + self.batch_size]
                values_pred = self.critic(critic_states[mb]).squeeze(-1)
                critic_loss = F.mse_loss(values_pred, critic_targets[mb]) * self.value_loss_coeff

                self.critic_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_opt.step()

                critic_losses.append(float(critic_loss.item()))

        mean_actor = float(np.mean(actor_losses)) if actor_losses else 0.0
        mean_critic = float(np.mean(critic_losses)) if critic_losses else 0.0
        mean_entropy = float(np.mean(entropies)) if entropies else 0.0
        return mean_actor, mean_critic, mean_entropy

    def _run_block(
        self,
        env,
        context: str,
        trials_target: int,
        phase_name: str,
        solo_agent: Optional[int] = None,
    ) -> None:
        self.current_context = context
        if solo_agent is not None:
            self.current_solo_agent = int(solo_agent)
        self.current_obs = env.reset(context=context)

        trials_done = 0
        total_high = 0
        total_choice = 0

        while trials_done < trials_target:
            buffer, stats = self.collect_rollout(env, n_steps=self.rollout_length)
            actor_loss, critic_loss, entropy = self.update(buffer)

            trials_done += stats.trials_completed
            total_high += stats.trial_high_count
            total_choice += stats.trial_choice_count

            p_high = float(total_high / max(1, total_choice))
            mean_reward = float(stats.reward_sum / max(1, stats.steps_collected))

            phase_value = phase_name
            self.log["trial"].append(float(self.total_trials))
            self.log["phase"].append(phase_value)
            self.log["p_high"].append(p_high)
            self.log["actor_loss"].append(actor_loss)
            self.log["critic_loss"].append(critic_loss)
            self.log["entropy"].append(entropy)
            self.log["mean_reward"].append(mean_reward)

    def _estimate_p_high(self, env, actor_id: int, eval_trials: int = 200) -> float:
        self.current_context = "solo"
        self.current_solo_agent = int(actor_id)
        self.current_obs = env.reset(context="solo")

        trials = 0
        high = 0

        while trials < eval_trials:
            obs_dict = self.current_obs
            agent_obs = np.asarray(obs_dict["agent_obs"], dtype=np.float32)
            phase = str(obs_dict.get("phase", "iti"))

            if phase == "choice":
                act, _, _ = self._select_action(
                    actor=self.actors[actor_id],
                    obs_vec=agent_obs[0],
                )
                env_action = int(act)
            else:
                env_action = 8

            next_obs, _, dones, info = env.step(env_action)

            if info.get("event") == "trial_end":
                trials += 1
                tr = info.get("trial_result", {})
                if isinstance(tr, dict) and tr.get("trial_type") == "choice":
                    if tr.get("choice_role") == "high":
                        high += 1

            if bool(dones["__all__"]):
                next_obs = env.reset(context="solo")

            self.current_obs = next_obs

        return float(high / max(1, trials))

    def _save_checkpoint(self, trial: int) -> None:
        ckpt = self.state_dict()
        ckpt["trial"] = int(trial)
        path = self.checkpoint_dir / f"mappo_trial_{int(trial):07d}.pt"
        torch.save(ckpt, path)

    def train(self, env, n_phases: int = 2, trials_per_phase: int = 500) -> Dict[str, List]:
        _ = n_phases
        block_trials = int(trials_per_phase)

        # Phase 1: SOLO pre-training, 5000 trials per agent with reliability target.
        for agent_id in [0, 1]:
            self.current_context = "solo"
            self.current_solo_agent = agent_id
            self.current_obs = env.reset(context="solo")

            phase_trials = 0
            high_count = 0
            choice_count = 0

            while True:
                buffer, stats = self.collect_rollout(env, n_steps=self.rollout_length)
                actor_loss, critic_loss, entropy = self.update(buffer)

                phase_trials += stats.trials_completed
                high_count += stats.trial_high_count
                choice_count += stats.trial_choice_count

                p_high = float(high_count / max(1, choice_count))
                mean_reward = float(stats.reward_sum / max(1, stats.steps_collected))

                self.log["trial"].append(float(self.total_trials))
                self.log["phase"].append(f"phase1_solo_agent{agent_id}")
                self.log["p_high"].append(p_high)
                self.log["actor_loss"].append(actor_loss)
                self.log["critic_loss"].append(critic_loss)
                self.log["entropy"].append(entropy)
                self.log["mean_reward"].append(mean_reward)

                if phase_trials >= 5000 and p_high > 0.70:
                    break

        # Phase 2: 30 alternating blocks of 500 trials each.
        # Solo blocks train both actors for the same block budget in sequence.
        # Social blocks train both actors simultaneously with centralized critic.
        for block in range(30):
            if block % 2 == 0:
                self._run_block(
                    env,
                    context="solo",
                    trials_target=block_trials,
                    phase_name=f"phase2_block{block}_solo_agent0",
                    solo_agent=0,
                )
                self._run_block(
                    env,
                    context="solo",
                    trials_target=block_trials,
                    phase_name=f"phase2_block{block}_solo_agent1",
                    solo_agent=1,
                )
            else:
                self._run_block(
                    env,
                    context="social",
                    trials_target=block_trials,
                    phase_name=f"phase2_block{block}_social",
                )

        self._save_checkpoint(trial=self.total_trials)
        return self.log
