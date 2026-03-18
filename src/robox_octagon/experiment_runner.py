from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from .env import OctagonEnv
from .mappo import MAPPOTrainer


def _patch_xy(patch_id: int, inradius: float = 1.0) -> np.ndarray:
    theta = np.deg2rad(float(patch_id) * 45.0)
    return np.array([np.cos(theta), np.sin(theta)], dtype=float) * float(inradius)


def _extract_high_low(trial_result: Dict[str, Any]) -> Tuple[float, float]:
    patch_indices = trial_result.get("patch_indices", [-1, -1])
    patch_roles = trial_result.get("patch_roles", [None, None])

    high_patch_id = np.nan
    low_patch_id = np.nan
    for idx, role in zip(patch_indices, patch_roles):
        if role == "high":
            high_patch_id = float(idx)
        if role == "low":
            low_patch_id = float(idx)

    if np.isnan(high_patch_id) and len(patch_indices) > 0:
        high_patch_id = float(patch_indices[0])
    if np.isnan(low_patch_id):
        if len(patch_indices) > 1:
            low_patch_id = float(patch_indices[1])
        elif len(patch_indices) > 0:
            low_patch_id = float(patch_indices[0])

    return high_patch_id, low_patch_id


def _patch_sep_deg(high_patch_id: float, low_patch_id: float) -> float:
    if np.isnan(high_patch_id) or np.isnan(low_patch_id):
        return np.nan
    d = abs(int(high_patch_id) - int(low_patch_id))
    sep = min(d, 8 - d)
    return float(sep * 45)


class ExperimentLogger:
    def __init__(self, pair_dir: Path) -> None:
        self.pair_dir = pair_dir
        self.pair_dir.mkdir(parents=True, exist_ok=True)
        self.trials_csv = self.pair_dir / "trials.csv"
        self.blocks_csv = self.pair_dir / "blocks.csv"
        self.summary_h5 = self.pair_dir / "summary.h5"

        self.trial_records: List[Dict[str, Any]] = []
        self.block_records: List[Dict[str, Any]] = []

        if self.trials_csv.exists() or self.blocks_csv.exists():
            self.load()

    def log_trial(self, row: Dict[str, Any]) -> None:
        self.trial_records.append(dict(row))

    def log_block(self, row: Dict[str, Any]) -> None:
        self.block_records.append(dict(row))

    def save(self) -> None:
        trials_df = pd.DataFrame(self.trial_records)
        blocks_df = pd.DataFrame(self.block_records)

        trials_df.to_csv(self.trials_csv, index=False)
        blocks_df.to_csv(self.blocks_csv, index=False)

        summary = {
            "n_trial_rows": int(len(trials_df)),
            "n_blocks": int(len(blocks_df)),
            "mean_raw_reward": float(trials_df["raw_reward"].mean()) if len(trials_df) > 0 else 0.0,
            "mean_shaped_reward": float(trials_df["shaped_reward"].mean()) if len(trials_df) > 0 else 0.0,
        }
        summary_df = pd.DataFrame([summary])

        with pd.HDFStore(self.summary_h5, mode="w") as store:
            store.put("trials", trials_df, format="table")
            store.put("blocks", blocks_df, format="table")
            store.put("summary", summary_df, format="table")

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        trials_df = pd.read_csv(self.trials_csv) if self.trials_csv.exists() else pd.DataFrame()
        blocks_df = pd.read_csv(self.blocks_csv) if self.blocks_csv.exists() else pd.DataFrame()
        self.trial_records = trials_df.to_dict(orient="records")
        self.block_records = blocks_df.to_dict(orient="records")
        return trials_df, blocks_df


@dataclass
class RunState:
    phase: str
    phase1_agent: int
    phase1_trials_done: List[int]
    phase1_choice_history: List[List[int]]
    phase1_log_counter: List[int]
    phase2_block: int
    phase2_stage: str
    phase2_stage_trials_done: int
    next_trial_id: int


def _init_state() -> RunState:
    return RunState(
        phase="phase1",
        phase1_agent=0,
        phase1_trials_done=[0, 0],
        phase1_choice_history=[[], []],
        phase1_log_counter=[0, 0],
        phase2_block=0,
        phase2_stage="solo_A",
        phase2_stage_trials_done=0,
        next_trial_id=0,
    )


def _latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    if not checkpoint_dir.exists():
        return None
    candidates = sorted(checkpoint_dir.glob("experiment_state_*.pt"))
    return candidates[-1] if candidates else None


def _save_run_checkpoint(
    checkpoint_dir: Path,
    trainer: MAPPOTrainer,
    state: RunState,
    seed_A: int,
    seed_B: int,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed_A": int(seed_A),
        "seed_B": int(seed_B),
        "trainer_state": trainer.state_dict(),
        "run_state": {
            "phase": state.phase,
            "phase1_agent": state.phase1_agent,
            "phase1_trials_done": list(state.phase1_trials_done),
            "phase1_choice_history": [list(x) for x in state.phase1_choice_history],
            "phase1_log_counter": list(state.phase1_log_counter),
            "phase2_block": state.phase2_block,
            "phase2_stage": state.phase2_stage,
            "phase2_stage_trials_done": state.phase2_stage_trials_done,
            "next_trial_id": state.next_trial_id,
        },
    }
    trial_id = int(trainer.total_trials)
    out = checkpoint_dir / f"experiment_state_{trial_id:07d}.pt"
    torch.save(payload, out)
    return out


def _load_run_checkpoint(path: Path, trainer: MAPPOTrainer) -> RunState:
    payload = torch.load(path, map_location="cpu")
    trainer.load_state_dict(payload["trainer_state"])
    rs = payload["run_state"]
    return RunState(
        phase=str(rs["phase"]),
        phase1_agent=int(rs["phase1_agent"]),
        phase1_trials_done=[int(x) for x in rs["phase1_trials_done"]],
        phase1_choice_history=[list(map(int, x)) for x in rs["phase1_choice_history"]],
        phase1_log_counter=[int(x) for x in rs["phase1_log_counter"]],
        phase2_block=int(rs["phase2_block"]),
        phase2_stage=str(rs["phase2_stage"]),
        phase2_stage_trials_done=int(rs["phase2_stage_trials_done"]),
        next_trial_id=int(rs["next_trial_id"]),
    )


def _build_trial_rows(
    trial_result: Dict[str, Any],
    phase_label: str,
    block_id: int,
    next_trial_id: int,
    global_agent_map: List[int],
    inradius: float,
    context: str,
) -> Tuple[List[Dict[str, Any]], int, bool]:
    rows: List[Dict[str, Any]] = []
    n_agents = len(global_agent_map)

    high_patch_id, low_patch_id = _extract_high_low(trial_result)
    patch_separation_deg = _patch_sep_deg(high_patch_id, low_patch_id)

    high_xy = _patch_xy(int(high_patch_id), inradius=inradius) if not np.isnan(high_patch_id) else np.array([np.nan, np.nan])
    low_xy = _patch_xy(int(low_patch_id), inradius=inradius) if not np.isnan(low_patch_id) else np.array([np.nan, np.nan])

    onset_positions = trial_result.get("stimulus_onset_positions", [[np.nan, np.nan] for _ in range(n_agents)])
    onset_speeds = trial_result.get("stimulus_onset_speeds", [np.nan for _ in range(n_agents)])
    reward_details = trial_result.get("reward_details", [])
    raw_rewards = trial_result.get("raw_rewards", [0.0 for _ in range(n_agents)])
    shaped_rewards = trial_result.get("shaped_rewards", [0.0 for _ in range(n_agents)])

    winner_id = trial_result.get("winner_agent", np.nan)
    rt_value = trial_result.get("response_time", np.nan)

    is_choice_high = bool(trial_result.get("choice_role") == "high")

    for local_agent in range(n_agents):
        global_agent = int(global_agent_map[local_agent])
        onset = np.asarray(onset_positions[local_agent], dtype=float)

        dist2high = float(np.linalg.norm(onset - high_xy)) if not np.isnan(high_xy).any() else np.nan
        dist2low = float(np.linalg.norm(onset - low_xy)) if not np.isnan(low_xy).any() else np.nan

        if context == "social":
            opp_local = 1 - local_agent
            opp_onset = np.asarray(onset_positions[opp_local], dtype=float)
            opp_dist2high = float(np.linalg.norm(opp_onset - high_xy)) if not np.isnan(high_xy).any() else np.nan
            opp_dist2low = float(np.linalg.norm(opp_onset - low_xy)) if not np.isnan(low_xy).any() else np.nan
            opp_speed = float(onset_speeds[opp_local])
            winner_out = float(winner_id) if winner_id is not None else np.nan
        else:
            opp_dist2high = np.nan
            opp_dist2low = np.nan
            opp_speed = np.nan
            winner_out = np.nan

        detail = reward_details[local_agent] if local_agent < len(reward_details) else {}
        chosen_patch_id = detail.get("chosen_patch_id", np.nan)
        if chosen_patch_id is None or int(chosen_patch_id) < 0:
            choice = np.nan
        elif not np.isnan(high_patch_id) and int(chosen_patch_id) == int(high_patch_id):
            choice = 0.0
        elif not np.isnan(low_patch_id) and int(chosen_patch_id) == int(low_patch_id):
            choice = 1.0
        else:
            choice = np.nan

        row = {
            "trial_id": int(next_trial_id),
            "block_id": int(block_id),
            "phase": phase_label,
            "agent_id": int(global_agent),
            "stimulus_onset_pos_x": float(onset[0]),
            "stimulus_onset_pos_y": float(onset[1]),
            "high_patch_id": float(high_patch_id),
            "low_patch_id": float(low_patch_id),
            "patch_separation_deg": float(patch_separation_deg),
            "choice": choice,
            "RT": float(rt_value) if rt_value is not None else np.nan,
            "travel_distance": float(detail.get("travel_distance", np.nan)),
            "dist2high_at_onset": dist2high,
            "dist2low_at_onset": dist2low,
            "opp_dist2high_at_onset": opp_dist2high,
            "opp_dist2low_at_onset": opp_dist2low,
            "opp_speed_at_onset": opp_speed,
            "winner_id": winner_out,
            "raw_reward": float(raw_rewards[local_agent]) if local_agent < len(raw_rewards) else 0.0,
            "shaped_reward": float(shaped_rewards[local_agent]) if local_agent < len(shaped_rewards) else 0.0,
        }
        rows.append(row)

    return rows, next_trial_id + 1, is_choice_high


def run_experiment(pair_id: int, seed_A: int, seed_B: int, config: Dict[str, Any]) -> Dict[str, Any]:
    exp_cfg = config["experiment"]
    env_cfg = config["env"]
    mappo_cfg = config["mappo"]
    protocol = config["protocol"]

    pair_dir = Path(exp_cfg["log_dir"]) / f"pair_{seed_A}_{seed_B}"
    checkpoints_dir = pair_dir / "checkpoints"
    logger = ExperimentLogger(pair_dir)

    env = OctagonEnv(
        dt=float(env_cfg["dt"]),
        inradius=float(env_cfg["inradius"]),
        max_speed=float(env_cfg["max_speed"]),
        max_trials=int(env_cfg["max_trials"]),
        seed=int(pair_id),
    )

    torch.manual_seed(int(seed_A))
    trainer = MAPPOTrainer(
        obs_dim=int(env.obs_dim),
        global_state_dim=int(2 * env.obs_dim + 3),
        action_dim=int(mappo_cfg["action_dim"]),
        gamma=float(mappo_cfg["gamma"]),
        lambda_gae=float(mappo_cfg["lambda_gae"]),
        clip_eps=float(mappo_cfg["clip_eps"]),
        lr_actor=float(mappo_cfg["lr_actor"]),
        lr_critic=float(mappo_cfg["lr_critic"]),
        n_epochs=int(mappo_cfg["n_epochs"]),
        batch_size=int(mappo_cfg["batch_size"]),
        rollout_length=int(mappo_cfg["rollout_length"]),
        entropy_coeff=float(mappo_cfg["entropy_coeff"]),
        value_loss_coeff=float(mappo_cfg["value_loss_coeff"]),
        max_grad_norm=float(mappo_cfg["max_grad_norm"]),
        checkpoint_dir=str(checkpoints_dir),
        actor_seeds=(int(seed_A), int(seed_B)),
    )

    state = _init_state()
    latest = _latest_checkpoint(checkpoints_dir)
    if latest is not None:
        state = _load_run_checkpoint(latest, trainer)

    ckpt_every = int(exp_cfg["checkpoint_every_trials"])

    phase1_trials = int(protocol["phase1_trials"])
    phase1_log_every = int(protocol["phase1_log_every"])
    phase1_window = int(protocol["phase1_early_stop_window"])
    phase1_thresh = float(protocol["phase1_early_stop_p_high"])

    phase2_blocks = int(protocol["phase2_blocks"])
    block_trials = int(protocol["block_trials"])

    # Phase 1: solo pre-training for each actor independently.
    while state.phase == "phase1":
        agent = state.phase1_agent
        trainer.current_context = "solo"
        trainer.current_solo_agent = int(agent)
        trainer.current_obs = env.reset(context="solo")

        buffer, stats = trainer.collect_rollout(env, n_steps=trainer.rollout_length)
        actor_loss, critic_loss, entropy = trainer.update(buffer)

        for tr in stats.trial_results:
            rows, state.next_trial_id, is_high = _build_trial_rows(
                trial_result=tr,
                phase_label="solo",
                block_id=-1,
                next_trial_id=state.next_trial_id,
                global_agent_map=[agent],
                inradius=float(env.inradius),
                context="solo",
            )
            for row in rows:
                logger.log_trial(row)

            state.phase1_trials_done[agent] += 1
            if tr.get("trial_type") == "choice":
                state.phase1_choice_history[agent].append(1 if is_high else 0)
                if len(state.phase1_choice_history[agent]) > phase1_window:
                    state.phase1_choice_history[agent] = state.phase1_choice_history[agent][-phase1_window:]

                if state.phase1_trials_done[agent] % phase1_log_every == 0:
                    p_high_now = float(np.mean(state.phase1_choice_history[agent])) if state.phase1_choice_history[agent] else 0.0
                    logger.log_block(
                        {
                            "block_id": -1,
                            "phase": f"phase1_agent_{agent}",
                            "agent_id": int(agent),
                            "trials_done": int(state.phase1_trials_done[agent]),
                            "p_high": p_high_now,
                            "actor_loss": float(actor_loss),
                            "critic_loss": float(critic_loss),
                            "entropy": float(entropy),
                        }
                    )

            early_stop_ready = len(state.phase1_choice_history[agent]) >= phase1_window
            early_stop_pass = early_stop_ready and float(np.mean(state.phase1_choice_history[agent][-phase1_window:])) > phase1_thresh
            done_by_trials = state.phase1_trials_done[agent] >= phase1_trials

            if done_by_trials or early_stop_pass:
                state.phase1_agent += 1
                break

        if state.phase1_agent >= 2:
            state.phase = "phase2"
            state.phase2_block = 0
            state.phase2_stage = "solo_A"
            state.phase2_stage_trials_done = 0

        if trainer.total_trials > 0 and trainer.total_trials % ckpt_every == 0:
            logger.save()
            _save_run_checkpoint(checkpoints_dir, trainer, state, seed_A, seed_B)

    # Phase 2: 30 interleaved blocks (solo then social).
    while state.phase == "phase2" and state.phase2_block < phase2_blocks:
        block_id = state.phase2_block

        if state.phase2_stage == "solo_A":
            trainer.current_context = "solo"
            trainer.current_solo_agent = 0
            trainer.current_obs = env.reset(context="solo")

            buffer, stats = trainer.collect_rollout(env, n_steps=trainer.rollout_length)
            actor_loss, critic_loss, entropy = trainer.update(buffer)

            for tr in stats.trial_results:
                rows, state.next_trial_id, _ = _build_trial_rows(
                    trial_result=tr,
                    phase_label="solo",
                    block_id=block_id,
                    next_trial_id=state.next_trial_id,
                    global_agent_map=[0],
                    inradius=float(env.inradius),
                    context="solo",
                )
                for row in rows:
                    logger.log_trial(row)
                state.phase2_stage_trials_done += 1
                if state.phase2_stage_trials_done >= block_trials:
                    state.phase2_stage = "solo_B"
                    state.phase2_stage_trials_done = 0
                    break

        elif state.phase2_stage == "solo_B":
            trainer.current_context = "solo"
            trainer.current_solo_agent = 1
            trainer.current_obs = env.reset(context="solo")

            buffer, stats = trainer.collect_rollout(env, n_steps=trainer.rollout_length)
            actor_loss, critic_loss, entropy = trainer.update(buffer)

            for tr in stats.trial_results:
                rows, state.next_trial_id, _ = _build_trial_rows(
                    trial_result=tr,
                    phase_label="solo",
                    block_id=block_id,
                    next_trial_id=state.next_trial_id,
                    global_agent_map=[1],
                    inradius=float(env.inradius),
                    context="solo",
                )
                for row in rows:
                    logger.log_trial(row)
                state.phase2_stage_trials_done += 1
                if state.phase2_stage_trials_done >= block_trials:
                    state.phase2_stage = "social"
                    state.phase2_stage_trials_done = 0
                    break

        elif state.phase2_stage == "social":
            trainer.current_context = "social"
            trainer.current_obs = env.reset(context="social")

            buffer, stats = trainer.collect_rollout(env, n_steps=trainer.rollout_length)
            actor_loss, critic_loss, entropy = trainer.update(buffer)

            for tr in stats.trial_results:
                rows, state.next_trial_id, _ = _build_trial_rows(
                    trial_result=tr,
                    phase_label="social",
                    block_id=block_id,
                    next_trial_id=state.next_trial_id,
                    global_agent_map=[0, 1],
                    inradius=float(env.inradius),
                    context="social",
                )
                for row in rows:
                    logger.log_trial(row)
                state.phase2_stage_trials_done += 1
                if state.phase2_stage_trials_done >= block_trials:
                    # Summarize this full interleaved block from logger cache.
                    block_rows = [r for r in logger.trial_records if int(r["block_id"]) == block_id]
                    solo_rows = [r for r in block_rows if r["phase"] == "solo"]
                    social_rows = [r for r in block_rows if r["phase"] == "social"]

                    p_high_solo = float(np.mean([1.0 if r["choice"] == 0.0 else 0.0 for r in solo_rows if not pd.isna(r["choice"])])) if solo_rows else np.nan
                    p_high_social = float(np.mean([1.0 if r["choice"] == 0.0 else 0.0 for r in social_rows if not pd.isna(r["choice"])])) if social_rows else np.nan
                    mean_rt = float(np.nanmean([r["RT"] for r in social_rows])) if social_rows else np.nan
                    social_agent0 = [r for r in social_rows if int(r["agent_id"]) == 0]
                    win_rate_A = float(np.nanmean([1.0 if r["winner_id"] == 0.0 else 0.0 for r in social_agent0])) if social_agent0 else np.nan

                    logger.log_block(
                        {
                            "block_id": int(block_id),
                            "phase": "phase2",
                            "p_high_solo": p_high_solo,
                            "p_high_social": p_high_social,
                            "mean_RT": mean_rt,
                            "win_rate_A": win_rate_A,
                            "actor_loss": float(actor_loss),
                            "critic_loss": float(critic_loss),
                            "entropy": float(entropy),
                        }
                    )

                    state.phase2_block += 1
                    state.phase2_stage = "solo_A"
                    state.phase2_stage_trials_done = 0
                    break

        if trainer.total_trials > 0 and trainer.total_trials % ckpt_every == 0:
            logger.save()
            _save_run_checkpoint(checkpoints_dir, trainer, state, seed_A, seed_B)

    logger.save()
    _save_run_checkpoint(checkpoints_dir, trainer, state, seed_A, seed_B)

    return {
        "pair_id": int(pair_id),
        "seed_A": int(seed_A),
        "seed_B": int(seed_B),
        "pair_dir": str(pair_dir),
        "n_trial_rows": int(len(logger.trial_records)),
        "n_blocks": int(len(logger.block_records)),
    }


def _run_experiment_worker(args: Tuple[int, int, int, Dict[str, Any]]) -> Dict[str, Any]:
    pair_id, seed_A, seed_B, config = args
    return run_experiment(pair_id=pair_id, seed_A=seed_A, seed_B=seed_B, config=config)


def _pair_seeds(config: Dict[str, Any]) -> List[Tuple[int, int]]:
    n_pairs = int(config["experiment"]["n_pairs"])
    seeds = list(config["experiment"]["seeds"])

    if len(seeds) < 2 * n_pairs:
        a = seeds[:n_pairs]
        b = [s + 1000 for s in a]
        return list(zip(a, b))

    return list(zip(seeds[:n_pairs], seeds[n_pairs : 2 * n_pairs]))


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OctagonEnv MAPPO experimental protocol")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--parallel", action="store_true", help="Run pairs in parallel")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    pairs = _pair_seeds(config)

    run_parallel = bool(args.parallel or config["experiment"].get("parallel", False))
    num_workers = int(args.num_workers or config["experiment"].get("num_workers", 2))

    tasks = [(i, int(sa), int(sb), config) for i, (sa, sb) in enumerate(pairs)]

    results: List[Dict[str, Any]] = []
    if run_parallel:
        with mp.Pool(processes=num_workers) as pool:
            for out in tqdm(pool.imap_unordered(_run_experiment_worker, tasks), total=len(tasks), desc="Pairs"):
                results.append(out)
    else:
        for task in tqdm(tasks, total=len(tasks), desc="Pairs"):
            results.append(_run_experiment_worker(task))

    out_df = pd.DataFrame(results)
    logs_dir = Path(config["experiment"]["log_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(logs_dir / "experiment_index.csv", index=False)


if __name__ == "__main__":
    main()
