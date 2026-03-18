from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class FitResult:
    params: Dict[str, float]
    log_likelihood: float
    n_params: int
    n_trials: int


class AttractorModel:
    """Bistable attractor model for high-vs-low decisions (paper Figure 4 style)."""

    def __init__(
        self,
        w_e: float = 1.5,
        w_i: float = 4.0,
        tau: float = 0.1,
        dt: float = 0.001,
        threshold: float = 1.0,
        tmax: float = 0.3,
        v_low: float = 0.5,
        rng_seed: Optional[int] = None,
        shotgun_samples: int = 10000,
        n_starts: int = 10,
    ) -> None:
        self.w_e = float(w_e)
        self.w_i = float(w_i)
        self.tau = float(tau)
        self.dt = float(dt)
        self.threshold = float(threshold)
        self.tmax = float(tmax)
        self.v_low = float(v_low)

        self.steps = int(np.ceil(self.tmax / self.dt))
        self.time = np.arange(self.steps, dtype=float) * self.dt

        self.shotgun_samples = int(shotgun_samples)
        self.n_starts = int(n_starts)

        self.rng = np.random.default_rng(rng_seed)

        self.solo_param_names = ["E", "sigma", "V_H", "m", "r", "b"]
        self.social_param_names = ["E", "sigma", "V_H", "m", "r", "b", "n", "s", "c"]

        # Parameter ranges for shotgun initialization.
        self.solo_ranges = {
            "E": (0.0, 3.0),
            "sigma": (0.05, 6.0),
            "V_H": (0.5, 4.0),
            "m": (-3.0, 0.0),
            "r": (0.5, 2.5),
            "b": (0.0, 3.0),
        }
        self.social_ranges = {
            **self.solo_ranges,
            "n": (-2.0, 2.0),
            "s": (0.5, 2.5),
            "c": (0.0, 3.0),
        }

        # Paper Table 2 best fits (defaults for quick simulation if no explicit fit).
        self.paper_solo = {
            "E": 1.239,
            "sigma": 3.775,
            "V_H": 2.199,
            "m": -1.155,
            "r": 1.540,
            "b": 1.471,
        }
        self.paper_social = {
            "E": 1.916,
            "sigma": 3.899,
            "V_H": 1.815,
            "m": -1.623,
            "r": 1.238,
            "b": 0.777,
            "n": -0.741,
            "s": 1.000,
            "c": 1.534,
        }

        self.best_params_solo: Optional[Dict[str, float]] = None
        self.best_params_social: Optional[Dict[str, float]] = None
        self.fit_history: Dict[str, List[Tuple[float, Dict[str, float]]]] = {
            "solo": [],
            "social": [],
        }

    @staticmethod
    def _f(x: float) -> float:
        return 0.5 * np.tanh(x) + 0.5

    @staticmethod
    def _safe_pow(x: float, p: float) -> float:
        return float(np.power(max(1e-8, float(x)), float(p)))

    def _psi(self, z: float, params: Dict[str, float], context: str) -> float:
        if context == "solo":
            return 0.0
        if any(k not in params for k in ["n", "s", "c"]):
            return 0.0
        return float(params["n"] * self._safe_pow(z, params["s"]) + params["c"])

    def _single_sim(
        self,
        dist_self_high: float,
        dist_self_low: float,
        dist_opp_high: float,
        dist_opp_low: float,
        context: str,
        params: Dict[str, float],
    ) -> Tuple[int, float, np.ndarray, np.ndarray]:
        m = float(params["m"])
        r = float(params["r"])
        b = float(params["b"])

        x1 = m * self._safe_pow(dist_self_high, r) + b
        x2 = m * self._safe_pow(dist_self_low, r) + b

        e = float(params["E"])
        sigma = float(params["sigma"])
        v1 = float(params["V_H"])
        v2 = float(self.v_low)

        tr1 = np.zeros(self.steps, dtype=float)
        tr2 = np.zeros(self.steps, dtype=float)

        crossed = False
        choice_high = 0
        rt = self.tmax

        for t in range(self.steps):
            tr1[t] = x1
            tr2[t] = x2

            eta1 = float(self.rng.normal(e, sigma))
            eta2 = float(self.rng.normal(e, sigma))

            psi1 = self._psi(dist_opp_high, params, context=context)
            psi2 = self._psi(dist_opp_low, params, context=context)

            dx1 = (
                -x1
                + self.w_e * self._f(x1)
                - self.w_i * self._f(x2)
                + v1
                + psi1
                + eta1
            ) / self.tau
            dx2 = (
                -x2
                + self.w_e * self._f(x2)
                - self.w_i * self._f(x1)
                + v2
                + psi2
                + eta2
            ) / self.tau

            x1 = x1 + self.dt * dx1
            x2 = x2 + self.dt * dx2

            if not crossed:
                hit1 = x1 >= self.threshold
                hit2 = x2 >= self.threshold
                if hit1 or hit2:
                    crossed = True
                    rt = (t + 1) * self.dt
                    if hit1 and hit2:
                        choice_high = 1 if x1 >= x2 else 0
                    else:
                        choice_high = 1 if hit1 else 0
                    break

        if not crossed:
            # If no threshold crossing in 0.3 s, choose higher node.
            choice_high = 1 if x1 >= x2 else 0
            rt = self.tmax

        return int(choice_high), float(rt), tr1, tr2

    def simulate_trial(
        self,
        dist_self_high: float,
        dist_self_low: float,
        dist_opp_high: float,
        dist_opp_low: float,
        context: str = "solo",
        n_sims: int = 200,
        params: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float, Dict[str, np.ndarray]]:
        """Simulate repeated realizations for one trial and return P_high and mean RT."""
        context = str(context).lower()
        if context not in {"solo", "social"}:
            raise ValueError("context must be 'solo' or 'social'")

        if params is None:
            params = self.paper_solo if context == "solo" else self.paper_social

        n_sims = int(n_sims)
        choices = np.zeros(n_sims, dtype=float)
        rts = np.zeros(n_sims, dtype=float)
        tr1_acc = np.zeros(self.steps, dtype=float)
        tr2_acc = np.zeros(self.steps, dtype=float)

        d_oh = float(0.0 if not np.isfinite(dist_opp_high) else dist_opp_high)
        d_ol = float(0.0 if not np.isfinite(dist_opp_low) else dist_opp_low)

        for i in range(n_sims):
            ch, rt, tr1, tr2 = self._single_sim(
                dist_self_high=float(dist_self_high),
                dist_self_low=float(dist_self_low),
                dist_opp_high=d_oh,
                dist_opp_low=d_ol,
                context=context,
                params=params,
            )
            choices[i] = ch
            rts[i] = rt
            tr1_acc += tr1
            tr2_acc += tr2

        p_high = float(np.mean(choices))
        mean_rt = float(np.mean(rts))

        trajectories = {
            "time": self.time.copy(),
            "x1_mean": tr1_acc / float(n_sims),
            "x2_mean": tr2_acc / float(n_sims),
            "choice_samples": choices.copy(),
            "rt_samples": rts.copy(),
        }
        return p_high, mean_rt, trajectories

    def simulate_trajectories(
        self,
        dist_self_high: float,
        dist_self_low: float,
        dist_opp_high: float,
        dist_opp_low: float,
        context: str = "solo",
        n_sims: int = 20,
        params: Optional[Dict[str, float]] = None,
    ) -> Dict[str, np.ndarray]:
        """Return full per-simulation trajectories for visualization."""
        context = str(context).lower()
        if context not in {"solo", "social"}:
            raise ValueError("context must be 'solo' or 'social'")

        if params is None:
            params = self.paper_solo if context == "solo" else self.paper_social

        n_sims = int(n_sims)
        x1 = np.zeros((n_sims, self.steps), dtype=float)
        x2 = np.zeros((n_sims, self.steps), dtype=float)
        choices = np.zeros(n_sims, dtype=float)
        rts = np.zeros(n_sims, dtype=float)

        d_oh = float(0.0 if not np.isfinite(dist_opp_high) else dist_opp_high)
        d_ol = float(0.0 if not np.isfinite(dist_opp_low) else dist_opp_low)

        for i in range(n_sims):
            ch, rt, tr1, tr2 = self._single_sim(
                dist_self_high=float(dist_self_high),
                dist_self_low=float(dist_self_low),
                dist_opp_high=d_oh,
                dist_opp_low=d_ol,
                context=context,
                params=params,
            )
            x1[i] = tr1
            x2[i] = tr2
            choices[i] = ch
            rts[i] = rt

        return {
            "time": self.time.copy(),
            "x1": x1,
            "x2": x2,
            "choices": choices,
            "rts": rts,
            "p_high": float(np.mean(choices)),
            "mean_rt": float(np.mean(rts)),
        }

    def _dataset_dist_cols(self, trial_df: pd.DataFrame) -> Tuple[str, str, str, str]:
        # Accept canonical columns from logger.
        candidates = {
            "self_high": ["dist2high_at_onset", "dist_self_high"],
            "self_low": ["dist2low_at_onset", "dist_self_low"],
            "opp_high": ["opp_dist2high_at_onset", "dist_opp_high"],
            "opp_low": ["opp_dist2low_at_onset", "dist_opp_low"],
        }

        out = []
        for key in ["self_high", "self_low", "opp_high", "opp_low"]:
            found = None
            for c in candidates[key]:
                if c in trial_df.columns:
                    found = c
                    break
            if found is None:
                # Opp columns can be missing in solo data.
                if key.startswith("opp"):
                    out.append("")
                    continue
                raise ValueError(f"Missing required distance column for {key}: {candidates[key]}")
            out.append(found)
        return out[0], out[1], out[2], out[3]

    def simulate_dataset(
        self,
        trial_df: pd.DataFrame,
        n_sims: int = 200,
        params: Optional[Dict[str, float]] = None,
        context: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return per-trial predicted P(high) and mean RT."""
        dsh, dsl, doh, dol = self._dataset_dist_cols(trial_df)

        p_list: List[float] = []
        rt_list: List[float] = []

        for _, r in trial_df.iterrows():
            row_context = context
            if row_context is None:
                row_context = str(r.get("phase", "solo")).lower()
                row_context = "social" if row_context == "social" else "solo"

            opp_h = float(r[doh]) if doh and np.isfinite(float(r[doh])) else 0.0
            opp_l = float(r[dol]) if dol and np.isfinite(float(r[dol])) else 0.0

            p, rt, _ = self.simulate_trial(
                dist_self_high=float(r[dsh]),
                dist_self_low=float(r[dsl]),
                dist_opp_high=opp_h,
                dist_opp_low=opp_l,
                context=row_context,
                n_sims=n_sims,
                params=params,
            )
            p_list.append(p)
            rt_list.append(rt)

        return np.asarray(p_list, dtype=float), np.asarray(rt_list, dtype=float)

    def log_likelihood(
        self,
        trial_df: pd.DataFrame,
        params: Dict[str, float],
        context: Optional[str] = None,
        n_sims: int = 200,
    ) -> float:
        """Bernoulli log-likelihood of observed choices under model-predicted P(high)."""
        if "choice" not in trial_df.columns:
            raise ValueError("trial_df must contain observed 'choice' column (0=high,1=low)")

        p_high, _ = self.simulate_dataset(trial_df, n_sims=n_sims, params=params, context=context)
        obs_high = (trial_df["choice"].astype(float).to_numpy() == 0.0).astype(float)

        eps = 1e-6
        p = np.clip(p_high, eps, 1.0 - eps)
        ll = np.sum(obs_high * np.log(p) + (1.0 - obs_high) * np.log(1.0 - p))
        return float(ll)

    def _sample_param_set(self, context: str) -> Dict[str, float]:
        rng = self.solo_ranges if context == "solo" else self.social_ranges
        out: Dict[str, float] = {}
        for k, (lo, hi) in rng.items():
            out[k] = float(self.rng.uniform(lo, hi))
        return out

    def _vector_to_params(self, x: np.ndarray, context: str) -> Dict[str, float]:
        names = self.solo_param_names if context == "solo" else self.social_param_names
        return {k: float(v) for k, v in zip(names, x.tolist())}

    def _params_to_vector(self, params: Dict[str, float], context: str) -> np.ndarray:
        names = self.solo_param_names if context == "solo" else self.social_param_names
        return np.asarray([float(params[k]) for k in names], dtype=float)

    def _clip_to_ranges(self, x: np.ndarray, context: str) -> np.ndarray:
        ranges = self.solo_ranges if context == "solo" else self.social_ranges
        names = self.solo_param_names if context == "solo" else self.social_param_names
        y = x.copy().astype(float)
        for i, k in enumerate(names):
            lo, hi = ranges[k]
            y[i] = float(np.clip(y[i], lo, hi))
        return y

    def _fit_context(self, trial_df: pd.DataFrame, context: str) -> Dict[str, float]:
        context = str(context).lower()
        if context not in {"solo", "social"}:
            raise ValueError("context must be 'solo' or 'social'")

        df_ctx = trial_df.copy()
        if "phase" in df_ctx.columns:
            if context == "solo":
                df_ctx = df_ctx[df_ctx["phase"] != "social"].copy()
            else:
                df_ctx = df_ctx[df_ctx["phase"] == "social"].copy()
        if len(df_ctx) == 0:
            raise ValueError(f"No trials available for context={context}")

        # 1) Shotgun random search.
        scored: List[Tuple[float, Dict[str, float]]] = []
        for _ in range(self.shotgun_samples):
            p = self._sample_param_set(context=context)
            ll = self.log_likelihood(df_ctx, params=p, context=context, n_sims=200)
            scored.append((ll, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        self.fit_history[context] = [(float(ll), dict(p)) for ll, p in scored]
        top = scored[: self.n_starts]

        # 2) Local Nelder-Mead optimization from top starts.
        best_ll = -np.inf
        best_params: Dict[str, float] = top[0][1].copy()

        def objective(x: np.ndarray) -> float:
            x_clip = self._clip_to_ranges(x, context=context)
            p = self._vector_to_params(x_clip, context=context)
            ll = self.log_likelihood(df_ctx, params=p, context=context, n_sims=200)
            return float(-ll)

        for _, p0 in top:
            x0 = self._params_to_vector(p0, context=context)
            res = minimize(objective, x0=x0, method="Nelder-Mead")
            x_opt = self._clip_to_ranges(np.asarray(res.x, dtype=float), context=context)
            p_opt = self._vector_to_params(x_opt, context=context)
            ll_opt = self.log_likelihood(df_ctx, params=p_opt, context=context, n_sims=200)
            if ll_opt > best_ll:
                best_ll = ll_opt
                best_params = p_opt

        if context == "solo":
            self.best_params_solo = dict(best_params)
        else:
            self.best_params_social = dict(best_params)

        return best_params

    def fit_solo(self, trial_df: pd.DataFrame) -> Dict[str, float]:
        """Fit solo model with 6 free parameters."""
        return self._fit_context(trial_df=trial_df, context="solo")

    def fit_social(self, trial_df: pd.DataFrame) -> Dict[str, float]:
        """Fit social model with 9 free parameters."""
        return self._fit_context(trial_df=trial_df, context="social")
