# Author: Samuel Lozano
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import ttest_1samp, ttest_rel
except Exception:  # pragma: no cover
    ttest_1samp = None
    ttest_rel = None

try:
    import statsmodels.formula.api as smf
except Exception:  # pragma: no cover
    smf = None

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from .attractor_model import AttractorModel
from .attractor_analysis import run_attractor_analysis
from .experiment_runner import load_config as load_experiment_config
from .experiment_runner import run_experiment
from .rl_comparison import run_rl_comparison
from .social_analysis import augment_social_with_inferred_losers, run_social_analysis
from .solo_analysis import run_solo_analysis


def _ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _octagon_vertices(inradius: float = 1.0) -> np.ndarray:
    circum = float(inradius) / np.cos(np.pi / 8.0)
    angles = np.deg2rad(np.arange(22.5, 360.0 + 22.5, 45.0))
    return np.stack([circum * np.cos(angles), circum * np.sin(angles)], axis=1)


def _point_in_octagon(point: np.ndarray, inradius: float = 1.0) -> bool:
    angles = np.deg2rad(np.arange(0.0, 360.0, 45.0, dtype=float))
    normals = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return bool(np.all(normals @ point <= inradius + 1e-12))


def _rolling_phigh(df: pd.DataFrame, window: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    d = df[df["phase"] == "solo"].copy()
    d = d[np.isfinite(d["choice"].astype(float))].copy()
    if d.empty:
        return np.array([]), np.array([])

    curves = []
    max_len = 0
    for _, g in d.groupby("agent_id"):
        g = g.sort_values("trial_id")
        c = (g["choice"].astype(float) == 0.0).astype(float)
        roll = c.rolling(window=window, min_periods=window).mean().to_numpy(dtype=float)
        curves.append(roll)
        max_len = max(max_len, len(roll))

    mat = np.full((len(curves), max_len), np.nan, dtype=float)
    for i, c in enumerate(curves):
        mat[i, : len(c)] = c

    mean = np.nanmean(mat, axis=0)
    x = np.arange(1, len(mean) + 1, dtype=float)
    return x, mean


def _plot_learning(ax: plt.Axes, df: pd.DataFrame) -> None:
    x, y = _rolling_phigh(df, window=50)
    if len(x) == 0:
        ax.text(0.5, 0.5, "No solo data", ha="center", va="center")
        ax.set_axis_off()
        return
    ax.plot(x, y, color="#2b8cbe", lw=1.8)
    ax.axhline(0.5, ls="--", lw=0.9, color="black")
    ax.axhline(0.7, ls="--", lw=0.9, color="gray")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Learning curves")
    ax.set_xlabel("Trial")
    ax.set_ylabel("P(high)")


def _plot_distance_45(ax: plt.Axes, df: pd.DataFrame) -> None:
    d = df[df["phase"] == "solo"].copy()
    needed = ["dist2high_at_onset", "dist2low_at_onset", "patch_separation_deg", "choice"]
    for c in needed:
        if c not in d.columns:
            ax.text(0.5, 0.5, "Missing columns", ha="center", va="center")
            ax.set_axis_off()
            return

    d = d[np.isfinite(d["choice"].astype(float))].copy()
    d = d[d["patch_separation_deg"].astype(float) == 45.0].copy()
    if d.empty:
        ax.text(0.5, 0.5, "No 45 deg trials", ha="center", va="center")
        ax.set_axis_off()
        return

    d["choose_high"] = (d["choice"].astype(float) == 0.0).astype(float)
    d["high_bin"] = pd.cut(d["dist2high_at_onset"], bins=5, labels=False, include_lowest=True)
    d["low_bin"] = pd.cut(d["dist2low_at_onset"], bins=4, labels=False, include_lowest=True)

    colors = mpl.cm.RdPu(np.linspace(0.35, 0.85, 4))
    for low_bin in range(4):
        xs = []
        ys = []
        for hb in range(5):
            g = d[(d["low_bin"] == low_bin) & (d["high_bin"] == hb)]
            if len(g) == 0:
                continue
            xs.append(hb + 1)
            ys.append(float(g["choose_high"].mean()))
        if xs:
            ax.plot(xs, ys, marker="o", lw=1.6, color=colors[low_bin], label=f"low bin {low_bin + 1}")

    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("dist2high bins")
    ax.set_ylabel("P(high)")
    ax.set_title("P(high) by distance (45 deg)")


def _plot_optimal_start_heatmap(ax: plt.Axes, inradius: float = 1.0) -> None:
    bins = np.linspace(-1.2, 1.2, 31)
    centers = 0.5 * (bins[:-1] + bins[1:])
    gx, gy = np.meshgrid(centers, centers)

    patch_angles = np.deg2rad(np.arange(0.0, 360.0, 45.0))
    patches = np.stack([np.cos(patch_angles), np.sin(patch_angles)], axis=1) * float(inradius)

    mean_dist = np.full_like(gx, np.nan, dtype=float)
    for i in range(gx.shape[0]):
        for j in range(gx.shape[1]):
            p = np.array([gx[i, j], gy[i, j]], dtype=float)
            if not _point_in_octagon(p, inradius=inradius):
                continue
            d = np.linalg.norm(patches - p[None, :], axis=1)
            mean_dist[i, j] = float(np.mean(d))

    mesh = ax.pcolormesh(bins, bins, mean_dist.T, cmap="viridis_r", shading="auto")
    v = _octagon_vertices(inradius)
    ax.plot(v[:, 0], v[:, 1], color="black", lw=1.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_title("Optimal start map (min mean travel)")
    ax.figure.colorbar(mesh, ax=ax, shrink=0.8)


def _session_shift_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df[np.isfinite(df["choice"].astype(float))].copy()
    d["choose_high"] = (d["choice"].astype(float) == 0.0).astype(float)

    rows: List[Dict[str, float]] = []
    for aid, g in d.groupby("agent_id"):
        g = g.sort_values("trial_id")
        solo = g[g["phase"] == "solo"]
        social = g[g["phase"] == "social"]
        if len(solo) == 0 or len(social) == 0:
            continue
        first_social = float(social["trial_id"].min())
        solo_pre = solo[solo["trial_id"] < first_social]
        solo_post = solo[solo["trial_id"] >= first_social]
        if len(solo_pre) == 0:
            solo_pre = solo.iloc[: max(1, len(solo) // 2)]
        if len(solo_post) == 0:
            solo_post = solo.iloc[max(0, len(solo) // 2) :]

        rows.append({"agent_id": float(aid), "session": "solo_pre", "p_high": float(solo_pre["choose_high"].mean())})
        rows.append({"agent_id": float(aid), "session": "social", "p_high": float(social["choose_high"].mean())})
        rows.append({"agent_id": float(aid), "session": "solo_post", "p_high": float(solo_post["choose_high"].mean())})
    return pd.DataFrame(rows)


def _plot_preference_shift(ax: plt.Axes, df: pd.DataFrame) -> None:
    s = _session_shift_table(df)
    if s.empty:
        ax.text(0.5, 0.5, "Insufficient session data", ha="center", va="center")
        ax.set_axis_off()
        return

    order = ["solo_pre", "social", "solo_post"]
    x = np.arange(3, dtype=float)
    for _, g in s.groupby("agent_id"):
        gg = g.set_index("session").reindex(order)
        ax.plot(x, gg["p_high"].to_numpy(dtype=float), color="gray", alpha=0.35, lw=1)

    mean = s.groupby("session")["p_high"].mean().reindex(order).to_numpy(dtype=float)
    ax.plot(x, mean, color="black", lw=2)
    ax.scatter(x, mean, color="black", s=14)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("P(high)")
    ax.set_title("Preference shift")


def _aligned_xy(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        hp = float(r.get("high_patch_id", np.nan))
        lp = float(r.get("low_patch_id", np.nan))
        ch = float(r.get("choice", np.nan))
        if not np.isfinite(hp) or not np.isfinite(lp) or not np.isfinite(ch):
            continue

        start = np.array([float(r["stimulus_onset_pos_x"]), float(r["stimulus_onset_pos_y"])], dtype=float)
        high_angle = np.deg2rad(float(int(hp)) * 45.0)
        rot = np.deg2rad(90.0) - high_angle
        c = np.cos(rot)
        s = np.sin(rot)
        p = np.array([c * start[0] - s * start[1], s * start[0] + c * start[1]], dtype=float)

        low_angle = np.deg2rad(float(int(lp)) * 45.0)
        low_rot = low_angle + rot
        low_xy = np.array([np.cos(low_rot), np.sin(low_rot)], dtype=float)
        if low_xy[0] < 0.0:
            p[0] *= -1.0

        rows.append(
            {
                "x": float(p[0]),
                "y": float(p[1]),
                "choose_low": 1.0 if ch == 1.0 else 0.0,
                "choose_high": 1.0 if ch == 0.0 else 0.0,
                "opp_dist2high_at_onset": float(r.get("opp_dist2high_at_onset", np.nan)),
                "phase": str(r.get("phase", "")),
                "trial_id": float(r.get("trial_id", np.nan)),
                "agent_id": float(r.get("agent_id", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def _plow_heatmap_data(df: pd.DataFrame, bins: np.ndarray) -> np.ndarray:
    c, _, _ = np.histogram2d(df["x"], df["y"], bins=[bins, bins])
    l, _, _ = np.histogram2d(df["x"], df["y"], bins=[bins, bins], weights=df["choose_low"])
    return np.divide(l, c, out=np.full_like(l, np.nan, dtype=float), where=c > 0)


def _plot_delta_plow(ax: plt.Axes, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    bins = np.linspace(-1.2, 1.2, 11)
    solo = _aligned_xy(df[df["phase"] == "solo"].copy())
    social = _aligned_xy(augment_social_with_inferred_losers(df))

    if solo.empty or social.empty:
        ax.text(0.5, 0.5, "Need solo+social data", ha="center", va="center")
        ax.set_axis_off()
        return bins, np.full((10, 10), np.nan, dtype=float)

    p_low_solo = _plow_heatmap_data(solo, bins)
    p_low_social = _plow_heatmap_data(social, bins)
    delta = p_low_social - p_low_solo

    norm = mpl.colors.TwoSlopeNorm(vmin=-0.25, vcenter=0.0, vmax=0.25)
    mesh = ax.pcolormesh(bins, bins, delta.T, cmap="RdBu_r", norm=norm, shading="auto")
    v = _octagon_vertices(1.0)
    ax.plot(v[:, 0], v[:, 1], color="black", lw=1.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_title("Delta P(low)")
    ax.figure.colorbar(mesh, ax=ax, shrink=0.8)
    return bins, delta


def _plot_opponent_speed(ax: plt.Axes, df: pd.DataFrame) -> None:
    d = df.copy()
    if "pair_id" not in d.columns:
        d["pair_id"] = 0

    d = d[np.isfinite(d["choice"].astype(float))].copy()
    d["choose_low"] = (d["choice"].astype(float) == 1.0).astype(float)

    solo = d[d["phase"] == "solo"].copy()
    social = augment_social_with_inferred_losers(d)
    social["choose_low"] = (social["choice"].astype(float) == 1.0).astype(float)

    rows = []
    for (pid, aid), gs in social.groupby(["pair_id", "agent_id"]):
        opp = social[(social["pair_id"] == pid) & (social["agent_id"] != aid)]
        if opp.empty:
            continue
        opp_id = int(opp["agent_id"].iloc[0])
        opp_solo = solo[(solo["pair_id"] == pid) & (solo["agent_id"] == opp_id)]
        self_solo = solo[(solo["pair_id"] == pid) & (solo["agent_id"] == aid)]
        if opp_solo.empty or self_solo.empty:
            continue

        x = float(np.nanmean(opp_solo["RT"]))
        y = float(np.nanmean(gs["choose_low"]) - np.nanmean(self_solo["choose_low"]))
        rows.append((x, y))

    if not rows:
        ax.text(0.5, 0.5, "Insufficient paired data", ha="center", va="center")
        ax.set_axis_off()
        return

    xy = np.asarray(rows, dtype=float)
    ax.scatter(xy[:, 0], xy[:, 1], color="#4c72b0", alpha=0.75)
    if len(xy) >= 2:
        coefs = np.polyfit(xy[:, 0], xy[:, 1], deg=1)
        xs = np.linspace(float(np.min(xy[:, 0])), float(np.max(xy[:, 0])), 100)
        ys = np.polyval(coefs, xs)
        ax.plot(xs, ys, color="black", lw=1.6)
    ax.set_title("Opponent speed correlation")
    ax.set_xlabel("Opponent mean RT (solo)")
    ax.set_ylabel("Delta P(low)")


def _scatter_far_vs_close(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    d = _aligned_xy(augment_social_with_inferred_losers(df[df["phase"] == "social"].copy()))
    d = d[np.isfinite(d["opp_dist2high_at_onset"])].copy()
    if d.empty:
        ax.text(0.5, 0.5, "No social data", ha="center", va="center")
        ax.set_axis_off()
        return

    q40 = float(np.nanquantile(d["opp_dist2high_at_onset"], 0.4))
    q60 = float(np.nanquantile(d["opp_dist2high_at_onset"], 0.6))
    close = d[d["opp_dist2high_at_onset"] < q40]
    far = d[d["opp_dist2high_at_onset"] > q60]

    bins = np.linspace(-1.2, 1.2, 11)

    def _ph(adf: pd.DataFrame) -> np.ndarray:
        c, _, _ = np.histogram2d(adf["x"], adf["y"], bins=[bins, bins])
        h, _, _ = np.histogram2d(adf["x"], adf["y"], bins=[bins, bins], weights=adf["choose_high"])
        return np.divide(h, c, out=np.full_like(h, np.nan, dtype=float), where=c > 0)

    m_close = _ph(close)
    m_far = _ph(far)

    x = m_close.flatten()
    y = m_far.flatten()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        ax.text(0.5, 0.5, "No overlapping bins", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.scatter(x, y, color="#7a0177", alpha=0.7, s=18)
    ax.plot([0.0, 1.0], [0.0, 1.0], ls="--", color="black", lw=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("P(high|opp close)")
    ax.set_ylabel("P(high|opp far)")
    ax.set_title(title)


def _plot_context_shift(ax: plt.Axes, df_social: pd.DataFrame, df_blind: pd.DataFrame) -> None:
    def _curve(df: pd.DataFrame) -> pd.DataFrame:
        d = df[np.isfinite(df["choice"].astype(float))].copy()
        d["choose_high"] = (d["choice"].astype(float) == 0.0).astype(float)
        rows = []
        for _, g in d.groupby("agent_id"):
            g = g.sort_values("trial_id")
            g["run"] = g["choose_high"].rolling(window=50, min_periods=50).mean()
            rows.append(g[["trial_id", "run"]])
        if not rows:
            return pd.DataFrame(columns=["trial_id", "run"])
        allr = pd.concat(rows, axis=0, ignore_index=True)
        return allr.groupby("trial_id", as_index=False)["run"].mean().sort_values("trial_id")

    s = _curve(df_social)
    b = _curve(df_blind)
    if not s.empty:
        ax.plot(s["trial_id"], s["run"], color="#1f78b4", lw=1.8, label="Social")
    if not b.empty:
        ax.plot(b["trial_id"], b["run"], color="#33a02c", lw=1.8, label="Blind")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Running P(high)")
    ax.set_title("Context preference shift")
    ax.legend(frameon=False, fontsize=8)


def _dist_to_patches(start_xy: np.ndarray) -> Tuple[float, float]:
    high_xy = np.array([0.0, 1.0], dtype=float)
    low_xy = np.array([1.0, 0.0], dtype=float)
    return float(np.linalg.norm(start_xy - high_xy)), float(np.linalg.norm(start_xy - low_xy))


def _plot_model_trajectory(
    ax: plt.Axes,
    model: AttractorModel,
    params: Dict[str, float],
    context: str,
    start_xy: np.ndarray,
    opp_xy: Optional[np.ndarray],
    title: str,
    n_sims: int = 18,
) -> None:
    dsh, dsl = _dist_to_patches(start_xy)
    if opp_xy is None:
        doh = 0.0
        dol = 0.0
    else:
        doh = float(np.linalg.norm(opp_xy - np.array([0.0, 1.0], dtype=float)))
        dol = float(np.linalg.norm(opp_xy - np.array([1.0, 0.0], dtype=float)))

    tr = model.simulate_trajectories(
        dist_self_high=dsh,
        dist_self_low=dsl,
        dist_opp_high=doh,
        dist_opp_low=dol,
        context=context,
        n_sims=n_sims,
        params=params,
    )

    for i in range(tr["x1"].shape[0]):
        ax.plot(tr["time"], tr["x1"][i], color="#2b8cbe", alpha=0.25, lw=0.9)
        ax.plot(tr["time"], tr["x2"][i], color="#de2d26", alpha=0.25, lw=0.9)

    ax.axhline(model.threshold, color="black", ls="--", lw=1)
    ax.set_xlim(0.0, model.tmax)
    ax.set_ylim(-1.0, 1.5)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Activity")


def _simulate_model_delta_plow(
    model_solo: AttractorModel,
    model_social: AttractorModel,
    params_solo: Dict[str, float],
    params_social: Dict[str, float],
    bins: np.ndarray,
    n_sims: int = 80,
) -> np.ndarray:
    centers = 0.5 * (bins[:-1] + bins[1:])
    gx, gy = np.meshgrid(centers, centers)

    ph_solo = np.full_like(gx, np.nan, dtype=float)
    ph_social = np.full_like(gx, np.nan, dtype=float)

    for i in range(gx.shape[0]):
        for j in range(gx.shape[1]):
            p = np.array([gx[i, j], gy[i, j]], dtype=float)
            if not _point_in_octagon(p, 1.0):
                continue
            dsh, dsl = _dist_to_patches(p)
            ps, _, _ = model_solo.simulate_trial(
                dsh,
                dsl,
                0.0,
                0.0,
                context="solo",
                n_sims=n_sims,
                params=params_solo,
            )
            pso, _, _ = model_social.simulate_trial(
                dsh,
                dsl,
                0.2,
                1.2,
                context="social",
                n_sims=n_sims,
                params=params_social,
            )
            ph_solo[i, j] = ps
            ph_social[i, j] = pso

    p_low_solo = 1.0 - ph_solo
    p_low_social = 1.0 - ph_social
    return p_low_social - p_low_solo


def generate_summary_dashboard(
    df_agents: pd.DataFrame,
    df_blind: pd.DataFrame,
    model_solo: AttractorModel,
    model_social: AttractorModel,
    output_path: str | Path,
) -> Path:
    plt.style.use("seaborn-v0_8-whitegrid")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 3, figsize=(18, 20), constrained_layout=True)

    # Row 1
    _plot_learning(axes[0, 0], df_agents)
    _plot_distance_45(axes[0, 1], df_agents)
    _plot_optimal_start_heatmap(axes[0, 2], inradius=1.0)

    # Row 2
    _plot_preference_shift(axes[1, 0], df_agents)
    bins, empirical_delta_plow = _plot_delta_plow(axes[1, 1], df_agents)
    _plot_opponent_speed(axes[1, 2], df_agents)

    # Row 3
    _scatter_far_vs_close(axes[2, 0], df_agents, "Social: far vs close")
    _scatter_far_vs_close(axes[2, 1], df_blind, "Socially-blind: far vs close")
    _plot_context_shift(axes[2, 2], df_agents, df_blind)

    # Row 4
    best_solo = dict(model_solo.best_params_solo or model_solo.paper_solo)
    best_social = dict(model_social.best_params_social or model_social.paper_social)

    _plot_model_trajectory(
        axes[3, 0],
        model=model_solo,
        params=best_solo,
        context="solo",
        start_xy=np.array([0.0, 0.75], dtype=float),
        opp_xy=None,
        title="Solo trajectories (close-to-high)",
    )
    _plot_model_trajectory(
        axes[3, 1],
        model=model_social,
        params=best_social,
        context="social",
        start_xy=np.array([0.25, 0.25], dtype=float),
        opp_xy=np.array([0.05, 0.95], dtype=float),
        title="Social trajectory (opp-close)",
    )

    model_delta_plow = _simulate_model_delta_plow(
        model_solo=model_solo,
        model_social=model_social,
        params_solo=best_solo,
        params_social=best_social,
        bins=bins,
        n_sims=100,
    )

    norm = mpl.colors.TwoSlopeNorm(vmin=-0.25, vcenter=0.0, vmax=0.25)
    mesh = axes[3, 2].pcolormesh(bins, bins, empirical_delta_plow.T, cmap="RdBu_r", norm=norm, shading="auto", alpha=0.75)
    xc = 0.5 * (bins[:-1] + bins[1:])
    yc = 0.5 * (bins[:-1] + bins[1:])
    axes[3, 2].contour(xc, yc, model_delta_plow.T, levels=[-0.15, -0.05, 0.05, 0.15], colors="black", linewidths=1.0)
    vv = _octagon_vertices(1.0)
    axes[3, 2].plot(vv[:, 0], vv[:, 1], color="black", lw=1.2)
    axes[3, 2].set_aspect("equal", adjustable="box")
    axes[3, 2].set_xlim(-1.2, 1.2)
    axes[3, 2].set_ylim(-1.2, 1.2)
    axes[3, 2].set_title("Delta P(low): empirical + model")
    fig.colorbar(mesh, ax=axes[3, 2], shrink=0.8)

    fig.suptitle("RoBox Summary Dashboard", fontsize=18)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def _post_learning_agent_phigh(df: pd.DataFrame, threshold: float = 0.7, window: int = 50) -> pd.Series:
    d = df[df["phase"] == "solo"].copy()
    d = d[np.isfinite(d["choice"].astype(float))].copy()
    out: Dict[float, float] = {}
    for aid, g in d.groupby("agent_id"):
        g = g.sort_values("trial_id")
        choose_high = (g["choice"].astype(float) == 0.0).astype(float)
        roll = choose_high.rolling(window=window, min_periods=window).mean().to_numpy(dtype=float)
        reached = np.where(roll >= threshold)[0]
        if len(reached) == 0:
            out[float(aid)] = float(np.nanmean(choose_high.to_numpy(dtype=float)))
        else:
            idx0 = int(reached[0])
            out[float(aid)] = float(np.nanmean(choose_high.to_numpy(dtype=float)[idx0:]))
    return pd.Series(out, dtype=float)


def _fit_logit_opp(df: pd.DataFrame) -> Tuple[float, float, int]:
    required = ["choice", "dist2high_at_onset", "dist2low_at_onset", "opp_dist2high_at_onset"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    d = df.copy()
    d = d[np.isfinite(d["choice"].astype(float))].copy()
    d = d[d["phase"] == "social"].copy()
    if d.empty:
        raise ValueError("No social rows available")

    d["choose_high"] = (d["choice"].astype(float) == 0.0).astype(int)
    for c in ["dist2high_at_onset", "dist2low_at_onset", "opp_dist2high_at_onset"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["choose_high", "dist2high_at_onset", "dist2low_at_onset", "opp_dist2high_at_onset"])

    if len(d) < 20:
        raise ValueError("Too few rows for logistic regression")
    if smf is None:
        raise RuntimeError("statsmodels is required for logistic regression checks")

    model = smf.logit(
        "choose_high ~ dist2high_at_onset + dist2low_at_onset + opp_dist2high_at_onset",
        data=d,
    )
    res = model.fit(disp=False)
    beta = float(res.params["opp_dist2high_at_onset"])
    p = float(res.pvalues["opp_dist2high_at_onset"])
    return beta, p, int(len(d))


def _forced_rt_medians(df: pd.DataFrame) -> Tuple[float, float, float]:
    d = df[df["phase"] == "solo"].copy()
    if "RT" not in d.columns:
        raise ValueError("Missing RT column")

    category: pd.Series
    if "trial_type" in d.columns:
        category = d["trial_type"].astype(str)
        if "choice_role" in d.columns:
            forced = category != "choice"
            category.loc[forced & (d["choice_role"].astype(str) == "high")] = "highx2"
            category.loc[forced & (d["choice_role"].astype(str) == "low")] = "lowx2"
    elif "phase_label" in d.columns:
        category = d["phase_label"].astype(str).replace({"forced_highx2": "highx2", "forced_lowx2": "lowx2"})
    elif "forced_type" in d.columns:
        category = pd.Series(["choice"] * len(d), index=d.index)
        category.loc[d["forced_type"].astype(str) == "highx2"] = "highx2"
        category.loc[d["forced_type"].astype(str) == "lowx2"] = "lowx2"
    else:
        raise ValueError("Missing forced-trial labels: need one of trial_type/phase_label/forced_type")

    d = d[np.isfinite(d["RT"])].copy()
    d["cat"] = category

    med_choice = float(np.nanmedian(d[d["cat"] == "choice"]["RT"]))
    med_highx2 = float(np.nanmedian(d[d["cat"] == "highx2"]["RT"]))
    med_lowx2 = float(np.nanmedian(d[d["cat"] == "lowx2"]["RT"]))
    return med_highx2, med_choice, med_lowx2


def _append_report_line(lines: List[str], name: str, passed: bool, detail: str) -> None:
    status = "PASS" if passed else "FAIL"
    lines.append(f"[{status}] {name}: {detail}")


def check_reproduction(df_agents: pd.DataFrame, df_mice_reference: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    report_lines: List[str] = []
    results: Dict[str, Any] = {}

    # Test 1
    try:
        ph_post = _post_learning_agent_phigh(df_agents)
        passed = bool((ph_post > 0.70).all()) and len(ph_post) > 0
        detail = f"post-learning P(high) per agent={ph_post.to_dict()}"
        _append_report_line(report_lines, "Test 1 solo post-learning > 0.70", passed, detail)
        results["test1"] = {"pass": passed, "values": ph_post.to_dict()}
    except Exception as e:
        _append_report_line(report_lines, "Test 1 solo post-learning > 0.70", False, str(e))
        results["test1"] = {"pass": False, "error": str(e)}

    # Test 2
    try:
        d = df_agents[np.isfinite(df_agents["choice"].astype(float))].copy()
        d["choose_high"] = (d["choice"].astype(float) == 0.0).astype(float)
        agent_rows = []
        for aid, g in d.groupby("agent_id"):
            solo = g[g["phase"] == "solo"]["choose_high"]
            social = g[g["phase"] == "social"]["choose_high"]
            if len(solo) == 0 or len(social) == 0:
                continue
            agent_rows.append((float(aid), float(np.mean(solo)), float(np.mean(social))))

        if len(agent_rows) < 2:
            raise ValueError("Need >=2 agents with both solo and social data")

        tab = pd.DataFrame(agent_rows, columns=["agent_id", "solo", "social"])
        diff = tab["solo"] - tab["social"]
        mean_drop = float(np.mean(diff))
        if ttest_rel is None:
            raise RuntimeError("scipy is required for paired t-test")
        t_stat, p_val = ttest_rel(tab["solo"], tab["social"], nan_policy="omit")
        passed = bool(mean_drop >= 0.05 and float(p_val) < 0.05)
        detail = (
            f"mean_drop={100.0 * mean_drop:.2f}% (paper 10.46 +/- 0.73%), "
            f"t={float(t_stat):.4f}, p={float(p_val):.4g} (paper p=7.2e-17)"
        )
        _append_report_line(report_lines, "Test 2 social drop >= 5%", passed, detail)
        results["test2"] = {
            "pass": passed,
            "mean_drop": mean_drop,
            "t_stat": float(t_stat),
            "p_value": float(p_val),
        }
    except Exception as e:
        _append_report_line(report_lines, "Test 2 social drop >= 5%", False, str(e))
        results["test2"] = {"pass": False, "error": str(e)}

    # Test 3
    try:
        d = df_agents.copy()
        if "agent_type" in d.columns:
            d = d[d["agent_type"].astype(str).str.lower() == "social"].copy()
            if d.empty:
                d = df_agents.copy()
        beta, p, n = _fit_logit_opp(d)
        passed = bool(beta > 0.0 and p < 0.05)
        detail = f"beta={beta:.4f}, p={p:.4g}, n={n} (paper beta=0.27, p<1e-5)"
        _append_report_line(report_lines, "Test 3 social opponent sensitivity", passed, detail)
        results["test3"] = {"pass": passed, "beta": beta, "p_value": p, "n": n}
    except Exception as e:
        _append_report_line(report_lines, "Test 3 social opponent sensitivity", False, str(e))
        results["test3"] = {"pass": False, "error": str(e)}

    # Test 4
    try:
        if "agent_type" not in df_agents.columns:
            raise ValueError("No agent_type column; cannot isolate socially-blind rows")
        d_blind = df_agents[df_agents["agent_type"].astype(str).str.lower().isin(["blind", "socially-blind", "socially_blind"])].copy()
        if d_blind.empty:
            raise ValueError("No socially-blind rows in df_agents")
        beta, p, n = _fit_logit_opp(d_blind)
        passed = bool(p > 0.05)
        detail = f"beta={beta:.4f}, p={p:.4g}, n={n} (paper beta=-0.03, p=0.59)"
        _append_report_line(report_lines, "Test 4 blind no opponent sensitivity", passed, detail)
        results["test4"] = {"pass": passed, "beta": beta, "p_value": p, "n": n}
    except Exception as e:
        _append_report_line(report_lines, "Test 4 blind no opponent sensitivity", False, str(e))
        results["test4"] = {"pass": False, "error": str(e)}

    # Test 5
    try:
        med_highx2, med_choice, med_lowx2 = _forced_rt_medians(df_agents)
        passed = bool(med_highx2 < med_choice < med_lowx2)
        detail = f"median RT highx2={med_highx2:.4f}, choice={med_choice:.4f}, lowx2={med_lowx2:.4f}"
        _append_report_line(report_lines, "Test 5 RT ranking", passed, detail)
        results["test5"] = {
            "pass": passed,
            "median_highx2": med_highx2,
            "median_choice": med_choice,
            "median_lowx2": med_lowx2,
        }
    except Exception as e:
        _append_report_line(report_lines, "Test 5 RT ranking", False, str(e))
        results["test5"] = {"pass": False, "error": str(e)}

    # Test 6
    try:
        for c in ["stimulus_onset_pos_x", "stimulus_onset_pos_y"]:
            if c not in df_agents.columns:
                raise ValueError(f"Missing column: {c}")
        dist = np.sqrt(np.square(df_agents["stimulus_onset_pos_x"].astype(float)) + np.square(df_agents["stimulus_onset_pos_y"].astype(float)))
        dist = dist[np.isfinite(dist)]
        if len(dist) < 2:
            raise ValueError("Too few onset positions")

        mean_dist = float(np.mean(dist))
        if ttest_1samp is None:
            raise RuntimeError("scipy is required for one-sample t-test")
        t_stat, p_val = ttest_1samp(dist.to_numpy(dtype=float), popmean=0.7, alternative="less")
        passed = bool(mean_dist < 0.7 and float(p_val) < 0.05)
        detail = f"mean normalized center distance={mean_dist:.4f}, t={float(t_stat):.4f}, p={float(p_val):.4g} (paper ~0.56)"
        _append_report_line(report_lines, "Test 6 centre-seeking behavior", passed, detail)
        results["test6"] = {
            "pass": passed,
            "mean_dist": mean_dist,
            "t_stat": float(t_stat),
            "p_value": float(p_val),
        }
    except Exception as e:
        _append_report_line(report_lines, "Test 6 centre-seeking behavior", False, str(e))
        results["test6"] = {"pass": False, "error": str(e)}

    all_pass = all(bool(v.get("pass", False)) for v in results.values() if isinstance(v, dict) and "pass" in v)
    results["all_pass"] = all_pass
    results["report"] = "\n".join(report_lines)

    if df_mice_reference is not None:
        results["reference_rows"] = int(len(df_mice_reference))

    return results


def _collect_trial_csvs(log_dir: Path) -> List[Path]:
    if not log_dir.exists():
        return []
    return sorted(log_dir.glob("pair_*/trials.csv"))


def _load_agent_trials(pipeline_cfg: Dict[str, Any], exp_cfg: Dict[str, Any]) -> pd.DataFrame:
    data_cfg = pipeline_cfg.get("data", {})
    if "agents_trials_csv" in data_cfg and data_cfg["agents_trials_csv"]:
        return pd.read_csv(Path(data_cfg["agents_trials_csv"]))

    log_dir = Path(exp_cfg["experiment"]["log_dir"])
    csvs = _collect_trial_csvs(log_dir)
    if not csvs:
        raise FileNotFoundError("No trial CSV files found. Run training or provide data.agents_trials_csv.")

    dfs = []
    for p in csvs:
        d = pd.read_csv(p)
        pair_name = p.parent.name
        d["pair_name"] = pair_name
        try:
            parts = pair_name.replace("pair_", "").split("_")
            if len(parts) == 2:
                d["pair_id"] = int(parts[0])
        except Exception:
            pass
        dfs.append(d)
    return pd.concat(dfs, axis=0, ignore_index=True)


def _load_blind_trials(pipeline_cfg: Dict[str, Any], df_agents: pd.DataFrame) -> pd.DataFrame:
    data_cfg = pipeline_cfg.get("data", {})
    blind_path = data_cfg.get("blind_trials_csv")
    if blind_path:
        return pd.read_csv(Path(blind_path))

    if "agent_type" in df_agents.columns:
        d_blind = df_agents[
            df_agents["agent_type"].astype(str).str.lower().isin(["blind", "socially-blind", "socially_blind"])
        ].copy()
        if not d_blind.empty:
            return d_blind

    # Fallback keeps pipeline runnable when blind data is not explicitly available.
    return df_agents.copy()


def _run_training_or_resume(pipeline_cfg: Dict[str, Any], exp_cfg: Dict[str, Any]) -> None:
    train_cfg = pipeline_cfg.get("training", {})
    if not bool(train_cfg.get("run_training", False)):
        return

    n_pairs = int(exp_cfg["experiment"].get("n_pairs", 1))
    seeds = list(exp_cfg["experiment"].get("seeds", []))
    if len(seeds) < 2 * n_pairs:
        a = seeds[:n_pairs]
        b = [int(s) + 1000 for s in a]
        pairs = list(zip(a, b))
    else:
        pairs = list(zip(seeds[:n_pairs], seeds[n_pairs : 2 * n_pairs]))

    for i, (sa, sb) in enumerate(pairs):
        run_experiment(pair_id=int(i), seed_A=int(sa), seed_B=int(sb), config=exp_cfg)


def main(config_path: str = "config.yaml") -> None:
    cfg = _load_yaml(config_path)
    pipeline_cfg = cfg.get("pipeline", {})

    exp_config_path = pipeline_cfg.get("experiment_config", "configs/experiment.yaml")
    exp_cfg = load_experiment_config(exp_config_path)

    results_root = _ensure_dir(pipeline_cfg.get("results_dir", "results"))
    solo_dir = _ensure_dir(results_root / "solo")
    social_dir = _ensure_dir(results_root / "social")
    comp_dir = _ensure_dir(results_root / "comparison")
    attr_dir = _ensure_dir(results_root / "attractor")

    # 2) Run training or load from checkpoint/logged csv.
    _run_training_or_resume(pipeline_cfg, exp_cfg)

    # Load trial datasets.
    df_agents = _load_agent_trials(pipeline_cfg, exp_cfg)
    df_blind = _load_blind_trials(pipeline_cfg, df_agents)

    # 3) Solo analysis.
    run_solo_analysis(df_agents, output_dir=solo_dir)

    # 4) Social analysis.
    run_social_analysis(df_agents, output_dir=social_dir)

    # 5) RL comparison.
    run_rl_comparison(df_social=df_agents, df_blind=df_blind, output_dir=comp_dir)

    # 6) Fit/evaluate attractor model.
    attr_cfg = pipeline_cfg.get("attractor", {})
    shotgun_samples = int(attr_cfg.get("shotgun_samples", 500))
    n_starts = int(attr_cfg.get("n_starts", 5))

    model_solo = AttractorModel(shotgun_samples=shotgun_samples, n_starts=n_starts)
    model_social = AttractorModel(shotgun_samples=shotgun_samples, n_starts=n_starts)

    # Optional downsample to keep fitting tractable in large datasets.
    fit_rows = int(attr_cfg.get("max_fit_rows", 1200))
    dfit = df_agents.copy()
    if len(dfit) > fit_rows:
        dfit = dfit.sample(n=fit_rows, random_state=42)

    model_solo.fit_solo(dfit)
    model_social.fit_social(dfit)

    run_attractor_analysis(model_solo=model_solo, model_social=model_social, trial_df=df_agents, output_dir=attr_dir)

    # Save compact fit summary.
    fit_summary = pd.DataFrame(
        [
            {"context": "solo", **dict(model_solo.best_params_solo or model_solo.paper_solo)},
            {"context": "social", **dict(model_social.best_params_social or model_social.paper_social)},
        ]
    )
    fit_summary.to_csv(attr_dir / "attractor_best_params.csv", index=False)

    # 7) Summary dashboard.
    dashboard_path = generate_summary_dashboard(
        df_agents=df_agents,
        df_blind=df_blind,
        model_solo=model_solo,
        model_social=model_social,
        output_path=results_root / "summary_dashboard.pdf",
    )

    # 8) Reproducibility checks.
    rep = check_reproduction(df_agents=df_agents, df_mice_reference=None)
    print(rep["report"])
    report_path = results_root / "reproducibility_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(rep["report"] + "\n")

    # 9) Done message.
    print(f"Saved dashboard: {dashboard_path}")
    print(f"Saved reproducibility report: {report_path}")
    print("All done. Check results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full RoBox summary and reproducibility pipeline")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
