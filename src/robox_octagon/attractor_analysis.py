from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from .attractor_model import AttractorModel


def _save_fig(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", dpi=300, bbox_inches="tight")


def _octagon_vertices(inradius: float = 1.0) -> np.ndarray:
    circum = float(inradius) / np.cos(np.pi / 8.0)
    angles = np.deg2rad(np.arange(22.5, 360.0 + 22.5, 45.0))
    return np.stack([circum * np.cos(angles), circum * np.sin(angles)], axis=1)


def _point_in_octagon(point: np.ndarray, inradius: float = 1.0) -> bool:
    angles = np.deg2rad(np.arange(0, 360, 45, dtype=float))
    normals = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return bool(np.all(normals @ point <= inradius + 1e-12))


def _default_best(model: AttractorModel, context: str) -> Dict[str, float]:
    if context == "solo":
        return dict(model.best_params_solo or model.paper_solo)
    return dict(model.best_params_social or model.paper_social)


def _dist_to_patches(start_xy: np.ndarray) -> Tuple[float, float]:
    high_xy = np.array([0.0, 1.0], dtype=float)
    low_xy = np.array([1.0, 0.0], dtype=float)
    return float(np.linalg.norm(start_xy - high_xy)), float(np.linalg.norm(start_xy - low_xy))


def _draw_inset(
    ax: plt.Axes,
    agent_xy: np.ndarray,
    opp_xy: Optional[np.ndarray] = None,
    high_xy: np.ndarray = np.array([0.0, 1.0]),
    low_xy: np.ndarray = np.array([1.0, 0.0]),
) -> None:
    iax = ax.inset_axes([0.70, 0.52, 0.25, 0.4])
    v = _octagon_vertices(1.0)
    iax.plot(v[:, 0], v[:, 1], color="black", lw=1)
    iax.scatter([agent_xy[0]], [agent_xy[1]], color="green", s=30)
    iax.scatter([high_xy[0]], [high_xy[1]], color="blue", marker="^", s=35)
    iax.scatter([low_xy[0]], [low_xy[1]], color="red", marker="^", s=35)
    if opp_xy is not None:
        iax.scatter([opp_xy[0]], [opp_xy[1]], color="orange", marker="s", s=30)
    iax.set_xlim(-1.25, 1.25)
    iax.set_ylim(-1.25, 1.25)
    iax.set_aspect("equal", adjustable="box")
    iax.set_xticks([])
    iax.set_yticks([])


def _simulate_grid_phigh(
    model: AttractorModel,
    params: Dict[str, float],
    context: str,
    opp_dist_high: float,
    opp_dist_low: float,
    n_sims: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bins = np.linspace(-1.2, 1.2, 11)
    centers = 0.5 * (bins[:-1] + bins[1:])
    grid_x, grid_y = np.meshgrid(centers, centers)

    ph = np.full_like(grid_x, np.nan, dtype=float)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            p = np.array([grid_x[i, j], grid_y[i, j]], dtype=float)
            if not _point_in_octagon(p, inradius=1.0):
                continue
            dsh, dsl = _dist_to_patches(p)
            p_high, _, _ = model.simulate_trial(
                dist_self_high=dsh,
                dist_self_low=dsl,
                dist_opp_high=opp_dist_high,
                dist_opp_low=opp_dist_low,
                context=context,
                n_sims=n_sims,
                params=params,
            )
            ph[i, j] = p_high

    return bins, grid_x, ph


def _plot_trajectories_panel(
    ax: plt.Axes,
    model: AttractorModel,
    params: Dict[str, float],
    context: str,
    start_xy: np.ndarray,
    opp_xy: Optional[np.ndarray],
    n_sims: int,
    title_prefix: str,
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

    t = tr["time"]
    for k in range(tr["x1"].shape[0]):
        ax.plot(t, tr["x1"][k], color="blue", alpha=0.25, lw=1)
        ax.plot(t, tr["x2"][k], color="red", alpha=0.25, lw=1)

    ax.axhline(model.threshold, color="black", ls="--", lw=1)
    ax.set_xlim(0.0, model.tmax)
    ax.set_ylim(-1.0, 1.5)
    ax.set_xlabel("Time from stimulus onset (s)")
    ax.set_ylabel("Node activity")
    ax.set_title(f"{title_prefix} | p(high) = {tr['p_high']:.2f}")
    _draw_inset(ax, agent_xy=start_xy, opp_xy=opp_xy)


def _plot_forced_rt(model: AttractorModel, params: Dict[str, float], n_sims: int = 400) -> Tuple[List[str], np.ndarray]:
    # Use one representative ambiguous start location.
    start = np.array([0.0, 0.0], dtype=float)
    dsh, dsl = _dist_to_patches(start)

    baseline_vlow = float(model.v_low)

    try:
        # choice: V1=V_H, V2=V_L
        _, rt_choice, _ = model.simulate_trial(
            dsh, dsl, 0.0, 0.0, context="solo", n_sims=n_sims, params=params
        )

        # highx2: V1=V2=V_H
        highx2_params = dict(params)
        model.v_low = float(params["V_H"])
        _, rt_highx2, _ = model.simulate_trial(
            dsh, dsl, 0.0, 0.0, context="solo", n_sims=n_sims, params=highx2_params
        )

        # lowx2: V1=V2=V_low
        lowx2_params = dict(params)
        lowx2_params["V_H"] = baseline_vlow
        model.v_low = baseline_vlow
        _, rt_lowx2, _ = model.simulate_trial(
            dsh, dsl, 0.0, 0.0, context="solo", n_sims=n_sims, params=lowx2_params
        )
    finally:
        # Always restore model state even if simulation fails.
        model.v_low = baseline_vlow

    meds = np.array([rt_choice, rt_highx2, rt_lowx2], dtype=float)
    meds = meds / max(1e-12, rt_highx2)
    return ["choice", "highx2", "lowx2"], meds


def _top_percent_cloud(
    model: AttractorModel,
    trial_df: pd.DataFrame,
    context: str,
    top_pct: float = 2.5,
    n_sims: int = 80,
    min_samples: int = 400,
) -> pd.DataFrame:
    hist = model.fit_history.get(context, [])

    if len(hist) < min_samples:
        # Build approximate cloud if fit history is unavailable/short.
        samples = max(min_samples, len(hist))
        hist = []
        for _ in range(samples):
            p = model._sample_param_set(context=context)
            ll = model.log_likelihood(trial_df, params=p, context=context, n_sims=n_sims)
            hist.append((ll, p))

    scored = sorted(hist, key=lambda x: x[0], reverse=True)
    k = max(1, int(np.ceil((top_pct / 100.0) * len(scored))))
    top = scored[:k]

    rows = []
    for ll, p in top:
        rows.append({"sigma": float(p["sigma"]), "V_H": float(p["V_H"]), "log_likelihood": float(ll)})
    return pd.DataFrame(rows)


def run_attractor_analysis(
    model_solo: AttractorModel,
    model_social: AttractorModel,
    trial_df: pd.DataFrame,
    output_dir: str | Path,
) -> Dict[str, Path]:
    """Generate Figure-4 style attractor-model visualization suite."""
    plt.style.use("seaborn-v0_8-whitegrid")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_solo = _default_best(model_solo, context="solo")
    best_social = _default_best(model_social, context="social")

    outputs: Dict[str, Path] = {}

    # Plot 16: solo trajectories (close to high vs close to low).
    fig16, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    _plot_trajectories_panel(
        axes[0],
        model=model_solo,
        params=best_solo,
        context="solo",
        start_xy=np.array([0.0, 0.75], dtype=float),
        opp_xy=None,
        n_sims=20,
        title_prefix="Solo, self close to high",
    )
    _plot_trajectories_panel(
        axes[1],
        model=model_solo,
        params=best_solo,
        context="solo",
        start_xy=np.array([0.75, 0.0], dtype=float),
        opp_xy=None,
        n_sims=20,
        title_prefix="Solo, self close to low",
    )
    _save_fig(fig16, out_dir, "plot16_phase_space_solo")
    plt.close(fig16)
    outputs["plot16"] = out_dir / "plot16_phase_space_solo.png"

    # Plot 17: solo model P(high) heatmap.
    fig17, ax17 = plt.subplots(figsize=(10, 8))
    bins, _, ph_solo = _simulate_grid_phigh(
        model=model_solo,
        params=best_solo,
        context="solo",
        opp_dist_high=0.0,
        opp_dist_low=0.0,
        n_sims=120,
    )
    norm = mpl.colors.TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    mesh = ax17.pcolormesh(bins, bins, ph_solo.T, cmap="RdBu", norm=norm, shading="auto")
    v = _octagon_vertices(1.0)
    ax17.plot(v[:, 0], v[:, 1], color="black", lw=1.5)
    ax17.set_aspect("equal", adjustable="box")
    ax17.set_xlim(-1.2, 1.2)
    ax17.set_ylim(-1.2, 1.2)
    ax17.set_title("Solo model P(high) heatmap")
    cbar = fig17.colorbar(mesh, ax=ax17, shrink=0.85)
    cbar.set_label("P(choose high)")
    _save_fig(fig17, out_dir, "plot17_solo_phigh_heatmap")
    plt.close(fig17)
    outputs["plot17"] = out_dir / "plot17_solo_phigh_heatmap.png"

    # Plot 18: forced trial RT analysis.
    labels, meds = _plot_forced_rt(model_solo, params=best_solo, n_sims=600)
    fig18, ax18 = plt.subplots(figsize=(10, 8))
    ax18.bar(np.arange(len(labels)), meds, color=["#4c72b0", "#55a868", "#c44e52"], alpha=0.85)
    ax18.set_xticks(np.arange(len(labels)))
    ax18.set_xticklabels(labels)
    ax18.set_ylabel("Median RT / RT(highx2)")
    ax18.set_title("Simulated reaction times (forced-trial analysis)")
    _save_fig(fig18, out_dir, "plot18_forced_rt")
    plt.close(fig18)
    outputs["plot18"] = out_dir / "plot18_forced_rt.png"

    # Plot 19: social trajectories (opp far vs close to high).
    fig19, axes19 = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)
    _plot_trajectories_panel(
        axes19[0],
        model=model_social,
        params=best_social,
        context="social",
        start_xy=np.array([0.25, 0.25], dtype=float),
        opp_xy=np.array([0.9, -0.1], dtype=float),
        n_sims=20,
        title_prefix="Social, opponent far from high",
    )
    _plot_trajectories_panel(
        axes19[1],
        model=model_social,
        params=best_social,
        context="social",
        start_xy=np.array([0.25, 0.25], dtype=float),
        opp_xy=np.array([0.05, 0.95], dtype=float),
        n_sims=20,
        title_prefix="Social, opponent close to high",
    )
    _save_fig(fig19, out_dir, "plot19_social_trajectories")
    plt.close(fig19)
    outputs["plot19"] = out_dir / "plot19_social_trajectories.png"

    # Plot 20: social P(high) and delta heatmaps.
    bins20, _, ph_social = _simulate_grid_phigh(
        model=model_social,
        params=best_social,
        context="social",
        opp_dist_high=0.15,
        opp_dist_low=1.2,
        n_sims=120,
    )
    _, _, ph_solo20 = _simulate_grid_phigh(
        model=model_solo,
        params=best_solo,
        context="solo",
        opp_dist_high=0.0,
        opp_dist_low=0.0,
        n_sims=120,
    )
    delta = ph_social - ph_solo20

    fig20, axes20 = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    n1 = mpl.colors.TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    m1 = axes20[0].pcolormesh(bins20, bins20, ph_social.T, cmap="RdBu", norm=n1, shading="auto")
    n2 = mpl.colors.TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)
    m2 = axes20[1].pcolormesh(bins20, bins20, delta.T, cmap="RdBu_r", norm=n2, shading="auto")

    for ax in axes20:
        vv = _octagon_vertices(1.0)
        ax.plot(vv[:, 0], vv[:, 1], color="black", lw=1.2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
    axes20[0].set_title("Social model P(high)")
    axes20[1].set_title("Delta P(high): social - solo")
    fig20.colorbar(m1, ax=axes20[0], shrink=0.85)
    fig20.colorbar(m2, ax=axes20[1], shrink=0.85)
    _save_fig(fig20, out_dir, "plot20_social_and_delta_heatmaps")
    plt.close(fig20)
    outputs["plot20"] = out_dir / "plot20_social_and_delta_heatmaps.png"

    # Plot 21: opponent position influence map: far - close.
    bins21, _, ph_far = _simulate_grid_phigh(
        model=model_social,
        params=best_social,
        context="social",
        opp_dist_high=1.5,
        opp_dist_low=0.2,
        n_sims=120,
    )
    _, _, ph_close = _simulate_grid_phigh(
        model=model_social,
        params=best_social,
        context="social",
        opp_dist_high=0.2,
        opp_dist_low=1.5,
        n_sims=120,
    )
    diff_opp = ph_far - ph_close

    fig21, ax21 = plt.subplots(figsize=(10, 8))
    n21 = mpl.colors.TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)
    m21 = ax21.pcolormesh(bins21, bins21, diff_opp.T, cmap="RdBu_r", norm=n21, shading="auto")
    vv = _octagon_vertices(1.0)
    ax21.plot(vv[:, 0], vv[:, 1], color="black", lw=1.2)
    ax21.set_aspect("equal", adjustable="box")
    ax21.set_xlim(-1.2, 1.2)
    ax21.set_ylim(-1.2, 1.2)
    ax21.set_title("Opponent position influence: P(high|opp_far) - P(high|opp_close)")
    fig21.colorbar(m21, ax=ax21, shrink=0.85)
    _save_fig(fig21, out_dir, "plot21_opponent_position_influence")
    plt.close(fig21)
    outputs["plot21"] = out_dir / "plot21_opponent_position_influence.png"

    # Plot 22: sigma vs V_H top 2.5% fit cloud.
    d_solo = trial_df.copy()
    if "phase" in d_solo.columns:
        d_solo = d_solo[d_solo["phase"] != "social"].copy()
    d_social = trial_df.copy()
    if "phase" in d_social.columns:
        d_social = d_social[d_social["phase"] == "social"].copy()

    cloud_solo = _top_percent_cloud(model_solo, d_solo, context="solo", top_pct=2.5)
    cloud_social = _top_percent_cloud(model_social, d_social if len(d_social) > 0 else trial_df, context="social", top_pct=2.5)

    fig22, ax22 = plt.subplots(figsize=(10, 8))
    if len(cloud_solo) > 0:
        ax22.scatter(cloud_solo["sigma"], cloud_solo["V_H"], color="black", s=20, alpha=0.6, label="solo top 2.5%")
    if len(cloud_social) > 0:
        ax22.scatter(cloud_social["sigma"], cloud_social["V_H"], color="purple", s=20, alpha=0.6, label="social top 2.5%")

    ax22.scatter([best_solo["sigma"]], [best_solo["V_H"]], facecolors="white", edgecolors="black", s=120, linewidths=2, label="best solo")
    ax22.scatter([best_social["sigma"]], [best_social["V_H"]], facecolors="magenta", edgecolors="purple", s=120, linewidths=2, label="best social")

    ax22.set_xlabel("sigma (noise)")
    ax22.set_ylabel("V_H (high-value input)")
    ax22.set_title("Parameter comparison: solo vs social fits")
    ax22.legend(frameon=False)
    _save_fig(fig22, out_dir, "plot22_parameter_comparison")
    plt.close(fig22)
    outputs["plot22"] = out_dir / "plot22_parameter_comparison.png"

    return outputs
