# Author: Samuel Lozano
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

try:
    from scipy.stats import pearsonr
except Exception:  # pragma: no cover
    pearsonr = None

try:
    import statsmodels.formula.api as smf
except Exception:  # pragma: no cover
    smf = None

from .social_analysis import augment_social_with_inferred_losers


def _require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _save_fig(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", dpi=300, bbox_inches="tight")


def _octagon_vertices(inradius: float = 1.0) -> np.ndarray:
    circum = float(inradius) / np.cos(np.pi / 8.0)
    angles = np.deg2rad(np.arange(22.5, 360.0 + 22.5, 45.0))
    return np.stack([circum * np.cos(angles), circum * np.sin(angles)], axis=1)


def _rotate_points(xy: np.ndarray, angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    r = np.array([[c, -s], [s, c]], dtype=float)
    return xy @ r.T


def _transform_trial_frame(start_xy: np.ndarray, high_patch_id: int, low_patch_id: int) -> np.ndarray:
    high_angle = np.deg2rad(float(high_patch_id) * 45.0)
    rot = np.deg2rad(90.0) - high_angle
    p = _rotate_points(start_xy[None, :], rot)[0]

    low_angle = np.deg2rad(float(low_patch_id) * 45.0)
    low_rot = low_angle + rot
    low_xy = np.array([np.cos(low_rot), np.sin(low_rot)], dtype=float)
    if low_xy[0] < 0.0:
        p[0] *= -1.0
    return p


def _prepare_social_aligned(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        df,
        [
            "phase",
            "trial_id",
            "agent_id",
            "choice",
            "high_patch_id",
            "low_patch_id",
            "stimulus_onset_pos_x",
            "stimulus_onset_pos_y",
            "opp_dist2high_at_onset",
        ],
    )

    social = df[df["phase"] == "social"].copy()
    social = augment_social_with_inferred_losers(social)
    social = social[np.isfinite(social["choice"].astype(float))].copy()

    rows: List[Dict[str, float]] = []
    for _, r in social.iterrows():
        hp = float(r.get("high_patch_id", np.nan))
        lp = float(r.get("low_patch_id", np.nan))
        if not np.isfinite(hp) or not np.isfinite(lp):
            continue

        start = np.array([float(r["stimulus_onset_pos_x"]), float(r["stimulus_onset_pos_y"])], dtype=float)
        p = _transform_trial_frame(start, int(hp), int(lp))
        rows.append(
            {
                "trial_id": float(r["trial_id"]),
                "agent_id": float(r["agent_id"]),
                "x": float(p[0]),
                "y": float(p[1]),
                "choose_high": 1.0 if float(r["choice"]) == 0.0 else 0.0,
                "choose_low": 1.0 if float(r["choice"]) == 1.0 else 0.0,
                "opp_dist2high_at_onset": float(r.get("opp_dist2high_at_onset", np.nan)),
                "dist2high_at_onset": float(r.get("dist2high_at_onset", np.nan)),
                "dist2low_at_onset": float(r.get("dist2low_at_onset", np.nan)),
                "phase": "social",
            }
        )

    return pd.DataFrame(rows)


def _phigh_map(adf: pd.DataFrame, bins: np.ndarray) -> np.ndarray:
    c, _, _ = np.histogram2d(adf["x"], adf["y"], bins=[bins, bins])
    h, _, _ = np.histogram2d(adf["x"], adf["y"], bins=[bins, bins], weights=adf["choose_high"])
    return np.divide(h, c, out=np.full_like(h, np.nan, dtype=float), where=c > 0)


def _plow_map_weighted(adf: pd.DataFrame, bins: np.ndarray, rw: float) -> np.ndarray:
    d = adf.sort_values("trial_id").copy()
    t = d["trial_id"].to_numpy(dtype=float)
    if len(t) == 0:
        return np.full((len(bins) - 1, len(bins) - 1), np.nan, dtype=float)

    # Exponential recency weight: larger rw => faster decay of old trials.
    tmax = float(np.max(t))
    w = np.exp(-float(rw) * (tmax - t))
    d["w"] = w

    c, _, _ = np.histogram2d(d["x"], d["y"], bins=[bins, bins], weights=d["w"])
    l, _, _ = np.histogram2d(d["x"], d["y"], bins=[bins, bins], weights=d["w"] * d["choose_low"])
    return np.divide(l, c, out=np.full_like(l, np.nan, dtype=float), where=c > 1e-12)


def plot_agent_sensitivity(df: pd.DataFrame, agent_type: str, output_dir: str | Path) -> plt.Figure:
    """Figure-3 style opponent sensitivity panels for one agent type."""
    plt.style.use("seaborn-v0_8-whitegrid")

    d = _prepare_social_aligned(df)
    if d.empty:
        raise ValueError("No social trials available after loser inference")

    d = d[np.isfinite(d["opp_dist2high_at_onset"])].copy()
    if d.empty:
        raise ValueError("No finite opp_dist2high_at_onset values")

    q40 = float(np.nanquantile(d["opp_dist2high_at_onset"], 0.4))
    q60 = float(np.nanquantile(d["opp_dist2high_at_onset"], 0.6))

    d_close = d[d["opp_dist2high_at_onset"] < q40]
    d_far = d[d["opp_dist2high_at_onset"] > q60]

    bins = np.linspace(-1.2, 1.2, 11)
    m_close = _phigh_map(d_close, bins)
    m_far = _phigh_map(d_far, bins)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    ax1, ax2, ax3 = axes

    norm = mpl.colors.TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)

    im1 = ax1.pcolormesh(bins, bins, m_far.T, cmap="RdBu", norm=norm, shading="auto")
    im2 = ax2.pcolormesh(bins, bins, m_close.T, cmap="RdBu", norm=norm, shading="auto")

    for ax in [ax1, ax2]:
        v = _octagon_vertices(1.0)
        ax.plot(v[:, 0], v[:, 1], color="black", lw=1.2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)

    ax1.set_title(f"{agent_type}: opp far from high")
    ax2.set_title(f"{agent_type}: opp close to high")

    fig.colorbar(im1, ax=ax1, shrink=0.8)
    fig.colorbar(im2, ax=ax2, shrink=0.8)

    x = m_close.flatten()
    y = m_far.flatten()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    ax3.scatter(x, y, color="#7a0177", alpha=0.75)
    lims = [0.0, 1.0]
    ax3.plot(lims, lims, "k--", lw=1)
    ax3.set_xlim(lims)
    ax3.set_ylim(lims)
    ax3.set_xlabel("P(high | opp close)")
    ax3.set_ylabel("P(high | opp far)")
    ax3.set_title(f"{agent_type}: bin-wise far vs close")

    if len(x) >= 3 and pearsonr is not None:
        r, p = pearsonr(x, y)
        ax3.text(0.05, 0.95, f"r={r:.2f}, p={p:.3g}", transform=ax3.transAxes, va="top")

    out = Path(output_dir)
    _save_fig(fig, out, f"plot_agent_sensitivity_{agent_type.lower().replace(' ', '_')}")
    return fig


def _running_curve(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    d = df.copy()
    d = d[np.isfinite(d["choice"].astype(float))].sort_values(["agent_id", "trial_id"])
    rows = []
    for aid, g in d.groupby("agent_id"):
        g = g.copy()
        g["choose_high"] = (g["choice"].astype(float) == 0.0).astype(float)
        g["run"] = g["choose_high"].rolling(window=window, min_periods=window).mean()
        rows.append(g[["agent_id", "trial_id", "phase", "run"]])
    if not rows:
        return pd.DataFrame(columns=["trial_id", "phase", "run"])

    allr = pd.concat(rows, axis=0, ignore_index=True)
    out = (
        allr.groupby(["trial_id", "phase"], as_index=False)["run"]
        .mean()
        .sort_values("trial_id")
    )
    return out


def fit_logistic_regression(df: pd.DataFrame) -> pd.DataFrame:
    """Per-agent-type logistic regression summary for opp_dist2high term."""
    _require_columns(
        df,
        [
            "agent_type",
            "phase",
            "choice",
            "dist2high_at_onset",
            "dist2low_at_onset",
            "opp_dist2high_at_onset",
        ],
    )

    d = df.copy()
    d = d[d["phase"] == "social"].copy()
    d = d[np.isfinite(d["choice"].astype(float))]
    d["choose_high"] = (d["choice"].astype(float) == 0.0).astype(int)

    rows = []
    for atype, g in d.groupby("agent_type"):
        gg = g.copy()
        for c in ["dist2high_at_onset", "dist2low_at_onset", "opp_dist2high_at_onset"]:
            gg[c] = pd.to_numeric(gg[c], errors="coerce")
        gg = gg.dropna(subset=["dist2high_at_onset", "dist2low_at_onset", "opp_dist2high_at_onset", "choose_high"])

        coef = np.nan
        pval = np.nan
        n = len(gg)

        if n >= 20 and smf is not None:
            try:
                model = smf.logit(
                    "choose_high ~ dist2high_at_onset + dist2low_at_onset + opp_dist2high_at_onset + dist2high_at_onset:dist2low_at_onset",
                    data=gg,
                )
                res = model.fit(disp=False)
                coef = float(res.params.get("opp_dist2high_at_onset", np.nan))
                pval = float(res.pvalues.get("opp_dist2high_at_onset", np.nan))
            except Exception:
                pass

        rows.append(
            {
                "agent_type": str(atype),
                "n_rows": int(n),
                "coef_opp_dist2high": coef,
                "pvalue_opp_dist2high": pval,
            }
        )

    return pd.DataFrame(rows)


def run_rl_comparison(df_social: pd.DataFrame, df_blind: pd.DataFrame, output_dir: str | Path) -> Dict[str, Path]:
    """Generate Figure-3 style social-vs-blind comparison analysis outputs."""
    plt.style.use("seaborn-v0_8-whitegrid")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tag source agent type for comparison/regression.
    social_df = df_social.copy()
    social_df["agent_type"] = "social"
    blind_df = df_blind.copy()
    blind_df["agent_type"] = "blind"

    outputs: Dict[str, Path] = {}

    # Plot 12: social sensitivity.
    fig12 = plot_agent_sensitivity(social_df, agent_type="Social", output_dir=out_dir)
    _save_fig(fig12, out_dir, "plot12_social_agent_sensitivity")
    plt.close(fig12)
    outputs["plot12"] = out_dir / "plot12_social_agent_sensitivity.png"

    # Plot 13: socially-blind sensitivity.
    fig13 = plot_agent_sensitivity(blind_df, agent_type="Socially-blind", output_dir=out_dir)
    _save_fig(fig13, out_dir, "plot13_blind_agent_sensitivity")
    plt.close(fig13)
    outputs["plot13"] = out_dir / "plot13_blind_agent_sensitivity.png"

    # Plot 14: context preference-shift comparison.
    fig14, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    axl, axr = axes

    social_curve = _running_curve(social_df, window=50)
    blind_curve = _running_curve(blind_df, window=50)

    # Background shading by dominant phase schedule from social curve.
    sched = social_curve.sort_values("trial_id")
    if not sched.empty:
        sched = sched.groupby("trial_id", as_index=False)["phase"].agg(lambda x: x.iloc[0])
        trials = sched["trial_id"].to_numpy(dtype=float)
        phases = sched["phase"].to_numpy(dtype=str)
        if len(trials) > 1:
            starts = np.r_[trials[0], trials[1:]]
            ends = np.r_[trials[1:], trials[-1] + 1]
            for s, e, ph in zip(starts, ends, phases):
                color = "#bdd7e7" if ph == "solo" else "#fdd0a2"
                axl.axvspan(s, e, color=color, alpha=0.25, lw=0)

    axl.plot(social_curve["trial_id"], social_curve["run"], color="#1f78b4", lw=2, label="Social")
    axl.plot(blind_curve["trial_id"], blind_curve["run"], color="#33a02c", lw=2, label="Socially-blind")
    axl.set_xlabel("Trial number")
    axl.set_ylabel("Running P(high), window=50")
    axl.set_ylim(0.0, 1.0)
    axl.set_title("Context preference shift over training")
    axl.legend(frameon=False)

    # Right panel bars: P(high) in solo vs social by agent type.
    def _mean_ph_by_phase(dfi: pd.DataFrame) -> Tuple[float, float]:
        dd = dfi[np.isfinite(dfi["choice"].astype(float))].copy()
        dd["choose_high"] = (dd["choice"].astype(float) == 0.0).astype(float)
        solo = float(dd[dd["phase"] == "solo"]["choose_high"].mean()) if (dd["phase"] == "solo").any() else np.nan
        social = float(dd[dd["phase"] == "social"]["choose_high"].mean()) if (dd["phase"] == "social").any() else np.nan
        return solo, social

    s_solo, s_social = _mean_ph_by_phase(social_df)
    b_solo, b_social = _mean_ph_by_phase(blind_df)

    x = np.arange(2)
    w = 0.35
    axr.bar(x - w / 2, [s_solo, s_social], width=w, color="#1f78b4", alpha=0.9, label="Social")
    axr.bar(x + w / 2, [b_solo, b_social], width=w, color="#33a02c", alpha=0.9, label="Socially-blind")
    axr.set_xticks(x)
    axr.set_xticklabels(["solo", "social"])
    axr.set_ylim(0.0, 1.0)
    axr.set_ylabel("P(high)")
    axr.set_title("Context-wise preference")
    axr.legend(frameon=False)

    _save_fig(fig14, out_dir, "plot14_context_shift_comparison")
    plt.close(fig14)
    outputs["plot14"] = out_dir / "plot14_context_shift_comparison.png"

    # Plot 15: self-position-specific Delta P(low) for rw settings.
    fig15, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    bins = np.linspace(-1.2, 1.2, 11)

    def _aligned_social_solo(dfi: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # social with inferred losers
        social_aug = augment_social_with_inferred_losers(dfi[dfi["phase"] == "social"].copy())
        # build aligned frames
        def _build(adf: pd.DataFrame) -> pd.DataFrame:
            rows = []
            for _, r in adf.iterrows():
                if not np.isfinite(float(r.get("choice", np.nan))):
                    continue
                hp = float(r.get("high_patch_id", np.nan))
                lp = float(r.get("low_patch_id", np.nan))
                if not np.isfinite(hp) or not np.isfinite(lp):
                    continue
                sxy = np.array([float(r["stimulus_onset_pos_x"]), float(r["stimulus_onset_pos_y"])], dtype=float)
                p = _transform_trial_frame(sxy, int(hp), int(lp))
                rows.append(
                    {
                        "trial_id": float(r["trial_id"]),
                        "x": float(p[0]),
                        "y": float(p[1]),
                        "choose_low": 1.0 if float(r["choice"]) == 1.0 else 0.0,
                    }
                )
            return pd.DataFrame(rows)

        solo_al = _build(dfi[dfi["phase"] == "solo"].copy())
        social_al = _build(social_aug)
        return solo_al, social_al

    s_solo_al, s_social_al = _aligned_social_solo(social_df)
    b_solo_al, b_social_al = _aligned_social_solo(blind_df)

    rws = [0.001, 0.1]
    datasets = [("social", s_solo_al, s_social_al), ("blind", b_solo_al, b_social_al)]

    for ri, rw in enumerate(rws):
        for ci, (name, solo_al, social_al) in enumerate(datasets):
            ax = axes[ri, ci]
            if solo_al.empty or social_al.empty:
                ax.set_title(f"{name}, rw={rw} (insufficient data)")
                ax.axis("off")
                continue

            p_low_solo = _plow_map_weighted(solo_al, bins=bins, rw=rw)
            p_low_social = _plow_map_weighted(social_al, bins=bins, rw=rw)
            delta = p_low_social - p_low_solo

            norm = mpl.colors.TwoSlopeNorm(vmin=-0.25, vcenter=0.0, vmax=0.25)
            mesh = ax.pcolormesh(bins, bins, delta.T, cmap="RdBu_r", norm=norm, shading="auto")
            v = _octagon_vertices(1.0)
            ax.plot(v[:, 0], v[:, 1], color="black", lw=1.2)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_title(f"{name}, rw={rw}")

            fig15.colorbar(mesh, ax=ax, shrink=0.8)

    fig15.suptitle("Self-position-specific Delta P(low): social - solo")
    _save_fig(fig15, out_dir, "plot15_self_position_shift_recency")
    plt.close(fig15)
    outputs["plot15"] = out_dir / "plot15_self_position_shift_recency.png"

    # Statistical comparison table.
    reg_df = pd.concat([social_df, blind_df], axis=0, ignore_index=True)
    reg_summary = fit_logistic_regression(reg_df)
    reg_summary.to_csv(out_dir / "rl_comparison_logistic_summary.csv", index=False)
    outputs["logistic_summary"] = out_dir / "rl_comparison_logistic_summary.csv"

    return outputs
