from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

try:
    from scipy.stats import pearsonr, ttest_1samp
except Exception:  # pragma: no cover
    pearsonr = None
    ttest_1samp = None


def _require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _solo_df(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, ["phase", "agent_id", "trial_id", "choice"])
    out = df[df["phase"] == "solo"].copy()
    out = out.sort_values(["agent_id", "trial_id"]).reset_index(drop=True)
    return out


def _choose_high_indicator(series: pd.Series) -> pd.Series:
    # choice: 0=high, 1=low
    return (series.astype(float) == 0.0).astype(float)


def _rolling_phigh_by_agent(df_solo: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    rows = []
    for agent_id, g in df_solo.groupby("agent_id"):
        g = g.sort_values("trial_id").copy()
        g["choose_high"] = _choose_high_indicator(g["choice"])
        g["trial_idx"] = np.arange(1, len(g) + 1)
        g["p_high_roll"] = g["choose_high"].rolling(window=window, min_periods=window).mean()
        rows.append(g[["agent_id", "trial_id", "trial_idx", "p_high_roll"]])
    return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()


def _mean_sem(values_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values_2d, axis=0)
    n = np.sum(np.isfinite(values_2d), axis=0)
    sd = np.nanstd(values_2d, axis=0, ddof=1)
    sem = np.where(n > 1, sd / np.sqrt(n), np.nan)
    return mean, sem


def _post_learning_mask(df_solo: pd.DataFrame, threshold: float = 0.7, window: int = 50) -> pd.Series:
    mask = np.zeros(len(df_solo), dtype=bool)
    for _, idx in df_solo.groupby("agent_id").groups.items():
        g = df_solo.loc[idx].sort_values("trial_id")
        choose_high = _choose_high_indicator(g["choice"]).to_numpy(dtype=float)
        roll = pd.Series(choose_high).rolling(window=window, min_periods=window).mean().to_numpy()
        reached = np.where(roll >= threshold)[0]
        if reached.size == 0:
            continue
        first = int(reached[0])
        post_idx = g.index.to_numpy()[first:]
        mask[np.isin(df_solo.index.to_numpy(), post_idx)] = True
    return pd.Series(mask, index=df_solo.index)


def _octagon_vertices(inradius: float = 1.0) -> np.ndarray:
    circum = float(inradius) / np.cos(np.pi / 8.0)
    angles = np.deg2rad(np.arange(22.5, 360.0 + 22.5, 45.0))
    return np.stack([circum * np.cos(angles), circum * np.sin(angles)], axis=1)


def _rotate_points(xy: np.ndarray, angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    r = np.array([[c, -s], [s, c]], dtype=float)
    return xy @ r.T


def _transform_trial_frame(start_xy: np.ndarray, high_patch_id: int, low_patch_id: int) -> Tuple[np.ndarray, int]:
    # Rotate so high patch moves to north (90 deg).
    high_angle = np.deg2rad(float(high_patch_id) * 45.0)
    rot = np.deg2rad(90.0) - high_angle
    p = _rotate_points(start_xy[None, :], rot)[0]

    # Reflect across y-axis so low is to the right.
    low_angle = np.deg2rad(float(low_patch_id) * 45.0)
    low_rot = low_angle + rot
    low_xy = np.array([np.cos(low_rot), np.sin(low_rot)], dtype=float)
    if low_xy[0] < 0.0:
        p[0] *= -1.0

    sep_steps = min(abs(int(high_patch_id) - int(low_patch_id)), 8 - abs(int(high_patch_id) - int(low_patch_id)))
    return p, int(sep_steps)


def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = float(k) / float(n)
    den = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / den
    half = (z / den) * np.sqrt((p * (1.0 - p) / n) + (z**2 / (4.0 * n * n)))
    return center - half, center + half


def _trial_category(df: pd.DataFrame) -> pd.Series:
    # Preferred explicit columns if present.
    if "trial_type" in df.columns:
        cat = df["trial_type"].astype(str).copy()
        if "choice_role" in df.columns:
            forced_mask = cat != "choice"
            cat.loc[forced_mask & (df["choice_role"].astype(str) == "high")] = "highx2"
            cat.loc[forced_mask & (df["choice_role"].astype(str) == "low")] = "lowx2"
        return cat

    if "phase_label" in df.columns:
        c = df["phase_label"].astype(str).copy()
        c = c.replace({"forced_highx2": "highx2", "forced_lowx2": "lowx2", "choice": "choice"})
        return c

    # Fallback: keep all as choice if forced classes are not available.
    cat = pd.Series(["choice"] * len(df), index=df.index)
    if "forced_type" in df.columns:
        forced = df["forced_type"].astype(str)
        cat.loc[forced == "highx2"] = "highx2"
        cat.loc[forced == "lowx2"] = "lowx2"
    return cat


def _save_fig(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", dpi=300, bbox_inches="tight")


def plot_learning_curves(df: pd.DataFrame) -> plt.Figure:
    plt.style.use("seaborn-v0_8-whitegrid")
    df_solo = _solo_df(df)
    roll = _rolling_phigh_by_agent(df_solo, window=50)

    if roll.empty:
        raise ValueError("No solo trials available for learning curve plot")

    max_len = int(roll["trial_idx"].max())
    agents = sorted(roll["agent_id"].unique())
    mat = np.full((len(agents), max_len), np.nan, dtype=float)
    for ai, aid in enumerate(agents):
        g = roll[roll["agent_id"] == aid]
        idx = g["trial_idx"].to_numpy(dtype=int) - 1
        mat[ai, idx] = g["p_high_roll"].to_numpy(dtype=float)

    mean, sem = _mean_sem(mat)
    x = np.arange(1, max_len + 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x, mean, color="tab:blue", lw=2)
    ax.fill_between(x, mean - sem, mean + sem, color="tab:blue", alpha=0.25)
    ax.axhline(0.5, color="black", ls="--", lw=1)
    ax.axhline(0.7, color="gray", ls="--", lw=1)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("Trial number")
    ax.set_ylabel("P(choose high)")
    ax.set_title("Solo learning curves")
    return fig


def plot_location_heatmap(df: pd.DataFrame) -> plt.Figure:
    plt.style.use("seaborn-v0_8-whitegrid")
    df_solo = _solo_df(df)
    _require_columns(df_solo, ["high_patch_id"])

    post_mask = _post_learning_mask(df_solo, threshold=0.7, window=50)
    d = df_solo[post_mask].copy()
    if d.empty:
        d = df_solo.copy()

    d = d[np.isfinite(d["high_patch_id"].astype(float))]
    d["high_patch_id"] = d["high_patch_id"].astype(int)
    d["choose_high"] = _choose_high_indicator(d["choice"])

    ph = np.full(8, np.nan, dtype=float)
    for pid in range(8):
        g = d[d["high_patch_id"] == pid]
        if len(g) > 0:
            ph[pid] = float(np.nanmean(g["choose_high"]))

    angles = np.deg2rad(np.arange(0, 360, 45, dtype=float))
    coords = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    verts = _octagon_vertices(1.0)
    ax.plot(verts[:, 0], verts[:, 1], color="black", lw=2)

    norm = mpl.colors.TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    cmap = plt.get_cmap("RdBu")

    for i, (x, y) in enumerate(coords):
        val = ph[i]
        color = cmap(norm(val)) if np.isfinite(val) else (0.8, 0.8, 0.8, 1.0)
        ax.scatter([x], [y], s=450, marker="s", c=[color], edgecolors="k", linewidths=1.0)
        label = f"{val:.2f}" if np.isfinite(val) else "NA"
        ax.text(x, y, label, ha="center", va="center", fontsize=10)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("P(choose high)")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_title("P(high) across patch locations")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig


def plot_spatial_phigh(df: pd.DataFrame) -> plt.Figure:
    plt.style.use("seaborn-v0_8-whitegrid")
    df_solo = _solo_df(df)
    _require_columns(
        df_solo,
        [
            "stimulus_onset_pos_x",
            "stimulus_onset_pos_y",
            "high_patch_id",
            "low_patch_id",
            "choice",
        ],
    )

    valid = df_solo.copy()
    valid = valid[np.isfinite(valid["high_patch_id"].astype(float))]
    valid = valid[np.isfinite(valid["low_patch_id"].astype(float))]

    transformed = []
    for _, r in valid.iterrows():
        start = np.array([float(r["stimulus_onset_pos_x"]), float(r["stimulus_onset_pos_y"])], dtype=float)
        high_id = int(r["high_patch_id"])
        low_id = int(r["low_patch_id"])
        p, sep_steps = _transform_trial_frame(start, high_id, low_id)
        if sep_steps not in {1, 2}:
            continue
        transformed.append(
            {
                "x": float(p[0]),
                "y": float(p[1]),
                "sep_steps": int(sep_steps),
                "choose_high": 1.0 if float(r["choice"]) == 0.0 else 0.0,
            }
        )

    td = pd.DataFrame(transformed)
    if td.empty:
        raise ValueError("Insufficient data for spatial P(high) plot")

    bins = np.linspace(-1.2, 1.2, 11)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    seps = [1, 2]
    titles = {1: "45° separation", 2: "90° separation"}

    norm = mpl.colors.TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    cmap = plt.get_cmap("RdBu")

    for ax, sep in zip(axes, seps):
        d = td[td["sep_steps"] == sep]

        counts, _, _ = np.histogram2d(d["x"], d["y"], bins=[bins, bins])
        highs, _, _ = np.histogram2d(d["x"], d["y"], bins=[bins, bins], weights=d["choose_high"]) 
        ph = np.divide(highs, counts, out=np.full_like(highs, np.nan, dtype=float), where=counts > 0)

        mesh = ax.pcolormesh(bins, bins, ph.T, cmap=cmap, norm=norm, shading="auto")

        verts = _octagon_vertices(1.0)
        ax.plot(verts[:, 0], verts[:, 1], color="black", lw=1.5)

        high_xy = np.array([0.0, 1.0], dtype=float)
        low_xy = np.array([np.cos(np.deg2rad(45.0 * (2 - sep))), np.sin(np.deg2rad(45.0 * (2 - sep)))], dtype=float)
        # sep=1 -> low at 45 deg, sep=2 -> low at 0 deg.
        ax.arrow(0.0, 0.0, high_xy[0], high_xy[1], color="blue", width=0.01, length_includes_head=True)
        ax.arrow(0.0, 0.0, low_xy[0], low_xy[1], color="red", width=0.01, length_includes_head=True)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(titles[sep])
        ax.set_xlabel("Aligned x")
        ax.set_ylabel("Aligned y")

    cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("P(choose high)")
    fig.suptitle("Spatial heatmap of P(high) by start position")
    return fig


def plot_distance_phigh(df: pd.DataFrame) -> plt.Figure:
    plt.style.use("seaborn-v0_8-whitegrid")
    d = _solo_df(df)
    _require_columns(d, ["dist2high_at_onset", "dist2low_at_onset", "patch_separation_deg", "choice"])

    d = d[np.isfinite(d["dist2high_at_onset"]) & np.isfinite(d["dist2low_at_onset"])].copy()
    d = d[d["patch_separation_deg"].isin([45.0, 90.0])]
    d["choose_high"] = _choose_high_indicator(d["choice"]).astype(int)

    if d.empty:
        raise ValueError("Insufficient data for distance-based P(high) plot")

    d["high_bin"] = pd.cut(d["dist2high_at_onset"], bins=5, labels=False, include_lowest=True)
    d["low_bin"] = pd.cut(d["dist2low_at_onset"], bins=4, labels=False, include_lowest=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    palette = sns.color_palette("RdPu", n_colors=5)[1:]

    for ax, sep in zip(axes, [45.0, 90.0]):
        ds = d[d["patch_separation_deg"] == sep]
        for low_bin in sorted([x for x in ds["low_bin"].dropna().unique()]):
            low_bin = int(low_bin)
            xs = []
            ys = []
            yerr_lo = []
            yerr_hi = []
            for hb in range(5):
                g = ds[(ds["high_bin"] == hb) & (ds["low_bin"] == low_bin)]
                n = len(g)
                if n == 0:
                    continue
                k = int(g["choose_high"].sum())
                p = float(k / n)
                ci_l, ci_u = _wilson_ci(k, n, z=1.96)
                xs.append(hb + 1)
                ys.append(p)
                yerr_lo.append(p - ci_l)
                yerr_hi.append(ci_u - p)

            if len(xs) == 0:
                continue

            ax.errorbar(
                xs,
                ys,
                yerr=np.vstack([yerr_lo, yerr_hi]),
                marker="o",
                color=palette[min(low_bin, len(palette) - 1)],
                lw=1.8,
                capsize=3,
                label=f"dist2low bin {low_bin + 1}",
            )

        ax.set_title(f"{int(sep)}° separation")
        ax.set_xlabel("Distance-to-high bin (near to far)")
        ax.set_ylabel("P(choose high)")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.legend(frameon=False)

    fig.suptitle("P(high) as function of distances")
    return fig


def plot_rt_analysis(df: pd.DataFrame) -> plt.Figure:
    plt.style.use("seaborn-v0_8-whitegrid")
    d = _solo_df(df)
    _require_columns(d, ["agent_id", "RT", "choice", "trial_id"])

    d = d[np.isfinite(d["RT"])].copy()
    if d.empty:
        raise ValueError("No finite RT values available")

    d["trial_category"] = _trial_category(d)
    d = d[d["trial_category"].isin(["choice", "highx2", "lowx2"])].copy()
    if d.empty:
        raise ValueError(
            "RT analysis requires trial categories {'choice','highx2','lowx2'} in columns trial_type/choice_role or phase_label"
        )

    # Median RT per agent per category.
    med = (
        d.groupby(["agent_id", "trial_category"], as_index=False)["RT"]
        .median()
        .rename(columns={"RT": "median_RT"})
    )

    piv = med.pivot(index="agent_id", columns="trial_category", values="median_RT")
    if "highx2" not in piv.columns:
        # Fallback baseline if highx2 unavailable.
        piv["highx2"] = piv.get("choice", np.nan)

    eps = 1e-8
    log_choice = np.log((piv.get("choice", np.nan) + eps) / (piv["highx2"] + eps))
    log_high = np.log((piv["highx2"] + eps) / (piv["highx2"] + eps))
    log_low = np.log((piv.get("lowx2", np.nan) + eps) / (piv["highx2"] + eps))

    vals = {
        "choice/highx2": log_choice.dropna().to_numpy(dtype=float),
        "highx2/highx2": log_high.dropna().to_numpy(dtype=float),
        "lowx2/highx2": log_low.dropna().to_numpy(dtype=float),
    }

    means = [float(np.nanmean(vals[k])) if len(vals[k]) > 0 else np.nan for k in vals]
    sems = [float(np.nanstd(vals[k], ddof=1) / np.sqrt(len(vals[k]))) if len(vals[k]) > 1 else np.nan for k in vals]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    ax = axes[0]
    labels = list(vals.keys())
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=sems, color=["#4c72b0", "#999999", "#c44e52"], alpha=0.85, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("log(RT_type / RT_highx2)")
    ax.set_title("Forced-trial RT analysis")

    # Significance marks against zero.
    for i, k in enumerate(labels):
        arr = vals[k]
        if len(arr) < 2:
            continue
        if ttest_1samp is not None:
            _, p = ttest_1samp(arr, popmean=0.0, nan_policy="omit")
            if p < 0.01:
                y = means[i] + (sems[i] if np.isfinite(sems[i]) else 0.02) + 0.02
                ax.text(i, y, "**", ha="center", va="bottom", fontsize=14)

    # Scatter: P(high) per session vs log RT ratio lowx2/highx2.
    agent_phigh = d.groupby("agent_id")["choice"].apply(lambda s: np.mean((s.astype(float) == 0.0).astype(float)))
    ratio = log_low
    common_agents = agent_phigh.index.intersection(ratio.dropna().index)
    xs = agent_phigh.loc[common_agents].to_numpy(dtype=float)
    ys = ratio.loc[common_agents].to_numpy(dtype=float)

    ax2 = axes[1]
    ax2.scatter(xs, ys, c="#7a0177", alpha=0.8)
    ax2.set_xlabel("P(high) per session")
    ax2.set_ylabel("log RT ratio (lowx2/highx2)")
    ax2.set_title("Behavior-performance relationship")

    if len(xs) >= 3 and pearsonr is not None:
        r, p = pearsonr(xs, ys)
        ax2.text(0.05, 0.95, f"r={r:.2f}, p={p:.3g}", transform=ax2.transAxes, va="top")

    return fig


def run_solo_analysis(df: pd.DataFrame, output_dir: str | Path) -> Dict[str, Path]:
    plt.style.use("seaborn-v0_8-whitegrid")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, Path] = {}

    fig1 = plot_learning_curves(df)
    _save_fig(fig1, out_dir, "plot1_learning_curves")
    plt.close(fig1)
    outputs["plot1"] = out_dir / "plot1_learning_curves.png"

    fig2 = plot_location_heatmap(df)
    _save_fig(fig2, out_dir, "plot2_location_heatmap")
    plt.close(fig2)
    outputs["plot2"] = out_dir / "plot2_location_heatmap.png"

    fig3 = plot_spatial_phigh(df)
    _save_fig(fig3, out_dir, "plot3_spatial_phigh")
    plt.close(fig3)
    outputs["plot3"] = out_dir / "plot3_spatial_phigh.png"

    fig4 = plot_distance_phigh(df)
    _save_fig(fig4, out_dir, "plot4_distance_phigh")
    plt.close(fig4)
    outputs["plot4"] = out_dir / "plot4_distance_phigh.png"

    fig5 = plot_rt_analysis(df)
    _save_fig(fig5, out_dir, "plot5_rt_analysis")
    plt.close(fig5)
    outputs["plot5"] = out_dir / "plot5_rt_analysis.png"

    return outputs
