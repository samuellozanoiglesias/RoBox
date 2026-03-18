from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

try:
    from scipy.stats import pearsonr, ttest_rel
except Exception:  # pragma: no cover
    pearsonr = None
    ttest_rel = None

try:
    import statsmodels.formula.api as smf
except Exception:  # pragma: no cover
    smf = None

try:
    import pingouin as pg
except Exception:  # pragma: no cover
    pg = None

from .navigation import clip_to_octagon


def _require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _save_fig(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", dpi=300, bbox_inches="tight")


def _choose_high_indicator(choice: pd.Series) -> pd.Series:
    return (choice.astype(float) == 0.0).astype(float)


def _choose_low_indicator(choice: pd.Series) -> pd.Series:
    return (choice.astype(float) == 1.0).astype(float)


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
    high_angle = np.deg2rad(float(high_patch_id) * 45.0)
    rot = np.deg2rad(90.0) - high_angle
    p = _rotate_points(start_xy[None, :], rot)[0]

    low_angle = np.deg2rad(float(low_patch_id) * 45.0)
    low_rot = low_angle + rot
    low_xy = np.array([np.cos(low_rot), np.sin(low_rot)], dtype=float)
    if low_xy[0] < 0.0:
        p[0] *= -1.0

    sep_steps = min(abs(int(high_patch_id) - int(low_patch_id)), 8 - abs(int(high_patch_id) - int(low_patch_id)))
    return p, int(sep_steps)


def _patch_xy(patch_id: int, inradius: float = 1.0) -> np.ndarray:
    th = np.deg2rad(float(patch_id) * 45.0)
    return np.array([np.cos(th), np.sin(th)], dtype=float) * float(inradius)


def _infer_loser_choice_for_trial(
    trial_df: pd.DataFrame,
    max_speed: float = 0.5,
    infer_time_offset: float = 1.0,
    include_threshold: float = 0.12,
    inradius: float = 1.0,
) -> pd.DataFrame:
    if len(trial_df) != 2:
        return trial_df.copy()

    d = trial_df.copy()

    # Winner row uses explicit choice.
    winner_mask = d["winner_id"].notna() & (d["winner_id"].astype(float) == d["agent_id"].astype(float))
    if winner_mask.sum() != 1:
        return d

    winner_row = d[winner_mask].iloc[0]
    loser_idx = d.index[~winner_mask]
    if len(loser_idx) != 1:
        return d

    li = loser_idx[0]
    loser = d.loc[li].copy()

    if np.isfinite(float(loser.get("choice", np.nan))):
        # Already has a choice, keep as-is.
        return d

    high_patch = int(loser["high_patch_id"])
    low_patch = int(loser["low_patch_id"])
    winner_rt = float(winner_row.get("RT", np.nan))
    if not np.isfinite(winner_rt):
        return d

    t = max(0.0, winner_rt + float(infer_time_offset))

    start = np.array([float(loser["stimulus_onset_pos_x"]), float(loser["stimulus_onset_pos_y"])], dtype=float)
    high_xy = _patch_xy(high_patch, inradius=inradius)
    low_xy = _patch_xy(low_patch, inradius=inradius)

    # Infer heading by nearest active patch from onset.
    d_high0 = float(np.linalg.norm(start - high_xy))
    d_low0 = float(np.linalg.norm(start - low_xy))
    heading_target = high_xy if d_high0 <= d_low0 else low_xy

    vec = heading_target - start
    norm = float(np.linalg.norm(vec))
    if norm > 1e-12:
        direction = vec / norm
    else:
        direction = np.zeros(2, dtype=float)

    pos_t = start + direction * float(max_speed) * t
    pos_t = clip_to_octagon(pos_t, inradius=float(inradius)).astype(float)

    dh = float(np.linalg.norm(pos_t - high_xy))
    dl = float(np.linalg.norm(pos_t - low_xy))

    nearest = min(dh, dl)
    if nearest >= float(include_threshold):
        return d

    inferred_choice = 0.0 if dh <= dl else 1.0
    d.loc[li, "choice"] = inferred_choice
    return d


def augment_social_with_inferred_losers(
    df: pd.DataFrame,
    max_speed: float = 0.5,
    infer_time_offset: float = 1.0,
    include_threshold: float = 0.12,
    inradius: float = 1.0,
) -> pd.DataFrame:
    _require_columns(
        df,
        [
            "phase",
            "trial_id",
            "agent_id",
            "winner_id",
            "RT",
            "choice",
            "high_patch_id",
            "low_patch_id",
            "stimulus_onset_pos_x",
            "stimulus_onset_pos_y",
        ],
    )

    social = df[df["phase"] == "social"].copy()
    if social.empty:
        return social

    out = []
    for _, g in social.groupby("trial_id"):
        out.append(
            _infer_loser_choice_for_trial(
                g,
                max_speed=max_speed,
                infer_time_offset=infer_time_offset,
                include_threshold=include_threshold,
                inradius=inradius,
            )
        )

    return pd.concat(out, axis=0, ignore_index=True)


def _glmm_equivalent(df: pd.DataFrame, formula: str, group_col: str = "agent_id") -> Optional[Tuple[float, float]]:
    # Returns approximate (effect, p-value) for primary condition term.
    if smf is not None:
        try:
            model = smf.mixedlm(formula, df, groups=df[group_col])
            res = model.fit(reml=False, method="lbfgs", disp=False)
            # Use first non-intercept term.
            terms = [t for t in res.params.index if t != "Intercept"]
            if terms:
                term = terms[0]
                return float(res.params[term]), float(res.pvalues[term])
        except Exception:
            pass

    if pg is not None:
        try:
            # Approximate using repeated-measures ANOVA on aggregated means.
            # Requires columns: y, cond, subject
            # This fallback only handles simple two-level contrasts externally.
            return None
        except Exception:
            return None

    return None


def _build_aligned(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        if not np.isfinite(float(r.get("choice", np.nan))):
            continue
        high_patch_id = float(r.get("high_patch_id", np.nan))
        low_patch_id = float(r.get("low_patch_id", np.nan))
        if not np.isfinite(high_patch_id) or not np.isfinite(low_patch_id):
            continue

        start = np.array([float(r["stimulus_onset_pos_x"]), float(r["stimulus_onset_pos_y"])], dtype=float)
        p, sep_steps = _transform_trial_frame(start, int(high_patch_id), int(low_patch_id))
        if sep_steps not in {1, 2}:
            continue

        rows.append(
            {
                "trial_id": int(r["trial_id"]),
                "agent_id": int(r["agent_id"]),
                "pair_id": r.get("pair_id", 0),
                "phase": r["phase"],
                "block_id": r.get("block_id", np.nan),
                "x": float(p[0]),
                "y": float(p[1]),
                "sep_steps": int(sep_steps),
                "choose_high": 1.0 if float(r["choice"]) == 0.0 else 0.0,
                "choose_low": 1.0 if float(r["choice"]) == 1.0 else 0.0,
                "dist2high_at_onset": float(r.get("dist2high_at_onset", np.nan)),
                "dist2low_at_onset": float(r.get("dist2low_at_onset", np.nan)),
                "opp_dist2high_at_onset": float(r.get("opp_dist2high_at_onset", np.nan)),
                "RT": float(r.get("RT", np.nan)),
            }
        )

    return pd.DataFrame(rows)


def plot_preference_shift(df: pd.DataFrame) -> plt.Figure:
    plt.style.use("seaborn-v0_8-whitegrid")
    _require_columns(df, ["phase", "agent_id", "choice", "trial_id"])

    d = df.copy()
    d = d[np.isfinite(d["choice"].astype(float))]
    d["choose_high"] = _choose_high_indicator(d["choice"])

    # Construct solo_pre and solo_post around social exposure.
    rows = []
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

        rows.append({"agent_id": aid, "session": "solo_pre", "p_high": float(solo_pre["choose_high"].mean())})
        rows.append({"agent_id": aid, "session": "social", "p_high": float(social["choose_high"].mean())})
        rows.append({"agent_id": aid, "session": "solo_post", "p_high": float(solo_post["choose_high"].mean())})

    s = pd.DataFrame(rows)
    if s.empty:
        raise ValueError("Insufficient data for preference-shift plot")

    order = ["solo_pre", "social", "solo_post"]
    xmap = {k: i for i, k in enumerate(order)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for aid, g in s.groupby("agent_id"):
        g = g.set_index("session").reindex(order).reset_index()
        ax.plot([xmap[x] for x in g["session"]], g["p_high"], color="gray", alpha=0.4, lw=1)
        ax.scatter([xmap[x] for x in g["session"]], g["p_high"], color="gray", alpha=0.6, s=15)

    mean = s.groupby("session")["p_high"].mean().reindex(order)
    sem = s.groupby("session")["p_high"].sem().reindex(order)
    xs = np.array([xmap[k] for k in order], dtype=float)
    ax.plot(xs, mean.to_numpy(), color="black", lw=3)
    ax.errorbar(xs, mean.to_numpy(), yerr=sem.to_numpy(), color="black", lw=2, capsize=5)

    ax.set_xticks(xs)
    ax.set_xticklabels(order)
    ax.set_ylabel("P(choose high)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Session-averaged preference shift")

    piv = s.pivot(index="agent_id", columns="session", values="p_high")
    p1 = np.nan
    p2 = np.nan
    if ttest_rel is not None:
        try:
            p1 = float(ttest_rel(piv["solo_pre"], piv["social"], nan_policy="omit").pvalue)
            p2 = float(ttest_rel(piv["social"], piv["solo_post"], nan_policy="omit").pvalue)
        except Exception:
            pass

    y_top = float(np.nanmax(s["p_high"]) + 0.08)
    ax.plot([0, 1], [y_top, y_top], color="black", lw=1)
    ax.text(0.5, y_top + 0.01, "***" if np.isfinite(p1) and p1 < 0.001 else "NS", ha="center")
    ax.plot([1, 2], [y_top - 0.05, y_top - 0.05], color="black", lw=1)
    ax.text(1.5, y_top - 0.04, "NS" if (not np.isfinite(p2) or p2 >= 0.05) else "***", ha="center")

    glmm = _glmm_equivalent(
        s.replace({"session": {"solo_pre": 0, "social": 1, "solo_post": 2}}),
        formula="p_high ~ session",
        group_col="agent_id",
    )
    if glmm is not None:
        eff, p = glmm
        ax.text(0.02, 0.02, f"GLMM approx: beta={eff:.3f}, p={p:.3g}", transform=ax.transAxes)

    return fig


def plot_social_spatial(df: pd.DataFrame) -> plt.Figure:
    plt.style.use("seaborn-v0_8-whitegrid")
    social_aug = augment_social_with_inferred_losers(df)
    aligned_social = _build_aligned(social_aug)
    aligned_solo = _build_aligned(df[df["phase"] == "solo"].copy())

    if aligned_social.empty:
        raise ValueError("No social data (with inferred loser choices) available")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # Panel A: P(high) vs dist2high, solo dashed vs social solid, color by dist2low bins.
    ax = axes[0]

    def _prep_lines(adf: pd.DataFrame) -> pd.DataFrame:
        d = adf.copy()
        d = d[np.isfinite(d["dist2high_at_onset"]) & np.isfinite(d["dist2low_at_onset"])].copy()
        d["high_bin"] = pd.cut(d["dist2high_at_onset"], bins=5, labels=False, include_lowest=True)
        d["low_bin"] = pd.cut(d["dist2low_at_onset"], bins=4, labels=False, include_lowest=True)
        return d

    ds_solo = _prep_lines(aligned_solo)
    ds_social = _prep_lines(aligned_social)
    palette = sns.color_palette("RdPu", n_colors=5)[1:]

    for low_bin in range(4):
        for label, dset, ls in [("solo", ds_solo, "--"), ("social", ds_social, "-")]:
            xs = []
            ys = []
            for hb in range(5):
                g = dset[(dset["low_bin"] == low_bin) & (dset["high_bin"] == hb)]
                if len(g) == 0:
                    continue
                xs.append(hb + 1)
                ys.append(float(g["choose_high"].mean()))
            if len(xs) > 0:
                ax.plot(
                    xs,
                    ys,
                    ls=ls,
                    lw=2,
                    color=palette[min(low_bin, len(palette) - 1)],
                    label=f"{label}, low-bin {low_bin + 1}" if low_bin == 0 else None,
                )

    ax.set_xlabel("Distance-to-high bin (close to far)")
    ax.set_ylabel("P(choose high)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("P(high) by self-position: solo vs social")

    # Panel B: social spatial heatmap.
    ax2 = axes[1]
    bins = np.linspace(-1.2, 1.2, 11)
    counts, _, _ = np.histogram2d(aligned_social["x"], aligned_social["y"], bins=[bins, bins])
    highs, _, _ = np.histogram2d(
        aligned_social["x"], aligned_social["y"], bins=[bins, bins], weights=aligned_social["choose_high"]
    )
    ph = np.divide(highs, counts, out=np.full_like(highs, np.nan, dtype=float), where=counts > 0)

    norm = mpl.colors.TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    mesh = ax2.pcolormesh(bins, bins, ph.T, cmap="RdBu", norm=norm, shading="auto")
    verts = _octagon_vertices(1.0)
    ax2.plot(verts[:, 0], verts[:, 1], color="black", lw=1.5)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_title("Social P(high) spatial heatmap")
    cbar = fig.colorbar(mesh, ax=ax2, shrink=0.9)
    cbar.set_label("P(choose high)")

    # GLMM-equivalent annotation on panel A (social vs solo, controlling dist bins).
    joint = pd.concat(
        [
            aligned_solo.assign(context=0),
            aligned_social.assign(context=1),
        ],
        axis=0,
        ignore_index=True,
    )
    joint = joint[np.isfinite(joint["dist2high_at_onset"]) & np.isfinite(joint["dist2low_at_onset"])].copy()
    glmm = _glmm_equivalent(joint, formula="choose_high ~ context + dist2high_at_onset + dist2low_at_onset")
    if glmm is not None:
        eff, p = glmm
        ax.text(0.02, 0.02, f"GLMM approx: beta={eff:.3f}, p={p:.3g}", transform=ax.transAxes)

    return fig


def plot_delta_plow(df: pd.DataFrame) -> plt.Figure:
    plt.style.use("seaborn-v0_8-whitegrid")

    solo = _build_aligned(df[df["phase"] == "solo"].copy())
    social = _build_aligned(augment_social_with_inferred_losers(df))
    if solo.empty or social.empty:
        raise ValueError("Need both solo and social data for delta P(low) heatmap")

    bins = np.linspace(-1.2, 1.2, 11)

    def _plow_map(adf: pd.DataFrame) -> np.ndarray:
        c, _, _ = np.histogram2d(adf["x"], adf["y"], bins=[bins, bins])
        l, _, _ = np.histogram2d(adf["x"], adf["y"], bins=[bins, bins], weights=adf["choose_low"])
        return np.divide(l, c, out=np.full_like(l, np.nan, dtype=float), where=c > 0)

    p_low_solo = _plow_map(solo)
    p_low_social = _plow_map(social)
    delta = p_low_social - p_low_solo

    fig, ax = plt.subplots(figsize=(10, 8))
    norm = mpl.colors.TwoSlopeNorm(vmin=-0.25, vcenter=0.0, vmax=0.25)
    mesh = ax.pcolormesh(bins, bins, delta.T, cmap="RdBu_r", norm=norm, shading="auto")
    verts = _octagon_vertices(1.0)
    ax.plot(verts[:, 0], verts[:, 1], color="black", lw=1.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_title("Delta P(low): social minus solo")
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.85)
    cbar.set_label("Delta P(low)")

    return fig


def plot_opponent_speed_corr(df: pd.DataFrame) -> plt.Figure:
    plt.style.use("seaborn-v0_8-whitegrid")
    _require_columns(df, ["phase", "agent_id", "RT", "choice", "block_id"])

    d = df.copy()
    d = d[np.isfinite(d["choice"].astype(float))]
    d["choose_low"] = _choose_low_indicator(d["choice"])
    d["pair_id"] = d.get("pair_id", 0)

    solo = d[d["phase"] == "solo"].copy()
    social = augment_social_with_inferred_losers(d)
    social["choose_low"] = _choose_low_indicator(social["choice"])
    social["pair_id"] = social.get("pair_id", 0)

    rows = []
    for (pid, aid), gs in social.groupby(["pair_id", "agent_id"]):
        opp_candidates = social[(social["pair_id"] == pid) & (social["agent_id"] != aid)]["agent_id"].unique()
        if len(opp_candidates) == 0:
            continue
        opp_id = opp_candidates[0]

        opp_solo = solo[(solo["pair_id"] == pid) & (solo["agent_id"] == opp_id)]
        if len(opp_solo) == 0:
            continue
        x = float(np.nanmean(opp_solo["RT"]))

        self_solo = solo[(solo["pair_id"] == pid) & (solo["agent_id"] == aid)]
        if len(self_solo) == 0:
            continue

        # Session-level deltas over shared block_id where possible.
        by_block = []
        for bid, gss in gs.groupby("block_id"):
            if pd.isna(bid):
                continue
            gsolo = self_solo[self_solo["block_id"] == bid]
            if len(gsolo) == 0:
                continue
            p_low_social = float(np.mean(gss["choose_low"]))
            p_low_solo = float(np.mean(gsolo["choose_low"]))
            by_block.append(p_low_social - p_low_solo)

        if len(by_block) == 0:
            y = float(np.mean(gs["choose_low"]) - np.mean(self_solo["choose_low"]))
            y_sem = np.nan
        else:
            y = float(np.mean(by_block))
            y_sem = float(np.std(by_block, ddof=1) / np.sqrt(len(by_block))) if len(by_block) > 1 else 0.0

        rows.append({"pair_id": pid, "agent_id": aid, "x": x, "y": y, "y_sem": y_sem})

    corr = pd.DataFrame(rows)
    if corr.empty:
        raise ValueError("Insufficient data for opponent-speed correlation")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.errorbar(corr["x"], corr["y"], yerr=corr["y_sem"], fmt="o", color="#4c72b0", alpha=0.85, capsize=3)

    # Linear regression line.
    if len(corr) >= 2:
        coefs = np.polyfit(corr["x"], corr["y"], deg=1)
        xs = np.linspace(float(np.min(corr["x"])), float(np.max(corr["x"])), 100)
        ys = np.polyval(coefs, xs)
        ax.plot(xs, ys, color="black", lw=2)

    ax.set_xlabel("Opponent mean RT in solo (1/speed proxy)")
    ax.set_ylabel("Delta P(low): social - solo")
    ax.set_title("Preference shift vs opponent speed")

    if len(corr) >= 3 and pearsonr is not None:
        r, p = pearsonr(corr["x"], corr["y"])
        ax.text(0.02, 0.98, f"r={r:.2f}, p={p:.3g}", transform=ax.transAxes, va="top")

    return fig


def plot_opponent_position(df: pd.DataFrame) -> plt.Figure:
    plt.style.use("seaborn-v0_8-whitegrid")

    social = augment_social_with_inferred_losers(df)
    aligned = _build_aligned(social)
    if aligned.empty:
        raise ValueError("No social data available for opponent-position modulation")

    aligned = aligned[np.isfinite(aligned["opp_dist2high_at_onset"])].copy()
    if aligned.empty:
        raise ValueError("No opp_dist2high_at_onset values for opponent-position modulation")

    q40 = float(np.nanquantile(aligned["opp_dist2high_at_onset"], 0.4))
    q60 = float(np.nanquantile(aligned["opp_dist2high_at_onset"], 0.6))

    d_close = aligned[aligned["opp_dist2high_at_onset"] < q40]
    d_far = aligned[aligned["opp_dist2high_at_onset"] > q60]

    bins = np.linspace(-1.2, 1.2, 11)

    def _ph_map(adf: pd.DataFrame) -> np.ndarray:
        c, _, _ = np.histogram2d(adf["x"], adf["y"], bins=[bins, bins])
        h, _, _ = np.histogram2d(adf["x"], adf["y"], bins=[bins, bins], weights=adf["choose_high"])
        return np.divide(h, c, out=np.full_like(h, np.nan, dtype=float), where=c > 0)

    m_close = _ph_map(d_close)
    m_far = _ph_map(d_far)
    m_diff = m_far - m_close

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.ravel()

    norm = mpl.colors.TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    diff_norm = mpl.colors.TwoSlopeNorm(vmin=-0.25, vcenter=0.0, vmax=0.25)

    im1 = ax1.pcolormesh(bins, bins, m_far.T, cmap="RdBu", norm=norm, shading="auto")
    im2 = ax2.pcolormesh(bins, bins, m_close.T, cmap="RdBu", norm=norm, shading="auto")
    im3 = ax3.pcolormesh(bins, bins, m_diff.T, cmap="RdBu_r", norm=diff_norm, shading="auto")

    for ax in [ax1, ax2, ax3]:
        v = _octagon_vertices(1.0)
        ax.plot(v[:, 0], v[:, 1], color="black", lw=1.2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)

    ax1.set_title("Opponent far from high")
    ax2.set_title("Opponent close to high")
    ax3.set_title("Difference: far - close")

    fig.colorbar(im1, ax=ax1, shrink=0.8)
    fig.colorbar(im2, ax=ax2, shrink=0.8)
    fig.colorbar(im3, ax=ax3, shrink=0.8)

    # Scatter per position bin.
    x = m_close.flatten()
    y = m_far.flatten()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    ax4.scatter(x, y, alpha=0.75, color="#7a0177")
    lims = [0.0, 1.0]
    ax4.plot(lims, lims, "k--", lw=1)
    ax4.set_xlim(lims)
    ax4.set_ylim(lims)
    ax4.set_xlabel("P(high | opp close)")
    ax4.set_ylabel("P(high | opp far)")
    ax4.set_title("Bin-wise comparison")

    if len(x) >= 3 and pearsonr is not None:
        r, p = pearsonr(x, y)
        ax4.text(0.05, 0.95, f"r={r:.2f}, p={p:.3g}", transform=ax4.transAxes, va="top")

    # GLMM-equivalent: choose_high ~ opp_far_close
    g = aligned.copy()
    g["opp_group"] = np.nan
    g.loc[g["opp_dist2high_at_onset"] < q40, "opp_group"] = 0
    g.loc[g["opp_dist2high_at_onset"] > q60, "opp_group"] = 1
    g = g[np.isfinite(g["opp_group"])].copy()
    glmm = _glmm_equivalent(g, formula="choose_high ~ opp_group")
    if glmm is not None:
        eff, p = glmm
        ax4.text(0.05, 0.05, f"GLMM approx: beta={eff:.3f}, p={p:.3g}", transform=ax4.transAxes)

    return fig


def run_social_analysis(df: pd.DataFrame, output_dir: str | Path) -> Dict[str, Path]:
    plt.style.use("seaborn-v0_8-whitegrid")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, Path] = {}

    fig6 = plot_preference_shift(df)
    _save_fig(fig6, out_dir, "plot6_preference_shift")
    plt.close(fig6)
    outputs["plot6"] = out_dir / "plot6_preference_shift.png"

    fig7_8 = plot_social_spatial(df)
    _save_fig(fig7_8, out_dir, "plot7_8_social_spatial")
    plt.close(fig7_8)
    outputs["plot7_8"] = out_dir / "plot7_8_social_spatial.png"

    fig9 = plot_delta_plow(df)
    _save_fig(fig9, out_dir, "plot9_delta_plow")
    plt.close(fig9)
    outputs["plot9"] = out_dir / "plot9_delta_plow.png"

    fig10 = plot_opponent_speed_corr(df)
    _save_fig(fig10, out_dir, "plot10_opponent_speed_corr")
    plt.close(fig10)
    outputs["plot10"] = out_dir / "plot10_opponent_speed_corr.png"

    fig11 = plot_opponent_position(df)
    _save_fig(fig11, out_dir, "plot11_opponent_position")
    plt.close(fig11)
    outputs["plot11"] = out_dir / "plot11_opponent_position.png"

    return outputs
