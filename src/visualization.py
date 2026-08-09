"""
visualization.py
================
Figure generation for the MCC feasibility study.

Produces two publication-quality figures:
  Figure A: Three-panel summary (RMSE, F1, daily energy) across participation rates.
  Figure B: One-week time-series excerpt illustrating load estimation and detection.

All figures are saved as both PDF (vector, for the paper) and PNG (300 dpi).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d
from pathlib import Path

# ── Colour palette ─────────────────────────────────────────────────────────────
BLUE       = "#1A5276"
BLUE_LIGHT = "#5DADE2"
BLUE_MID   = "#2874A6"
RED        = "#C0392B"
GREY       = "#7F8C8D"
GREY_LIGHT = "#BDC3C7"


def save_fig(fig: plt.Figure, stem: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {stem}.pdf / .png")


def figure_summary(
    rate_results,       # list[RateResult] from mcc_simulation
    feeder_gt: np.ndarray,
    energy_config,      # EnergyConfig from energy_model
    out_dir: Path,
    dataset_label: str = "Ausgrid 2012–2013",
) -> None:
    """
    Three-panel summary figure:
      (a) Normalised RMSE vs participation rate
      (b) Anomaly detection F1 score vs participation rate
      (c) Daily operational energy — MCC vs conventional edge baseline (log scale)
    """
    from .energy_model import mcc_daily_energy_wh, edge_daily_energy_wh

    rates_pct  = [int(r.participation_rate * 100) for r in rate_results]
    rmse_means = [r.rmse_mean for r in rate_results]
    rmse_stds  = [r.rmse_std  for r in rate_results]
    f1_means   = [r.f1_mean   for r in rate_results]
    f1_stds    = [r.f1_std    for r in rate_results]
    n_devices  = [r.n_devices for r in rate_results]

    feeder_mean = feeder_gt.mean()
    nrmse       = [v / feeder_mean * 100 for v in rmse_means]
    nrmse_std   = [v / feeder_mean * 100 for v in rmse_stds]

    mcc_energy_wh  = [mcc_daily_energy_wh(n, energy_config) for n in n_devices]
    edge_energy_wh = edge_daily_energy_wh(energy_config)

    fig = plt.figure(figsize=(14, 4.5))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.40)
    ax1, ax2, ax3 = [fig.add_subplot(gs[i]) for i in range(3)]

    # Panel (a) — RMSE
    ax1.errorbar(rates_pct, nrmse, yerr=nrmse_std,
                 fmt="o-", color=BLUE, linewidth=2, markersize=6,
                 capsize=4, capthick=1.5, elinewidth=1.5, label="MCC estimate")
    ax1.axhline(0, color=GREY, linestyle="--", linewidth=1,
                label="Feeder sensor (ideal)")
    ax1.set_xlabel("Participation rate (%)", fontsize=11)
    ax1.set_ylabel("Normalised RMSE (% of mean feeder load)", fontsize=10)
    ax1.set_title("(a) Load estimation accuracy", fontsize=11, fontweight="bold")
    ax1.set_xticks(rates_pct)
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle=":")
    for sp in ["top", "right"]: ax1.spines[sp].set_visible(False)

    # Panel (b) — F1
    ax2.errorbar(rates_pct, f1_means, yerr=f1_stds,
                 fmt="s-", color=BLUE, linewidth=2, markersize=6,
                 capsize=4, capthick=1.5, elinewidth=1.5, label="MCC detection")
    ax2.axhline(1.0, color=GREY, linestyle="--", linewidth=1,
                label="Feeder sensor (ideal, no spatial res.)")
    ax2.set_xlabel("Participation rate (%)", fontsize=11)
    ax2.set_ylabel("F1 score (anomaly detection)", fontsize=10)
    ax2.set_title("(b) Anomaly detection accuracy", fontsize=11, fontweight="bold")
    ax2.set_xticks(rates_pct)
    ax2.set_ylim(0.0, 1.10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle=":")
    for sp in ["top", "right"]: ax2.spines[sp].set_visible(False)

    # Panel (c) — Energy (log scale)
    ax3.bar(rates_pct, mcc_energy_wh, width=8, color=BLUE, alpha=0.85,
            label="MCC (all devices)", zorder=3)
    ax3.axhline(edge_energy_wh, color=RED, linestyle="--", linewidth=2,
                label=f"Conventional {energy_config.edge_nodes}-node baseline "
                      f"({edge_energy_wh:.0f} Wh/day)")
    ax3.set_xlabel("Participation rate (%)", fontsize=11)
    ax3.set_ylabel("Daily energy consumption (Wh)", fontsize=10)
    ax3.set_title("(c) Daily operational energy", fontsize=11, fontweight="bold")
    ax3.set_xticks(rates_pct)
    ax3.set_yscale("log")
    ax3.legend(fontsize=9)
    ax3.grid(True, axis="y", alpha=0.3, linestyle=":")
    for sp in ["top", "right"]: ax3.spines[sp].set_visible(False)

    # Annotate energy ratios
    for i, n in enumerate(n_devices):
        ratio = edge_energy_wh / mcc_energy_wh[i]
        ax3.text(rates_pct[i], mcc_energy_wh[i] * 1.4,
                 f"{ratio:.0f}×", ha="center", va="bottom",
                 fontsize=9, color=RED, fontweight="bold")

    fig.suptitle(
        f"MCC feasibility assessment — {dataset_label} "
        f"(N={round(rate_results[0].n_devices / rate_results[0].participation_rate)}"
        f" households, {rate_results[0].n_trials} Monte Carlo trials per rate)",
        fontsize=10, y=1.02, style="italic", color="#444444"
    )

    save_fig(fig, "fig_mcc_summary", out_dir)


def figure_timeseries(
    loads: np.ndarray,
    anomaly_mask: np.ndarray,
    feeder_gt: np.ndarray,
    rate_results,
    sim_config,
    out_dir: Path,
    week_start_day: int = 21,
) -> None:
    """
    Two-panel time-series figure for one representative week:
      (top)    Ground truth vs MCC estimates at three participation rates.
      (bottom) Anomaly detection probability across 20 trials.
    """
    from .mcc_simulation import SimulationConfig

    N_HH, N_TS = loads.shape
    N_INT = 48
    t_start = week_start_day * N_INT
    t_end   = (week_start_day + 7) * N_INT
    t_hours = np.arange(t_end - t_start) * 0.5

    gt_excerpt      = feeder_gt[t_start:t_end]
    anomaly_excerpt = anomaly_mask[t_start:t_end]

    gt_mean = feeder_gt.mean()
    gt_std  = feeder_gt.std()
    threshold = gt_mean + sim_config.anomaly_threshold_sigma * gt_std

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.patch.set_facecolor("white")

    rng = np.random.default_rng(0)

    # Top panel — load estimates
    style_map = {
        0.10: (BLUE_LIGHT, 1.0, "10% (30 devices)"),
        0.25: (BLUE_MID,   1.5, "25% (75 devices)"),
        0.75: (BLUE,       2.0, "75% (225 devices)"),
    }

    for rate, (color, lw, label) in style_map.items():
        n_dev = max(1, int(N_HH * rate))
        selected = rng.choice(N_HH, n_dev, replace=False)
        tau = rng.beta(sim_config.trust_alpha, sim_config.trust_beta, n_dev)
        tau /= tau.sum()
        sample = loads[selected, t_start:t_end]
        est = (tau @ sample) * N_HH
        axes[0].plot(t_hours, est, color=color, linewidth=lw,
                     alpha=0.85, label=label, zorder=3)

    axes[0].plot(t_hours, gt_excerpt, color="black", linewidth=2.0,
                 label="Ground truth (feeder aggregate)", zorder=5)

    # Shade anomaly intervals
    shaded = False
    for i in range(len(t_hours)):
        if anomaly_excerpt[i]:
            axes[0].axvspan(t_hours[i], t_hours[i] + 0.5,
                            color=RED, alpha=0.12, zorder=1,
                            label="Injected anomaly" if not shaded else "")
            shaded = True

    axes[0].set_ylabel("Feeder load (kWh per 30 min)", fontsize=11)
    axes[0].set_title("MCC load estimation — one-week excerpt",
                      fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9, loc="upper left")
    axes[0].grid(True, alpha=0.25, linestyle=":")
    for sp in ["top", "right"]: axes[0].spines[sp].set_visible(False)

    # Bottom panel — detection probability
    axes[1].fill_between(t_hours, 0, anomaly_excerpt.astype(float),
                         color=RED, alpha=0.25, label="True anomaly interval")

    rng2 = np.random.default_rng(1)
    for rate, color, label in [
        (0.25, BLUE_MID, "25% participation"),
        (0.75, BLUE,     "75% participation"),
    ]:
        n_dev = max(1, int(N_HH * rate))
        votes = np.zeros(t_end - t_start)
        n_det_trials = 30
        for _ in range(n_det_trials):
            sel = rng2.choice(N_HH, n_dev, replace=False)
            tau = rng2.beta(sim_config.trust_alpha, sim_config.trust_beta, n_dev)
            tau /= tau.sum()
            sample = loads[sel, t_start:t_end]
            est = (tau @ sample) * N_HH
            votes += (est > threshold).astype(float)
        axes[1].plot(t_hours, votes / n_det_trials,
                     color=color, linewidth=1.5, label=label)

    axes[1].set_xlabel("Time (hours from start of week)", fontsize=11)
    axes[1].set_ylabel("Detection probability", fontsize=11)
    axes[1].set_title(f"Anomaly detection probability ({n_det_trials} trials)",
                      fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(-0.05, 1.15)
    axes[1].grid(True, alpha=0.25, linestyle=":")
    for sp in ["top", "right"]: axes[1].spines[sp].set_visible(False)

    # Day boundaries
    for d in range(1, 7):
        for ax in axes:
            ax.axvline(d * 24, color=GREY_LIGHT, linewidth=0.6)
        axes[1].text(d * 24 + 1, 1.08, f"Day {week_start_day + d + 1}",
                     fontsize=8, color=GREY)

    plt.tight_layout()
    save_fig(fig, "fig_mcc_timeseries", out_dir)
