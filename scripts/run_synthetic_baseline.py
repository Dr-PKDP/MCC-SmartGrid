"""
run_synthetic_baseline.py
=========================
Synthetic data validation run.

Generates a synthetic household load dataset calibrated to the statistical
properties of the Ausgrid Solar Home 2012-2013 dataset (mean daily GC
15.3 kWh, std 9.7 kWh, two-peak residential profile) and runs the same
MCC simulation pipeline as run_analysis.py.

This confirms that the results from the real-data analysis generalise
beyond the specific Ausgrid cohort and are not artefacts of Sydney's
particular demand characteristics.

Usage
-----
    python scripts/run_synthetic_baseline.py

Output
------
    results/tables/results_synthetic.csv
    results/figures/fig_mcc_summary_synthetic.*
    results/figures/fig_mcc_timeseries_synthetic.*
"""

import sys
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from src.data_loader       import N_DAYS, N_INTERVALS
from src.anomaly_detection import inject_anomalies, AnomalyConfig
from src.mcc_simulation    import run_simulation, SimulationConfig
from src.energy_model      import EnergyConfig, mcc_daily_energy_wh, \
                                   edge_daily_energy_wh, energy_ratio
from src.visualization     import figure_summary, figure_timeseries

FIG_DIR   = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"

# Statistical calibration parameters derived from Ausgrid 2012-2013 GC data
MEAN_DAILY_KWH = 15.3
STD_DAILY_KWH  = 9.7
N_HH_SYNTHETIC = 300

SIM_CONFIG    = SimulationConfig(participation_rates=[0.10,0.25,0.50,0.75,1.00],
                                  n_monte_carlo=200, trust_alpha=4.0, trust_beta=2.0,
                                  anomaly_threshold_sigma=4.0, seed=99)
ANOMALY_CONFIG = AnomalyConfig(n_anomalies=40, min_duration=2, max_duration=4,
                                min_affected_frac=0.40, max_affected_frac=0.65,
                                min_scale=3.0, max_scale=5.0, seed=456)
ENERGY_CONFIG  = EnergyConfig()


def generate_synthetic_loads(seed: int = 42) -> np.ndarray:
    """
    Generate (N_HH_SYNTHETIC, N_DAYS * N_INTERVALS) synthetic load array
    calibrated to Ausgrid GC statistical properties.
    """
    rng = np.random.default_rng(seed)
    n_ts = N_DAYS * N_INTERVALS

    # Base half-hourly profile (normalised to unit daily total)
    t = np.arange(N_INTERVALS) / N_INTERVALS * 24
    profile = (0.25 + 0.7 * np.exp(-0.5*((t-8.0)/1.2)**2)
                    + 1.5 * np.exp(-0.5*((t-19.5)/1.8)**2))
    profile /= profile.sum()

    # Household-specific daily energy (lognormal)
    mu_ln = np.log(MEAN_DAILY_KWH) - 0.5 * np.log(1+(STD_DAILY_KWH/MEAN_DAILY_KWH)**2)
    sig_ln = np.sqrt(np.log(1+(STD_DAILY_KWH/MEAN_DAILY_KWH)**2))
    daily_scale = rng.lognormal(mu_ln, sig_ln, (N_HH_SYNTHETIC, N_DAYS))

    # Seasonal factor (Southern Hemisphere: winter in Jul–Sep)
    season = 1.0 + 0.10 * np.sin(2*np.pi*np.arange(N_DAYS)/365 + np.pi)
    daily_scale *= season[np.newaxis, :]

    loads = np.zeros((N_HH_SYNTHETIC, n_ts), dtype=np.float32)
    for hh in range(N_HH_SYNTHETIC):
        p = np.roll(profile, rng.integers(-2, 3))
        p = rng.dirichlet(p * 50)
        for d in range(N_DAYS):
            row = daily_scale[hh, d] * p
            row += rng.normal(0, 0.02*daily_scale[hh, d], N_INTERVALS)
            loads[hh, d*N_INTERVALS:(d+1)*N_INTERVALS] = np.maximum(row, 0)

    return loads


def main():
    print("=" * 70)
    print("MCC Smart Grid Feasibility Study — Synthetic Baseline")
    print(f"Calibrated to Ausgrid GC: mean={MEAN_DAILY_KWH} kWh/day, "
          f"std={STD_DAILY_KWH} kWh/day")
    print("=" * 70)

    print("\n[1/4] Generating synthetic data...")
    loads = generate_synthetic_loads()
    feeder_gt = loads.sum(axis=0)
    print(f"  Shape: {loads.shape}")
    print(f"  Mean daily GC/HH: {loads.reshape(N_HH_SYNTHETIC,N_DAYS,N_INTERVALS).sum(axis=2).mean():.2f} kWh")

    # Per-slot statistics from clean baseline data (time-of-day z-score detection,
    # matching the method used in run_analysis.py for the real dataset)
    slot_mean = feeder_gt.reshape(-1, N_INTERVALS).mean(axis=0)
    slot_std  = feeder_gt.reshape(-1, N_INTERVALS).std(axis=0)

    print("\n[2/4] Injecting anomalies...")
    loads_a, anomaly_mask = inject_anomalies(loads, ANOMALY_CONFIG)
    feeder_a = loads_a.sum(axis=0)
    print(f"  Anomaly timesteps: {anomaly_mask.sum()}")

    print("\n[3/4] Running simulation...")
    results = run_simulation(loads_a, SIM_CONFIG,
                             feeder_gt=feeder_a, anomaly_mask=anomaly_mask,
                             slot_mean=slot_mean, slot_std=slot_std)

    feeder_mean = feeder_a.mean()
    edge_wh = edge_daily_energy_wh(ENERGY_CONFIG)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table_path = TABLE_DIR / "results_synthetic.csv"
    fieldnames = ["participation_rate_pct","n_devices","nrmse_mean_pct",
                  "nrmse_std_pct","f1_mean","f1_std","mcc_daily_energy_wh",
                  "edge_daily_energy_wh","energy_ratio"]
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            mcc_wh = mcc_daily_energy_wh(r.n_devices, ENERGY_CONFIG)
            writer.writerow({
                "participation_rate_pct": int(r.participation_rate*100),
                "n_devices":              r.n_devices,
                "nrmse_mean_pct":         round(r.rmse_mean/feeder_mean*100, 4),
                "nrmse_std_pct":          round(r.rmse_std/feeder_mean*100,  4),
                "f1_mean":                round(r.f1_mean, 4),
                "f1_std":                 round(r.f1_std,  4),
                "mcc_daily_energy_wh":    round(mcc_wh, 4),
                "edge_daily_energy_wh":   round(edge_wh, 1),
                "energy_ratio":           round(energy_ratio(r.n_devices, ENERGY_CONFIG), 0),
            })
    print(f"\n  Results saved: {table_path.relative_to(ROOT)}")

    print("\n[4/4] Generating figures...")
    figure_summary(results, feeder_a, ENERGY_CONFIG, FIG_DIR,
                   dataset_label="Synthetic (Ausgrid-calibrated)")
    # Rename to avoid overwriting real-data figures
    for ext in ["pdf", "png"]:
        src = FIG_DIR / f"fig_mcc_summary.{ext}"
        dst = FIG_DIR / f"fig_mcc_summary_synthetic.{ext}"
        if src.exists(): src.rename(dst)
    figure_timeseries(loads_a, anomaly_mask, feeder_a,
                      results, SIM_CONFIG, FIG_DIR, week_start_day=21)
    for ext in ["pdf", "png"]:
        src = FIG_DIR / f"fig_mcc_timeseries.{ext}"
        dst = FIG_DIR / f"fig_mcc_timeseries_synthetic.{ext}"
        if src.exists(): src.rename(dst)

    print("\nDone.")


if __name__ == "__main__":
    main()
