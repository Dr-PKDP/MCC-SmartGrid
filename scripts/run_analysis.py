"""
run_analysis.py
===============
Main analysis script. Reproduces all results in Section 5.6 of:

    Pramanik, P.K.D. (2025). Mobile Crowd Computing as a Sustainable
    Edge Computing Paradigm for Smart Grids. [Journal TBD].

Usage
-----
    python scripts/run_analysis.py

Output
------
    results/tables/results_real_data.csv   — numerical results table
    results/figures/fig_mcc_summary.*      — three-panel summary figure
    results/figures/fig_mcc_timeseries.*   — time-series excerpt figure

Requirements
------------
    See requirements.txt. Install with:
        pip install -r requirements.txt
"""

import sys
import csv
from pathlib import Path

# Make src importable regardless of working directory
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from src.data_loader      import load_ausgrid, dataset_summary, N_DAYS, N_INTERVALS
from src.anomaly_detection import inject_anomalies, AnomalyConfig
from src.mcc_simulation   import run_simulation, SimulationConfig
from src.energy_model     import EnergyConfig, mcc_daily_energy_wh, edge_daily_energy_wh, energy_ratio
from src.visualization    import figure_summary, figure_timeseries

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_FILE   = ROOT / "data" / "2012-2013-Solar-home-electricity-data.csv"
FIG_DIR     = ROOT / "results" / "figures"
TABLE_DIR   = ROOT / "results" / "tables"

# ── Configuration ──────────────────────────────────────────────────────────────
SIM_CONFIG = SimulationConfig(
    participation_rates    = [0.10, 0.25, 0.50, 0.75, 1.00],
    n_monte_carlo          = 200,
    trust_alpha            = 4.0,
    trust_beta             = 2.0,
    anomaly_threshold_sigma = 4.0,   # per-slot z-score; 4σ for high-precision detection
    seed                   = 42,
)

ANOMALY_CONFIG = AnomalyConfig(
    n_anomalies          = 40,
    min_duration         = 2,
    max_duration         = 4,
    min_affected_frac    = 0.40,
    max_affected_frac    = 0.65,
    min_scale            = 3.0,
    max_scale            = 5.0,
    seed                 = 123,
)

ENERGY_CONFIG = EnergyConfig(
    e_session_j       = 2.97,   # Patterson et al. (2024)
    sessions_per_day  = 12,
    edge_nodes        = 100,
    edge_node_power_w = 5.0,
)


def main():
    print("=" * 70)
    print("MCC Smart Grid Feasibility Study — Real Data Analysis")
    print("Dataset: Ausgrid Solar Home Electricity 2012-2013")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading dataset...")
    if not DATA_FILE.exists():
        print(f"ERROR: Dataset not found at {DATA_FILE}")
        print("Please place the Ausgrid CSV at that path and retry.")
        sys.exit(1)

    loads, customer_ids = load_ausgrid(DATA_FILE, category="GC", min_days=365)
    summary = dataset_summary(loads, customer_ids)
    print(f"  Households retained (365 complete days): {summary['n_households']}")
    print(f"  Total timesteps: {summary['n_timesteps']}")
    print(f"  Mean daily GC per household: {summary['mean_daily_kwh_per_hh']:.2f} kWh")
    print(f"  Std daily GC per household:  {summary['std_daily_kwh_per_hh']:.2f} kWh")
    print(f"  Mean feeder load per interval: {summary['mean_feeder_load_per_interval_kwh']:.2f} kWh")

    feeder_gt = loads.sum(axis=0)
    N_INT = 48

    # Per-slot statistics from clean baseline data (for time-of-day z-score detection)
    slot_mean = feeder_gt.reshape(-1, N_INT).mean(axis=0)
    slot_std  = feeder_gt.reshape(-1, N_INT).std(axis=0)

    # ── 2. Inject anomalies ───────────────────────────────────────────────────
    print("\n[2/5] Injecting anomaly events...")
    loads_a, anomaly_mask = inject_anomalies(loads, ANOMALY_CONFIG)
    feeder_a = loads_a.sum(axis=0)
    n_anomaly = int(anomaly_mask.sum())
    print(f"  Anomaly timesteps: {n_anomaly} of {loads.shape[1]} "
          f"({100 * n_anomaly / loads.shape[1]:.1f}%)")

    # ── 3. Run MCC simulation ─────────────────────────────────────────────────
    print("\n[3/5] Running MCC participation simulation...")
    results = run_simulation(loads_a, SIM_CONFIG,
                             feeder_gt=feeder_a, anomaly_mask=anomaly_mask,
                             slot_mean=slot_mean, slot_std=slot_std)

    feeder_mean = feeder_a.mean()
    edge_wh = edge_daily_energy_wh(ENERGY_CONFIG)

    print(f"\n{'Rate':>6} {'N_dev':>6} {'RMSE%':>8} {'±':>5} "
          f"{'F1':>7} {'±':>5} {'MCC_Wh':>9} {'Ratio':>8}")
    print("-" * 65)
    for r in results:
        nrmse = r.rmse_mean / feeder_mean * 100
        nrmse_s = r.rmse_std / feeder_mean * 100
        mcc_wh = mcc_daily_energy_wh(r.n_devices, ENERGY_CONFIG)
        ratio = energy_ratio(r.n_devices, ENERGY_CONFIG)
        print(f"{int(r.participation_rate*100):5d}% {r.n_devices:6d} "
              f"{nrmse:8.2f} {nrmse_s:5.2f} "
              f"{r.f1_mean:7.3f} {r.f1_std:5.3f} "
              f"{mcc_wh:9.3f} {ratio:8.0f}×")
    print(f"\n  Conventional 100-node baseline: {edge_wh:.0f} Wh/day")

    # ── 4. Save results table ─────────────────────────────────────────────────
    print("\n[4/5] Saving results table...")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table_path = TABLE_DIR / "results_real_data.csv"
    fieldnames = [
        "participation_rate_pct", "n_devices",
        "nrmse_mean_pct", "nrmse_std_pct",
        "mae_mean_kwh", "mae_std_kwh",
        "f1_mean", "f1_std",
        "precision_mean", "recall_mean",
        "mcc_daily_energy_wh", "edge_daily_energy_wh", "energy_ratio",
    ]
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            mcc_wh = mcc_daily_energy_wh(r.n_devices, ENERGY_CONFIG)
            writer.writerow({
                "participation_rate_pct": int(r.participation_rate * 100),
                "n_devices":              r.n_devices,
                "nrmse_mean_pct":         round(r.rmse_mean / feeder_mean * 100, 4),
                "nrmse_std_pct":          round(r.rmse_std  / feeder_mean * 100, 4),
                "mae_mean_kwh":           round(r.mae_mean, 4),
                "mae_std_kwh":            round(r.mae_std,  4),
                "f1_mean":                round(r.f1_mean,  4),
                "f1_std":                 round(r.f1_std,   4),
                "precision_mean":         round(r.precision_mean, 4),
                "recall_mean":            round(r.recall_mean,    4),
                "mcc_daily_energy_wh":    round(mcc_wh,           4),
                "edge_daily_energy_wh":   round(edge_wh,          1),
                "energy_ratio":           round(energy_ratio(r.n_devices, ENERGY_CONFIG), 0),
            })
    print(f"  Saved: {table_path.relative_to(ROOT)}")

    # ── 5. Generate figures ───────────────────────────────────────────────────
    print("\n[5/5] Generating figures...")
    figure_summary(results, feeder_a, ENERGY_CONFIG, FIG_DIR,
                   dataset_label="Ausgrid Solar Home 2012–2013")
    figure_timeseries(loads_a, anomaly_mask, feeder_a,
                      results, SIM_CONFIG, FIG_DIR, week_start_day=28)

    print("\n" + "=" * 70)
    print("Analysis complete. Outputs:")
    print(f"  Figures : {FIG_DIR.relative_to(ROOT)}/")
    print(f"  Table   : {table_path.relative_to(ROOT)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
