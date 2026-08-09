"""
mcc_simulation.py
=================
Core MCC participation simulation implementing the computational model
(Equations 1–5) from:

    Pramanik, P.K.D. (2025). Mobile Crowd Computing as a Sustainable
    Edge Computing Paradigm for Smart Grids. [Journal TBD].

Each household in the dataset is treated as a potential MCC device.
Its half-hourly GC reading is the proxy observation that an MCC-enabled
smartphone in that household would contribute — an indirect signal
consistent with Section 5.1's indirect sensing model.

The MCC system estimates feeder-level load by aggregating contributions
from a random subset of participating households using trust-weighted
aggregation (Eq. 5), then scales the sample estimate to the full
population.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SimulationConfig:
    """
    Configuration for an MCC participation simulation run.

    Attributes
    ----------
    participation_rates : list[float]
        Fractions of the total household population to simulate as
        MCC participants. Each value in [0, 1].
    n_monte_carlo : int
        Number of independent random-sample trials per participation rate.
        200 gives stable mean estimates; 500 for publication-quality CIs.
    trust_alpha : float
        Alpha parameter of the Beta distribution for device trust weights
        (Eq. 1, τ_i component). Beta(4, 2) produces a right-skewed
        distribution modelling a realistic device population where most
        devices have moderate-to-high reliability.
    trust_beta : float
        Beta parameter of the trust weight distribution.
    anomaly_threshold_sigma : float
        Number of standard deviations above mean feeder load at which
        a timestep is classified as an anomaly for detection evaluation.
    seed : int
        Random seed for reproducibility.
    """
    participation_rates: list = field(
        default_factory=lambda: [0.10, 0.25, 0.50, 0.75, 1.00]
    )
    n_monte_carlo: int = 200
    trust_alpha: float = 4.0
    trust_beta: float = 2.0
    anomaly_threshold_sigma: float = 2.0
    seed: int = 42


@dataclass
class TrialResult:
    """Results for a single Monte Carlo trial."""
    rmse: float
    mae: float
    f1: float
    precision: float
    recall: float


@dataclass
class RateResult:
    """Aggregated results across all Monte Carlo trials for one participation rate."""
    participation_rate: float
    n_devices: int
    n_trials: int
    rmse_mean: float
    rmse_std: float
    mae_mean: float
    mae_std: float
    f1_mean: float
    f1_std: float
    precision_mean: float
    recall_mean: float


def run_simulation(
    loads: np.ndarray,
    config: SimulationConfig,
    feeder_gt: Optional[np.ndarray] = None,
    anomaly_mask: Optional[np.ndarray] = None,
    slot_mean: Optional[np.ndarray] = None,
    slot_std: Optional[np.ndarray] = None,
) -> list[RateResult]:
    """
    Run MCC participation simulation across all specified participation rates.

    Parameters
    ----------
    loads : np.ndarray, shape (N_HH, N_TIMESTEPS)
        Half-hourly load values for all households (kWh per interval).
        Each row is a household; each column is a timestep.
    config : SimulationConfig
        Simulation configuration.
    feeder_gt : np.ndarray, shape (N_TIMESTEPS,), optional
        Ground-truth feeder load (sum of all households). Computed from
        ``loads`` if not provided.
    anomaly_mask : np.ndarray, shape (N_TIMESTEPS,), dtype bool, optional
        Boolean mask marking anomalous timesteps (True = anomaly).
        If provided, anomaly detection metrics (F1, precision, recall)
        are computed against this ground truth.
    slot_mean : np.ndarray, shape (N_INTERVALS,), optional
        Per-half-hour-slot mean of baseline feeder load. When provided,
        anomaly detection uses a time-of-day-aware z-score threshold
        (estimate > slot_mean + sigma * slot_std) rather than a global
        threshold. Recommended for real-world datasets with strong
        diurnal load patterns.
    slot_std : np.ndarray, shape (N_INTERVALS,), optional
        Per-half-hour-slot standard deviation of baseline feeder load.
        Required if slot_mean is provided.

    Returns
    -------
    list[RateResult]
        One result object per participation rate in ``config.participation_rates``.
    """
    rng = np.random.default_rng(config.seed)
    n_hh, n_ts = loads.shape

    if feeder_gt is None:
        feeder_gt = loads.sum(axis=0)

    n_ts = loads.shape[1]
    n_int = 48  # half-hourly intervals per day
    n_days = n_ts // n_int

    # Determine anomaly detection mode
    use_slot_threshold = (slot_mean is not None and slot_std is not None)

    if not use_slot_threshold:
        # Global threshold: mean + k * std of ground-truth feeder load
        gt_mean = feeder_gt.mean()
        gt_std  = feeder_gt.std()
        threshold = gt_mean + config.anomaly_threshold_sigma * gt_std

    if anomaly_mask is None:
        if use_slot_threshold:
            z = ((feeder_gt.reshape(n_days, n_int) - slot_mean[np.newaxis, :]) /
                 (slot_std[np.newaxis, :] + 1e-12)).flatten()
            anomaly_mask = z > config.anomaly_threshold_sigma
        else:
            anomaly_mask = feeder_gt > threshold

    results = []

    for rate in config.participation_rates:
        n_devices = max(1, int(np.floor(n_hh * rate)))
        trials: list[TrialResult] = []

        for _ in range(config.n_monte_carlo):
            # --- Eq. 3: select S*_j — random subset of n_devices households ---
            selected = rng.choice(n_hh, n_devices, replace=False)

            # --- Eq. 1: trust weights τ_i ~ Beta(α, β) ---
            tau = rng.beta(config.trust_alpha, config.trust_beta, n_devices)
            tau = tau / tau.sum()   # normalise to sum to 1

            # --- Eq. 5: trust-weighted aggregation ---
            sample_loads = loads[selected, :]          # (n_devices, N_TIMESTEPS)
            weighted_mean = tau @ sample_loads          # (N_TIMESTEPS,)
            # Scale from n_devices to full population (N_HH)
            estimate = weighted_mean * n_hh

            # --- Estimation accuracy ---
            diff = estimate - feeder_gt
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            mae = float(np.mean(np.abs(diff)))

            # --- Anomaly detection: threshold crossing ---
            if use_slot_threshold:
                z_est = ((estimate.reshape(n_days, n_int) - slot_mean[np.newaxis, :]) /
                         (slot_std[np.newaxis, :] + 1e-12)).flatten()
                predicted = z_est > config.anomaly_threshold_sigma
            else:
                predicted = estimate > threshold
            tp = int(np.logical_and(predicted,  anomaly_mask).sum())
            fp = int(np.logical_and(predicted,  ~anomaly_mask).sum())
            fn = int(np.logical_and(~predicted, anomaly_mask).sum())

            precision = tp / (tp + fp + 1e-12)
            recall    = tp / (tp + fn + 1e-12)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)

            trials.append(TrialResult(rmse=rmse, mae=mae, f1=f1,
                                      precision=precision, recall=recall))

        # Aggregate across trials
        rmse_arr  = np.array([t.rmse      for t in trials])
        mae_arr   = np.array([t.mae       for t in trials])
        f1_arr    = np.array([t.f1        for t in trials])
        prec_arr  = np.array([t.precision for t in trials])
        rec_arr   = np.array([t.recall    for t in trials])

        results.append(RateResult(
            participation_rate = rate,
            n_devices          = n_devices,
            n_trials           = config.n_monte_carlo,
            rmse_mean          = float(rmse_arr.mean()),
            rmse_std           = float(rmse_arr.std()),
            mae_mean           = float(mae_arr.mean()),
            mae_std            = float(mae_arr.std()),
            f1_mean            = float(f1_arr.mean()),
            f1_std             = float(f1_arr.std()),
            precision_mean     = float(prec_arr.mean()),
            recall_mean        = float(rec_arr.mean()),
        ))

    return results
