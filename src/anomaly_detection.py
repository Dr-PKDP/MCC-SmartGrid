"""
anomaly_detection.py
====================
Anomaly injection and detection evaluation for the MCC feasibility study.

Anomalies represent high-load voltage-stress events: periods in which a
spatially clustered subset of households simultaneously draws elevated power,
producing feeder-level load exceedances that are the signature of incipient
voltage regulation problems in residential distribution networks.

In the MCC model, each participating device detects such events through
proxy signals (charging behaviour, power-quality effects on electronics)
and the aggregated trust-weighted output (Eq. 5) crosses a detection
threshold when enough affected devices contribute correlated observations.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class AnomalyConfig:
    """
    Parameters controlling anomaly injection.

    Attributes
    ----------
    n_anomalies : int
        Number of distinct high-load events to inject.
    min_duration : int
        Minimum event duration in half-hourly intervals.
    max_duration : int
        Maximum event duration in half-hourly intervals.
    min_affected_frac : float
        Minimum fraction of households simultaneously affected per event.
    max_affected_frac : float
        Maximum fraction of households simultaneously affected per event.
    min_scale : float
        Minimum load scaling factor applied to affected households.
    max_scale : float
        Maximum load scaling factor applied to affected households.
    seed : int
        Random seed for reproducibility.
    """
    n_anomalies: int = 40
    min_duration: int = 2
    max_duration: int = 4
    min_affected_frac: float = 0.40
    max_affected_frac: float = 0.65
    min_scale: float = 3.0
    max_scale: float = 5.0
    seed: int = 123


def inject_anomalies(
    loads: np.ndarray,
    config: AnomalyConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inject synthetic high-load events into the load array.

    A copy of ``loads`` is returned with events injected; the original
    array is not modified. A boolean mask marks all timesteps belonging
    to at least one injected event.

    Parameters
    ----------
    loads : np.ndarray, shape (N_HH, N_TIMESTEPS)
        Original household load values (kWh per interval).
    config : AnomalyConfig
        Injection parameters.

    Returns
    -------
    loads_with_anomalies : np.ndarray, shape (N_HH, N_TIMESTEPS)
        Load array with events injected.
    anomaly_mask : np.ndarray, shape (N_TIMESTEPS,), dtype bool
        True at every timestep that belongs to an injected event.
    """
    n_hh, n_ts = loads.shape
    rng = np.random.default_rng(config.seed)

    loads_out = loads.copy()
    anomaly_mask = np.zeros(n_ts, dtype=bool)

    # Guard band: avoid the first and last full day
    guard = 48
    t_min = guard
    t_max = n_ts - guard * 2

    for _ in range(config.n_anomalies):
        t_start = int(rng.integers(t_min, t_max))
        duration = int(rng.integers(config.min_duration, config.max_duration + 1))

        n_affected = max(1, int(n_hh * rng.uniform(
            config.min_affected_frac, config.max_affected_frac
        )))
        affected = rng.choice(n_hh, n_affected, replace=False)
        scale = rng.uniform(config.min_scale, config.max_scale)

        for offset in range(duration):
            t = t_start + offset
            if t < n_ts:
                loads_out[affected, t] = loads_out[affected, t] * scale
                anomaly_mask[t] = True

    return loads_out, anomaly_mask


def detection_stats(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> dict:
    """
    Compute precision, recall, and F1 for binary anomaly detection.

    Parameters
    ----------
    predicted : np.ndarray, dtype bool
        Model's binary anomaly predictions.
    ground_truth : np.ndarray, dtype bool
        True anomaly labels.

    Returns
    -------
    dict with keys 'precision', 'recall', 'f1', 'tp', 'fp', 'fn'.
    """
    tp = int(np.logical_and(predicted,  ground_truth).sum())
    fp = int(np.logical_and(predicted, ~ground_truth).sum())
    fn = int(np.logical_and(~predicted, ground_truth).sum())

    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1        = 2 * precision * recall / (precision + recall + 1e-12)

    return dict(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)
