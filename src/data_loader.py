"""
data_loader.py
==============
Load and preprocess the Ausgrid Solar Home Electricity dataset
(2012–2013, half-hourly, 300 residential solar households, Sydney NSW).

Dataset source:
    Ausgrid. (2014). Solar Home Electricity Data.
    https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data
    Creative Commons Attribution 4.0 International (CC BY 4.0)

Statistical reference:
    Ratnam, E.L., Weller, S.R., Kellett, C.M., and Murray, A.T. (2017).
    Residential load and rooftop PV generation: an Australian distribution
    network dataset. International Journal of Sustainable Energy, 36(8),
    787–806. https://doi.org/10.1080/14786451.2015.1100196
"""

import numpy as np
import pandas as pd
from pathlib import Path


# Half-hourly interval column labels (48 per day, 00:30 through 00:00)
INTERVAL_COLS = [
    "0:30","1:00","1:30","2:00","2:30","3:00","3:30","4:00","4:30","5:00",
    "5:30","6:00","6:30","7:00","7:30","8:00","8:30","9:00","9:30","10:00",
    "10:30","11:00","11:30","12:00","12:30","13:00","13:30","14:00","14:30",
    "15:00","15:30","16:00","16:30","17:00","17:30","18:00","18:30","19:00",
    "19:30","20:00","20:30","21:00","21:30","22:00","22:30","23:00","23:30",
    "0:00",
]
N_INTERVALS = 48   # half-hourly intervals per day
N_DAYS = 365       # July 2012 – June 2013
N_TIMESTEPS = N_DAYS * N_INTERVALS  # 17,520


def load_ausgrid(csv_path: str, category: str = "GC",
                 min_days: int = 365) -> tuple[np.ndarray, list[int]]:
    """
    Load the Ausgrid Solar Home CSV and return a (N_HH, N_TIMESTEPS) array.

    Parameters
    ----------
    csv_path : str
        Path to '2012-2013 Solar home electricity data v2.csv'.
    category : str
        Consumption category to extract. One of:
            'GC'  — general consumption from grid (kWh per interval)
            'GG'  — gross solar PV generation (kWh per interval)
            'CL'  — controlled load (kWh per interval)
        Default: 'GC'.
    min_days : int
        Minimum number of complete days required to include a customer.
        Default: 365 (retain only customers with a full year).

    Returns
    -------
    loads : np.ndarray, shape (N_HH, N_TIMESTEPS)
        Half-hourly load values in kWh for each retained household, ordered
        chronologically from 1 July 2012 00:30 to 30 June 2013 00:00.
    customer_ids : list[int]
        Customer IDs corresponding to each row in ``loads``.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    # Row 0 is a banner note; row 1 is the actual header
    df = pd.read_csv(csv_path, header=1, low_memory=False)

    # Filter to requested consumption category
    df = df[df["Consumption Category"] == category].copy()

    # Parse dates (format: DD/MM/YYYY)
    df["date_parsed"] = pd.to_datetime(df["date"], dayfirst=True)

    # Retain only customers with sufficient coverage
    days_per_customer = df.groupby("Customer")["date_parsed"].count()
    valid_customers = sorted(
        days_per_customer[days_per_customer >= min_days].index.tolist()
    )

    df = df[df["Customer"].isin(valid_customers)].copy()

    # Sort by customer then date for deterministic ordering
    df = df.sort_values(["Customer", "date_parsed"]).reset_index(drop=True)

    # Build canonical date index for the year
    date_index = pd.date_range("2012-07-01", periods=N_DAYS, freq="D")

    n_hh = len(valid_customers)
    loads = np.zeros((n_hh, N_TIMESTEPS), dtype=np.float32)

    for hh_idx, cust_id in enumerate(valid_customers):
        cust_df = df[df["Customer"] == cust_id].set_index("date_parsed")
        # Align to canonical date index; fill missing days with column median
        cust_df = cust_df.reindex(date_index)
        interval_data = cust_df[INTERVAL_COLS].values.astype(float)
        # Fill any NaN with the per-interval median across available days
        for col_i in range(N_INTERVALS):
            col_vals = interval_data[:, col_i]
            col_median = np.nanmedian(col_vals[~np.isnan(col_vals)]) if not np.all(np.isnan(col_vals)) else 0.0
            col_vals[np.isnan(col_vals)] = col_median
            interval_data[:, col_i] = col_vals
        # Flatten: row-major gives chronological half-hourly sequence
        loads[hh_idx, :] = interval_data.flatten()

    return loads, valid_customers


def dataset_summary(loads: np.ndarray, customer_ids: list[int]) -> dict:
    """
    Return a dictionary of key descriptive statistics for a loaded dataset.
    """
    n_hh = loads.shape[0]
    daily_kwh = loads.reshape(n_hh, N_DAYS, N_INTERVALS).sum(axis=2)
    feeder = loads.sum(axis=0)

    return {
        "n_households": n_hh,
        "n_timesteps": loads.shape[1],
        "n_days": N_DAYS,
        "mean_daily_kwh_per_hh": float(np.mean(daily_kwh)),
        "std_daily_kwh_per_hh": float(np.std(daily_kwh)),
        "min_daily_kwh": float(np.min(daily_kwh)),
        "max_daily_kwh": float(np.max(daily_kwh)),
        "mean_feeder_load_per_interval_kwh": float(np.mean(feeder)),
        "peak_feeder_load_kwh": float(np.max(feeder)),
    }
