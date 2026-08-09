# Dataset: Ausgrid Solar Home Electricity 2012–2013

## Source

**Provider:** Ausgrid (Australian electricity distribution network operator, Sydney NSW)  
**URL:** https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data  
**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)  
**Period:** 1 July 2012 – 30 June 2013 (full year)

## File

| File | Size | Description |
|---|---|---|
| `2012-2013-Solar-home-electricity-data.csv` | ~61 MB | Half-hourly electricity data for 300 solar households |

## Structure

The CSV has a single-line header preceded by one title row. Skip the first row when loading; use the second row as column headers.

**Metadata columns (5):**

| Column | Description |
|---|---|
| `Customer` | Anonymous customer ID (1–300) |
| `Generator Capacity` | Installed PV system size (kW) |
| `Postcode` | NSW postcode (anonymised area) |
| `Consumption Category` | GC, GG, or CL (see below) |
| `date` | Date in DD/MM/YYYY format |

**Interval columns (48):**  
Columns labelled `0:30`, `1:00`, `1:30`, ..., `0:00` — each holding the energy consumed or generated in that 30-minute interval (kWh). The sequence runs from 00:30 through 00:00 (midnight), Eastern Standard Time.

**Trailing column:**  
`Row Quality` — Ausgrid data quality indicator (usually empty).

## Consumption Categories

| Code | Meaning |
|---|---|
| `GC` | General consumption — electricity drawn from the grid (kWh) |
| `GG` | Gross generation — solar PV output (kWh) |
| `CL` | Controlled load — separately metered tariff load (e.g., hot water) |

This study uses **GC only** (household electricity drawn from the grid), consistent with using household load as a proxy signal for distribution-level grid analytics.

## Dataset Characteristics

| Metric | Value |
|---|---|
| Households | 300 (299 with complete 365 days) |
| Days | 365 (1 Jul 2012 – 30 Jun 2013) |
| Intervals per day | 48 (30-minute) |
| Total timesteps per household | 17,520 |
| Mean daily GC per household | 15.3 kWh |
| Std of daily GC | 9.7 kWh |
| Missing GC values | 0 (in retained households) |

## Data Quality Notes

Customer 2 has 284 of 365 days and is excluded by the data loader (`min_days=365`). All other 299 customers have complete records with no missing half-hourly values in the GC category. The dataset was quality-checked by Ausgrid; see the original data notes PDF for details.

## How to Load

```python
from src.data_loader import load_ausgrid

loads, customer_ids = load_ausgrid(
    "data/2012-2013-Solar-home-electricity-data.csv",
    category="GC",
    min_days=365,
)
# loads: np.ndarray, shape (299, 17520) — kWh per half-hourly interval
# customer_ids: list of 299 customer IDs
```

## Citation

If you use this dataset, please cite:

> Ratnam, E.L., Weller, S.R., Kellett, C.M., and Murray, A.T. (2017).
> Residential load and rooftop PV generation: an Australian distribution network dataset.
> *International Journal of Sustainable Energy*, 36(8), 787–806.
> https://doi.org/10.1080/14786451.2015.1100196

And acknowledge the data source:
> Ausgrid. (2014). Solar Home Electricity Data. Retrieved from
> https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data
> Licensed under CC BY 4.0.
