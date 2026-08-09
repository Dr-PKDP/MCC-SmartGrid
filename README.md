# MCC Smart Grid Feasibility Study

**Companion code and data for:**

> Pramanik, P.K.D. (2025). Mobile Crowd Computing as a Sustainable Edge Computing Paradigm for Smart Grids. *[Journal TBD]*.

This repository contains the complete code, dataset, and results for the analytical feasibility assessment (Section 5.6 of the paper). The study evaluates whether Mobile Crowd Computing (MCC) can provide useful distribution-level grid monitoring using real residential smart meter data, and quantifies the operational energy cost advantage of MCC relative to a conventional dedicated-edge-node baseline.

---

## What this study does

Each of 299 residential households in the Ausgrid Solar Home dataset is treated as a potential MCC device. Its half-hourly general consumption (GC) reading is the proxy observation that an MCC-enabled smartphone in that household would contribute to the grid analytics layer — an indirect signal consistent with the paper's indirect sensing model (Section 5.1).

The MCC system estimates feeder-level load by aggregating contributions from a random subset of participating households using trust-weighted aggregation (Equation 5 of the paper), then scales from the sample to the full population. Performance is evaluated across five participation rates (10%–100%) with 200 Monte Carlo trials each.

---

## Key Results

| Participation rate | Devices | Normalised RMSE | Anomaly F1 | MCC daily energy | Energy ratio |
|---|---|---|---|---|---|
| 10% | 29 | 20.8 ± 2.9% | 0.376 ± 0.121 | 0.29 Wh | ~42,000× |
| 25% | 74 | 11.9 ± 1.2% | 0.564 ± 0.072 | 0.73 Wh | ~16,400× |
| 50% | 149 | 7.1 ± 0.9% | 0.649 ± 0.037 | 1.48 Wh | ~8,100× |
| 75% | 224 | 4.3 ± 0.5% | 0.672 ± 0.018 | 2.22 Wh | ~5,400× |
| 100% | 299 | 1.8 ± 0.2% | 0.682 ± 0.008 | 2.96 Wh | ~4,100× |

**Conventional 100-node edge baseline:** 12,000 Wh/day (5 W per node, continuous).

Energy figures use per-session smartphone FL energy of 2.97 J from Patterson et al. (2024).

---

## Repository Structure

```
├── data/
│   ├── README.md                          ← Dataset description and provenance
│   └── 2012-2013-Solar-home-electricity-data.csv   ← Ausgrid dataset (CC BY 4.0)
│
├── src/
│   ├── data_loader.py                     ← Load and preprocess Ausgrid CSV
│   ├── anomaly_detection.py               ← Anomaly injection and detection
│   ├── mcc_simulation.py                  ← Core MCC participation simulation (Eqs 1–5)
│   ├── energy_model.py                    ← Marginal energy cost model (Eq. 4)
│   └── visualization.py                   ← Figure generation
│
├── scripts/
│   ├── run_analysis.py                    ← Main script: real Ausgrid data
│   └── run_synthetic_baseline.py          ← Synthetic validation run
│
├── results/
│   ├── figures/                           ← Generated figures (PDF + PNG)
│   └── tables/                            ← Results tables (CSV)
│
├── paper/
│   └── section_5_6_draft.md              ← Paper section text (Section 5.6)
│
├── requirements.txt
├── LICENSE                                ← CC BY 4.0
└── CITATION.cff                           ← Machine-readable citation
```

---

## Reproducing the Results

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+ on Ubuntu 22.04 and macOS 13. No GPU required.

### 2. Run the real-data analysis (Ausgrid)

```bash
python scripts/run_analysis.py
```

This loads the Ausgrid 2012–2013 dataset, injects 40 synthetic high-load anomaly events, runs 200 Monte Carlo trials at each of five participation rates, saves results to `results/tables/results_real_data.csv`, and generates two figures in `results/figures/`.

Runtime: approximately 3–5 minutes on a modern laptop.

### 3. Run the synthetic baseline

```bash
python scripts/run_synthetic_baseline.py
```

Generates a synthetic dataset calibrated to the Ausgrid GC statistical properties (mean 15.3 kWh/day, std 9.7 kWh/day) and runs the identical simulation pipeline. Results are saved as `results_synthetic.csv` and figures as `fig_mcc_summary_synthetic.*`.

---

## Methodology

### Dataset

The Ausgrid Solar Home Electricity dataset contains half-hourly general consumption (GC) readings for 300 Sydney-area residential solar households over the period 1 July 2012 – 30 June 2013. Customer 2 (284 of 365 days) is excluded; 299 customers with complete records are retained.

### Anomaly injection

Synthetic high-load events representing voltage-stress conditions (EV charging surges, extreme demand peaks) are injected by scaling the load of 40–65% of households simultaneously by a factor of 3–5× for 2–4 consecutive intervals. Forty events are injected, producing 119 anomalous timesteps (0.7% of all timesteps).

Anomaly detection uses a time-of-day-aware z-score threshold: a timestep is flagged when the MCC aggregate estimate exceeds the per-half-hour-slot mean plus 4 standard deviations of the baseline feeder load. This approach accounts for the strong diurnal pattern in residential feeder loads; a global threshold would produce excessive false positives during natural evening peaks.

### MCC simulation (Equations 1–5)

For each participation rate and trial:

1. **Device selection (Eq. 3):** A random subset of N devices is drawn from the 299-household pool.
2. **Trust weights (Eq. 1):** Each device is assigned τᵢ ~ Beta(4, 2), representing a realistic population where most devices have moderate-to-high reliability. Weights are normalised to sum to 1.
3. **Aggregation (Eq. 5):** The trust-weighted mean of selected device loads is computed and scaled by 299/N to estimate the full-feeder load.
4. **Evaluation:** RMSE, MAE, and F1 score are computed against ground truth.

### Energy model (Eq. 4)

Per-session device energy: 2.97 J (Patterson et al., 2024, Table 3 — lightweight ~10K-parameter model on Android smartphone).  
Sessions per day: 12 (one per two hours during active-use periods, conservative).  
Conventional baseline: 100 nodes at 5 W continuous = 12,000 Wh/day.

---

## Dependencies

| Package | Purpose |
|---|---|
| numpy | Array operations, random sampling |
| pandas | CSV loading, date parsing |
| matplotlib | Figure generation |
| scipy | Smoothing (uniform_filter1d in visualization) |

---

## Citation

If you use this code or data, please cite both the paper and the Ausgrid dataset:

**Paper:**
```
Pramanik, P.K.D. (2025). Mobile Crowd Computing as a Sustainable Edge
Computing Paradigm for Smart Grids. [Journal TBD].
```

**Dataset:**
```
Ratnam, E.L., Weller, S.R., Kellett, C.M., and Murray, A.T. (2017).
Residential load and rooftop PV generation: an Australian distribution
network dataset. International Journal of Sustainable Energy, 36(8), 787–806.
https://doi.org/10.1080/14786451.2015.1100196
```

**Dataset source:**
```
Ausgrid. (2014). Solar Home Electricity Data.
https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data
License: CC BY 4.0
```

**Energy figures:**
```
Patterson, D., Gilbert, J.M., Gruteser, M., Robles, E., Sekar, K., Wei, Y.,
and Zhu, T. (2024). Energy and Emissions of Machine Learning on Smartphones
vs. the Cloud. Communications of the ACM, 67(2), 87–95.
```

---

## Contact

Pijush Kanti Dutta Pramanik  
School of Computer Applications and Technology, Galgotias University, India  
McWilliams School of Biomedical Informatics, UTHealth Houston, USA  
pijushjld@yahoo.co.in | ORCID: 0000-0001-9438-9309
