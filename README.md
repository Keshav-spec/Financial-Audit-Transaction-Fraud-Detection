# Financial Audit & Transaction Fraud Detection: Project Walkthrough

An end-to-end financial transaction audit and fraud detection pipeline. The system processes raw banking transactions, runs data quality audits, executes a multi-signal anomaly detection engine (rule-based + unsupervised machine learning), models the results into an analytics-ready star schema, and feeds a Power BI reporting suite.

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/c34b9d8b-265b-42e5-bb67-b7384eac854f" width="100%"></td>
    <td><img src="https://github.com/user-attachments/assets/08298f04-edec-4b5d-8188-66ae61f72748" width="100%"></td>
    <td><img src="https://github.com/user-attachments/assets/3a4386da-e939-4e66-ab95-4b23fa906f23" width="100%"></td>
  </tr>
</table>


---

## Repository Structure

```
audit/
├── data/
│   ├── raw/
│   │   └── synthetic_paysim.csv        # 50,145 records with injected anomalies
│   └── processed/
│       ├── audit_warehouse.db          # Indexed SQLite data warehouse
│       ├── clean_transactions.csv      # Preprocessed & deduplicated transactions
│       ├── clean_transactions.parquet  # High-performance parquet format
│       ├── data_quality_report.csv     # Column completeness & range audit
│       ├── dim_category.csv / .parquet # Category dimension (5 payment types)
│       ├── dim_date.csv / .parquet     # Calendar date dimension
│       ├── dim_vendor.csv / .parquet   # Merchant dimension with risk tiers
│       ├── fact_transactions.csv       # Star-schema fact table with 7 flags
│       └── flagged_transactions.csv    # Scored transaction records
├── sql/
│   └── schema_and_views.sql            # CTE staging, clean views, reporting views
├── scripts/
│   ├── preprocess.py                   # Datetime parsing, standardization, dedup
│   ├── anomaly_detection.py            # 7-signal anomaly engine & composite scoring
│   ├── export.py                       # Star-schema dimensional modeling & SQLite writer
│   ├── run_pipeline.py                 # Master CLI pipeline orchestrator
│   ├── generate_synthetic_data.py      # Controlled anomaly test fixture generator
│   ├── sample_paysim.py                # Memory-efficient stratified sampler
│   ├── test_sql.py                     # SQL views execution & test script
│   └── utils.py                        # Shared data utilities
├── dashboard/
│   ├── dax_measures.dax                # 20+ production DAX measures for Power BI
│   └── powerbi_setup_guide.md          # Dashboard layout & visual specs
├── notebooks/
│   └── 01_financial_audit_and_fraud_detection.ipynb  # Interactive EDA & model evaluation
├── tests/
│   └── test_pipeline.py                # Automated pytest suite (5 passing tests)
├── docs/
│   └── false_positives_tradeoffs.md    # Asymmetric cost matrix & interview guide
├── requirements.txt                    # Pinned project dependencies
└── README.md
```

---

## Pipeline Architecture

```
Raw transactions (PaySim / synthetic)
        |
Preprocessing & enrichment (preprocess.py)
        |
Data quality audit (data_quality_report.csv)
        |
Anomaly detection engine (anomaly_detection.py) — 7 signals -> composite score
        |
Star-schema export (export.py) -> CSV / Parquet / SQLite
        |
Power BI dashboard (dashboard/)
```

---

## Phase-by-Phase Implementation

### Phase 1: Data Acquisition & Environment Setup
- Configured Kaggle API integration for automated retrieval of the PaySim dataset (~6.3M rows).
- Added a synthetic generator (`generate_synthetic_data.py`) with controlled anomaly injections — missing values, exact duplicates, temporal near-duplicates, vendor spikes, velocity bursts, extreme amounts — for fast, repeatable verification without needing the full download.

### Phase 2: Preprocessing & Standardization (`scripts/preprocess.py`)
- Parsed the elapsed `step` counter into calendar datetimes (`timestamp`).
- Extracted temporal features: `hour_of_day`, `day_of_week`, `is_weekend`, `is_night` (1 AM-5 AM off-hours proxy).
- Standardized text casing and rounded numeric values to 2 decimal places.
- Handled missing values with explicit audit flags (`is_missing_amount`, `is_missing_orig_balance`) rather than silent drops.
- Implemented exact deduplication (`flag_exact_duplicate`) and near-deduplication (`flag_near_duplicate`: same origin user, destination merchant, and amount within a 2-hour window).

### Phase 3: Pre-Cleaning Data Quality Auditing (`data_quality_report.csv`)
- Automated column-level completeness report (null counts, null %, unique values, datatypes).
- Schema and domain-logic consistency checks:
  - Negative amount validation (amount < 0).
  - Account balance reconciliation (old balance - amount ~ new balance).
- Findings exported to `data_quality_report.csv`.

### Phase 4: Multi-Signal Anomaly Detection Engine (`scripts/anomaly_detection.py`)
Seven independent anomaly signals:
1. **Duplicates** (`flag_duplicate`) — exact and temporal near-duplicates.
2. **Category outliers** (`flag_large_transaction`) — category-segmented IQR (Q3 + 2.5 x IQR) and log z-scores.
3. **Unusual timing** (`flag_unusual_timing`) — global off-hours (1 AM-5 AM) and vendor-specific operating-hour distributions.
4. **Vendor spikes** (`flag_vendor_spike`) — rolling 7-day volume baseline (mean + 2.5 x std).
5. **Velocity bursts** (`flag_velocity`) — user transacting 3+ times within a 2-hour window.
6. **Multivariate outliers** (`flag_multivariate_outlier`) — unsupervised Isolation Forest on scaled multidimensional space (amount, hour, step, balance delta).
7. **Concentration risk** (`flag_concentration`) — Herfindahl-Hirschman Index (HHI >= 0.85).

**Composite scoring:**
- Normalized score from 0-100 based on signal weights.
- Segmented into risk tiers: CRITICAL (>=75), HIGH (>=45), MEDIUM (>=20), LOW (<20).
- Each flagged transaction carries a human-readable `flag_reasons` string for auditor explainability.

### Phase 5: Dimensional Star Schema & SQLite Data Warehouse (`scripts/export.py`)
- `fact_transactions` — granular transaction line items, surrogate keys, 7 anomaly flags, composite scores, risk levels, and audit reasons.
- `dim_vendor` — merchant master with brand names, cumulative transaction counts, total volume, flagged rates, and risk ratings (HIGH_RISK, MEDIUM_RISK, STANDARD).
- `dim_category` — payment type dimension with risk weightings.
- `dim_date` — calendar date dimension for time intelligence.
- All tables written to `data/processed/audit_warehouse.db` with indexes on foreign keys.

### Phase 6: Power BI Reporting Suite (`dashboard/`)
20+ DAX measures in `dashboard/dax_measures.dax` and a 5-page layout guide in `dashboard/powerbi_setup_guide.md`. Build status:

| Page | Contents | Status |
|---|---|---|
| 1. Executive Audit Overview | KPI cards (total transactions, total value, flagged value, anomaly %, high/critical risk count), transaction volume + anomaly % trend, risk-level donut | Built |
| 2. Vendor Risk & Counterparty Intelligence | Top suspicious vendors by flagged value, drill-through transaction table, vendor risk profile card | Built |
| 3. Transaction Trends & Outliers | Unusual-timing distribution by hour, amount vs. anomaly-score scatter by risk tier | Built |

### Phase 7: Documentation & Trade-off Analysis
- `README.md` — architecture, execution steps, thresholds (this file).
- `docs/false_positives_tradeoffs.md` — asymmetric cost matrices, customer friction, Tier-1 auditor triage, interview discussion framework.
- `notebooks/01_financial_audit_and_fraud_detection.ipynb` — interactive walkthrough of all 7 phases with plots and model evaluation.

---

## Verification & Test Results

### Automated Unit Tests (pytest)
```text
tests/test_pipeline.py::test_data_quality_report PASSED       [ 20%]
tests/test_pipeline.py::test_clean_pipeline PASSED            [ 40%]
tests/test_pipeline.py::test_duplicate_detection PASSED       [ 60%]
tests/test_pipeline.py::test_velocity_burst PASSED            [ 80%]
tests/test_pipeline.py::test_dimensional_generation PASSED    [100%]
============================== 5 passed in 3.01s ==============================
```

### End-to-End Pipeline Execution
`scripts/run_pipeline.py` on 50,145 synthetic transaction records:
```text
===========================================================================
 PIPELINE EXECUTION SUMMARY
===========================================================================
Total Processed Transactions: 50,095
Flagged Anomalies:            903 (1.80%)
High / Critical Risk Txns:    357
Total Execution Time:         6.88 seconds
===========================================================================
```
- 50 missing amounts identified and dropped cleanly with audit logs.
- 30 exact duplicates and 65 near-duplicates detected.
- All tables exported to CSV, Parquet, and the indexed SQLite database (`audit_warehouse.db`).

### SQL Data Warehouse Verification
- **Top risky vendor:** Walgreens Health (M59873), 74.55% flagged ratio.
- **Category risk breakdown:**
  - Wire & P2P Transfer: $40,365,261 volume, 528 flagged (5.24% anomaly rate).
  - Cash Withdrawal: $27,455,031 volume, 213 flagged (1.72% anomaly rate).
  - Retail & Merchant Payment: $5,804,356 volume, 88 flagged (0.50% anomaly rate).
- Flag explanations generated as clean, human-readable strings, e.g.: "High Amount vs Category; Abnormal Vendor Spike; High Velocity Burst; Multivariate Outlier; Unusual Timing / Night; Reported Fraud Ground Truth."

---

## Setup & Usage

```bash
# 1. Clone / open the project
cd audit

# 2. Create environment and install dependencies
pip install -r requirements.txt

# 3. (Optional) Download the real PaySim dataset
#    Requires ~/.kaggle/kaggle.json API token
kaggle datasets download -d ealaxi/paysim1 -p data/raw --unzip

#    Or skip the download and use the synthetic generator instead:
python scripts/generate_synthetic_data.py

# 4. Run the full pipeline
python scripts/run_pipeline.py

# 5. Run tests
pytest -q

# 6. Open the exported tables in Power BI
#    Point Power BI Desktop at data/processed/ (CSV/Parquet) or
#    data/processed/audit_warehouse.db, then follow
#    dashboard/powerbi_setup_guide.md and import dashboard/dax_measures.dax
```

---

## Next Steps
- Build Power BI page 4: Category Distribution & Concentration (spend treemap, HHI histogram).
- Build Power BI page 5: Data Quality & Pipeline Health (completeness report, confusion matrix, schema validation cards).
- Re-run the pipeline against the full PaySim download (6.3M rows) rather than the synthetic/sampled set, and refresh the dashboard.
