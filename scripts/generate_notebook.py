"""
Generate an interactive, comprehensive Jupyter Notebook covering all 7 phases:
1. Setup & Environment
2. Data Quality Reporting & Inspection
3. Preprocessing, Standardization & Feature Engineering
4. Anomaly Detection Engine (Rule-based & ML Models)
5. Model Evaluation (Precision, Recall, ROC-AUC, Confusion Matrix)
6. Star-Schema Modeling & SQLite Data Warehouse
7. Power BI Readiness & Insights
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# Financial Audit & Transaction Anomaly Detection
### End-to-End Enterprise Data Pipeline & Fraud Intelligence
---
This notebook provides a complete interactive walkthrough of the **7-Phase Financial Audit & Anomaly Detection System**:
1. **Phase 1:** Data Ingestion & Schema Exploration
2. **Phase 2:** Cleaning, Standardization, Datetime Parsing & Deduplication
3. **Phase 3:** Pre-pipeline Data Quality Auditing (Schema, Range, Math Consistency)
4. **Phase 4:** Multi-Signal Anomaly Detection Engine (IQR, Z-Score, Timing, Spikes, Velocity, Isolation Forest, HHI)
5. **Phase 5:** Composite Scoring, Risk Tiers & Model Evaluation
6. **Phase 6:** Dimensional Star-Schema Transformation (`fact_transactions`, `dim_vendor`, `dim_category`, `dim_date`)
7. **Phase 7:** SQLite Warehouse & Power BI Integration
"""))

# Cell: Imports
cells.append(nbf.v4.new_code_cell("""import os
import sys
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score

# Set visualization styles
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11

print("Environment configured successfully.")"""))

# Section: Load Raw Data & Data Quality Checks
cells.append(nbf.v4.new_markdown_cell("""## 1. Data Ingestion & Quality Audit (Phase 1 & 3)
Before executing cleaning or anomaly detection, we run a non-destructive audit of raw records to identify missing values, negative amounts, and balance discrepancies.
"""))

cells.append(nbf.v4.new_code_cell("""# Check available data files
data_raw_path = "../data/raw/synthetic_paysim.csv"
if not os.path.exists(data_raw_path):
    # Fallback to current directory path if running directly from project root
    data_raw_path = "./data/raw/synthetic_paysim.csv"

raw_df = pd.read_csv(data_raw_path)
print(f"Loaded {len(raw_df):,} raw transaction records.")
raw_df.head()"""))

cells.append(nbf.v4.new_code_cell("""# Data Quality Summary Report
dq_summary = []
for col in raw_df.columns:
    dq_summary.append({
        "column": col,
        "dtype": str(raw_df[col].dtype),
        "missing_count": raw_df[col].isna().sum(),
        "pct_missing": round(100 * raw_df[col].isna().mean(), 3),
        "unique_values": raw_df[col].nunique()
    })
dq_df = pd.DataFrame(dq_summary)

# Targeted consistency checks
neg_amounts = (raw_df["amount"] < 0).sum()
bad_balance_math = ((raw_df["oldbalanceOrg"] - raw_df["amount"] - raw_df["newbalanceOrig"]).abs() > 1.0).sum()

print(f"Negative Amounts Detected: {neg_amounts}")
print(f"Balance Math Inconsistencies: {bad_balance_math:,}")
dq_df"""))

# Section: Preprocessing & Cleaning
cells.append(nbf.v4.new_markdown_cell("""## 2. Preprocessing & Feature Engineering (Phase 2)
We parse the elapsed `step` counter into calendar datetimes, derive cyclical time features, standardize categories, and flag exact/near duplicates.
"""))

cells.append(nbf.v4.new_code_cell("""# Add datetime and time-of-day features
df_clean = raw_df.dropna(subset=["amount"]).copy()
base_date = pd.Timestamp("2024-01-01")
df_clean["timestamp"] = base_date + pd.to_timedelta(df_clean["step"], unit="h")
df_clean["hour_of_day"] = df_clean["timestamp"].dt.hour
df_clean["day_of_week"] = df_clean["timestamp"].dt.dayofweek
df_clean["is_weekend"] = df_clean["day_of_week"].isin([5, 6])
df_clean["is_night"] = df_clean["hour_of_day"].between(1, 5)

# Standardize vendor & category
df_clean["vendor_id"] = df_clean["nameDest"]
df_clean["category"] = df_clean["type"].str.strip().str.upper()
df_clean["is_merchant_dest"] = df_clean["nameDest"].str.startswith("M")

print(f"Cleaned dataset: {len(df_clean):,} valid records.")
df_clean[["timestamp", "category", "amount", "hour_of_day", "is_weekend", "is_night"]].head()"""))

# Section: Anomaly Detection Visuals
cells.append(nbf.v4.new_markdown_cell("""## 3. Anomaly Detection & Scoring (Phase 4 & 5)
Load the flagged transaction dataset produced by the anomaly engine and inspect the distribution of risk scores and detection flags.
"""))

cells.append(nbf.v4.new_code_cell("""flagged_path = "../data/processed/flagged_transactions.parquet"
if not os.path.exists(flagged_path):
    flagged_path = "./data/processed/flagged_transactions.parquet"
    if not os.path.exists(flagged_path):
        flagged_path = "./data/processed/flagged_transactions.csv"

flagged_df = pd.read_parquet(flagged_path) if flagged_path.endswith(".parquet") else pd.read_csv(flagged_path)
print(f"Total flagged transactions: {flagged_df['is_flagged'].sum():,} / {len(flagged_df):,} ({flagged_df['is_flagged'].mean()*100:.2f}%)")"""))

cells.append(nbf.v4.new_code_cell("""# Visualization: Distribution of Anomaly Scores by Risk Level
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

sns.histplot(data=flagged_df, x="anomaly_score", hue="risk_level", multiple="stack", bins=30, ax=axes[0], palette="turbo")
axes[0].set_title("Distribution of Transaction Anomaly Scores", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Anomaly Score (0 - 100)")
axes[0].set_ylabel("Transaction Count")

# Anomaly flag breakdown
flag_cols = [
    "flag_duplicate", "flag_large_transaction", "flag_unusual_timing",
    "flag_vendor_spike", "flag_velocity", "flag_multivariate_outlier", "flag_concentration"
]
flag_counts = flagged_df[flag_cols].sum()
flag_counts.index = [c.replace("flag_", "").replace("_", " ").title() for c in flag_counts.index]

flag_counts.sort_values().plot(kind="barh", ax=axes[1], color="#2563EB")
axes[1].set_title("Incident Count by Anomaly Detection Signal", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Flagged Incidents")

plt.tight_layout()
plt.show()"""))

# Section: Model Validation & Confusion Matrix
cells.append(nbf.v4.new_markdown_cell("""## 4. Ground-Truth Validation & Confusion Matrix
Using PaySim's ground truth `isFraud` column, we evaluate the system's precision, recall, and false positive trade-offs.
"""))

cells.append(nbf.v4.new_code_cell("""if "is_fraud_ground_truth" in flagged_df.columns:
    y_true = flagged_df["is_fraud_ground_truth"]
elif "isFraud" in flagged_df.columns:
    y_true = flagged_df["isFraud"]
else:
    y_true = np.zeros(len(flagged_df))

y_pred = flagged_df["is_flagged"].astype(int)

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Normal", "Fraud / Anomaly"]))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False,
            xticklabels=["Predicted Clean", "Predicted Flagged"],
            yticklabels=["Actual Clean", "Actual Fraud"])
plt.title("Audit Anomaly Engine - Confusion Matrix", fontweight="bold")
plt.ylabel("Ground Truth")
plt.xlabel("Pipeline Decision")
plt.show()"""))

# Section: Querying SQLite Data Warehouse
cells.append(nbf.v4.new_markdown_cell("""## 5. Star-Schema Data Warehouse & SQL Views (Phase 5 & 6)
Query the generated SQLite database (`audit_warehouse.db`) to simulate the analytics queries powering the Power BI dashboard.
"""))

cells.append(nbf.v4.new_code_cell("""db_path = "../data/processed/audit_warehouse.db"
if not os.path.exists(db_path):
    db_path = "./data/processed/audit_warehouse.db"

conn = sqlite3.connect(db_path)

# Top Suspicious Vendors
query_vendors = \"\"\"
SELECT 
    v.vendor_name,
    v.vendor_risk_tier,
    COUNT(f.transaction_id) AS total_txns,
    ROUND(SUM(f.amount), 2) AS total_volume,
    SUM(CASE WHEN f.is_flagged = 1 THEN 1 ELSE 0 END) AS flagged_count,
    ROUND(100.0 * SUM(CASE WHEN f.is_flagged = 1 THEN 1 ELSE 0 END) / COUNT(f.transaction_id), 2) AS flagged_pct
FROM dim_vendor v
JOIN fact_transactions f ON v.vendor_id = f.vendor_id
GROUP BY v.vendor_name, v.vendor_risk_tier
HAVING COUNT(f.transaction_id) >= 10
ORDER BY flagged_count DESC, flagged_pct DESC
LIMIT 10;
\"\"\"
top_vendors = pd.read_sql(query_vendors, conn)
print("Top 10 Suspicious Vendors:")
top_vendors"""))

cells.append(nbf.v4.new_code_cell("""# Daily Trend Summary
query_trend = \"\"\"
SELECT 
    d.full_date,
    COUNT(f.transaction_id) AS total_txns,
    SUM(CASE WHEN f.is_flagged = 1 THEN 1 ELSE 0 END) AS flagged_txns,
    ROUND(SUM(CASE WHEN f.is_flagged = 1 THEN f.amount ELSE 0 END), 2) AS flagged_volume
FROM dim_date d
JOIN fact_transactions f ON d.date_key = f.date_key
GROUP BY d.full_date
ORDER BY d.full_date;
\"\"\"
daily_trend = pd.read_sql(query_trend, conn)
daily_trend["full_date"] = pd.to_datetime(daily_trend["full_date"])

plt.figure(figsize=(14, 5))
plt.plot(daily_trend["full_date"], daily_trend["total_txns"], label="Total Daily Volume", color="#64748B", lw=2)
plt.plot(daily_trend["full_date"], daily_trend["flagged_txns"], label="Flagged Anomalies", color="#DC2626", lw=2.5, marker="o")
plt.title("Daily Transaction Activity & Flagged Anomalies", fontsize=14, fontweight="bold")
plt.xlabel("Date")
plt.ylabel("Transaction Volume")
plt.legend()
plt.tight_layout()
plt.show()

conn.close()"""))

nb.cells = cells

notebook_path = "c:/Users/vinay/Desktop/Projects/data science/audit/notebooks/01_financial_audit_and_fraud_detection.ipynb"
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Jupyter Notebook generated at {notebook_path}")
