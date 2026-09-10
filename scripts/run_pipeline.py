"""
Master Execution Pipeline: Financial Audit & Fraud Detection
Orchestrates Phase 2 through Phase 5 in an end-to-end automated sequence:
1. Data Preprocessing & Cleaning (preprocess.py)
2. Quality Reporting (data_quality_report.csv)
3. Anomaly Engine & Multi-signal Scoring (anomaly_detection.py)
4. Star-Schema Modeling & SQLite / Power BI Export (export.py)
"""

import os
import sys
import time
import argparse
import pandas as pd

# Import local modules
from preprocess import load_raw, quality_report, clean
from anomaly_detection import run_anomaly_pipeline
from export import run_export_pipeline


def run_full_pipeline(input_file: str, processed_dir: str = "./data/processed", db_name: str = "audit_warehouse.db"):
    start_time = time.time()
    os.makedirs(processed_dir, exist_ok=True)
    
    print("=" * 75)
    print(" FINANCIAL AUDIT & FRAUD DETECTION DATA PIPELINE")
    print("=" * 75)
    print(f"Input Data:      {input_file}")
    print(f"Output Folder:   {processed_dir}")
    print(f"SQLite Database: {os.path.join(processed_dir, db_name)}")
    print("=" * 75)
    
    # -------------------------------------------------------------
    # Step 1: Preprocessing & Data Quality Audit
    # -------------------------------------------------------------
    print("\n[PHASE 1-3] Loading Raw Data & Performing Data Quality Audit...")
    raw_df = load_raw(input_file)
    print(f"Loaded {len(raw_df):,} raw records.")
    
    dq_rep = quality_report(raw_df)
    dq_csv = os.path.join(processed_dir, "data_quality_report.csv")
    dq_rep.to_csv(dq_csv, index=False)
    print(f"Data Quality Report saved to {dq_csv}")
    
    print("\n[PHASE 2] Cleaning & Feature Engineering...")
    clean_df = clean(raw_df)
    clean_csv = os.path.join(processed_dir, "clean_transactions.csv")
    clean_parquet = os.path.join(processed_dir, "clean_transactions.parquet")
    clean_df.to_csv(clean_csv, index=False)
    try:
        clean_df.to_parquet(clean_parquet, index=False)
    except Exception as e:
        print(f"Parquet export skipped: {e}")
    print(f"Cleaned dataset saved ({len(clean_df):,} rows).")
    
    # -------------------------------------------------------------
    # Step 2: Anomaly Detection Engine
    # -------------------------------------------------------------
    print("\n[PHASE 4] Executing Anomaly Detection Engine & Composite Scoring...")
    flagged_df = run_anomaly_pipeline(clean_df)
    flagged_parquet = os.path.join(processed_dir, "flagged_transactions.parquet")
    flagged_csv = os.path.join(processed_dir, "flagged_transactions.csv")
    flagged_df.to_csv(flagged_csv, index=False)
    try:
        flagged_df.to_parquet(flagged_parquet, index=False)
    except Exception as e:
        print(f"Parquet export skipped: {e}")
        
    # -------------------------------------------------------------
    # Step 3: Star-Schema Dimensional Modeling & Export
    # -------------------------------------------------------------
    print("\n[PHASE 5] Transforming to Star-Schema for Power BI & SQL Database...")
    run_export_pipeline(flagged_csv, processed_dir, db_name)
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print(" PIPELINE EXECUTION SUMMARY")
    print("=" * 75)
    print(f"Total Processed Transactions: {len(flagged_df):,}")
    print(f"Flagged Anomalies:            {flagged_df['is_flagged'].sum():,} ({flagged_df['is_flagged'].mean()*100:.2f}%)")
    print(f"High / Critical Risk Txns:    {(flagged_df['risk_level'].isin(['HIGH', 'CRITICAL'])).sum():,}")
    print(f"Total Execution Time:         {elapsed:.2f} seconds")
    print("=" * 75)
    print("All artifacts exported successfully and ready for Power BI!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete fraud detection and audit pipeline.")
    parser.add_argument("--input", required=True, help="Path to input raw CSV file")
    parser.add_argument("--outdir", default="./data/processed", help="Output directory")
    parser.add_argument("--dbname", default="audit_warehouse.db", help="SQLite database filename")
    args = parser.parse_args()
    
    run_full_pipeline(args.input, args.outdir, args.dbname)
