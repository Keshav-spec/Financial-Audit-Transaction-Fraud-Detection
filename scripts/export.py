"""
Phase 5: Star-Schema Dimensional Modeling & Export
Transforms processed, flagged transaction data into a formal Star Schema for Power BI and SQL databases:
- fact_transactions
- dim_vendor
- dim_category
- dim_date
- data_quality_summary

Exports to Parquet, CSV, and writes tables directly into a local SQLite database (audit_warehouse.db).
"""

import os
import argparse
import sqlite3
import numpy as np
import pandas as pd
from faker import Faker


def generate_dim_category(categories: list) -> pd.DataFrame:
    """Build dim_category with descriptive attributes."""
    category_meta = {
        "PAYMENT": {
            "name": "Retail & Merchant Payment",
            "desc": "Consumer point-of-sale and online merchant checkout transactions",
            "risk_weight": "Low",
        },
        "TRANSFER": {
            "name": "Wire & P2P Transfer",
            "desc": "Direct electronic account-to-account funds transfers",
            "risk_weight": "High",
        },
        "CASH_OUT": {
            "name": "Cash Withdrawal",
            "desc": "Disbursement of physical currency via agent or ATM",
            "risk_weight": "High",
        },
        "DEBIT": {
            "name": "Debit / POS Card",
            "desc": "Direct debit from bank account or debit card processing",
            "risk_weight": "Low",
        },
        "CASH_IN": {
            "name": "Deposit & Funding",
            "desc": "Inflow of funds into customer wallet or bank account",
            "risk_weight": "Low",
        },
    }
    
    rows = []
    for cat in sorted(set(categories)):
        meta = category_meta.get(cat, {
            "name": cat.title(),
            "desc": f"General {cat} transaction",
            "risk_weight": "Medium"
        })
        rows.append({
            "category_id": cat,
            "category_name": meta["name"],
            "description": meta["desc"],
            "category_risk_profile": meta["risk_weight"]
        })
    return pd.DataFrame(rows)


def generate_dim_vendor(df: pd.DataFrame, fake_seed: int = 42) -> pd.DataFrame:
    """Build dim_vendor with realistic corporate naming, aggregate volumes, and risk profiles."""
    fake = Faker()
    Faker.seed(fake_seed)
    np.random.seed(fake_seed)
    
    vendor_stats = df.groupby("vendor_id").agg(
        total_transactions=("amount", "count"),
        total_volume=("amount", "sum"),
        avg_transaction_amount=("amount", "mean"),
        flagged_count=("is_flagged", "sum") if "is_flagged" in df.columns else ("amount", lambda x: 0),
        is_merchant=("is_merchant_dest", "first") if "is_merchant_dest" in df.columns else ("vendor_id", lambda s: s.str.startswith("M"))
    ).reset_index()
    
    vendor_stats["flagged_rate_pct"] = (vendor_stats["flagged_count"] / vendor_stats["total_transactions"].replace(0, 1) * 100).round(2)
    
    # Generate realistic vendor names
    vendor_names = []
    merchant_brands = [
        "Amazon Web Store", "Starbucks Coffee", "Shell Oil Station", "Walmart Supercenter",
        "Target Retail", "Apple Services", "Uber Technologies", "Delta Air Lines",
        "Costco Wholesale", "Home Depot", "Netflix Streaming", "McDonald's Global",
        "Best Buy Electronics", "CVS Pharmacy", "Walgreens Health", "Chevron Gas",
        "Kroger Groceries", "Nike Direct", "FedEx Freight", "Stripe Online Merchant"
    ]
    
    unique_vendors = vendor_stats["vendor_id"].values
    for vid in unique_vendors:
        if str(vid).startswith("M"):
            # Select from realistic brand pool or generate company name
            idx = abs(hash(vid)) % len(merchant_brands)
            vendor_names.append(f"{merchant_brands[idx]} ({vid[:6]})")
        else:
            vendor_names.append(f"Private Account {vid}")
            
    vendor_stats["vendor_name"] = vendor_names
    
    # Assign vendor risk rating
    conditions = [
        vendor_stats["flagged_rate_pct"] >= 15.0,
        vendor_stats["flagged_rate_pct"] >= 5.0,
    ]
    vendor_stats["vendor_risk_tier"] = np.select(conditions, ["HIGH_RISK", "MEDIUM_RISK"], default="STANDARD")
    
    return vendor_stats


def generate_dim_date(df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
    """Build dim_date for standard Power BI time intelligence."""
    df[time_col] = pd.to_datetime(df[time_col])
    min_date = df[time_col].dt.date.min()
    max_date = df[time_col].dt.date.max()
    
    date_range = pd.date_range(min_date, max_date, freq="D")
    dim_date = pd.DataFrame({"full_date": date_range})
    
    dim_date["date_key"] = dim_date["full_date"].dt.strftime("%Y%m%d").astype(int)
    dim_date["year"] = dim_date["full_date"].dt.year
    dim_date["quarter"] = "Q" + dim_date["full_date"].dt.quarter.astype(str)
    dim_date["month"] = dim_date["full_date"].dt.month
    dim_date["month_name"] = dim_date["full_date"].dt.strftime("%B")
    dim_date["day"] = dim_date["full_date"].dt.day
    dim_date["day_of_week"] = dim_date["full_date"].dt.dayofweek
    dim_date["day_name"] = dim_date["full_date"].dt.strftime("%A")
    dim_date["is_weekend"] = dim_date["day_of_week"].isin([5, 6])
    
    return dim_date


def generate_fact_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Format and align the fact table with surrogate keys and audit metrics."""
    fact = df.copy()
    
    if "transaction_id" not in fact.columns:
        fact.insert(0, "transaction_id", ["TXN_" + str(i).zfill(8) for i in range(1, len(fact) + 1)])
        
    fact["timestamp"] = pd.to_datetime(fact["timestamp"])
    fact["date_key"] = fact["timestamp"].dt.strftime("%Y%m%d").astype(int)
    
    # Standard column mapping
    rename_cols = {
        "nameOrig": "user_orig_id",
        "oldbalanceOrg": "old_balance_orig",
        "newbalanceOrig": "new_balance_orig",
        "oldbalanceDest": "old_balance_dest",
        "newbalanceDest": "new_balance_dest",
        "isFraud": "is_fraud_ground_truth"
    }
    fact = fact.rename(columns={k: v for k, v in rename_cols.items() if k in fact.columns})
    
    # Ensure boolean flags are clean integer/boolean
    bool_flags = [
        "is_weekend", "is_night", "flag_exact_duplicate", "flag_near_duplicate",
        "flag_duplicate", "flag_large_transaction", "flag_unusual_timing",
        "flag_vendor_spike", "flag_velocity", "flag_multivariate_outlier",
        "flag_concentration", "is_flagged"
    ]
    for bf in bool_flags:
        if bf in fact.columns:
            fact[bf] = fact[bf].fillna(False).astype(bool)
            
    return fact


def export_to_sqlite(tables: dict, db_path: str):
    """Write star-schema tables to a local SQLite database for SQL practice."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    print(f"-> Writing tables to SQLite database: {db_path}...")
    for table_name, df in tables.items():
        # SQLite cannot store datetime objects natively, convert to string
        df_sql = df.copy()
        for col in df_sql.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
            df_sql[col] = df_sql[col].astype(str)
        df_sql.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"   [SQLite] Loaded {table_name}: {len(df_sql):,} rows")
        
    # Create indexes for fast joins
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_date ON fact_transactions(date_key);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_vendor ON fact_transactions(vendor_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_cat ON fact_transactions(category);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fact_flag ON fact_transactions(is_flagged);")
    conn.commit()
    conn.close()
    print("-> SQLite database ready with optimized indexes.")


def run_export_pipeline(input_path: str, outdir: str, db_name: str = "audit_warehouse.db"):
    """Execute complete star-schema export workflow."""
    os.makedirs(outdir, exist_ok=True)
    
    print(f"-> Loading flagged transactions from {input_path}...")
    if input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
        
    print(f"Loaded {len(df):,} transactions for dimensional modeling.")
    
    # 1. Dimension Tables
    print("-> Building dim_category...")
    dim_category = generate_dim_category(df["category"].unique())
    
    print("-> Building dim_vendor...")
    dim_vendor = generate_dim_vendor(df)
    
    print("-> Building dim_date...")
    dim_date = generate_dim_date(df)
    
    print("-> Building fact_transactions...")
    fact_transactions = generate_fact_transactions(df)
    
    # 2. Data Quality Summary
    dq_path = os.path.join(outdir, "data_quality_report.csv")
    if os.path.exists(dq_path):
        data_quality_summary = pd.read_csv(dq_path)
    else:
        data_quality_summary = pd.DataFrame([{"status": "Completed", "records_evaluated": len(df)}])
        
    tables = {
        "fact_transactions": fact_transactions,
        "dim_vendor": dim_vendor,
        "dim_category": dim_category,
        "dim_date": dim_date,
        "data_quality_summary": data_quality_summary
    }
    
    # 3. Export to CSV & Parquet
    print("-> Exporting CSV and Parquet files for Power BI...")
    for name, tbl in tables.items():
        csv_file = os.path.join(outdir, f"{name}.csv")
        parquet_file = os.path.join(outdir, f"{name}.parquet")
        tbl.to_csv(csv_file, index=False)
        try:
            tbl.to_parquet(parquet_file, index=False)
        except Exception as e:
            print(f"   Note: parquet export skipped for {name}: {e}")
        print(f"   Exported {name} -> {csv_file}")
        
    # 4. Export to SQLite
    sqlite_db_path = os.path.join(outdir, db_name)
    export_to_sqlite(tables, sqlite_db_path)
    
    print("\n[SUCCESS] Phase 5 Export Complete! All files ready for Power BI.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export transactions to Star-Schema for Power BI and SQLite.")
    parser.add_argument("--input", default="./data/processed/flagged_transactions.parquet", help="Path to flagged transactions")
    parser.add_argument("--outdir", default="./data/processed", help="Output directory")
    args = parser.parse_args()
    
    run_export_pipeline(args.input, args.outdir)
