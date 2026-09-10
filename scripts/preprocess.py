import argparse
import os
import numpy as np
import pandas as pd

RAW_COLS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_cols = set(RAW_COLS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input is missing expected PaySim columns: {missing_cols}")
    return df


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Column-level null / dtype / range summary. Run BEFORE cleaning
    so the report reflects the raw data's actual problems."""
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "pct_missing": round(100 * s.isna().mean(), 3),
            "n_unique": int(s.nunique(dropna=True)),
        })
    rep = pd.DataFrame(rows)

    # a few targeted consistency checks beyond per-column nulls
    neg_amounts = int((df["amount"] < 0).sum())
    zero_amounts = int((df["amount"] == 0).sum())
    bad_balance_math = int((
        (df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]).abs() > 1.0
    ).sum())

    print("--- Data quality summary ---")
    print(rep.to_string(index=False))
    print(f"\nNegative amounts: {neg_amounts}")
    print(f"Zero amounts: {zero_amounts}")
    print(f"Rows where oldbalanceOrg - amount != newbalanceOrig (>$1 diff): {bad_balance_math}")
    return rep


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_missing_amount"] = df["amount"].isna()
    df["is_missing_orig_balance"] = df["oldbalanceOrg"].isna()
    n_before = len(df)
    df = df.dropna(subset=["amount"]).copy()
    print(f"Dropped {n_before - len(df)} rows with missing amount")
    for col in ["oldbalanceOrg", "oldbalanceDest"]:
        if df[col].isna().any():
            med = df[col].median()
            df[col] = df[col].fillna(med)
    df["type"] = df["type"].str.strip().str.upper()
    df["amount"] = df["amount"].round(2)
    base = pd.Timestamp("2024-01-01")
    df["timestamp"] = base + pd.to_timedelta(df["step"], unit="h")
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    df["is_night"] = df["hour_of_day"].between(1, 5)
    df["vendor_id"] = df["nameDest"]
    df["category"] = df["type"]
    df["is_merchant_dest"] = df["nameDest"].str.startswith("M")
    exact_dupes = df.duplicated(subset=RAW_COLS, keep="first")
    print(f"Exact duplicate rows: {exact_dupes.sum()}")
    df["is_exact_duplicate"] = exact_dupes
    df_sorted = df.sort_values(["nameOrig", "amount", "step"])
    same_group = (
        (df_sorted["nameOrig"] == df_sorted["nameOrig"].shift())
        & (df_sorted["amount"] == df_sorted["amount"].shift())
        & ((df_sorted["step"] - df_sorted["step"].shift()).abs() <= 1)
    )
    df_sorted["is_near_duplicate"] = same_group
    df = df_sorted.sort_index()
    print(f"Near-duplicate rows (same user+amount within 1hr): {df['is_near_duplicate'].sum()}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="./data/processed")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    raw = load_raw(args.input)
    print(f"Loaded {len(raw):,} raw rows\n")
    qrep = quality_report(raw)
    qrep.to_csv(os.path.join(args.outdir, "data_quality_report.csv"), index=False)
    clean_df = clean(raw)
    clean_df.to_csv(os.path.join(args.outdir, "clean_transactions.csv"), index=False)
    try:
        clean_df.to_parquet(os.path.join(args.outdir, "clean_transactions.parquet"), index=False)
    except ImportError:
        print("(skipped parquet export — install pyarrow for it: pip install pyarrow)")
    print(f"\nSaved {len(clean_df):,} cleaned rows -> {args.outdir}")
    print(clean_df[["timestamp", "type", "amount", "hour_of_day", "is_night",
                     "is_exact_duplicate", "is_near_duplicate"]].head())

if __name__ == "__main__":
    main()
