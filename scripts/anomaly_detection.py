"""
Phase 4: Anomaly Detection Engine
Computes granular rule-based and machine-learning anomaly signals:
1. Exact and near duplicates
2. Category-relative amount outliers (IQR / z-score)
3. Unusual timing (off-hours & vendor-specific operating hours)
4. Vendor abnormal spike (rolling window baseline)
5. User transaction velocity
6. Multivariate outliers using Isolation Forest
7. Concentration risk using Herfindahl-Hirschman Index (HHI)
Combines all flags into a composite anomaly_score, risk_level, and is_flagged indicator.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


def detect_duplicates(df: pd.DataFrame, time_col: str = "step") -> pd.DataFrame:
    """Detect exact and near duplicate transactions."""
    df = df.copy()
    raw_cols = ["step", "type", "amount", "nameOrig", "nameDest"]
    subset_cols = [c for c in raw_cols if c in df.columns]
    
    # Exact duplicates
    df["flag_exact_duplicate"] = df.duplicated(subset=subset_cols, keep="first")
    
    # Near-duplicates: same user, same amount, same destination within 2 hours
    df_sorted = df.sort_values(["nameOrig", "nameDest", "amount", time_col]).reset_index(drop=True)
    same_tuple = (
        (df_sorted["nameOrig"] == df_sorted["nameOrig"].shift(1))
        & (df_sorted["nameDest"] == df_sorted["nameDest"].shift(1))
        & (df_sorted["amount"] == df_sorted["amount"].shift(1))
        & ((df_sorted[time_col] - df_sorted[time_col].shift(1)).abs() <= 2)
    )
    df_sorted["flag_near_duplicate"] = same_tuple
    df = df_sorted.sort_index()
    
    df["flag_duplicate"] = df["flag_exact_duplicate"] | df["flag_near_duplicate"]
    return df


def detect_large_transactions(df: pd.DataFrame, category_col: str = "category", amount_col: str = "amount") -> pd.DataFrame:
    """Detect category-specific amount outliers using IQR and log-z-score."""
    df = df.copy()
    
    def calc_iqr_outliers(group):
        q25 = group[amount_col].quantile(0.25)
        q75 = group[amount_col].quantile(0.75)
        iqr = q75 - q25
        cutoff = q75 + (2.5 * iqr) if iqr > 0 else group[amount_col].mean() + 3 * group[amount_col].std()
        return group[amount_col] > cutoff

    df["flag_large_transaction"] = df.groupby(category_col, group_keys=False).apply(
        calc_iqr_outliers, include_groups=False
    ).fillna(False).astype(bool)
    
    return df


def detect_unusual_timing(df: pd.DataFrame, hour_col: str = "hour_of_day", vendor_col: str = "vendor_id") -> pd.DataFrame:
    """Detect transactions outside regular business hours and outside vendor standard hours."""
    df = df.copy()
    if hour_col not in df.columns and "timestamp" in df.columns:
        df[hour_col] = pd.to_datetime(df["timestamp"]).dt.hour
        
    # Global night hours: 1am - 5am
    global_night = df[hour_col].between(1, 5)
    
    # Vendor-specific profile: if vendor has >= 10 transactions, check if outside 5th-95th percentile hours
    vendor_counts = df[vendor_col].value_counts()
    active_vendors = vendor_counts[vendor_counts >= 10].index
    
    vendor_hours = df[df[vendor_col].isin(active_vendors)].groupby(vendor_col)[hour_col].agg(
        p05=lambda x: x.quantile(0.05),
        p95=lambda x: x.quantile(0.95)
    ).reset_index()
    
    merged = df.merge(vendor_hours, on=vendor_col, how="left")
    vendor_atypical = (
        merged["p05"].notna() & 
        ((merged[hour_col] < merged["p05"]) | (merged[hour_col] > merged["p95"]))
    )
    
    df["flag_unusual_timing"] = (global_night | vendor_atypical).astype(bool)
    return df


def detect_vendor_spikes(df: pd.DataFrame, vendor_col: str = "vendor_id", step_col: str = "step") -> pd.DataFrame:
    """Detect sudden spikes in vendor transaction count/volume vs rolling baseline."""
    df = df.copy()
    # Bucket transactions into daily intervals (24 steps = 1 day)
    df["day_bucket"] = (df[step_col] // 24).astype(int)
    
    daily_vendor = df.groupby([vendor_col, "day_bucket"]).agg(
        daily_count=("amount", "count"),
        daily_volume=("amount", "sum")
    ).reset_index()
    
    daily_vendor = daily_vendor.sort_values([vendor_col, "day_bucket"])
    daily_vendor["rolling_count_mean"] = daily_vendor.groupby(vendor_col)["daily_count"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=2).mean()
    )
    daily_vendor["rolling_count_std"] = daily_vendor.groupby(vendor_col)["daily_count"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=2).std()
    )
    
    count_threshold = daily_vendor["rolling_count_mean"] + (2.5 * daily_vendor["rolling_count_std"].fillna(0))
    daily_vendor["is_vendor_spike_day"] = (
        (daily_vendor["daily_count"] > 5) & 
        (daily_vendor["daily_count"] > count_threshold)
    )
    
    spike_days = daily_vendor[daily_vendor["is_vendor_spike_day"]][[vendor_col, "day_bucket"]]
    spike_days["flag_vendor_spike"] = True
    
    df = df.merge(spike_days, on=[vendor_col, "day_bucket"], how="left")
    df["flag_vendor_spike"] = df["flag_vendor_spike"].fillna(False).astype(bool)
    df = df.drop(columns=["day_bucket"])
    return df


def detect_user_velocity(df: pd.DataFrame, user_col: str = "nameOrig", step_col: str = "step") -> pd.DataFrame:
    """Flag users transacting >= 3 times within a short 2-hour window."""
    df = df.copy()
    df_sorted = df.sort_values([user_col, step_col]).reset_index(drop=True)
    
    # 2-transaction lag within 2 steps
    lag2_match = (
        (df_sorted[user_col] == df_sorted[user_col].shift(2)) &
        ((df_sorted[step_col] - df_sorted[step_col].shift(2)) <= 2)
    )
    # 1-transaction forward match for symmetric burst identification
    lead1_match = (
        (df_sorted[user_col] == df_sorted[user_col].shift(-1)) &
        ((df_sorted[step_col].shift(-1) - df_sorted[step_col]) <= 2)
    )
    
    df_sorted["flag_velocity"] = lag2_match | (lag2_match.shift(-1).fillna(False)) | lead1_match
    df = df_sorted.sort_index()
    return df


def detect_multivariate_outliers(df: pd.DataFrame, sample_size: int = 200000, random_state: int = 42) -> pd.DataFrame:
    """Use Isolation Forest to detect complex non-linear multivariate anomalies."""
    df = df.copy()
    
    features = []
    for col in ["amount", "hour_of_day", "step"]:
        if col in df.columns:
            features.append(col)
            
    if "oldbalanceOrg" in df.columns and "newbalanceOrig" in df.columns:
        df["balance_delta_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
        features.append("balance_delta_orig")
        
    X = df[features].fillna(0)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest on representative sample if dataset is large
    iso = IsolationForest(
        n_estimators=100,
        contamination=0.015,
        random_state=random_state,
        n_jobs=-1
    )
    
    if len(X_scaled) > sample_size:
        idx_sample = np.random.RandomState(random_state).choice(len(X_scaled), sample_size, replace=False)
        iso.fit(X_scaled[idx_sample])
    else:
        iso.fit(X_scaled)
        
    preds = iso.predict(X_scaled)
    df["flag_multivariate_outlier"] = (preds == -1)
    return df


def detect_concentration_hhi(df: pd.DataFrame, user_col: str = "nameOrig", vendor_col: str = "vendor_id") -> pd.DataFrame:
    """Calculate user vendor concentration using Herfindahl-Hirschman Index (HHI)."""
    df = df.copy()
    
    # Compute total spend and vendor spend per user
    user_totals = df.groupby(user_col)["amount"].sum().rename("total_user_amount")
    uv_spend = df.groupby([user_col, vendor_col])["amount"].sum().reset_index()
    uv_spend = uv_spend.merge(user_totals, on=user_col)
    uv_spend["share"] = uv_spend["amount"] / uv_spend["total_user_amount"].replace(0, 1)
    uv_spend["share_sq"] = uv_spend["share"] ** 2
    
    hhi = uv_spend.groupby(user_col)["share_sq"].sum().reset_index().rename(columns={"share_sq": "user_hhi"})
    
    # Flag concentration if user transacted >= 3 times and HHI > 0.85 (heavily concentrated)
    user_counts = df[user_col].value_counts().rename("txn_count")
    hhi = hhi.merge(user_counts, on=user_col)
    hhi["flag_concentration"] = (hhi["txn_count"] >= 3) & (hhi["user_hhi"] >= 0.85)
    
    df = df.merge(hhi[[user_col, "user_hhi", "flag_concentration"]], on=user_col, how="left")
    df["flag_concentration"] = df["flag_concentration"].fillna(False).astype(bool)
    df["user_hhi"] = df["user_hhi"].fillna(0.0).round(4)
    return df


def calculate_anomaly_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate composite anomaly score (0-100), risk tiers, and reason strings."""
    df = df.copy()
    
    # Weights for each flag
    weights = {
        "flag_duplicate": 25,
        "flag_large_transaction": 20,
        "flag_vendor_spike": 20,
        "flag_velocity": 15,
        "flag_multivariate_outlier": 15,
        "flag_unusual_timing": 10,
        "flag_concentration": 10,
    }
    
    # If isFraud is present (PaySim ground truth), incorporate heavy weight
    if "isFraud" in df.columns:
        weights["isFraud"] = 50
        
    score = pd.Series(0, index=df.index)
    reasons = pd.Series("", index=df.index)
    
    flag_descriptions = {
        "flag_duplicate": "Duplicate Transaction",
        "flag_large_transaction": "High Amount vs Category",
        "flag_vendor_spike": "Abnormal Vendor Spike",
        "flag_velocity": "High Velocity Burst",
        "flag_multivariate_outlier": "Multivariate Outlier",
        "flag_unusual_timing": "Unusual Timing / Night",
        "flag_concentration": "High Spend Concentration",
        "isFraud": "Reported Fraud Ground Truth"
    }
    
    for flag_col, weight in weights.items():
        if flag_col in df.columns:
            mask = df[flag_col].astype(bool)
            score += mask.astype(int) * weight
            desc = flag_descriptions.get(flag_col, flag_col)
            reasons = np.where(mask, reasons + desc + "; ", reasons)
            
    df["anomaly_score"] = score.clip(upper=100)
    df["flag_reasons"] = pd.Series(reasons).str.rstrip("; ")
    
    # Final threshold for is_flagged: score >= 35 OR exact duplicate OR reported fraud
    is_critical_flag = (
        (df["anomaly_score"] >= 35) | 
        df.get("flag_duplicate", False) | 
        df.get("isFraud", False).astype(bool)
    )
    df["is_flagged"] = is_critical_flag
    
    # Risk Level Tiers
    conditions = [
        df["anomaly_score"] >= 75,
        df["anomaly_score"] >= 45,
        df["anomaly_score"] >= 20
    ]
    choices = ["CRITICAL", "HIGH", "MEDIUM"]
    df["risk_level"] = np.select(conditions, choices, default="LOW")
    
    return df


def run_anomaly_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run all anomaly detection modules sequentially."""
    print("-> Running duplicate detection...")
    df = detect_duplicates(df)
    
    print("-> Running category-based amount outlier detection...")
    df = detect_large_transactions(df)
    
    print("-> Running unusual timing detection...")
    df = detect_unusual_timing(df)
    
    print("-> Running vendor spike detection...")
    df = detect_vendor_spikes(df)
    
    print("-> Running user velocity detection...")
    df = detect_user_velocity(df)
    
    print("-> Running multivariate Isolation Forest outlier detection...")
    df = detect_multivariate_outliers(df)
    
    print("-> Running concentration (HHI) analysis...")
    df = detect_concentration_hhi(df)
    
    print("-> Calculating composite anomaly score and risk tiers...")
    df = calculate_anomaly_score(df)
    
    flagged_pct = (df["is_flagged"].sum() / len(df)) * 100
    print(f"Anomaly Detection Complete! Flagged {df['is_flagged'].sum():,} / {len(df):,} transactions ({flagged_pct:.2f}%)")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run anomaly detection on cleaned transaction data.")
    parser.add_argument("--input", default="./data/processed/clean_transactions.parquet", help="Path to clean transactions")
    parser.add_argument("--outdir", default="./data/processed", help="Output directory")
    args = parser.parse_args()
    
    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
        
    flagged_df = run_anomaly_pipeline(df)
    
    out_csv = os.path.join(args.outdir, "flagged_transactions.csv")
    out_parquet = os.path.join(args.outdir, "flagged_transactions.parquet")
    
    flagged_df.to_csv(out_csv, index=False)
    try:
        flagged_df.to_parquet(out_parquet, index=False)
    except Exception as e:
        print(f"Parquet export notice: {e}")
        
    print(f"Saved flagged data to {out_csv} and {out_parquet}")
