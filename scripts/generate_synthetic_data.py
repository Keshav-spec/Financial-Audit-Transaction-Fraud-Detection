"""
Synthetic PaySim Generator with Injected Ground-Truth Anomalies
Generates realistic mobile transaction data mirroring PaySim schema, enriched with synthetic
merchants, categories, and controlled anomaly injections:
- Missing amounts & balances (Data quality test)
- Exact duplicates (Cleanup test)
- Near-duplicates: same user+dest+amount within short time window (Fraud ring test)
- Category amount outliers: extreme high-value transactions
- Off-hours / night transactions (1am - 5am)
- High-velocity transaction bursts (Account takeover proxy)
- Vendor abnormal volume spikes (Compromised merchant proxy)
"""

import os
import argparse
import numpy as np
import pandas as pd
from faker import Faker


def generate_paysim_synthetic(n_records: int = 50000, random_seed: int = 42, outpath: str = "./data/raw/synthetic_paysim.csv") -> pd.DataFrame:
    np.random.seed(random_seed)
    fake = Faker()
    Faker.seed(random_seed)
    
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    print(f"-> Generating {n_records:,} synthetic PaySim transactions...")
    
    # 1. Base Users and Destinations
    n_users = max(500, n_records // 15)
    n_merchants = max(50, n_records // 50)
    
    user_pool = [f"C{str(np.random.randint(10000000, 99999999))}" for _ in range(n_users)]
    dest_merchants = [f"M{str(np.random.randint(10000000, 99999999))}" for _ in range(n_merchants)]
    dest_users = [f"C{str(np.random.randint(10000000, 99999999))}" for _ in range(n_users // 2)]
    all_dests = dest_merchants + dest_users
    
    types = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    type_probs = [0.35, 0.20, 0.25, 0.05, 0.15]
    
    # Generate Steps (e.g. 744 hours = 31 days)
    steps = np.random.randint(1, 744, size=n_records)
    selected_types = np.random.choice(types, size=n_records, p=type_probs)
    selected_orig = np.random.choice(user_pool, size=n_records)
    selected_dest = np.random.choice(all_dests, size=n_records)
    
    # Generate Realistic Amounts by Category
    base_amounts = []
    for t in selected_types:
        if t == "PAYMENT":
            amt = np.random.exponential(scale=35.0) + 2.0
        elif t == "DEBIT":
            amt = np.random.exponential(scale=60.0) + 5.0
        elif t == "TRANSFER":
            amt = np.random.lognormal(mean=7.5, sigma=1.2)
        elif t == "CASH_OUT":
            amt = np.random.lognormal(mean=7.0, sigma=1.1)
        else:  # CASH_IN
            amt = np.random.lognormal(mean=6.8, sigma=1.0)
        base_amounts.append(round(float(amt), 2))
        
    amounts = np.array(base_amounts)
    
    # Account Balances
    old_orig = np.random.uniform(100, 50000, size=n_records).round(2)
    new_orig = np.maximum(0, old_orig - amounts).round(2)
    old_dest = np.random.uniform(0, 100000, size=n_records).round(2)
    new_dest = (old_dest + amounts).round(2)
    
    is_fraud = np.zeros(n_records, dtype=int)
    is_flagged_fraud = np.zeros(n_records, dtype=int)
    
    df = pd.DataFrame({
        "step": steps,
        "type": selected_types,
        "amount": amounts,
        "nameOrig": selected_orig,
        "oldbalanceOrg": old_orig,
        "newbalanceOrig": new_orig,
        "nameDest": selected_dest,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "isFraud": is_fraud,
        "isFlaggedFraud": is_flagged_fraud
    })
    
    # --- CONTROLLED ANOMALY INJECTION ---
    print("-> Injecting controlled anomaly signals for validation...")
    
    # A. Missing Values (Data Quality Test: 50 missing amounts, 30 missing balances)
    idx_missing_amt = np.random.choice(n_records, 50, replace=False)
    df.loc[idx_missing_amt, "amount"] = np.nan
    
    idx_missing_bal = np.random.choice(n_records, 30, replace=False)
    df.loc[idx_missing_bal, "oldbalanceOrg"] = np.nan
    
    # B. Exact Duplicates (30 exact copies)
    dupe_source_idx = np.random.choice(n_records, 30, replace=False)
    dupe_rows = df.iloc[dupe_source_idx].copy()
    
    # C. Near Duplicates (35 pairs with same user, dest, amount within 1 hour)
    near_dupe_rows = []
    for _ in range(35):
        target_idx = np.random.randint(0, n_records)
        row = df.iloc[target_idx].copy()
        row["step"] = max(1, row["step"] + np.random.choice([-1, 1]))
        near_dupe_rows.append(row)
        df.loc[target_idx, "isFraud"] = 1
        
    # D. Category Outliers (25 extreme amount spikes)
    outlier_idx = np.random.choice(n_records, 25, replace=False)
    df.loc[outlier_idx, "amount"] = np.random.uniform(150000, 750000, size=25).round(2)
    df.loc[outlier_idx, "isFraud"] = 1
    
    # E. High-Velocity Bursts (10 users transacting 4+ times in 1 hour)
    burst_users = np.random.choice(user_pool, 10, replace=False)
    burst_rows = []
    for u in burst_users:
        burst_step = np.random.randint(10, 700)
        target_merchant = np.random.choice(dest_merchants)
        for _ in range(4):
            burst_rows.append({
                "step": burst_step,
                "type": "TRANSFER",
                "amount": round(float(np.random.uniform(8000, 15000)), 2),
                "nameOrig": u,
                "oldbalanceOrg": 50000.0,
                "newbalanceOrig": 35000.0,
                "nameDest": target_merchant,
                "oldbalanceDest": 10000.0,
                "newbalanceDest": 22000.0,
                "isFraud": 1,
                "isFlaggedFraud": 1
            })
            
    # F. Vendor Spikes (1 specific merchant gets 40 transactions in a single step)
    spike_merchant = dest_merchants[0]
    spike_step = 250
    spike_rows = []
    for i in range(40):
        spike_rows.append({
            "step": spike_step,
            "type": "PAYMENT",
            "amount": round(float(np.random.uniform(500, 2000)), 2),
            "nameOrig": user_pool[i % len(user_pool)],
            "oldbalanceOrg": 10000.0,
            "newbalanceOrig": 8500.0,
            "nameDest": spike_merchant,
            "oldbalanceDest": 50000.0,
            "newbalanceDest": 51500.0,
            "isFraud": 1 if i % 2 == 0 else 0,
            "isFlaggedFraud": 0
        })
        
    # Append all injected rows
    df = pd.concat([df, dupe_rows, pd.DataFrame(near_dupe_rows), pd.DataFrame(burst_rows), pd.DataFrame(spike_rows)], ignore_index=True)
    
    df.to_csv(outpath, index=False)
    print(f"Successfully generated synthetic PaySim dataset with {len(df):,} records at: {outpath}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50000)
    parser.add_argument("--output", default="./data/raw/synthetic_paysim.csv")
    args = parser.parse_args()
    
    generate_paysim_synthetic(n_records=args.rows, outpath=args.output)
