"""
Utility to stratify-sample PaySim while keeping 100% of all fraud records (isFraud=1).
PaySim has ~6.3 million rows with only ~8,213 frauds (~0.13%).
Sampling without stratification risks dropping rare fraud cases.
"""

import os
import argparse
import pandas as pd


def sample_paysim(input_path: str, output_path: str, n_normal: int = 150000, random_state: int = 42):
    print(f"-> Loading PaySim from {input_path} in chunks to conserve memory...")
    
    frauds = []
    normal_chunks = []
    chunksize = 250000
    
    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        # Extract all fraud cases
        fraud_rows = chunk[chunk["isFraud"] == 1]
        if len(fraud_rows) > 0:
            frauds.append(fraud_rows)
            
        # Sample normal cases
        normal_sample = chunk[chunk["isFraud"] == 0].sample(
            n=min(len(chunk[chunk["isFraud"] == 0]), n_normal // 25),
            random_state=random_state
        )
        normal_chunks.append(normal_sample)
        
    df_fraud = pd.concat(frauds, ignore_index=True) if frauds else pd.DataFrame()
    df_normal = pd.concat(normal_chunks, ignore_index=True)
    
    if len(df_normal) > n_normal:
        df_normal = df_normal.sample(n=n_normal, random_state=random_state)
        
    sampled_df = pd.concat([df_fraud, df_normal], ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sampled_df.to_csv(output_path, index=False)
    
    print(f"Sampled Dataset Created: {len(sampled_df):,} total records")
    print(f" - Verified Fraud Records (isFraud=1): {len(df_fraud):,} (100% retained)")
    print(f" - Clean Records (isFraud=0):           {len(df_normal):,}")
    print(f"Saved to: {output_path}")
    return sampled_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./data/raw/PS_20174392719_1491204439457_log.csv")
    parser.add_argument("--output", default="./data/raw/paysim_sampled.csv")
    parser.add_argument("--normal", type=int, default=150000)
    args = parser.parse_args()
    
    if os.path.exists(args.input):
        sample_paysim(args.input, args.output, n_normal=args.normal)
    else:
        print(f"Input file not found: {args.input}")
