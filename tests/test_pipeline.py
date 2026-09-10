import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from preprocess import clean, quality_report
from anomaly_detection import (
    detect_duplicates,
    detect_large_transactions,
    detect_unusual_timing,
    detect_vendor_spikes,
    detect_user_velocity,
    detect_multivariate_outliers,
    calculate_anomaly_score
)
from export import generate_dim_category, generate_dim_vendor, generate_dim_date


@pytest.fixture
def sample_raw_data():
    return pd.DataFrame({
        "step": [1, 2, 2, 3, 25],
        "type": ["PAYMENT", "TRANSFER", "TRANSFER", "CASH_OUT", "PAYMENT"],
        "amount": [10.50, 50000.0, 50000.0, np.nan, 25.0],
        "nameOrig": ["C1001", "C1002", "C1002", "C1003", "C1004"],
        "oldbalanceOrg": [100.0, 60000.0, 60000.0, 500.0, 200.0],
        "newbalanceOrig": [89.50, 10000.0, 10000.0, 500.0, 175.0],
        "nameDest": ["M1001", "C2001", "C2001", "M1002", "M1001"],
        "oldbalanceDest": [0.0, 100.0, 100.0, 0.0, 0.0],
        "newbalanceDest": [10.50, 50100.0, 50100.0, 0.0, 25.0],
        "isFraud": [0, 1, 1, 0, 0],
        "isFlaggedFraud": [0, 1, 1, 0, 0]
    })


def test_data_quality_report(sample_raw_data):
    rep = quality_report(sample_raw_data)
    assert len(rep) == len(sample_raw_data.columns)
    amt_row = rep[rep["column"] == "amount"].iloc[0]
    assert amt_row["n_missing"] == 1


def test_clean_pipeline(sample_raw_data):
    cleaned = clean(sample_raw_data)
    # 1 row with NaN amount dropped
    assert len(cleaned) == 4
    assert "timestamp" in cleaned.columns
    assert "hour_of_day" in cleaned.columns
    assert "is_weekend" in cleaned.columns
    assert "is_night" in cleaned.columns
    assert "is_merchant_dest" in cleaned.columns


def test_duplicate_detection():
    df = pd.DataFrame({
        "step": [1, 1, 2],
        "type": ["PAYMENT", "PAYMENT", "PAYMENT"],
        "amount": [100.0, 100.0, 100.0],
        "nameOrig": ["C1", "C1", "C1"],
        "nameDest": ["M1", "M1", "M1"]
    })
    result = detect_duplicates(df)
    assert result["flag_exact_duplicate"].iloc[1] == True
    assert result["flag_near_duplicate"].iloc[2] == True


def test_velocity_burst():
    df = pd.DataFrame({
        "step": [10, 11, 11, 12, 50],
        "nameOrig": ["C99", "C99", "C99", "C99", "C99"],
        "amount": [100, 200, 300, 400, 500]
    })
    result = detect_user_velocity(df)
    assert result["flag_velocity"].iloc[0] == True
    assert result["flag_velocity"].iloc[1] == True


def test_dimensional_generation():
    cats = ["PAYMENT", "TRANSFER", "CASH_OUT"]
    dim_cat = generate_dim_category(cats)
    assert len(dim_cat) == 3
    assert "category_risk_profile" in dim_cat.columns

    date_df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=5, freq="D")})
    dim_d = generate_dim_date(date_df)
    assert len(dim_d) == 5
    assert "date_key" in dim_d.columns
