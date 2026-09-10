-- ====================================================================
-- FINANCIAL AUDIT & FRAUD DETECTION ANALYTICS PIPELINE
-- SQL Schema, CTE Preprocessing Pipeline, and Reporting Views
-- Compatible with SQLite, PostgreSQL, and MySQL
-- ====================================================================

-- Create Indexes for High-Performance Joins
CREATE INDEX IF NOT EXISTS idx_fact_date ON fact_transactions(date_key);
CREATE INDEX IF NOT EXISTS idx_fact_vendor ON fact_transactions(vendor_id);
CREATE INDEX IF NOT EXISTS idx_fact_cat ON fact_transactions(category);
CREATE INDEX IF NOT EXISTS idx_fact_flag ON fact_transactions(is_flagged);
CREATE INDEX IF NOT EXISTS idx_dim_vendor_id ON dim_vendor(vendor_id);
CREATE INDEX IF NOT EXISTS idx_dim_cat_id ON dim_category(category_id);
CREATE INDEX IF NOT EXISTS idx_dim_date_key ON dim_date(date_key);

-- 1. STAGING TABLE (Raw PaySim Structure)
CREATE TABLE IF NOT EXISTS stg_transactions (
    step INTEGER,
    type VARCHAR(32),
    amount DECIMAL(18, 2),
    nameOrig VARCHAR(64),
    oldbalanceOrg DECIMAL(18, 2),
    newbalanceOrig DECIMAL(18, 2),
    nameDest VARCHAR(64),
    oldbalanceDest DECIMAL(18, 2),
    newbalanceDest DECIMAL(18, 2),
    isFraud INTEGER,
    isFlaggedFraud INTEGER
);

-- 2. CLEAN TRANSACTIONS PIPELINE (CTE + View)
DROP VIEW IF EXISTS clean_transactions;
CREATE VIEW clean_transactions AS
WITH parsed_base AS (
    SELECT
        ROW_NUMBER() OVER () AS row_id,
        step,
        UPPER(TRIM(type)) AS category,
        ROUND(amount, 2) AS amount,
        nameOrig AS user_orig_id,
        oldbalanceOrg AS old_balance_orig,
        newbalanceOrig AS new_balance_orig,
        nameDest AS vendor_id,
        oldbalanceDest AS old_balance_dest,
        newbalanceDest AS new_balance_dest,
        isFraud AS is_fraud_ground_truth,
        isFlaggedFraud AS is_flagged_fraud_system,
        DATETIME('2024-01-01 00:00:00', '+' || step || ' hours') AS transaction_timestamp,
        (step % 24) AS hour_of_day,
        ((step / 24) % 7) AS day_of_week_num,
        CASE WHEN nameDest LIKE 'M%' THEN 1 ELSE 0 END AS is_merchant_dest
    FROM stg_transactions
    WHERE amount IS NOT NULL AND amount >= 0
),
flagged_time_and_dupes AS (
    SELECT
        p.*,
        CASE WHEN p.hour_of_day BETWEEN 1 AND 5 THEN 1 ELSE 0 END AS is_night,
        CASE WHEN p.day_of_week_num IN (5, 6) THEN 1 ELSE 0 END AS is_weekend,
        CASE WHEN ABS((p.old_balance_orig - p.amount) - p.new_balance_orig) > 1.00 THEN 1 ELSE 0 END AS is_balance_inconsistent,
        COUNT(*) OVER (
            PARTITION BY p.step, p.category, p.amount, p.user_orig_id, p.vendor_id
        ) AS exact_dupe_count,
        LAG(p.step, 1) OVER (
            PARTITION BY p.user_orig_id, p.vendor_id, p.amount ORDER BY p.step
        ) AS prev_step_same_user_vendor
    FROM parsed_base p
)
SELECT
    row_id,
    step,
    transaction_timestamp,
    hour_of_day,
    day_of_week_num,
    is_night,
    is_weekend,
    category,
    amount,
    user_orig_id,
    old_balance_orig,
    new_balance_orig,
    is_balance_inconsistent,
    vendor_id,
    is_merchant_dest,
    old_balance_dest,
    new_balance_dest,
    is_fraud_ground_truth,
    is_flagged_fraud_system,
    CASE WHEN exact_dupe_count > 1 THEN 1 ELSE 0 END AS flag_exact_duplicate,
    CASE 
        WHEN prev_step_same_user_vendor IS NOT NULL 
             AND (step - prev_step_same_user_vendor) <= 1 
        THEN 1 
        ELSE 0 
    END AS flag_near_duplicate
FROM flagged_time_and_dupes;

-- 3. AUDIT & REPORTING VIEWS FOR POWER BI & EXECUTIVES

-- View A: Suspicious & Flagged High-Risk Transactions
DROP VIEW IF EXISTS v_suspicious_transactions;
CREATE VIEW v_suspicious_transactions AS
SELECT
    f.transaction_id,
    f.timestamp,
    f.amount,
    f.category,
    v.vendor_name,
    f.vendor_id,
    f.user_orig_id,
    f.anomaly_score,
    f.risk_level,
    f.flag_reasons,
    f.flag_duplicate,
    f.flag_large_transaction,
    f.flag_unusual_timing,
    f.flag_vendor_spike,
    f.flag_velocity,
    f.flag_multivariate_outlier,
    f.flag_concentration,
    f.is_fraud_ground_truth
FROM fact_transactions f
LEFT JOIN dim_vendor v ON f.vendor_id = v.vendor_id
WHERE f.is_flagged = 1
ORDER BY f.anomaly_score DESC;

-- View B: Vendor Risk Profiling & Concentration (Direct from pre-aggregated dim_vendor)
DROP VIEW IF EXISTS v_vendor_risk_summary;
CREATE VIEW v_vendor_risk_summary AS
SELECT
    v.vendor_id,
    v.vendor_name,
    v.vendor_risk_tier,
    v.total_transactions,
    v.total_volume,
    v.avg_transaction_amount AS avg_transaction_size,
    v.flagged_count AS flagged_transaction_count,
    v.flagged_rate_pct AS flagged_ratio_pct
FROM dim_vendor v
ORDER BY v.flagged_count DESC, v.total_volume DESC;

-- View C: Daily Audit Trends & Anomaly Rate Overlay
DROP VIEW IF EXISTS v_daily_audit_trend;
CREATE VIEW v_daily_audit_trend AS
SELECT
    d.date_key,
    d.full_date,
    d.day_name,
    d.is_weekend,
    COUNT(f.transaction_id) AS total_txns,
    SUM(f.amount) AS total_value,
    SUM(CASE WHEN f.is_flagged = 1 THEN 1 ELSE 0 END) AS flagged_txns,
    SUM(CASE WHEN f.is_flagged = 1 THEN f.amount ELSE 0 END) AS flagged_value,
    ROUND(100.0 * SUM(CASE WHEN f.is_flagged = 1 THEN 1 ELSE 0 END) / COUNT(f.transaction_id), 2) AS anomaly_pct
FROM dim_date d
JOIN fact_transactions f ON d.date_key = f.date_key
GROUP BY d.date_key, d.full_date, d.day_name, d.is_weekend
ORDER BY d.date_key ASC;

-- View D: Category Risk Breakdown
DROP VIEW IF EXISTS v_category_risk_breakdown;
CREATE VIEW v_category_risk_breakdown AS
SELECT
    c.category_id,
    c.category_name,
    c.category_risk_profile,
    COUNT(f.transaction_id) AS txn_count,
    SUM(f.amount) AS total_amount,
    ROUND(AVG(f.amount), 2) AS avg_amount,
    MAX(f.amount) AS max_amount,
    SUM(CASE WHEN f.is_flagged = 1 THEN 1 ELSE 0 END) AS flagged_count,
    ROUND(100.0 * SUM(CASE WHEN f.is_flagged = 1 THEN 1 ELSE 0 END) / COUNT(f.transaction_id), 2) AS flagged_rate_pct
FROM dim_category c
JOIN fact_transactions f ON c.category_id = f.category
GROUP BY c.category_id, c.category_name, c.category_risk_profile
ORDER BY total_amount DESC;
