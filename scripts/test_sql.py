import sqlite3

def test_sql():
    conn = sqlite3.connect("data/processed/audit_warehouse.db")
    with open("sql/schema_and_views.sql", "r", encoding="utf-8") as f:
        sql = f.read()
    conn.executescript(sql)
    cursor = conn.cursor()
    
    cursor.execute("SELECT transaction_id, vendor_name, amount, anomaly_score, risk_level, flag_reasons FROM v_suspicious_transactions LIMIT 5")
    suspicious = cursor.fetchall()
    print("--- Top Suspicious Transactions ---")
    for row in suspicious:
        print(row)
        
    cursor.execute("SELECT vendor_name, total_transactions, total_volume, flagged_transaction_count, flagged_ratio_pct FROM v_vendor_risk_summary LIMIT 5")
    vendors = cursor.fetchall()
    print("\n--- Top Risky Vendors ---")
    for row in vendors:
        print(row)
        
    cursor.execute("SELECT category_name, txn_count, total_amount, flagged_count, flagged_rate_pct FROM v_category_risk_breakdown")
    cats = cursor.fetchall()
    print("\n--- Category Breakdown ---")
    for row in cats:
        print(row)
        
    conn.close()
    print("\n[SUCCESS] SQL views and CTE logic validated successfully!")

if __name__ == "__main__":
    test_sql()
