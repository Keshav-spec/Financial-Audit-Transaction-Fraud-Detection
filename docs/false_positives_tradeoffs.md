# False Positives vs. False Negatives in Fraud & Financial Audit: Strategic Trade-Offs

In financial auditing, anti-money laundering (AML), and transaction fraud detection, model evaluation fundamentally diverges from traditional classification tasks. Achieving 99% accuracy is trivial in heavily imbalanced data (e.g., fraud rate ~0.1%), but financially disastrous if the model misses multi-million dollar fraud rings or blocks legitimate VIP transactions.

This document outlines the core economic and operational tradeoffs governing threshold selection and anomaly scoring in this pipeline.

---

## 1. Defining the Trade-off Matrix

| Metric | Machine Learning Definition | Operational Meaning | Business Consequence |
|---|---|---|---|
| **False Negative (FN)** | Type II Error: Fraud exists, but model marks it `is_flagged = False`. | **Missed Fraud:** An illegal transfer, unauthorized cash-out, or fraudulent merchant goes undetected. | **Direct Capital Loss & Fines:** Direct financial write-off, chargeback penalties (Visa/Mastercard rules), regulatory AML fines, and catastrophic reputational damage. |
| **False Positive (FP)** | Type I Error: Clean transaction, but model marks it `is_flagged = True`. | **False Alarm:** A legitimate customer’s purchase is frozen or diverted to manual audit. | **Operational Cost & Customer Churn:** Human auditor review costs (~$5–$30 per investigation), customer checkout friction, abandoned carts, and churn to competing fintech platforms. |

---

## 2. The Cost-Asymmetry Function

In consumer fraud, the cost function is severely asymmetric:

$$\text{Total Cost} = C_{\text{FN}} \cdot \text{FN} + C_{\text{FP}} \cdot \text{FP}$$

Where:
- $C_{\text{FN}} = \text{Transaction Amount} + \text{Chargeback Fee} + \text{Regulatory Penalty Risk}$
- $C_{\text{FP}} = \text{Auditor Labor Cost} + \text{Customer Lifetime Value (LTV) Depreciation Risk}$

### Case 1: High-Value Wire Transfers (`TRANSFER` / `CASH_OUT`)
- If an account initiates a $450,000 cash-out, $C_{\text{FN}} \approx \$450,000$, whereas auditor review cost $C_{\text{FP}} \approx \$25$.
- **Strategy:** Prioritize **Recall** (Sensitivity). Lower the threshold; accept a higher false-positive rate because a single missed fraud wipes out years of margin.

### Case 2: Low-Value Point-of-Sale (`PAYMENT` - e.g., $4.50 Coffee)
- If a customer card is declined at a drive-thru, $C_{\text{FN}} \le \$4.50$, but $C_{\text{FP}}$ includes customer embarrassment and potential loss of the cardholder account (average LTV > $1,200).
- **Strategy:** Prioritize **Precision**. Set high confidence thresholds; utilize silent step-up authentication (SMS 2FA, biometric push) rather than hard declines.

---

## 3. Why Composite Scoring Trumps Binary Single-Rule Triggers

Single-indicator rules (e.g., `flag_large_transaction` alone) generate unacceptable false positive spikes:
- A high-net-worth client buying luxury furniture at 2:00 AM triggers both `flag_large_transaction` and `flag_unusual_timing`.
- However, if their historical velocity is normal, the recipient is a verified merchant, and balance math reconciles, a hard block creates unnecessary customer friction.

### Multi-Tiered Action Matrix Implemented in this Project

Our composite scoring engine (`anomaly_score` 0 to 100) separates transactions into actionable operational tiers:

```
+------------------+-----------------------------------------------------------+
| Score Range      | Action & Operational Workflow                             |
+------------------+-----------------------------------------------------------+
| 0 - 19 (LOW)     | Auto-Approve: Pass through instantly without friction.    |
| 20 - 49 (MEDIUM) | Step-Up Auth: Require OTP / biometric confirmation.       |
| 50 - 74 (HIGH)   | Queue for Tier-1 Auditor: Same-day manual investigation.  |
| 75 - 100 (CRIT)  | Immediate Freeze: Auto-hold funds + SAR Filing Protocol.  |
+------------------+-----------------------------------------------------------+
```

---

## 4. Key Interview Discussion Points

When discussing this pipeline in technical interviews, highlight:
1. **Cost Curve Optimization:** "Instead of selecting an arbitrary 0.5 probability cutoff, we calibrated the `anomaly_score` threshold to minimize total operational loss based on dollar exposure."
2. **Explainability & Reason Codes:** "Auditors cannot act on an opaque black-box probability. Our engine attaches a string of trigger reasons (`flag_reasons`: *Duplicate Transaction; Abnormal Vendor Spike*) enabling tier-1 analysts to resolve alerts in under 60 seconds."
3. **Adaptive Baselines:** "Static thresholds decay. By computing rolling 7-day vendor activity baselines ($\mu \pm 2.5\sigma$) and category-segmented IQRs, our system accommodates seasonal demand spikes (e.g. Black Friday) without drowning the audit desk in false alarms."
