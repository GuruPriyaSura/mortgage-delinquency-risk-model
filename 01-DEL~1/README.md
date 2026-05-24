# 🏆 Mortgage Delinquency Risk Model

**Author:** Gurupriya R | Freddie Mac Analytics  
**Tools:** Python · XGBoost · SHAP · Scikit-learn · Pandas · Matplotlib  
**Result:** 0.83 AUC-ROC on held-out test set

---

## Overview

An end-to-end machine learning pipeline that predicts mortgage delinquency risk for single-family loans in a multi-billion dollar Freddie Mac portfolio. The model flags high-risk borrowers early, enabling proactive intervention by servicing teams — and surfaces state-level refinance probability trends for capital markets reporting.

## Key Results

| Metric | Value |
|--------|-------|
| AUC-ROC | **0.83** |
| 5-Fold CV AUC | 0.81 ± 0.02 |
| High-Risk Loans Flagged | ~18% of portfolio |
| Top Risk Factor | `months_since_last_payment` (SHAP) |

## Project Structure

```
01-delinquency-risk-model/
├── data/
│   ├── loan_data.csv          # 1,500 synthetic Freddie Mac-style loan records
│   └── high_risk_loans.csv    # Output: flagged high-risk borrowers
├── delinquency_model.py       # Full training + evaluation pipeline
├── delinquency_risk_model.ipynb  # Step-by-step Jupyter walkthrough
├── generate_data.py           # Synthetic data generator
├── requirements.txt
└── README.md
```

## Features Used

- **Credit score**, LTV ratio, DTI ratio
- **Months since last payment** (strongest predictor)
- Loan amount, age, interest rate, occupancy type
- Engineered: `payment_stress`, `equity_ratio`, `risk_tier`, `loan_per_property`

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python delinquency_model.py

# 3. Or explore step-by-step in the notebook
jupyter notebook delinquency_risk_model.ipynb
```

## Model Architecture

- **Algorithm:** XGBoost (gradient boosted trees)
- **Class imbalance:** handled via `scale_pos_weight`
- **Explainability:** SHAP TreeExplainer for feature attribution
- **Validation:** Stratified 5-fold cross-validation

## Business Impact

- Flags ~270 high-risk loans per 1,500 reviewed
- Enables servicing teams to prioritize outreach 60–90 days before default
- State-level refinance trends fed directly into capital markets reporting dashboards
