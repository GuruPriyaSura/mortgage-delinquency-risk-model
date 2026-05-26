# 🏆 Mortgage Delinquency Risk Model

**Author:** Gurupriya R | Freddie Mac Analytics  
**Tools:** Python · XGBoost · SHAP · Scikit-learn · Pandas · Matplotlib  
**Result:** 0.83 AUC-ROC on held-out test set

---

## Overview

An end-to-end machine learning pipeline that predicts mortgage delinquency risk for single-family loans. The model flags high-risk borrowers early, enabling proactive intervention by servicing teams.

| Metric | Value |
|--------|-------|
| AUCROC | **0.83** |
| 5-Fold CV AUC | 0.81 ± 0.02 |

## Run

```bash
pip install -r requirements.txt
python delinquency_model.py
```
