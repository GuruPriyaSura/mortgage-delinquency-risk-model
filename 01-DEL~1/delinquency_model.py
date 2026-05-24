"""
Mortgage Delinquency Risk Model
================================
Author: Gurupriya R | Freddie Mac Analytics
Description:
    XGBoost-based binary classifier to identify high-risk borrowers
    in a single-family mortgage portfolio. Uses SHAP for model
    explainability and flags refinance probability trends by state.

AUC-ROC achieved: ~0.83
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
import shap
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  Mortgage Delinquency Risk Model — Gurupriya R")
print("=" * 60)

df = pd.read_csv('data/loan_data.csv')
print(f"\n✅ Loaded {len(df):,} loan records")
print(f"   Delinquency rate: {df['is_delinquent'].mean():.1%}\n")

# ── 2. Feature Engineering ────────────────────────────────────────────────────
print("📐 Engineering features...")

# Encode categoricals
le = LabelEncoder()
for col in ['occupancy_type', 'loan_purpose', 'state']:
    df[col + '_enc'] = le.fit_transform(df[col])

# Derived features
df['payment_stress']    = df['dti_ratio'] * df['ltv_ratio'] / 100
df['equity_ratio']      = 100 - df['ltv_ratio']
df['risk_tier']         = pd.cut(df['credit_score'],
                                  bins=[0, 580, 620, 680, 740, 850],
                                  labels=[4, 3, 2, 1, 0]).astype(int)
df['high_ltv_flag']     = (df['ltv_ratio'] > 90).astype(int)
df['loan_per_property'] = df['loan_amount'] / df['property_value']

FEATURES = [
    'credit_score', 'ltv_ratio', 'dti_ratio', 'loan_amount',
    'loan_age_months', 'original_interest_rate', 'num_units',
    'num_borrowers', 'months_since_last_payment',
    'occupancy_type_enc', 'loan_purpose_enc', 'state_enc',
    'payment_stress', 'equity_ratio', 'risk_tier',
    'high_ltv_flag', 'loan_per_property'
]

X = df[FEATURES]
y = df['is_delinquent']

# ── 3. Train / Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

# ── 4. Train XGBoost Model ────────────────────────────────────────────────────
print("\n🚀 Training XGBoost classifier...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42,
    verbosity=0
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=False)

# ── 5. Evaluation ─────────────────────────────────────────────────────────────
print("\n📊 Model Performance")
print("-" * 40)

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred       = model.predict(X_test)
auc          = roc_auc_score(y_test, y_pred_proba)

print(f"   AUC-ROC Score : {auc:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Current','Delinquent'])}")

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
print(f"   5-Fold CV AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 6. SHAP Explainability ────────────────────────────────────────────────────
print("\n🔍 Computing SHAP values...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Mortgage Delinquency Risk Model — Results', fontsize=14, fontweight='bold')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0].plot(fpr, tpr, color='#e8924a', lw=2.5, label=f'AUC = {auc:.3f}')
axes[0].plot([0,1],[0,1], 'k--', lw=1)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend(loc='lower right')
axes[0].grid(alpha=0.3)

# Feature Importance
feat_imp = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES).sort_values(ascending=True).tail(10)
axes[1].barh(feat_imp.index, feat_imp.values, color='#e8924a')
axes[1].set_title('Top 10 Feature Importances (SHAP)')
axes[1].set_xlabel('Mean |SHAP value|')
axes[1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('delinquency_results.png', dpi=150, bbox_inches='tight')
print("   Saved → delinquency_results.png")

# ── 7. State-Level Refinance Trends ──────────────────────────────────────────
print("\n🗺️  State-Level Refinance Probability Trends")
print("-" * 40)
test_df = X_test.copy()
test_df['state'] = df.loc[X_test.index, 'state'].values
test_df['refinance_prob'] = df.loc[X_test.index, 'refinance_probability'].values
test_df['delinquency_risk'] = y_pred_proba

state_summary = (
    test_df.groupby('state')
    .agg(avg_refinance_prob=('refinance_prob', 'mean'),
         avg_delinquency_risk=('delinquency_risk', 'mean'),
         loan_count=('refinance_prob', 'count'))
    .sort_values('avg_delinquency_risk', ascending=False)
)
print(state_summary.to_string())

# ── 8. High-Risk Borrower Flagging ───────────────────────────────────────────
high_risk = df.copy()
high_risk['delinquency_risk_score'] = model.predict_proba(X)[:, 1]
high_risk_loans = high_risk[high_risk['delinquency_risk_score'] > 0.75][[
    'loan_id', 'state', 'credit_score', 'ltv_ratio', 'dti_ratio', 'delinquency_risk_score'
]].sort_values('delinquency_risk_score', ascending=False)

print(f"\n⚠️  High-Risk Loans Flagged (score > 0.75): {len(high_risk_loans):,}")
high_risk_loans.to_csv('data/high_risk_loans.csv', index=False)
print("   Saved → data/high_risk_loans.csv")

print("\n✅ Done! Final AUC-ROC:", round(auc, 4))
