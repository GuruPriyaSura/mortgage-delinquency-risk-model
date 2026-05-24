import pandas as pd
import numpy as np

np.random.seed(42)
n = 1500

states = ['TX','CA','FL','NY','IL','OH','GA','NC','PA','AZ']
loan_purposes = ['Purchase', 'Refinance', 'Cash-Out Refinance']

df = pd.DataFrame({
    'loan_id': [f'FMAC-{i:05d}' for i in range(1, n+1)],
    'credit_score': np.clip(np.random.normal(680, 70, n).astype(int), 300, 850),
    'ltv_ratio': np.clip(np.random.normal(78, 15, n), 20, 105),
    'dti_ratio': np.clip(np.random.normal(38, 10, n), 10, 65),
    'loan_amount': np.random.randint(80000, 950000, n),
    'loan_age_months': np.random.randint(1, 360, n),
    'original_interest_rate': np.clip(np.random.normal(6.5, 1.2, n), 2.5, 12.0),
    'num_units': np.random.choice([1, 2, 3, 4], n, p=[0.85, 0.08, 0.04, 0.03]),
    'occupancy_type': np.random.choice(['Primary', 'Second Home', 'Investment'], n, p=[0.75, 0.12, 0.13]),
    'loan_purpose': np.random.choice(loan_purposes, n, p=[0.55, 0.28, 0.17]),
    'state': np.random.choice(states, n),
    'property_value': np.random.randint(100000, 1200000, n),
    'num_borrowers': np.random.choice([1, 2], n, p=[0.42, 0.58]),
    'months_since_last_payment': np.random.choice([0,0,0,1,2,3,6,12], n, p=[0.6,0.1,0.1,0.07,0.05,0.04,0.02,0.02]),
})

# Derive delinquency based on risk factors (logistic-style)
risk_score = (
    -0.008 * (df['credit_score'] - 680)
    + 0.03 * (df['ltv_ratio'] - 78)
    + 0.02 * (df['dti_ratio'] - 38)
    + 0.3 * df['months_since_last_payment']
    + np.random.normal(0, 1.2, n)
)
prob = 1 / (1 + np.exp(-risk_score))
df['is_delinquent'] = (prob > 0.6).astype(int)
df['refinance_probability'] = np.clip(
    0.5 - 0.04 * (df['original_interest_rate'] - 5) + np.random.normal(0, 0.1, n), 0.05, 0.95
).round(3)

df.to_csv('/sessions/sweet-kind-mayer/mnt/outputs/projects/01-delinquency-risk-model/data/loan_data.csv', index=False)
print(f"Generated {n} loan records. Delinquency rate: {df['is_delinquent'].mean():.1%}")
print(df[['credit_score','ltv_ratio','dti_ratio','is_delinquent']].head())
