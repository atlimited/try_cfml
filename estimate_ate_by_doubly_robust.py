import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# 0. データ読み込み
df = pd.read_csv("df.csv")

# 1. Propensity Score モデル (age, homeownership)
ps_model = LogisticRegression(solver='lbfgs', max_iter=1000)
X_ps = df[['age', 'homeownership']]
ps_model.fit(X_ps, df['treatment'])
e = ps_model.predict_proba(X_ps)[:, 1]
df['e'] = e

# 2. Outcome モデル (S-Learner) logistic
out_model = LogisticRegression(solver='lbfgs', max_iter=1000)
X_out = df[['age', 'homeownership', 'treatment']]
out_model.fit(X_out, df['outcome'])

# 潜在結果予測
X1 = X_out.copy(); X1['treatment'] = 1
X0 = X_out.copy(); X0['treatment'] = 0
mu1 = out_model.predict_proba(X1)[:, 1]
mu0 = out_model.predict_proba(X0)[:, 1]

df['mu1'] = mu1
df['mu0'] = mu0

# 3. Doubly Robust 推定量
T = df['treatment'].values
Y = df['outcome'].values
dr = mu1 - mu0 + T * (Y - mu1) / e - (1 - T) * (Y - mu0) / (1 - e)
df['dr'] = dr

ate_dr = dr.mean()
true_ate = np.mean(df['y1'] - df['y0'])

# 4. 結果表示
print(df[['age', 'homeownership', 'treatment', 'outcome', 'e', 'mu0', 'mu1', 'dr']].head())
print(f"\nEstimated ATE (IPS): {((df['treatment']/e + (1-df['treatment'])/(1-e)) * df['outcome']).mean():.4f}")
print(f"Estimated ATE (S-Learner): {(mu1 - mu0).mean():.4f}")
print(f"Estimated ATE (Doubly Robust): {ate_dr:.4f}")
print(f"True ATE: {true_ate:.4f}")
