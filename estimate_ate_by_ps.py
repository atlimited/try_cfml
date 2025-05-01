#import numpy as np
#from sklearn.linear_model import LogisticRegression
#import matplotlib.pyplot as plt
#import pandas as pd
#
#df = pd.read_csv("df.csv")
#
## 1. 傾向スコア推定
#X_ps = df[['age', 'homeownership']]
#ps_model = LogisticRegression(solver='lbfgs', max_iter=1000)
#ps_model.fit(X_ps, df['treatment'])
#df['estimated_propensity_score'] = ps_model.predict_proba(X_ps)[:, 1]
#print(df)



import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
import matplotlib.pyplot as plt

# 0. データ読み込み
df = pd.read_csv("df.csv")

# 1. 傾向スコア推定
X_ps = df[['age', 'homeownership']]
ps_model = LogisticRegression(solver='lbfgs', max_iter=1000)
ps_model.fit(X_ps, df['treatment'])
df['estimated_propensity_score'] = ps_model.predict_proba(X_ps)[:, 1]

# 2. 分類予測（閾値0.5）
df['pred_treatment'] = (df['estimated_propensity_score'] >= 0.5).astype(int)

# 3. 予測精度指標
accuracy = accuracy_score(df['treatment'], df['pred_treatment'])
roc_auc = roc_auc_score(df['treatment'], df['estimated_propensity_score'])
brier = brier_score_loss(df['treatment'], df['estimated_propensity_score'])
ll = log_loss(df['treatment'], df['estimated_propensity_score'])

print(f"Classification accuracy: {accuracy:.4f}")
print(f"ROC AUC:              {roc_auc:.4f}")
print(f"Brier score:         {brier:.4f}")
print(f"Log loss:            {ll:.4f}")

# 4. 共通サポートの確認とトリミング
plt.hist(df[df['treatment']==1]['estimated_propensity_score'], bins=30, density=True, alpha=0.5, label='Treated')
plt.hist(df[df['treatment']==0]['estimated_propensity_score'], bins=30, density=True, alpha=0.5, label='Control')
plt.xlabel('Estimated Propensity Score')
plt.ylabel('Density')
plt.title('Estimated Propensity Score Distribution by Treatment')
plt.legend()
plt.savefig("estimated_propensity_score.png")

df_trim = df[(df['estimated_propensity_score'] >= 0.05) & (df['estimated_propensity_score'] <= 0.95)].copy()

# 5. ATE推定 (IPS)
df_trim['w'] = df_trim['treatment'] / df_trim['estimated_propensity_score'] + (1 - df_trim['treatment']) / (1 - df_trim['estimated_propensity_score'])
y1 = (df_trim[df_trim['treatment']==1]['outcome'] * df_trim[df_trim['treatment']==1]['w']).sum() / df_trim[df_trim['treatment']==1]['w'].sum()
y0 = (df_trim[df_trim['treatment']==0]['outcome'] * df_trim[df_trim['treatment']==0]['w']).sum() / df_trim[df_trim['treatment']==0]['w'].sum()
ate_ipw = y1 - y0

print("Trimmed sample size:", len(df_trim))
print(f"Estimated ATE (IPW): {ate_ipw:.2f}")

true_ate = np.mean(df['y1'] - df['y0'])
print(f"True ATE (Conversion Lift): {true_ate:.4f}")

plt.show()