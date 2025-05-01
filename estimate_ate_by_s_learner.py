import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv("df.csv")

# S-Learner モデル学習
model = LogisticRegression(solver='lbfgs', max_iter=1000)
X = df[['age', 'homeownership', 'treatment']]
y = df['outcome']
model.fit(X, y)

# ITE & ATE 推定
X1 = X.copy(); X1['treatment'] = 1
X0 = X.copy(); X0['treatment'] = 0
pred1 = model.predict(X1)
pred0 = model.predict(X0)
ite = pred1 - pred0
ate = ite.mean()

df['estimated_ite'] = ite

# 予測確率（線形回帰出力を [0,1] にクリッピング）
# binary outcome の場合、output を確率として解釈
df['pred_prob'] = np.clip(model.predict(X), 0, 1)

# 二値予測（閾値0.5）
df['pred_outcome'] = (df['pred_prob'] >= 0.5).astype(int)

# 精度指標の計算
accuracy = accuracy_score(df['outcome'], df['pred_outcome'])
roc_auc = roc_auc_score(df['outcome'], df['pred_prob'])
brier = brier_score_loss(df['outcome'], df['pred_prob'])
ll = log_loss(df['outcome'], df['pred_prob'])

# 結果表示
print(df.head())
print(f"Estimated ATE (S-Learner): {ate:.4f}")
print(f"True ATE (Conversion Lift): {np.mean(df['y1'] - df['y0']):.4f}\n")
print("S-Learner Prediction Metrics:")
print(f"  Classification accuracy: {accuracy:.4f}")
print(f"  ROC AUC:                {roc_auc:.4f}")
print(f"  Brier score:            {brier:.4f}")
print(f"  Log loss:               {ll:.4f}")