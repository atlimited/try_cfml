import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression

# 1. データ読み込み
df = pd.read_csv("df.csv")

# 2. Propensity Score モデル推定
ps_model = LogisticRegression(solver='lbfgs', max_iter=1000)
X_ps = df[['age', 'homeownership']]
ps_model.fit(X_ps, df['treatment'])
df['e'] = ps_model.predict_proba(X_ps)[:, 1]

# 3. Outcome モデル（S-Learner）推定
out_model = LinearRegression()
X_out = df[['age', 'homeownership', 'treatment']]
out_model.fit(X_out, df['outcome'])

# 4. 潜在アウトカム予測
X1 = df.copy()
X1['treatment'] = 1
X0 = df.copy()
X0['treatment'] = 0
df['mu1'] = out_model.predict(X1[['age', 'homeownership', 'treatment']])
df['mu0'] = out_model.predict(X0[['age', 'homeownership', 'treatment']])

# 5. ポリシー定義：Upliftスコア上位N人に処置を割り当てる
N = 2000
df['uplift'] = df['mu1'] - df['mu0']  # upliftスコアを計算
# 上位N人にpolicy=1を割り当て、他は0
df['policy'] = 0  # デフォルトですべてに0を割り当て
# upliftの大きい順に上位N人のインデックスを取得し、そこにpolicy=1を設定
top_indices = df['uplift'].nlargest(N).index
df.loc[top_indices, 'policy'] = 1

print(df)

# 6. Doubly Robust 評価
T = df['treatment']
Y = df['outcome']
e = df['e']
mu1 = df['mu1']
mu0 = df['mu0']
pi = df['policy']

# DR推定量
df['dr_term'] = pi * (Y - mu1) / e - (1 - pi) * (Y - mu0) / (1 - e)
df['dr_value'] = pi * mu1 + (1 - pi) * mu0 + df['dr_term']
dr_value = df['dr_value'].mean()

# 7. 結果表示
print(f"Policy: treat top {N} uplift")
print(f"Policy coverage (treated fraction): {pi.mean():.3f}")
print(f"Doubly Robust estimated policy value (expected outcome): {dr_value:.3f}")

# 真のポリシー上限（Oracle Policy Value）
#df["y1_minus_y0"] = df["y1"] - df["y0"]
df["y1_minus_y0"] = df["treat_prob"] - df["base_prob"]
df_sorted = df.sort_values('y1_minus_y0', ascending=False)
top = df_sorted.iloc[:N]
val_oracle = (top['y1'].sum() + df_sorted.iloc[N:]['y0'].sum()) / len(df)
#val_oracle = (top['treat_prob'].sum() + df_sorted.iloc[N:]['base_prob'].sum()) / len(df)
print(f"Oracle Policy Value: {val_oracle:.4f}")


# 真の policy 実行時の平均アウトカム（真の y1, y0 があれば比較可能）
if {'y0','y1'}.issubset(df.columns):
    true_value = np.mean(pi * df['y1'] + (1 - pi) * df['y0'])
    print(f"True policy value: {true_value:.3f}")
