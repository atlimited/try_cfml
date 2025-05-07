import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# 1. データ読み込み
df = pd.read_csv("df.csv")

# 2. Propensity Score モデル推定
ps_model = LogisticRegression(solver='lbfgs', max_iter=1000)
X_ps = df[['age', 'homeownership']]
ps_model.fit(X_ps, df['treatment'])
df['e'] = ps_model.predict_proba(X_ps)[:, 1]

# 3. 特徴量エンジニアリング - 交互作用項の追加
# 年齢と持ち家状況の標準化
scaler = StandardScaler()
df['age_scaled'] = scaler.fit_transform(df[['age']])
df['homeownership_scaled'] = df['homeownership']

# 交互作用項
df['age_home'] = df['age_scaled'] * df['homeownership_scaled']  # 年齢 × 持ち家
df['age_squared'] = df['age_scaled'] ** 2  # 年齢の二乗項

# 4. Outcome モデル（非線形 S-Learner）推定
# ランダムフォレストモデル
out_model = RandomForestRegressor(
    n_estimators=100,       # 木の数
    max_depth=10,           # 木の深さ制限
    min_samples_leaf=50,    # 葉ノードの最小サンプル数
    random_state=42,
    n_jobs=-1               # 並列計算
)

# 特徴量
X_out = df[['age_scaled', 'homeownership_scaled', 'age_home', 'age_squared', 'treatment']]
out_model.fit(X_out, df['outcome'])

# 特徴量の重要度表示
feature_importance = pd.DataFrame({
    'feature': X_out.columns,
    'importance': out_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\n===== 特徴量重要度 =====")
print(feature_importance)

# 5. 潜在アウトカム予測
X1 = X_out.copy()
X1['treatment'] = 1
X0 = X_out.copy()
X0['treatment'] = 0
df['mu1'] = out_model.predict(X1)
df['mu0'] = out_model.predict(X0)
df['uplift'] = df['mu1'] - df['mu0']  # upliftスコアを計算

# 6. upliftスコア分布のプロット
print("\n===== upliftスコア統計 =====")
print("平均:", df['uplift'].mean())
print("標準偏差:", df['uplift'].std())
print("最小値:", df['uplift'].min())
print("最大値:", df['uplift'].max())

# 各反応グループごとのupliftスコア統計
print("\n===== グループ別upliftスコア統計 =====")
for group in sorted(df['response_group'].unique()):
    group_df = df[df['response_group'] == group]
    print(f"グループ{group}:")
    print(f"  サンプル数: {len(group_df)}")
    print(f"  平均uplift: {group_df['uplift'].mean():.4f}")
    print(f"  標準偏差: {group_df['uplift'].std():.4f}")
    print(f"  最小値: {group_df['uplift'].min():.4f}")
    print(f"  最大値: {group_df['uplift'].max():.4f}")
    print(f"  真のuplift: {group_df['y1'].mean() - group_df['y0'].mean():.4f}")

# 7. ポリシー定義：Upliftスコア上位N人に処置を割り当てる
N = 4000
# 上位N人にpolicy=1を割り当て、他は0
df['policy'] = 0  # デフォルトですべてに0を割り当て
# upliftの大きい順に上位N人のインデックスを取得し、そこにpolicy=1を設定
top_indices = df['uplift'].nlargest(N).index
df.loc[top_indices, 'policy'] = 1

print(df)
print(df[["treatment", "policy", "response_group"]])
print("policy=1のtreatmentの合計", df[df["policy"] == 1]["treatment"].sum())
print("policy=1のtreatmentの割合", df[df["policy"] == 1]["treatment"].sum() / df["treatment"].sum())
print("policy=1のoutcomの合計", df[df["policy"] == 1]["outcome"].sum())
print("policy=1のoutcomの割合", df[df["policy"] == 1]["outcome"].sum() / df["outcome"].sum())

# 8. 混同行列: response_group vs treatment（絶対数）
print("\n===== 反応グループ vs 処置 (絶対数) =====")
treatment_confusion = pd.crosstab(df['response_group'], df['treatment'], margins=True)
print(treatment_confusion)

# 混同行列: response_group vs treatment（行方向割合）
print("\n===== 反応グループ vs 処置 (行方向%) =====")
treatment_confusion_row_pct = pd.crosstab(df['response_group'], df['treatment'], normalize='index') * 100
print(treatment_confusion_row_pct.round(2))

# 混同行列: response_group vs treatment（列方向割合）
print("\n===== 反応グループ vs 処置 (列方向%) =====")
treatment_confusion_col_pct = pd.crosstab(df['response_group'], df['treatment'], normalize='columns') * 100
print(treatment_confusion_col_pct.round(2))

# 混同行列: response_group vs policy（絶対数）
print("\n===== 反応グループ vs ポリシー (絶対数) =====")
policy_confusion = pd.crosstab(df['response_group'], df['policy'], margins=True)
print(policy_confusion)

# 混同行列: response_group vs policy（行方向割合）
print("\n===== 反応グループ vs ポリシー (行方向%) =====")
policy_confusion_row_pct = pd.crosstab(df['response_group'], df['policy'], normalize='index') * 100
print(policy_confusion_row_pct.round(2))

# 混同行列: response_group vs policy（列方向割合）
print("\n===== 反応グループ vs ポリシー (列方向%) =====")
policy_confusion_col_pct = pd.crosstab(df['response_group'], df['policy'], normalize='columns') * 100
print(policy_confusion_col_pct.round(2))

# 9. Doubly Robust 評価
T = df['treatment']
Y = df['outcome']

# 傾向スコアの極値を防ぐためのトリミング（安定化）
min_ps = 0.01  # 最小値
max_ps = 0.99  # 最大値
e_orig = df['e'].copy()  # 元の傾向スコアを保存
e = df['e'].clip(min_ps, max_ps)  # トリミング

mu1 = df['mu1']
mu0 = df['mu0']
pi = df['policy']

# DR推定量 (安定版)
dr_term_raw = pi * (Y - mu1) * T / e - pi * (Y - mu0) * (1-T) / (1-e)
# 外れ値の影響を軽減するために、dr_termも上下限でトリミング
dr_term_percentiles = np.percentile(dr_term_raw, [1, 99])
df['dr_term'] = np.clip(dr_term_raw, dr_term_percentiles[0], dr_term_percentiles[1])
df['dr_value'] = pi * mu1 + (1 - pi) * mu0 + df['dr_term']
dr_value = df['dr_value'].mean()

# IPW推定量も計算
ipw_value = np.mean(Y * pi * T / e + Y * (1-pi) * (1-T) / (1-e))

# 10. 結果表示
print(f"Policy: treat top {N} uplift")
print(f"Policy coverage (treated fraction): {pi.mean():.3f}")
print(f"Doubly Robust estimated policy value (expected outcome): {dr_value:.3f}")
print(f"IPW estimated policy value: {ipw_value:.3f}")

# 真の policy 実行時の平均アウトカム（真の y1, y0 があれば比較可能）
if {'y0', 'y1'}.issubset(df.columns):
    true_value = np.mean(pi * df['y1'] + (1 - pi) * df['y0'])
    print(f"True policy value: {true_value:.3f}")

    # オラクルポリシー（真のupliftでソート）
    df['true_uplift'] = df['y1'] - df['y0']
    oracle_indices = df['true_uplift'].nlargest(N).index
    df['oracle_policy'] = 0
    df.loc[oracle_indices, 'oracle_policy'] = 1
    oracle_policy_value = np.mean(df['oracle_policy'] * df['y1'] + (1 - df['oracle_policy']) * df['y0'])
    print(f"Oracle policy value (best possible): {oracle_policy_value:.3f}")

    # モデル予測upliftと真のupliftの相関
    uplift_corr = df['uplift'].corr(df['true_uplift'])
    print(f"Correlation between predicted and true uplift: {uplift_corr:.3f}")

# 11. グループ別のポリシー適用率
print("\n===== グループ別ポリシー適用率 =====")
group_policy = df.groupby('response_group')['policy'].mean()
print(group_policy)
