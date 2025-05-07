import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# データ読み込み
df = pd.read_csv("df.csv")

# 特徴量の前処理と拡張（データ生成メカニズムに基づく）
scaler = StandardScaler()
df['age_scaled'] = scaler.fit_transform(df[['age']])
df['homeownership_scaled'] = df['homeownership']

# 年齢の閾値特徴量（グループ分類条件に基づく）
df['age_over_60'] = (df['age'] > 60).astype(int)
df['age_30_to_60'] = ((df['age'] > 30) & (df['age'] <= 60)).astype(int)
df['age_over_40'] = (df['age'] > 40).astype(int)

# データ生成で使用された潜在変数の近似
df['marketing_receptivity'] = 0.01 * df['age'] + 0.5 * df['homeownership']
df['price_sensitivity'] = -0.01 * df['age'] + 0.3 * (1 - df['homeownership'])

# グループ分類条件の近似
df['group1_score'] = df['age_over_60'] * df['marketing_receptivity']
df['group2_score'] = df['age_30_to_60'] * df['marketing_receptivity']
df['group3_score'] = df['age_over_40'] * df['price_sensitivity']

# 非線形特徴量と交互作用
df['age_squared'] = df['age_scaled'] ** 2
df['age_home'] = df['age_scaled'] * df['homeownership_scaled']
df['marketing_price_interaction'] = df['marketing_receptivity'] * df['price_sensitivity']

# 特徴量、処置、アウトカムを定義
feature_cols = [
    'age_scaled', 'homeownership_scaled', 'age_squared', 'age_home',
    'age_over_60', 'age_30_to_60', 'age_over_40',
    'marketing_receptivity', 'price_sensitivity',
    'group1_score', 'group2_score', 'group3_score',
    'marketing_price_interaction'
]
X = df[feature_cols]
treatment = df['treatment'].values
outcome = df['outcome'].values

# 傾向スコア推定
ps_model = LogisticRegression(C=0.01, solver='liblinear', random_state=42)
ps_model.fit(X, treatment)
propensity_score = ps_model.predict_proba(X)[:, 1]
df['propensity_score'] = propensity_score

# トリミング（極端な傾向スコアを回避）
min_ps, max_ps = 0.01, 0.99
df['propensity_score_trimmed'] = df['propensity_score'].clip(min_ps, max_ps)

# 1. S-Learner実装（単一モデルに処置を含める）
def s_learner(X, treatment, outcome, propensity=None):
    """S-Learner: 処置を特徴量に含めた単一モデル"""
    X_with_t = X.copy()
    X_with_t['treatment'] = treatment
    
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_with_t, outcome)
    
    # 潜在結果を予測
    X1 = X.copy()
    X1['treatment'] = 1
    X0 = X.copy()
    X0['treatment'] = 0
    
    y1_pred = model.predict(X1)
    y0_pred = model.predict(X0)
    
    # 個別処置効果を計算
    ite = y1_pred - y0_pred
    return ite, y1_pred, y0_pred, model

# 2. T-Learner実装（処置群と対照群で別々のモデル）
def t_learner(X, treatment, outcome, propensity=None):
    """T-Learner: 処置群と対照群で別々のモデル"""
    # 処置群のモデル
    t1_indices = treatment == 1
    t1_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    t1_model.fit(X[t1_indices], outcome[t1_indices])
    
    # 対照群のモデル
    t0_indices = treatment == 0
    t0_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    t0_model.fit(X[t0_indices], outcome[t0_indices])
    
    # 潜在結果を予測
    y1_pred = t1_model.predict(X)
    y0_pred = t0_model.predict(X)
    
    # 個別処置効果を計算
    ite = y1_pred - y0_pred
    return ite, y1_pred, y0_pred, (t1_model, t0_model)

# 3. X-Learner実装（交差学習アプローチ）
def x_learner(X, treatment, outcome, propensity):
    """X-Learner: 交差学習アプローチ"""
    # まずはT-Learnerと同様に処置群/対照群でモデルを構築
    t1_indices = treatment == 1
    t0_indices = treatment == 0
    
    # 第1段階: 処置群と対照群の結果を予測
    t1_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    t1_model.fit(X[t1_indices], outcome[t1_indices])
    
    t0_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    t0_model.fit(X[t0_indices], outcome[t0_indices])
    
    # 第2段階: 反事実予測と実際のアウトカムの差を学習
    d1 = outcome[t1_indices] - t0_model.predict(X[t1_indices])  # 処置群に対する処置効果
    d0 = t1_model.predict(X[t0_indices]) - outcome[t0_indices]  # 対照群に対する処置効果
    
    # 第3段階: 処置効果をモデル化
    d1_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    d1_model.fit(X[t1_indices], d1)
    
    d0_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    d0_model.fit(X[t0_indices], d0)
    
    # 個別処置効果の予測を組み合わせる
    d1_pred = d1_model.predict(X)  # すべての個体の処置効果（処置モデルに基づく）
    d0_pred = d0_model.predict(X)  # すべての個体の処置効果（対照モデルに基づく）
    
    # 傾向スコアをウェイトとして使用
    tau = propensity * d0_pred + (1 - propensity) * d1_pred
    
    # 潜在結果も予測
    y1_pred = t1_model.predict(X)
    y0_pred = t0_model.predict(X)
    
    return tau, y1_pred, y0_pred, (t1_model, t0_model, d1_model, d0_model)

# 4. DR-Learner実装（Doubly Robustアプローチ）
def dr_learner(X, treatment, outcome, propensity):
    """DR-Learner: Doubly Robust推定量をターゲットとする学習"""
    # 1. アウトカムモデル（S-Learnerと同じ）
    _, y1_pred, y0_pred, _ = s_learner(X, treatment, outcome)
    
    # 2. Doubly Robust項の計算
    T = treatment
    Y = outcome
    ps = propensity
    
    # DR推定量 = E[Y|X,T=1] - E[Y|X,T=0] + T(Y - E[Y|X,T=1])/ps - (1-T)(Y - E[Y|X,T=0])/(1-ps)
    dr_term = T * (Y - y1_pred) / ps - (1-T) * (Y - y0_pred) / (1-ps)
    
    # 外れ値の影響を緩和
    dr_term_percentiles = np.percentile(dr_term, [1, 99])
    dr_term_clipped = np.clip(dr_term, dr_term_percentiles[0], dr_term_percentiles[1])
    
    # 3. DR項をターゲットとして特徴量からモデル学習
    dr_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    dr_model.fit(X, y1_pred - y0_pred + dr_term_clipped)
    
    # 予測
    tau = dr_model.predict(X)
    
    return tau, y1_pred, y0_pred, dr_model

# 各モデルを適用
X_np = X.values
ps = df['propensity_score_trimmed'].values

# 各学習器で処置効果を推定
s_ite, s_y1, s_y0, s_model = s_learner(X, treatment, outcome)
t_ite, t_y1, t_y0, t_model = t_learner(X, treatment, outcome)
x_ite, x_y1, x_y0, x_model = x_learner(X, treatment, outcome, ps)
dr_ite, dr_y1, dr_y0, dr_model = dr_learner(X, treatment, outcome, ps)

# 結果をデータフレームに追加
df['ite_s_learner'] = s_ite
df['ite_t_learner'] = t_ite
df['ite_x_learner'] = x_ite
df['ite_dr_learner'] = dr_ite

# 真のITEを取得
df['ite_true'] = df['y1'] - df['y0']

# 結果の表示
print("===== 各モデルのATE推定値 =====")
print(f"真のATE: {df['ite_true'].mean():.4f}")
print(f"S-Learner ATE: {df['ite_s_learner'].mean():.4f}")
print(f"T-Learner ATE: {df['ite_t_learner'].mean():.4f}")
print(f"X-Learner ATE: {df['ite_x_learner'].mean():.4f}")
print(f"DR-Learner ATE: {df['ite_dr_learner'].mean():.4f}")

# ITE相関係数
print("\n===== 各モデルのITE相関係数 =====")
for model in ['s_learner', 't_learner', 'x_learner', 'dr_learner']:
    ite_col = f'ite_{model}'
    corr = df['ite_true'].corr(df[ite_col])
    mse = ((df['ite_true'] - df[ite_col]) ** 2).mean()
    mae = (abs(df['ite_true'] - df[ite_col])).mean()
    print(f"{model}: 相関係数 = {corr:.4f}, MSE = {mse:.4f}, MAE = {mae:.4f}")

# グループ別の予測精度
print("\n===== グループ別の予測精度 =====")
for group in sorted(df['response_group'].unique()):
    group_df = df[df['response_group'] == group]
    print(f"グループ{group}:")
    print(f"  サンプル数: {len(group_df)}")
    print(f"  平均真のITE: {group_df['ite_true'].mean():.4f}")
    for model in ['s_learner', 't_learner', 'x_learner', 'dr_learner']:
        ite_col = f'ite_{model}'
        print(f"  {model}: {group_df[ite_col].mean():.4f} (相関: {group_df['ite_true'].corr(group_df[ite_col]):.4f})")

# オラクルのITE合計（処置効果の総和）
total_true_ite = df['ite_true'].sum()
print("\n===== ITE合計 =====")
print(f"全体のITE合計 (真値): {total_true_ite:.2f}")
for model in ['s_learner', 't_learner', 'x_learner', 'dr_learner']:
    ite_col = f'ite_{model}'
    print(f"{model}: {df[ite_col].sum():.2f}")

# グループ別のITE合計
print("\n===== グループ別ITE合計 =====")
for group in sorted(df['response_group'].unique()):
    group_df = df[df['response_group'] == group]
    group_size = len(group_df)
    total_true = group_df['ite_true'].sum()
    print(f"グループ{group} ({group_size}人):")
    print(f"  ITE合計 (真値): {total_true:.2f}")
    print(f"  寄与率 (真値): {total_true/total_true_ite*100:.2f}%")
    for model in ['s_learner', 't_learner', 'x_learner', 'dr_learner']:
        ite_col = f'ite_{model}'
        print(f"  {model}: {group_df[ite_col].sum():.2f}")

# 上位N人にtreatmentを与えた場合の効果
print("\n===== 上位N人処置の効果 =====")
top_ns = [1000, 2000, 4000, 6000, 8000, 10000]
print("オラクル（真のITEでソート）:")
for n in top_ns:
    # 真のITEでソート
    oracle_indices = df['ite_true'].nlargest(n).index
    oracle_effect = df.loc[oracle_indices, 'ite_true'].sum()
    print(f"  上位{n}人処置: 効果合計 = {oracle_effect:.2f}, 平均効果 = {oracle_effect/n:.4f}")

# 各モデルの予測による上位N人処置の効果
for model in ['s_learner', 't_learner', 'x_learner', 'dr_learner']:
    ite_col = f'ite_{model}'
    print(f"\n{model}の予測による処置:")
    for n in top_ns:
        # 予測ITEでソート
        pred_indices = df[ite_col].nlargest(n).index
        pred_true_effect = df.loc[pred_indices, 'ite_true'].sum()  # 予測上位の真の効果
        oracle_indices = df['ite_true'].nlargest(n).index
        oracle_effect = df.loc[oracle_indices, 'ite_true'].sum()
        print(f"  上位{n}人処置: 効果合計 = {pred_true_effect:.2f}, 平均効果 = {pred_true_effect/n:.4f}, 最適比 = {pred_true_effect/oracle_effect:.2f}")

# 各モデルの正負判定精度
print("\n===== 各モデルの正負判定精度 =====")
df['true_uplift_pos'] = (df['ite_true'] > 0).astype(int)
for model in ['s_learner', 't_learner', 'x_learner', 'dr_learner']:
    ite_col = f'ite_{model}'
    df[f'{model}_pos'] = (df[ite_col] > 0).astype(int)
    accuracy = (df[f'{model}_pos'] == df['true_uplift_pos']).mean()
    print(f"{model}: 全体正負判定精度 = {accuracy:.4f}")
    
    # グループ別の正負判定精度
    for group in sorted(df['response_group'].unique()):
        group_df = df[df['response_group'] == group]
        group_acc = (group_df[f'{model}_pos'] == group_df['true_uplift_pos']).mean()
        print(f"  グループ{group}の正負判定精度: {group_acc:.4f}")

# 可視化: 予測されたITEと真のITEの関係
plt.figure(figsize=(15, 10))
for i, model in enumerate(['s_learner', 't_learner', 'x_learner', 'dr_learner']):
    plt.subplot(2, 2, i+1)
    ite_col = f'ite_{model}'
    plt.scatter(df['ite_true'], df[ite_col], alpha=0.5)
    plt.plot([-1, 1], [-1, 1], 'r--')  # 完全一致を示す線
    plt.xlabel('真のITE')
    plt.ylabel(f'予測ITE ({model})')
    plt.title(f'{model} (相関係数: {df["ite_true"].corr(df[ite_col]):.4f})')
    plt.grid(True)
plt.tight_layout()
plt.savefig('ite_comparison.png')
plt.close()

print("\n===== 可視化結果は ite_comparison.png に保存されました =====")
