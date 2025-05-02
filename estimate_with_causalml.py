import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

# CausalML ライブラリをインポート - 正しいクラス名を使用
from causalml.inference.meta import (
    BaseSRegressor,    # S-Learnerベース
    BaseTRegressor,    # T-Learnerベース
    BaseXRegressor,    # X-Learnerベース
    BaseRRegressor     # R-Learnerベース
)
from causalml.inference.meta.utils import (
    check_treatment_vector, 
    check_p_conditions
)
from causalml.metrics import auuc_score, qini_score

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
treatment = df['treatment']
outcome = df['outcome']

# 傾向スコアモデル
propensity_model = LogisticRegression(random_state=42, max_iter=5000, solver='saga')
propensity_model.fit(X, treatment)
p_score = propensity_model.predict_proba(X)[:, 1]

# 1. S-Learner（単一モデルに処置を特徴量として含める）
s_learner = BaseSRegressor(
    learner=RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
)
s_learner.fit(X=X, treatment=treatment, y=outcome)
te_s = s_learner.predict(X=X, p=p_score)
df['ite_s_learner'] = te_s

# 2. T-Learner（処置群と対照群で別々のモデル）
t_learner = BaseTRegressor(
    learner=RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
)
t_learner.fit(X=X, treatment=treatment, y=outcome)
te_t = t_learner.predict(X=X, p=p_score)
df['ite_t_learner'] = te_t

# 3. X-Learner（両方向の処置効果を推定して組み合わせる）
x_learner = BaseXRegressor(
    learner=RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
)
x_learner.fit(X=X, treatment=treatment, y=outcome)
te_x = x_learner.predict(X=X, p=p_score)
df['ite_x_learner'] = te_x

# 4. R-Learner（残差ベースのアプローチ）
r_learner = BaseRRegressor(
    learner=RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
)
r_learner.fit(X=X, treatment=treatment, y=outcome, p=p_score)
te_r = r_learner.predict(X=X)
df['ite_r_learner'] = te_r

# 真のITEを取得
df['ite_true'] = df['y1'] - df['y0']

# 結果の表示
print("===== 各モデルのATE推定値 =====")
print(f"真のATE: {df['ite_true'].mean():.4f}")
print(f"S-Learner ATE: {df['ite_s_learner'].mean():.4f}")
print(f"T-Learner ATE: {df['ite_t_learner'].mean():.4f}")
print(f"X-Learner ATE: {df['ite_x_learner'].mean():.4f}")
print(f"R-Learner ATE: {df['ite_r_learner'].mean():.4f}")

# ITE相関係数
print("\n===== 各モデルのITE相関係数 =====")
for model in ['s_learner', 't_learner', 'x_learner', 'r_learner']:
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
    for model in ['s_learner', 't_learner', 'x_learner', 'r_learner']:
        ite_col = f'ite_{model}'
        print(f"  {model}: {group_df[ite_col].mean():.4f} (相関: {group_df['ite_true'].corr(group_df[ite_col]):.4f})")

# オラクルのITE合計（処置効果の総和）
total_true_ite = df['ite_true'].sum()
print("\n===== ITE合計 =====")
print(f"全体のITE合計 (真値): {total_true_ite:.2f}")
for model in ['s_learner', 't_learner', 'x_learner', 'r_learner']:
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
    for model in ['s_learner', 't_learner', 'x_learner', 'r_learner']:
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
for model in ['s_learner', 't_learner', 'x_learner', 'r_learner']:
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
for model in ['s_learner', 't_learner', 'x_learner', 'r_learner']:
    ite_col = f'ite_{model}'
    df[f'{model}_pos'] = (df[ite_col] > 0).astype(int)
    accuracy = (df[f'{model}_pos'] == df['true_uplift_pos']).mean()
    print(f"{model}: 全体正負判定精度 = {accuracy:.4f}")
    
    # グループ別の正負判定精度
    for group in sorted(df['response_group'].unique()):
        group_df = df[df['response_group'] == group]
        group_acc = (group_df[f'{model}_pos'] == group_df['true_uplift_pos']).mean()
        print(f"  グループ{group}の正負判定精度: {group_acc:.4f}")

# 可視化: UpliftとLiftカーブ（CausalMLの機能を使用）
try:
    # Uplift Curve（簡易版）
    plt.figure(figsize=(10, 6))
    
    # 上位N%の場合の効果を計算
    percentiles = np.arange(0, 101, 5)
    models = ['s_learner', 't_learner', 'x_learner', 'r_learner']
    
    for model in models:
        ite_col = f'ite_{model}'
        
        # 予測ITEでソート
        df_sorted = df.sort_values(by=ite_col, ascending=False).reset_index(drop=True)
        
        # 累積効果を計算
        cumulative_effect = []
        for p in percentiles:
            if p == 0:
                cumulative_effect.append(0)
            else:
                n_samples = int(len(df) * p / 100)
                effect = df_sorted.iloc[:n_samples]['ite_true'].sum()
                cumulative_effect.append(effect)
        
        plt.plot(percentiles, cumulative_effect, label=model)
    
    # オラクル（真のITEでソート）
    df_oracle = df.sort_values(by='ite_true', ascending=False).reset_index(drop=True)
    oracle_effect = []
    for p in percentiles:
        if p == 0:
            oracle_effect.append(0)
        else:
            n_samples = int(len(df) * p / 100)
            effect = df_oracle.iloc[:n_samples]['ite_true'].sum()
            oracle_effect.append(effect)
    
    plt.plot(percentiles, oracle_effect, label='Oracle', linestyle='--', color='black')
    
    plt.xlabel('Percentile of population treated')
    plt.ylabel('Cumulative treatment effect')
    plt.title('Uplift Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('uplift_curve.png')
    plt.close()
    
    print("\n===== 可視化結果は uplift_curve.png に保存されました =====")
except Exception as e:
    print(f"可視化中にエラーが発生しました: {e}")
