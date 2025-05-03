import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from causalml.inference.meta import (
    BaseSRegressor,    # S-Learnerベース
    BaseTRegressor,    # T-Learnerベース
    BaseXRegressor,    # X-Learnerベース
    BaseRRegressor,    # R-Learnerベース
    BaseDRLearner      # Doubly Robustベース
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
    'age_scaled',
    'homeownership_scaled',
#    'age_squared',
#    'age_home',
#    'age_over_60',
#    'age_30_to_60',
#    'age_over_40',
#    'marketing_receptivity',
#    'price_sensitivity',
#    'group1_score',
#    'group2_score',
#    'group3_score',
#    'marketing_price_interaction'
]
X = df[feature_cols]
treatment = df['treatment']
outcome = df['outcome']

# LightGBMの警告メッセージを抑制
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

# LightGBMの共通パラメータ
lgb_params = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 8,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbose': -1  # 警告メッセージを抑制
}

# LightGBMの分類器パラメータ
lgb_cls_params = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 8,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbose': -1,
    'objective': 'binary'  # バイナリ分類
}

# カテゴリカル特徴量の指定
categorical_features = ['age_over_60', 'age_30_to_60', 'age_over_40', 'homeownership']

# 単純なoutcome予測モデル - 全データでアウトカムを予測（treatmentは特徴量として使わない）
outcome_predictor = lgb.LGBMClassifier(**lgb_cls_params)
outcome_predictor.fit(X, outcome)  # 全データで学習（treatmentは特徴量に含まれていない）
df['predicted_outcome'] = outcome_predictor.predict_proba(X)[:, 1]  # 確率値（クラス1の確率）を取得

# 傾向スコアモデルを分類器に変更（より適切）
propensity_model = LogisticRegression(random_state=42, max_iter=10000, solver='liblinear', C=0.1)
propensity_model.fit(X, treatment)
p_score = propensity_model.predict_proba(X)[:, 1]

# 1. S-Learner（単一モデルに処置を特徴量として含める）- 分類器版
s_learner = BaseSRegressor(
    learner=lgb.LGBMClassifier(**lgb_cls_params)
)
s_learner.fit(X=X, treatment=treatment, y=outcome)
te_s = s_learner.predict(X=X, p=p_score)
df['ite_s_learner'] = te_s

# 2. T-Learner（処置群と対照群で別々のモデル）- 分類器版
t_learner = BaseTRegressor(
    learner=lgb.LGBMClassifier(**lgb_cls_params)
)
t_learner.fit(X=X, treatment=treatment, y=outcome)
te_t = t_learner.predict(X=X)
df['ite_t_learner'] = te_t

# 3. X-Learner（T-Learnerを拡張し、異質性を考慮）- 回帰器版
x_learner = BaseXRegressor(
    learner=lgb.LGBMRegressor(**lgb_params)
)
# 傾向スコアを明示的に渡す
x_learner.fit(X=X, treatment=treatment, y=outcome, p=p_score)
te_x = x_learner.predict(X=X, p=p_score)  # 予測時にも傾向スコアを渡す
df['ite_x_learner'] = te_x

# 4. R-Learner（バイアス低減のための直交化を使用）- 回帰器版
r_learner = BaseRRegressor(
    learner=lgb.LGBMRegressor(**lgb_params)
)
r_learner.fit(X=X, treatment=treatment, y=outcome, p=p_score)
te_r = r_learner.predict(X=X)
df['ite_r_learner'] = te_r

# 5. DR-Learner（Doubly Robust推定量に基づく学習）- 回帰器版
dr_learner = BaseDRLearner(
    learner=lgb.LGBMRegressor(**lgb_params),
    control_outcome_learner=lgb.LGBMRegressor(**lgb_params),
    treatment_outcome_learner=lgb.LGBMRegressor(**lgb_params),
    treatment_effect_learner=lgb.LGBMRegressor(**lgb_params)
)
dr_learner.fit(X=X, treatment=treatment, y=outcome, p=p_score)  # 傾向スコアを明示的に渡す
te_dr = dr_learner.predict(X=X)
df['ite_dr_learner'] = te_dr

# 真のITEを取得
df['ite_true'] = df['y1'] - df['y0']

# 結果の表示
print("===== 各モデルのATE推定値 =====")
print(f"真のATE: {df['ite_true'].mean():.4f}")
print(f"S-Learner ATE: {df['ite_s_learner'].mean():.4f}")
print(f"T-Learner ATE: {df['ite_t_learner'].mean():.4f}")
print(f"X-Learner ATE: {df['ite_x_learner'].mean():.4f}")
print(f"R-Learner ATE: {df['ite_r_learner'].mean():.4f}")
print(f"DR-Learner ATE: {df['ite_dr_learner'].mean():.4f}")

# ITE相関係数
print("\n===== 各モデルのITE相関係数 =====")
for model in ['s_learner', 't_learner', 'x_learner', 'r_learner', 'dr_learner']:
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
    for model in ['s_learner', 't_learner', 'x_learner', 'r_learner', 'dr_learner']:
        ite_col = f'ite_{model}'
        print(f"  {model}: {group_df[ite_col].mean():.4f} (相関: {group_df['ite_true'].corr(group_df[ite_col]):.4f})")

# ITE合計（処置効果の総和）
total_true_ite = df['ite_true'].sum()
print("\n===== ITE合計 =====")
print(f"全体のITE合計 (真値): {total_true_ite:.2f}")
for model in ['s_learner', 't_learner', 'x_learner', 'r_learner', 'dr_learner']:
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
    for model in ['s_learner', 't_learner', 'x_learner', 'r_learner', 'dr_learner']:
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
for model in ['s_learner', 't_learner', 'x_learner', 'r_learner', 'dr_learner']:
    ite_col = f'ite_{model}'
    print(f"\n{model}の予測による処置:")
    for n in top_ns:
        # 予測ITEでソート
        pred_indices = df[ite_col].nlargest(n).index
        pred_true_effect = df.loc[pred_indices, 'ite_true'].sum()  # 予測上位の真の効果
        oracle_indices = df['ite_true'].nlargest(n).index
        oracle_effect = df.loc[oracle_indices, 'ite_true'].sum()
        print(f"  上位{n}人処置: 効果合計 = {pred_true_effect:.2f}, 平均効果 = {pred_true_effect/n:.4f}, 最適比 = {pred_true_effect/oracle_effect:.2f}")

# 単純なoutcome予測モデルによる上位N人処置の効果
print("\n単純なoutcome予測モデルによる処置:")
for n in top_ns:
    # 予測アウトカムでソート
    pred_indices = df['predicted_outcome'].nlargest(n).index
    pred_true_effect = df.loc[pred_indices, 'ite_true'].sum()  # 予測上位の真の効果
    oracle_indices = df['ite_true'].nlargest(n).index
    oracle_effect = df.loc[oracle_indices, 'ite_true'].sum()
    print(f"  上位{n}人処置: 効果合計 = {pred_true_effect:.2f}, 平均効果 = {pred_true_effect/n:.4f}, 最適比 = {pred_true_effect/oracle_effect:.2f}")

# 各モデルの正負判定精度
print("\n===== 各モデルの正負判定精度 =====")
df['true_uplift_pos'] = (df['ite_true'] > 0).astype(int)
for model in ['s_learner', 't_learner', 'x_learner', 'r_learner', 'dr_learner']:
    ite_col = f'ite_{model}'
    df[f'{model}_pos'] = (df[ite_col] > 0).astype(int)
    accuracy = (df[f'{model}_pos'] == df['true_uplift_pos']).mean()
    print(f"{model}: 全体正負判定精度 = {accuracy:.4f}")
    
    # グループ別の正負判定精度
    for group in sorted(df['response_group'].unique()):
        group_df = df[df['response_group'] == group]
        group_acc = (group_df[f'{model}_pos'] == group_df['true_uplift_pos']).mean()
        print(f"  グループ{group}の正負判定精度: {group_acc:.4f}")

# 実際の処置ポリシーの評価
print("\n===== 実際の処置ポリシーの評価 =====")
# 実際に処置された人数
N_treated = df['treatment'].sum()
print(f"実際の処置人数: {N_treated}人")

# 実際に処置された人たちの真のITE総和
actual_policy_effect = df[df['treatment'] == 1]['ite_true'].sum()
print(f"実際の処置効果の総和: {actual_policy_effect:.2f}")

# 理想的な処置配分（上位N人）の効果
oracle_indices = df['ite_true'].nlargest(N_treated).index
oracle_effect = df.loc[oracle_indices, 'ite_true'].sum()
print(f"理想的な処置配分の効果: {oracle_effect:.2f}")

# 処置効率（実際/理想）
efficiency = actual_policy_effect / oracle_effect
print(f"処置効率（実際/理想）: {efficiency:.2f} ({efficiency*100:.1f}%)")

# 各モデルによる処置ポリシーの推定効果
print("\n===== 各モデルによる処置ポリシーの推定効果 =====")
for model in ['s_learner', 't_learner', 'x_learner', 'r_learner', 'dr_learner']:
    ite_col = f'ite_{model}'
    model_indices = df[ite_col].nlargest(N_treated).index
    model_policy_effect = df.loc[model_indices, 'ite_true'].sum()
    model_efficiency = model_policy_effect / oracle_effect
    improvement = model_policy_effect / actual_policy_effect
    
    print(f"{model}:")
    print(f"  推定処置効果の総和: {model_policy_effect:.2f}")
    print(f"  処置効率（推定/理想）: {model_efficiency:.2f} ({model_efficiency*100:.1f}%)")
    print(f"  実際の処置に対する改善率: {improvement:.2f}倍")

# 単純なoutcome予測モデルによる処置ポリシーの推定効果
print("\n単純なoutcome予測モデルによる処置ポリシーの推定効果:")
model_indices = df['predicted_outcome'].nlargest(N_treated).index
model_policy_effect = df.loc[model_indices, 'ite_true'].sum()
model_efficiency = model_policy_effect / oracle_effect
improvement = model_policy_effect / actual_policy_effect
    
print(f"  推定処置効果の総和: {model_policy_effect:.2f}")
print(f"  処置効率（推定/理想）: {model_efficiency:.2f} ({model_efficiency*100:.1f}%)")
print(f"  実際の処置に対する改善率: {improvement:.2f}倍")

# 可視化: UpliftとLiftカーブ（CausalMLの機能を使用）
try:
    # Uplift Curve（簡易版）
    plt.figure(figsize=(10, 6))
    
    # 上位N%の場合の効果を計算
    percentiles = np.arange(0, 101, 5)
    models = ['s_learner', 't_learner', 'x_learner', 'r_learner', 'dr_learner']
    
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
    
    # 単純なoutcome予測モデル
    df_sorted = df.sort_values(by='predicted_outcome', ascending=False).reset_index(drop=True)
    outcome_effect = []
    for p in percentiles:
        if p == 0:
            outcome_effect.append(0)
        else:
            n_samples = int(len(df) * p / 100)
            effect = df_sorted.iloc[:n_samples]['ite_true'].sum()
            outcome_effect.append(effect)
    
    plt.plot(percentiles, outcome_effect, label='Outcome Model', linestyle='-.', color='gray')
    
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
