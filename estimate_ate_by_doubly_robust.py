import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 0. データ読み込み
df = pd.read_csv("df.csv")

# 1. Propensity Score モデル
ps_model = LogisticRegression(solver='lbfgs', max_iter=1000)
X_ps = df[['age', 'homeownership']]
ps_model.fit(X_ps, df['treatment'])
e = ps_model.predict_proba(X_ps)[:, 1]
df['e'] = e

# 2. 特徴量の前処理と拡張（データ生成メカニズムに基づく）
# 基本的なスケーリング
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

# 特徴量リスト
feature_cols = [
    'age_scaled', 'homeownership_scaled', 'age_squared', 'age_home',
    'age_over_60', 'age_30_to_60', 'age_over_40',
    'marketing_receptivity', 'price_sensitivity',
    'group1_score', 'group2_score', 'group3_score',
    'marketing_price_interaction'
]

# 3. S-Learnerの実装（処置を特徴量に含めた単一モデル）
# 特徴量セットに処置を追加
X_base = df[feature_cols].copy()
X_train = X_base.copy()
X_train['treatment'] = df['treatment']

# 最適化したランダムフォレスト設定でS-Learnerを構築
s_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)
s_model.fit(X_train, df['outcome'])

# 4. 潜在結果予測（S-Learnerアプローチ）
# 全ユーザーに処置した場合の予測
X1 = X_base.copy()
X1['treatment'] = 1
mu1 = s_model.predict_proba(X1)[:, 1]

# 全ユーザーに処置しなかった場合の予測
X0 = X_base.copy()
X0['treatment'] = 0
mu0 = s_model.predict_proba(X0)[:, 1]

df['mu1'] = mu1
df['mu0'] = mu0

# 5. Doubly Robust 推定量
T = df['treatment'].values
Y = df['outcome'].values

# 傾向スコアの極値を防ぐためのトリミング（安定化）
min_ps = 0.01
max_ps = 0.99
e_orig = df['e'].copy()
e = df['e'].clip(min_ps, max_ps)

# DR推定量の計算
dr_term = T * (Y - mu1) / e - (1 - T) * (Y - mu0) / (1 - e)
# 外れ値の影響を緩和
dr_term_percentiles = np.percentile(dr_term, [1, 99])
dr = mu1 - mu0 + np.clip(dr_term, dr_term_percentiles[0], dr_term_percentiles[1])
df['dr'] = dr

ate_dr = dr.mean()
df['ite_pred'] = mu1 - mu0  # 予測された個別処置効果
df['ite_true'] = df['y1'] - df['y0']  # 真の個別処置効果
true_ate = np.mean(df['ite_true'])

# 6. 特徴量重要度の表示
print("===== 特徴量重要度（S-Learnerモデル） =====")
# 特徴量名に'treatment'を追加
feature_imp_cols = feature_cols + ['treatment']
s_importances = pd.DataFrame({
    'feature': feature_imp_cols,
    'importance': s_model.feature_importances_
}).sort_values('importance', ascending=False)
print(s_importances)

# 7. 結果表示
print("\n===== サンプルデータ =====")
print(df[['age', 'homeownership', 'treatment', 'outcome', 'e', 'mu0', 'mu1', 'dr']].head())
print(f"\nEstimated ATE (IPS): {((df['treatment']/e + (1-df['treatment'])/(1-e)) * df['outcome']).mean():.4f}")
print(f"Estimated ATE (S-Learner): {(mu1 - mu0).mean():.4f}")
print(f"Estimated ATE (Doubly Robust): {ate_dr:.4f}")
print(f"True ATE: {true_ate:.4f}")

# 8. オラクル情報（真の情報を利用した理想的な指標）
print("\n===== オラクル情報 =====")
# ITE相関係数（予測と真値）
ite_corr = df['ite_pred'].corr(df['ite_true'])
print(f"ITE correlation (predicted vs true): {ite_corr:.4f}")

# MSE（平均二乗誤差）
ite_mse = ((df['ite_pred'] - df['ite_true'])**2).mean()
print(f"ITE MSE: {ite_mse:.4f}")

# MAE（平均絶対誤差）
ite_mae = (abs(df['ite_pred'] - df['ite_true'])).mean()
print(f"ITE MAE: {ite_mae:.4f}")

# オラクルのITE合計（処置効果の総和）
total_true_ite = df['ite_true'].sum()
total_pred_ite = df['ite_pred'].sum()
print(f"\n全体のITE合計 (真値): {total_true_ite:.2f}")
print(f"全体のITE合計 (予測値): {total_pred_ite:.2f}")
print(f"全体のITE平均 (真値): {df['ite_true'].mean():.4f} (=ATE)")

# グループ別のITE合計
print("\n===== グループ別ITE合計 =====")
for group in sorted(df['response_group'].unique()):
    group_df = df[df['response_group'] == group]
    group_size = len(group_df)
    total_true = group_df['ite_true'].sum()
    total_pred = group_df['ite_pred'].sum()
    print(f"グループ{group} ({group_size}人):")
    print(f"  ITE合計 (真値): {total_true:.2f}")
    print(f"  ITE合計 (予測値): {total_pred:.2f}")
    print(f"  寄与率 (真値): {total_true/total_true_ite*100:.2f}%")
    
# 年齢層別のITE合計
print("\n===== 年齢層別ITE合計 =====")
df['age_group'] = pd.cut(df['age'], bins=[19, 30, 40, 50, 60, 80], 
                        labels=['20代', '30代', '40代', '50代', '60代以上'])
age_group_ite = df.groupby('age_group')['ite_true'].agg(['count', 'sum', 'mean'])
age_group_ite['寄与率'] = age_group_ite['sum'] / total_true_ite * 100
print(age_group_ite)

# 持ち家状況別のITE合計
print("\n===== 持ち家状況別ITE合計 =====")
home_ite = df.groupby('homeownership')['ite_true'].agg(['count', 'sum', 'mean'])
home_ite['寄与率'] = home_ite['sum'] / total_true_ite * 100
home_ite.index = ['持ち家なし', '持ち家あり']
print(home_ite)

# 上位N人にtreatmentを与えた場合の効果
print("\n===== 上位N人処置の効果 =====")
top_ns = [1000, 2000, 4000, 6000, 8000, 10000]
print("オラクル（真のITEでソート）:")
for n in top_ns:
    # 真のITEでソート
    oracle_indices = df['ite_true'].nlargest(n).index
    oracle_effect = df.loc[oracle_indices, 'ite_true'].sum()
    print(f"  上位{n}人処置: 効果合計 = {oracle_effect:.2f}, 平均効果 = {oracle_effect/n:.4f}")

print("\n予測ITE（モデル予測）:")
for n in top_ns:
    # 予測ITEでソート
    pred_indices = df['ite_pred'].nlargest(n).index
    pred_true_effect = df.loc[pred_indices, 'ite_true'].sum()  # 予測上位の真の効果
    oracle_indices = df['ite_true'].nlargest(n).index
    oracle_effect = df.loc[oracle_indices, 'ite_true'].sum()
    print(f"  上位{n}人処置: 効果合計 = {pred_true_effect:.2f}, 平均効果 = {pred_true_effect/n:.4f}, 最適比 = {pred_true_effect/oracle_effect:.2f}")

# グループ別のITE予測精度
if 'response_group' in df.columns:
    print("\n===== グループ別ITE予測精度 =====")
    for group in sorted(df['response_group'].unique()):
        group_df = df[df['response_group'] == group]
        group_corr = group_df['ite_pred'].corr(group_df['ite_true'])
        group_mse = ((group_df['ite_pred'] - group_df['ite_true'])**2).mean()
        print(f"グループ{group}:")
        print(f"  平均真のITE: {group_df['ite_true'].mean():.4f}")
        print(f"  平均予測ITE: {group_df['ite_pred'].mean():.4f}")
        print(f"  ITE相関係数: {group_corr:.4f}")
        print(f"  ITE MSE: {group_mse:.4f}")

# 9. 予測の混同行列 - グループごとの正答率
print("\n===== グループごとの予測正確度 =====")
# upliftの正負の識別がどれだけ正確かを測定
df['true_uplift_pos'] = (df['ite_true'] > 0).astype(int)
df['pred_uplift_pos'] = (df['ite_pred'] > 0).astype(int)

# 全体の精度
overall_acc = (df['true_uplift_pos'] == df['pred_uplift_pos']).mean()
print(f"全体の正負判定精度: {overall_acc:.4f}")

# グループごとの精度
for group in sorted(df['response_group'].unique()):
    group_df = df[df['response_group'] == group]
    acc = (group_df['true_uplift_pos'] == group_df['pred_uplift_pos']).mean()
    print(f"グループ{group}の正負判定精度: {acc:.4f}")
