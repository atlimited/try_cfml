"""
傾向スコアの比較と可視化
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

# データの読み込み
df = pd.read_csv("df.csv")

# 傾向スコアの計算（endpoint.pyから必要な関数をインポート）
from propensity_score import compute_propensity_scores, compare_propensity_scores
from data_preprocessing import create_features, get_feature_sets

# 特徴量の生成
df = create_features(df)
feature_cols = get_feature_sets('original')

# 傾向スコアの計算
X = df[feature_cols]
treatment = df['treatment']
oracle_score = df['propensity_score'].values if 'propensity_score' in df.columns else None
ps_dict = compare_propensity_scores(X, treatment, oracle_score)

# 相関係数の計算
print("\n===== 傾向スコアの相関係数 =====")
if oracle_score is not None:
    lightgbm_corr = pearsonr(oracle_score, ps_dict['lightgbm'])
    logistic_corr = pearsonr(oracle_score, ps_dict['logistic'])
    
    print(f"Oracle PS vs LightGBM PS correlation: {lightgbm_corr[0]:.4f} (p-value: {lightgbm_corr[1]:.4e})")
    print(f"Oracle PS vs Logistic PS correlation: {logistic_corr[0]:.4f} (p-value: {logistic_corr[1]:.4e})")
    
    # LightGBMとLogisticの相関
    lgbm_logistic_corr = pearsonr(ps_dict['lightgbm'], ps_dict['logistic'])
    print(f"LightGBM PS vs Logistic PS correlation: {lgbm_logistic_corr[0]:.4f} (p-value: {lgbm_logistic_corr[1]:.4e})")
else:
    print("Oracle propensity score not available.")
    
# 平均二乗誤差（MSE）の計算
if oracle_score is not None:
    lightgbm_mse = np.mean((oracle_score - ps_dict['lightgbm'])**2)
    logistic_mse = np.mean((oracle_score - ps_dict['logistic'])**2)
    
    print(f"\nOracle PS vs LightGBM PS MSE: {lightgbm_mse:.4f}")
    print(f"Oracle PS vs Logistic PS MSE: {logistic_mse:.4f}")
    
    # 平均絶対誤差（MAE）の計算
    lightgbm_mae = np.mean(np.abs(oracle_score - ps_dict['lightgbm']))
    logistic_mae = np.mean(np.abs(oracle_score - ps_dict['logistic']))
    
    print(f"Oracle PS vs LightGBM PS MAE: {lightgbm_mae:.4f}")
    print(f"Oracle PS vs Logistic PS MAE: {logistic_mae:.4f}")

# 散布図の作成
plt.figure(figsize=(15, 5))

# オラクルとLightGBMの散布図
plt.subplot(1, 3, 1)
plt.scatter(oracle_score, ps_dict['lightgbm'], alpha=0.5)
plt.plot([0, 1], [0, 1], 'r--')  # 対角線
plt.xlabel('Oracle Propensity Score')
plt.ylabel('LightGBM Propensity Score')
plt.title(f'Correlation: {lightgbm_corr[0]:.4f}')
plt.grid(True)

# オラクルとLogisticの散布図
plt.subplot(1, 3, 2)
plt.scatter(oracle_score, ps_dict['logistic'], alpha=0.5)
plt.plot([0, 1], [0, 1], 'r--')  # 対角線
plt.xlabel('Oracle Propensity Score')
plt.ylabel('Logistic Propensity Score')
plt.title(f'Correlation: {logistic_corr[0]:.4f}')
plt.grid(True)

# LightGBMとLogisticの散布図
plt.subplot(1, 3, 3)
plt.scatter(ps_dict['lightgbm'], ps_dict['logistic'], alpha=0.5)
plt.plot([0, 1], [0, 1], 'r--')  # 対角線
plt.xlabel('LightGBM Propensity Score')
plt.ylabel('Logistic Propensity Score')
plt.title(f'Correlation: {lgbm_logistic_corr[0]:.4f}')
plt.grid(True)

plt.tight_layout()
plt.savefig('propensity_score_comparison.png')
#plt.show()

# 処置群と非処置群の傾向スコア分布比較
plt.figure(figsize=(15, 5))

# オラクル傾向スコアの分布
plt.subplot(1, 3, 1)
sns.histplot(data=df, x='propensity_score', hue='treatment', bins=30, 
             element="step", common_norm=False, stat="density")
plt.xlabel('Oracle Propensity Score')
plt.title('Oracle PS Distribution')
plt.grid(True)

# LightGBM傾向スコアの分布
plt.subplot(1, 3, 2)
df['lightgbm_ps'] = ps_dict['lightgbm']
sns.histplot(data=df, x='lightgbm_ps', hue='treatment', bins=30, 
             element="step", common_norm=False, stat="density")
plt.xlabel('LightGBM Propensity Score')
plt.title('LightGBM PS Distribution')
plt.grid(True)

# Logistic傾向スコアの分布
plt.subplot(1, 3, 3)
df['logistic_ps'] = ps_dict['logistic']
sns.histplot(data=df, x='logistic_ps', hue='treatment', bins=30, 
             element="step", common_norm=False, stat="density")
plt.xlabel('Logistic Propensity Score')
plt.title('Logistic PS Distribution')
plt.grid(True)

plt.tight_layout()
plt.savefig('propensity_score_distributions.png')
#plt.show()
