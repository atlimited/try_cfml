import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from statsmodels.distributions.empirical_distribution import ECDF

# 日本語フォントの設定
plt.rcParams['font.family'] = ['Hiragino Sans', 'AppleGothic', 'Arial', 'sans-serif']  # 複数フォントを指定
plt.rcParams['axes.unicode_minus'] = False   # マイナス記号を正しく表示

# タイトルなどを英語に変更して回避
USE_ENGLISH = True

output_dir = "./output"

# データ読み込み
df = pd.read_csv("df.csv")

# 基本統計を表示
print("=== 基本統計 ===")
print(f"サンプル数: {len(df)}")
print(f"処置群: {df['treatment'].sum()} サンプル ({df['treatment'].mean()*100:.1f}%)")
print(f"アウトカム（全体）: {df['outcome'].mean()*100:.1f}%")
print(f"アウトカム（処置群）: {df[df['treatment']==1]['outcome'].mean()*100:.1f}%")
print(f"アウトカム（対照群）: {df[df['treatment']==0]['outcome'].mean()*100:.1f}%")

# グループごとの統計
print("\n=== 反応グループ統計 ===")
for group in range(1, 5):
    g_df = df[df['response_group'] == group]
    n_samples = len(g_df)
    print(f"グループ{group}: {n_samples}サンプル ({n_samples/len(df)*100:.1f}%)")
    print(f"  処置群比率: {g_df['treatment'].mean()*100:.1f}%")
    print(f"  潜在アウトカム y0（処置なし）: {g_df['y0'].mean()*100:.1f}%")
    print(f"  潜在アウトカム y1（処置あり）: {g_df['y1'].mean()*100:.1f}%")
    print(f"  Uplift (y1-y0): {(g_df['y1'].mean() - g_df['y0'].mean())*100:.1f}%")
    print(f"  観測アウトカム: {g_df['outcome'].mean()*100:.1f}%")
    print()

# 年齢とグループの関係
plt.figure(figsize=(10, 6))
sns.boxplot(x='response_group', y='age', data=df)
if USE_ENGLISH:
    plt.title('Age Distribution by Group')
else:
    plt.title('年齢分布（グループ別）')
plt.savefig(f"{output_dir}/eda_age_by_group.png")
plt.close()

# 持ち家状況とグループの関係
plt.figure(figsize=(10, 6))
homeown_by_group = df.groupby('response_group')['homeownership'].mean()
homeown_by_group.plot(kind='bar')
if USE_ENGLISH:
    plt.title('Homeownership Rate by Group')
    plt.ylabel('Homeownership Rate')
else:
    plt.title('持ち家比率（グループ別）')
    plt.ylabel('持ち家比率')
plt.savefig(f"{output_dir}/eda_homeown_by_group.png")
plt.close()

# 各グループのuplift分布
plt.figure(figsize=(12, 8))
for group in range(1, 5):
    g_df = df[df['response_group'] == group]
    uplift = g_df['y1'] - g_df['y0']
    plt.hist(uplift, alpha=0.5, bins=3, label=f'Group {group}')
if USE_ENGLISH:
    plt.title('Uplift Distribution by Group')
    plt.xlabel('Uplift (y1-y0)')
else:
    plt.title('Uplift分布（グループ別）')
    plt.xlabel('Uplift（潜在アウトカムの差）')
plt.legend()
plt.savefig(f"{output_dir}/eda_uplift_by_group.png")
plt.close()

# base_probとtreat_probの分布（グループ別）
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
if USE_ENGLISH:
    fig.suptitle('Probability Distribution by Group', fontsize=16)
else:
    fig.suptitle('処置なし/処置あり確率分布（グループ別）', fontsize=16)

for i, group in enumerate(range(1, 5)):
    ax = axes[i//2, i%2]
    g_df = df[df['response_group'] == group]
    ax.hist(g_df['base_prob'], alpha=0.5, bins=20, label='base_prob')
    ax.hist(g_df['treat_prob'], alpha=0.5, bins=20, label='treat_prob')
    if USE_ENGLISH:
        ax.set_title(f'Group {group}')
        ax.set_xlabel('Probability')
        ax.set_ylabel('Frequency')
    else:
        ax.set_title(f'Group {group}')
        ax.set_xlabel('確率')
        ax.set_ylabel('頻度')
    ax.legend()
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{output_dir}/eda_probs_by_group.png")
plt.close()

# 年齢と持ち家状況による傾向スコア
plt.figure(figsize=(12, 8))
pivot = df.pivot_table(index='age', columns='homeownership', values='propensity_score', aggfunc='mean')
sns.heatmap(pivot, cmap='viridis')
if USE_ENGLISH:
    plt.title('Propensity Score by Age and Homeownership')
    plt.xlabel('Homeownership (0=Rent, 1=Own)')
else:
    plt.title('傾向スコア（年齢×持ち家状況）')
    plt.xlabel('持ち家 (0=賃貸, 1=持ち家)')
plt.ylabel('Age')
plt.savefig(f"{output_dir}/eda_ps_by_features.png")
plt.close()

# 年齢と持ち家状況による反応グループの分布
plt.figure(figsize=(12, 8))
age_bins = pd.cut(df['age'], bins=6)
group_pivot = pd.crosstab(age_bins, df['homeownership'], values=df['response_group'], aggfunc='mean')
sns.heatmap(group_pivot, cmap='viridis', annot=True, fmt='.2f')
if USE_ENGLISH:
    plt.title('Average Response Group by Age and Homeownership')
    plt.xlabel('Homeownership (0=Rent, 1=Own)')
else:
    plt.title('平均反応グループ（年齢×持ち家状況）')
    plt.xlabel('持ち家 (0=賃貸, 1=持ち家)')
plt.ylabel('Age')
plt.savefig(f"{output_dir}/eda_group_by_features.png")
plt.close()

# グループごとの処置効果（実際のoutcome）
treatment_effect = pd.DataFrame()
for group in range(1, 5):
    g_df = df[df['response_group'] == group]
    effect = g_df[g_df['treatment']==1]['outcome'].mean() - g_df[g_df['treatment']==0]['outcome'].mean()
    treatment_effect.loc[f'Group {group}', 'Observed Effect'] = effect

# 潜在アウトカム（y1-y0）と比較
for group in range(1, 5):
    g_df = df[df['response_group'] == group]
    true_effect = g_df['y1'].mean() - g_df['y0'].mean()
    treatment_effect.loc[f'Group {group}', 'True Effect'] = true_effect

plt.figure(figsize=(10, 6))
treatment_effect.plot(kind='bar')
if USE_ENGLISH:
    plt.title('Treatment Effect: Observed vs True (by Group)')
else:
    plt.title('処置効果: 観測値 vs 真の値（グループ別）')
plt.axhline(y=0, color='k', linestyle='--')
plt.savefig(f"{output_dir}/eda_treatment_effect_by_group.png")
plt.close()

# 処置と反応グループによるアウトカム
outcome_by_group_treatment = df.groupby(['response_group', 'treatment'])['outcome'].mean().unstack()
plt.figure(figsize=(10, 6))
outcome_by_group_treatment.plot(kind='bar')
if USE_ENGLISH:
    plt.title('Outcome by Group and Treatment')
    plt.ylabel('Outcome Probability')
else:
    plt.title('アウトカム（グループ×処置）')
    plt.ylabel('アウトカム確率')
plt.savefig(f"{output_dir}/eda_outcome_by_group_treatment.png")
plt.close()

# upliftスコアとグループの関係（S-Learnerモデルの場合）
from sklearn.linear_model import LinearRegression
X_out = df[['age', 'homeownership', 'treatment']]
out_model = LinearRegression()
out_model.fit(X_out, df['outcome'])

# 潜在アウトカム予測
X1 = df.copy()
X1['treatment'] = 1
X0 = df.copy()
X0['treatment'] = 0
df['pred_mu1'] = out_model.predict(X1[['age', 'homeownership', 'treatment']])
df['pred_mu0'] = out_model.predict(X0[['age', 'homeownership', 'treatment']])
df['pred_uplift'] = df['pred_mu1'] - df['pred_mu0']
df['true_uplift'] = df['y1'] - df['y0']

# 予測upliftと真のupliftの関係（グループ別）
plt.figure(figsize=(12, 8))
for group in range(1, 5):
    g_df = df[df['response_group'] == group]
    plt.scatter(g_df['true_uplift'], g_df['pred_uplift'], alpha=0.5, label=f'Group {group}')
if USE_ENGLISH:
    plt.title('Predicted Uplift vs True Uplift (by Group)')
    plt.xlabel('True Uplift (y1-y0)')
    plt.ylabel('Predicted Uplift (S-Learner)')
else:
    plt.title('予測uplift vs 真のuplift（グループ別）')
    plt.xlabel('真のuplift (y1-y0)')
    plt.ylabel('予測uplift (S-Learner)')
plt.axhline(y=0, color='k', linestyle='--')
plt.axvline(x=0, color='k', linestyle='--')
plt.legend()
plt.savefig(f"{output_dir}/eda_pred_vs_true_uplift.png")
plt.close()

# 予測upliftの分布（グループ別）
plt.figure(figsize=(12, 8))
for group in range(1, 5):
    g_df = df[df['response_group'] == group]
    plt.hist(g_df['pred_uplift'], alpha=0.5, bins=20, label=f'Group {group}')
if USE_ENGLISH:
    plt.title('Predicted Uplift Distribution (by Group)')
    plt.xlabel('Predicted Uplift (S-Learner)')
else:
    plt.title('予測uplift分布（グループ別）')
    plt.xlabel('予測uplift (S-Learner)')
plt.legend()
plt.savefig(f"{output_dir}/eda_pred_uplift_by_group.png")
plt.close()

print("\n=== ファイル出力完了 ===")
if USE_ENGLISH:
    print("All visualization results have been saved as .png files.")
else:
    print("すべての可視化結果は .png ファイルとして保存されました。")
