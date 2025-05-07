import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from psmpy import PsmPy
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from obp.ope.estimators import InverseProbabilityWeighting, DirectMethod, DoublyRobust
from obp.ope import OffPolicyEvaluation

# 日本語フォントの設定（macOS向け）
plt.rcParams['font.family'] = 'Hiragino Sans GB'
# フォールバックとしてシステムのデフォルトフォントを使用
mpl.rcParams['axes.unicode_minus'] = False  # マイナス記号を正しく表示

# データの読み込み
#df = pd.read_csv("df.csv")
#df = pd.read_csv("df_only_1group.csv")
df = pd.read_csv("df_balanced_group.csv")

# データの基本統計情報を表示
print("===== マッチング前のデータ基本統計情報 =====")
print(f"データの総数: {len(df)}件")
print(f"トリートメント総数: {df['treatment'].sum()}件 (処置率: {df['treatment'].sum()/len(df):.2%})")
print(f"アウトカム総数: {df['outcome'].sum():.2f} (平均: {df['outcome'].mean():.4f})")
print(f"非処置時の潜在的アウトカム総数: {df['y0'].sum():.2f} (平均: {df['y0'].mean():.4f})")
print(f"処置時の潜在的アウトカム総数: {df['y1'].sum():.2f} (平均: {df['y1'].mean():.4f})")
print(f"真のITE総数: {(df['y1'] - df['y0']).sum():.2f} (平均: {(df['y1'] - df['y0']).mean():.4f})")
print(f"真のITE総数(実数): {(df['treat_prob'] - df['base_prob']).sum():.2f} (平均: {(df['treat_prob'] - df['base_prob']).mean():.4f})")

# 共変量の定義（年齢と住宅所有状況）
covariates = ['age', 'homeownership']

# データにIDカラムを追加
df['id'] = range(len(df))

# 独自の傾向スコアマッチングを実装
print("\n===== 傾向スコアマッチングを実行 =====")

# 1. 傾向スコアの計算
# 共変量の標準化
X = df[covariates].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

## ロジスティック回帰で傾向スコアを計算
#model = LogisticRegression(random_state=42)
#model.fit(X_scaled, df['treatment'])
#
## 傾向スコアを計算
#df['ps_score'] = model.predict_proba(X_scaled)[:, 1]
#print("傾向スコア推定完了")

df["ps_score"] = df["propensity_score"]

# 2. PsmPyを使用した傾向スコアマッチング

# データフレームにインデックス列を追加
df = df.reset_index().rename(columns={'index': 'original_index'})

# PsmPyオブジェクトを初期化
psm = PsmPy(df, treatment='treatment', indx='original_index', exclude=['outcome', 'y0', 'y1', 'base_prob', 'treat_prob', 'propensity_score', 'ps_score', 'response_group', 'id'])

# 傾向スコアを計算
psm.logistic_ps(balance=True)

# マッチングを実行 (最近傾向スコアマッチング)
psm.knn_matched(matcher='propensity_score', replacement=False, caliper=None)

# マッチング結果を取得
try:
    # バージョンによってはこちら
    matched_df = psm.df_matched
    print("psm.df_matchedを使用")
    
    # マッチング結果に必要なカラムが含まれているか確認
    required_columns = ['treatment', 'outcome', 'y0', 'y1', 'base_prob', 'treat_prob', 'propensity_score', 'ps_score', 'response_group']
    missing_columns = [col for col in required_columns if col not in matched_df.columns]
    
    if missing_columns:
        print(f"\n警告: マッチング結果に次のカラムが含まれていません: {missing_columns}")
        print("元のデータから必要なカラムを追加します")
        
        # マッチング結果に含まれるインデックスを取得
        if 'original_index' in matched_df.columns:
            # original_indexを使用して元のデータから必要なカラムを取得
            matched_indices = matched_df['original_index'].values
            original_data = df.loc[matched_indices]
            
            # 必要なカラムを追加
            for col in missing_columns:
                if col in df.columns:
                    matched_df[col] = original_data[col].values
        else:
            print("警告: original_indexが見つからないため、独自のマッチング実装に切り替えます")
            # 独自のマッチング実装に切り替え
            use_custom_matching = True
except AttributeError:
    try:
        # 別のバージョンではこちら
        matched_df = psm.matched_data
        print("psm.matched_dataを使用")
        
        # マッチング結果に必要なカラムが含まれているか確認
        required_columns = ['treatment', 'outcome', 'y0', 'y1', 'base_prob', 'treat_prob', 'propensity_score', 'ps_score', 'response_group']
        missing_columns = [col for col in required_columns if col not in matched_df.columns]
        
        if missing_columns:
            print(f"\n警告: マッチング結果に次のカラムが含まれていません: {missing_columns}")
            print("元のデータから必要なカラムを追加します")
            
            # マッチング結果に含まれるインデックスを取得
            if 'original_index' in matched_df.columns:
                # original_indexを使用して元のデータから必要なカラムを取得
                matched_indices = matched_df['original_index'].values
                original_data = df.loc[matched_indices]
                
                # 必要なカラムを追加
                for col in missing_columns:
                    if col in df.columns:
                        matched_df[col] = original_data[col].values
            else:
                print("警告: original_indexが見つからないため、独自のマッチング実装に切り替えます")
                # 独自のマッチング実装に切り替え
                use_custom_matching = True
    except AttributeError:
        # 状況に応じて独自にマッチング結果を取得
        print("\n警告: PsmPyからマッチング結果を取得できませんでした")
        print("\n独自のマッチング実装に切り替えます")
        use_custom_matching = True

# # 独自のマッチング実装が必要な場合
# if 'use_custom_matching' in locals() and use_custom_matching:
#     # 処置群と対照群に分ける
#     treated = df[df['treatment'] == 1].copy()
#     control = df[df['treatment'] == 0].copy()
    
#     # 各処置群サンプルに対して、最も傾向スコアが近い対照群サンプルを見つける
#     matched_control_indices = []
#     matched_treated_indices = treated.index.tolist()
    
#     for t_idx in treated.index:
#         t_ps = treated.loc[t_idx, 'ps_score']
        
#         # 既にマッチングされたサンプルを除外
#         available_control = control[~control.index.isin(matched_control_indices)].copy()
        
#         if len(available_control) > 0:
#             # 傾向スコアの差の絶対値を計算
#             available_control.loc[:, 'ps_diff'] = abs(available_control['ps_score'] - t_ps)
            
#             # 最も差が小さいサンプルを選択
#             best_match_idx = available_control['ps_diff'].idxmin()
#             matched_control_indices.append(best_match_idx)
    
#     # マッチングされたサンプルを結合
#     matched_indices = matched_treated_indices + matched_control_indices
#     matched_df = df.loc[matched_indices].copy()

# マッチングの詳細情報を表示
print(f"\n===== PsmPyを使用したマッチング結果 =====")
print(f"  マッチングタイプ: 最近傾向スコアマッチング")
print(f"  マッチング前の処置群サンプル数: {sum(df['treatment'] == 1)}")
print(f"  マッチング前の対照群サンプル数: {sum(df['treatment'] == 0)}")
print(f"  マッチング後の処置群サンプル数: {sum(matched_df['treatment'] == 1)}")
print(f"  マッチング後の対照群サンプル数: {sum(matched_df['treatment'] == 0)}")
print(f"  マッチング率: {len(matched_df)/len(df):.2%}")

# マッチングの詳細な評価を実行
try:
    psm.plot_match(matched_entity='propensity_score', Title='Propensity Score Matching Results', Ylabel='Frequency', Xlabel='Propensity Score')
    print("マッチング結果のグラフを表示しました")
except Exception as e:
    print(f"マッチング結果のグラフ表示でエラーが発生しました: {e}")

print(f"マッチング完了: {len(matched_df)}件のデータが残りました (元データの{len(matched_df)/len(df):.2%})")

# マッチング後のデータの基本統計情報を表示
print("\n===== マッチング後のデータ基本統計情報 =====")
print(f"データの総数: {len(matched_df)}件")
print(f"トリートメント総数: {matched_df['treatment'].sum()}件 (処置率: {matched_df['treatment'].sum()/len(matched_df):.2%})")
print(f"アウトカム総数: {matched_df['outcome'].sum():.2f} (平均: {matched_df['outcome'].mean():.4f})")
print(f"非処置時の潜在的アウトカム総数: {matched_df['y0'].sum():.2f} (平均: {matched_df['y0'].mean():.4f})")
print(f"処置時の潜在的アウトカム総数: {matched_df['y1'].sum():.2f} (平均: {matched_df['y1'].mean():.4f})")
print(f"真のITE総数: {(matched_df['y1'] - matched_df['y0']).sum():.2f} (平均: {(matched_df['y1'] - matched_df['y0']).mean():.4f})")
print(f"真のITE総数(実数): {(matched_df['treat_prob'] - matched_df['base_prob']).sum():.2f} (平均: {(matched_df['treat_prob'] - matched_df['base_prob']).mean():.4f})")

# 共変量のバランスを確認
print("\n===== 共変量のバランス =====")
for cov in covariates:
    if cov == 'age':
        print(f"{cov}:")
        print(f"  処置群平均: {matched_df[matched_df['treatment']==1][cov].mean():.2f}")
        print(f"  対照群平均: {matched_df[matched_df['treatment']==0][cov].mean():.2f}")
    else:
        print(f"{cov}:")
        print(f"  処置群割合: {matched_df[matched_df['treatment']==1][cov].mean():.2%}")
        print(f"  対照群割合: {matched_df[matched_df['treatment']==0][cov].mean():.2%}")

# マッチング前後の傾向スコア分布を表示
plt.figure(figsize=(10, 6))

# マッチング前の分布
plt.subplot(1, 2, 1)
plt.hist(df[df['treatment']==1]['ps_score'], alpha=0.5, bins=20, label='Treatment')
plt.hist(df[df['treatment']==0]['ps_score'], alpha=0.5, bins=20, label='Control')
plt.title('Propensity Score Distribution (Before Matching)')
plt.legend()

# マッチング後の分布
plt.subplot(1, 2, 2)
plt.hist(matched_df[matched_df['treatment']==1]['ps_score'], alpha=0.5, bins=20, label='Treatment')
plt.hist(matched_df[matched_df['treatment']==0]['ps_score'], alpha=0.5, bins=20, label='Control')
plt.title('Propensity Score Distribution (After Matching)')
plt.legend()

plt.tight_layout()
plt.savefig("psm_balance.png")
print("傾向スコア分布のグラフを 'psm_balance.png' に保存しました")

# response_groupごとの分布をマッチング前後で比較
plt.figure(figsize=(12, 10))

# マッチング前のresponse_group分布
plt.subplot(2, 2, 1)
response_counts_before_t = df[df['treatment']==1]['response_group'].value_counts().sort_index()
response_counts_before_c = df[df['treatment']==0]['response_group'].value_counts().sort_index()

# 全てのグループが存在することを確認
all_groups = sorted(set(df['response_group'].unique()))
for group in all_groups:
    if group not in response_counts_before_t.index:
        response_counts_before_t[group] = 0
    if group not in response_counts_before_c.index:
        response_counts_before_c[group] = 0
response_counts_before_t = response_counts_before_t.sort_index()
response_counts_before_c = response_counts_before_c.sort_index()

# パーセンテージに変換
response_pct_before_t = response_counts_before_t / response_counts_before_t.sum() * 100
response_pct_before_c = response_counts_before_c / response_counts_before_c.sum() * 100

# 棒グラフの位置を調整
x = np.arange(len(all_groups))
width = 0.35

plt.bar(x - width/2, response_pct_before_t, width, label='Treatment')
plt.bar(x + width/2, response_pct_before_c, width, label='Control')
plt.xlabel('Response Group')
plt.ylabel('Percentage (%)')
plt.title('Response Group Distribution (Before Matching)')
plt.xticks(x, all_groups)
plt.legend()

# マッチング後のresponse_group分布
plt.subplot(2, 2, 2)
response_counts_after_t = matched_df[matched_df['treatment']==1]['response_group'].value_counts().sort_index()
response_counts_after_c = matched_df[matched_df['treatment']==0]['response_group'].value_counts().sort_index()

# 全てのグループが存在することを確認
for group in all_groups:
    if group not in response_counts_after_t.index:
        response_counts_after_t[group] = 0
    if group not in response_counts_after_c.index:
        response_counts_after_c[group] = 0
response_counts_after_t = response_counts_after_t.sort_index()
response_counts_after_c = response_counts_after_c.sort_index()

# パーセンテージに変換
response_pct_after_t = response_counts_after_t / response_counts_after_t.sum() * 100
response_pct_after_c = response_counts_after_c / response_counts_after_c.sum() * 100

plt.bar(x - width/2, response_pct_after_t, width, label='Treatment')
plt.bar(x + width/2, response_pct_after_c, width, label='Control')
plt.xlabel('Response Group')
plt.ylabel('Percentage (%)')
plt.title('Response Group Distribution (After Matching)')
plt.xticks(x, all_groups)
plt.legend()

# マッチング前後の差分（処置群）
plt.subplot(2, 2, 3)
diff_t = response_pct_after_t - response_pct_before_t
plt.bar(x, diff_t, width, color='skyblue')
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Response Group')
plt.ylabel('Difference in Percentage (pp)')
plt.title('Change in Treatment Group Distribution (After - Before)')
plt.xticks(x, all_groups)

# マッチング前後の差分（対照群）
plt.subplot(2, 2, 4)
diff_c = response_pct_after_c - response_pct_before_c
plt.bar(x, diff_c, width, color='orange')
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Response Group')
plt.ylabel('Difference in Percentage (pp)')
plt.title('Change in Control Group Distribution (After - Before)')
plt.xticks(x, all_groups)

plt.tight_layout()
plt.savefig("response_group_balance.png")
print("Response Groupの分布グラフを 'response_group_balance.png' に保存しました")

# マッチング前後のデータでOPE評価を比較

# マッチング前のデータでOPE評価
print("\n===== マッチング前のデータでOPE評価 =====")

# マッチング前のデータでBanditFeedback形式に変換
bandit_feedback_before = {
    "n_rounds": len(df),
    "n_actions": 2,
    "action": df["treatment"].values,
    "reward": df["outcome"].values,
    "pscore": df["propensity_score"].values,
    "position": np.zeros(len(df), dtype=int),
    "context": df[["age", "homeownership"]].values,
}

# 真のITEを計算（オラクル評価用）
true_ite_before = df["y1"] - df["y0"]

# 真の期待報酬 (oracle)
true_rewards_before = np.zeros((len(df), 2, 1))
true_rewards_before[:, 0, 0] = df["base_prob"].values
true_rewards_before[:, 1, 0] = df["treat_prob"].values

# ランダムポリシー（50%の確率で処置）
np.random.seed(42)
random_policy_before = np.random.binomial(1, 0.5, size=len(df))

# ランダムポリシーのaction_distを作成
def create_action_dist(policy):
    action_dist = np.zeros((len(policy), 2, 1))
    action_dist[:, 1, 0] = policy
    action_dist[:, 0, 0] = 1 - policy
    return action_dist

random_action_dist_before = create_action_dist(random_policy_before)

# 傾向スコアのクリッピング（最小値を0.01に設定）
clipped_pscore_before = np.maximum(bandit_feedback_before["pscore"], 0.01)

# OPE実行関数
def run_ope(bandit_feedback, reward_model, action_dist):
    # クリップした傾向スコアを使用してOPEを実行
    ope = OffPolicyEvaluation(
        bandit_feedback=bandit_feedback,
        ope_estimators=[
            InverseProbabilityWeighting(estimator_name="IPW"),
            DirectMethod(estimator_name="DM"),
            DoublyRobust(estimator_name="DR"),
        ],
    )
    return ope.estimate_policy_values(
        action_dist=action_dist,
        estimated_rewards_by_reg_model=reward_model,
    )

# マッチング前のランダムポリシーのOPE評価
print("\n===== ランダムポリシーのオフポリシー評価結果（マッチング前） =====")
ope_results_before = run_ope(bandit_feedback_before, true_rewards_before, random_action_dist_before)

# 結果の表示
print("Estimator | Est-model | True-ITE-mean")
for estimator_name, value in ope_results_before.items():
    print(f"{estimator_name:>8} | {value:.4f}     | {true_ite_before.mean():.4f}")

# マッチング後のデータでOPE評価
print("\n===== マッチング後のデータでOPE評価 =====")

# マッチング後のデータに対して傾向スコアを新たに予測
print("\nマッチング後のデータに対して傾向スコアを新たに予測")
X_after = matched_df[covariates].values
X_after_scaled = scaler.transform(X_after)  # 同じスケーラーを使用

# 新しいロジスティック回帰モデルで傾向スコアを予測
model_after = LogisticRegression(random_state=42)
model_after.fit(X_after_scaled, matched_df['treatment'])

# 予測傾向スコアを計算
matched_df['predicted_ps'] = model_after.predict_proba(X_after_scaled)[:, 1]

# 予測傾向スコアの分布を確認
print(f"\n予測傾向スコアの分布:")
print(f"  最小値: {matched_df['predicted_ps'].min():.4f}")
print(f"  最大値: {matched_df['predicted_ps'].max():.4f}")
print(f"  平均値: {matched_df['predicted_ps'].mean():.4f}")
print(f"  標準偏差: {matched_df['predicted_ps'].std():.4f}")

# マッチング後のデータでBanditFeedback形式に変換
bandit_feedback_after = {
    "n_rounds": len(matched_df),
    "n_actions": 2,
    "action": matched_df["treatment"].values,
    "reward": matched_df["outcome"].values,
    "pscore": matched_df["predicted_ps"].values,  # 予測傾向スコアを使用
    "position": np.zeros(len(matched_df), dtype=int),
    "context": matched_df[["age", "homeownership"]].values,
}

# 真のITEを計算（オラクル評価用）
true_ite_after = matched_df["y1"] - matched_df["y0"]

# 真の期待報酬 (oracle)
true_rewards_after = np.zeros((len(matched_df), 2, 1))
true_rewards_after[:, 0, 0] = matched_df["base_prob"].values
true_rewards_after[:, 1, 0] = matched_df["treat_prob"].values

# ランダムポリシー（50%の確率で処置）
np.random.seed(42)
random_policy_after = np.random.binomial(1, 0.5, size=len(matched_df))

# ランダムポリシーのaction_distを作成
random_action_dist_after = create_action_dist(random_policy_after)

# 傾向スコアのクリッピング（最小値を0.01に設定）
clipped_pscore_after = np.maximum(bandit_feedback_after["pscore"], 0.01)

# マッチング後のランダムポリシーのOPE評価
print("\n===== ランダムポリシーのオフポリシー評価結果（マッチング後） =====")
ope_results_after = run_ope(bandit_feedback_after, true_rewards_after, random_action_dist_after)

# 結果の表示
print("Estimator | Est-model | True-ITE-mean")
for estimator_name, value in ope_results_after.items():
    print(f"{estimator_name:>8} | {value:.4f}     | {true_ite_after.mean():.4f}")

# マッチング前後の差分を計算して表示
print("\n===== マッチング前後の差分（後 - 前） =====")
print("Estimator | Est-model-diff | True-ITE-diff | Ratio-Before | Ratio-After")
for estimator_name in ope_results_before.keys():
    est_before = ope_results_before[estimator_name]
    est_after = ope_results_after[estimator_name]
    true_before = true_ite_before.mean()
    true_after = true_ite_after.mean()
    
    est_diff = est_after - est_before
    true_diff = true_after - true_before
    
    # 推定値と真の値の比率
    ratio_before = est_before / true_before if true_before != 0 else float('inf')
    ratio_after = est_after / true_after if true_after != 0 else float('inf')
    
    print(f"{estimator_name:>8} | {est_diff:+.4f}        | {true_diff:+.4f}      | {ratio_before:.4f}      | {ratio_after:.4f}")

# ATEの計算
ate_before = np.mean(true_rewards_before[:, 1, 0] - true_rewards_before[:, 0, 0])
ate_after = np.mean(true_rewards_after[:, 1, 0] - true_rewards_after[:, 0, 0])

print("\n===== ATEの比較 =====")
print(f"全体のATE (Before): {ate_before:.4f}")
print(f"全体のATE (After): {ate_after:.4f}")
print(f"ATEの差分 (After - Before): {ate_after - ate_before:.4f}")

# 真のITE上位ポリシーの作成と評価
print("\n===== 真のITE上位ポリシーの評価 =====")
# 真のITEが正のサンプルのみを処置するポリシー
true_ite_policy_after = np.zeros(len(matched_df))
true_ite_policy_after[true_ite_after > 0] = 1
true_ite_policy_after_action_dist = create_action_dist(true_ite_policy_after)

# 真のITE上位ポリシーのOPE評価
ope_results_true_ite_after = run_ope(bandit_feedback_after, true_rewards_after, true_ite_policy_after_action_dist)

# 真のITE上位ポリシーのATE計算
true_ite_treated = matched_df[true_ite_policy_after == 1]
true_ite_control = matched_df[true_ite_policy_after == 0]
true_ite_policy_ate = np.mean(true_ite_after[true_ite_policy_after == 1]) if len(true_ite_treated) > 0 else 0

# 結果の表示
print(f"真のITE上位ポリシーの処置数: {np.sum(true_ite_policy_after)}")
print(f"真のITE上位ポリシーのATE: {true_ite_policy_ate:.4f}")
print(f"全体のATEとの比率: {true_ite_policy_ate/ate_after*100:.2f}%")

print("\nOPE推定結果:")
print("Estimator | Est-model | True-ATE")
for estimator_name, value in ope_results_true_ite_after.items():
    print(f"{estimator_name:>8} | {value:.4f}     | {true_ite_policy_ate:.4f}")

# アップリフトモデルの作成と評価
print("\n===== アップリフトモデルポリシーの評価 =====")
# アップリフトモデルの学習
uplift_model = LogisticRegression(random_state=42)
uplift_model.fit(X_after_scaled, true_ite_after > 0)  # ITEが正かどうかを予測

# アップリフトスコアの計算
uplift_scores = uplift_model.predict_proba(X_after_scaled)[:, 1]
matched_df['uplift_score'] = uplift_scores

# 上位20%を処置するポリシー（元のデータと同じ処置率を維持）
treatment_ratio = matched_df['treatment'].mean()
uplift_threshold = np.percentile(uplift_scores, 100 - treatment_ratio * 100)
uplift_policy_after = np.zeros(len(matched_df))
uplift_policy_after[uplift_scores > uplift_threshold] = 1
uplift_policy_after_action_dist = create_action_dist(uplift_policy_after)

# アップリフトモデルポリシーのOPE評価
ope_results_uplift_after = run_ope(bandit_feedback_after, true_rewards_after, uplift_policy_after_action_dist)

# アップリフトモデルポリシーのATE計算
uplift_treated = matched_df[uplift_policy_after == 1]
uplift_control = matched_df[uplift_policy_after == 0]
uplift_policy_ate = np.mean(true_ite_after[uplift_policy_after == 1]) if len(uplift_treated) > 0 else 0

# 結果の表示
print(f"アップリフトモデルポリシーの処置数: {np.sum(uplift_policy_after)}")
print(f"アップリフトモデルポリシーのATE: {uplift_policy_ate:.4f}")
print(f"全体のATEとの比率: {uplift_policy_ate/ate_after*100:.2f}%")
print(f"真のITE上位ポリシーとの比率: {uplift_policy_ate/true_ite_policy_ate*100:.2f}%")

print("\nOPE推定結果:")
print("Estimator | Est-model | True-ATE")
for estimator_name, value in ope_results_uplift_after.items():
    print(f"{estimator_name:>8} | {value:.4f}     | {uplift_policy_ate:.4f}")

# 処置集団と非処置集団のATE比較
print("\n===== 処置集団と非処置集団のATE比較 =====")
treated_ate = np.mean(true_ite_after[matched_df["treatment"] == 1])
control_ate = np.mean(true_ite_after[matched_df["treatment"] == 0])

print(f"処置集団のATE: {treated_ate:.4f}")
print(f"非処置集団のATE: {control_ate:.4f}")
print(f"差分 (処置 - 非処置): {treated_ate - control_ate:.4f}")

# 各ポリシーの詳細な統計情報表示
print("\n===== 各ポリシーの詳細な統計情報 =====")
print("Policy      | Sum-pred-ITE | Sum-treated | Sum-true-ITE | Sum-outcome | ATE")

# ランダムポリシー
random_sum_pred_ite = np.sum(uplift_scores[random_policy_after == 1])
random_sum_treated = np.sum(random_policy_after)
random_sum_true_ite = np.sum(true_ite_after[random_policy_after == 1])
random_sum_outcome = np.sum(matched_df["outcome"][random_policy_after == 1])
random_ate = np.mean(true_ite_after[random_policy_after == 1])
print(f"Random      | {random_sum_pred_ite:.1f}      | {random_sum_treated:.1f}     | {random_sum_true_ite:.1f}      | {random_sum_outcome:.1f}     | {random_ate:.4f}")

# 真のITE上位ポリシー
true_ite_sum_pred_ite = np.sum(uplift_scores[true_ite_policy_after == 1])
true_ite_sum_treated = np.sum(true_ite_policy_after)
true_ite_sum_true_ite = np.sum(true_ite_after[true_ite_policy_after == 1])
true_ite_sum_outcome = np.sum(matched_df["outcome"][true_ite_policy_after == 1])
print(f"True ITE    | {true_ite_sum_pred_ite:.1f}      | {true_ite_sum_treated:.1f}     | {true_ite_sum_true_ite:.1f}      | {true_ite_sum_outcome:.1f}     | {true_ite_policy_ate:.4f}")

# アップリフトモデルポリシー
uplift_sum_pred_ite = np.sum(uplift_scores[uplift_policy_after == 1])
uplift_sum_treated = np.sum(uplift_policy_after)
uplift_sum_true_ite = np.sum(true_ite_after[uplift_policy_after == 1])
uplift_sum_outcome = np.sum(matched_df["outcome"][uplift_policy_after == 1])
print(f"Uplift      | {uplift_sum_pred_ite:.1f}      | {uplift_sum_treated:.1f}     | {uplift_sum_true_ite:.1f}      | {uplift_sum_outcome:.1f}     | {uplift_policy_ate:.4f}")

# マッチング後のデータをCSVに保存
matched_df.to_csv("matched_df.csv", index=False)
print("\nマッチング後のデータを 'matched_df.csv' に保存しました")
