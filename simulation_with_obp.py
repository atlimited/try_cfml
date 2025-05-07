import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.model_selection import train_test_split
from obp.ope.estimators import InverseProbabilityWeighting, DirectMethod, DoublyRobust
from obp.ope import OffPolicyEvaluation

# 1) 読み込む
#df = pd.read_csv("df.csv")
df = pd.read_csv("matched_df.csv")

# データの総数、トリートメント総数、アウトカム総数を計算
total_samples = len(df)
total_treatments = df["treatment"].sum()
total_outcomes = df["outcome"].sum()
total_y0 = df["y0"].sum()  # 非処置時の潜在的アウトカム総数
total_y1 = df["y1"].sum()  # 処置時の潜在的アウトカム総数
total_true_ite = (df["y1"] - df["y0"]).sum()  # 真のITE総数
total_true_ite_num = (df["treat_prob"] - df["base_prob"]).sum()  # 真のITE総数

# データの基本統計情報を表示
print("===== データの基本統計情報 =====")
print(f"データの総数: {total_samples}件")
print(f"トリートメント総数: {total_treatments}件 (処置率: {total_treatments/total_samples:.2%})")
print("オラクル利用: ")
print(f"アウトカム総数: {total_outcomes:.2f} (平均: {total_outcomes/total_samples:.4f})")
print(f"非処置時の潜在的アウトカム総数: {total_y0:.2f} (平均: {total_y0/total_samples:.4f})")
print(f"処置時の潜在的アウトカム総数: {total_y1:.2f} (平均: {total_y1/total_samples:.4f})")
print(f"真のITE総数: {total_true_ite:.2f} (平均: {total_true_ite/total_samples:.4f})")
print(f"真のITE総数(実数): {total_true_ite_num:.2f} (平均: {total_true_ite_num/total_samples:.4f})")

# 2) BanditFeedback 形式に変換
bandit_feedback = {
    "n_rounds": len(df),
    "n_actions": 2,
    "action": df["treatment"].values,
    "reward": df["outcome"].values,
    "pscore": df["propensity_score"].values,
    "position": np.zeros(len(df), dtype=int),  # 位置情報を追加（すべて0）
    "context": df[["age", "homeownership"]].values,  # コンテキスト情報も追加
}

# 真のITEを計算（オラクル評価用）
true_ite = df["y1"] - df["y0"]  # 真のITE

# 3) 評価ポリシーの定義
#k = 2000  # 処置するユーザー数
k = total_treatments

# 3-1) ランダム抽出ポリシー
np.random.seed(42)  # 再現性のため
random_indices = np.random.choice(len(df), k, replace=False)
random_is_selected = np.zeros(len(df), dtype=bool)
random_is_selected[random_indices] = True
random_policy = random_is_selected.astype(int)

# 3-2) 真のITE上位k件を選択するポリシー（理論的最適）
true_topk_indices = np.argsort(-true_ite)[:k]
true_is_topk = np.zeros(len(true_ite), dtype=bool)
true_is_topk[true_topk_indices] = True
true_policy = true_is_topk.astype(int)

# 3-3) アップリフトモデルによるポリシー
file_name = "s_learner_classification_uplift_predictions.csv"
#file_name = "s_learner_regression_gpr_rbf_uplift_predictions.csv"
df_policy = pd.read_csv(file_name)  
predicted_ite = df_policy["ite_pred"].values  # 予測ITEを使用
model_topk_indices = np.argsort(-predicted_ite)[:k]  # 厳密に上位k件を選択
model_is_topk = np.zeros(len(predicted_ite), dtype=bool)
model_is_topk[model_topk_indices] = True
model_policy = model_is_topk.astype(int)

# 各ポリシーの統計情報を表示
print("\n===== ポリシー統計情報 =====")
print(f"処置ユーザー数: {k}人 (全体の{k/len(df):.2%})")

print("\n【ランダム抽出ポリシー】")
print(f"  処置数: {random_policy.sum()} 件")
print(f"  真のITEの平均値: {true_ite[random_is_selected].mean():.4f}")
print(f"  真のITEが正の割合: {(true_ite[random_is_selected] > 0).mean():.2%}")

print("\n【真のITE上位ポリシー】")
print(f"  処置数: {true_policy.sum()} 件")
print(f"  真のITEの平均値: {true_ite[true_is_topk].mean():.4f}")
print(f"  真のITEが正の割合: {(true_ite[true_is_topk] > 0).mean():.2%}")

print("\n【アップリフトモデルポリシー】")
print(f"  処置数: {model_policy.sum()} 件")
print(f"  真のITEの平均値: {true_ite[model_is_topk].mean():.4f}")
print(f"  真のITEが正の割合: {(true_ite[model_is_topk] > 0).mean():.2%}")
print(f"  真のITE上位{k}件との重複率: {np.sum(model_is_topk & true_is_topk) / k:.2%}")

# 4) 真の期待報酬 (oracle)
true_rewards = np.zeros((len(df), 2, 1))
true_rewards[:, 0, 0] = df["base_prob"].values
true_rewards[:, 1, 0] = df["treat_prob"].values

# 5) 推定期待報酬モデル学習用に 50/50 分割
X = df[["age", "homeownership", "treatment"]]
y = df["outcome"]
X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=0)

# ガウス過程回帰（GPR）モデルを使用
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)  # RBFカーネルを使用

# GPRモデルの初期化と学習
est_model = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-6,  # ノイズレベル
    n_restarts_optimizer=5,  # 最適化の再起動回数
    random_state=0
)

# 学習データが大きい場合、サブサンプリングを行う
n_samples = min(2000, len(X_train))  # 最大2000サンプルを使用
indices = np.random.choice(len(X_train), n_samples, replace=False)
X_train_sub = X_train.iloc[indices]
y_train_sub = y_train.iloc[indices]

print(f"\nGPRモデルの学習を開始します（サンプル数: {n_samples}）...")
est_model.fit(X_train_sub, y_train_sub)
print("GPRモデルの学習が完了しました")

# 6) 推定期待報酬を生成
estimated_rewards = np.zeros((len(df), 2, 1))
for a in [0, 1]:
    Xa = df[["age", "homeownership"]].copy()
    Xa["treatment"] = a
    estimated_rewards[:, a, 0] = est_model.predict(Xa)

# 7) 各ポリシーのaction_distを作成
def create_action_dist(policy):
    action_dist = np.zeros((len(policy), 2, 1))
    action_dist[:, 1, 0] = policy           # 選択されたなら action=1 の確率 1、それ以外は 0
    action_dist[:, 0, 0] = 1 - policy       # 非処置の確率を補完
    return action_dist

random_action_dist = create_action_dist(random_policy)
true_action_dist = create_action_dist(true_policy)
model_action_dist = create_action_dist(model_policy)

# 傾向スコアのクリッピング（最小値を0.01に設定）
clipped_pscore = np.maximum(bandit_feedback["pscore"], 0.01)


# 8) OPE 実行関数
def run_ope(reward_model, action_dist):
    #ope = OffPolicyEvaluation(
    #    bandit_feedback=bandit_feedback,
    #    ope_estimators=[InverseProbabilityWeighting(),
    #                     DirectMethod(),
    #                     DoublyRobust()]
    #)

    # クリップした傾向スコアを使用してOPEを実行
    ope = OffPolicyEvaluation(
        bandit_feedback={**bandit_feedback, "pscore": clipped_pscore},
        ope_estimators=[InverseProbabilityWeighting(),
                        DirectMethod(),
                        DoublyRobust()]
    )
    return ope.estimate_policy_values(
        action_dist=action_dist,
        estimated_rewards_by_reg_model=reward_model
    )
    return ope.estimate_policy_values(
        action_dist=action_dist,
        estimated_rewards_by_reg_model=reward_model
    )

# 9) 各ポリシーの真のITEの和と平均を計算
def calculate_true_ite_sum(policy):
    return np.sum(true_ite * policy)

def calculate_true_ite_mean(policy):
    selected_count = np.sum(policy)
    if selected_count > 0:
        return np.sum(true_ite * policy) / selected_count
    return 0

random_ite_sum = calculate_true_ite_sum(random_policy)
random_ite_mean = calculate_true_ite_mean(random_policy)

true_ite_sum = calculate_true_ite_sum(true_policy)
true_ite_mean = calculate_true_ite_mean(true_policy)

model_ite_sum = calculate_true_ite_sum(model_policy)
model_ite_mean = calculate_true_ite_mean(model_policy)

# 10) 全体のATEと各ポリシーのATEを計算
def calculate_ate():
    # 全体のATE
    overall_ate = (df["y1"] - df["y0"]).mean()
    
    # 各ポリシーのATE
    random_ate = (df["y1"][random_is_selected] - df["y0"][random_is_selected]).mean()
    true_ate = (df["y1"][true_is_topk] - df["y0"][true_is_topk]).mean()
    model_ate = (df["y1"][model_is_topk] - df["y0"][model_is_topk]).mean()
    
    print("\n===== 全体と各ポリシーのATE比較 =====")
    print(f"全体のATE: {overall_ate:.4f}")
    print(f"ランダム抽出ポリシーのATE: {random_ate:.4f}")
    print(f"真のITE上位ポリシーのATE: {true_ate:.4f}")
    print(f"アップリフトモデルポリシーのATE: {model_ate:.4f}")
    
    # 全体のATEとの差分
    print("\n全体のATEとの差分:")
    print(f"ランダム抽出ポリシー: {random_ate - overall_ate:+.4f} ({(random_ate - overall_ate)/overall_ate:+.2%})")
    print(f"真のITE上位ポリシー: {true_ate - overall_ate:+.4f} ({(true_ate - overall_ate)/overall_ate:+.2%})")
    print(f"アップリフトモデルポリシー: {model_ate - overall_ate:+.4f} ({(model_ate - overall_ate)/overall_ate:+.2%})")
    
    # ランダム抽出ポリシーとの差分
    print("\nランダム抽出ポリシーとの差分:")
    print(f"真のITE上位ポリシー: {true_ate - random_ate:+.4f} ({(true_ate - random_ate)/random_ate:+.2%})")
    print(f"アップリフトモデルポリシー: {model_ate - random_ate:+.4f} ({(model_ate - random_ate)/random_ate:+.2%})")
    
    # 全体のATEを計算する別の方法
    # 各ポリシーで処置される集団と処置されない集団の全体のATE
    print("\n===== 全体のATE（全ユーザーに対する平均処置効果） =====")
    print(f"全ユーザーの平均処置効果: {overall_ate:.4f}")
    
    # 各ポリシーの処置される集団と処置されない集団のATE
    print("\n===== 各ポリシーの処置集団と非処置集団のATE =====")
    
    # ランダム抽出ポリシー
    random_treated_ate = (df["y1"][random_is_selected] - df["y0"][random_is_selected]).mean()
    random_untreated_ate = (df["y1"][~random_is_selected] - df["y0"][~random_is_selected]).mean()
    print(f"ランダム抽出ポリシー:")
    print(f"  処置集団のATE: {random_treated_ate:.4f}")
    print(f"  非処置集団のATE: {random_untreated_ate:.4f}")
    print(f"  差分: {random_treated_ate - random_untreated_ate:+.4f}")
    
    # 真のITE上位ポリシー
    true_treated_ate = (df["y1"][true_is_topk] - df["y0"][true_is_topk]).mean()
    true_untreated_ate = (df["y1"][~true_is_topk] - df["y0"][~true_is_topk]).mean()
    print(f"真のITE上位ポリシー:")
    print(f"  処置集団のATE: {true_treated_ate:.4f}")
    print(f"  非処置集団のATE: {true_untreated_ate:.4f}")
    print(f"  差分: {true_treated_ate - true_untreated_ate:+.4f}")
    
    # アップリフトモデルポリシー
    model_treated_ate = (df["y1"][model_is_topk] - df["y0"][model_is_topk]).mean()
    model_untreated_ate = (df["y1"][~model_is_topk] - df["y0"][~model_is_topk]).mean()
    print(f"アップリフトモデルポリシー:")
    print(f"  処置集団のATE: {model_treated_ate:.4f}")
    print(f"  非処置集団のATE: {model_untreated_ate:.4f}")
    print(f"  差分: {model_treated_ate - model_untreated_ate:+.4f}")
    
    return overall_ate, random_ate, true_ate, model_ate

# 11) 各ポリシーのOPE結果を計算
random_results_true = run_ope(true_rewards, random_action_dist)
random_results_est = run_ope(estimated_rewards, random_action_dist)

true_results_true = run_ope(true_rewards, true_action_dist)
true_results_est = run_ope(estimated_rewards, true_action_dist)

model_results_true = run_ope(true_rewards, model_action_dist)
model_results_est = run_ope(estimated_rewards, model_action_dist)

# 12) 結果表示
print("\n===== ポリシー価値比較（真のITEの和と平均） =====")
print(f"ランダム抽出ポリシー: 和={random_ite_sum:.4f}, 平均={random_ite_mean:.4f}")
print(f"真のITE上位ポリシー: 和={true_ite_sum:.4f}, 平均={true_ite_mean:.4f}")
print(f"アップリフトモデルポリシー: 和={model_ite_sum:.4f}, 平均={model_ite_mean:.4f}")
print(f"アップリフトモデルの改善率（ランダム比）: {(model_ite_mean - random_ite_mean) / random_ite_mean:.2%}")
print(f"アップリフトモデルの最適性（理論的最適比）: {model_ite_mean / true_ite_mean:.2%}")

# 全体のATEと各ポリシーのATEを計算
calculate_ate()

# データの基本統計情報からの重要な洞察
print("\n===== データ分析からの重要な洞察 =====")
print("1. 処置効果の大きさ:")
print(f"   - 処置時の潜在的アウトカム平均({total_y1/total_samples:.4f})は非処置時({total_y0/total_samples:.4f})よりも約{total_y1/total_y0:.1f}倍高い")
print(f"   - 真のITEの平均は{total_true_ite/total_samples:.4f}で、処置時と非処置時の潜在的アウトカム平均の差に等しい")

print("2. ポリシー価値の定量的分析:")
print(f"   - ランダム抽出: 全体の20%のユーザーを処置して、真のITE総数の{random_ite_sum/total_true_ite:.2%}({random_ite_sum:.1f})を獲得")
print(f"   - 真のITE上位: 全体の20%のユーザーを処置して、真のITE総数の{true_ite_sum/total_true_ite:.2%}({true_ite_sum:.1f})を獲得")
print(f"   - アップリフトモデル: 全体の20%のユーザーを処置して、真のITE総数の{model_ite_sum/total_true_ite:.2%}({model_ite_sum:.1f})を獲得")
print(f"   - 理論的最適の{true_ite_mean:.2%}の効率({model_ite_mean/true_ite_mean:.2%})を達成")

# OPEが出力するオラクル平均ITEとの比較
print("\n===== OPEが出力するオラクル平均ITEとの比較 =====")
# 全員処置の場合のオラクル値
# これはOBPが出力するオラクル値に相当する
all_treat_value = true_rewards[:, 1, 0].mean()
print(f"全員処置のオラクル値: {all_treat_value:.4f}")
print(f"真のITE平均値: {total_true_ite/total_samples:.4f}")
print(f"差異: {all_treat_value - total_true_ite/total_samples:.4f}")
print(f"相対誤差: {(all_treat_value - total_true_ite/total_samples)/(total_true_ite/total_samples):.2%}")

# 各ポリシーのOPE推定値と真のITE平均の比較
print("\n各ポリシーのOPE推定値と真のITE平均の比較:")
print(f"1. ランダム抽出ポリシー (真のITE平均: {random_ite_mean:.4f})")
print(f"   - IPW: {random_results_true['ipw']:.4f} (バイアス: {random_results_true['ipw']-random_ite_mean:+.4f}, 相対誤差: {(random_results_true['ipw']-random_ite_mean)/random_ite_mean:+.2%})")
print(f"   - DM: {random_results_true['dm']:.4f} (バイアス: {random_results_true['dm']-random_ite_mean:+.4f}, 相対誤差: {(random_results_true['dm']-random_ite_mean)/random_ite_mean:+.2%})")
print(f"   - DR: {random_results_true['dr']:.4f} (バイアス: {random_results_true['dr']-random_ite_mean:+.4f}, 相対誤差: {(random_results_true['dr']-random_ite_mean)/random_ite_mean:+.2%})")

print(f"2. 真のITE上位ポリシー (真のITE平均: {true_ite_mean:.4f})")
print(f"   - IPW: {true_results_true['ipw']:.4f} (バイアス: {true_results_true['ipw']-true_ite_mean:+.4f}, 相対誤差: {(true_results_true['ipw']-true_ite_mean)/true_ite_mean:+.2%})")
print(f"   - DM: {true_results_true['dm']:.4f} (バイアス: {true_results_true['dm']-true_ite_mean:+.4f}, 相対誤差: {(true_results_true['dm']-true_ite_mean)/true_ite_mean:+.2%})")
print(f"   - DR: {true_results_true['dr']:.4f} (バイアス: {true_results_true['dr']-true_ite_mean:+.4f}, 相対誤差: {(true_results_true['dr']-true_ite_mean)/true_ite_mean:+.2%})")

print(f"3. アップリフトモデルポリシー (真のITE平均: {model_ite_mean:.4f})")
print(f"   - IPW: {model_results_true['ipw']:.4f} (バイアス: {model_results_true['ipw']-model_ite_mean:+.4f}, 相対誤差: {(model_results_true['ipw']-model_ite_mean)/model_ite_mean:+.2%})")
print(f"   - DM: {model_results_true['dm']:.4f} (バイアス: {model_results_true['dm']-model_ite_mean:+.4f}, 相対誤差: {(model_results_true['dm']-model_ite_mean)/model_ite_mean:+.2%})")
print(f"   - DR: {model_results_true['dr']:.4f} (バイアス: {model_results_true['dr']-model_ite_mean:+.4f}, 相対誤差: {(model_results_true['dr']-model_ite_mean)/model_ite_mean:+.2%})")

print("\n結論: OPEの推定手法によってバイアスが大きく異なります。ポリシーの種類によって最適な推定手法が異なります。")
print("- ランダムポリシーの評価にはDRが最適 (相対誤差: {:.2%})".format((random_results_true['dr']-random_ite_mean)/random_ite_mean))
print("- 選択的ポリシーの評価にはIPWが比較的良い性能を示す場合があります")
print("- 複数の推定手法を併用して結果の頻健性を確認することが重要です")

# 真のITEと予測ITEの相関
correlation = np.corrcoef(true_ite, predicted_ite)[0, 1]
print(f"\n真のITEと予測ITEの相関係数: {correlation:.4f}")

# 選択精度
true_positives = np.sum(true_is_topk & model_is_topk)
precision = true_positives / np.sum(model_is_topk)
recall = true_positives / np.sum(true_is_topk)
print(f"上位{k}件の選択精度:")
print(f"  適合率（Precision）: {precision:.2%}")
print(f"  再現率（Recall）: {recall:.2%}")
print(f"  F1スコア: {2 * precision * recall / (precision + recall):.2%}")

# 各ポリシーのOPE結果を表示
# 真の値は各ポリシーの真のITE平均値とし、バイアスはそれとの差とする
def print_ope_results(policy_name, results_est):
    print(f"\n===== {policy_name}のオフポリシー評価結果 =====")
    print("Estimator | Est-model | True-ITE-mean | Bias(est-true) | ATE | Overall-ATE | Sum-pred-ITE | Sum-treated | Sum-true-ITE | Sum-outcome")
    
    # 全体のATEを計算
    overall_ate = (df["y1"] - df["y0"]).mean()
    
    # 各ポリシーの真のITE平均値を取得
    if policy_name == "ランダム抽出ポリシー":
        true_value = random_ite_mean
        policy = random_policy
        pred_ite_sum = 0  # ランダムポリシーには予測値がない
    elif policy_name == "真のITE上位ポリシー":
        true_value = true_ite_mean
        policy = true_policy
        pred_ite_sum = np.sum(true_ite * policy)  # 真のITEを予測値として使用
    elif policy_name == "アップリフトモデルポリシー":
        true_value = model_ite_mean
        policy = model_policy
        pred_ite_sum = np.sum(predicted_ite * policy)  # モデルの予測値を使用
    
    # ポリシーのATEを計算
    treated_indices = np.where(policy == 1)[0]
    ate = np.mean(df["y1"].values[treated_indices] - df["y0"].values[treated_indices])
    
    # 各種合計値を計算
    treated_sum = np.sum(policy)
    true_ite_sum = np.sum(true_ite * policy)
    
    # 実際のアウトカムの合計を計算
    # 処置されたユーザーはy1、非処置ユーザーはy0を使用
    outcome_sum = np.sum(df["y1"].values[treated_indices]) + np.sum(df["y0"].values[~np.isin(np.arange(len(df)), treated_indices)])
    
    for name in ["ipw", "dm", "dr"]:
        e = results_est[name]   # 推定モデルを使ったOPE推定値
        print(f"{name.upper():>9} | {e:.4f}     | {true_value:.4f}      | {e-true_value:+.4f}     | {ate:.4f}     | {overall_ate:.4f}     | {pred_ite_sum:.1f}     | {treated_sum:.0f}     | {true_ite_sum:.1f}     | {outcome_sum:.1f}")

print_ope_results("ランダム抽出ポリシー", random_results_est)
print_ope_results("真のITE上位ポリシー", true_results_est)
print_ope_results("アップリフトモデルポリシー", model_results_est)