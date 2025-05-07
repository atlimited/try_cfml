import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.model_selection import train_test_split
from obp.ope.estimators import InverseProbabilityWeighting, DirectMethod, DoublyRobust
from obp.ope import OffPolicyEvaluation

# 1) 読み込む
df = pd.read_csv("df.csv")

# データの総数、トリートメント総数、アウトカム総数を計算
total_samples = len(df)
total_treatments = df["treatment"].sum()
total_outcomes = df["outcome"].sum()
total_y0 = df["y0"].sum()  # 非処置時の潜在的アウトカム総数
total_y1 = df["y1"].sum()  # 処置時の潜在的アウトカム総数
total_true_ite = (df["y1"] - df["y0"]).sum()  # 真のITE総数

# データの基本統計情報を表示
print("===== データの基本統計情報 =====")
print(f"データの総数: {total_samples}件")
print(f"トリートメント総数: {total_treatments}件 (処置率: {total_treatments/total_samples:.2%})")
print(f"アウトカム総数: {total_outcomes:.2f} (平均: {total_outcomes/total_samples:.4f})")
print(f"非処置時の潜在的アウトカム総数: {total_y0:.2f} (平均: {total_y0/total_samples:.4f})")
print(f"処置時の潜在的アウトカム総数: {total_y1:.2f} (平均: {total_y1/total_samples:.4f})")
print(f"真のITE総数: {total_true_ite:.2f} (平均: {total_true_ite/total_samples:.4f})")

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
k = 2000  # 処置するユーザー数

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
df_policy = pd.read_csv("s_learner_classification_uplift_predictions.csv")  
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

# 8) OPE 実行関数
def run_ope(reward_model, action_dist):
    ope = OffPolicyEvaluation(
        bandit_feedback=bandit_feedback,
        ope_estimators=[InverseProbabilityWeighting(),
                         DirectMethod(),
                         DoublyRobust()]
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

# 10) 各ポリシーのOPE結果を計算
random_results_true = run_ope(true_rewards, random_action_dist)
random_results_est = run_ope(estimated_rewards, random_action_dist)

true_results_true = run_ope(true_rewards, true_action_dist)
true_results_est = run_ope(estimated_rewards, true_action_dist)

model_results_true = run_ope(true_rewards, model_action_dist)
model_results_est = run_ope(estimated_rewards, model_action_dist)

# 11) 結果表示
print("\n===== ポリシー価値比較（真のITEの和と平均） =====")
print(f"ランダム抽出ポリシー: 和={random_ite_sum:.4f}, 平均={random_ite_mean:.4f}")
print(f"真のITE上位ポリシー: 和={true_ite_sum:.4f}, 平均={true_ite_mean:.4f}")
print(f"アップリフトモデルポリシー: 和={model_ite_sum:.4f}, 平均={model_ite_mean:.4f}")
print(f"アップリフトモデルの改善率（ランダム比）: {(model_ite_sum - random_ite_sum) / abs(random_ite_sum):.2%}")
print(f"アップリフトモデルの最適性（理論的最適比）: {(model_ite_sum / true_ite_sum):.2%}")

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
def print_ope_results(policy_name, results_true, results_est, true_ite_mean):
    print(f"\n===== {policy_name}のオフポリシー評価結果 =====")
    print("Estimator | True-model | Est-model | Bias(true) | Bias(est)")
    for name in ["ipw", "dm", "dr"]:
        t = results_true[name]
        e = results_est[name]
        print(f"{name.upper():>9} | {t:.4f}      | {e:.4f}     | {t-true_ite_mean:+.4f}     | {e-true_ite_mean:+.4f}")

print_ope_results("ランダム抽出ポリシー", random_results_true, random_results_est, random_ite_mean)
print_ope_results("真のITE上位ポリシー", true_results_true, true_results_est, true_ite_mean)
print_ope_results("アップリフトモデルポリシー", model_results_true, model_results_est, model_ite_mean)