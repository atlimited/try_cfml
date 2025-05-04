"""
Open Bandit Platformを使用したDoubly Robust評価スクリプト
因果推論データをOff-Policy評価するためのコード
"""
import numpy as np
import pandas as pd
import argparse
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgbm

# OBP インポート
from obp.policy import Random
from obp.ope.regression_model import RegressionModel
from obp.ope.estimators import DoublyRobust, DirectMethod, InverseProbabilityWeighting
from obp.ope import OffPolicyEvaluation

def load_and_preprocess_data(csv_path):
    """CSVファイルからデータを読み込み、OBP形式に変換する"""
    print(f"CSVファイルを読み込み: {csv_path}")
    # データの読み込み
    df = pd.read_csv(csv_path)
    print(f"データフレームの形状: {df.shape}")
    print(f"カラム: {df.columns.tolist()}")
    
    # 必要な列の抽出
    # 特徴量 (context)
    #X = df[['age', 'homeownership']].values
    X = df[['age', 'homeownership', 'response_group']].values
    
    # 処置 (action) - 0/1をactionとして扱う
    A = df['treatment'].values
    
    # 結果 (reward) - outcomeを報酬として扱う
    R = df['outcome'].values
    
    # 傾向スコア (propensity_score)
    if 'propensity_score' in df.columns:
        p_b = df['propensity_score'].values
    else:
        # 傾向スコアがない場合は一様分布を仮定
        p_b = np.ones_like(A) / 2
    
    # OBP用のバンディットフィードバックを作成
    n_rounds = len(df)
    n_actions = 2  # 0と1の2つのアクション
    
    bandit_feedback = {
        "n_rounds": n_rounds,
        "n_actions": n_actions,
        "action": A,
        "reward": R,
        "position": np.zeros(n_rounds, dtype=int),  # positionは使用しない場合は0
        "pscore": p_b,
        "context": X,
        "action_context": np.eye(n_actions),  # 各アクションの特徴量（単位行列）
    }
    
    # ランダム方策のアクション分布を計算（評価対象の方策）
    random_policy = Random(n_actions=n_actions)
    
    # 均等な確率分布ではなく、ランダムに選んだ2000人に処置する方策を作成
    p_e_random = np.zeros((n_rounds, n_actions))
    # デフォルトでは全員に処置しない
    p_e_random[:, 0] = 1.0
    p_e_random[:, 1] = 0.0
    
    # ランダムに2000人を選択して処置
    np.random.seed(42)  # 再現性のために乱数シードを固定
    random_indices = np.random.choice(n_rounds, size=2000, replace=False)
    p_e_random[random_indices, 0] = 0.0
    p_e_random[random_indices, 1] = 1.0
    
    # ディメンションの確認
    print(f"ランダム方策のアクション分布の形状: {p_e_random.shape}, 次元数: {p_e_random.ndim}")
    print(f"ランダム方策での処置人数: {np.sum(p_e_random[:, 1])}")
    
    # 必要なら3次元に変換
    if p_e_random.ndim == 2:
        p_e_random = np.expand_dims(p_e_random, axis=2)
        print(f"3次元に変換後のランダム方策の形状: {p_e_random.shape}")

    # ITE予測値に基づいた方策
    # ite_predの上位2000人を選択する方策
    p_e_ite = np.zeros((n_rounds, n_actions))
    
    if 'ite_pred' in df.columns:
        # ite_predを取得
        ite_pred = df['ite_pred'].values
        
        # ite_predの上位2000人のインデックスを取得
        top_2000_indices = np.argsort(-ite_pred)[:2000]
        
        # 方策分布の作成
        # デフォルトでは全員に処置しない（action=0）
        p_e_ite[:, 0] = 1.0
        p_e_ite[:, 1] = 0.0
        
        # 上位2000人には処置する（action=1）
        p_e_ite[top_2000_indices, 0] = 0.0
        p_e_ite[top_2000_indices, 1] = 1.0
        
        print(f"ITE方策の形状: {p_e_ite.shape}, 次元数: {p_e_ite.ndim}")
        
        # 3次元に変換
        if p_e_ite.ndim == 2:
            p_e_ite = np.expand_dims(p_e_ite, axis=2)
            print(f"3次元に変換後のITE方策の形状: {p_e_ite.shape}")
        
        print(f"ITE予測値の上位2000人に処置するポリシーを作成しました")
        print(f"処置割合: {len(top_2000_indices) / n_rounds * 100:.2f}%")
    else:
        print("警告: ite_predカラムが見つかりません。ランダム方策を使用します。")
        p_e_ite = p_e_random
    
    # オラクル（true_ite）に基づく方策
    p_e_oracle = np.zeros((n_rounds, n_actions))
    p_e_oracle_top = np.zeros((n_rounds, n_actions))
    
    if 'true_ite' in df.columns:
        # true_iteを取得
        true_ite = df['true_ite'].values
        
        # 方策1: true_iteが正の人全員に処置する最適方策
        # デフォルトでは全員に処置しない
        p_e_oracle[:, 0] = 1.0
        p_e_oracle[:, 1] = 0.0
        
        # true_iteが正の人には処置する
        positive_ite_indices = np.where(true_ite > 0)[0]
        p_e_oracle[positive_ite_indices, 0] = 0.0
        p_e_oracle[positive_ite_indices, 1] = 1.0
        
        print(f"オラクル最適方策での処置人数: {len(positive_ite_indices)}")
        
        # 方策2: true_iteの上位2000人に処置する方策
        # デフォルトでは全員に処置しない
        p_e_oracle_top[:, 0] = 1.0
        p_e_oracle_top[:, 1] = 0.0
        
        # true_iteの上位2000人を選択
        oracle_top_indices = np.argsort(-true_ite)[:2000]
        p_e_oracle_top[oracle_top_indices, 0] = 0.0
        p_e_oracle_top[oracle_top_indices, 1] = 1.0
        
        print(f"オラクル上位2000人方策での処置人数: {len(oracle_top_indices)}")
        
        # 3次元に変換
        if p_e_oracle.ndim == 2:
            p_e_oracle = np.expand_dims(p_e_oracle, axis=2)
        if p_e_oracle_top.ndim == 2:
            p_e_oracle_top = np.expand_dims(p_e_oracle_top, axis=2)
        
        # 予測ITEとtrue_ITEの相関係数を計算
        if 'ite_pred' in df.columns:
            corr = np.corrcoef(ite_pred, true_ite)[0, 1]
            print(f"\n予測ITEと真のITEの相関係数: {corr:.4f}")
            
            # 予測上位2000人と真のITE上位2000人の一致度
            match_count = np.intersect1d(top_2000_indices, oracle_top_indices).size
            match_percentage = match_count / 2000 * 100
            print(f"予測上位2000人と真のITE上位2000人の一致数: {match_count}人")
            print(f"一致率: {match_percentage:.2f}%")
            
            # 方策の比較
            pred_policy_agreement = np.mean(p_e_ite[:, :, 0] == p_e_oracle_top[:, :, 0])
            print(f"予測方策と真のITE上位方策の一致率: {pred_policy_agreement * 100:.2f}%")
    else:
        print("警告: true_iteカラムが見つかりません。オラクル方策は評価できません。")
        p_e_oracle = None
        p_e_oracle_top = None
    
    return X, A, R, p_b, bandit_feedback, p_e_random, p_e_ite, p_e_oracle, p_e_oracle_top, df

def main():
    parser = argparse.ArgumentParser(description='OBPを使用したDoubly Robust評価')
    parser.add_argument('--csv', type=str, default='df.csv', help='評価するCSVファイルのパス')
    args = parser.parse_args()
    
    # 1) データの読み込みと前処理
    X, A, R, p_b, bandit_feedback, p_e_random, p_e_ite, p_e_oracle, p_e_oracle_top, df = load_and_preprocess_data(args.csv)
    
    print("\n=== データサンプル ===")
    print("特徴量 (X)の最初の5行:")
    print(X[:5])
    print("\n処置 (A)の最初の5行:")
    print(A[:5])
    print("\n結果 (R)の最初の5行:")
    print(R[:5])
    print("\n傾向スコア (p_b)の最初の5行:")
    print(p_b[:5])
    
    # 2) 回帰モデルの初期化と学習
    rm = RegressionModel(
        base_model=lgbm.LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1),
        n_actions=bandit_feedback["n_actions"],
        len_list=1  # 単一アクションの場合は1
    )
    
    # 3) fit_predictで予測報酬を取得
    print("\n回帰モデルを学習して予測報酬を取得しています...")
    estimated_rewards_by_reg_model = rm.fit_predict(
        context=X,
        action=A,
        reward=R,
        pscore=p_b,
        position=bandit_feedback.get("position", None),
        action_dist=None
    )
    
    # 4) Off-Policy Evaluationのセットアップ
    print("\nOff-Policy評価を設定しています...")
    ope = OffPolicyEvaluation(
        bandit_feedback=bandit_feedback,
        ope_estimators=[
            DoublyRobust(),
            DirectMethod(), 
            InverseProbabilityWeighting()
        ]
    )
    
    # 5) OPE実行 - ランダム方策
    print("\n=== ランダム方策の評価 ===")
    result_random = ope.estimate_policy_values(
        action_dist=p_e_random,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model
    )
    
    print("\n=== ランダム方策のOff-Policy Evaluation Results ===")
    for estimator_name, value in result_random.items():
        print(f"{estimator_name}: {value:.4f}")
    
    # 6) OPE実行 - ITE予測値に基づく方策
    print("\n=== ITE予測値の上位2000人に処置する方策の評価 ===")
    result_ite = ope.estimate_policy_values(
        action_dist=p_e_ite,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model
    )
    
    print("\n=== ITE上位2000人処置方策のOff-Policy Evaluation Results ===")
    for estimator_name, value in result_ite.items():
        print(f"{estimator_name}: {value:.4f}")
    
    # 7) オラクル方策の評価
    results_dict = {
        'ランダム方策': result_random,
        'ITE上位方策': result_ite
    }
    
    if p_e_oracle is not None:
        print("\n=== オラクル最適方策の評価（true_ite > 0の全員に処置） ===")
        result_oracle = ope.estimate_policy_values(
            action_dist=p_e_oracle,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model
        )
        
        print("\n=== オラクル最適方策のOff-Policy Evaluation Results ===")
        for estimator_name, value in result_oracle.items():
            print(f"{estimator_name}: {value:.4f}")
        
        results_dict['オラクル最適方策'] = result_oracle
    
    if p_e_oracle_top is not None:
        print("\n=== オラクル上位2000人方策の評価 ===")
        result_oracle_top = ope.estimate_policy_values(
            action_dist=p_e_oracle_top,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model
        )
        
        print("\n=== オラクル上位2000人方策のOff-Policy Evaluation Results ===")
        for estimator_name, value in result_oracle_top.items():
            print(f"{estimator_name}: {value:.4f}")
        
        results_dict['オラクル上位方策'] = result_oracle_top
    
    # 8) 全方策比較
    print("\n=== 全方策比較 ===")
    
    # オラクル評価値の取得
    if 'true_ite' in df.columns:
        true_ite = df['true_ite'].values
        
        # 各方策が選んだ人々のtrue_iteの平均
        random_indices = np.where(p_e_random[:, :, 0][..., 1] == 1.0)[0]
        random_policy_true_effect = np.mean(true_ite[random_indices])
        
        ite_top_indices = np.where(p_e_ite[:, :, 0][..., 1] == 1.0)[0]
        ite_policy_true_effect = np.mean(true_ite[ite_top_indices])
        
        oracle_top_indices = np.argsort(-true_ite)[:2000]
        oracle_top_mean_ite = np.mean(true_ite[oracle_top_indices])
        
        # オラクル行を追加
        print(f"{'推定手法':20} " + " ".join([f"{name:20}" for name in results_dict.keys()]))
        
        # DR, DM, IPWの結果表示
        for estimator in ['dr', 'dm', 'ipw']:
            values = [results[estimator] for results in results_dict.values()]
            values_str = " ".join([f"{value:.4f}              " for value in values])
            print(f"{estimator:20} {values_str}")
        
        # オラクル評価行を追加
        oracle_values = [random_policy_true_effect, ite_policy_true_effect]
        if 'オラクル最適方策' in results_dict:
            # true_ite > 0の人全員の平均値も加える
            positive_ite_mean = np.mean(true_ite[true_ite > 0])
            oracle_values.append(positive_ite_mean)
        if 'オラクル上位方策' in results_dict:
            oracle_values.append(oracle_top_mean_ite)
        
        oracle_values_str = " ".join([f"{value:.4f}              " for value in oracle_values])
        print(f"{'オラクル評価':20} {oracle_values_str}")
    else:
        # 通常の表示（オラクルなし）
        print(f"{'推定手法':20} " + " ".join([f"{name:20}" for name in results_dict.keys()]))
        
        for estimator in ['dr', 'dm', 'ipw']:
            values = [results[estimator] for results in results_dict.values()]
            values_str = " ".join([f"{value:.4f}              " for value in values])
            print(f"{estimator:20} {values_str}")
    
    # 9) ランダム方策との比較（改善率）
    print("\n=== ランダム方策からの改善率 ===")
    print(f"{'推定手法':20} " + " ".join([f"{name:20}" for name in list(results_dict.keys())[1:]]))
    
    for estimator in ['dr', 'dm', 'ipw']:
        random_value = results_dict['ランダム方策'][estimator]
        improvements = [(results[estimator] - random_value) / random_value * 100 
                        for name, results in results_dict.items() 
                        if name != 'ランダム方策']
        improvements_str = " ".join([f"{imp:.1f}%              " for imp in improvements])
        print(f"{estimator:20} {improvements_str}")
    
    # 10) オラクル情報（true_ite）を使用した各方策の直接評価
    if 'true_ite' in df.columns:
        print("\n=== オラクル情報（true_ite）による各方策の直接評価 ===")
        true_ite = df['true_ite'].values
        
        # ランダム方策の評価（ランダムに選んだ2000人）
        random_indices = np.where(p_e_random[:, :, 0][..., 1] == 1.0)[0]
        random_policy_true_effect = np.mean(true_ite[random_indices])
        print(f"\nランダム方策が選んだ2000人の真の平均処置効果: {random_policy_true_effect:.4f}")
        
        # ITE予測上位方策の評価
        ite_top_indices = np.where(p_e_ite[:, :, 0][..., 1] == 1.0)[0]
        ite_policy_true_effect = np.mean(true_ite[ite_top_indices])
        print(f"ITE予測上位2000人の真の平均処置効果: {ite_policy_true_effect:.4f}")
        print(f"ランダム方策からの改善率: {(ite_policy_true_effect - random_policy_true_effect) / random_policy_true_effect * 100:.1f}%")
        
        # 真のITE統計
        positive_ite_count = np.sum(true_ite > 0)
        print(f"\n真のITE > 0の人数: {positive_ite_count}人 ({positive_ite_count/len(true_ite)*100:.1f}%)")
        print(f"真のITEの平均値: {np.mean(true_ite):.4f}")
        print(f"真のITEの最大値: {np.max(true_ite):.4f}")
        print(f"真のITEの最小値: {np.min(true_ite):.4f}")
        
        # 真のITE上位2000人の統計
        oracle_top_indices = np.argsort(-true_ite)[:2000]
        oracle_top_mean_ite = np.mean(true_ite[oracle_top_indices])
        print(f"\n真のITE上位2000人の平均処置効果: {oracle_top_mean_ite:.4f}")
        print(f"ランダム方策からの改善率: {(oracle_top_mean_ite - random_policy_true_effect) / random_policy_true_effect * 100:.1f}%")
        print(f"ITE予測上位方策からの改善率: {(oracle_top_mean_ite - ite_policy_true_effect) / ite_policy_true_effect * 100:.1f}%")
        
        # 方策の意思決定の比較表
        print("\n=== 方策の意思決定の比較 ===")
        print("各方策がどのITE値の人々を選択したか：")
        print(f"{'方策':20} {'真のITE平均値':15} {'選んだ正のITE人数':15} {'正のITE割合':15}")
        
        random_policy_positive_count = np.sum(true_ite[random_indices] > 0)
        ite_policy_positive_count = np.sum(true_ite[ite_top_indices] > 0)
        oracle_policy_positive_count = np.sum(true_ite[oracle_top_indices] > 0)
        
        print(f"{'ランダム方策':20} {random_policy_true_effect:.4f}       {random_policy_positive_count}       {random_policy_positive_count/len(random_indices)*100:.1f}%")
        print(f"{'ITE予測上位方策':20} {ite_policy_true_effect:.4f}       {ite_policy_positive_count}       {ite_policy_positive_count/len(ite_top_indices)*100:.1f}%")
        print(f"{'オラクル上位方策':20} {oracle_top_mean_ite:.4f}       {oracle_policy_positive_count}       {oracle_policy_positive_count/len(oracle_top_indices)*100:.1f}%")
        
        # グラフ化可能な形式で保存
        policy_evaluation = {
            'policy': ['ランダム方策', 'ITE予測上位方策', 'オラクル上位方策'],
            'true_effect': [random_policy_true_effect, ite_policy_true_effect, oracle_top_mean_ite],
            'positive_rate': [random_policy_positive_count/len(random_indices), 
                              ite_policy_positive_count/len(ite_top_indices),
                              oracle_policy_positive_count/len(oracle_top_indices)]
        }
    
    # 11) 処置効果の成功度合いを計算
    treatment_success = np.mean(R[A == 1])
    control_success = np.mean(R[A == 0])
    print(f"\n=== 観測データからの単純な処置効果 ===")
    print(f"処置群の成功率: {treatment_success:.4f}")
    print(f"対照群の成功率: {control_success:.4f}")
    print(f"観測された処置効果: {treatment_success - control_success:.4f}")

if __name__ == "__main__":
    main()
