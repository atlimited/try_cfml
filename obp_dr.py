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
    """データの読み込みと前処理"""
    print(f"CSVファイルを読み込み: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"データフレームの形状: {df.shape}")
    print(f"カラム: {df.columns.tolist()}")
    print(df.head())
    
    # 特徴量、処置、結果を抽出
    X = df[['age', 'homeownership']].values
    # 特徴量はage, homeownershipに加えて、response_groupも使用
    #X = df[['age', 'homeownership', 'response_group']].values
    A = df['treatment'].values
    R = df['outcome'].values
    
    # 傾向スコア（オラクル情報がある場合はそれを使用、なければ0.5で固定）
    if 'propensity_score' in df.columns:
        p_b = df['propensity_score'].values
    else:
        p_b = np.ones(len(df)) * 0.5
    
    # 元のログ方策の行動確率を計算
    # 処置を受けた場合は傾向スコア、受けなかった場合は1-傾向スコア
    p_b_original = np.zeros(len(df))
    p_b_original[A == 1] = p_b[A == 1]  # 処置を受けた場合
    p_b_original[A == 0] = 1 - p_b[A == 0]  # 処置を受けなかった場合
    
    # バンディットフィードバックの形式に変換
    bandit_feedback = {
        "n_rounds": len(df),
        "n_actions": 2,  # 処置あり/なしの2アクション
        "context": X,
        "action": A,
        "reward": R,
        "position": np.zeros(len(df), dtype=int),  # 単一アクションの場合は0で固定
        "pscore": p_b  # 傾向スコア
    }
    
    # ランダム方策の行動確率を計算
    # 処置割合を20%に設定
    treatment_ratio = 0.2
    
    # ランダムに20%のユーザーを選んで処置する方策
    np.random.seed(42)  # 再現性のために乱数シードを固定
    random_indices = np.random.choice(len(df), size=int(len(df) * treatment_ratio), replace=False)
    
    # 初期化（すべて非処置）
    p_e_random = np.zeros((len(df), 2))
    p_e_random[:, 0] = 1.0  # すべての行の非処置確率を1に
    
    # ランダムに選んだユーザーのみ処置に変更
    p_e_random[random_indices, 0] = 0.0  # 非処置確率を0に
    p_e_random[random_indices, 1] = 1.0  # 処置確率を1に
    
    # 確率分布の検証（各行の合計が1になることを確認）
    assert np.allclose(np.sum(p_e_random, axis=1), 1.0), "ランダム方策が確率分布になっていません"
    
    print(f"ランダム方策のアクション分布の形状: {p_e_random.shape}, 次元数: {p_e_random.ndim}")
    print(f"ランダム方策での処置人数: {np.sum(p_e_random[:, 1])}")
    
    # 3次元に変換（n_rounds, n_actions, len_list）
    p_e_random = p_e_random.reshape(len(df), 2, 1)
    print(f"3次元に変換後のランダム方策の形状: {p_e_random.shape}")
    
    # 元のログ方策の行動確率を計算
    p_e_original = np.zeros((len(df), 2))
    # 処置を受けた人は処置確率=1、非処置確率=0
    p_e_original[A == 1, 1] = 1.0
    p_e_original[A == 1, 0] = 0.0
    # 処置を受けなかった人は処置確率=0、非処置確率=1
    p_e_original[A == 0, 0] = 1.0
    p_e_original[A == 0, 1] = 0.0
    
    # 確率分布の検証（各行の合計が1になることを確認）
    assert np.allclose(np.sum(p_e_original, axis=1), 1.0), "元のログ方策が確率分布になっていません"
    
    p_e_original = p_e_original.reshape(len(df), 2, 1)
    
    # ITE予測値に基づく方策の行動確率を計算
    if 'ite_pred' in df.columns:
        # ITE予測値の上位20%に処置を割り当てる方策
        ite_pred = df['ite_pred'].values
        top_indices = np.argsort(-ite_pred)[:int(len(df) * treatment_ratio)]
        
        # 初期化（すべて非処置）
        p_e_ite = np.zeros((len(df), 2))
        p_e_ite[:, 0] = 1.0  # すべての行の非処置確率を1に
        
        # 上位20%のみ処置に変更
        p_e_ite[top_indices, 0] = 0.0  # 非処置確率を0に
        p_e_ite[top_indices, 1] = 1.0  # 処置確率を1に
        
        print(f"ITE方策の形状: {p_e_ite.shape}, 次元数: {p_e_ite.ndim}")
        
        # 3次元に変換
        p_e_ite = p_e_ite.reshape(len(df), 2, 1)
        print(f"3次元に変換後のITE方策の形状: {p_e_ite.shape}")
        
        print(f"ITE予測値の上位{int(len(df) * treatment_ratio)}人に処置するポリシーを作成しました")
        print(f"処置割合: {treatment_ratio*100:.2f}%")
    else:
        p_e_ite = None
    
    # オラクル情報（真のITE）がある場合、それに基づく最適方策を作成
    if 'true_ite' in df.columns:
        true_ite = df['true_ite'].values
        
        # 真のITEが正の人すべてに処置を割り当てる方策
        positive_ite_indices = np.where(true_ite > 0)[0]
        
        # 初期化（すべて非処置）
        p_e_oracle = np.zeros((len(df), 2))
        p_e_oracle[:, 0] = 1.0  # すべての行の非処置確率を1に
        
        # 真のITEが正の人のみ処置に変更
        p_e_oracle[positive_ite_indices, 0] = 0.0  # 非処置確率を0に
        p_e_oracle[positive_ite_indices, 1] = 1.0  # 処置確率を1に
        
        # 3次元に変換
        p_e_oracle = p_e_oracle.reshape(len(df), 2, 1)
        
        print(f"オラクル最適方策での処置人数: {len(positive_ite_indices)}")
        
        # 真のITEの上位20%に処置を割り当てる方策
        top_true_ite_indices = np.argsort(-true_ite)[:int(len(df) * treatment_ratio)]
        
        # 初期化（すべて非処置）
        p_e_oracle_top = np.zeros((len(df), 2))
        p_e_oracle_top[:, 0] = 1.0  # すべての行の非処置確率を1に
        
        # 上位20%のみ処置に変更
        p_e_oracle_top[top_true_ite_indices, 0] = 0.0  # 非処置確率を0に
        p_e_oracle_top[top_true_ite_indices, 1] = 1.0  # 処置確率を1に
        
        # 3次元に変換
        p_e_oracle_top = p_e_oracle_top.reshape(len(df), 2, 1)
        
        print(f"オラクル上位{int(len(df) * treatment_ratio)}人方策での処置人数: {int(len(df) * treatment_ratio)}")
        
        # 予測ITEと真のITEの相関を計算
        if 'ite_pred' in df.columns:
            correlation = np.corrcoef(ite_pred, true_ite)[0, 1]
            print(f"\n予測ITEと真のITEの相関係数: {correlation:.4f}")
            
            # 予測上位と真のITE上位の一致度を計算
            pred_top = set(np.argsort(-ite_pred)[:int(len(df) * treatment_ratio)])
            true_top = set(np.argsort(-true_ite)[:int(len(df) * treatment_ratio)])
            overlap = len(pred_top.intersection(true_top))
            print(f"予測上位{int(len(df) * treatment_ratio)}人と真のITE上位{int(len(df) * treatment_ratio)}人の一致数: {overlap}人")
            print(f"一致率: {overlap/int(len(df) * treatment_ratio):.2%}")
            
            # 予測方策と真のITE上位方策の一致率
            policy_match = np.mean((p_e_ite[:, 1, 0] > 0) == (p_e_oracle_top[:, 1, 0] > 0))
            print(f"予測方策と真のITE上位方策の一致率: {policy_match:.2%}")
    else:
        p_e_oracle = None
        p_e_oracle_top = None
    
    return X, A, R, p_b, bandit_feedback, p_e_random, p_e_original, p_e_ite, p_e_oracle, p_e_oracle_top, df

def main():
    parser = argparse.ArgumentParser(description='OBPを使用したDoubly Robust評価')
    parser.add_argument('--csv', type=str, default='df.csv', help='評価するCSVファイルのパス')
    args = parser.parse_args()
    
    # 1) データの読み込みと前処理
    X, A, R, p_b, bandit_feedback, p_e_random, p_e_original, p_e_ite, p_e_oracle, p_e_oracle_top, df = load_and_preprocess_data(args.csv)
    
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
    
    # 6) OPE実行 - 元のログ方策
    print("\n=== 元のログ方策の評価 ===")
    result_original = ope.estimate_policy_values(
        action_dist=p_e_original,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model
    )
    
    print("\n=== 元のログ方策のOff-Policy Evaluation Results ===")
    for estimator_name, value in result_original.items():
        print(f"{estimator_name}: {value:.4f}")
    
    # 7) OPE実行 - ITE予測値に基づく方策
    print("\n=== ITE予測値の上位2000人に処置する方策の評価 ===")
    result_ite = ope.estimate_policy_values(
        action_dist=p_e_ite,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model
    )
    
    print("\n=== ITE上位2000人処置方策のOff-Policy Evaluation Results ===")
    for estimator_name, value in result_ite.items():
        print(f"{estimator_name}: {value:.4f}")
    
    # 8) オラクル方策の評価
    results_dict = {
        'ランダム方策': result_random,
        '元のログ方策': result_original,
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
    
    # 9) 全方策比較
    print("\n=== 全方策比較 ===")
    
    # オラクル評価値の取得
    if 'true_ite' in df.columns:
        true_ite = df['true_ite'].values
        
        random_indices = np.where(p_e_random[:, :, 0][..., 1] == 1.0)[0]
        random_policy_true_effect = np.mean(true_ite[random_indices])
        
        original_indices = np.where(p_e_original[:, :, 0][..., 1] == 1.0)[0]
        original_policy_true_effect = np.mean(true_ite[original_indices])
        
        ite_top_indices = np.where(p_e_ite[:, :, 0][..., 1] == 1.0)[0]
        ite_policy_true_effect = np.mean(true_ite[ite_top_indices])
        
        oracle_top_indices = np.where(p_e_oracle_top[:, :, 0][..., 1] == 1.0)[0]
        oracle_top_mean_ite = np.mean(true_ite[oracle_top_indices])
        
        # 表のデータを作成
        comparison_data = {
            '推定手法': ['dr', 'dm', 'ipw', 'オラクル評価'],
            'ランダム方策': [result_random['dr'], result_random['dm'], result_random['ipw'], random_policy_true_effect],
            '元のログ方策': [result_original['dr'], result_original['dm'], result_original['ipw'], original_policy_true_effect],
            'ITE上位方策': [result_ite['dr'], result_ite['dm'], result_ite['ipw'], ite_policy_true_effect]
        }
        
        if 'オラクル最適方策' in results_dict:
            comparison_data['オラクル最適方策'] = [
                results_dict['オラクル最適方策']['dr'],
                results_dict['オラクル最適方策']['dm'],
                results_dict['オラクル最適方策']['ipw'],
                np.mean(true_ite[true_ite > 0])
            ]
            
        if 'オラクル上位方策' in results_dict:
            comparison_data['オラクル上位方策'] = [
                results_dict['オラクル上位方策']['dr'],
                results_dict['オラクル上位方策']['dm'],
                results_dict['オラクル上位方策']['ipw'],
                oracle_top_mean_ite
            ]
        
        # DataFrameに変換して表示
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.set_index('推定手法')
        print(comparison_df.round(4))
    
    # 10) ランダム方策との比較（改善率）
    print("\n=== ランダム方策からの改善率 ===")
    
    # 改善率のデータを作成
    improvement_data = {'推定手法': ['dr', 'dm', 'ipw']}
    
    for policy_name in list(results_dict.keys())[1:]:
        improvements = []
        for estimator in ['dr', 'dm', 'ipw']:
            base_value = result_random[estimator]
            improvement = (results_dict[policy_name][estimator] - base_value) / base_value * 100
            improvements.append(f"{improvement:.1f}%")
        improvement_data[policy_name] = improvements
    
    # DataFrameに変換して表示
    improvement_df = pd.DataFrame(improvement_data)
    improvement_df = improvement_df.set_index('推定手法')
    print(improvement_df)
    
    # 11) オラクル情報（true_ite）を使用した各方策の直接評価
    if 'true_ite' in df.columns:
        print("\n=== オラクル情報（true_ite）による各方策の直接評価 ===")
        true_ite = df['true_ite'].values
        
        # ランダム方策の評価（ランダムに選んだ2000人）
        random_indices = np.where(p_e_random[:, :, 0][..., 1] == 1.0)[0]
        random_policy_true_effect = np.mean(true_ite[random_indices])
        print(f"\nランダム方策が選んだ2000人の真の平均処置効果: {random_policy_true_effect:.4f}")
        
        # 元のログ方策の評価
        original_indices = np.where(p_e_original[:, :, 0][..., 1] == 1.0)[0]
        original_policy_true_effect = np.mean(true_ite[original_indices])
        print(f"\n元のログ方策が選んだ人の真の平均処置効果: {original_policy_true_effect:.4f}")
        
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
        original_policy_positive_count = np.sum(true_ite[original_indices] > 0)
        ite_policy_positive_count = np.sum(true_ite[ite_top_indices] > 0)
        oracle_policy_positive_count = np.sum(true_ite[oracle_top_indices] > 0)
        
        print(f"{'ランダム方策':20} {random_policy_true_effect:.4f}       {random_policy_positive_count}       {random_policy_positive_count/len(random_indices)*100:.1f}%")
        print(f"{'元のログ方策':20} {original_policy_true_effect:.4f}       {original_policy_positive_count}       {original_policy_positive_count/len(original_indices)*100:.1f}%")
        print(f"{'ITE予測上位方策':20} {ite_policy_true_effect:.4f}       {ite_policy_positive_count}       {ite_policy_positive_count/len(ite_top_indices)*100:.1f}%")
        print(f"{'オラクル上位方策':20} {oracle_top_mean_ite:.4f}       {oracle_policy_positive_count}       {oracle_policy_positive_count/len(oracle_top_indices)*100:.1f}%")
        
        # グラフ化可能な形式で保存
        policy_evaluation = {
            'policy': ['ランダム方策', '元のログ方策', 'ITE予測上位方策', 'オラクル上位方策'],
            'true_effect': [random_policy_true_effect, original_policy_true_effect, ite_policy_true_effect, oracle_top_mean_ite],
            'positive_rate': [random_policy_positive_count/len(random_indices), 
                              original_policy_positive_count/len(original_indices),
                              ite_policy_positive_count/len(ite_top_indices),
                              oracle_policy_positive_count/len(oracle_top_indices)]
        }
    
    # 12) 処置効果の成功度合いを計算
    treatment_success = np.mean(R[A == 1])
    control_success = np.mean(R[A == 0])
    print(f"\n=== 観測データからの単純な処置効果 ===")
    print(f"処置群の成功率: {treatment_success:.4f}")
    print(f"対照群の成功率: {control_success:.4f}")
    print(f"観測された処置効果: {treatment_success - control_success:.4f}")

if __name__ == "__main__":
    main()
