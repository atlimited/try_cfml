"""
因果推論モデルを用いた処置効果推定
モジュール化されたコード構造を使用
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import warnings
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score, precision_score, recall_score, f1_score
from scipy.stats import pearsonr
from typing import Dict, Tuple, List, Optional, Union
import argparse
from tqdm import tqdm

# 自作モジュールのインポート
from data_preprocessing import prepare_data
from causal_models import (
    train_all_models
)
from model_evaluation import (
    evaluate_all_models_cv,
    print_cv_results,
    run_cv_with_statistics,
    evaluate_model_cv,
    evaluate_cv_with_top_n
)
from causal_evaluation import (
    compute_propensity_scores,
    estimate_ate_with_ipw,
    evaluate_models,
    evaluate_targeting,
    evaluate_policy_efficiency,
    plot_uplift_curve
)
from outcome_predictors import (
    train_outcome_predictor
)
from model_utils import (
    ensure_dataframe,
    get_model_trainer
)



def estimate_ate_with_propensity_scores(df, X, treatment, outcome):
    """
    傾向スコアを用いたATE推定
    """
    # 傾向スコアによるATE推定
    print("\n===== 傾向スコアによるATE推定値 =====")
    true_ate = df['true_ite'].mean() if 'true_ite' in df.columns else None
    if true_ate is not None:
        print(f"真のATE (y1-y0の平均): {true_ate:.4f}")
    
    # 観測データからの素朴なATE計算
    observed_ate = df.groupby('treatment')['outcome'].mean().diff().iloc[-1]
    print(f"観測データからの素朴なATE推定値 (処置群vs対照群の平均の差): {observed_ate:.4f}")
    
    # 傾向スコアの取得
    propensity_score = df['propensity_score'].values
    
    # LightGBM傾向スコアを使用
    from causal_evaluation import compute_propensity_scores, estimate_ate_with_ipw
    
    # LightGBM傾向スコアの計算
    p_score_lgbm = compute_propensity_scores(X, treatment, method='lightgbm')
    
    # Logistic傾向スコアの計算
    p_score_logistic = compute_propensity_scores(X, treatment, method='logistic')
    
    # Seriesを配列に変換して渡す
    treatment_array = treatment.values if hasattr(treatment, 'values') else np.array(treatment)
    outcome_array = outcome.values if hasattr(outcome, 'values') else np.array(outcome)
    
    # IPWでATEを推定
    if 'propensity_score' in df.columns:
        ate_oracle = estimate_ate_with_ipw(outcome_array, treatment_array, propensity_score)
        print(f"オラクル傾向スコアによるIPW-ATE推定値: {ate_oracle:.4f}")
    
    ate_lgbm = estimate_ate_with_ipw(outcome_array, treatment_array, p_score_lgbm)
    print(f"LightGBM傾向スコアによるIPW-ATE推定値: {ate_lgbm:.4f}")
    
    ate_logistic = estimate_ate_with_ipw(outcome_array, treatment_array, p_score_logistic)
    print(f"Logistic傾向スコアによるIPW-ATE推定値: {ate_logistic:.4f}")
    
    # 傾向スコアを返す（後続の処理で使用するため）
    return p_score_lgbm

def compare_propensity_scores(X, treatment, oracle_score=None):
    """異なる手法で傾向スコアを計算して比較"""
    propensity_scores = {}
    
    # オラクルスコア
    if oracle_score is not None:
        propensity_scores['oracle'] = oracle_score
        print(f"オラクル傾向スコアの統計: 最小値={oracle_score.min():.4f}, 最大値={oracle_score.max():.4f}, 平均={oracle_score.mean():.4f}")
    
    # LightGBM傾向スコア
    from causal_evaluation import compute_propensity_scores
    p_score_lgbm = compute_propensity_scores(X, treatment, method='lightgbm')
    propensity_scores['lightgbm'] = p_score_lgbm
    print(f"LightGBM傾向スコアの統計: 最小値={p_score_lgbm.min():.4f}, 最大値={p_score_lgbm.max():.4f}, 平均={p_score_lgbm.mean():.4f}")
    
    # Logistic傾向スコア
    p_score_logistic = compute_propensity_scores(X, treatment, method='logistic')
    propensity_scores['logistic'] = p_score_logistic
    print(f"Logistic傾向スコアの統計: 最小値={p_score_logistic.min():.4f}, 最大値={p_score_logistic.max():.4f}, 平均={p_score_logistic.mean():.4f}")
    
    return propensity_scores

def main():
    """
    メイン関数
    """
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='CausalMLを使った因果推論モデルの評価')
    parser.add_argument('--causal_model_name', type=str, default=None,
                      help='評価するモデル名（従来方式: s_learner_classification, t_learner_regression等）')
    parser.add_argument('--uplift_method', type=str, default=None,
                      help='評価するモデルタイプ (s_learner, t_learner, x_learner, r_learner, dr_learner)')
    parser.add_argument('--prediction_method', type=str, default=None,
                      help='評価する学習器タイプ (classification, regression)')
    parser.add_argument('--cv', action='store_true',
                      help='クロスバリデーションを実行するかどうか')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='分類評価に使用する閾値（デフォルト: 0.5）')
    parser.add_argument('--folds', type=int, default=2,
                      help='交差検証のfold数（デフォルト: 2）')
    args = parser.parse_args()
    
    # モデル名の決定（新方式と旧方式の両方をサポート）
    #if args.model is not None:
    #    model_name = args.model
    #    # model引数が指定された場合
    #    if args.model.endswith('_classification') or args.model.endswith('_regression'):
    #        # すでに完全な形式（例：'s_learner_classification'）
    #        model_type = args.model
    #        base_model_type, learner_type = args.model.rsplit('_', 1)
    #    else:
    #        # モデル名のみ指定された場合（例：'s_learner'）
    #        base_model_type = args.model
    #        learner_type = args.learner_type if args.learner_type else 'classification'
    #        model_type = f"{base_model_type}_{learner_type}"
    #elif args.model_type is not None:
    #    # モデルタイプが指定された場合
    #    base_model_type = args.model_type
    #    learner_type = args.learner_type if args.learner_type else 'classification'
    #    model_type = f"{base_model_type}_{learner_type}"
    #else:
    #    # デフォルト値
    #    base_model_type = "s_learner"
    #    learner_type = "classification"
    #    model_type = f"{base_model_type}_{learner_type}"

    if args.causal_model_name is not None:
        causal_model_name = args.causal_model_name
        uplift_method, prediction_method = causal_model_name.split("_")
    elif args.uplift_method is not None and args.prediction_method is not None:
        causal_model_name = f"{args.uplift_method}_{args.prediction_method}"j
    else:
        print("invalid model name")
        exit()
    
    print(f"\n===== {causal_model_name}モデルの評価を開始 =====")
    
    # データ読み込み
    df = pd.read_csv("df.csv")

    # データフレームの構造確認
    print("\n===== データフレームの構造 =====")
    print("カラム名:", df.columns.tolist())
    print("最初の5行:")
    print(df.head())

    # データの前処理
    # 基本的特徴量のみ使用する場合は'minimal'、すべての特徴量を使用する場合は'full'
    #feature_level = 'minimal'
    feature_level = 'original'
    df, X, treatment, outcome, oracle_info, feature_cols = prepare_data(df, feature_level)

    print("X:\n", X)
    print("treatment:\n", treatment)
    print("outcome:\n", outcome)

    # オラクルの傾向スコアを取得（存在する場合）
    oracle_propensity = None
    if 'propensity_score' in df.columns:
        oracle_propensity = df['propensity_score'].values
        # 極端な値を避けるためのクリッピング
        oracle_propensity = np.clip(oracle_propensity, 0.05, 0.95)
        print(f"\n===== オラクル傾向スコア情報 =====")
        print(f"オラクル傾向スコアの統計: 最小値={oracle_propensity.min():.4f}, 最大値={oracle_propensity.max():.4f}, 平均={oracle_propensity.mean():.4f}")

    # 傾向スコアの計算（LightGBMとロジスティック回帰の両方）
    propensity_scores = compare_propensity_scores(X, treatment, oracle_score=oracle_propensity)
    
    # LightGBM傾向スコアを使用
    p_score_lgbm = propensity_scores['lightgbm']
    p_score_logistic = propensity_scores['logistic']
    
    # 傾向スコアの統計情報表示
    print("\n===== 複数の傾向スコア計算方法を比較 =====")
    if 'oracle' in propensity_scores:
        p_score_oracle = propensity_scores['oracle']
        print(f"オラクル傾向スコアの統計: 最小値={p_score_oracle.min():.4f}, 最大値={p_score_oracle.max():.4f}, 平均={p_score_oracle.mean():.4f}")
    
    print(f"LightGBM傾向スコアの統計: 最小値={p_score_lgbm.min():.4f}, 最大値={p_score_lgbm.max():.4f}, 平均={p_score_lgbm.mean():.4f}")
    print(f"Logistic傾向スコアの統計: 最小値={p_score_logistic.min():.4f}, 最大値={p_score_logistic.max():.4f}, 平均={p_score_logistic.mean():.4f}")

    # 傾向スコアによるATE推定
    p_score = estimate_ate_with_propensity_scores(df, X, treatment, outcome)
    
    # ここから、指定したモデルだけを評価するように変更
    # 分類器と回帰器のモデル比較部分は、指定したモデルのみ実行
    if args.model or (args.model_type and args.learner_type):
        # モデルトレーナーの取得
        model_trainer = get_model_trainer(causal_model_name)

        if model_trainer:
            # モデルタイプの判別
            #learner_type = 'classification' if 'classification' in model_name else 'regression'

            if args.cv:
                print(f"\n===== {causal_model_name}のクロスバリデーション評価 =====")
                cv_top_n_results = evaluate_cv_with_top_n(model_trainer, X, outcome, treatment, fold_count=5)
                #print(cv_top_n_results)
            
            # モデルを学習し、ITE予測
            print(f"\n===== {causal_model_name}のITE予測 =====")

            # モデルのトレーニング
            print(f"モデル学習開始: {causal_model_name}")
            print(f"特徴量: {X.columns.tolist()}")
            print(f"特徴量の形状: {X.shape}")
            model = model_trainer(X, treatment, outcome, propensity_score=p_score)
            print(f"モデル学習完了: {model}, モデルタイプ: {type(model)}")

            # ITE予測
            print(f"ITE予測開始: {causal_model_name}")
            ite_pred = model.predict(X)
            print(f"ITE予測値の形状: {ite_pred.shape}")
            print(ite_pred)
            print(f"ITE予測完了: {causal_model_name}, 予測形状: {ite_pred.shape}")

            # データフレームに結果を保存
            result_df = df.copy()
            result_df['ite_pred'] = ite_pred
            print(result_df)
            
            # CSV出力
            result_file = f'{causal_model_name}_uplift_predictions.csv'
            result_df.to_csv(result_file, index=False)
            print(f"予測結果を保存しました: {result_file}")

            # DoublyRobust評価
            from obp.dataset import OpenBanditDataset
            from obp.policy import RandomPolicy
            from obp.ope import DoublyRobust as DRLearner
            
            X, A, R, p_b = (
                df[['age', 'homeownership']].values,
                df['treatment'].values,
                df['outcome'].values,
                p_score  # または適切な傾向スコア（propensity_score）
            )
            
            policy = RandomPolicy(n_actions=2, random_state=42)
            p_e = policy.probability(X)
            
            dr = DRLearner()  
            V_dr = dr.estimate_policy_value(reward=R, action=A, pscore=p_b, evaluation_policy_pscore=p_e)
            print("DR-OPE 推定ポリシー価値:", V_dr)


            # モデルを学習し、真値ITEを使った評価
            print(f"\n===== {model_type}の真値ITEを使った評価 =====")

            # 真のITEがある場合、アップリフト評価を実行
            if 'true_ite' in df.columns:
                print("真のITEを使った評価を実行します")
                true_ite = df['true_ite'].values
                
                # 真のITEに基づく理想的な処置割り当て
                ideal_treatment = np.where(true_ite > 0, 1, 0)
                    
                # 理想的な処置効果の合計を計算
                ideal_effect_mask = (ideal_treatment == 1)
                ideal_effect_sum = np.sum(true_ite[ideal_effect_mask])
                
                # ITEの予測値でソート（降順）
                sorted_indices = np.argsort(-ite_pred.flatten())
                
                # 実際の処置による効果の合計を計算
                treatment_array = np.array(treatment)
                actual_effect_mask = (treatment_array == 1)
                actual_effect_sum = np.sum(true_ite[actual_effect_mask])
                
                # 実際に処置した人数を計算
                actual_treatment_count = np.sum(treatment_array)
                
                # 実際の処置人数での評価を追加
                top_actual_count_indices = sorted_indices[:actual_treatment_count]
                top_actual_count_effect_sum = np.sum(true_ite[top_actual_count_indices])
                top_actual_count_efficiency = top_actual_count_effect_sum / ideal_effect_sum if ideal_effect_sum > 0 else 0
                top_actual_count_improvement = top_actual_count_effect_sum / actual_effect_sum if actual_effect_sum > 0 else 0
                
                print(f"実際に処置した人数: {actual_treatment_count}")
                print("  N\t効果総和\t効率\t改善率\t備考")
                print(f"  {actual_treatment_count}\t{top_actual_count_effect_sum:.2f}\t{top_actual_count_efficiency:.2f} ({top_actual_count_efficiency*100:.1f}%)\t{top_actual_count_improvement:.2f}倍\t実際の処置人数")
                
                # 上位N人による評価（実際の処置数を含む）
                print("\n上位N人に処置した場合の効果:")
                print("  N\t効果総和\t効率\t改善率\t備考")
                # その他の1000刻みの評価も表示
                for n in range(1000, 10001, 1000):
                    # 上位n人を処置対象として選択
                    top_n_indices = sorted_indices[:n]
                    top_n_effect_sum = np.sum(true_ite[top_n_indices])
                    
                    # 効率とROI計算
                    top_n_efficiency = top_n_effect_sum / ideal_effect_sum if ideal_effect_sum > 0 else 0
                    top_n_improvement = top_n_effect_sum / actual_effect_sum if actual_effect_sum > 0 else 0
                    
                    # 実際の処置人数と同じ場合は注釈を追加
                    note = "実際の処置人数" if n == actual_treatment_count else ""
                    
                    print(f"  {n}\t{top_n_effect_sum:.2f}\t{top_n_efficiency:.2f} ({top_n_efficiency*100:.1f}%)\t{top_n_improvement:.2f}倍\t{note}")

        else:
            print(f"エラー: モデル{causal_model_name}の学習関数がありません")
    
    else:
        # 引数指定がない場合は、以前のコードで全モデルを比較
        print("\n===== 分類器と回帰器の両方を使ったモデル比較 =====")
        # ここに従来の全モデル比較コードが続く

    print(f"\n===== {causal_model_name}モデルの評価完了 =====")

if __name__ == "__main__":
    main()
