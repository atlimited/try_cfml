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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern

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
    evaluate_cv_with_top_n,
    calculate_top_n_ate
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
                      help='評価するモデルタイプ (classification, regression)')
    parser.add_argument('--model_type', type=str, default='lgbm', choices=['lgbm', 'gpr'],
                      help='使用するモデルタイプ (lgbm: LightGBM, gpr: ガウス過程回帰)')
    parser.add_argument('--kernel_type', type=str, default='rbf', choices=['rbf', 'matern', 'combined'],
                      help='GPRを使用する場合のカーネルタイプ (rbf, matern, combined)')
    parser.add_argument('--subsample', type=float, default=0.2, 
                      help='データのサブサンプリング率（0.0-1.0）。GPRモデルの高速化に有効。')

    parser.add_argument('--verbose', action='store_true',
                      help='詳細な進捗表示を行うかどうか')
    parser.add_argument('--cv', action='store_true',
                      help='クロスバリデーションを実行するかどうか')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='分類評価に使用する閾値（デフォルト: 0.5）')
    parser.add_argument('--folds', type=int, default=2,
                      help='交差検証のfold数（デフォルト: 2）')
    args = parser.parse_args()
    
    # 引数の解析
    if args.causal_model_name is not None and args.uplift_method is None and args.prediction_method is None:
        # 従来方式: s_learner_classificationのような形式
        causal_model_name = args.causal_model_name
        # s_learner_classificationのように複数の_がある場合に対応
        if '_' in causal_model_name:
            parts = causal_model_name.split("_")
            if len(parts) >= 2:
                uplift_method = parts[0]
                prediction_method = "_".join(parts[1:])
            else:
                print("無効なモデル名です。'uplift_method_prediction_method'の形式で指定してください。")
                exit()
        else:
            # _がない場合はuplift_methodのみと解釈し、デフォルトのprediction_methodを使用
            uplift_method = causal_model_name
            prediction_method = "classification"  # デフォルト値
    elif args.uplift_method is not None and args.prediction_method is not None:
        # 新方式: --uplift_method s_learner --prediction_method classification
        uplift_method = args.uplift_method
        prediction_method = args.prediction_method
        causal_model_name = f"{uplift_method}_{prediction_method}"
    elif args.causal_model_name is not None and (args.uplift_method is not None or args.prediction_method is not None):
        # 混合方式: 両方指定された場合は個別指定を優先
        uplift_method = args.uplift_method if args.uplift_method is not None else args.causal_model_name
        prediction_method = args.prediction_method if args.prediction_method is not None else "classification"
        causal_model_name = f"{uplift_method}_{prediction_method}"
    else:
        print("モデル名が指定されていません。--causal_model_name または --uplift_method と --prediction_method を指定してください。")
        exit()
    
    print(f"\n===== {causal_model_name}モデルの評価を開始 =====")
    
    # データ読み込み
    #df = pd.read_csv("df.csv")
    #df = pd.read_csv("matched_df.csv")
    #df = pd.read_csv("df_only_1group.csv")
    df = pd.read_csv("df_balanced_group.csv")
    
    # 元のデータを保存（後で予測に使用）
    df_orig = df.copy()

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

    # サブサンプリング（GPRモデルの高速化のため）
    if args.subsample < 1.0 and args.subsample > 0.0:
        from sklearn.model_selection import train_test_split
        sample_size = int(len(df) * args.subsample)
        print(f"\n===== データをサブサンプリング（{args.subsample * 100:.1f}%、{sample_size}件）=====")
        # 層化サンプリング（処置群と対照群の比率を保持）
        indices, _ = train_test_split(
            np.arange(len(df)), 
            test_size=1.0-args.subsample, 
            stratify=treatment,
            random_state=42
        )
        df = df.iloc[indices].reset_index(drop=True)
        X = X.iloc[indices].reset_index(drop=True)
        treatment = treatment.iloc[indices].reset_index(drop=True)
        outcome = outcome.iloc[indices].reset_index(drop=True)
        print(f"サブサンプリング後のデータサイズ: {len(df)}件")
        print(f"処置群: {sum(treatment)}件 ({sum(treatment)/len(treatment)*100:.1f}%)")
        print(f"対照群: {len(treatment)-sum(treatment)}件 ({(1-sum(treatment)/len(treatment))*100:.1f}%)")

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
    #if args.causal_model_name:
        # モデルトレーナーの取得
    model_trainer = get_model_trainer(
        uplift_method, 
        prediction_method, 
        model_type=args.model_type, 
        kernel_type=args.kernel_type,
        verbose=args.verbose
    )

    if model_trainer:
        # モデルタイプの判別
        model_type_str = f"{args.model_type}"
        if args.model_type == 'gpr':
            model_type_str += f"_{args.kernel_type}"
        
        print(f"\n===== {causal_model_name} ({model_type_str})の評価 =====")

        if args.cv:
            print(f"\n===== {causal_model_name} ({model_type_str})のクロスバリデーション評価 =====")
            # GPRモデルを使用する場合は、use_gpr=Trueを渡す
            if args.model_type == 'gpr':
                from causal_models import evaluate_all_models_cv
                cv_results = evaluate_all_models_cv(
                    X, treatment, outcome, 
                    true_ite=df['true_ite'].values if 'true_ite' in df.columns else None,
                    propensity_score=p_score, 
                    n_splits=args.folds, 
                    random_state=42,
                    use_gpr=True,
                    kernel_type=args.kernel_type
                )
                print_cv_results(cv_results)
            else:
                cv_top_n_results = evaluate_cv_with_top_n(model_trainer, X, outcome, treatment, fold_count=args.folds)
                #print(cv_top_n_results)
        
        # モデルを学習し、ITE予測
        print(f"\n===== {causal_model_name} ({model_type_str})のITE予測 =====")

        # モデルのトレーニング
        print(f"モデル学習開始: {causal_model_name}")
        print(f"特徴量: {X.columns.tolist()}")
        print(f"特徴量の形状: {X.shape}")
        model = model_trainer(X, treatment, outcome, propensity_score=p_score)
        print(f"モデル学習完了: {model}, モデルタイプ: {type(model)}")

        # 元の全データを読み込み（予測用）
        print("元の全データを読み込み中（予測用）...")
        full_df = df_orig.copy()
        
        # 予測用の特徴量を準備
        # 元の特徴量リストを使用
        X_full = full_df[['age', 'homeownership']].copy()
        print(f"予測用全データの形状: {X_full.shape}")
        
        # ITE予測
        print(f"ITE予測開始: {causal_model_name}")
        ite_pred = model.predict(X_full)
        print(f"ITE予測値の形状: {ite_pred.shape}")
        print(ite_pred[:10])  # 最初の10件だけ表示
        print(f"ITE予測完了: {causal_model_name}, 予測形状: {ite_pred.shape}")

        # データフレームに結果を保存
        result_df = full_df.copy()
        result_df['ite_pred'] = ite_pred
        print(result_df.head())
        
        # CSV出力
        result_file = f'{causal_model_name}_{args.model_type}'
        if args.model_type == 'gpr':
            result_file += f'_{args.kernel_type}'
        result_file += '_uplift_predictions.csv'
        result_df.to_csv(result_file, index=False)
        print(f"予測結果を保存しました: {result_file}")
        print(result_df.shape)

        # 予測上位N人のATE計算
        print("\n===== 予測上位N人のATE =====")
        from model_evaluation import calculate_top_n_ate
        
        # 全データのtreatmentとoutcomeを取得
        full_treatment = full_df['treatment'].values
        full_outcome = full_df['outcome'].values
        
        # 計算するN値のリスト
        ns = [500, 1000, 2000]
        
        top_n_results = calculate_top_n_ate(ite_pred, full_treatment, full_outcome, ns=ns)

        # 結果の表示
        for n, result in top_n_results.items():
            print(f"\n上位{n}人の結果:")
            print(f"  処置群サイズ: {result['n_treated']}")
            print(f"  非処置群サイズ: {result['n_control']}")
            
            if result.get('method') == 'observed':
                # 通常の計算結果がある場合
                print(f"  上位{n}人の処置群平均: {result['top_n_treated_mean']:.4f}")
                print(f"  上位{n}人の非処置群平均: {result['top_n_control_mean']:.4f}")
                print(f"  上位{n}人の処置効果: {result['top_n_effect']:.4f}")
                
                # 元の処置戦略との比較
                original_count = result['original_treatment_count']
                print(f"\n  元の処置戦略との比較（処置数: {original_count}人）:")
                print(f"  同数処置時の効果比較:")
                print(f"    上位{n}人処置の効果: {result['top_n_virtual_effect']:.4f}")
                print(f"    元の処置戦略の効果: {result['original_treatment_effect'] * n / original_count if n <= original_count else result['original_treatment_effect']:.4f}")
                print(f"    改善率: {result['improvement_ratio']:.2f}倍")
            else:
                # 処置群または非処置群のサンプルが不足している場合
                print(f"  注意: 上位{n}人の中に処置群と非処置群の両方が十分にありません")
            
            ## 真のITEがある場合
            #if 'true_ite_mean' in result:
            #    print(f"\n  真のITE情報:")
            #    print(f"    真のITE平均: {result['true_ite_mean']:.4f}")
            #    print(f"    正のITE割合: {result['positive_ite_ratio']*100:.1f}%")
            #    print(f"    真のITEに基づく総効果: {result['true_total_effect']:.4f}")
            #    
            #    # 真の改善率
            #    print(f"\n  真の効果での比較:")
        # 真のITEがある場合
        if 'true_ite' in full_df.columns:
            true_ite = full_df['true_ite'].values
            print(f"\nオラクルを使った評価")
            print("\n上位N人に処置した場合の効果:")
            print("  N     効果総和    効率      改善率     備考")
            
            # 元の処置戦略の効果
            original_treated_indices = np.where(full_treatment == 1)[0]
            original_effect_sum = np.sum(true_ite[original_treated_indices])
            
            # 計算するN値のリストを拡張
            all_ns = sorted(list(top_n_results.keys()) + [len(original_treated_indices)])
            
            for n in all_ns:
                # 上位N人のインデックスを取得
                top_n_indices = np.argsort(-ite_pred.flatten())[:n]
                top_n_effect_sum = np.sum(true_ite[top_n_indices])
                
                # 元の処置数との比較
                original_n = len(original_treated_indices)
                efficiency = top_n_effect_sum / n * 100 if n > 0 else 0
                improvement = top_n_effect_sum / original_effect_sum if original_effect_sum > 0 else 0
                
                note = "元の処置人数" if n == original_n else ""
                
                print(f"  {n:<5} {top_n_effect_sum:<8.2f}   {efficiency/100:<4.2f} ({efficiency:.1f}%)   {improvement:<5.2f}倍   {note}")
            
            # さらに大きなN値で評価
            larger_ns = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
            larger_ns = [n for n in larger_ns if n > max(all_ns) and n <= len(ite_pred)]
            
            if larger_ns:
                print("\n上位N人に処置した場合の効果:")
                print("  N     効果総和    効率      改善率     備考")
                
                for n in larger_ns:
                    top_n_indices = np.argsort(-ite_pred.flatten())[:n]
                    top_n_effect_sum = np.sum(true_ite[top_n_indices])
                    
                    efficiency = top_n_effect_sum / n * 100 if n > 0 else 0
                    improvement = top_n_effect_sum / original_effect_sum if original_effect_sum > 0 else 0
                    
                    print(f"  {n:<5} {top_n_effect_sum:<8.2f}   {efficiency/100:<4.2f} ({efficiency:.1f}%)   {improvement:<5.2f}倍   ")

    else:
        print(f"エラー: モデル{causal_model_name}の学習関数がありません")

    print(f"\n===== {causal_model_name}モデルの評価完了 =====")

if __name__ == "__main__":
    main()
