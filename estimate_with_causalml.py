"""
因果推論モデルを用いた処置効果推定
モジュール化されたコード構造を使用
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import warnings

# 自作モジュールのインポート
from data_preprocessing import prepare_data
from causal_models import train_all_models
from causal_evaluation import (
    compute_propensity_scores,
    estimate_ate_with_ipw,
    evaluate_models,
    evaluate_targeting,
    evaluate_policy_efficiency,
    plot_uplift_curve
)

# LightGBMの警告メッセージを抑制
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

def main():
    # データ読み込み
    df = pd.read_csv("df.csv")

    # データフレームの構造確認
    print("\n===== データフレームの構造 =====")
    print("カラム名:", df.columns.tolist())
    print("最初の5行:")
    print(df.head())

    # データの前処理
    # 基本的特徴量のみ使用する場合は'minimal'、すべての特徴量を使用する場合は'full'
    feature_level = 'minimal'
    df, X, treatment, outcome, oracle_info, feature_cols = prepare_data(df, feature_level)

    # 傾向スコアの計算（LightGBMとロジスティック回帰の両方）
    propensity_scores = compute_propensity_scores(X, treatment, method='all')
    
    # LightGBM傾向スコアを使用
    p_score_lgbm = propensity_scores['lightgbm']
    p_score_logistic = propensity_scores['logistic']
    
    # 傾向スコアの統計情報表示
    print("\n===== 複数の傾向スコア計算方法を比較 =====")
    print(f"LightGBM傾向スコアの統計: 最小値={p_score_lgbm.min():.4f}, 最大値={p_score_lgbm.max():.4f}, 平均={p_score_lgbm.mean():.4f}")
    print(f"Logistic傾向スコアの統計: 最小値={p_score_logistic.min():.4f}, 最大値={p_score_logistic.max():.4f}, 平均={p_score_logistic.mean():.4f}")

    # 傾向スコア単体でのATE評価（逆確率重み付け法）
    ate_lgbm = estimate_ate_with_ipw(outcome, treatment, p_score_lgbm)
    ate_logistic = estimate_ate_with_ipw(outcome, treatment, p_score_logistic)
    observed_ate = df.groupby('treatment')['outcome'].mean().diff().iloc[-1]
    true_ate = df['true_ite'].mean() if 'true_ite' in df.columns else None
    
    print(f"LightGBM傾向スコアによるIPW-ATE推定値: {ate_lgbm:.4f}")
    print(f"Logistic傾向スコアによるIPW-ATE推定値: {ate_logistic:.4f}")
    print(f"観測データからの素朴なATE推定値 (処置群vs対照群の平均の差): {observed_ate:.4f}")
    if true_ate is not None:
        print(f"真のATE (y1-y0の平均): {true_ate:.4f}")

    # 今回使用する傾向スコア
    p_score = p_score_lgbm  # LightGBMの傾向スコアを使用

    # ===== 分類器と回帰器の両方を使ったモデル比較 =====
    print("\n===== 分類器と回帰器の両方を使ったモデル比較 =====")
    
    # すべてのモデルを学習・予測
    model_results = train_all_models(X, treatment, outcome, p_score)
    
    # モデルの予測結果をデータフレームに格納
    for model_name, (model, predictions) in model_results.items():
        df[f'ite_{model_name}'] = predictions
    
    # 単純なoutcome予測モデル - 全データでアウトカムを予測（treatmentは特徴量として使わない）
    outcome_predictor = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31, 
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='binary'
    )
    outcome_predictor.fit(X, outcome)
    df['predicted_outcome'] = outcome_predictor.predict_proba(X)[:, 1]

    # ===== モデル評価 =====
    # モデル名のリスト
    model_names = ['s_learner_cls', 't_learner_cls', 's_learner_reg', 't_learner_reg', 
                   'x_learner', 'r_learner', 'dr_learner']
    
    # 1. ATE評価
    print("\n===== 各モデルのATE推定値 =====")
    print(f"真のATE (y1-y0の平均): {true_ate:.4f}")
    print(f"観測データからの素朴なATE推定値 (処置群vs対照群の平均の差): {observed_ate:.4f}")
    print(f"LightGBM傾向スコアによるIPW-ATE推定値: {ate_lgbm:.4f}")
    print(f"Logistic傾向スコアによるIPW-ATE推定値: {ate_logistic:.4f}")
    
    for model_name in model_names:
        ite_col = f'ite_{model_name}'
        print(f"{model_name}: {df[ite_col].mean():.4f}")
    
    # 2. ITE評価（相関係数、MSE、MAE）
    print("\n===== 各モデルのITE相関係数 (真のITEとの相関) =====")
    for model_name in model_names:
        ite_col = f'ite_{model_name}'
        # 真のITEとの相関を計算
        oracle_corr = df[ite_col].corr(df['true_ite'])
        oracle_mse = ((df['true_ite'] - df[ite_col]) ** 2).mean()
        oracle_mae = abs(df['true_ite'] - df[ite_col]).mean()
        print(f"{model_name}: 相関係数 = {oracle_corr:.4f}, MSE = {oracle_mse:.4f}, MAE = {oracle_mae:.4f}")
    
    # 3. グループ別の予測精度
    print("\n===== グループ別の予測精度 =====")
    if 'response_group' in df.columns:
        for group in sorted(df['response_group'].unique()):
            group_df = df[df['response_group'] == group]
            print(f"グループ{group}:")
            print(f"  サンプル数: {len(group_df)}")
            print(f"  平均真のITE: {group_df['true_ite'].mean():.4f}")
            
            # 各モデルのグループ平均ITE
            for model_name in model_names:
                ite_col = f'ite_{model_name}'
                pred_mean = group_df[ite_col].mean()
                corr = group_df['true_ite'].corr(group_df[ite_col]) if len(group_df) > 1 else float('nan')
                print(f"  {model_name}: {pred_mean:.4f} (相関: {corr:.4f}, 真値との差: {pred_mean-group_df['true_ite'].mean():.4f})")
    
    # 4. グループ別のITE合計
    print("\n===== グループ別のITE合計 =====")
    if 'response_group' in df.columns:
        total_true_ite = df['true_ite'].sum()
        print(f"全体の真のITE合計: {total_true_ite:.2f}")
        for group in sorted(df['response_group'].unique()):
            group_df = df[df['response_group'] == group]
            group_size = len(group_df)
            true_ite_sum = group_df['true_ite'].sum()
            true_ite_contribution = true_ite_sum / total_true_ite * 100 if total_true_ite != 0 else 0
            print(f"グループ{group} ({group_size}人):")
            print(f"  真のITE合計: {true_ite_sum:.2f} (寄与率: {true_ite_contribution:.2f}%)")
            
            # 各モデルのグループ別ITE合計
            for model_name in model_names:
                ite_col = f'ite_{model_name}'
                print(f"  {model_name}: {group_df[ite_col].sum():.2f}")
    
    # 5. 上位N人処置の効果
    print("\n===== 上位N人処置の効果 =====")
    top_ns = [1000, 2000, 4000, 6000, 8000, 10000]
    
    # オラクル（真のITEでソート）
    print("オラクル（真のITEでソート）:")
    for n in top_ns:
        # 真のITEでソート
        oracle_indices = df['true_ite'].nlargest(n).index
        oracle_effect = df.loc[oracle_indices, 'true_ite'].sum()
        print(f"  上位{n}人処置: 効果合計 = {oracle_effect:.2f}, 平均効果 = {oracle_effect/n:.4f}")
    
    # 各モデルの予測による上位N人処置の効果と真の効果の比較
    for model_name in model_names:
        ite_col = f'ite_{model_name}'
        print(f"\n{model_name}の予測による処置:")
        for n in top_ns:
            # 予測ITEでソート
            pred_indices = df[ite_col].nlargest(n).index
            pred_effect = df.loc[pred_indices, ite_col].sum()  # モデルの予測効果
            true_effect = df.loc[pred_indices, 'true_ite'].sum()  # 実際の効果
            oracle_effect = df['true_ite'].nlargest(n).sum()  # オラクル効果
            optimal_ratio = true_effect / oracle_effect if oracle_effect != 0 else 0
            print(f"  上位{n}人処置: 予測効果 = {pred_effect:.2f}, 実際の効果 = {true_effect:.2f}, 最適比 = {optimal_ratio:.2f}")
    
    # 単純なoutcome予測モデルによる上位N人処置の効果
    print("\n単純なoutcome予測モデルによる処置:")
    for n in top_ns:
        # 予測アウトカムでソート
        pred_indices = df['predicted_outcome'].nlargest(n).index
        true_effect = df.loc[pred_indices, 'true_ite'].sum()
        oracle_effect = df['true_ite'].nlargest(n).sum()  # オラクル効果
        optimal_ratio = true_effect / oracle_effect if oracle_effect != 0 else 0
        print(f"  上位{n}人処置: 実際の効果 = {true_effect:.2f}, 最適比 = {optimal_ratio:.2f}")
    
    # 6. 各モデルの正負判定精度
    print("\n===== 各モデルの正負判定精度 =====")
    df['true_uplift_pos'] = (df['true_ite'] > 0).astype(int)  # 真のITEが正の場合を1とする
    
    for model_name in model_names:
        ite_col = f'ite_{model_name}'
        df[f'{model_name}_pos'] = (df[ite_col] > 0).astype(int)
        accuracy = (df[f'{model_name}_pos'] == df['true_uplift_pos']).mean()
        print(f"{model_name}: 全体正負判定精度 = {accuracy:.4f}")
        
        # グループ別の正負判定精度
        if 'response_group' in df.columns:
            for group in sorted(df['response_group'].unique()):
                group_df = df[df['response_group'] == group]
                group_accuracy = (group_df[f'{model_name}_pos'] == group_df['true_uplift_pos']).mean()
                print(f"  グループ{group}の正負判定精度: {group_accuracy:.4f}")
    
    # 7. 実際の処置ポリシーの評価
    print("\n===== 実際の処置ポリシーの評価 =====")
    N_treated = df['treatment'].sum()
    print(f"実際の処置人数: {N_treated}人")
    
    # 実際に処置された人たちの真のITE総和
    actual_policy_effect = df[df['treatment'] == 1]['true_ite'].sum()
    print(f"実際の処置効果の総和: {actual_policy_effect:.2f}")
    
    # 理想的な処置配分（上位N人）の効果
    oracle_indices = df['true_ite'].nlargest(N_treated).index
    oracle_effect = df.loc[oracle_indices, 'true_ite'].sum()
    print(f"理想的な処置配分の効果: {oracle_effect:.2f}")
    
    # 処置効率（実際/理想）
    efficiency = actual_policy_effect / oracle_effect
    print(f"処置効率（実際/理想）: {efficiency:.2f} ({efficiency*100:.1f}%)")
    
    # 8. 各モデルによる処置ポリシーの推定効果
    print("\n===== 各モデルによる処置ポリシーの推定効果 =====")
    for model_name in model_names:
        ite_col = f'ite_{model_name}'
        model_indices = df[ite_col].nlargest(N_treated).index
        model_policy_effect = df.loc[model_indices, 'true_ite'].sum()  # 真のITEに基づく効果
        model_efficiency = model_policy_effect / oracle_effect
        improvement = model_policy_effect / actual_policy_effect
        
        print(f"{model_name}:")
        print(f"  推定処置効果の総和: {model_policy_effect:.2f}")
        print(f"  処置効率（推定/理想）: {model_efficiency:.2f} ({model_efficiency*100:.1f}%)")
        print(f"  実際の処置に対する改善率: {improvement:.2f}倍")
    
    # 単純なoutcome予測モデルによる処置ポリシーの推定効果
    print("\n単純なoutcome予測モデルによる処置ポリシーの推定効果:")
    model_indices = df['predicted_outcome'].nlargest(N_treated).index
    model_policy_effect = df.loc[model_indices, 'true_ite'].sum()
    model_efficiency = model_policy_effect / oracle_effect
    improvement = model_policy_effect / actual_policy_effect
        
    print(f"  推定処置効果の総和: {model_policy_effect:.2f}")
    print(f"  処置効率（推定/理想）: {model_efficiency:.2f} ({model_efficiency*100:.1f}%)")
    print(f"  実際の処置に対する改善率: {improvement:.2f}倍")
    
    # 9. 可視化
    try:
        plot_path = 'uplift_curve.png'
        plot_uplift_curve(
            df=df,
            models=model_names,
            true_ite_col='true_ite',
            prefix='ite_',
            outcome_col='predicted_outcome',
            save_path=plot_path
        )
    except Exception as e:
        print(f"可視化中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
