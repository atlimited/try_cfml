"""
因果推論モデルの評価とユーティリティ関数
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
import lightgbm as lgb
import warnings
from sklearn.linear_model import LogisticRegression
import sklearn

# 警告は根本解決するので削除
# warnings.filterwarnings("ignore", message="X does not have valid feature names")
# warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
# warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")
# warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", message="The objective parameter is deprecated")

# 日本語フォント設定
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
# MacOSの場合はHiragino Sans、Windowsの場合はMS Gothicを使用
matplotlib.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Arial', 'MS Gothic', 'DejaVu Sans']
# 負の値を表示するためのマイナス記号の設定
matplotlib.rcParams['axes.unicode_minus'] = False

def ensure_dataframe(X):
    """入力をDataFrameに変換し、特徴量名を確保"""
    if isinstance(X, pd.DataFrame):
        return X
    elif hasattr(X, 'columns'):  # numpyではなくpandasオブジェクト
        return pd.DataFrame(X, columns=X.columns)
    else:  # numpy配列
        return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

def estimate_ate_with_ipw(y, t, p):
    """逆確率重み付け法でATEを推定する"""
    # NumPy配列に変換
    y_arr = np.array(y)
    t_arr = np.array(t)
    p_arr = np.array(p)
    
    # ブール型インデックスを使用
    treat_mask = (t_arr == 1)
    control_mask = (t_arr == 0)
    
    # 処置群と対照群のアウトカムと傾向スコア
    y1 = y_arr[treat_mask]
    y0 = y_arr[control_mask]
    p1 = p_arr[treat_mask]
    p0 = p_arr[control_mask]
    
    # 処置群の逆確率重み
    w1 = 1 / p1
    # 対照群の逆確率重み
    w0 = 1 / (1 - p0)
    
    # 重み付き平均の計算
    weighted_y1 = np.sum(y1 * w1) / np.sum(w1)
    weighted_y0 = np.sum(y0 * w0) / np.sum(w0)
    
    # ATE = 処置群の重み付き平均 - 対照群の重み付き平均
    ate = weighted_y1 - weighted_y0
    
    return ate

def compute_propensity_scores(X, treatment, method='lightgbm', clip_bounds=(0.05, 0.95), oracle_score=None):
    """複数の手法で傾向スコアを計算し、結果を返す
    
    Args:
        X: 特徴量
        treatment: 処置フラグ
        method: 'lightgbm', 'logistic', 'all' のいずれか
        clip_bounds: 傾向スコアの下限と上限
        oracle_score: オラクルの傾向スコア（存在する場合）
    """
    # 確実にNumPy配列として処理
    treatment_array = np.array(treatment)
    
    # DataFrameに変換
    X_df = ensure_dataframe(X)
    
    # 返す傾向スコア
    result = None
    
    if method == 'lightgbm' or method == 'all':
        # LightGBMによる傾向スコア計算
        ps_lgbm = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            verbose=-1
        )
        ps_lgbm.fit(X_df, treatment_array)
        ps_pred_lgbm = ps_lgbm.predict_proba(X_df)[:, 1]
        
        # 極端な値を避けるためのクリッピング
        ps_pred_lgbm = np.clip(ps_pred_lgbm, clip_bounds[0], clip_bounds[1])
        
        if method == 'lightgbm':
            result = ps_pred_lgbm
    
    if method == 'logistic' or method == 'all':
        # ロジスティック回帰による傾向スコア計算
        ps_logistic = LogisticRegression(random_state=42, max_iter=1000)
        ps_logistic.fit(X_df, treatment_array)
        ps_pred_logistic = ps_logistic.predict_proba(X_df)[:, 1]
        
        # 極端な値を避けるためのクリッピング
        ps_pred_logistic = np.clip(ps_pred_logistic, clip_bounds[0], clip_bounds[1])
        
        if method == 'logistic':
            result = ps_pred_logistic
    
    if method == 'all':
        # 辞書で複数の傾向スコアを返す
        propensity_scores = {}
        
        # オラクルスコアがある場合は追加
        if oracle_score is not None:
            propensity_scores['oracle'] = np.array(oracle_score)
        
        propensity_scores['lightgbm'] = ps_pred_lgbm
        propensity_scores['logistic'] = ps_pred_logistic
        result = propensity_scores
    
    return result

def evaluate_models(df, models, true_ite_col='true_ite', prefix='ite_'):
    """各モデルのITE予測と真のITEを比較し、評価指標を計算"""
    evaluation_results = {}
    
    # ATE評価
    evaluation_results['ate'] = {}
    true_ate = df[true_ite_col].mean()
    evaluation_results['ate']['true'] = true_ate
    
    # 各モデルの平均処置効果
    for model_name in models:
        col_name = f'{prefix}{model_name}'
        if col_name in df.columns:
            evaluation_results['ate'][model_name] = df[col_name].mean()
    
    # 相関係数、MSE、MAE
    evaluation_results['metrics'] = {}
    for model_name in models:
        col_name = f'{prefix}{model_name}'
        if col_name in df.columns:
            model_metrics = {}
            # 相関係数
            model_metrics['correlation'] = df[col_name].corr(df[true_ite_col])
            # MSE
            model_metrics['mse'] = ((df[true_ite_col] - df[col_name]) ** 2).mean()
            # MAE
            model_metrics['mae'] = abs(df[true_ite_col] - df[col_name]).mean()
            evaluation_results['metrics'][model_name] = model_metrics
    
    return evaluation_results

def evaluate_targeting(df, models, true_ite_col='true_ite', prefix='ite_', top_ns=None):
    """各モデルによる上位N人処置の効果を評価"""
    if top_ns is None:
        top_ns = [1000, 2000, 4000, 6000, 8000, 10000]
    
    targeting_results = {}
    
    # オラクル（真のITEでソート）
    targeting_results['oracle'] = {}
    for n in top_ns:
        oracle_indices = df[true_ite_col].nlargest(n).index
        oracle_effect = df.loc[oracle_indices, true_ite_col].sum()
        targeting_results['oracle'][n] = {
            'effect': oracle_effect,
            'avg_effect': oracle_effect / n
        }
    
    # 各モデルの予測による上位N人
    for model_name in models:
        col_name = f'{prefix}{model_name}'
        if col_name in df.columns:
            targeting_results[model_name] = {}
            for n in top_ns:
                pred_indices = df[col_name].nlargest(n).index
                pred_effect = df.loc[pred_indices, col_name].sum()  # モデルの予測効果
                true_effect = df.loc[pred_indices, true_ite_col].sum()  # 実際の効果
                oracle_effect = df[true_ite_col].nlargest(n).sum()  # オラクル効果
                optimal_ratio = true_effect / oracle_effect if oracle_effect != 0 else 0
                
                targeting_results[model_name][n] = {
                    'pred_effect': pred_effect,
                    'true_effect': true_effect,
                    'optimal_ratio': optimal_ratio
                }
    
    return targeting_results

def evaluate_policy_efficiency(df, treatment_col, true_ite_col, models, prefix='ite_'):
    """現在の処置ポリシーと各モデルによる処置ポリシーの効率を評価"""
    policy_results = {}
    
    # 実際の処置人数
    N_treated = df[treatment_col].sum()
    policy_results['n_treated'] = N_treated
    
    # 実際に処置された人たちの真のITE総和
    actual_policy_effect = df[df[treatment_col] == 1][true_ite_col].sum()
    policy_results['actual_effect'] = actual_policy_effect
    
    # 理想的な処置配分（上位N人）の効果
    oracle_indices = df[true_ite_col].nlargest(N_treated).index
    oracle_effect = df.loc[oracle_indices, true_ite_col].sum()
    policy_results['oracle_effect'] = oracle_effect
    
    # 処置効率（実際/理想）
    efficiency = actual_policy_effect / oracle_effect if oracle_effect != 0 else 0
    policy_results['actual_efficiency'] = efficiency
    
    # 各モデルによる処置ポリシーの推定効果
    policy_results['models'] = {}
    for model_name in models:
        col_name = f'{prefix}{model_name}'
        if col_name in df.columns:
            model_indices = df[col_name].nlargest(N_treated).index
            model_policy_effect = df.loc[model_indices, true_ite_col].sum()  # 真のITEに基づく効果
            model_efficiency = model_policy_effect / oracle_effect if oracle_effect != 0 else 0
            improvement = model_policy_effect / actual_policy_effect if actual_policy_effect != 0 else float('inf')
            
            policy_results['models'][model_name] = {
                'effect': model_policy_effect,
                'efficiency': model_efficiency,
                'improvement': improvement
            }
    
    return policy_results

def plot_uplift_curve(df, models, true_ite_col='true_ite', prefix='ite_', 
                     outcome_col=None, percentiles=None, save_path=None):
    """Uplift Curveを描画（各モデルの上位N%処置の効果を可視化）"""
    plt.figure(figsize=(12, 8))
    
    if percentiles is None:
        percentiles = np.arange(0, 101, 5)
    
    # オラクル（真のITEでソート）
    df_oracle = df.sort_values(by=true_ite_col, ascending=False).reset_index(drop=True)
    oracle_effect = []
    for p in percentiles:
        if p == 0:
            oracle_effect.append(0)
        else:
            n_samples = int(len(df) * p / 100)
            effect = df_oracle.iloc[:n_samples][true_ite_col].sum()
            oracle_effect.append(effect)
    
    plt.plot(percentiles, oracle_effect, label='Oracle (True ITE)', linestyle='--', color='black', linewidth=2)
    
    # 各モデルのカーブ
    for model_name in models:
        col_name = f'{prefix}{model_name}'
        if col_name in df.columns:
            df_sorted = df.sort_values(by=col_name, ascending=False).reset_index(drop=True)
            
            cumulative_effect = []
            for p in percentiles:
                if p == 0:
                    cumulative_effect.append(0)
                else:
                    n_samples = int(len(df) * p / 100)
                    effect = df_sorted.iloc[:n_samples][true_ite_col].sum()  # 真のITEに基づく効果
                    cumulative_effect.append(effect)
            
            plt.plot(percentiles, cumulative_effect, label=model_name)
    
    # アウトカム予測でソートした場合（指定があれば）
    if outcome_col is not None and outcome_col in df.columns:
        df_sorted = df.sort_values(by=outcome_col, ascending=False).reset_index(drop=True)
        outcome_effect = []
        for p in percentiles:
            if p == 0:
                outcome_effect.append(0)
            else:
                n_samples = int(len(df) * p / 100)
                effect = df_sorted.iloc[:n_samples][true_ite_col].sum()  # 真のITEに基づく効果
                outcome_effect.append(effect)
        
        plt.plot(percentiles, outcome_effect, label='Outcome Model', linestyle='-.', color='gray')
    
    plt.xlabel('処置割合 (%)')
    plt.ylabel('累積処置効果')
    plt.title('Uplift Curve（真のITEに基づく効果）')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n===== 可視化結果は {save_path} に保存されました =====")
    else:
        plt.show()
    
    return plt.gcf()
