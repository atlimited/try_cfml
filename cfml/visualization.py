"""
可視化関連のモジュール
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple

def set_plot_style(style: str = 'seaborn-v0_8-whitegrid'):
    """
    Matplotlibのプロットスタイルを設定します。

    Args:
        style (str): 使用するスタイル名。デフォルトは 'seaborn-v0_8-whitegrid'。
    """
    try:
        plt.style.use(style)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [
            'IPAexGothic', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 
            'TakaoPGothic', 'Noto Sans CJK JP', 'sans-serif'
        ]
        plt.rcParams['axes.unicode_minus'] = False 
    except OSError:
        print(f"Warning: Style '{style}' not found. Using default style.")
        plt.style.use('default') 
        try:
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [
                'IPAexGothic', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 
                'TakaoPGothic', 'Noto Sans CJK JP', 'sans-serif'
            ]
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            print(f"Warning: Failed to set Japanese fonts for default style. {e}")

def plot_uplift_distribution(ite_pred: np.ndarray, true_ite: Optional[np.ndarray] = None):
    """
    アップリフトの分布をプロット
    
    Parameters
    ----------
    ite_pred : np.ndarray
        予測ITE
    true_ite : Optional[np.ndarray], optional
        真のITE, by default None
    """
    set_plot_style()
    
    plt.figure(figsize=(12, 6))
    
    # 予測ITEの分布
    plt.subplot(1, 2, 1)
    sns.histplot(ite_pred.flatten(), kde=True, label='Predicted ITE', color='skyblue', stat="density")
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('予測ITEの分布')
    plt.xlabel('予測ITE')
    plt.ylabel('頻度')
    
    # 真のITEとの比較（存在する場合）
    if true_ite is not None:
        sns.histplot(true_ite, kde=True, label='True ITE', color='orange', stat="density")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.scatter(true_ite, ite_pred.flatten(), alpha=0.5)
        plt.xlabel('True ITE')
        plt.ylabel('Predicted ITE')
        # 相関係数を計算する際も flatten する
        correlation_coef = np.corrcoef(true_ite, ite_pred.flatten())[0, 1]
        plt.title(f'真のITEと予測ITEの比較 (相関係数: {correlation_coef:.4f})')
        plt.plot([min(true_ite.min(), ite_pred.min()), max(true_ite.max(), ite_pred.max())], 
                 [min(true_ite.min(), ite_pred.min()), max(true_ite.max(), ite_pred.max())], 
                 'r--')  # 対角線
    
    plt.tight_layout()
    #plt.show()

def plot_uplift_curves(treatment: np.ndarray, outcome: np.ndarray, ite_pred: np.ndarray):
    """
    アップリフトカーブをプロット（causalmlの関数を使用）
    
    Parameters
    ----------
    treatment : np.ndarray
        処置変数
    outcome : np.ndarray
        アウトカム変数
    ite_pred : np.ndarray
        予測ITE
    """
    from causalml.metrics import plot_cumgain, plot_qini
    
    set_plot_style()
    
    # causalmlの関数を使用してアップリフトカーブをプロット
    plot_cumgain(treatment, outcome, ite_pred)
    plot_qini(treatment, outcome, ite_pred)

def plot_propensity_scores(propensity_scores: Dict[str, np.ndarray]):
    """
    傾向スコアの分布をプロット
    
    Parameters
    ----------
    propensity_scores : Dict[str, np.ndarray]
        傾向スコアの辞書
    """
    set_plot_style()
    
    n_methods = len(propensity_scores)
    plt.figure(figsize=(5 * n_methods, 4))
    
    for i, (name, scores) in enumerate(propensity_scores.items()):
        plt.subplot(1, n_methods, i + 1)
        sns.histplot(scores, kde=True)
        plt.title(f'{name}傾向スコアの分布')
        plt.xlabel('傾向スコア')
        plt.ylabel('頻度')
    
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path)
        plt.close()
    else:
        plt.show()

def plot_policy_comparison(policy_stats: Dict[str, Dict]):
    """
    ポリシーの比較をプロット
    
    Parameters
    ----------
    policy_stats : Dict[str, Dict]
        ポリシーの統計情報の辞書
    """
    set_plot_style()
    
    # プロットするメトリクス
    metrics = ['true_ite_mean', 'true_ite_sum', 'outcome_mean', 'outcome_sum']
    available_metrics = []
    
    # 利用可能なメトリクスを確認
    for metric in metrics:
        if all(metric in stats for stats in policy_stats.values()):
            available_metrics.append(metric)
    
    if not available_metrics:
        print("プロットするメトリクスがありません")
        return
    
    # プロット
    n_metrics = len(available_metrics)
    plt.figure(figsize=(5 * n_metrics, 4))
    
    for i, metric in enumerate(available_metrics):
        plt.subplot(1, n_metrics, i + 1)
        
        names = list(policy_stats.keys())
        values = [stats[metric] for stats in policy_stats.values()]
        
        plt.bar(names, values)
        plt.title(f'{metric}の比較')
        plt.ylabel(metric)
        
        # 値を表示
        for j, value in enumerate(values):
            plt.text(j, value, f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path)
        plt.close()
    else:
        plt.show()

def plot_ope_results(ope_results: Dict[str, Dict], true_values: Optional[Dict[str, float]] = None, fig_path: Optional[str] = None):
    """
    OPE結果をプロット
    
    Parameters
    ----------
    ope_results : Dict[str, Dict]
        OPE結果の辞書
    true_values : Optional[Dict[str, float]], optional
        真の値の辞書, by default None
    """
    set_plot_style()
    
    # ポリシー名とOPE手法の取得
    policy_names = list(ope_results.keys())
    ope_methods = list(ope_results[policy_names[0]].keys())
    
    # プロット
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(policy_names))
    width = 0.8 / len(ope_methods)
    
    for i, method in enumerate(ope_methods):
        values = [results[method] for results in ope_results.values()]
        plt.bar(x + i * width - 0.4 + width / 2, values, width, label=method)
    
    # 真の値がある場合はプロット
    if true_values is not None:
        true_vals = [true_values.get(name, np.nan) for name in policy_names]
        plt.plot(x, true_vals, 'k--', label='True Value')
        plt.scatter(x, true_vals, color='k')
    
    plt.xlabel('ポリシー')
    plt.ylabel('推定値')
    plt.title('OPE結果の比較')
    plt.xticks(x, policy_names)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path)
        plt.close()
    else:
        plt.show()
