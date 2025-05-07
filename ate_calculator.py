"""
ATE（平均処置効果）の計算を行う関数
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Optional


def calculate_true_ite_sum(true_ite: np.ndarray, policy: np.ndarray) -> float:
    """
    ポリシーに基づく真のITEの合計を計算します
    
    Parameters
    ----------
    true_ite : np.ndarray
        真のITE配列
    policy : np.ndarray
        0/1のポリシー配列（1が処置を表す）
        
    Returns
    -------
    float
        真のITEの合計
    """
    return np.sum(true_ite * policy)


def calculate_true_ite_mean(true_ite: np.ndarray, policy: np.ndarray) -> float:
    """
    ポリシーに基づく真のITEの平均を計算します
    
    Parameters
    ----------
    true_ite : np.ndarray
        真のITE配列
    policy : np.ndarray
        0/1のポリシー配列（1が処置を表す）
        
    Returns
    -------
    float
        真のITEの平均
    """
    selected_count = np.sum(policy)
    if selected_count > 0:
        return np.sum(true_ite * policy) / selected_count
    return 0


def calculate_policy_statistics(df: pd.DataFrame, policy: np.ndarray, policy_name: str) -> Dict:
    """
    ポリシーの統計情報を計算します
    
    Parameters
    ----------
    df : pd.DataFrame
        データフレーム（y0, y1, true_iteを含む）
    policy : np.ndarray
        0/1のポリシー配列（1が処置を表す）
    policy_name : str
        ポリシー名
        
    Returns
    -------
    Dict
        ポリシーの統計情報を含む辞書
    """
    # ポリシーに基づく選択
    is_selected = policy == 1
    
    # 基本統計
    stats = {
        "name": policy_name,
        "count": np.sum(is_selected),
        "ratio": np.mean(is_selected)
    }
    
    # 真のITE情報（存在する場合）
    if 'true_ite' in df.columns:
        true_ite = df['true_ite'].values
        stats["true_ite_sum"] = calculate_true_ite_sum(true_ite, policy)
        stats["true_ite_mean"] = calculate_true_ite_mean(true_ite, policy)
        stats["true_ite_positive_ratio"] = np.mean(true_ite[is_selected] > 0) if stats["count"] > 0 else 0
    
    # アウトカム情報
    if 'outcome' in df.columns:
        stats["outcome_sum"] = np.sum(df['outcome'].values * policy)
        stats["outcome_mean"] = np.mean(df['outcome'].values[is_selected]) if stats["count"] > 0 else 0
    
    # 予測ITE情報（存在する場合）
    if 'ite_pred' in df.columns:
        stats["pred_ite_sum"] = np.sum(df['ite_pred'].values * policy)
        stats["pred_ite_mean"] = np.mean(df['ite_pred'].values[is_selected]) if stats["count"] > 0 else 0
    
    return stats


def calculate_ate(df: pd.DataFrame, policies: Dict[str, np.ndarray]) -> Dict:
    """
    全体のATEと各ポリシーのATEを計算します
    
    Parameters
    ----------
    df : pd.DataFrame
        データフレーム（y0, y1を含む）
    policies : Dict[str, np.ndarray]
        ポリシー名とポリシー配列の辞書
        
    Returns
    -------
    Dict
        ATE計算結果の辞書
    """
    results = {}
    
    # 全体のATE
    if 'y0' in df.columns and 'y1' in df.columns:
        overall_ate = (df["y1"] - df["y0"]).mean()
        results["overall"] = {
            "ate": overall_ate,
            "treated_ate": overall_ate,  # 全体では処置群と非処置群のATEは同じ
            "control_ate": overall_ate
        }
    
    # 各ポリシーのATE
    for name, policy in policies.items():
        is_selected = policy == 1
        is_not_selected = ~is_selected
        
        # ポリシーの処置群と非処置群のATE
        if 'y0' in df.columns and 'y1' in df.columns:
            # 処置群のATE
            treated_ate = (df["y1"][is_selected] - df["y0"][is_selected]).mean() if np.any(is_selected) else 0
            
            # 非処置群のATE
            control_ate = (df["y1"][is_not_selected] - df["y0"][is_not_selected]).mean() if np.any(is_not_selected) else 0
            
            # ポリシー全体のATE（処置群のATE）
            policy_ate = treated_ate
            
            results[name] = {
                "ate": policy_ate,
                "treated_ate": treated_ate,
                "control_ate": control_ate,
                "ate_diff": treated_ate - control_ate,
                "overall_ratio": policy_ate / overall_ate if overall_ate != 0 else 0
            }
    
    return results


def print_ate_results(ate_results: Dict) -> None:
    """
    ATE計算結果を整形して表示します
    
    Parameters
    ----------
    ate_results : Dict
        ATE計算結果の辞書
    """
    print("\n===== 全体と各ポリシーのATE比較 =====")
    
    # 全体のATE
    if "overall" in ate_results:
        overall = ate_results["overall"]
        print(f"全体のATE: {overall['ate']:.4f}")
    
    # 各ポリシーのATE
    for name, result in ate_results.items():
        if name == "overall":
            continue
        
        print(f"\n【{name}】")
        print(f"  ポリシーのATE: {result['ate']:.4f} (全体比{result['overall_ratio']*100:+.2f}%)")
        print(f"  処置集団のATE: {result['treated_ate']:.4f}")
        print(f"  非処置集団のATE: {result['control_ate']:.4f}")
        print(f"  ATE差分: {result['ate_diff']:+.4f}")


def calculate_top_n_ate(ite_pred: np.ndarray, treatment: np.ndarray, outcome: np.ndarray, 
                       ns: List[int], true_ite: Optional[np.ndarray] = None) -> Dict:
    """
    予測ITE上位N人のATEを計算します
    
    Parameters
    ----------
    ite_pred : np.ndarray
        ITE予測値の配列
    treatment : np.ndarray
        処置変数
    outcome : np.ndarray
        アウトカム変数
    ns : List[int]
        評価するN値のリスト
    true_ite : Optional[np.ndarray], optional
        真のITE配列（存在する場合）, by default None
        
    Returns
    -------
    Dict
        各N値に対する結果の辞書
    """
    results = {}
    
    # 元の処置戦略の効果
    original_treated_indices = np.where(treatment == 1)[0]
    original_treated_count = len(original_treated_indices)
    original_treated_mean = np.mean(outcome[original_treated_indices]) if original_treated_count > 0 else 0
    original_control_mean = np.mean(outcome[treatment == 0]) if np.any(treatment == 0) else 0
    original_effect = original_treated_mean - original_control_mean
    
    # 各N値に対する評価
    for n in ns:
        # 上位N人のインデックスを取得
        top_n_indices = np.argsort(-ite_pred.flatten())[:n]
        
        # 上位N人の処置群と非処置群に分ける
        top_n_treatment = treatment[top_n_indices]
        top_n_treated_indices = top_n_indices[top_n_treatment == 1]
        top_n_control_indices = top_n_indices[top_n_treatment == 0]
        
        # 上位N人の処置群と非処置群の平均を計算
        n_treated = len(top_n_treated_indices)
        n_control = len(top_n_control_indices)
        
        top_n_treated_mean = np.mean(outcome[top_n_treated_indices]) if n_treated > 0 else 0
        top_n_control_mean = np.mean(outcome[top_n_control_indices]) if n_control > 0 else 0
        
        # 上位N人の処置効果
        top_n_effect = top_n_treated_mean - top_n_control_mean
        
        # 結果を格納
        results[n] = {
            "n_treated": n_treated,
            "n_control": n_control,
            "treated_mean": top_n_treated_mean,
            "control_mean": top_n_control_mean,
            "effect": top_n_effect,
            "original_effect": original_effect,
            "improvement_ratio": top_n_effect / original_effect if original_effect != 0 else 0
        }
        
        # 同数処置時の効果比較
        if original_treated_count > 0:
            top_n_effect_scaled = top_n_effect * original_treated_count / n if n > 0 else 0
            results[n]["scaled_effect"] = top_n_effect_scaled
            results[n]["scaled_improvement_ratio"] = top_n_effect_scaled / original_effect if original_effect != 0 else 0
        
        # 真のITEが提供されている場合
        if true_ite is not None:
            top_n_true_effect = np.mean(true_ite[top_n_indices])
            results[n]["true_effect"] = top_n_true_effect
            
            # 真のITEに基づく改善率
            true_original_effect = np.mean(true_ite[original_treated_indices]) if original_treated_count > 0 else 0
            results[n]["true_improvement_ratio"] = top_n_true_effect / true_original_effect if true_original_effect != 0 else 0
    
    return results


def print_top_n_results(top_n_results: Dict, original_treated_count: int) -> None:
    """
    上位N人のATE結果を整形して表示します
    
    Parameters
    ----------
    top_n_results : Dict
        上位N人のATE結果の辞書
    original_treated_count : int
        元の処置人数
    """
    for n, result in top_n_results.items():
        print(f"\n上位{n}人の結果:")
        print(f"  処置群サイズ: {result['n_treated']}")
        print(f"  非処置群サイズ: {result['n_control']}")
        print(f"  上位{n}人の処置群平均: {result['treated_mean']:.4f}")
        print(f"  上位{n}人の非処置群平均: {result['control_mean']:.4f}")
        print(f"  上位{n}人の処置効果: {result['effect']:.4f}")
        
        if "true_effect" in result:
            print(f"  上位{n}人の真の処置効果: {result['true_effect']:.4f}")
        
        print(f"\n  元の処置戦略との比較（処置数: {original_treated_count}人）:")
        print(f"  同数処置時の効果比較:")
        print(f"    上位{n}人処置の効果: {result['scaled_effect']:.4f}")
        print(f"    元の処置戦略の効果: {result['original_effect']:.4f}")
        print(f"    改善率: {result['scaled_improvement_ratio']:.2f}倍")
        
        if "true_improvement_ratio" in result:
            print(f"    真の改善率: {result['true_improvement_ratio']:.2f}倍")
