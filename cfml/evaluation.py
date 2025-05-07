"""
評価指標とオフポリシー評価のためのモジュール
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_uplift_model(ite_pred: np.ndarray, true_ite: Optional[np.ndarray] = None) -> Dict:
    """
    アップリフトモデルの評価指標を計算
    
    Parameters
    ----------
    ite_pred : np.ndarray
        予測ITE
    true_ite : Optional[np.ndarray], optional
        真のITE, by default None
        
    Returns
    -------
    Dict
        評価指標の辞書
    """
    metrics = {}
    
    # 基本統計
    metrics['mean'] = np.mean(ite_pred)
    metrics['std'] = np.std(ite_pred)
    metrics['min'] = np.min(ite_pred)
    metrics['max'] = np.max(ite_pred)
    metrics['positive_ratio'] = np.mean(ite_pred > 0)
    
    # 真のITEが存在する場合の評価
    if true_ite is not None:
        # MSE, MAE, 相関係数の計算
        metrics['mse'] = mean_squared_error(true_ite, ite_pred)
        metrics['mae'] = mean_absolute_error(true_ite, ite_pred)
        # ite_pred が (n, 1) のような形状の場合、1Dに変換
        if ite_pred.ndim > 1 and ite_pred.shape[1] == 1:
            ite_pred_flat = ite_pred.flatten()
        else:
            ite_pred_flat = ite_pred
        metrics['correlation'] = np.corrcoef(ite_pred_flat, true_ite)[0, 1]
        
        # 符号一致率（正負が一致する割合）
        metrics['sign_agreement'] = np.mean(np.sign(ite_pred) == np.sign(true_ite))
    
    return metrics

def create_policy(df: pd.DataFrame, ite_values: np.ndarray, k: int) -> np.ndarray:
    """
    予測ITEに基づくポリシーを作成
    
    Parameters
    ----------
    df : pd.DataFrame
        データフレーム
    ite_values : np.ndarray
        ITE予測値
    k : int
        処置するユーザー数
        
    Returns
    -------
    np.ndarray
        ポリシー配列（0/1）
    """
    # 上位k件を選択
    topk_indices = np.argsort(-ite_values)[:k]
    is_selected = np.zeros(len(df), dtype=bool)
    is_selected[topk_indices] = True
    
    return is_selected.astype(int)

def create_action_dist(policy: np.ndarray) -> np.ndarray:
    """
    ポリシーからaction_distを作成
    
    Parameters
    ----------
    policy : np.ndarray
        ポリシー配列（0/1）
        
    Returns
    -------
    np.ndarray
        action_dist配列
    """
    action_dist = np.zeros((len(policy), 2, 1))
    action_dist[:, 1, 0] = policy           # 選択されたなら action=1 の確率 1、それ以外は 0
    action_dist[:, 0, 0] = 1 - policy       # 非処置の確率を補完
    return action_dist

def run_ope_evaluation(bandit_feedback: Dict, policies: Dict[str, np.ndarray], 
                     reward_model: Optional[np.ndarray] = None) -> Dict:
    """
    オフポリシー評価を実行
    
    Parameters
    ----------
    bandit_feedback : Dict
        BanditFeedback形式の辞書
    policies : Dict[str, np.ndarray]
        評価するポリシーの辞書
    reward_model : Optional[np.ndarray], optional
        報酬モデル, by default None
        
    Returns
    -------
    Dict
        OPE結果の辞書
    """
    from obp.ope import OffPolicyEvaluation
    from obp.ope.estimators import InverseProbabilityWeighting, DirectMethod, DoublyRobust
    
    # 傾向スコアのクリッピング
    clipped_pscore = np.maximum(bandit_feedback.get("pscore", np.ones(len(bandit_feedback["action"])) * 0.5), 0.01)
    bandit_feedback_clipped = {**bandit_feedback, "pscore": clipped_pscore}
    
    # OPEインスタンスの作成
    ope = OffPolicyEvaluation(
        bandit_feedback=bandit_feedback_clipped,
        ope_estimators=[InverseProbabilityWeighting(), DirectMethod(), DoublyRobust()]
    )
    
    # 各ポリシーのaction_distを作成
    action_dists = {}
    for name, policy in policies.items():
        action_dists[name] = create_action_dist(policy)
    
    # OPE実行
    results = {}
    for name, action_dist in action_dists.items():
        results[name] = ope.estimate_policy_values(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=reward_model
        )
    
    return results

def calculate_policy_statistics(df: pd.DataFrame, policy: np.ndarray, name: str) -> Dict:
    """
    ポリシーの統計情報を計算
    
    Parameters
    ----------
    df : pd.DataFrame
        データフレーム
    policy : np.ndarray
        ポリシー配列（0/1）
    name : str
        ポリシー名
        
    Returns
    -------
    Dict
        統計情報の辞書
    """
    stats = {}
    
    # 処置数と割合
    stats['count'] = int(np.sum(policy))
    stats['ratio'] = float(np.sum(policy) / len(policy))
    
    # 真のITEの統計（存在する場合）
    if 'true_ite' in df.columns:
        true_ite = df['true_ite'].values
        stats['true_ite_mean'] = float(np.mean(true_ite[policy == 1]))
        stats['true_ite_sum'] = float(np.sum(true_ite[policy == 1]))
        stats['true_ite_positive_ratio'] = float(np.mean(true_ite[policy == 1] > 0))
    
    # アウトカムの統計
    if 'outcome' in df.columns:
        outcome = df['outcome'].values
        stats['outcome_mean'] = float(np.mean(outcome[policy == 1]))
        stats['outcome_sum'] = float(np.sum(outcome[policy == 1]))
    
    # 予測ITEの統計（存在する場合）
    if 'ite_pred' in df.columns:
        pred_ite = df['ite_pred'].values
        stats['pred_ite_mean'] = float(np.mean(pred_ite[policy == 1]))
        stats['pred_ite_sum'] = float(np.sum(pred_ite[policy == 1]))
    
    return stats

def calculate_ate(df: pd.DataFrame, policies: Dict[str, np.ndarray]) -> Dict:
    """
    全体のATEと各ポリシーのATEを計算
    
    Parameters
    ----------
    df : pd.DataFrame
        データフレーム
    policies : Dict[str, np.ndarray]
        ポリシーの辞書
        
    Returns
    -------
    Dict
        ATE計算結果の辞書
    """
    # (この関数の実装は表示されていませんでした)
    pass # TODO: 実装を補完
