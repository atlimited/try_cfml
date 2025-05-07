"""
OPE（オフポリシー評価）関連の処理を行う関数
"""
import numpy as np
import pandas as pd
from obp.ope.estimators import InverseProbabilityWeighting, DirectMethod, DoublyRobust
from obp.ope import OffPolicyEvaluation
from typing import Dict, List, Union, Tuple, Optional


def create_action_dist(policy: np.ndarray) -> np.ndarray:
    """
    ポリシーからaction_dist形式を作成します
    
    Parameters
    ----------
    policy : np.ndarray
        0/1のポリシー配列（1が処置を表す）
        
    Returns
    -------
    np.ndarray
        OBP形式のaction_dist配列
    """
    action_dist = np.zeros((len(policy), 2, 1))
    action_dist[:, 1, 0] = policy           # 選択されたなら action=1 の確率 1、それ以外は 0
    action_dist[:, 0, 0] = 1 - policy       # 非処置の確率を補完
    return action_dist


def run_ope(bandit_feedback: Dict, reward_model: np.ndarray, action_dist: np.ndarray,
           clip_pscore: bool = True, min_pscore: float = 0.01) -> Dict:
    """
    オフポリシー評価を実行します
    
    Parameters
    ----------
    bandit_feedback : Dict
        OBP形式のバンディットフィードバック
    reward_model : np.ndarray
        報酬モデル（真または推定）
    action_dist : np.ndarray
        評価するポリシーのaction_dist
    clip_pscore : bool, optional
        傾向スコアをクリップするかどうか, by default True
    min_pscore : float, optional
        傾向スコアの最小値, by default 0.01
        
    Returns
    -------
    Dict
        OPE結果の辞書
    """
    # 傾向スコアのクリッピング（最小値を設定）
    if clip_pscore and "pscore" in bandit_feedback:
        clipped_pscore = np.maximum(bandit_feedback["pscore"], min_pscore)
        bandit_feedback_used = {**bandit_feedback, "pscore": clipped_pscore}
    else:
        bandit_feedback_used = bandit_feedback
    
    # OPEの実行
    ope = OffPolicyEvaluation(
        bandit_feedback=bandit_feedback_used,
        ope_estimators=[InverseProbabilityWeighting(),
                        DirectMethod(),
                        DoublyRobust()]
    )
    
    # ポリシー価値の推定
    return ope.estimate_policy_values(
        action_dist=action_dist,
        estimated_rewards_by_reg_model=reward_model
    )


def generate_estimated_rewards(df: pd.DataFrame, model, feature_cols: List[str]) -> np.ndarray:
    """
    推定期待報酬を生成します
    
    Parameters
    ----------
    df : pd.DataFrame
        データフレーム
    model : object
        学習済みモデル（predict メソッドを持つ）
    feature_cols : List[str]
        特徴量のカラム名リスト
        
    Returns
    -------
    np.ndarray
        推定期待報酬の配列
    """
    estimated_rewards = np.zeros((len(df), 2, 1))
    
    for a in [0, 1]:
        # 特徴量に処置を追加
        Xa = df[feature_cols].copy()
        Xa["treatment"] = a
        
        # モデルで予測
        estimated_rewards[:, a, 0] = model.predict(Xa)
    
    return estimated_rewards


def generate_true_rewards(df: pd.DataFrame) -> np.ndarray:
    """
    真の期待報酬を生成します（オラクル情報が必要）
    
    Parameters
    ----------
    df : pd.DataFrame
        データフレーム（base_probとtreat_probを含む）
        
    Returns
    -------
    np.ndarray
        真の期待報酬の配列
    """
    if 'base_prob' not in df.columns or 'treat_prob' not in df.columns:
        raise ValueError("真の期待報酬を計算するには base_prob と treat_prob が必要です")
    
    true_rewards = np.zeros((len(df), 2, 1))
    true_rewards[:, 0, 0] = df["base_prob"].values  # 非処置時の期待報酬
    true_rewards[:, 1, 0] = df["treat_prob"].values  # 処置時の期待報酬
    
    return true_rewards


def print_ope_results(policy_name: str, results: Dict, true_value: Optional[float] = None) -> None:
    """
    OPE結果を整形して表示します
    
    Parameters
    ----------
    policy_name : str
        ポリシー名
    results : Dict
        OPE結果の辞書
    true_value : Optional[float], optional
        真の値（存在する場合）, by default None
    """
    print(f"\n===== {policy_name}のオフポリシー評価結果 =====")
    
    # 各推定手法の結果を表示
    print("IPW推定値:", results["ipw"])
    print("DM推定値:", results["dm"])
    print("DR推定値:", results["dr"])
    
    # 真の値が提供されている場合は比較も表示
    if true_value is not None:
        print(f"\n真の値との比較（真の値: {true_value:.4f}）:")
        print(f"IPW: バイアス={results['ipw']-true_value:+.4f}, 相対誤差={(results['ipw']-true_value)/true_value:+.2%}")
        print(f"DM: バイアス={results['dm']-true_value:+.4f}, 相対誤差={(results['dm']-true_value)/true_value:+.2%}")
        print(f"DR: バイアス={results['dr']-true_value:+.4f}, 相対誤差={(results['dr']-true_value)/true_value:+.2%}")


def create_policy_from_ite(ite: np.ndarray, k: int) -> np.ndarray:
    """
    ITE予測値から上位k件を選択するポリシーを作成します
    
    Parameters
    ----------
    ite : np.ndarray
        ITE予測値の配列
    k : int
        選択する件数
        
    Returns
    -------
    np.ndarray
        0/1のポリシー配列（1が処置を表す）
    """
    # ITE上位k件のインデックスを取得
    topk_indices = np.argsort(-ite.flatten())[:k]
    
    # ポリシー配列を初期化（すべて0）
    policy = np.zeros(len(ite), dtype=int)
    
    # 上位k件を1に設定
    policy[topk_indices] = 1
    
    return policy
