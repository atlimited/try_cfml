"""
データ前処理のためのモジュール
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional

def preprocess_data(df: pd.DataFrame, feature_level: str = 'original') -> Tuple[pd.DataFrame, Dict, List[str]]:
    """
    データを前処理する関数
    
    Parameters
    ----------
    df : pd.DataFrame
        データフレーム
    feature_level : str, optional
        特徴量レベル, by default 'original'
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict, List[str]]
        前処理されたデータフレーム、オラクル情報、特徴量カラムのリスト
    """
    # オラクル情報の確認
    oracle_info = {}
    if 'y0' in df.columns and 'y1' in df.columns:
        oracle_info['has_oracle'] = True
        oracle_info['y0'] = df['y0'].values
        oracle_info['y1'] = df['y1'].values
        
        # 真のITEを計算
        if 'true_ite' not in df.columns:
            df['true_ite'] = df['y1'] - df['y0']
        
        oracle_info['true_ite'] = df['true_ite'].values
    
    # 特徴量セットの取得
    feature_cols = get_feature_columns(df, feature_level)
    
    # 特徴量エンジニアリング
    if feature_level == 'extended':
        # 拡張特徴量の作成
        for col in [c for c in feature_cols if c not in ['age_squared', 'age_homeownership']]:
            if 'age' in col:
                df['age_squared'] = df['age'] ** 2
            if 'age' in col and 'homeownership' in df.columns:
                df['age_homeownership'] = df['age'] * df['homeownership']
        
        # 拡張特徴量を特徴量リストに追加
        if 'age_squared' in df.columns and 'age_squared' not in feature_cols:
            feature_cols.append('age_squared')
        if 'age_homeownership' in df.columns and 'age_homeownership' not in feature_cols:
            feature_cols.append('age_homeownership')
    
    return df, oracle_info, feature_cols

def get_feature_columns(df: pd.DataFrame, feature_level: str = 'original') -> List[str]:
    """
    特徴量カラムを取得する関数
    
    Parameters
    ----------
    df : pd.DataFrame
        データフレーム
    feature_level : str, optional
        特徴量レベル, by default 'original'
        
    Returns
    -------
    List[str]
        特徴量カラムのリスト
    """
    # 除外するカラム
    exclude_cols = ['treatment', 'outcome', 'y0', 'y1', 'true_ite', 'propensity_score',
                   'base_prob', 'treat_prob', 'ite_pred']
    
    # 基本特徴量
    basic_cols = [col for col in df.columns if col not in exclude_cols]
    
    if feature_level == 'original':
        return basic_cols
    elif feature_level == 'extended':
        # 拡張特徴量（基本特徴量 + 追加特徴量）
        return basic_cols
    else:
        raise ValueError(f"未知の特徴量レベル: {feature_level}")

def convert_to_bandit_feedback(df: pd.DataFrame, propensity_scores: Optional[np.ndarray] = None) -> Dict:
    """
    データフレームをBanditFeedback形式に変換する関数
    
    Parameters
    ----------
    df : pd.DataFrame
        データフレーム
    propensity_scores : Optional[np.ndarray], optional
        傾向スコア, by default None
        
    Returns
    -------
    Dict
        BanditFeedback形式の辞書
    """
    # 基本的なBanditFeedback形式
    bandit_feedback = {
        "n_rounds": len(df),
        "n_actions": 2,
        "action": df["treatment"].values,
        "reward": df["outcome"].values,
        "position": np.zeros(len(df), dtype=int),
    }
    
    # コンテキスト情報の追加
    feature_cols = [col for col in df.columns if col not in 
                   ['treatment', 'outcome', 'y0', 'y1', 'true_ite', 'propensity_score',
                    'base_prob', 'treat_prob', 'ite_pred']]
    
    if len(feature_cols) > 0:
        bandit_feedback["context"] = df[feature_cols].values
    
    # 傾向スコアの設定
    if propensity_scores is not None:
        bandit_feedback["pscore"] = propensity_scores
    elif 'propensity_score' in df.columns:
        bandit_feedback["pscore"] = df['propensity_score'].values
    
    return bandit_feedback
