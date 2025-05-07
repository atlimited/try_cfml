"""
傾向スコア計算のための関数
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from typing import Union, Optional, Tuple, Dict


def compute_propensity_scores(X: pd.DataFrame, treatment: Union[pd.Series, np.ndarray], 
                             method: str = 'logistic') -> np.ndarray:
    """
    指定された方法で傾向スコアを計算します
    
    Parameters
    ----------
    X : pd.DataFrame
        特徴量のデータフレーム
    treatment : Union[pd.Series, np.ndarray]
        処置変数
    method : str, optional
        計算方法 ('logistic', 'lightgbm'), by default 'logistic'
        
    Returns
    -------
    np.ndarray
        計算された傾向スコア
    """
    # 処置変数を配列に変換
    treatment_array = treatment.values if hasattr(treatment, 'values') else np.array(treatment)
    
    if method == 'logistic':
        # 特徴量の標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # ロジスティック回帰で傾向スコアを計算
        model = LogisticRegression(random_state=42)
        model.fit(X_scaled, treatment_array)
        
        # 傾向スコアを計算（処置の確率）
        propensity_scores = model.predict_proba(X_scaled)[:, 1]
        
    elif method == 'lightgbm':
        # LightGBMで傾向スコアを計算
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'random_state': 42
        }
        
        # データセットの作成
        lgb_train = lgb.Dataset(X, treatment_array)
        
        # モデルの学習
        model = lgb.train(lgb_params, lgb_train, num_boost_round=100)
        
        # 傾向スコアを計算
        propensity_scores = model.predict(X)
    else:
        raise ValueError(f"サポートされていない方法です: {method}")
    
    # 極端な値を防ぐためのクリッピング（0.05-0.95）
    propensity_scores = np.clip(propensity_scores, 0.05, 0.95)
    
    return propensity_scores


def compare_propensity_scores(X: pd.DataFrame, treatment: Union[pd.Series, np.ndarray], 
                             oracle_score: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    異なる手法で傾向スコアを計算して比較します
    
    Parameters
    ----------
    X : pd.DataFrame
        特徴量のデータフレーム
    treatment : Union[pd.Series, np.ndarray]
        処置変数
    oracle_score : Optional[np.ndarray], optional
        オラクル傾向スコア（存在する場合）, by default None
        
    Returns
    -------
    Dict[str, np.ndarray]
        各手法で計算された傾向スコアの辞書
    """
    from scipy.stats import pearsonr
    propensity_scores = {}
    
    # オラクルスコア
    if oracle_score is not None:
        propensity_scores['oracle'] = oracle_score
        print(f"オラクル傾向スコアの統計: 最小値={oracle_score.min():.4f}, 最大値={oracle_score.max():.4f}, 平均={oracle_score.mean():.4f}")
    
    # LightGBM傾向スコア
    p_score_lgbm = compute_propensity_scores(X, treatment, method='lightgbm')
    propensity_scores['lightgbm'] = p_score_lgbm
    print(f"LightGBM傾向スコアの統計: 最小値={p_score_lgbm.min():.4f}, 最大値={p_score_lgbm.max():.4f}, 平均={p_score_lgbm.mean():.4f}")
    
    # Logistic傾向スコア
    p_score_logistic = compute_propensity_scores(X, treatment, method='logistic')
    propensity_scores['logistic'] = p_score_logistic
    print(f"Logistic傾向スコアの統計: 最小値={p_score_logistic.min():.4f}, 最大値={p_score_logistic.max():.4f}, 平均={p_score_logistic.mean():.4f}")
    
    # 相関係数の計算と表示
    if oracle_score is not None:
        print("\n===== 傾向スコアの相関係数 =====")
        # LightGBMとオラクルの相関
        lgbm_corr = pearsonr(oracle_score, p_score_lgbm)
        print(f"Oracle PS vs LightGBM PS correlation: {lgbm_corr[0]:.4f} (p-value: {lgbm_corr[1]:.4e})")
        
        # Logisticとオラクルの相関
        logistic_corr = pearsonr(oracle_score, p_score_logistic)
        print(f"Oracle PS vs Logistic PS correlation: {logistic_corr[0]:.4f} (p-value: {logistic_corr[1]:.4e})")
        
        # LightGBMとLogisticの相関
        lgbm_logistic_corr = pearsonr(p_score_lgbm, p_score_logistic)
        print(f"LightGBM PS vs Logistic PS correlation: {lgbm_logistic_corr[0]:.4f} (p-value: {lgbm_logistic_corr[1]:.4e})")
        
        # 平均二乗誤差（MSE）と平均絶対誤差（MAE）の計算
        print("\n===== 傾向スコアの誤差指標 =====")
        lgbm_mse = np.mean((oracle_score - p_score_lgbm)**2)
        logistic_mse = np.mean((oracle_score - p_score_logistic)**2)
        print(f"Oracle PS vs LightGBM PS MSE: {lgbm_mse:.4f}")
        print(f"Oracle PS vs Logistic PS MSE: {logistic_mse:.4f}")
        
        lgbm_mae = np.mean(np.abs(oracle_score - p_score_lgbm))
        logistic_mae = np.mean(np.abs(oracle_score - p_score_logistic))
        print(f"Oracle PS vs LightGBM PS MAE: {lgbm_mae:.4f}")
        print(f"Oracle PS vs Logistic PS MAE: {logistic_mae:.4f}")
    
    return propensity_scores


def estimate_ate_with_ipw(outcome: np.ndarray, treatment: np.ndarray, 
                         propensity_score: np.ndarray) -> float:
    """
    逆確率重み付け法（IPW）でATEを推定します
    
    Parameters
    ----------
    outcome : np.ndarray
        アウトカム変数
    treatment : np.ndarray
        処置変数
    propensity_score : np.ndarray
        傾向スコア
        
    Returns
    -------
    float
        IPWで推定されたATE
    """
    # 処置群と対照群に分ける
    treated = treatment == 1
    control = treatment == 0
    
    # 傾向スコアのクリッピング（極端な値を防ぐため）
    clipped_ps = np.clip(propensity_score, 0.01, 0.99)
    
    # 処置群の逆確率重み
    treated_weights = 1.0 / clipped_ps[treated]
    
    # 対照群の逆確率重み
    control_weights = 1.0 / (1.0 - clipped_ps[control])
    
    # 重み付き平均を計算
    treated_mean = np.sum(outcome[treated] * treated_weights) / np.sum(treated_weights)
    control_mean = np.sum(outcome[control] * control_weights) / np.sum(control_weights)
    
    # ATEを計算
    ate = treated_mean - control_mean
    
    return ate
