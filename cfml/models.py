"""
causalmlライブラリを活用したモデルのラッパー
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union, List, Any
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
import logging

logger = logging.getLogger(__name__)

def get_uplift_model(model_type: str = 's_learner', random_state: int = 42, **kwargs):
    """
    causalmlのアップリフトモデルを取得
    
    Parameters
    ----------
    model_type : str
        モデルタイプ ('s_learner', 't_learner', 'x_learner')
    random_state : int
        乱数シード
    
    Returns
    -------
    object
        causalmlのモデルインスタンス
    """
    if model_type == 's_learner':
        from causalml.inference.meta import LRSRegressor
        # LRSRegressorはrandom_stateを直接取らないので削除
        return LRSRegressor(**kwargs) 
    elif model_type == 't_learner':
        from causalml.inference.meta import XGBTRegressor
        return XGBTRegressor(random_state=random_state, **kwargs)
    elif model_type == 'x_learner':
        from causalml.inference.meta import XGBRRegressor
        return XGBRRegressor(random_state=random_state, **kwargs)
    else:
        raise ValueError(f"未知のモデルタイプ: {model_type}")

def train_uplift_model(
    X: pd.DataFrame,
    treatment: np.ndarray,
    outcome: np.ndarray,
    model_type: str = 's_learner',
    propensity_scores: Optional[np.ndarray] = None,
    random_state: int = 42,
    **kwargs
) -> Tuple[object, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    causalmlを使用してアップリフトモデルを学習（責務分離版）
    - causalml uplift modelのfit/predictのみを担当
    - S-Learnerの学習・予測はtrain_s_learnerで独立して管理
    
    Returns
    -------
    Tuple[object, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
        学習済みモデル、ITE予測値、mu0_pred, mu1_pred（取得不可の場合はNone）
    """
    model = get_uplift_model(model_type, random_state=random_state, **kwargs)
    model.fit(X=X, treatment=treatment, y=outcome, p=propensity_scores)
    # causalml uplift modelのpredict
    ite_pred = model.predict(X)
    mu0_pred = None
    mu1_pred = None
    return model, ite_pred, mu0_pred, mu1_pred

def train_custom_s_learner(X: np.ndarray, t: np.ndarray, y: np.ndarray, model_type: str = 'lr', random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger.info(f"Training Custom S-Learner with model_type: {model_type}")
    X_s = np.concatenate([X, t.reshape(-1, 1)], axis=1)
    
    if model_type == 'lr':
        # y が連続値の場合、LogisticRegression は使えないため LinearRegression を使用
        # y の型や値の範囲をチェックして分岐することも可能
        if len(np.unique(y)) > 2 and (y.dtype == float or y.dtype == np.float32 or y.dtype == np.float64) : # heuristic for continuous
             model_s = LinearRegression() 
        else: # そうでなければ LogisticRegression (二値分類を想定)
             model_s = LogisticRegression(random_state=random_state, solver='liblinear')
    elif model_type == 'rf':
        # RandomForestRegressor は連続値を扱える
        model_s = RandomForestRegressor(random_state=random_state)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model_s.fit(X_s, y)
    
    X_s_t1 = np.concatenate([X, np.ones_like(t.reshape(-1, 1))], axis=1)
    X_s_t0 = np.concatenate([X, np.zeros_like(t.reshape(-1, 1))], axis=1)
    
    # LogisticRegression の場合、predict_proba を使うのが一般的
    # それ以外 (LinearRegression, RandomForestRegressor) の場合は predict を使う
    if isinstance(model_s, LogisticRegression):
        mu1_pred = model_s.predict_proba(X_s_t1)[:, 1] 
        mu0_pred = model_s.predict_proba(X_s_t0)[:, 1]
    else:
        mu1_pred = model_s.predict(X_s_t1)
        mu0_pred = model_s.predict(X_s_t0)
    
    y_pred_s = mu1_pred - mu0_pred
    logger.info(f"S-Learner training complete. Predicted ITE shape: {y_pred_s.shape}")
    return y_pred_s.reshape(-1, 1), mu0_pred.reshape(-1, 1), mu1_pred.reshape(-1, 1)

def compute_propensity_scores(X: pd.DataFrame, treatment: np.ndarray, method: str = 'logistic') -> np.ndarray:
    """
    causalmlを使用して傾向スコアを計算
    
    Parameters
    ----------
    X : pd.DataFrame
        特徴量
    treatment : np.ndarray
        処置変数
    method : str
        計算方法 ('logistic', 'gbm')
    
    Returns
    -------
    np.ndarray
        計算された傾向スコア
    """
    if method == 'logistic':
        from causalml.propensity import ElasticNetPropensityModel
        ps_model = ElasticNetPropensityModel()
    elif method in ['gbm', 'lightgbm']:
        from causalml.propensity import GradientBoostedPropensityModel
        ps_model = GradientBoostedPropensityModel()
    else:
        raise ValueError(f"未知の方法: {method}")
    
    # 傾向スコアの計算
    ps_model.fit(X, treatment)
    propensity_scores = ps_model.predict(X)
    
    # 極端な値を防ぐためのクリッピング（0.05-0.95）
    propensity_scores = np.clip(propensity_scores, 0.05, 0.95)
    
    return propensity_scores

def compare_propensity_scores(X: pd.DataFrame, treatment: np.ndarray, 
                             oracle_score: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    異なる手法で傾向スコアを計算して比較
    
    Parameters
    ----------
    X : pd.DataFrame
        特徴量
    treatment : np.ndarray
        処置変数
    oracle_score : Optional[np.ndarray]
        オラクル傾向スコア
        
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
