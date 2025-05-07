"""
アップリフトモデル（因果効果予測モデル）の実装
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from typing import Dict, List, Union, Tuple, Optional


class SLearner:
    """
    S-Learner: 単一モデルで処置変数を特徴量として学習するアップリフトモデル
    """
    
    def __init__(self, model=None, random_state=42):
        """
        初期化
        
        Parameters
        ----------
        model : object, optional
            基本モデル（デフォルトはRandomForestRegressor）
        random_state : int, optional
            乱数シード, by default 42
        """
        self.model = model if model is not None else RandomForestRegressor(n_estimators=100, random_state=random_state)
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, treatment: np.ndarray, outcome: np.ndarray) -> 'SLearner':
        """
        モデルを学習
        
        Parameters
        ----------
        X : pd.DataFrame
            特徴量
        treatment : np.ndarray
            処置変数
        outcome : np.ndarray
            アウトカム変数
            
        Returns
        -------
        SLearner
            学習済みモデル
        """
        # 特徴量に処置変数を追加
        X_with_treatment = X.copy()
        X_with_treatment['treatment'] = treatment
        
        # モデルの学習
        self.model.fit(X_with_treatment, outcome)
        self.is_fitted = True
        
        return self
    
    def predict_ite(self, X: pd.DataFrame) -> np.ndarray:
        """
        個別処置効果（ITE）を予測
        
        Parameters
        ----------
        X : pd.DataFrame
            特徴量
            
        Returns
        -------
        np.ndarray
            予測ITE
        """
        if not self.is_fitted:
            raise ValueError("モデルが学習されていません。先にfit()を呼び出してください。")
        
        # 処置あり・なしの両方で予測
        X_with_treatment = X.copy()
        X_with_treatment['treatment'] = 1
        y1_pred = self.model.predict(X_with_treatment)
        
        X_with_treatment['treatment'] = 0
        y0_pred = self.model.predict(X_with_treatment)
        
        # ITE = y1 - y0
        return y1_pred - y0_pred


class TLearner:
    """
    T-Learner: 処置群と対照群で別々のモデルを学習するアップリフトモデル
    """
    
    def __init__(self, model_t=None, model_c=None, random_state=42):
        """
        初期化
        
        Parameters
        ----------
        model_t : object, optional
            処置群のモデル（デフォルトはRandomForestRegressor）
        model_c : object, optional
            対照群のモデル（デフォルトはRandomForestRegressor）
        random_state : int, optional
            乱数シード, by default 42
        """
        self.model_t = model_t if model_t is not None else RandomForestRegressor(n_estimators=100, random_state=random_state)
        self.model_c = model_c if model_c is not None else RandomForestRegressor(n_estimators=100, random_state=random_state)
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, treatment: np.ndarray, outcome: np.ndarray) -> 'TLearner':
        """
        モデルを学習
        
        Parameters
        ----------
        X : pd.DataFrame
            特徴量
        treatment : np.ndarray
            処置変数
        outcome : np.ndarray
            アウトカム変数
            
        Returns
        -------
        TLearner
            学習済みモデル
        """
        # 処置群と対照群に分割
        treatment_mask = treatment == 1
        control_mask = ~treatment_mask
        
        # 処置群のモデル学習
        if np.any(treatment_mask):
            self.model_t.fit(X[treatment_mask], outcome[treatment_mask])
        
        # 対照群のモデル学習
        if np.any(control_mask):
            self.model_c.fit(X[control_mask], outcome[control_mask])
        
        self.is_fitted = True
        
        return self
    
    def predict_ite(self, X: pd.DataFrame) -> np.ndarray:
        """
        個別処置効果（ITE）を予測
        
        Parameters
        ----------
        X : pd.DataFrame
            特徴量
            
        Returns
        -------
        np.ndarray
            予測ITE
        """
        if not self.is_fitted:
            raise ValueError("モデルが学習されていません。先にfit()を呼び出してください。")
        
        # 処置群モデルと対照群モデルで予測
        y1_pred = self.model_t.predict(X)
        y0_pred = self.model_c.predict(X)
        
        # ITE = y1 - y0
        return y1_pred - y0_pred


class XLearner:
    """
    X-Learner: T-Learnerを拡張し、効果を直接推定するアップリフトモデル
    """
    
    def __init__(self, models_t=None, models_c=None, model_final=None, random_state=42):
        """
        初期化
        
        Parameters
        ----------
        models_t : List[object], optional
            処置群のモデルリスト（デフォルトはRandomForestRegressor）
        models_c : List[object], optional
            対照群のモデルリスト（デフォルトはRandomForestRegressor）
        model_final : object, optional
            最終的な効果推定モデル（デフォルトはLinearRegression）
        random_state : int, optional
            乱数シード, by default 42
        """
        # 第1段階のモデル（処置群と対照群の結果を予測）
        self.model_t = models_t if models_t is not None else RandomForestRegressor(n_estimators=100, random_state=random_state)
        self.model_c = models_c if models_c is not None else RandomForestRegressor(n_estimators=100, random_state=random_state)
        
        # 第2段階のモデル（処置効果を直接予測）
        self.model_effect_t = RandomForestRegressor(n_estimators=100, random_state=random_state)
        self.model_effect_c = RandomForestRegressor(n_estimators=100, random_state=random_state)
        
        # 最終的な効果を組み合わせるモデル
        self.model_final = model_final if model_final is not None else LinearRegression()
        
        # 傾向スコア（重み付けに使用）
        self.propensity_scores = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, treatment: np.ndarray, outcome: np.ndarray, 
           propensity_scores: Optional[np.ndarray] = None) -> 'XLearner':
        """
        モデルを学習
        
        Parameters
        ----------
        X : pd.DataFrame
            特徴量
        treatment : np.ndarray
            処置変数
        outcome : np.ndarray
            アウトカム変数
        propensity_scores : Optional[np.ndarray], optional
            傾向スコア, by default None
            
        Returns
        -------
        XLearner
            学習済みモデル
        """
        # 処置群と対照群に分割
        treatment_mask = treatment == 1
        control_mask = ~treatment_mask
        
        X_t = X[treatment_mask]
        X_c = X[control_mask]
        y_t = outcome[treatment_mask]
        y_c = outcome[control_mask]
        
        # 第1段階: 処置群と対照群の結果を予測
        self.model_t.fit(X_t, y_t)
        self.model_c.fit(X_c, y_c)
        
        # 対照群に対する処置群モデルの予測（反実仮想）
        y_t_pred_c = self.model_t.predict(X_c)
        
        # 処置群に対する対照群モデルの予測（反実仮想）
        y_c_pred_t = self.model_c.predict(X_t)
        
        # 第2段階: 処置効果を直接予測
        # 処置群の効果: 実際の結果 - 予測された対照群の結果
        d_t = y_t - y_c_pred_t
        self.model_effect_t.fit(X_t, d_t)
        
        # 対照群の効果: 予測された処置群の結果 - 実際の結果
        d_c = y_t_pred_c - y_c
        self.model_effect_c.fit(X_c, d_c)
        
        # 傾向スコアの保存（重み付けに使用）
        if propensity_scores is not None:
            self.propensity_scores = propensity_scores
        else:
            # 傾向スコアが提供されない場合は均等重み
            self.propensity_scores = np.ones(len(treatment)) * 0.5
        
        self.is_fitted = True
        
        return self
    
    def predict_ite(self, X: pd.DataFrame, propensity_scores: Optional[np.ndarray] = None) -> np.ndarray:
        """
        個別処置効果（ITE）を予測
        
        Parameters
        ----------
        X : pd.DataFrame
            特徴量
        propensity_scores : Optional[np.ndarray], optional
            傾向スコア, by default None
            
        Returns
        -------
        np.ndarray
            予測ITE
        """
        if not self.is_fitted:
            raise ValueError("モデルが学習されていません。先にfit()を呼び出してください。")
        
        # 処置効果の予測
        tau_t = self.model_effect_t.predict(X)
        tau_c = self.model_effect_c.predict(X)
        
        # 傾向スコアによる重み付け
        if propensity_scores is None:
            propensity_scores = np.ones(len(X)) * 0.5
        
        # 最終的な処置効果: 傾向スコアで重み付けした加重平均
        tau = propensity_scores * tau_c + (1 - propensity_scores) * tau_t
        
        return tau


def train_uplift_model(X: pd.DataFrame, treatment: np.ndarray, outcome: np.ndarray, 
                      model_type: str = 's_learner', propensity_scores: Optional[np.ndarray] = None,
                      random_state: int = 42) -> Tuple[object, np.ndarray]:
    """
    アップリフトモデルを学習し、ITE予測値を返す
    
    Parameters
    ----------
    X : pd.DataFrame
        特徴量
    treatment : np.ndarray
        処置変数
    outcome : np.ndarray
        アウトカム変数
    model_type : str, optional
        モデルタイプ ('s_learner', 't_learner', 'x_learner'), by default 's_learner'
    propensity_scores : Optional[np.ndarray], optional
        傾向スコア（X-Learnerで使用）, by default None
    random_state : int, optional
        乱数シード, by default 42
        
    Returns
    -------
    Tuple[object, np.ndarray]
        学習済みモデルとITE予測値のタプル
    """
    # モデルの選択
    if model_type == 's_learner':
        model = SLearner(random_state=random_state)
    elif model_type == 't_learner':
        model = TLearner(random_state=random_state)
    elif model_type == 'x_learner':
        model = XLearner(random_state=random_state)
    else:
        raise ValueError(f"未知のモデルタイプ: {model_type}。's_learner', 't_learner', 'x_learner'のいずれかを指定してください。")
    
    # モデルの学習
    if model_type == 'x_learner':
        model.fit(X, treatment, outcome, propensity_scores)
    else:
        model.fit(X, treatment, outcome)
    
    # ITE予測
    ite_pred = model.predict_ite(X)
    
    return model, ite_pred


def evaluate_uplift_model(ite_pred: np.ndarray, true_ite: Optional[np.ndarray] = None) -> Dict:
    """
    アップリフトモデルの評価指標を計算
    
    Parameters
    ----------
    ite_pred : np.ndarray
        予測ITE
    true_ite : Optional[np.ndarray], optional
        真のITE（存在する場合）, by default None
        
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
        # 相関係数
        metrics['correlation'] = np.corrcoef(ite_pred, true_ite)[0, 1]
        
        # 平均二乗誤差
        metrics['mse'] = np.mean((ite_pred - true_ite)**2)
        
        # 平均絶対誤差
        metrics['mae'] = np.mean(np.abs(ite_pred - true_ite))
        
        # 符号一致率（正負が一致する割合）
        metrics['sign_agreement'] = np.mean(np.sign(ite_pred) == np.sign(true_ite))
    
    return metrics
