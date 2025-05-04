"""
因果推論モデルの共通ユーティリティ関数
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor, BaseRRegressor, BaseDRLearner

def ensure_dataframe(X):
    """入力をDataFrameに変換し、特徴量名を確保"""
    if isinstance(X, pd.DataFrame):
        return X
    elif hasattr(X, 'columns'):  # numpyではなくpandasオブジェクト
        return pd.DataFrame(X, columns=X.columns)
    else:  # numpy配列
        return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

def get_model_params(model_type='classification'):
    """LightGBMモデルのパラメータを取得"""
    # 共通パラメータ
    lgb_params = {
        'n_estimators': 100,
        'learning_rate': 0.05,
        'num_leaves': 31, 
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbose': -1  # 警告メッセージを抑制
    }
    
    if model_type == 'classification':
        return {**lgb_params, 'objective': 'binary'}  # 分類モデル
    else:
        return {**lgb_params, 'objective': 'regression'}  # 回帰モデル

def _binarize_predictions(y_pred, threshold=0.5):
    """
    連続値の予測を閾値で二値化する
    
    Parameters:
    ----------
    y_pred : array-like
        予測値（確率値を想定）
    threshold : float, default=0.5
        二値化の閾値
        
    Returns:
    -------
    array
        二値化された予測値（0または1）
    """
    return (y_pred > threshold).astype(int)

def get_model_trainer(uplift_method, prediction_method):
    """
    モデル名に対応するトレーナー関数を返す
    
    Parameters:
    ----------
    uplift_method : str
        モデルタイプ ('s_learner', 't_learner', 'x_learner', 'r_learner', 'dr_learner')
    prediction_method : str, default='classification'
        学習器タイプ ('classification', 'regression')
        
    Returns:
    -------
    function
        対応するモデルトレーナー関数
    """
    # 複合名を正しく処理する
    valid_uplift_methods = ['s_learner', 't_learner', 'x_learner', 'r_learner', 'dr_learner']
    valid_prediction_methods = ['classification', 'regression']

    
    ## '_'が含まれていて、かつvalid_model_typesに含まれない場合は、旧形式の命名規則と見なす
    #if '_' in model_type and model_type not in valid_model_types:
    #    parts = model_type.rsplit('_', 1)
    #    if len(parts) == 2:
    #        model_type, learner_type = parts
    
    ## learner_typeを固定値として使用
    #learner_type_fixed = learner_type
    
    print(f"[DEBUG] get_model_trainer: uplift_method={uplift_method}, prediction_method={prediction_method}")
    
    # s_learnerと't_learner'は分類器と回帰器の両方をサポート
    if uplift_method == 's_learner':
        from causal_models import train_s_learner
        def s_learner_wrapper(X, treatment, outcome, propensity_score=None):
            print(f"[DEBUG] s_learner_wrapper: X.shape={X.shape}, prediction_method={prediction_method}")
            # 学習器タイプは外部から固定値として渡される
            model = train_s_learner(
                X, treatment, outcome, prediction_method, propensity_score=propensity_score
            )
            print(f"[DEBUG] s_learner_wrapper: モデルタイプ={type(model)}")
            return model
        return s_learner_wrapper
    
    # T-Learner
    elif uplift_method == 't_learner':
        from causal_models import train_t_learner
        def t_learner_wrapper(X, treatment, outcome, propensity_score=None):
            model = train_t_learner(
                X, treatment, outcome, prediction_method
            )
            return model
        return t_learner_wrapper
    
    # X-Learner（常に回帰）
    elif uplift_method == 'x_learner':
        from causal_models import train_x_learner
        def x_learner_wrapper(X, treatment, outcome, propensity_score=None):
            model = train_x_learner(
                X, treatment, outcome, prediction_method, propensity_score=propensity_score
            )
            return model
        return x_learner_wrapper
    
    # R-Learner（常に回帰）
    elif uplift_method == 'r_learner':
        from causal_models import train_r_learner
        def r_learner_wrapper(X, treatment, outcome, propensity_score=None):
            model = train_r_learner(
                X, treatment, outcome, prediction_method
            )
            return model
        return r_learner_wrapper
    
    # DR-Learner（常に回帰） 
    elif uplift_method == 'dr_learner':
        from causal_models import train_dr_learner
        def dr_learner_wrapper(X, treatment, outcome, propensity_score=None):
            model = train_dr_learner(
                X, treatment, outcome, prediction_method, propensity_score=propensity_score
            )
            return model
        return dr_learner_wrapper
    
    else:
        print(f"警告: 未知のuplift_method '{uplift_method}'")
        return None
