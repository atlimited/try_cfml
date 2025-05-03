"""
因果推論モデル（S-Learner、T-Learner、X-Learner、R-Learner、DR-Learner）の実装
"""
import numpy as np
import lightgbm as lgb
from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor, BaseRRegressor, BaseDRLearner

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
        'random_state': 42
    }
    
    if model_type == 'classification':
        return {**lgb_params, 'objective': 'binary'}  # 分類モデル
    else:
        return {**lgb_params, 'objective': 'regression'}  # 回帰モデル

def train_s_learner(X, treatment, outcome, model_type='classification', propensity_score=None):
    """S-Learnerモデルの学習と予測"""
    if model_type == 'classification':
        model = BaseSRegressor(learner=lgb.LGBMClassifier(**get_model_params('classification')))
    else:
        model = BaseSRegressor(learner=lgb.LGBMRegressor(**get_model_params('regression')))
    
    model.fit(X=X, treatment=treatment, y=outcome)
    
    # 予測（傾向スコアがあれば使用）
    if propensity_score is not None:
        predictions = model.predict(X=X, p=propensity_score)
    else:
        predictions = model.predict(X=X)
    
    return model, predictions

def train_t_learner(X, treatment, outcome, model_type='classification'):
    """T-Learnerモデルの学習と予測"""
    if model_type == 'classification':
        model = BaseTRegressor(learner=lgb.LGBMClassifier(**get_model_params('classification')))
    else:
        model = BaseTRegressor(learner=lgb.LGBMRegressor(**get_model_params('regression')))
    
    model.fit(X=X, treatment=treatment, y=outcome)
    predictions = model.predict(X=X)
    
    return model, predictions

def train_x_learner(X, treatment, outcome, propensity_score=None):
    """X-Learnerモデルの学習と予測"""
    model = BaseXRegressor(learner=lgb.LGBMRegressor(**get_model_params('regression')))
    model.fit(X=X, treatment=treatment, y=outcome)
    
    # 予測（傾向スコアがあれば使用）
    if propensity_score is not None:
        predictions = model.predict(X=X, p=propensity_score)
    else:
        predictions = model.predict(X=X)
    
    return model, predictions

def train_r_learner(X, treatment, outcome):
    """R-Learnerモデルの学習と予測"""
    model = BaseRRegressor(learner=lgb.LGBMRegressor(**get_model_params('regression')))
    model.fit(X=X, treatment=treatment, y=outcome)
    predictions = model.predict(X=X)
    
    return model, predictions

def train_dr_learner(X, treatment, outcome, propensity_score):
    """DR-Learner (Doubly Robust) モデルの学習と予測"""
    model = BaseDRLearner(learner=lgb.LGBMRegressor(**get_model_params('regression')))
    model.fit(X=X, treatment=treatment, y=outcome, p=propensity_score)
    predictions = model.predict(X=X)
    
    return model, predictions

def train_all_models(X, treatment, outcome, propensity_score):
    """すべてのモデルを学習し、予測を返す"""
    model_results = {}
    
    # 1. S-Learner (分類器)
    model_results['s_learner_cls'] = train_s_learner(
        X, treatment, outcome, 'classification', propensity_score)
    
    # 2. T-Learner (分類器)
    model_results['t_learner_cls'] = train_t_learner(
        X, treatment, outcome, 'classification')
    
    # 3. S-Learner (回帰器)
    model_results['s_learner_reg'] = train_s_learner(
        X, treatment, outcome, 'regression', propensity_score)
    
    # 4. T-Learner (回帰器)
    model_results['t_learner_reg'] = train_t_learner(
        X, treatment, outcome, 'regression')
    
    # 5. X-Learner
    model_results['x_learner'] = train_x_learner(
        X, treatment, outcome, propensity_score)
    
    # 6. R-Learner
    model_results['r_learner'] = train_r_learner(
        X, treatment, outcome)
    
    # 7. DR-Learner
    model_results['dr_learner'] = train_dr_learner(
        X, treatment, outcome, propensity_score)
    
    return model_results
