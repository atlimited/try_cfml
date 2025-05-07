"""
各モデルタイプごとの目的変数（outcome）予測機能を提供するモジュール
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# causal_modelsからモデルパラメータ取得関数をインポート
from causal_models import get_model_params

def ensure_dataframe(X):
    """入力をDataFrameに変換し、特徴量名を確保"""
    if isinstance(X, pd.DataFrame):
        return X
    elif hasattr(X, 'columns'):  # numpyではなくpandasオブジェクト
        return pd.DataFrame(X, columns=X.columns)
    else:  # numpy配列
        return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

def train_outcome_predictor(X, outcome, learner_type='classification', random_state=42):
    """
    目的変数予測モデルを訓練する関数。
    
    Parameters:
    ----------
    X : array-like or DataFrame
        特徴量行列
    outcome : array-like
        目的変数
    learner_type : str, default='classification'
        'classification'または'regression'
    random_state : int, default=42
        乱数シード
        
    Returns:
    -------
    model
        訓練された予測モデル
    """
    # DataFrameに変換
    X_df = ensure_dataframe(X)
    
    if learner_type == 'classification':
        # 分類モデル
        model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31, 
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbose=-1,  # 警告メッセージを抑制
            objective='binary'
        )
    else:
        # 回帰モデル
        model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31, 
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbose=-1,  # 警告メッセージを抑制
            objective='regression'
        )
    
    # モデルの訓練
    model.fit(X_df, outcome)
    
    return model

def predict_outcomes_s_learner(model, X_test, treatment_train=None, outcome_train=None, treatment_test=None, learner_type='classification', random_state=42):
    """
    S-Learnerモデルを使って目的変数を予測する
    
    Parameters:
    ----------
    model : model
        訓練済みのS-Learnerモデル
    X_test : array-like or DataFrame
        テストデータの特徴量
    treatment_train : array-like, optional
        訓練データの処置変数（1=処置、0=非処置）
    outcome_train : array-like, optional
        訓練データの目的変数
    treatment_test : array-like, optional
        テストデータの処置変数（実際の処置情報）
    learner_type : str, default='classification'
        'classification'または'regression'
    random_state : int, default=42
        乱数シード
        
    Returns:
    -------
    y_pred_treated : array-like
        処置群予測（全ての特徴量に処置が適用された場合の予測値）
    y_pred_control : array-like
        非処置群予測（全ての特徴量に処置が適用されなかった場合の予測値）
    y_pred_actual : array-like
        実際の処置を適用した場合の予測値（直接比較用）
    success : bool
        予測が成功したかどうか
    """
    X_test_df = ensure_dataframe(X_test)
    y_pred_actual = None
    
    try:
        # ITEの予測（treatment effectの予測）
        ite_pred = model.predict(X_test_df)
        
        # 処置群と非処置群の予測を初期化
        y_pred_treated = None
        y_pred_control = None
        
        # S-Learnerモデル特有の処理
        try:
            # カスタム実装: 処置効果（ITE）と平均アウトカムから予測値を計算
            # 基準値の計算（全体の平均）
            if outcome_train is not None:
                base_value = np.mean(outcome_train)
            else:
                # アウトカムデータがない場合は0.5を基準とする（分類の場合）
                base_value = 0.5 if learner_type == 'classification' else 0
            
            # 処置効果を利用して処置群と非処置群の予測を計算
            if learner_type == 'classification':
                # 分類の場合は確率値として扱う（0-1に制限）
                y_pred_control = np.clip(base_value - ite_pred/2, 0, 1)
                y_pred_treated = np.clip(base_value + ite_pred/2, 0, 1)
            else:
                # 回帰の場合はそのまま計算
                y_pred_control = base_value - ite_pred/2
                y_pred_treated = base_value + ite_pred/2
        except Exception as e:
            print(f"S-LearnerのITE予測からのアウトカム計算中にエラー: {str(e)}")
            y_pred_treated, y_pred_control = None, None
            return None, None, None, False
            
        # 実際の処置情報を使った直接アウトカム予測
        if treatment_test is not None:
            try:
                # ITEベースの予測（フォールバック方法）
                # すでに計算済みの処置群・非処置群の予測を使用
                # 修正: y_pred_treatedとy_pred_controlの次元を確認
                if hasattr(y_pred_treated, 'shape') and len(y_pred_treated.shape) > 1:
                    y_pred_treated_1d = y_pred_treated.flatten()
                else:
                    y_pred_treated_1d = y_pred_treated
                    
                if hasattr(y_pred_control, 'shape') and len(y_pred_control.shape) > 1:
                    y_pred_control_1d = y_pred_control.flatten()
                else:
                    y_pred_control_1d = y_pred_control
                    
                # 修正: 予測値の長さを調整
                n_samples = len(treatment_test)
                if len(y_pred_treated_1d) > n_samples:
                    y_pred_treated_1d = y_pred_treated_1d[:n_samples]
                if len(y_pred_control_1d) > n_samples:
                    y_pred_control_1d = y_pred_control_1d[:n_samples]
                
                # 修正: np.whereを使って処置群と非処置群の予測値を選択
                y_pred_actual = np.zeros_like(treatment_test, dtype=float)
                treated_mask = (treatment_test == 1)
                control_mask = (treatment_test == 0)
                
                # 各マスクを使って対応する予測値を代入
                y_pred_actual[treated_mask] = y_pred_treated_1d[:sum(treated_mask)]
                y_pred_actual[control_mask] = y_pred_control_1d[:sum(control_mask)]
                
                success_flag = True
            except Exception as e:
                print(f"S-Learnerの直接アウトカム予測中にエラー: {str(e)}")
                # すでに計算済みのITEベースの予測が使用される
                pass
        
        return y_pred_treated, y_pred_control, y_pred_actual, True
    except Exception as e:
        print(f"S-Learnerの目的変数予測中にエラー: {str(e)}")
        return None, None, None, False

def predict_outcomes_t_learner(model, X_test, treatment_train=None, outcome_train=None, learner_type='classification', random_state=42):
    """T-Learnerモデルを使って目的変数を予測する"""
    X_test_df = ensure_dataframe(X_test)
    
    try:
        if hasattr(model, 'models') and len(model.models) >= 2:
            # T-Learnerモデルの場合、models[0]が非処置群、models[1]が処置群
            if learner_type == 'classification':
                # 分類モデルの場合は確率を取得
                y_pred_treated = model.models[1].predict_proba(X_test_df)[:, 1]
                y_pred_control = model.models[0].predict_proba(X_test_df)[:, 1]
            else:
                # 回帰モデルの場合は値を直接取得
                y_pred_treated = model.models[1].predict(X_test_df)
                y_pred_control = model.models[0].predict(X_test_df)
            
            return y_pred_treated, y_pred_control, True
        else:
            print("警告: T-Learnerモデルに期待された構造がありません")
            return None, None, False
    except Exception as e:
        print(f"T-Learnerの目的変数予測中にエラー: {str(e)}")
        return None, None, False

def predict_outcomes_x_learner(model, X_test, treatment_train=None, outcome_train=None, learner_type='classification', random_state=42):
    """X-Learnerモデルを使って目的変数を予測する"""
    try:
        # X-Learnerは複雑な構造を持つため、モデルの内部にアクセスして処置効果予測関数を取得
        if hasattr(model, 'models_mu_0') and hasattr(model, 'models_mu_1'):
            X_test_df = ensure_dataframe(X_test)
            
            # X-Learnerは複雑な構造で、複数のモデルを持っている
            # models_mu_0: 非処置群のアウトカム予測モデル
            # models_mu_1: 処置群のアウトカム予測モデル
            if learner_type == 'classification':
                # 確率を取得
                y_pred_treated = np.array([m.predict_proba(X_test_df)[:, 1] for m in model.models_mu_1]).mean(axis=0)
                y_pred_control = np.array([m.predict_proba(X_test_df)[:, 1] for m in model.models_mu_0]).mean(axis=0)
            else:
                # 値を直接取得
                y_pred_treated = np.array([m.predict(X_test_df) for m in model.models_mu_1]).mean(axis=0)
                y_pred_control = np.array([m.predict(X_test_df) for m in model.models_mu_0]).mean(axis=0)
            
            return y_pred_treated, y_pred_control, True
        else:
            print("警告: X-Learnerモデルに期待された構造がありません")
            return None, None, False
    except Exception as e:
        print(f"X-Learnerの目的変数予測中にエラー: {str(e)}")
        return None, None, False

def predict_outcomes_r_learner(model, X_test, treatment_train=None, outcome_train=None, learner_type='classification', random_state=42):
    """R-Learnerモデルを使って目的変数を予測する"""
    try:
        # R-Learnerは複雑な構造を持つため、内部モデルにアクセスして予測
        if hasattr(model, 'model_mu') and hasattr(model, 'model_tau'):
            X_test_df = ensure_dataframe(X_test)
            
            # model_mu: アウトカム予測モデル
            # model_tau: 処置効果予測モデル
            if learner_type == 'classification':
                # アウトカム期待値の予測 (すべてのユーザーの平均的なアウトカム)
                mu_pred = np.array([m.predict_proba(X_test_df)[:, 1] for m in model.models_mu]).mean(axis=0)
                # 処置効果の予測
                tau_pred = np.array([m.predict(X_test_df) for m in model.models_tau]).mean(axis=0)
                
                # 処置群と非処置群のアウトカム予測値を計算
                y_pred_treated = mu_pred + tau_pred/2
                y_pred_control = mu_pred - tau_pred/2
                
                # 確率値を0-1の範囲に制限
                y_pred_treated = np.clip(y_pred_treated, 0, 1)
                y_pred_control = np.clip(y_pred_control, 0, 1)
            else:
                # アウトカム期待値の予測
                mu_pred = np.array([m.predict(X_test_df) for m in model.models_mu]).mean(axis=0)
                # 処置効果の予測
                tau_pred = np.array([m.predict(X_test_df) for m in model.models_tau]).mean(axis=0)
                
                # 処置群と非処置群のアウトカム予測値を計算
                y_pred_treated = mu_pred + tau_pred/2
                y_pred_control = mu_pred - tau_pred/2
            
            return y_pred_treated, y_pred_control, True
        else:
            print("警告: R-Learnerモデルに期待された構造がありません")
            return None, None, False
    except Exception as e:
        print(f"R-Learnerの目的変数予測中にエラー: {str(e)}")
        return None, None, False

def predict_outcomes_dr_learner(model, X_test, treatment_train=None, outcome_train=None, learner_type='classification', random_state=42):
    """DR-Learnerモデルを使って目的変数を予測する"""
    try:
        # DR-Learnerは複雑な構造を持つため、内部モデルにアクセスして予測
        if hasattr(model, 'models_mu_0') and hasattr(model, 'models_mu_1'):
            X_test_df = ensure_dataframe(X_test)
            
            # models_mu_0: 非処置群のアウトカム予測モデル
            # models_mu_1: 処置群のアウトカム予測モデル
            if learner_type == 'classification':
                # 確率を取得
                y_pred_treated = np.array([m.predict_proba(X_test_df)[:, 1] for m in model.models_mu_1]).mean(axis=0)
                y_pred_control = np.array([m.predict_proba(X_test_df)[:, 1] for m in model.models_mu_0]).mean(axis=0)
            else:
                # 値を直接取得
                y_pred_treated = np.array([m.predict(X_test_df) for m in model.models_mu_1]).mean(axis=0)
                y_pred_control = np.array([m.predict(X_test_df) for m in model.models_mu_0]).mean(axis=0)
            
            return y_pred_treated, y_pred_control, True
        else:
            print("警告: DR-Learnerモデルに期待された構造がありません")
            return None, None, False
    except Exception as e:
        print(f"DR-Learnerの目的変数予測中にエラー: {str(e)}")
        return None, None, False

def get_outcome_predictor(model_type):
    """モデルタイプに応じた目的変数予測関数を返す"""
    predictors = {
        's_learner': predict_outcomes_s_learner,
        't_learner': predict_outcomes_t_learner,
        'x_learner': predict_outcomes_x_learner,
        'r_learner': predict_outcomes_r_learner,
        'dr_learner': predict_outcomes_dr_learner
    }
    return predictors.get(model_type, None)

def fallback_predict_outcomes(X_train, treatment_train, outcome_train, X_test, 
                              learner_type='classification', random_state=42):
    """代替の目的変数予測モデル（これはどのモデルでも内部予測ができない場合のフォールバック）"""
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    treatment_train_np = treatment_train.values if hasattr(treatment_train, 'values') else treatment_train
    outcome_train_np = outcome_train.values if hasattr(outcome_train, 'values') else outcome_train
    
    # 処置群と非処置群に分割
    treated_indices = treatment_train_np == 1
    control_indices = treatment_train_np == 0
    
    if learner_type == 'classification':
        # 処置群モデル
        if np.sum(treated_indices) > 0:
            treated_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
            treated_model.fit(X_train_np[treated_indices], outcome_train_np[treated_indices])
            y_pred_treated = treated_model.predict_proba(X_test_np)[:, 1]
        else:
            print("警告: 処置群のデータがありません")
            y_pred_treated = np.zeros(len(X_test_np))
        
        # 非処置群モデル
        if np.sum(control_indices) > 0:
            control_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
            control_model.fit(X_train_np[control_indices], outcome_train_np[control_indices])
            y_pred_control = control_model.predict_proba(X_test_np)[:, 1]
        else:
            print("警告: 非処置群のデータがありません")
            y_pred_control = np.zeros(len(X_test_np))
    else:
        # 処置群モデル
        if np.sum(treated_indices) > 0:
            treated_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
            treated_model.fit(X_train_np[treated_indices], outcome_train_np[treated_indices])
            y_pred_treated = treated_model.predict(X_test_np)
        else:
            print("警告: 処置群のデータがありません")
            y_pred_treated = np.zeros(len(X_test_np))
        
        # 非処置群モデル
        if np.sum(control_indices) > 0:
            control_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
            control_model.fit(X_train_np[control_indices], outcome_train_np[control_indices])
            y_pred_control = control_model.predict(X_test_np)
        else:
            print("警告: 非処置群のデータがありません")
            y_pred_control = np.zeros(len(X_test_np))
    
    return y_pred_treated, y_pred_control, True
