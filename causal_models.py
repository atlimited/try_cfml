"""
因果推論モデル（S-Learner、T-Learner、X-Learner、R-Learner、DR-Learner）の実装
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
from causalml.inference.meta import BaseSClassifier, BaseSRegressor, BaseTClassifier, BaseTRegressor, BaseXRegressor, BaseRRegressor, BaseDRLearner
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

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

def ensure_dataframe(X):
    """入力をDataFrameに変換し、特徴量名を確保"""
    if isinstance(X, pd.DataFrame):
        return X
    elif hasattr(X, 'columns'):  # numpyではなくpandasオブジェクト
        return pd.DataFrame(X, columns=X.columns)
    else:  # numpy配列
        return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

def train_s_learner(X, treatment, outcome, model_type='classification', propensity_score=None, test_data=None):
    """S-Learnerモデルの学習と予測"""
    if model_type == 'classification':
        model = BaseSClassifier(learner=lgb.LGBMClassifier(**get_model_params('classification')))
    else:
        model = BaseSRegressor(learner=lgb.LGBMRegressor(**get_model_params('regression')))
    
    # DataFrameに変換
    X_df = ensure_dataframe(X)
    
    # モデル学習
    model.fit(X=X_df, treatment=treatment, y=outcome)

    return model
    
#    # 予測（傾向スコアがあれば使用）- 必ずDataFrameで渡す
#    if propensity_score is not None:
#        if test_data is None:
#            predictions = model.predict(X=X_df, p=propensity_score)
#        else:
#            predictions = model.predict(X=ensure_dataframe(test_data), p=propensity_score)
#    else:
#        if test_data is None:
#            predictions = model.predict(X=X_df)
#        else:
#            predictions = model.predict(X=ensure_dataframe(test_data))
#    
#    return model, predictions

def train_t_learner(X, treatment, outcome, model_type='classification', test_data=None):
    """T-Learnerモデルの学習と予測"""
    if model_type == 'classification':
        model = BaseTClassifier(learner=lgb.LGBMClassifier(**get_model_params('classification')))
    else:
        model = BaseTRegressor(learner=lgb.LGBMRegressor(**get_model_params('regression')))
    
    # DataFrameに変換
    X_df = ensure_dataframe(X)
    
    # モデル学習
    model.fit(X=X_df, treatment=treatment, y=outcome)

    return model
    
#    # 予測 - 必ずDataFrameで渡す
#    if test_data is None:
#        predictions = model.predict(X=X_df)
#    else:
#        predictions = model.predict(X=ensure_dataframe(test_data))
#    
#    return model, predictions

def train_x_learner(X, treatment, outcome, propensity_score=None, test_data=None):
    """X-Learnerモデルの学習と予測"""
    model = BaseXRegressor(learner=lgb.LGBMRegressor(**get_model_params('regression')))
    
    # DataFrameに変換
    X_df = ensure_dataframe(X)
    
    # モデル学習
    model.fit(X=X_df, treatment=treatment, y=outcome)

    return model
    
#    # 予測（傾向スコアがあれば使用）- 必ずDataFrameで渡す
#    if propensity_score is not None:
#        if test_data is None:
#            predictions = model.predict(X=X_df, p=propensity_score)
#        else:
#            predictions = model.predict(X=ensure_dataframe(test_data), p=propensity_score)
#    else:
#        if test_data is None:
#            predictions = model.predict(X=X_df)
#        else:
#            predictions = model.predict(X=ensure_dataframe(test_data))
#    
#    return model, predictions

def train_r_learner(X, treatment, outcome, test_data=None):
    """R-Learnerモデルの学習と予測"""
    model = BaseRRegressor(learner=lgb.LGBMRegressor(**get_model_params('regression')))
    
    # DataFrameに変換
    X_df = ensure_dataframe(X)
    
    # モデル学習
    model.fit(X=X_df, treatment=treatment, y=outcome)

    return model
    
#    # 予測 - 必ずDataFrameで渡す
#    if test_data is None:
#        predictions = model.predict(X=X_df)
#    else:
#        predictions = model.predict(X=ensure_dataframe(test_data))
#    
#    return model, predictions

def train_dr_learner(X, treatment, outcome, propensity_score=None, test_data=None):
    """DR-Learner (Doubly Robust) モデルの学習と予測"""
    model = BaseDRLearner(learner=lgb.LGBMRegressor(**get_model_params('regression')))
    
    # DataFrameに変換
    X_df = ensure_dataframe(X)
    
    # モデル学習 - 傾向スコアがある場合のみ使用
    if propensity_score is not None:
        model.fit(X=X_df, treatment=treatment, y=outcome, p=propensity_score)
    else:
        # 傾向スコアなしの場合は内部で推定
        model.fit(X=X_df, treatment=treatment, y=outcome)

    return model
    
#    # 予測 - 必ずDataFrameで渡す
#    if test_data is None:
#        predictions = model.predict(X=X_df)
#    else:
#        predictions = model.predict(X=ensure_dataframe(test_data))
#    
#    return model, predictions

def train_all_models(X, treatment, outcome, propensity_score):
    """すべてのモデルを学習し、予測を返す"""
    model_results = {}
    
    # DataFrameに変換
    X_df = ensure_dataframe(X)
    
    # 1. S-Learner (分類器)
    model_results['s_learner_cls'] = train_s_learner(
        X_df, treatment, outcome, 'classification', propensity_score)
    
    # 2. T-Learner (分類器)
    model_results['t_learner_cls'] = train_t_learner(
        X_df, treatment, outcome, 'classification')
    
    # 3. S-Learner (回帰器)
    model_results['s_learner_reg'] = train_s_learner(
        X_df, treatment, outcome, 'regression', propensity_score)
    
    # 4. T-Learner (回帰器)
    model_results['t_learner_reg'] = train_t_learner(
        X_df, treatment, outcome, 'regression')
    
    # 5. X-Learner
    model_results['x_learner'] = train_x_learner(
        X_df, treatment, outcome, propensity_score)
    
    # 6. R-Learner
    model_results['r_learner'] = train_r_learner(
        X_df, treatment, outcome)
    
    # 7. DR-Learner
    model_results['dr_learner'] = train_dr_learner(
        X_df, treatment, outcome, propensity_score)
    
    return model_results

# 以下、k-foldクロスバリデーション関連の関数を追加

def evaluate_model_cv(X, treatment, outcome, true_ite=None, model_type='s_learner', 
                     learner_type='classification', propensity_score=None, n_splits=5, random_state=42):
    """k-foldクロスバリデーションを使ってモデルを評価する
    
    Args:
        X: 特徴量データフレーム
        treatment: 処置フラグ
        outcome: アウトカム
        true_ite: 真のITE（存在する場合）
        model_type: 'all', 's_learner', 't_learner', 'x_learner', 'r_learner', 'dr_learner'のいずれか
        learner_type: 'classification', 'regression'のいずれか
        propensity_score: 傾向スコア（オプション）
        n_splits: 交差検証の分割数
        random_state: 乱数シード
        
    Returns:
        評価結果の辞書
    """
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.metrics import mean_squared_error, r2_score
    
    X = ensure_dataframe(X)
    treatment = pd.Series(treatment) if not isinstance(treatment, pd.Series) else treatment
    outcome = pd.Series(outcome) if not isinstance(outcome, pd.Series) else outcome
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # 評価指標の初期化
    metrics = {
        'train_mse': [],
        'test_mse': [],
        'train_r2': [],
        'test_r2': [],
        # 目的変数（outcome）に対する評価指標
        'outcome_treated_mse': [],      # 処置群の目的変数予測MSE
        'outcome_control_mse': [],      # 非処置群の目的変数予測MSE
        'outcome_overall_mse': [],      # 全体の目的変数予測MSE
        'outcome_treated_r2': [],       # 処置群の目的変数予測R2
        'outcome_control_r2': [],       # 非処置群の目的変数予測R2
        'outcome_overall_r2': [],       # 全体の目的変数予測R2
        'outcome_treated_accuracy': [], # 処置群の分類精度（分類モデルの場合）
        'outcome_control_accuracy': [], # 非処置群の分類精度（分類モデルの場合）
        'outcome_overall_accuracy': []  # 全体の分類精度（分類モデルの場合）
    }
    
    if true_ite is not None:
        metrics.update({
            'test_ite_corr': [],     # 真のITEとの相関
            'test_ite_mse': [],      # 真のITEとのMSE
            'test_sign_accuracy': [] # 正負の一致度
        })
    
    fold_idx = 1
    for train_idx, test_idx in kf.split(X):
        print(f"Fold {fold_idx}/{n_splits}...")
        
        # トレーニングデータとテストデータを分割 (numpy配列のインデックスを使用)
        X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
        treatment_train = treatment.iloc[train_idx].reset_index(drop=True)
        treatment_test = treatment.iloc[test_idx].reset_index(drop=True)
        outcome_train = outcome.iloc[train_idx].reset_index(drop=True)
        outcome_test = outcome.iloc[test_idx].reset_index(drop=True)
        
        # numpy配列に変換（インデックス操作のためにインデックスリセットは必要）
        X_train_np = X_train.values
        X_test_np = X_test.values
        treatment_train_np = treatment_train.values
        treatment_test_np = treatment_test.values
        outcome_train_np = outcome_train.values
        outcome_test_np = outcome_test.values
        
        # 傾向スコアが与えられている場合は分割
        ps_train, ps_test = None, None
        if propensity_score is not None:
            ps_train = propensity_score[train_idx]
            ps_test = propensity_score[test_idx]
        
        # モデルの学習と予測
        train_pred = None
        test_pred = None
        y_pred_treated = None
        y_pred_control = None
        outcome_prediction_success = False  # 目的変数予測の成功フラグ
        
        try:
            # モデルの学習と処置効果の予測
            if model_type == 's_learner':
                model, train_pred = train_s_learner(X_train, treatment_train, outcome_train, 
                                               learner_type, ps_train)
                _, test_pred = train_s_learner(X_train, treatment_train, outcome_train, 
                                             learner_type, ps_train, test_data=X_test)
                
                # 目的変数の予測 - 直接訓練されたモデルを使用
                try:
                    # モデル内部に直接アクセスして予測
                    if learner_type == 'classification':
                        # S-Learnerの場合、内部モデルを直接使用する
                        if hasattr(model, 'model'):
                            X_test_df = ensure_dataframe(X_test)
                            
                            # 処置群の予測
                            X_test_treated = X_test_df.copy()
                            X_test_treated['treatment'] = 1
                            y_pred_treated = model.model.predict_proba(X_test_treated)[:, 1]
                            
                            # 非処置群の予測
                            X_test_control = X_test_df.copy()
                            X_test_control['treatment'] = 0
                            y_pred_control = model.model.predict_proba(X_test_control)[:, 1]
                            
                            outcome_prediction_success = True
                        else:
                            print(f"警告: S-Learnerモデルに期待されるモデル属性がありません")
                    else:
                        # 回帰モデルの場合も同様
                        if hasattr(model, 'model'):
                            X_test_df = ensure_dataframe(X_test)
                            
                            # 処置群の予測
                            X_test_treated = X_test_df.copy()
                            X_test_treated['treatment'] = 1
                            y_pred_treated = model.model.predict(X_test_treated)
                            
                            # 非処置群の予測
                            X_test_control = X_test_df.copy()
                            X_test_control['treatment'] = 0
                            y_pred_control = model.model.predict(X_test_control)
                            
                            outcome_prediction_success = True
                        else:
                            print(f"警告: S-Learnerモデルに期待されるモデル属性がありません")
                except Exception as e:
                    print(f"S-Learnerの目的変数予測中にエラー: {str(e)}")
                    # バックアップとして別のモデルを訓練
                    outcome_prediction_success = False
            
            elif model_type == 't_learner':
                model, train_pred = train_t_learner(X_train, treatment_train, outcome_train, 
                                               learner_type)
                _, test_pred = train_t_learner(X_train, treatment_train, outcome_train, 
                                             learner_type, test_data=X_test)
                
                # 目的変数の予測 - T-Learnerの場合、処置群と対照群の別々のモデルがある
                try:
                    if hasattr(model, 'models') and len(model.models) >= 2:
                        # T-Learnerモデルの場合、models[0]が非処置群、models[1]が処置群
                        X_test_df = ensure_dataframe(X_test)
                        
                        if learner_type == 'classification':
                            # 分類モデルの場合は確率を取得
                            y_pred_treated = model.models[1].predict_proba(X_test_df)[:, 1]
                            y_pred_control = model.models[0].predict_proba(X_test_df)[:, 1]
                        else:
                            # 回帰モデルの場合は値を直接取得
                            y_pred_treated = model.models[1].predict(X_test_df)
                            y_pred_control = model.models[0].predict(X_test_df)
                        
                        outcome_prediction_success = True
                    else:
                        print(f"警告: T-Learnerモデルに期待されるモデル構造がありません")
                except Exception as e:
                    print(f"T-Learnerの目的変数予測中にエラー: {str(e)}")
                    outcome_prediction_success = False
            
            elif model_type == 'x_learner':
                model, train_pred = train_x_learner(X_train, treatment_train, outcome_train, ps_train)
                _, test_pred = train_x_learner(X_train, treatment_train, outcome_train, 
                                             ps_train, test_data=X_test)
                
                # X-Learnerの目的変数予測は複雑なので、代替モデルを使用
                outcome_prediction_success = False
            
            elif model_type == 'r_learner':
                model, train_pred = train_r_learner(X_train, treatment_train, outcome_train)
                _, test_pred = train_r_learner(X_train, treatment_train, outcome_train, 
                                             test_data=X_test)
                
                # R-Learnerの目的変数予測は複雑なので、代替モデルを使用
                outcome_prediction_success = False
            
            elif model_type == 'dr_learner':
                if ps_train is None:
                    print("DR Learnerには傾向スコアが必要です")
                    continue
                model, train_pred = train_dr_learner(X_train, treatment_train, outcome_train, ps_train)
                _, test_pred = train_dr_learner(X_train, treatment_train, outcome_train, 
                                              ps_train, test_data=X_test)
                
                # DR-Learnerの目的変数予測は複雑なので、代替モデルを使用
                outcome_prediction_success = False
            
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # 予測値が正しい形状であることを確認
            if train_pred is None or test_pred is None:
                print(f"警告: {model_type}のFold {fold_idx}で予測が取得できませんでした")
                continue
            
            # 次元が合わない場合はフラット化
            train_pred = np.ravel(train_pred)
            test_pred = np.ravel(test_pred)
            
            # 目的変数予測が成功しなかった場合は代替モデルを使用
            if not outcome_prediction_success:
                print(f"モデル{model_type}の目的変数予測のために代替モデルを使用します")
                
                # 処置群と非処置群それぞれに個別のモデルをトレーニング
                if learner_type == 'classification':
                    # 処置群モデル
                    treated_indices = treatment_train_np == 1
                    if np.sum(treated_indices) > 0:  # 処置群が存在する場合
                        treated_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
                        treated_model.fit(X_train_np[treated_indices], outcome_train_np[treated_indices])
                        y_pred_treated = treated_model.predict_proba(X_test_np)[:, 1]
                    else:
                        print("警告: 処置群のデータがありません")
                        y_pred_treated = np.zeros(len(X_test))
                    
                    # 非処置群モデル
                    control_indices = treatment_train_np == 0
                    if np.sum(control_indices) > 0:  # 非処置群が存在する場合
                        control_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
                        control_model.fit(X_train_np[control_indices], outcome_train_np[control_indices])
                        y_pred_control = control_model.predict_proba(X_test_np)[:, 1]
                    else:
                        print("警告: 非処置群のデータがありません")
                        y_pred_control = np.zeros(len(X_test))
                else:
                    # 処置群モデル
                    treated_indices = treatment_train_np == 1
                    if np.sum(treated_indices) > 0:  # 処置群が存在する場合
                        treated_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
                        treated_model.fit(X_train_np[treated_indices], outcome_train_np[treated_indices])
                        y_pred_treated = treated_model.predict(X_test_np)
                    else:
                        print("警告: 処置群のデータがありません")
                        y_pred_treated = np.zeros(len(X_test))
                    
                    # 非処置群モデル
                    control_indices = treatment_train_np == 0
                    if np.sum(control_indices) > 0:  # 非処置群が存在する場合
                        control_model = RandomForestRegressor(n_estimators=100, random_state=random_state)
                        control_model.fit(X_train_np[control_indices], outcome_train_np[control_indices])
                        y_pred_control = control_model.predict(X_test_np)
                    else:
                        print("警告: 非処置群のデータがありません")
                        y_pred_control = np.zeros(len(X_test))
            
            # デバッグ情報
            if fold_idx == 1:
                print(f"モデル: {model_type}_{learner_type}")
                print(f"ITE予測値の最初の5件: {test_pred[:5]}")
                print(f"ITE予測値の分散: {np.var(test_pred)}")
                print(f"ITE予測値のユニーク値: {np.unique(test_pred)[:10] if len(np.unique(test_pred)) > 10 else np.unique(test_pred)}")
                if true_ite is not None:
                    true_ite_sample = true_ite.iloc[test_idx].values[:5]
                    print(f"真のITEサンプル: {true_ite_sample}")
                    print(f"真のITEの分散: {np.var(true_ite.iloc[test_idx].values)}")
                
                if y_pred_treated is not None and y_pred_control is not None:
                    print(f"処置群予測値の最初の5件: {y_pred_treated[:5]}")
                    print(f"非処置群予測値の最初の5件: {y_pred_control[:5]}")
            
            # トレーニングデータに対する評価
            if true_ite is not None:
                true_ite_train = true_ite.iloc[train_idx].values
                train_mse = mean_squared_error(true_ite_train, train_pred)
                train_r2 = r2_score(true_ite_train, train_pred)
            else:
                # アウトカムの予測としての評価（真のITEがない場合）
                y1_train = outcome_train[treatment_train == 1]
                y0_train = outcome_train[treatment_train == 0]
                train_mse = np.mean((y1_train.mean() - y0_train.mean() - train_pred) ** 2)
                # R2は計算できない（単一値との比較になるため）
                train_r2 = np.nan
            
            metrics['train_mse'].append(train_mse)
            metrics['train_r2'].append(train_r2)
            
            # テストデータに対する評価
            if true_ite is not None:
                true_ite_test = true_ite.iloc[test_idx].values
                test_mse = mean_squared_error(true_ite_test, test_pred)
                test_r2 = r2_score(true_ite_test, test_pred)
                
                # 相関係数 - データに分散がある場合のみ計算
                test_pred_var = np.var(test_pred)
                true_ite_var = np.var(true_ite_test)
                
                if test_pred_var > 1e-10 and true_ite_var > 1e-10:
                    test_corr = np.corrcoef(true_ite_test, test_pred)[0, 1]
                else:
                    # 分散がほぼ0の場合は相関を計算できないので、代替値を使用
                    if learner_type == 'classification':
                        # 分類モデルの場合、符号精度を相関の代わりに使用
                        sign_accuracy = np.mean((np.sign(true_ite_test) == np.sign(test_pred)))
                        test_corr = sign_accuracy * 2 - 1  # [-1, 1]の範囲に変換
                    else:
                        test_corr = 0.0  # 分散がない場合は相関なしとみなす
                
                # 符号の一致度（正か負かの判断が合っているか）
                sign_accuracy = np.mean((np.sign(true_ite_test) == np.sign(test_pred)))
                
                metrics['test_ite_corr'].append(test_corr)
                metrics['test_ite_mse'].append(test_mse)
                metrics['test_sign_accuracy'].append(sign_accuracy)
            else:
                # アウトカムの予測としての評価（真のITEがない場合）
                y1_test = outcome_test[treatment_test == 1]
                y0_test = outcome_test[treatment_test == 0]
                test_mse = np.mean((y1_test.mean() - y0_test.mean() - test_pred) ** 2)
                # R2は計算できない（単一値との比較になるため）
                test_r2 = np.nan
            
            metrics['test_mse'].append(test_mse)
            metrics['test_r2'].append(test_r2)
            
            # 目的変数（outcome）に対する予測評価
            if y_pred_treated is not None and y_pred_control is not None:
                # 実際の処置と結果に基づいて評価
                treated_idx = treatment_test_np == 1
                control_idx = treatment_test_np == 0
                
                # 処置群の評価
                if np.sum(treated_idx) > 0:
                    # MSE
                    treated_mse = mean_squared_error(outcome_test_np[treated_idx], y_pred_treated[treated_idx])
                    metrics['outcome_treated_mse'].append(treated_mse)
                    
                    # R2
                    if np.var(outcome_test_np[treated_idx]) > 0:
                        treated_r2 = r2_score(outcome_test_np[treated_idx], y_pred_treated[treated_idx])
                    else:
                        treated_r2 = np.nan
                    metrics['outcome_treated_r2'].append(treated_r2)
                    
                    # 分類精度（分類モデルの場合）
                    if learner_type == 'classification':
                        treated_acc = np.mean((y_pred_treated[treated_idx] > 0.5) == outcome_test_np[treated_idx])
                        metrics['outcome_treated_accuracy'].append(treated_acc)
                
                # 非処置群の評価
                if np.sum(control_idx) > 0:
                    # MSE
                    control_mse = mean_squared_error(outcome_test_np[control_idx], y_pred_control[control_idx])
                    metrics['outcome_control_mse'].append(control_mse)
                    
                    # R2
                    if np.var(outcome_test_np[control_idx]) > 0:
                        control_r2 = r2_score(outcome_test_np[control_idx], y_pred_control[control_idx])
                    else:
                        control_r2 = np.nan
                    metrics['outcome_control_r2'].append(control_r2)
                    
                    # 分類精度（分類モデルの場合）
                    if learner_type == 'classification':
                        control_acc = np.mean((y_pred_control[control_idx] > 0.5) == outcome_test_np[control_idx])
                        metrics['outcome_control_accuracy'].append(control_acc)
                
                # 全体の評価（実際の処置に基づく予測値と実際の結果を比較）
                y_pred_combined = np.zeros_like(outcome_test_np, dtype=float)
                y_pred_combined[treated_idx] = y_pred_treated[treated_idx]
                y_pred_combined[control_idx] = y_pred_control[control_idx]
                
                # MSE
                overall_mse = mean_squared_error(outcome_test_np, y_pred_combined)
                metrics['outcome_overall_mse'].append(overall_mse)
                
                # R2
                if np.var(outcome_test_np) > 0:
                    overall_r2 = r2_score(outcome_test_np, y_pred_combined)
                else:
                    overall_r2 = np.nan
                metrics['outcome_overall_r2'].append(overall_r2)
                
                # 分類精度（分類モデルの場合）
                if learner_type == 'classification':
                    overall_acc = np.mean((y_pred_combined > 0.5) == outcome_test_np)
                    metrics['outcome_overall_accuracy'].append(overall_acc)
            
        except Exception as e:
            print(f"エラー発生: {model_type} - Fold {fold_idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
        
        fold_idx += 1
    
    # 平均値と標準偏差を計算
    result = {}
    for metric_name, values in metrics.items():
        if len(values) > 0:
            result[f'{metric_name}_mean'] = np.mean(values)
            result[f'{metric_name}_std'] = np.std(values)
    
    return result

def evaluate_all_models_cv(X, treatment, outcome, true_ite=None, propensity_score=None, 
                         n_splits=5, random_state=42):
    """すべてのモデルをk-foldクロスバリデーションで評価し、結果を返す"""
    X = ensure_dataframe(X)
    results = {}
    
    # モデルタイプとそのパラメータのリスト
    model_configs = [
        {'name': 's_learner_cls', 'type': 's_learner', 'learner': 'classification'},
        {'name': 't_learner_cls', 'type': 't_learner', 'learner': 'classification'},
        {'name': 's_learner_reg', 'type': 's_learner', 'learner': 'regression'},
        {'name': 't_learner_reg', 'type': 't_learner', 'learner': 'regression'},
        {'name': 'x_learner', 'type': 'x_learner', 'learner': 'regression'},
        {'name': 'r_learner', 'type': 'r_learner', 'learner': 'regression'},
        {'name': 'dr_learner', 'type': 'dr_learner', 'learner': 'regression'}
    ]
    
    for config in model_configs:
        print(f"\n評価モデル: {config['name']}")
        results[config['name']] = evaluate_model_cv(
            X, treatment, outcome, true_ite, 
            model_type=config['type'], 
            learner_type=config['learner'],
            propensity_score=propensity_score,
            n_splits=n_splits,
            random_state=random_state
        )
    
    return results

def print_cv_results(results):
    """クロスバリデーション結果を出力する"""
    print("\n===== クロスバリデーション評価結果 =====\n")
    
    # 各モデルについて評価指標を表示
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        
        # 真のITEとの相関（オラクル評価）
        if 'test_ite_corr_mean' in metrics:
            print(f"  真のITEとの相関: {metrics['test_ite_corr_mean']:.4f} (±{metrics['test_ite_corr_std']:.4f})")
        
        # MSE
        if 'test_ite_mse_mean' in metrics:
            print(f"  真のITEとのMSE: {metrics['test_ite_mse_mean']:.4f} (±{metrics['test_ite_mse_std']:.4f})")
        elif 'test_mse_mean' in metrics:
            print(f"  テストMSE: {metrics['test_mse_mean']:.4f} (±{metrics['test_mse_std']:.4f})")
        
        # 符号精度
        if 'test_sign_accuracy_mean' in metrics:
            print(f"  正負判定精度: {metrics['test_sign_accuracy_mean']:.4f} (±{metrics['test_sign_accuracy_std']:.4f})")
        
        # R^2
        if 'test_r2_mean' in metrics:
            print(f"  テストR²: {metrics['test_r2_mean']:.4f} (±{metrics['test_r2_std']:.4f})")
        
        # 目的変数（outcome）に対する評価指標
        if 'outcome_treated_mse_mean' in metrics:
            print(f"  処置群の目的変数MSE: {metrics['outcome_treated_mse_mean']:.4f} (±{metrics['outcome_treated_mse_std']:.4f})")
        if 'outcome_control_mse_mean' in metrics:
            print(f"  非処置群の目的変数MSE: {metrics['outcome_control_mse_mean']:.4f} (±{metrics['outcome_control_mse_std']:.4f})")
        if 'outcome_overall_mse_mean' in metrics:
            print(f"  全体の目的変数MSE: {metrics['outcome_overall_mse_mean']:.4f} (±{metrics['outcome_overall_mse_std']:.4f})")
        
        if 'outcome_treated_r2_mean' in metrics:
            print(f"  処置群の目的変数R²: {metrics['outcome_treated_r2_mean']:.4f} (±{metrics['outcome_treated_r2_std']:.4f})")
        if 'outcome_control_r2_mean' in metrics:
            print(f"  非処置群の目的変数R²: {metrics['outcome_control_r2_mean']:.4f} (±{metrics['outcome_control_r2_std']:.4f})")
        if 'outcome_overall_r2_mean' in metrics:
            print(f"  全体の目的変数R²: {metrics['outcome_overall_r2_mean']:.4f} (±{metrics['outcome_overall_r2_std']:.4f})")
        
        if 'outcome_treated_accuracy_mean' in metrics:
            print(f"  処置群の分類精度: {metrics['outcome_treated_accuracy_mean']:.4f} (±{metrics['outcome_treated_accuracy_std']:.4f})")
        if 'outcome_control_accuracy_mean' in metrics:
            print(f"  非処置群の分類精度: {metrics['outcome_control_accuracy_mean']:.4f} (±{metrics['outcome_control_accuracy_std']:.4f})")
        if 'outcome_overall_accuracy_mean' in metrics:
            print(f"  全体の分類精度: {metrics['outcome_overall_accuracy_mean']:.4f} (±{metrics['outcome_overall_accuracy_std']:.4f})")
        
        print()
