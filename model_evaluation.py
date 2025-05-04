"""
因果推論モデルの評価のための関数を提供するモジュール
特にk分割交差検証によるモデル評価に重点を置いています
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_squared_error, r2_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score, classification_report, accuracy_score
)
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from scipy.stats import pearsonr
from scipy import stats
from tqdm import tqdm

# ローカルモジュールのインポート
import causal_models
from outcome_predictors import get_outcome_predictor, fallback_predict_outcomes
from model_utils import get_model_trainer, ensure_dataframe, _binarize_predictions

def calculate_top_n_ate(ite_pred, treatment, outcome, ns=[500, 1000, 2000], true_ite=None, alpha=0.05):
    """
    ITE予測値の上位N人に対する実際の処置効果（ATE）を計算する
    
    Parameters:
    -----------
    ite_pred : array-like
        個別処置効果の予測値
    treatment : array-like
        処置フラグ（1=処置、0=非処置）
    outcome : array-like
        結果変数
    ns : list of int, default=[500, 1000, 2000]
        上位何人を選ぶかのリスト
    true_ite : array-like, optional
        真のITE値（オラクル情報）
    alpha : float, default=0.05
        信頼区間の有意水準
        
    Returns:
    --------
    dict
        各N値に対する計算結果を含む辞書
    """
    results = {}
    
    # NumPy配列に変換
    ite_pred_array = np.array(ite_pred).flatten()  # 2次元配列の場合は1次元に変換
    treatment_array = np.array(treatment)
    outcome_array = np.array(outcome)
    
    # 全体の処置群と非処置群の平均アウトカムを計算（ベースライン比較用）
    all_treated_indices = np.where(treatment_array == 1)[0]
    all_control_indices = np.where(treatment_array == 0)[0]
    
    all_treated_outcome = outcome_array[treatment_array == 1]
    all_control_outcome = outcome_array[treatment_array == 0]
    
    all_treated_mean = np.mean(all_treated_outcome) if len(all_treated_outcome) > 0 else np.nan
    all_control_mean = np.mean(all_control_outcome) if len(all_control_outcome) > 0 else np.nan
    all_ate = all_treated_mean - all_control_mean if not (np.isnan(all_treated_mean) or np.isnan(all_control_mean)) else np.nan
    
    # 元の処置戦略の効果を計算
    original_treatment_count = np.sum(treatment_array == 1)
    original_treatment_effect = all_ate * original_treatment_count if not np.isnan(all_ate) else np.nan
    
    # 元の処置戦略で処置された人々の真のITE合計（オラクル情報がある場合）
    if true_ite is not None:
        true_ite_array = np.array(true_ite)
        original_true_effect = np.sum(true_ite_array[all_treated_indices])
    else:
        original_true_effect = np.nan
    
    for n in ns:
        # 上位N人のインデックスを取得
        top_n_indices = np.argsort(ite_pred_array)[-n:]
        
        # 上位N人のデータを抽出
        top_n_treatment = treatment_array[top_n_indices]
        top_n_outcome = outcome_array[top_n_indices]
        
        # 処置群と非処置群に分ける
        treated_mask = top_n_treatment == 1
        control_mask = top_n_treatment == 0
        
        treated_outcome = top_n_outcome[treated_mask]
        control_outcome = top_n_outcome[control_mask]
        
        # 処置群と非処置群のサイズを確認
        n_treated = len(treated_outcome)
        n_control = len(control_outcome)
        
        # 結果を格納する辞書を初期化
        result = {
            'n': n,
            'n_treated': n_treated,
            'n_control': n_control,
            'original_treatment_count': original_treatment_count,
            'original_treatment_effect': original_treatment_effect,
            'original_true_effect': original_true_effect
        }
        
        # 上位N人の処置群と非処置群の平均を計算
        if n_treated > 0 and n_control > 0:
            # 上位N人の中での処置効果計算
            top_n_treated_mean = np.mean(treated_outcome)
            top_n_control_mean = np.mean(control_outcome)
            top_n_effect = top_n_treated_mean - top_n_control_mean
            
            # 仮想的な総効果 = 処置効果 × 非処置群の人数
            virtual_total_effect = top_n_effect * n_control
            
            # 元の処置戦略との比較（同数の処置を行った場合）
            if n <= original_treatment_count:
                # 上位n人に処置した場合の効果
                top_n_virtual_effect = top_n_effect * n
                
                # 改善率
                if not np.isnan(original_treatment_effect) and original_treatment_effect != 0:
                    improvement_ratio = top_n_virtual_effect / (original_treatment_effect * n / original_treatment_count)
                else:
                    improvement_ratio = np.nan
            else:
                # 元の処置人数より多い場合は、元の処置人数分だけで比較
                top_n_virtual_effect = top_n_effect * original_treatment_count
                
                # 改善率
                if not np.isnan(original_treatment_effect) and original_treatment_effect != 0:
                    improvement_ratio = top_n_virtual_effect / original_treatment_effect
                else:
                    improvement_ratio = np.nan
            
            result.update({
                'top_n_treated_mean': top_n_treated_mean,
                'top_n_control_mean': top_n_control_mean,
                'top_n_effect': top_n_effect,
                'virtual_total_effect': virtual_total_effect,
                'top_n_virtual_effect': top_n_virtual_effect,
                'improvement_ratio': improvement_ratio,
                'method': 'observed'
            })
        else:
            # 処置群または非処置群のサンプルがない場合
            result.update({
                'top_n_treated_mean': np.nan if n_treated == 0 else np.mean(treated_outcome),
                'top_n_control_mean': np.nan if n_control == 0 else np.mean(control_outcome),
                'top_n_effect': np.nan,
                'virtual_total_effect': np.nan,
                'top_n_virtual_effect': np.nan,
                'improvement_ratio': np.nan,
                'method': 'insufficient_data'
            })
        
        # 真のITEがある場合は、それとの比較も行う
        if true_ite is not None:
            # 上位N人の真のITE平均
            top_n_true_ite = true_ite_array[top_n_indices]
            true_ite_mean = np.mean(top_n_true_ite)
            
            # 真のITE > 0の割合
            positive_ite_ratio = np.mean(top_n_true_ite > 0)
            
            # 真のITEに基づく総効果
            true_total_effect = np.sum(top_n_true_ite)
            
            # 元の処置戦略との真の効果比較（同数の処置を行った場合）
            if n <= original_treatment_count:
                # 上位n人に処置した場合の真の効果
                true_n_effect = np.sum(top_n_true_ite)
                
                # 真の改善率
                if original_true_effect != 0:
                    true_improvement_ratio = true_n_effect / (original_true_effect * n / original_treatment_count)
                else:
                    true_improvement_ratio = np.nan
            else:
                # 元の処置人数より多い場合は、元の処置人数分だけで比較
                top_original_count_indices = np.argsort(true_ite_array)[-original_treatment_count:]
                true_n_effect = np.sum(true_ite_array[top_original_count_indices])
                
                # 真の改善率
                if original_true_effect != 0:
                    true_improvement_ratio = true_n_effect / original_true_effect
                else:
                    true_improvement_ratio = np.nan
            
            result.update({
                'true_ite_mean': true_ite_mean,
                'positive_ite_ratio': positive_ite_ratio,
                'true_total_effect': true_total_effect,
                'true_n_effect': true_n_effect,
                'true_improvement_ratio': true_improvement_ratio
            })
        
        results[n] = result
    
    return results

def evaluate_cv_with_top_n(model_trainer, X, outcome, treatment, fold_count=5, random_state=42, propensity_score=None):
    """CVを使って予測ITEの上位N人評価を行う関数"""
    from sklearn.model_selection import KFold
    
    # 実際にtreatment=1かつoutcome=1だった人数を計算
    treatment_array = np.array(treatment)
    outcome_array = np.array(outcome)
    converted_mask = (treatment_array == 1) & (outcome_array == 1)
    treatment_count = np.sum(treatment_array==1)
    converted_count = np.sum(converted_mask)
    
    print(f"実際にtreatment=1だった人数(全数): {treatment_count}")
    print(f"実際にtreatment=1かつoutcome=1だった人数(全数): {converted_count}")
    
    # 結果を保存する配列
    cv_predictions = []
    cv_true_values = []
    cv_treatments = []
    cv_scores = []
    
    kf = KFold(n_splits=fold_count, random_state=random_state, shuffle=True)
    
    for train_idx, test_idx in kf.split(X):
        # トレーニングデータとテストデータを分割
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = outcome.iloc[train_idx], outcome.iloc[test_idx]
        w_train, w_test = treatment.iloc[train_idx], treatment.iloc[test_idx]
        
        # model_trainerを使ってモデルをトレーニング
        fold_model = model_trainer(X_train, w_train, y_train, propensity_score=propensity_score)
        
        # テストデータでITE予測
        ite_preds = fold_model.predict(X_test)
        #print(f"ITE予測値の形状: {ite_preds.shape}")
        #print("ITE予測値:", ite_preds)
        
        # フォールドごとに評価
        # 予測ITEの上位(treatment_count / fold_count) (treatment全数のFold相当の人数)を選択
        sorted_indices = np.argsort(-ite_preds.flatten())
        fold_top_n_treatment_count = int(treatment_count/fold_count)
        top_n_indices = sorted_indices[:fold_top_n_treatment_count]
        
        # そのフォールドでのtreatment=1かつoutcome=1の人数
        fold_converted_mask = (w_test.values == 1) & (y_test.values == 1)
        fold_converted_count = np.sum(fold_converted_mask)
        print("fold_converted_count:", fold_converted_count)
        
        # 評価指標を計算
        #precision = np.sum(y_test.values[top_n_indices] * w_test.values[top_n_indices]) / treatment_count if treatment_count > 0 else 0
        #recall = np.sum(y_test.values[top_n_indices] * w_test.values[top_n_indices]) / fold_converted_count if fold_converted_count > 0 else 0

        precision = np.sum(y_test.values[top_n_indices]) / fold_top_n_treatment_count if fold_top_n_treatment_count > 0 else 0
        recall = np.sum(y_test.values[top_n_indices]) / fold_converted_count if fold_converted_count > 0 else 0
        
        # 結果を保存
        cv_scores.append({
            'precision': precision,
            'recall': recall,
            'fold_converted_count': fold_converted_count,
            'top_n_converted': np.sum(y_test.values[top_n_indices] * w_test.values[top_n_indices])
        })
        
        # フォールドごとの予測と実際の値を保存
        cv_predictions.extend(ite_preds.flatten())
        cv_true_values.extend(y_test.values)
        cv_treatments.extend(w_test.values)
    
    # 結果を出力
    print("\nCV評価 - 予測ITE上位N人での効果:")
    print(f"  N = {int(treatment_count/fold_count)} (treatment=1かつoutcome=1の人数)")
    
    avg_precision = np.mean([s['precision'] for s in cv_scores])
    avg_recall = np.mean([s['recall'] for s in cv_scores])
    print(f"  平均精度（precision）: {avg_precision:.4f}")
    print(f"  平均再現率（recall）: {avg_recall:.4f}")
    
    for i, score in enumerate(cv_scores):
        print(f"  Fold {i+1}: 上位{int(treatment_count/fold_count)}人中 {score['top_n_converted']}人が実際にコンバージョン (精度: {score['precision']:.4f}, 再現率: {score['recall']:.4f})")
    
    return {
        'cv_predictions': cv_predictions,
        'cv_true_values': cv_true_values,
        'cv_treatments': cv_treatments,
        'cv_scores': cv_scores,
        'avg_precision': avg_precision
    }

def evaluate_model_cv(X, treatment, outcome, true_ite=None, model_type='s_learner', 
                     learner_type='classification', propensity_score=None, n_splits=5, 
                     random_state=42, classification_threshold=0.5):
    """
    k分割交差検証でモデルを評価する関数
    
    Parameters:
    ----------
    X : array-like or DataFrame
        特徴量行列
    treatment : array-like
        処置変数（1=処置、0=非処置）
    outcome : array-like
        目的変数
    true_ite : array-like, optional
        真の個別処置効果（検証用）
    model_type : str, default='s_learner'
        モデルタイプ ('s_learner', 't_learner', 'x_learner', 'r_learner', 'dr_learner')
    learner_type : str, default='classification'
        学習器タイプ ('classification', 'regression')
    propensity_score : array-like, optional
        傾向スコア（与えられれば使用）
    n_splits : int, default=5
        交差検証の分割数
    random_state : int, default=42
        乱数シード
    classification_threshold : float, default=0.5
        分類評価に使用する二値化の閾値
        
    Returns:
    -------
    dict
        評価指標の辞書
    """
    from sklearn.model_selection import KFold
    import numpy as np
    from tqdm import tqdm
    import warnings
    
    # モデルのトレーニング関数を取得
    train_function = get_model_trainer(model_type, learner_type)
    
    if train_function is None:
        print(f"エラー: モデル{model_type}の学習関数がありません")
        return {}
    
    # 型変換（numpyに統一）
    X_np = X.values if hasattr(X, 'values') else X
    treatment_np = treatment.values if hasattr(treatment, 'values') else treatment
    outcome_np = outcome.values if hasattr(outcome, 'values') else outcome
    ps_np = propensity_score.values if propensity_score is not None and hasattr(propensity_score, 'values') else propensity_score
    true_ite_np = true_ite.values if true_ite is not None and hasattr(true_ite, 'values') else true_ite
    
    # K-分割交差検証の準備
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # 結果を格納するリスト
    all_fold_results = {}
    fold_results = []
    
    # 各foldの処理
    for fold_idx, (train_idx, test_idx) in enumerate(tqdm(kf.split(X_np), desc=f"CV for {model_type}", total=n_splits)):
        try:
            # トレーニングデータとテストデータに分割
            X_train, X_test = X_np[train_idx], X_np[test_idx]
            treatment_train, treatment_test = treatment_np[train_idx], treatment_np[test_idx]
            outcome_train, outcome_test = outcome_np[train_idx], outcome_np[test_idx]#

            # 真のITEがある場合は分割
            if true_ite is not None:
                true_ite_train, true_ite_test = true_ite_np[train_idx], true_ite_np[test_idx]
            
            # 傾向スコアがある場合は分割
            if propensity_score is not None:
                ps_train = ps_np[train_idx] 
                ps_test = ps_np[test_idx]
            else:
                ps_train = None
                ps_test = None
            
            # DataFrameに変換（特徴量名を保持するため）
            if hasattr(X, 'columns'):
                X_train_df = pd.DataFrame(X_train, columns=X.columns)
                X_test_df = pd.DataFrame(X_test, columns=X.columns)
            else:
                X_train_df = X_train
                X_test_df = X_test
            
            # モデル固有の学習関数でモデルを学習・予測
            try:
                if model_type == 's_learner':
                    model = train_function(
                        X_train_df, treatment_train, outcome_train, 
                        propensity_score=ps_train
                    )
                elif model_type == 't_learner':
                    model = train_function(
                        X_train_df, treatment_train, outcome_train
                    )
                elif model_type == 'x_learner':
                    model = train_function(
                        X_train_df, treatment_train, outcome_train,
                        propensity_score=ps_train
                    )
                elif model_type == 'r_learner':
                    model = train_function(
                        X_train_df, treatment_train, outcome_train
                    )
                elif model_type == 'dr_learner':
                    model = train_function(
                        X_train_df, treatment_train, outcome_train,
                        propensity_score=ps_train
                    )
                else:
                    # 未知のモデルタイプ
                    print(f"Fold {fold_idx} でエラーが発生しました: 未知のモデルタイプ {model_type}")
                    continue
            except Exception as e:
                print(f"Fold {fold_idx} でエラーが発生しました: {str(e)}")
                continue
            
            # テストデータでの予測
            try:
                # モデルタイプに応じて予測方法を変える
                if model_type in ['s_learner', 'x_learner', 'dr_learner'] and ps_test is not None:
                    # 傾向スコアを使用するモデル
                    test_predictions = model.predict(
                        X=X_test_df, 
                        p=ps_test
                    )
                else:
                    # 傾向スコアを使用しないモデル
                    test_predictions = model.predict(X=X_test_df)
                
                if isinstance(test_predictions, tuple):
                    # 予測が複数の値を返す場合（一部のモデル）
                    test_predictions = test_predictions[0]  # 最初の要素を使用
                
                # 予測値の形状を確認し、必要に応じて変換
                if hasattr(test_predictions, 'shape') and len(test_predictions.shape) > 1:
                    if test_predictions.shape[1] == 1:
                        # (n_samples, 1) -> (n_samples,)
                        test_predictions = test_predictions.flatten()
                
                # スカラー値の場合、配列に変換
                if np.isscalar(test_predictions):
                    test_predictions = np.array([test_predictions] * len(test_idx))
                    
                # 形状の確認
                print(f"  予測形状: {test_predictions.shape if hasattr(test_predictions, 'shape') else 'スカラー'}")
            except Exception as e:
                print(f"Fold {fold_idx} 予測でエラーが発生しました: {str(e)}")
                continue
            
            # モデル固有の目的変数予測関数を取得
            outcome_predictor = get_outcome_predictor(model_type)
            
            # モデル固有の方法でアウトカム予測
            if outcome_predictor:
                # 実際の処置情報を渡して直接比較用の予測も取得
                y_pred_treated, y_pred_control, y_pred_actual, success = outcome_predictor(
                    model, X_test_df, 
                    treatment_train=treatment_train, 
                    outcome_train=outcome_train,
                    treatment_test=treatment_test,  # 実際の処置情報を渡す
                    learner_type=learner_type,
                    random_state=random_state
                )

                print("###################")
                print(y_pred_treated)
                print(y_pred_treated.shape)
                print(y_pred_control)
                print(y_pred_control.shape)
                print(y_pred_actual)
                print(y_pred_actual.shape)
                
                # DataFrameにして詳細な情報を表示
                print("\n===== 予測値の詳細情報 =====")
                # サンプルデータ
                sample_size = min(10, len(X_test))
                debug_df = pd.DataFrame({
                    'treatment': treatment_test[:sample_size],
                    'outcome': outcome_test[:sample_size],
                })
                
                # 処置効果の予測値（ITE）を追加
                if hasattr(model, 'predict'):
                    try:
                        ite_pred = model.predict(X_test_df)
                        debug_df['ite_pred'] = ite_pred[:sample_size]
                        print("ITE予測値統計: 最小値={:.4f}, 最大値={:.4f}, 平均={:.4f}, 標準偏差={:.4f}"
                              .format(np.min(ite_pred), np.max(ite_pred), np.mean(ite_pred), np.std(ite_pred)))
                    except Exception as e:
                        print(f"ITEの予測でエラー: {str(e)}")
                
                # 予測値を処理
                if hasattr(y_pred_treated, 'shape'):
                    if len(y_pred_treated.shape) > 1 and y_pred_treated.shape[1] > 0:
                        debug_df['y_pred_treated'] = y_pred_treated[:sample_size, 0]
                    else:
                        debug_df['y_pred_treated'] = y_pred_treated[:sample_size]
                    
                    print("処置群予測値統計: 最小値={:.4f}, 最大値={:.4f}, 平均={:.4f}, 標準偏差={:.4f}"
                          .format(np.min(y_pred_treated), np.max(y_pred_treated), 
                                  np.mean(y_pred_treated), np.std(y_pred_treated)))
                
                if hasattr(y_pred_control, 'shape'):
                    if len(y_pred_control.shape) > 1 and y_pred_control.shape[1] > 0:
                        debug_df['y_pred_control'] = y_pred_control[:sample_size, 0]
                    else:
                        debug_df['y_pred_control'] = y_pred_control[:sample_size]
                    
                    print("対照群予測値統計: 最小値={:.4f}, 最大値={:.4f}, 平均={:.4f}, 標準偏差={:.4f}"
                          .format(np.min(y_pred_control), np.max(y_pred_control), 
                                  np.mean(y_pred_control), np.std(y_pred_control)))
                
                # y_pred_actualの問題を検出（形状が異常な場合）
                if y_pred_actual is not None:
                    if hasattr(y_pred_actual, 'shape') and len(y_pred_actual.shape) >= 2 and y_pred_actual.shape[1] > 1:
                        print(f"警告: y_pred_actualの形状が異常です: {y_pred_actual.shape}")
                        # 正しい形状に修正を試みる
                        try:
                            if y_pred_actual.shape[0] == len(treatment_test):
                                # 最初の列だけを使用
                                y_pred_actual = y_pred_actual[:, 0]
                                print(f"  y_pred_actualを修正しました: 新しい形状 {y_pred_actual.shape}")
                        except Exception as e:
                            print(f"  y_pred_actualの修正に失敗: {str(e)}")
                
                print(debug_df)
                print("###################")
                
                # モデル固有の予測が失敗した場合は予測結果をNoneに設定
                if not success:
                    print(f"モデル{model_type}の目的変数予測ができませんでした")
                    y_pred_treated, y_pred_control, y_pred_actual = None, None, None
            else:
                # 該当する予測機能がない場合も予測結果をNoneに設定
                print(f"モデル{model_type}には目的変数予測機能がありません")
                y_pred_treated, y_pred_control, y_pred_actual = None, None, None
            
            # このfoldの結果を評価
            fold_result = {}
            
            try:
                # ITE予測の評価
                if true_ite is not None:
                    true_ite_np = true_ite if not hasattr(true_ite, 'values') else true_ite.values
                    true_ite_test = true_ite_np[test_idx]
                    
                    # 形状の出力
                    print(f"  真のITE形状: {true_ite_test.shape}, 予測形状: {test_predictions.shape}")
                    
                    # 形状の一致
                    if hasattr(true_ite_test, 'shape') and hasattr(test_predictions, 'shape'):
                        if len(true_ite_test.shape) != len(test_predictions.shape):
                            # 次元が異なる場合は調整
                            if len(true_ite_test.shape) > len(test_predictions.shape):
                                # 真の値の次元が大きい場合
                                true_ite_test = true_ite_test.flatten()
                            else:
                                # 予測値の次元が大きい場合
                                test_predictions = test_predictions.flatten()
                        
                        # 長さの不一致
                        if len(true_ite_test) != len(test_predictions):
                            min_len = min(len(true_ite_test), len(test_predictions))
                            true_ite_test = true_ite_test[:min_len]
                            test_predictions = test_predictions[:min_len]
                            print(f"  警告: サイズ不一致を検出。最小サイズ{min_len}に調整しました。")
                    
                    # スカラー値の場合は配列に変換
                    if np.isscalar(true_ite_test):
                        true_ite_test = np.array([true_ite_test])
                    if np.isscalar(test_predictions):
                        test_predictions = np.array([test_predictions])
                    
                    # 相関係数（予測された処置効果 vs 真の処置効果）
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        print(f"  相関係数計算: {test_predictions.shape} vs {true_ite_test.shape}")
                        correlation, _ = pearsonr(test_predictions.flatten(), true_ite_test.flatten())
                        
                        # 予測精度指標
                        mse = mean_squared_error(true_ite_test, test_predictions)
                        r2 = r2_score(true_ite_test, test_predictions)
                        
                        # 符号一致率 (予測と真のITEの符号が一致する割合)
                        sign_accuracy = np.mean((test_predictions > 0) == (true_ite_test > 0))
                        
                        # 処置効果がバイナリ変数の場合のAUC-ROC
                        if learner_type == 'classification':
                            try:
                                if len(np.unique(true_ite_test)) == 2:  # 二値分類の場合
                                    auc_roc = roc_auc_score((true_ite_test > 0).astype(int), test_predictions)
                                    fold_result['auc_roc'] = auc_roc
                            except Exception as e:
                                print(f"  AUC計算でエラー: {str(e)}")
                        
                        # 処置効果の評価指標をfold結果に追加
                        fold_result['correlation'] = correlation if not np.isnan(correlation) else 0
                        fold_result['mse'] = mse
                        fold_result['r2'] = r2
                        fold_result['sign_accuracy'] = sign_accuracy
                        
                        # 成功
                        print(f"  Fold {fold_idx} ITE評価成功: 相関={correlation:.4f}, MSE={mse:.4f}, R2={r2:.4f}")
            except Exception as e:
                print(f"  Fold {fold_idx} 処置効果評価エラー: {str(e)}")
            
            # 目的変数の予測評価
            if y_pred_treated is not None and y_pred_control is not None:
                try:
                    # テストセットのデータを取得
                    test_treatment = treatment[test_idx]
                    test_outcome = outcome[test_idx]
                    
                    # 多次元の予測値を1次元に変換（必要な場合）
                    def ensure_1d(arr):
                        """配列を1次元に変換する"""
                        if arr is None:
                            return None
                        if hasattr(arr, 'shape') and len(arr.shape) > 1:
                            if arr.shape[1] == 1:
                                return arr.flatten()
                            elif arr.shape[1] > 1:
                                # 二値分類の場合、陽性クラスの確率を返す
                                return arr[:, 1]
                        return arr
                    
                    # 予測値を1次元に変換
                    y_pred_treated_1d = ensure_1d(y_pred_treated)
                    y_pred_control_1d = ensure_1d(y_pred_control)
                    
                    # デバッグ出力
                    print(f"  予測値次元: 処置群={y_pred_treated_1d.shape if hasattr(y_pred_treated_1d, 'shape') else 'None'}, "
                          f"対照群={y_pred_control_1d.shape if hasattr(y_pred_control_1d, 'shape') else 'None'}")
                    
                    # クロスバリデーションの結果を格納するデータフレームを作成
                    # 実際のデータと予測値を結合して評価しやすくする
                    test_df = pd.DataFrame({
                        'treatment': test_treatment,
                        'outcome': test_outcome
                    })
                    
                    # 予測値をデータフレームに追加
                    # 全体用の予測値サイズを確認
                    if len(y_pred_treated_1d) == len(test_idx) and len(y_pred_control_1d) == len(test_idx):
                        # 全サンプル分の予測がある場合
                        test_df['y_pred'] = np.where(test_df['treatment'] == 1, 
                                                    y_pred_treated_1d, 
                                                    y_pred_control_1d)
                    else:
                        # 処置群/対照群別々に予測がある場合
                        # 念のためサイズ調整
                        test_df['y_pred'] = np.nan
                        if np.sum(test_treatment) > 0:  # 処置群が存在
                            treated_mask = (test_df['treatment'] == 1)
                            test_df.loc[treated_mask, 'y_pred'] = y_pred_treated_1d[:sum(treated_mask)]
                        
                        if np.sum(1 - test_treatment) > 0:  # 対照群が存在
                            control_mask = (test_df['treatment'] == 0)
                            test_df.loc[control_mask, 'y_pred'] = y_pred_control_1d[:sum(control_mask)]
                    
                    # 処置群と対照群に分割
                    treated_df = test_df[test_df['treatment'] == 1]
                    control_df = test_df[test_df['treatment'] == 0]
                    
                    # デバッグ情報
                    print(f"  テストデータ: 全体={len(test_df)}, 処置群={len(treated_df)}, 対照群={len(control_df)}")
                    
                    # 各群の評価
                    if len(treated_df) > 0 and 'y_pred' in treated_df.columns:
                        # 処置群の評価
                        try:
                            if learner_type == 'classification':
                                # 二値分類に変換
                                treated_df['y_pred_binary'] = (treated_df['y_pred'] > classification_threshold).astype(int)
                                
                                # 分類指標の計算
                                accuracy = accuracy_score(treated_df['outcome'], treated_df['y_pred_binary'])
                                
                                try:
                                    roc_auc = roc_auc_score(treated_df['outcome'], treated_df['y_pred'])
                                except Exception as e:
                                    print(f"  処置群AUC計算エラー: {str(e)}")
                                    roc_auc = 0.5
                                
                                precision = precision_score(treated_df['outcome'], treated_df['y_pred_binary'], zero_division=0)
                                recall = recall_score(treated_df['outcome'], treated_df['y_pred_binary'], zero_division=0)
                                f1 = f1_score(treated_df['outcome'], treated_df['y_pred_binary'], zero_division=0)
                                
                                # 評価結果を保存
                                fold_result['treated_accuracy'] = accuracy
                                fold_result['treated_roc_auc'] = roc_auc
                                fold_result['treated_precision'] = precision
                                fold_result['treated_recall'] = recall
                                fold_result['treated_f1'] = f1
                                
                                # 混同行列
                                cm = confusion_matrix(treated_df['outcome'], treated_df['y_pred_binary'])
                                fold_result['treated_outcome_conf_mat'] = cm
                                
                                # 全体の混同行列も更新
                                if 'treated_outcome_conf_mat_all' not in all_fold_results:
                                    all_fold_results['treated_outcome_conf_mat_all'] = cm
                                else:
                                    all_fold_results['treated_outcome_conf_mat_all'] += cm
                            
                            # 回帰指標（分類/回帰共通）
                            mse = mean_squared_error(treated_df['outcome'], treated_df['y_pred'])
                            fold_result['treated_outcome_mse'] = mse
                            
                            # 追加のメトリック
                            fold_result['treated_size'] = len(treated_df)
                            
                            print(f"  処置群評価成功: サイズ={len(treated_df)}")
                            
                        except Exception as e:
                            print(f"  処置群評価エラー: {str(e)}")
                    
                    if len(control_df) > 0 and 'y_pred' in control_df.columns:
                        # 対照群の評価
                        try:
                            if learner_type == 'classification':
                                # 二値分類に変換
                                control_df['y_pred_binary'] = (control_df['y_pred'] > classification_threshold).astype(int)
                                
                                # 分類指標の計算
                                accuracy = accuracy_score(control_df['outcome'], control_df['y_pred_binary'])
                                
                                try:
                                    roc_auc = roc_auc_score(control_df['outcome'], control_df['y_pred'])
                                except Exception as e:
                                    print(f"  対照群AUC計算エラー: {str(e)}")
                                    roc_auc = 0.5
                                
                                precision = precision_score(control_df['outcome'], control_df['y_pred_binary'], zero_division=0)
                                recall = recall_score(control_df['outcome'], control_df['y_pred_binary'], zero_division=0)
                                f1 = f1_score(control_df['outcome'], control_df['y_pred_binary'], zero_division=0)
                                
                                # 評価結果を保存
                                fold_result['control_accuracy'] = accuracy
                                fold_result['control_roc_auc'] = roc_auc
                                fold_result['control_precision'] = precision
                                fold_result['control_recall'] = recall
                                fold_result['control_f1'] = f1
                                
                                # 混同行列
                                cm = confusion_matrix(control_df['outcome'], control_df['y_pred_binary'])
                                fold_result['control_outcome_conf_mat'] = cm
                                
                                # 全体の混同行列も更新
                                if 'control_outcome_conf_mat_all' not in all_fold_results:
                                    all_fold_results['control_outcome_conf_mat_all'] = cm
                                else:
                                    all_fold_results['control_outcome_conf_mat_all'] += cm
                            
                            # 回帰指標（分類/回帰共通）
                            mse = mean_squared_error(control_df['outcome'], control_df['y_pred'])
                            fold_result['control_outcome_mse'] = mse
                            
                            # 追加のメトリック
                            fold_result['control_size'] = len(control_df)
                            
                            print(f"  対照群評価成功: サイズ={len(control_df)}")
                            
                        except Exception as e:
                            print(f"  対照群評価エラー: {str(e)}")
                
                except Exception as e:
                    print(f"  Fold {fold_idx} 目的変数評価エラー: {str(e)}")
            
            # 実際の処置予測（y_pred_actual）がある場合の評価
            if y_pred_actual is not None:
                # 実際の予測のMSE
                actual_mse = mean_squared_error(outcome_test, y_pred_actual)
                fold_result['actual_outcome_mse'] = actual_mse
                
                # 実際の予測のR²
                actual_r2 = r2_score(outcome_test, y_pred_actual)
                fold_result['actual_outcome_r2'] = actual_r2
                
                # 分類の場合の精度
                if learner_type == 'classification':
                    actual_true_binary = outcome_test.astype(int)
                    actual_pred_binary = _binarize_predictions(y_pred_actual, classification_threshold)
                    
                    # 正解率
                    try:
                        actual_accuracy = accuracy_score(actual_true_binary, actual_pred_binary)
                        fold_result['actual_outcome_accuracy'] = actual_accuracy
                    except:
                        pass
                    
                    # AUC-ROC
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                            actual_auc_roc = roc_auc_score(actual_true_binary, y_pred_actual)
                        fold_result['actual_outcome_auc_roc'] = actual_auc_roc
                    except:
                        pass
                    
                    # 混同行列
                    try:
                        actual_conf_mat = confusion_matrix(actual_true_binary, actual_pred_binary)
                        fold_result['actual_outcome_conf_mat'] = actual_conf_mat
                    except:
                        pass
                    
                    # 精度・再現率・F1スコア
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                            actual_precision = precision_score(actual_true_binary, actual_pred_binary)
                            actual_recall = recall_score(actual_true_binary, actual_pred_binary)
                            actual_f1 = f1_score(actual_true_binary, actual_pred_binary)
                        fold_result['actual_outcome_precision'] = actual_precision
                        fold_result['actual_outcome_recall'] = actual_recall
                        fold_result['actual_outcome_f1'] = actual_f1
                    except:
                        pass
            
            # このfoldの結果を保存
            fold_result['fold_idx'] = fold_idx
            fold_result['test_size'] = len(test_idx)
            fold_result['treated_size'] = np.sum(treatment_test)
            fold_result['control_size'] = len(treatment_test) - np.sum(treatment_test)
            fold_results.append(fold_result)
            
            # foldのサンプル数を記録（平均値計算用）
            if 'fold_counts' not in all_fold_results:
                all_fold_results['fold_counts'] = {}
            all_fold_results['fold_counts'][fold_idx] = {
                'treated': np.sum(treatment_test),
                'control': len(treatment_test) - np.sum(treatment_test),
                'test_size': len(test_idx)
            }
            
        except Exception as e:
            print(f"Fold {fold_idx} でエラーが発生しました: {str(e)}")
            continue
    
    # 全foldの平均結果を計算
    if not fold_results:
        print("すべてのfoldで評価に失敗しました")
        return {}
    
    # 最終結果の辞書を作成
    final_results = {}
    
    # 各メトリクスについて平均と標準偏差を計算
    metric_keys = set()
    for result in fold_results:
        metric_keys.update(result.keys())
    
    # 混同行列とその他の特殊な項目を除外
    excluded_keys = {'fold_idx', 'conf_mat', 'treated_outcome_conf_mat', 'control_outcome_conf_mat', 
                     'overall_outcome_conf_mat', 'actual_outcome_conf_mat', 'class_report',
                     'treated_outcome_class_report', 'control_outcome_class_report', 
                     'overall_outcome_class_report', 'actual_outcome_class_report'}
    
    # 数値メトリクスの平均と標準偏差を計算
    for key in metric_keys:
        if key in excluded_keys:
            continue
        
        # 該当するメトリクスを持つfoldの結果を収集
        values = [result[key] for result in fold_results if key in result]
        
        if values:
            # 平均値を計算
            final_results[key] = np.mean(values)
            # 標準偏差も計算して保存（オプション）
            final_results[f"{key}_std"] = np.std(values)
    
    # サンプルサイズの情報を追加
    final_results['total_folds'] = len(fold_results)
    final_results['avg_test_size'] = np.mean([result['test_size'] for result in fold_results])
    final_results['avg_treated_size'] = np.mean([result['treated_size'] for result in fold_results])
    final_results['avg_control_size'] = np.mean([result['control_size'] for result in fold_results])
    
    # 混同行列は合計値として保存（オプション）
    for mat_key in ['conf_mat', 'treated_outcome_conf_mat', 'control_outcome_conf_mat', 
                   'overall_outcome_conf_mat', 'actual_outcome_conf_mat']:
        matrices = [result[mat_key] for result in fold_results if mat_key in result]
        if matrices:
            try:
                # すべての混同行列が同じ形状であることを確認
                if all(m.shape == matrices[0].shape for m in matrices):
                    # 合計の混同行列を計算
                    final_results[mat_key] = sum(matrices)
            except:
                # 合計できない場合は最初のfoldの混同行列を使用
                final_results[mat_key] = matrices[0]
    
    # 元の各foldの詳細結果も保存（オプション）
    final_results['fold_details'] = fold_results
    
    return final_results

def evaluate_all_models_cv(X, treatment, outcome, true_ite=None, propensity_score=None,
                          models_config=None, n_splits=5, random_state=42):
    """
    複数のモデルをk分割交差検証で評価する関数
    
    Parameters:
    ----------
    X : array-like or DataFrame
        特徴量行列
    treatment : array-like
        処置変数（1=処置、0=非処置）
    outcome : array-like
        目的変数
    true_ite : array-like, optional
        真の個別処置効果（検証用）
    propensity_score : array-like, optional
        傾向スコア（与えられれば使用）
    models_config : list of dict, optional
        評価するモデルの設定リスト。各辞書は以下のキーを持つ:
        - 'model_type': モデルタイプ ('s_learner', 't_learner', 'x_learner', 'r_learner', 'dr_learner')
        - 'learner_type': 学習器タイプ ('classification', 'regression')
    n_splits : int, default=5
        交差検証の分割数
    random_state : int, default=42
        乱数シード
        
    Returns:
    -------
    dict
        各モデルの評価指標の辞書
    """
    # デフォルトのモデル設定
    if models_config is None:
        models_config = [
            {'model_type': 's_learner', 'learner_type': 'classification'},
            {'model_type': 't_learner', 'learner_type': 'classification'},
            {'model_type': 's_learner', 'learner_type': 'regression'},
            {'model_type': 't_learner', 'learner_type': 'regression'},
            {'model_type': 'x_learner', 'learner_type': 'regression'},
            {'model_type': 'r_learner', 'learner_type': 'regression'},
            {'model_type': 'dr_learner', 'learner_type': 'regression'}
        ]
    
    # 各モデルの評価結果を格納するための辞書
    all_results = {}
    
    # 各モデルを評価
    for config in models_config:
        model_type = config['model_type']
        learner_type = config['learner_type']
        model_id = f"{model_type}_{learner_type.split('_')[0]}"
        
        print(f"\n{model_id}を評価中...")
        
        # モデルのCVによる評価を実行
        results = evaluate_model_cv(
            X, treatment, outcome,
            true_ite=true_ite,
            model_type=model_type,
            learner_type=learner_type, 
            propensity_score=propensity_score,
            n_splits=n_splits,
            random_state=random_state
        )
        
        # 結果を格納
        all_results[model_id] = results
    
    return all_results

def print_cv_results(all_results, n_splits=5):
    """
    交差検証の結果を整形して表示する関数
    
    Parameters:
    ----------
    all_results : dict
        evaluate_all_models_cvの戻り値
    n_splits : int, default=5
        交差検証の分割数
    """
    # 各モデルの結果を表示
    for model_id, results in all_results.items():
        print(f"\n{model_id}:")
        
        # 結果がない場合はスキップ
        if not results:
            print("  評価結果がありません")
            continue
        
        # 処置効果の評価結果
        if 'correlation' in results:
            print(f"  真のITEとの相関: {results['correlation']:.4f}")
        
        if 'mse' in results:
            print(f"  真のITEとのMSE: {results['mse']:.4f}")
        
        if 'sign_accuracy' in results:
            print(f"  正負判定精度: {results['sign_accuracy']:.4f}")
        
        if 'r2' in results:
            print(f"  テストR²: {results['r2']:.4f}")
        
        if 'auc_roc' in results:
            print(f"  AUC-ROC: {results['auc_roc']:.4f}")
        
        if 'conf_mat' in results:
            print(f"  混同行列:\n{results['conf_mat']}")
        
        if 'precision' in results:
            print(f"  精度: {results['precision']:.4f}")
        
        if 'recall' in results:
            print(f"  再現率: {results['recall']:.4f}")
        
        if 'f1' in results:
            print(f"  F1スコア: {results['f1']:.4f}")
        
        if 'class_report' in results:
            print(f"  分類レポート:\n{results['class_report']}")
        
        # 目的変数の予測評価結果
        if 'treated_outcome_mse' in results:
            print(f"  処置群の目的変数MSE: {results['treated_outcome_mse']:.4f}")
        
        if 'control_outcome_mse' in results:
            print(f"  非処置群の目的変数MSE: {results['control_outcome_mse']:.4f}")
        
        if 'overall_outcome_mse' in results:
            print(f"  全体の目的変数MSE: {results['overall_outcome_mse']:.4f}")
        
        if 'treated_outcome_r2' in results:
            print(f"  処置群の目的変数R²: {results['treated_outcome_r2']:.4f}")
        
        if 'control_outcome_r2' in results:
            print(f"  非処置群の目的変数R²: {results['control_outcome_r2']:.4f}")
        
        if 'overall_outcome_r2' in results:
            print(f"  全体の目的変数R²: {results['overall_outcome_r2']:.4f}")
        
        # 分類の場合の精度
        if 'treated_outcome_accuracy' in results:
            print(f"  処置群の分類精度: {results['treated_outcome_accuracy']:.4f}")
        
        if 'control_outcome_accuracy' in results:
            print(f"  非処置群の分類精度: {results['control_outcome_accuracy']:.4f}")
        
        if 'overall_outcome_accuracy' in results:
            print(f"  全体の分類精度: {results['overall_outcome_accuracy']:.4f}")
        
        # 直接予測の評価結果
        if 'actual_outcome_mse' in results:
            print(f"  直接予測の目的変数MSE: {results['actual_outcome_mse']:.4f}")
        
        if 'actual_outcome_r2' in results:
            print(f"  直接予測の目的変数R²: {results['actual_outcome_r2']:.4f}")
        
        if 'actual_outcome_accuracy' in results:
            print(f"  直接予測の分類精度: {results['actual_outcome_accuracy']:.4f}")

def run_cv_with_statistics(all_results_by_fold, fold_indices):
    """
    複数回のクロスバリデーション結果の統計量を計算して表示する関数
    
    Parameters:
    ----------
    all_results_by_fold : list of dict
        各foldのevaluate_all_models_cvの戻り値リスト
    fold_indices : array-like
        各foldのインデックス
        
    Returns:
    -------
    dict
        各モデル・各指標の平均と標準偏差を含む辞書
    """
    # 統計結果を格納する辞書
    stats_results = {}
    
    # 最初のfoldの結果からモデルIDを取得
    if not all_results_by_fold:
        print("結果がありません")
        return stats_results
    
    model_ids = list(all_results_by_fold[0].keys())
    
    # 各モデルについて処理
    for model_id in model_ids:
        stats_results[model_id] = {}
        
        # 各foldの結果を集める
        model_results_by_fold = []
        for fold_idx in fold_indices:
            if fold_idx < len(all_results_by_fold) and model_id in all_results_by_fold[fold_idx]:
                model_results_by_fold.append(all_results_by_fold[fold_idx][model_id])
        
        # 結果がなければスキップ
        if not model_results_by_fold:
            continue
        
        # 最初のfoldの結果から指標を取得
        metrics = list(model_results_by_fold[0].keys())
        
        # 各指標について統計量を計算
        for metric in metrics:
            metric_values = [results[metric] for results in model_results_by_fold if metric in results]
            
            if metric_values:
                mean_value = np.mean(metric_values)
                std_value = np.std(metric_values)
                stats_results[model_id][metric] = (mean_value, std_value)
    
    # 結果の表示
    for model_id, metrics in stats_results.items():
        print(f"\n{model_id}:")
        
        if not metrics:
            print("  統計結果がありません")
            continue
        
        # 処置効果の評価統計
        if 'correlation' in metrics:
            mean, std = metrics['correlation']
            print(f"  真のITEとの相関: {mean:.4f} (±{std:.4f})")
        
        if 'mse' in metrics:
            mean, std = metrics['mse']
            print(f"  真のITEとのMSE: {mean:.4f} (±{std:.4f})")
        
        if 'sign_accuracy' in metrics:
            mean, std = metrics['sign_accuracy']
            print(f"  正負判定精度: {mean:.4f} (±{std:.4f})")
        
        if 'r2' in metrics:
            mean, std = metrics['r2']
            print(f"  テストR²: {mean:.4f} (±{std:.4f})")
        
        if 'auc_roc' in metrics:
            mean, std = metrics['auc_roc']
            print(f"  AUC-ROC: {mean:.4f} (±{std:.4f})")
        
        if 'conf_mat' in metrics:
            print(f"  混同行列:\n{metrics['conf_mat']}")
        
        if 'precision' in metrics:
            mean, std = metrics['precision']
            print(f"  精度: {mean:.4f} (±{std:.4f})")
        
        if 'recall' in metrics:
            mean, std = metrics['recall']
            print(f"  再現率: {mean:.4f} (±{std:.4f})")
        
        if 'f1' in metrics:
            mean, std = metrics['f1']
            print(f"  F1スコア: {mean:.4f} (±{std:.4f})")
        
        if 'class_report' in metrics:
            print(f"  分類レポート:\n{metrics['class_report']}")
        
        # 目的変数の予測評価統計
        if 'treated_outcome_mse' in metrics:
            mean, std = metrics['treated_outcome_mse']
            print(f"  処置群の目的変数MSE: {mean:.4f} (±{std:.4f})")
        
        if 'control_outcome_mse' in metrics:
            mean, std = metrics['control_outcome_mse']
            print(f"  非処置群の目的変数MSE: {mean:.4f} (±{std:.4f})")
        
        if 'overall_outcome_mse' in metrics:
            mean, std = metrics['overall_outcome_mse']
            print(f"  全体の目的変数MSE: {mean:.4f} (±{std:.4f})")
        
        if 'treated_outcome_r2' in metrics:
            mean, std = metrics['treated_outcome_r2']
            print(f"  処置群の目的変数R²: {mean:.4f} (±{std:.4f})")
        
        if 'control_outcome_r2' in metrics:
            mean, std = metrics['control_outcome_r2']
            print(f"  非処置群の目的変数R²: {mean:.4f} (±{std:.4f})")
        
        if 'overall_outcome_r2' in metrics:
            mean, std = metrics['overall_outcome_r2']
            print(f"  全体の目的変数R²: {mean:.4f} (±{std:.4f})")
        
        # 分類の場合の精度統計
        if 'treated_outcome_accuracy' in metrics:
            mean, std = metrics['treated_outcome_accuracy']
            print(f"  処置群の分類精度: {mean:.4f} (±{std:.4f})")
        
        if 'control_outcome_accuracy' in metrics:
            mean, std = metrics['control_outcome_accuracy']
            print(f"  非処置群の分類精度: {mean:.4f} (±{std:.4f})")
        
        if 'overall_outcome_accuracy' in metrics:
            mean, std = metrics['overall_outcome_accuracy']
            print(f"  全体の分類精度: {mean:.4f} (±{std:.4f})")
        
        # 直接予測の評価統計
        if 'actual_outcome_mse' in metrics:
            mean, std = metrics['actual_outcome_mse']
            print(f"  直接予測の目的変数MSE: {mean:.4f} (±{std:.4f})")
        
        if 'actual_outcome_r2' in metrics:
            mean, std = metrics['actual_outcome_r2']
            print(f"  直接予測の目的変数R²: {mean:.4f} (±{std:.4f})")
        
        if 'actual_outcome_accuracy' in metrics:
            mean, std = metrics['actual_outcome_accuracy']
            print(f"  直接予測の分類精度: {mean:.4f} (±{std:.4f})")
    
    return stats_results
