"""
Open Bandit Platformを使用したDoubly Robust評価スクリプト
因果推論データをOff-Policy評価するためのコード
"""
import numpy as np
import pandas as pd
import argparse
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgbm
from causalml.inference.meta import BaseSLearner, BaseSRegressor, BaseXRegressor, BaseTRegressor, BaseRRegressor
from tqdm import tqdm
import time
import warnings

# 警告を無視する設定（LightGBMの特徴量名に関する警告を抑制）
warnings.filterwarnings("ignore", message="Feature names .+ are not unique")

# OBP インポート
from obp.policy import Random
from obp.ope.regression_model import RegressionModel
from obp.ope.estimators import (
    DirectMethod, 
    InverseProbabilityWeighting, 
    DoublyRobust,
    SwitchDoublyRobust,
    DoublyRobustWithShrinkage,
    SelfNormalizedInverseProbabilityWeighting
)
from obp.ope import OffPolicyEvaluation

# 様々な回帰モデルをラップするクラス
class CustomRegressionModel:
    """
    様々な回帰モデルをラップして、OBPのRegressionModelと互換性を持たせるクラス
    """
    def __init__(self, base_model=None):
        """
        Parameters
        ----------
        base_model : estimator object, optional
            基本となるモデル。指定しない場合はLGBMRegressorが使用される。
        """
        self.base_model = base_model or lgbm.LGBMRegressor(n_estimators=100, learning_rate=0.05)
        self.n_actions = 2
        self.len_list = 1
        self.estimated_rewards_by_reg_model = None
        self.action_models = {}  # 各アクションに対するモデル
    
    def fit_predict(self, context, action, reward, n_folds=2, random_state=12345):
        """
        各アクションごとに別々の回帰モデルを学習し、予測する
        
        Parameters
        ----------
        context : array-like, shape (n_samples, n_features)
            特徴量
        action : array-like, shape (n_samples,)
            行動
        reward : array-like, shape (n_samples,)
            報酬
        n_folds : int, optional
            交差検証の分割数（現在は使用していない）
        random_state : int, optional
            乱数シード
            
        Returns
        -------
        estimated_rewards_by_reg_model : array-like, shape (n_samples, n_actions, len_list)
            各サンプル、各行動に対する予測報酬
        """
        # 各行動に対する予測報酬を計算
        n_samples = len(context)
        self.estimated_rewards_by_reg_model = np.zeros((n_samples, self.n_actions, self.len_list))
        
        # 各アクションごとに別々のモデルを学習
        for action_id in range(self.n_actions):
            # そのアクションのデータだけを抽出
            action_indices = action == action_id
            if np.sum(action_indices) > 0:
                X_action = context[action_indices]
                y_action = reward[action_indices]
                
                # モデルのコピーを作成
                if isinstance(self.base_model, lgbm.LGBMRegressor):
                    model = lgbm.LGBMRegressor(**self.base_model.get_params())
                elif isinstance(self.base_model, RandomForestRegressor):
                    model = RandomForestRegressor(**self.base_model.get_params())
                elif isinstance(self.base_model, GradientBoostingRegressor):
                    model = GradientBoostingRegressor(**self.base_model.get_params())
                elif isinstance(self.base_model, LinearRegression):
                    model = LinearRegression(**self.base_model.get_params())
                elif isinstance(self.base_model, Ridge):
                    model = Ridge(**self.base_model.get_params())
                elif isinstance(self.base_model, Lasso):
                    model = Lasso(**self.base_model.get_params())
                elif isinstance(self.base_model, MLPRegressor):
                    model = MLPRegressor(**self.base_model.get_params())
                elif isinstance(self.base_model, GaussianProcessRegressor):
                    # GaussianProcessRegressorはカーネルが特殊なので、シンプルに初期化
                    model = GaussianProcessRegressor(
                        kernel=RBF(),
                        random_state=42
                    )
                else:
                    # その他のモデルの場合はそのまま使用
                    model = self.base_model
                
                # モデルを学習
                model.fit(X_action, y_action)
                self.action_models[action_id] = model
                
                # 全サンプルに対して予測
                self.estimated_rewards_by_reg_model[:, action_id, 0] = model.predict(context)
            else:
                # そのアクションのデータがない場合は0を予測
                self.estimated_rewards_by_reg_model[:, action_id, 0] = 0
        
        # 回帰モデルの性能評価
        self._evaluate_regression_performance(context, action, reward)
        
        # ITE予測の評価（真のITEがある場合）
        self._evaluate_ite_prediction()
        
        return self.estimated_rewards_by_reg_model
    
    def _evaluate_regression_performance(self, context, action, reward):
        """回帰モデルの性能を評価する"""
        print("\n=== 回帰モデルの性能評価 ===")
        
        # 各アクションごとの性能評価
        for action_id in range(self.n_actions):
            action_indices = action == action_id
            if np.sum(action_indices) > 0 and action_id in self.action_models:
                y_true = reward[action_indices]
                y_pred = self.estimated_rewards_by_reg_model[action_indices, action_id, 0]
                
                mse = mean_squared_error(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                
                print(f"\nアクション {action_id} の回帰モデル性能:")
                print(f"  サンプル数: {len(y_true)}")
                print(f"  MSE: {mse:.6f}")
                print(f"  MAE: {mae:.6f}")
                print(f"  R2スコア: {r2:.6f}")
        
        # 全体の性能評価
        y_true = reward
        y_pred = np.zeros_like(reward)
        for i in range(len(action)):
            y_pred[i] = self.estimated_rewards_by_reg_model[i, action[i], 0]
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print("\n全体の回帰モデル性能:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  R2スコア: {r2:.6f}")
    
    def _evaluate_ite_prediction(self):
        """ITE予測の性能を評価する（真のITEがある場合）"""
        if 'true_ite' in globals():
            true_ite = globals()['true_ite']
            # 予測ITEを計算
            pred_ite = self.estimated_rewards_by_reg_model[:, 1, 0] - self.estimated_rewards_by_reg_model[:, 0, 0]
            
            # 予測ITEと真のITEの相関係数を計算
            correlation = np.corrcoef(pred_ite, true_ite)[0, 1]
            print(f"\n=== ITE予測の性能評価 ===")
            print(f"予測ITEと真のITEの相関係数: {correlation:.4f}")
            
            # 上位k人の一致率を計算
            k = 2000  # 上位k人
            pred_top_k = np.argsort(pred_ite)[-k:]
            true_top_k = np.argsort(true_ite)[-k:]
            overlap = np.intersect1d(pred_top_k, true_top_k)
            print(f"予測上位{k}人と真のITE上位{k}人の一致数: {len(overlap)}人")
            print(f"一致率: {len(overlap) / k * 100:.2f}%")
            
            # 真のITEが正の人を何人選択できたか
            true_positive = np.sum(true_ite[pred_top_k] > 0)
            print(f"予測上位{k}人のうち、真のITEが正の人数: {true_positive}人")
            print(f"正の効果の選択率: {true_positive / k * 100:.2f}%")

# CausalMLのS-Learnerをラップするクラス
class SLearnerRegressionModel:
    """
    CausalMLのS-Learnerをラップして、OBPのRegressionModelと互換性を持たせるクラス
    """
    def __init__(self, base_model=None, model_type='regressor'):
        """
        Parameters
        ----------
        base_model : estimator object, optional
            基本となるモデル。指定しない場合はLGBMRegressorが使用される。
        model_type : str, optional
            'regressor'または'classifier'。デフォルトは'regressor'。
        """
        self.base_model = base_model or lgbm.LGBMRegressor(n_estimators=100, learning_rate=0.05)
        self.model_type = model_type
        self.n_actions = 2
        self.len_list = 1
        self.estimated_rewards_by_reg_model = None
        
        # S-Learnerの初期化
        if model_type == 'regressor':
            self.model = BaseSRegressor(learner=self.base_model)
        else:
            self.model = BaseSLearner(learner=self.base_model)
    
    def fit_predict(self, context, action, reward, n_folds=2, random_state=12345):
        """
        S-Learnerを使用して報酬関数を学習し、予測する
        
        Parameters
        ----------
        context : array-like, shape (n_samples, n_features)
            特徴量
        action : array-like, shape (n_samples,)
            行動
        reward : array-like, shape (n_samples,)
            報酬
        n_folds : int, optional
            交差検証の分割数
        random_state : int, optional
            乱数シード
            
        Returns
        -------
        estimated_rewards_by_reg_model : array-like, shape (n_samples, n_actions, len_list)
            各サンプル、各行動に対する予測報酬
        """
        # CausalMLのS-Learnerの入力形式に変換
        X = context
        treatment = action
        y = reward
        
        # S-Learnerを学習
        self.model.fit(X=X, treatment=treatment, y=y)
        
        # 各行動に対する予測報酬を計算
        n_samples = len(context)
        self.estimated_rewards_by_reg_model = np.zeros((n_samples, self.n_actions, self.len_list))
        
        # 処置なし（action=0）の場合の予測
        t0_pred = self.model.predict(X, treatment=np.zeros(n_samples))
        # 形状を調整（CausalMLは(n_samples, 1)を返す可能性がある）
        if t0_pred.ndim > 1 and t0_pred.shape[1] == 1:
            t0_pred = t0_pred.flatten()
        self.estimated_rewards_by_reg_model[:, 0, 0] = t0_pred
        
        # 処置あり（action=1）の場合の予測
        t1_pred = self.model.predict(X, treatment=np.ones(n_samples))
        # 形状を調整
        if t1_pred.ndim > 1 and t1_pred.shape[1] == 1:
            t1_pred = t1_pred.flatten()
        self.estimated_rewards_by_reg_model[:, 1, 0] = t1_pred
        
        # 性能評価
        self._evaluate_regression_performance(context, action, reward, t0_pred, t1_pred)
        
        # ITE予測の評価（真のITEがある場合）
        self._evaluate_ite_prediction(t0_pred, t1_pred)
        
        return self.estimated_rewards_by_reg_model
    
    def _evaluate_regression_performance(self, context, action, reward, t0_pred, t1_pred):
        """回帰モデルの性能を評価する"""
        print("\n=== S-Learnerの回帰モデル性能評価 ===")
        
        # 各アクションごとの性能評価
        action_0_indices = action == 0
        action_1_indices = action == 1
        
        # アクション0の回帰モデル性能
        if np.sum(action_0_indices) > 0:
            y_true_0 = reward[action_0_indices]
            y_pred_0 = t0_pred[action_0_indices]
            mse_0 = mean_squared_error(y_true_0, y_pred_0)
            mae_0 = mean_absolute_error(y_true_0, y_pred_0)
            r2_0 = r2_score(y_true_0, y_pred_0)
            print(f"\nアクション 0 の回帰モデル性能:")
            print(f"  サンプル数: {len(y_true_0)}")
            print(f"  MSE: {mse_0:.6f}")
            print(f"  MAE: {mae_0:.6f}")
            print(f"  R2スコア: {r2_0:.6f}")
        
        # アクション1の回帰モデル性能
        if np.sum(action_1_indices) > 0:
            y_true_1 = reward[action_1_indices]
            y_pred_1 = t1_pred[action_1_indices]
            mse_1 = mean_squared_error(y_true_1, y_pred_1)
            mae_1 = mean_absolute_error(y_true_1, y_pred_1)
            r2_1 = r2_score(y_true_1, y_pred_1)
            print(f"\nアクション 1 の回帰モデル性能:")
            print(f"  サンプル数: {len(y_true_1)}")
            print(f"  MSE: {mse_1:.6f}")
            print(f"  MAE: {mae_1:.6f}")
            print(f"  R2スコア: {r2_1:.6f}")
        
        # 全体の回帰モデル性能
        y_true = reward
        y_pred = np.zeros_like(reward)
        y_pred[action_0_indices] = t0_pred[action_0_indices]
        y_pred[action_1_indices] = t1_pred[action_1_indices]
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"\n全体の回帰モデル性能:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  R2スコア: {r2:.6f}")
    
    def _evaluate_ite_prediction(self, t0_pred, t1_pred):
        """ITE予測の性能を評価する（真のITEがある場合）"""
        if 'true_ite' in globals():
            true_ite = globals()['true_ite']
            # 予測ITEを計算
            pred_ite = t1_pred - t0_pred
            
            # 予測ITEと真のITEの相関係数を計算
            correlation = np.corrcoef(pred_ite, true_ite)[0, 1]
            print(f"\n=== S-Learnerのアップリフト予測性能 ===")
            print(f"予測ITEと真のITEの相関係数: {correlation:.4f}")
            
            # 上位k人の一致率を計算
            k = 2000  # 上位k人
            pred_top_k = np.argsort(pred_ite)[-k:]
            true_top_k = np.argsort(true_ite)[-k:]
            overlap = np.intersect1d(pred_top_k, true_top_k)
            print(f"予測上位{k}人と真のITE上位{k}人の一致数: {len(overlap)}人")
            print(f"一致率: {len(overlap) / k * 100:.2f}%")
            
            # 真のITEが正の人を何人選択できたか
            true_positive = np.sum(true_ite[pred_top_k] > 0)
            print(f"予測上位{k}人のうち、真のITEが正の人数: {true_positive}人")
            print(f"正の効果の選択率: {true_positive / k * 100:.2f}%")

def load_and_preprocess_data(csv_path):
    """CSVファイルからデータを読み込み、前処理する
    
    Parameters
    ----------
    csv_path : str
        CSVファイルのパス
        
    Returns
    -------
    X : numpy.ndarray
        特徴量
    A : numpy.ndarray
        行動
    R : numpy.ndarray
        報酬
    p_b : numpy.ndarray
        行動確率（propensity score）
    bandit_feedback : dict
        バンディットフィードバック
    p_e_random : numpy.ndarray
        ランダム方策の行動確率
    p_e_original : numpy.ndarray
        元のログ方策の行動確率
    p_e_ite : numpy.ndarray
        ITE予測値に基づく方策の行動確率
    p_e_oracle : numpy.ndarray
        オラクル最適方策の行動確率
    p_e_oracle_top : numpy.ndarray
        オラクル上位方策の行動確率
    df : pandas.DataFrame
        データフレーム
    true_ite : numpy.ndarray, optional
        真のITE（存在する場合）
    """
    # CSVファイルを読み込む
    print(f"CSVファイルを読み込み: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"データフレームの形状: {df.shape}")
    print(f"カラム: {df.columns.tolist()}")
    print(df.head())
    
    # 特徴量、行動、報酬を抽出
    feature_names = ['age', 'homeownership']
    X_df = df[feature_names]  # DataFrameとして保持
    X = X_df.values  # NumPy配列に変換
    A = df['treatment'].values
    R = df['outcome'].values
    
    # 傾向スコアを抽出または計算
    if 'propensity_score' in df.columns:
        p_b = df['propensity_score'].values
    else:
        # 傾向スコアがない場合は、ランダム方策として0.2を設定
        p_b = np.ones(len(df)) * 0.2
    
    # 真のITEを抽出（存在する場合）
    true_ite = None
    if 'true_ite' in df.columns:
        true_ite = df['true_ite'].values
        # グローバル変数に設定（SLearnerRegressionModelで使用）
        globals()['true_ite'] = true_ite
    
    # バンディットフィードバックを作成
    n_actions = 2  # 処置あり/なしの2つの行動
    len_list = 1   # 各ユーザーに対して1つの行動を選択
    
    bandit_feedback = {
        'n_rounds': len(df),
        'n_actions': n_actions,
        'context': X,
        'action': A,
        'reward': R,
        'position': np.zeros(len(df), dtype=int),  # 位置は常に0
        'pscore': p_b,
        'context_set': np.zeros((len(df), n_actions, len_list)),
        'action_set': np.zeros((len(df), n_actions, len_list)),
        'reward_set': np.zeros((len(df), n_actions, len_list)),
        'len_list': len_list
    }
    
    # ランダム方策の行動確率を計算
    # 処置割合を20%に設定
    treatment_ratio = 0.2
    
    # ランダムに20%のユーザーを選んで処置する方策
    np.random.seed(42)  # 再現性のために乱数シードを固定
    random_indices = np.random.choice(len(df), size=int(len(df) * treatment_ratio), replace=False)
    
    # 初期化（すべて非処置）
    p_e_random = np.zeros((len(df), 2))
    p_e_random[:, 0] = 1.0  # すべての行の非処置確率を1に
    
    # ランダムに選んだユーザーのみ処置に変更
    p_e_random[random_indices, 0] = 0.0  # 非処置確率を0に
    p_e_random[random_indices, 1] = 1.0  # 処置確率を1に
    
    # 確率分布の検証（各行の合計が1になることを確認）
    assert np.allclose(np.sum(p_e_random, axis=1), 1.0), "ランダム方策が確率分布になっていません"
    
    print(f"ランダム方策のアクション分布の形状: {p_e_random.shape}, 次元数: {p_e_random.ndim}")
    print(f"ランダム方策での処置人数: {np.sum(p_e_random[:, 1])}")
    
    # 3次元に変換（n_rounds, n_actions, len_list）
    p_e_random = p_e_random.reshape(len(df), 2, 1)
    print(f"3次元に変換後のランダム方策の形状: {p_e_random.shape}")
    
    # 元のログ方策の行動確率を計算
    p_e_original = np.zeros((len(df), 2))
    # 処置を受けた人は処置確率=1、非処置確率=0
    p_e_original[A == 1, 1] = 1.0
    p_e_original[A == 1, 0] = 0.0
    # 処置を受けなかった人は処置確率=0、非処置確率=1
    p_e_original[A == 0, 0] = 1.0
    p_e_original[A == 0, 1] = 0.0
    
    # 確率分布の検証（各行の合計が1になることを確認）
    assert np.allclose(np.sum(p_e_original, axis=1), 1.0), "元のログ方策が確率分布になっていません"
    
    p_e_original = p_e_original.reshape(len(df), 2, 1)
    
    # ITE予測値に基づく方策の行動確率を計算
    if 'ite_pred' in df.columns:
        # ITE予測値の上位20%に処置を割り当てる方策
        ite_pred = df['ite_pred'].values
        top_indices = np.argsort(-ite_pred)[:int(len(df) * treatment_ratio)]
        
        # 初期化（すべて非処置）
        p_e_ite = np.zeros((len(df), 2))
        p_e_ite[:, 0] = 1.0  # すべての行の非処置確率を1に
        
        # 上位20%のみ処置に変更
        p_e_ite[top_indices, 0] = 0.0  # 非処置確率を0に
        p_e_ite[top_indices, 1] = 1.0  # 処置確率を1に
        
        print(f"ITE方策の形状: {p_e_ite.shape}, 次元数: {p_e_ite.ndim}")
        
        # 3次元に変換
        p_e_ite = p_e_ite.reshape(len(df), 2, 1)
        print(f"3次元に変換後のITE方策の形状: {p_e_ite.shape}")
        
        print(f"ITE予測値の上位{int(len(df) * treatment_ratio)}人に処置するポリシーを作成しました")
        print(f"処置割合: {treatment_ratio*100:.2f}%")
    else:
        p_e_ite = None
    
    # オラクル情報（真のITE）がある場合、それに基づく最適方策を作成
    if 'true_ite' in df.columns:
        true_ite = df['true_ite'].values
        
        # 真のITEが正の人すべてに処置を割り当てる方策
        positive_ite_indices = np.where(true_ite > 0)[0]
        
        # 初期化（すべて非処置）
        p_e_oracle = np.zeros((len(df), 2))
        p_e_oracle[:, 0] = 1.0  # すべての行の非処置確率を1に
        
        # 真のITEが正の人のみ処置に変更
        p_e_oracle[positive_ite_indices, 0] = 0.0  # 非処置確率を0に
        p_e_oracle[positive_ite_indices, 1] = 1.0  # 処置確率を1に
        
        # 3次元に変換
        p_e_oracle = p_e_oracle.reshape(len(df), 2, 1)
        
        print(f"オラクル最適方策での処置人数: {len(positive_ite_indices)}")
        
        # 真のITEの上位20%に処置を割り当てる方策
        top_true_ite_indices = np.argsort(-true_ite)[:int(len(df) * treatment_ratio)]
        
        # 初期化（すべて非処置）
        p_e_oracle_top = np.zeros((len(df), 2))
        p_e_oracle_top[:, 0] = 1.0  # すべての行の非処置確率を1に
        
        # 上位20%のみ処置に変更
        p_e_oracle_top[top_true_ite_indices, 0] = 0.0  # 非処置確率を0に
        p_e_oracle_top[top_true_ite_indices, 1] = 1.0  # 処置確率を1に
        
        # 3次元に変換
        p_e_oracle_top = p_e_oracle_top.reshape(len(df), 2, 1)
        
        print(f"オラクル上位{int(len(df) * treatment_ratio)}人方策での処置人数: {int(len(df) * treatment_ratio)}")
        
        # 予測ITEと真のITEの相関を計算
        if 'ite_pred' in df.columns:
            correlation = np.corrcoef(ite_pred, true_ite)[0, 1]
            print(f"\n予測ITEと真のITEの相関係数: {correlation:.4f}")
            
            # 予測上位と真のITE上位の一致度を計算
            pred_top = set(np.argsort(-ite_pred)[:int(len(df) * treatment_ratio)])
            true_top = set(np.argsort(-true_ite)[:int(len(df) * treatment_ratio)])
            overlap = len(pred_top.intersection(true_top))
            print(f"予測上位{int(len(df) * treatment_ratio)}人と真のITE上位{int(len(df) * treatment_ratio)}人の一致数: {overlap}人")
            print(f"一致率: {overlap/int(len(df) * treatment_ratio):.2%}")
            
            # 予測方策と真のITE上位方策の一致率
            policy_match = np.mean((p_e_ite[:, 1, 0] > 0) == (p_e_oracle_top[:, 1, 0] > 0))
            print(f"予測方策と真のITE上位方策の一致率: {policy_match:.2%}")
    else:
        p_e_oracle = None
        p_e_oracle_top = None
    
    return X, A, R, p_b, bandit_feedback, p_e_random, p_e_original, p_e_ite, p_e_oracle, p_e_oracle_top, df, true_ite

def evaluate_with_multiple_methods(reward, action, action_dist, pscore, estimated_rewards, clip_value=None):
    """
    複数のオフポリシー評価手法を使用して方策を評価する関数
    
    Parameters
    ----------
    reward : array-like, shape (n_samples,)
        報酬
    action : array-like, shape (n_samples,)
        行動
    action_dist : array-like, shape (n_samples, n_actions, len_list)
        評価する方策の行動分布
    pscore : array-like, shape (n_samples,)
        傾向スコア
    estimated_rewards : array-like, shape (n_samples, n_actions, len_list)
        回帰モデルによる予測報酬
    clip_value : float, optional
        傾向スコアのクリッピング値（Noneの場合はクリッピングしない）
        
    Returns
    -------
    results : dict
        各手法の評価結果
    """
    # 傾向スコアのクリッピング
    if clip_value is not None:
        clipped_pscore = np.maximum(pscore, clip_value)
    else:
        clipped_pscore = pscore
    
    # 評価手法の初期化
    dm = DirectMethod()
    ipw = InverseProbabilityWeighting()
    dr = DoublyRobust()
    switch_dr = SwitchDoublyRobust(tau=0.05)  # 切り替え閾値
    drs = DoublyRobustWithShrinkage(lambda_=0.5)  # 収縮パラメータ
    snips = SelfNormalizedInverseProbabilityWeighting()
    
    # 各手法での評価
    results = {
        "DM": dm.estimate_policy_value(
            reward=reward,
            action=action,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards
        ),
        "IPW": ipw.estimate_policy_value(
            reward=reward,
            action=action,
            action_dist=action_dist,
            pscore=clipped_pscore
        ),
        "DR": dr.estimate_policy_value(
            reward=reward,
            action=action,
            action_dist=action_dist,
            pscore=clipped_pscore,
            estimated_rewards_by_reg_model=estimated_rewards
        ),
        "Switch-DR": switch_dr.estimate_policy_value(
            reward=reward,
            action=action,
            action_dist=action_dist,
            pscore=clipped_pscore,
            estimated_rewards_by_reg_model=estimated_rewards
        ),
        "DRs": drs.estimate_policy_value(
            reward=reward,
            action=action,
            action_dist=action_dist,
            pscore=clipped_pscore,
            estimated_rewards_by_reg_model=estimated_rewards
        ),
        "SNIPS": snips.estimate_policy_value(
            reward=reward,
            action=action,
            action_dist=action_dist,
            pscore=clipped_pscore
        )
    }
    
    return results

def bootstrap_confidence_intervals(df, p_e, p_b, A, R, regression_model, n_bootstrap=1000, alpha=0.05, clip_value=None):
    """
    ブートストラップ法を使用して信頼区間を計算する関数
    
    Parameters
    ----------
    df : pandas.DataFrame
        特徴量を含むデータフレーム
    p_e : array-like, shape (n_samples, n_actions)
        評価する方策の行動分布
    p_b : array-like, shape (n_samples,)
        傾向スコア
    A : array-like, shape (n_samples,)
        行動
    R : array-like, shape (n_samples,)
        報酬
    regression_model : RegressionModel
        報酬関数を推定するための回帰モデル
    n_bootstrap : int, default=1000
        ブートストラップ回数
    alpha : float, default=0.05
        信頼区間の有意水準
    clip_value : float, optional
        傾向スコアのクリッピング値（Noneの場合はクリッピングしない）
        
    Returns
    -------
    ci_df : pandas.DataFrame
        信頼区間を含むデータフレーム
    bootstrap_results : dict
        ブートストラップ結果
    """
    n_samples = len(df)
    feature_names = ['age', 'homeownership']
    
    # 各手法の結果を格納するリスト
    bootstrap_results = {
        "DM": [], "IPW": [], "DR": [], "Switch-DR": [], 
        "DRs": [], "SNIPS": []
    }
    
    X = df[feature_names].values  # NumPy配列に変換
    
    for _ in tqdm(range(n_bootstrap), desc="ブートストラップ実行中"):
        # リサンプリング
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X[indices]
        A_boot = A[indices]
        R_boot = R[indices]
        p_b_boot = p_b[indices]
        p_e_boot = p_e[indices]
        
        # 回帰モデルを学習 - NumPy配列を渡す
        estimated_rewards = regression_model.fit_predict(
            context=X_boot,
            action=A_boot,
            reward=R_boot,
            n_folds=2,
            random_state=12345
        )
        
        # 各手法での推定
        # OBPのAPIに合わせて引数を調整
        action_dist = p_e_boot
        
        # 複数の手法で評価
        results = evaluate_with_multiple_methods(
            reward=R_boot,
            action=A_boot,
            action_dist=action_dist,
            pscore=p_b_boot,
            estimated_rewards=estimated_rewards,
            clip_value=clip_value
        )
        
        # 結果を格納
        for method, value in results.items():
            bootstrap_results[method].append(value)
    
    # 信頼区間の計算
    ci_results = {}
    for method, values in bootstrap_results.items():
        values = np.array(values)
        mean_value = np.mean(values)
        lower = np.percentile(values, alpha/2 * 100)
        upper = np.percentile(values, (1 - alpha/2) * 100)
        std_err = np.std(values)
        
        ci_results[method] = {
            "推定値": mean_value,
            "下限": lower,
            "上限": upper,
            "標準誤差": std_err
        }
    
    # データフレームに変換
    ci_df = pd.DataFrame(ci_results).T
    
    return ci_df, bootstrap_results

def parse_args():
    """コマンドライン引数をパースする
    
    Returns
    -------
    args : argparse.Namespace
        パースされた引数
    """
    parser = argparse.ArgumentParser(description='Doubly Robust評価スクリプト')
    parser.add_argument('--csv', type=str, required=True, help='CSVファイルのパス')
    parser.add_argument('--policy', type=str, default='ite', choices=['random', 'original', 'ite', 'oracle', 'oracle_top'], help='評価する方策')
    parser.add_argument('--clip', type=float, default=None, help='傾向スコアのクリッピング値')
    parser.add_argument('--bootstrap', action='store_true', help='ブートストラップ法による信頼区間を計算するかどうか')
    parser.add_argument('--n_bootstrap', type=int, default=1000, help='ブートストラップ法の反復回数')
    parser.add_argument('--model', type=str, default='lgbm', 
                        choices=['lgbm', 'rf', 'gbr', 'linear', 'ridge', 'lasso', 'mlp', 'gpr'], 
                        help='回帰モデルの種類')
    parser.add_argument('--learner', type=str, default='s_learner', 
                        choices=['s_learner', 'custom'], 
                        help='学習器の種類（s_learner: CausalMLのS-Learner, custom: 各行動ごとに別々のモデル）')
    parser.add_argument('--model_type', type=str, default='regressor', 
                        choices=['regressor', 'classifier'], 
                        help='モデルのタイプ（S-Learnerの場合のみ使用）')
    return parser.parse_args()

def main():
    args = parse_args()
    start_time = time.time()
    
    # データの読み込みと前処理
    X, A, R, p_b, bandit_feedback, p_e_random, p_e_original, p_e_ite, p_e_oracle, p_e_oracle_top, df, true_ite = load_and_preprocess_data(args.csv)
    
    # データサンプルの表示
    print("\n=== データサンプル ===")
    print("特徴量 (X)の最初の5行:")
    print(df[['age', 'homeownership']].head())
    print("\n処置 (A)の最初の5行:")
    print(A[:5])
    print("\n結果 (R)の最初の5行:")
    print(R[:5])
    print("\n傾向スコア (p_b)の最初の5行:")
    print(p_b[:5])
    
    # 回帰モデルの設定
    model_name = args.model
    learner_type = args.learner
    model_classifier_type = args.model_type
    
    if model_name == 'lgbm':
        base_model = lgbm.LGBMRegressor(n_estimators=100, learning_rate=0.05)
    elif model_name == 'rf':
        base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == 'gbr':
        base_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05)
    elif model_name == 'linear':
        base_model = LinearRegression()
    elif model_name == 'ridge':
        base_model = Ridge()
    elif model_name == 'lasso':
        base_model = Lasso()
    elif model_name == 'mlp':
        base_model = MLPRegressor(hidden_layer_sizes=(10, 10), random_state=42)
    elif model_name == 'gpr':
        kernel = ConstantKernel() * RBF()
        base_model = GaussianProcessRegressor(kernel=kernel, random_state=42)
    
    if learner_type == 's_learner':
        regression_model = SLearnerRegressionModel(
            base_model=base_model,
            model_type=model_classifier_type
        )
    elif learner_type == 'custom':
        regression_model = CustomRegressionModel(
            base_model=base_model
        )
    else:
        raise ValueError("Invalid learner type. Please choose 's_learner' or 'custom'.")
    
    # 回帰モデルを学習して予測報酬を取得
    print("\n回帰モデルを学習して予測報酬を取得しています...")
    estimated_rewards = regression_model.fit_predict(
        context=X,
        action=A,
        reward=R,
        n_folds=2,
        random_state=12345
    )
    
    # 回帰モデルの性能評価
    print("\n=== 回帰モデルの性能評価 ===")
    # 予測値と実際の値を比較
    action_0_indices = A == 0
    action_1_indices = A == 1
    
    # アクション0の回帰モデル性能
    y_true_0 = R[action_0_indices]
    y_pred_0 = estimated_rewards[action_0_indices, 0, 0]
    mse_0 = mean_squared_error(y_true_0, y_pred_0)
    mae_0 = mean_absolute_error(y_true_0, y_pred_0)
    r2_0 = r2_score(y_true_0, y_pred_0)
    print(f"アクション 0 の回帰モデル性能:")
    print(f"  サンプル数: {len(y_true_0)}")
    print(f"  MSE: {mse_0:.6f}")
    print(f"  MAE: {mae_0:.6f}")
    print(f"  R2スコア: {r2_0:.6f}")
    
    # アクション1の回帰モデル性能
    y_true_1 = R[action_1_indices]
    y_pred_1 = estimated_rewards[action_1_indices, 1, 0]
    mse_1 = mean_squared_error(y_true_1, y_pred_1)
    mae_1 = mean_absolute_error(y_true_1, y_pred_1)
    r2_1 = r2_score(y_true_1, y_pred_1)
    print(f"\nアクション 1 の回帰モデル性能:")
    print(f"  サンプル数: {len(y_true_1)}")
    print(f"  MSE: {mse_1:.6f}")
    print(f"  MAE: {mae_1:.6f}")
    print(f"  R2スコア: {r2_1:.6f}")
    
    # 全体の回帰モデル性能
    y_true = R
    y_pred = np.zeros_like(R)
    y_pred[action_0_indices] = estimated_rewards[action_0_indices, 0, 0]
    y_pred[action_1_indices] = estimated_rewards[action_1_indices, 1, 0]
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n全体の回帰モデル性能:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R2スコア: {r2:.6f}")
    
    print("\nOff-Policy評価を設定しています...")
    
    # 方策名のマッピング
    policy_name_mapping = {
        'random': 'ランダム方策',
        'original': '元のログ方策',
        'ite': 'ITE上位方策',
        'oracle': 'オラクル最適方策',
        'oracle_top': 'オラクル上位方策'
    }
    
    # 評価する方策の設定
    policies_to_evaluate = []
    if args.policy == 'random' or args.policy == 'all':
        policies_to_evaluate.append(('random', p_e_random))
    if args.policy == 'original' or args.policy == 'all':
        policies_to_evaluate.append(('original', p_e_original))
    if args.policy == 'ite' or args.policy == 'all':
        policies_to_evaluate.append(('ite', p_e_ite))
    if args.policy == 'oracle' or args.policy == 'all':
        policies_to_evaluate.append(('oracle', p_e_oracle))
    if args.policy == 'oracle_top' or args.policy == 'all':
        policies_to_evaluate.append(('oracle_top', p_e_oracle_top))
    
    # 各方策の評価
    all_results = {}
    for policy_name, policy in policies_to_evaluate:
        display_name = policy_name_mapping.get(policy_name, policy_name)
        print(f"\n=== {display_name}の評価 ===")
        
        # 複数の評価手法で評価
        results = evaluate_with_multiple_methods(
            reward=R,
            action=A,
            action_dist=policy,
            pscore=p_b,
            estimated_rewards=estimated_rewards,
            clip_value=args.clip
        )
        
        all_results[policy_name] = results
        
        # 結果を表示
        print("\n=== Off-Policy Evaluation Results ===")
        for method, value in results.items():
            print(f"{method}: {value:.4f}")
    
    # 全方策の比較表を作成
    if len(policies_to_evaluate) > 1:
        print("\n=== 全方策比較 ===")
        comparison_data = {}
        
        # 各方策の結果を収集
        for policy_name, _ in policies_to_evaluate:
            display_name = policy_name_mapping.get(policy_name, policy_name)
            comparison_data[display_name] = {method: value for method, value in all_results[policy_name].items()}
        
        # データフレームに変換して表示
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.round(4))
        
        # ランダム方策からの改善率を計算（ランダム方策が含まれている場合）
        if 'random' in [p[0] for p in policies_to_evaluate]:
            print("\n=== ランダム方策からの改善率 ===")
            improvement_data = {}
            
            for policy_name, _ in policies_to_evaluate:
                if policy_name != 'random':
                    display_name = policy_name_mapping.get(policy_name, policy_name)
                    improvement = {}
                    
                    for method in all_results['random'].keys():
                        random_value = all_results['random'][method]
                        policy_value = all_results[policy_name][method]
                        improvement[method] = (policy_value - random_value) / random_value * 100
                    
                    improvement_data[display_name] = improvement
            
            # データフレームに変換して表示
            improvement_df = pd.DataFrame(improvement_data)
            print(improvement_df.round(1).astype(str) + '%')
    
    # ブートストラップ分析
    if args.bootstrap:
        bootstrap_results = {}
        bootstrap_values = {}
        policies_to_bootstrap = [p[0] for p in policies_to_evaluate]
        
        for policy_key, policy in policies_to_evaluate:
            display_name = policy_name_mapping.get(policy_key, policy_key)
            print(f"\n--- {display_name}のブートストラップ分析 ---")
            
            # ブートストラップ信頼区間の計算
            bootstrap_result, bootstrap_value = bootstrap_confidence_intervals(
                df=df,
                p_e=policy,
                p_b=p_b,
                A=A,
                R=R,
                regression_model=regression_model,
                n_bootstrap=args.n_bootstrap,
                alpha=0.05,
                clip_value=args.clip
            )
            
            bootstrap_results[policy_key] = bootstrap_result
            bootstrap_values[policy_key] = bootstrap_value
            
            # 結果を表示
            print("\n=== ブートストラップ95%信頼区間 ===")
            print(bootstrap_result.round(4))
            
            # 実行時間の表示
            elapsed_time = time.time() - start_time
            print(f"実行時間: {elapsed_time:.2f}秒")
            
            # 方策間の比較（ランダム方策との比較）
            if policy_key != 'random' and 'random' in policies_to_bootstrap:
                print("\n--- ランダム方策との差の信頼区間 ---")
                
                # 各手法ごとの差分を計算
                diff_results = {}
                p_values = {}
                
                for method in bootstrap_result.index:
                    # 現在の方策とランダム方策の結果を取得
                    current_values = np.array(bootstrap_values[policy_key][method])
                    random_values = np.array(bootstrap_values['random'][method])
                    
                    # 差分を計算
                    diff = current_values - random_values
                    
                    # 差分の信頼区間
                    mean_diff = np.mean(diff)
                    lower_diff = np.percentile(diff, 2.5)
                    upper_diff = np.percentile(diff, 97.5)
                    
                    # p値の計算（0との差が有意かどうか）
                    p_value = np.mean(diff <= 0) if mean_diff > 0 else np.mean(diff >= 0)
                    p_value = min(p_value, 1 - p_value) * 2  # 両側検定
                    
                    diff_results[method] = {
                        "平均差": mean_diff,
                        "下限": lower_diff,
                        "上限": upper_diff
                    }
                    p_values[method] = p_value
                
                # 結果をデータフレームに変換して表示
                diff_df = pd.DataFrame(diff_results).T
                diff_df["p値"] = pd.Series(p_values)
                print(diff_df.round(4))

if __name__ == "__main__":
    main()
