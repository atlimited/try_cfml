"""
Causal Inference and Machine Learning (CFML) Package
"""

from typing import Optional, Dict, List
import pandas as pd # CausalAnalysis クラスで使用するため
import numpy as np  # CausalAnalysis クラスで使用するため
import os # OPE結果の保存用

# OBP (Open Bandit Pipeline) から必要なモジュールをインポート
from obp.ope import OffPolicyEvaluation, DirectMethod, InverseProbabilityWeighting, DoublyRobust

# --- Imports from submodules ---
from .data import (
    preprocess_data,
    get_feature_columns,
    convert_to_bandit_feedback
)
from .models import (
    get_uplift_model,
    train_uplift_model,
    train_custom_s_learner,
    compute_propensity_scores,
    compare_propensity_scores
)
from .evaluation import (
    evaluate_uplift_model,
    create_policy,
    create_action_dist,
    run_ope_evaluation,
    calculate_policy_statistics,
    calculate_ate # 実装は後ほど
)
from .visualization import (
    set_plot_style,
    plot_uplift_distribution,
    plot_uplift_curves,
    plot_propensity_scores,
    plot_policy_comparison,
    plot_ope_results
)
# from .utils import (
#     # 今後追加するユーティリティ関数
# )

# --- Main CausalAnalysis Class ---
class CausalAnalysis:
    """
    因果推論分析を実行するためのメインクラス（リファクタ版）
    """
    def __init__(self,
                 data_df: pd.DataFrame,
                 treatment_col: str,
                 outcome_col: str,
                 feature_level: str = 'original',
                 propensity_score_col: str = 'pscore'):
        """
        コンストラクタ
        """
        self.raw_df: pd.DataFrame = data_df.copy()
        self.treatment_col: str = treatment_col
        self.outcome_col: str = outcome_col
        self.feature_level: str = feature_level
        self.propensity_score_col: str = propensity_score_col

        # 前処理
        self.df, oracle_info, self.feature_cols = preprocess_data(
            self.raw_df.copy(), feature_level=feature_level
        )
        # オラクル情報を属性化
        self.has_oracle: bool = oracle_info.get('has_oracle', False)
        self.true_ite: Optional[np.ndarray] = oracle_info.get('true_ite', None)
        self.true_ite_col: Optional[str] = 'true_ite' if 'true_ite' in self.df.columns else None
        self.y0: Optional[np.ndarray] = oracle_info.get('y0', None)
        self.y1: Optional[np.ndarray] = oracle_info.get('y1', None)

        # モデル・予測値
        self.uplift_model: Optional[object] = None
        self.ite_pred: Optional[np.ndarray] = None
        self.mu0_pred: Optional[np.ndarray] = None
        self.mu1_pred: Optional[np.ndarray] = None
        self.propensity_scores: Optional[np.ndarray] = None
        self.bandit_feedback: Optional[pd.DataFrame] = None

        print("CausalAnalysis initialized.")
        print(f"  Data shape: {self.df.shape}")
        print(f"  Feature columns: {self.feature_cols}")
        if self.has_oracle:
            print("  Oracle information (true ITE, y0, y1) is available.")

    def estimate_propensity_scores(self, method: str = 'logistic') -> np.ndarray:
        """
        傾向スコアを計算し、self.propensity_scores(np.ndarray)・self.df・self.bandit_feedback(DataFrame)に一貫して格納
        """
        print(f"Estimating propensity scores using {method} method...")
        self.propensity_scores: np.ndarray = compute_propensity_scores(
            X=self.df[self.feature_cols],
            treatment=self.df[self.treatment_col].values,
            method=method
        )
        self.df[self.propensity_score_col] = self.propensity_scores
        self.bandit_feedback: pd.DataFrame = self.df.copy()  # DataFrame型で保持
        print(f"  Propensity scores estimated. Shape: {self.propensity_scores.shape}")
        return self.propensity_scores


    def train_model(self, model_type: str = 's_learner', **kwargs):
        """
        モデルを学習し、ITE予測値等を属性に格納（責務分離設計）
        """
        # デフォルトはcausalmlのS-Learnerを使用
        if model_type == 's_learner':
            model, ite_pred, mu0_pred, mu1_pred = train_uplift_model(
                X=self.df[self.feature_cols],
                treatment=self.df[self.treatment_col].values,
                outcome=self.df[self.outcome_col].values,
                model_type='s_learner',
                propensity_scores=self.propensity_scores,
                random_state=kwargs.get('random_state', 42)
            )
            self.uplift_model = model
            self.ite_pred = ite_pred
            self.mu0_pred = mu0_pred
            self.mu1_pred = mu1_pred
        elif model_type == 'custom_s_learner':
            ite_pred, mu0_pred, mu1_pred = train_custom_s_learner(
                X=self.df[self.feature_cols].values,
                t=self.df[self.treatment_col].values,
                y=self.df[self.outcome_col].values,
                model_type='lr',
                random_state=kwargs.get('random_state', 42)
            )
            self.ite_pred = ite_pred
            self.mu0_pred = mu0_pred
            self.mu1_pred = mu1_pred
            self.uplift_model = None
        else:
            model, ite_pred, mu0_pred, mu1_pred = train_uplift_model(
                X=self.df[self.feature_cols],
                treatment=self.df[self.treatment_col].values,
                outcome=self.df[self.outcome_col].values,
                model_type=model_type,
                propensity_scores=self.propensity_scores,
                random_state=kwargs.get('random_state', 42)
            )
            self.uplift_model = model
            self.ite_pred = ite_pred
            self.mu0_pred = mu0_pred
            self.mu1_pred = mu1_pred
        print(f"  Model trained. Predicted ITE shape: {self.ite_pred.shape}")
        if self.mu0_pred is not None:
            print(f"  Predicted mu0 shape: {self.mu0_pred.shape}")
        if self.mu1_pred is not None:
            print(f"  Predicted mu1 shape: {self.mu1_pred.shape}")
        return self.uplift_model, self.ite_pred

    def evaluate_model(self) -> Optional[Dict]:
        """
        学習済みモデルを評価する
        """
        if self.ite_pred is None:
            print("Error: Model not trained yet. Call train_model() first.")
            
        
        print("Evaluating uplift model...")
        true_ite = self.true_ite
        metrics = evaluate_uplift_model(self.ite_pred, true_ite=true_ite)
        
        print("  Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}")
        return metrics

    def run_off_policy_evaluation(
        self,
        *,
        bandit_feedback: pd.DataFrame, 
        policies_to_evaluate: Dict[str, np.ndarray], 
        reward_col: str = 'outcome', 
        n_actions: int = 2
    ) -> Optional[pd.DataFrame]:
        """
        オフポリシー評価 (OPE) を実行します。
        obpライブラリを使用します。

        Args:
            bandit_feedback (pd.DataFrame): OPEに必要な情報を含むDataFrame。
                                         必須カラム: 'action', reward_col, 'pscore'
                                         オプションカラム: 'position' (スレート/推薦の場合)
            policies_to_evaluate (Dict[str, np.ndarray]): 評価するポリシー名→action配列の辞書
            reward_col (str): 報酬カラム名
            n_actions (int): アクション数

        Returns:
            Optional[pd.DataFrame]: OPE結果のDataFrame
        """
        print("Running Off-Policy Evaluation (OPE)...")

        # bandit_feedbackのactionカラムを整数型に変換（obpの仕様に合わせる）
        # また、カラム名が 'action' であることを確認
        # 'treatment' カラムが存在し、'action' が存在しない場合はリネームを試みる
        current_action_col = ''
        if 'action' in bandit_feedback.columns:
            current_action_col = 'action'
        elif self.treatment_col in bandit_feedback.columns:
            bandit_feedback = bandit_feedback.rename(columns={self.treatment_col: 'action'})
            current_action_col = 'action'
        else:
            print(f"Error: Neither 'action' nor '{self.treatment_col}' column found in bandit_feedback.")
            
        
        # pscore カラムの確認 ( self.propensity_score_col が 'pscore' でない可能性も考慮)
        current_pscore_col = ''
        if 'pscore' in bandit_feedback.columns:
            current_pscore_col = 'pscore'
        elif self.propensity_score_col in bandit_feedback.columns:
            bandit_feedback = bandit_feedback.rename(columns={self.propensity_score_col: 'pscore'})
            current_pscore_col = 'pscore'
        else:
            print(f"Error: Neither 'pscore' nor '{self.propensity_score_col}' column found in bandit_feedback.")
            

        if reward_col not in bandit_feedback.columns:
            print(f"Error: Reward column '{reward_col}' not found in bandit_feedback.")
            

        # OBPの仕様に合わせてカラム名を調整
        obp_bandit_feedback = {
            'action': bandit_feedback[current_action_col].astype(int).values,
            'reward': bandit_feedback[reward_col].values,
            'pscore': bandit_feedback[current_pscore_col].values,
            'n_actions': n_actions,
            'n_rounds': len(bandit_feedback),
            'position': np.zeros(len(bandit_feedback), dtype=int)  # ダミーの position を追加
        }

        # estimated_rewards_by_reg_model の準備
        n_rounds = obp_bandit_feedback['n_rounds']
        n_actions = obp_bandit_feedback['n_actions']
        estimated_rewards_by_reg_model = None
        if self.mu0_pred is not None and self.mu1_pred is not None:
            if len(self.mu0_pred) == n_rounds and len(self.mu1_pred) == n_rounds:
                mu0_reshaped = self.mu0_pred.flatten().reshape(-1, 1)
                mu1_reshaped = self.mu1_pred.flatten().reshape(-1, 1)
                # (n_rounds, 2)
                estimated_rewards_by_reg_model = np.hstack([mu0_reshaped, mu1_reshaped])
            else:
                # 長さ不一致の場合はゼロ埋め
                estimated_rewards_by_reg_model = np.zeros((n_rounds, n_actions))
        else:
            # 予測値がない場合もゼロ埋め
            estimated_rewards_by_reg_model = np.zeros((n_rounds, n_actions))
        # 必ず(n_rounds, n_actions, 1)の3次元で渡す
        estimated_rewards_by_reg_model = estimated_rewards_by_reg_model.reshape(n_rounds, n_actions, 1)
        obp_bandit_feedback['estimated_rewards_by_reg_model'] = estimated_rewards_by_reg_model

        ope = OffPolicyEvaluation(
            bandit_feedback=obp_bandit_feedback,
            ope_estimators=[
                DirectMethod(), 
                InverseProbabilityWeighting(), 
                DoublyRobust()
            ]
        )

        results_df_list = []
        for policy_name, action_dist_2d in policies_to_evaluate.items():  # 変数名を action_dist_2d に変更
            if action_dist_2d.shape != (obp_bandit_feedback['n_rounds'], obp_bandit_feedback['n_actions']):
                print(f"Skipping policy '{policy_name}': action_dist shape mismatch. Expected ({obp_bandit_feedback['n_rounds']}, {obp_bandit_feedback['n_actions']}), got {action_dist_2d.shape}")
                continue
            # 正しいインデントで2D/3D対応
            if action_dist_2d.ndim == 2:
                action_dist_3d = action_dist_2d[:, :, np.newaxis]
            elif action_dist_2d.ndim == 3:
                action_dist_3d = action_dist_2d
            else:
                raise ValueError("action_dist must be 2D or 3D")

            estimated_policy_value = ope.estimate_policy_values(
                action_dist=action_dist_3d,
                estimated_rewards_by_reg_model=obp_bandit_feedback.get('estimated_rewards_by_reg_model')
            )
            
            # --- ポリシーごとの真のATE（true ITEの平均）を計算 ---
            true_ate_for_policy = np.nan
            if self.true_ite_col and self.true_ite_col in self.df.columns:
                treated_by_policy_mask = action_dist_2d[:, 1] == 1.0
                if np.any(treated_by_policy_mask):
                    true_ate_for_policy = self.df.loc[treated_by_policy_mask, self.true_ite_col].mean()
                else:
                    true_ate_for_policy = 0.0
            # --- 全体のATE（true ITEの全体平均）を計算 ---
            overall_true_ate = np.nan
            if self.true_ite_col and self.true_ite_col in self.df.columns:
                overall_true_ate = self.df[self.true_ite_col].mean()

            for estimator_name, value in estimated_policy_value.items():
                results_df_list.append({
                    'Policy': policy_name,
                    'Estimator': estimator_name,
                    'Estimated_Value': value,
                    'True_Policy_ATE': true_ate_for_policy,
                    'Overall_True_ATE': overall_true_ate
                })

        if not results_df_list:
            print("No OPE results to display.")

        ope_results_df = pd.DataFrame(results_df_list)
        print("\nOff-Policy Evaluation Results:")
        print(ope_results_df)

        try:
            output_dir = 'output'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            fig_path = os.path.join(output_dir, 'ope_results_comparison.png')

            # DataFrameからDict[str, Dict[str, float]]形式に変換
            ope_results_dict = {}
            for _, row in ope_results_df.iterrows():
                policy = row['Policy']
                estimator = row['Estimator']
                value = row['Estimated_Value']
                if policy not in ope_results_dict:
                    ope_results_dict[policy] = {}
                ope_results_dict[policy][estimator] = value
            true_values = ope_results_df.groupby('Policy')['True_Policy_ATE'].first().to_dict()
            plot_ope_results(ope_results_dict, true_values=true_values, fig_path=fig_path)
            print(f"OPE results visualization saved to {fig_path}")
        except Exception as e:
            print(f"Could not visualize OPE results: {e}")
        
        return ope_results_df

    def visualize_uplift_distribution(self):
        """
        アップリフト分布を可視化
        """
        plot_uplift_distribution(self.ite_pred, true_ite=self.true_ite)

    def visualize_uplift_curves(self):
        """
        アップリフトカーブを可視化する
        """
        if self.ite_pred is None:
            print("Error: Model not trained yet. Call train_model() first.")
            return
        plot_uplift_curves(
            treatment=self.df['treatment'].values,
            outcome=self.df['outcome'].values,
            ite_pred=self.ite_pred
        )

    # --- 他のメソッドも同様に追加 ---

__all__ = [
    "CausalAnalysis",
    "preprocess_data",
    "get_feature_columns",
    "convert_to_bandit_feedback",
    "get_uplift_model",
    "train_uplift_model",
    "compute_propensity_scores",
    "compare_propensity_scores",
    "evaluate_uplift_model",
    "create_policy",
    "create_action_dist",
    "run_ope_evaluation",
    "calculate_policy_statistics",
    "calculate_ate",
    "set_plot_style",
    "plot_uplift_distribution",
    "plot_uplift_curves",
    "plot_propensity_scores",
    "plot_policy_comparison",
    "plot_ope_results",
]