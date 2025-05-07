"""
因果推論とマシンラーニングのためのエンドポイント
複数のスクリプト間で共有される機能を集約しています
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union

# 各モジュールをインポート
from data_utils import (
    set_plot_style,
    load_data,
    display_basic_stats,
    convert_to_bandit_feedback
)

from propensity_score import (
    compute_propensity_scores,
    compare_propensity_scores,
    estimate_ate_with_ipw
)

from ope_utils import (
    create_action_dist,
    run_ope,
    generate_estimated_rewards,
    generate_true_rewards,
    print_ope_results,
    create_policy_from_ite
)

from ate_calculator import (
    calculate_true_ite_sum,
    calculate_true_ite_mean,
    calculate_policy_statistics,
    calculate_ate,
    print_ate_results,
    calculate_top_n_ate,
    print_top_n_results
)

from data_preprocessing import (
    check_oracle_information,
    create_features,
    get_feature_sets,
    prepare_data
)


class CausalAnalysis:
    """
    因果分析のための統合クラス
    """
    
    def __init__(self, data_path: str = None, df: pd.DataFrame = None):
        """
        初期化
        
        Parameters
        ----------
        data_path : str, optional
            データファイルのパス, by default None
        df : pd.DataFrame, optional
            データフレーム（直接渡す場合）, by default None
        """
        self.df = None
        self.df_matched = None
        self.df_orig = None
        self.X = None
        self.treatment = None
        self.outcome = None
        self.oracle_info = {}
        self.feature_cols = []
        self.propensity_scores = {}
        self.bandit_feedback = None
        self.policies = {}
        self.policy_stats = {}
        self.ate_results = {}
        self.ope_results = {}
        
        # プロットスタイルの設定
        set_plot_style()
        
        # データの読み込み
        if data_path is not None:
            self.load_data(data_path)
        elif df is not None:
            self.df = df.copy()
            self.df_orig = df.copy()
    
    def load_data(self, data_path: str) -> None:
        """
        データを読み込む
        
        Parameters
        ----------
        data_path : str
            データファイルのパス
        """
        self.df = load_data(data_path)
        self.df_orig = self.df.copy()
        display_basic_stats(self.df)
    
    def preprocess_data(self, feature_level: str = 'original', additional_features: bool = True) -> None:
        """
        データを前処理する
        
        Parameters
        ----------
        feature_level : str, optional
            特徴量レベル, by default 'original'
        additional_features : bool, optional
            追加特徴量を生成するかどうか, by default True
        """
        # オラクル情報の確認
        self.df, self.oracle_info = check_oracle_information(self.df)
        
        # 特徴量の生成
        self.df = create_features(self.df, additional_features=additional_features)
        
        # 特徴量セットの取得
        feature_sets = get_feature_sets(feature_level)
        self.feature_cols = feature_sets
        
        # 特徴量、処置、アウトカムの抽出
        self.X = self.df[self.feature_cols]
        self.treatment = self.df['treatment']
        self.outcome = self.df['outcome']
    
    def compute_propensity_scores(self, methods: List[str] = ['logistic', 'lightgbm']) -> Dict:
        """
        傾向スコアを計算する
        
        Parameters
        ----------
        methods : List[str], optional
            計算方法のリスト, by default ['logistic', 'lightgbm']
            
        Returns
        -------
        Dict
            計算された傾向スコアの辞書
        """
        # オラクル傾向スコア（存在する場合）
        oracle_score = self.df['propensity_score'].values if 'propensity_score' in self.df.columns else None
        
        # 傾向スコアの計算と比較
        self.propensity_scores = compare_propensity_scores(self.X, self.treatment, oracle_score)
        
        return self.propensity_scores
    
    def estimate_ate(self) -> Dict:
        """
        ATEを推定する
        
        Returns
        -------
        Dict
            推定されたATEの辞書
        """
        # 傾向スコアによるATE推定
        print("\n===== 傾向スコアによるATE推定値 =====")
        
        # 真のATE（オラクル情報がある場合）
        true_ate = self.df['true_ite'].mean() if 'true_ite' in self.df.columns else None
        if true_ate is not None:
            print(f"真のATE (y1-y0の平均): {true_ate:.4f}")
        
        # 観測データからの素朴なATE計算
        observed_ate = self.df.groupby('treatment')['outcome'].mean().diff().iloc[-1]
        print(f"観測データからの素朴なATE推定値 (処置群vs対照群の平均の差): {observed_ate:.4f}")
        
        # 各傾向スコアでのIPW-ATE推定
        ate_estimates = {}
        for name, p_score in self.propensity_scores.items():
            ate_ipw = estimate_ate_with_ipw(self.outcome.values, self.treatment.values, p_score)
            print(f"{name}傾向スコアによるIPW-ATE推定値: {ate_ipw:.4f}")
            ate_estimates[name] = ate_ipw
        
        return ate_estimates
    
    def create_bandit_feedback(self) -> Dict:
        """
        BanditFeedback形式に変換する
        
        Returns
        -------
        Dict
            BanditFeedback形式の辞書
        """
        self.bandit_feedback = convert_to_bandit_feedback(self.df)
        
        # 傾向スコアが計算済みの場合は追加
        if 'lightgbm' in self.propensity_scores:
            self.bandit_feedback["pscore"] = self.propensity_scores['lightgbm']
        elif 'propensity_score' in self.df.columns:
            self.bandit_feedback["pscore"] = self.df['propensity_score'].values
        
        return self.bandit_feedback
    
    def create_policies(self, k: int = None) -> Dict:
        """
        評価ポリシーを作成する
        
        Parameters
        ----------
        k : int, optional
            処置するユーザー数, by default None（処置群と同数）
            
        Returns
        -------
        Dict
            ポリシー名とポリシー配列の辞書
        """
        # 処置するユーザー数（指定がなければ元の処置数と同じ）
        if k is None:
            k = self.df['treatment'].sum()
        
        # 3-1) ランダム抽出ポリシー
        np.random.seed(42)  # 再現性のため
        random_indices = np.random.choice(len(self.df), k, replace=False)
        random_is_selected = np.zeros(len(self.df), dtype=bool)
        random_is_selected[random_indices] = True
        random_policy = random_is_selected.astype(int)
        
        self.policies['random'] = random_policy
        
        # 3-2) 真のITE上位k件を選択するポリシー（理論的最適、オラクル情報がある場合）
        if 'true_ite' in self.df.columns:
            true_ite = self.df['true_ite'].values
            true_topk_indices = np.argsort(-true_ite)[:k]
            true_is_topk = np.zeros(len(true_ite), dtype=bool)
            true_is_topk[true_topk_indices] = True
            true_policy = true_is_topk.astype(int)
            
            self.policies['true_ite'] = true_policy
        
        # 3-3) アップリフトモデルによるポリシー（ITE予測値がある場合）
        if 'ite_pred' in self.df.columns:
            predicted_ite = self.df['ite_pred'].values
            model_topk_indices = np.argsort(-predicted_ite)[:k]
            model_is_topk = np.zeros(len(predicted_ite), dtype=bool)
            model_is_topk[model_topk_indices] = True
            model_policy = model_is_topk.astype(int)
            
            self.policies['model'] = model_policy
        
        return self.policies
    
    def calculate_policy_statistics(self) -> Dict:
        """
        各ポリシーの統計情報を計算する
        
        Returns
        -------
        Dict
            ポリシー名と統計情報の辞書
        """
        self.policy_stats = {}
        
        for name, policy in self.policies.items():
            self.policy_stats[name] = calculate_policy_statistics(self.df, policy, name)
        
        # 統計情報の表示
        print("\n===== ポリシー統計情報 =====")
        for name, stats in self.policy_stats.items():
            print(f"\n【{name}ポリシー】")
            print(f"  処置数: {stats['count']} 件 (全体の{stats['ratio']:.2%})")
            
            if 'true_ite_mean' in stats:
                print(f"  真のITEの平均値: {stats['true_ite_mean']:.4f}")
                print(f"  真のITEが正の割合: {stats['true_ite_positive_ratio']:.2%}")
            
            if 'outcome_mean' in stats:
                print(f"  アウトカムの平均値: {stats['outcome_mean']:.4f}")
                print(f"  アウトカムの合計: {stats['outcome_sum']:.2f}")
            
            if 'pred_ite_mean' in stats:
                print(f"  予測ITEの平均値: {stats['pred_ite_mean']:.4f}")
                print(f"  予測ITEの合計: {stats['pred_ite_sum']:.2f}")
        
        return self.policy_stats
    
    def calculate_ate_comparison(self) -> Dict:
        """
        全体のATEと各ポリシーのATEを計算する
        
        Returns
        -------
        Dict
            ATE計算結果の辞書
        """
        self.ate_results = calculate_ate(self.df, self.policies)
        print_ate_results(self.ate_results)
        
        return self.ate_results
    
    def run_ope_evaluation(self, model = None) -> Dict:
        """
        オフポリシー評価を実行する
        
        Parameters
        ----------
        model : object, optional
            学習済みモデル（predict メソッドを持つ）, by default None
            
        Returns
        -------
        Dict
            OPE結果の辞書
        """
        # BanditFeedbackの作成（まだ作成されていない場合）
        if self.bandit_feedback is None:
            self.create_bandit_feedback()
        
        # 真の期待報酬（オラクル情報がある場合）
        if 'base_prob' in self.df.columns and 'treat_prob' in self.df.columns:
            true_rewards = generate_true_rewards(self.df)
        else:
            true_rewards = None
        
        # 推定期待報酬（モデルがある場合）
        if model is not None:
            estimated_rewards = generate_estimated_rewards(self.df, model, self.feature_cols)
        else:
            estimated_rewards = true_rewards
        
        # 各ポリシーのaction_distを作成
        action_dists = {}
        for name, policy in self.policies.items():
            action_dists[name] = create_action_dist(policy)
        
        # OPE実行
        self.ope_results = {}
        for name, action_dist in action_dists.items():
            # 真の報酬でのOPE（オラクル評価）
            if true_rewards is not None:
                self.ope_results[f"{name}_true"] = run_ope(self.bandit_feedback, true_rewards, action_dist)
            
            # 推定報酬でのOPE（実際の評価）
            if estimated_rewards is not None:
                self.ope_results[f"{name}_est"] = run_ope(self.bandit_feedback, estimated_rewards, action_dist)
        
        # OPE結果の表示
        for name, policy in self.policies.items():
            # 真のITEの平均（存在する場合）
            true_ite_mean = None
            if 'true_ite' in self.df.columns:
                true_ite = self.df['true_ite'].values
                true_ite_mean = calculate_true_ite_mean(true_ite, policy)
            
            # 真の報酬でのOPE結果
            if f"{name}_true" in self.ope_results:
                print_ope_results(f"{name}ポリシー（真の報酬）", self.ope_results[f"{name}_true"], true_ite_mean)
            
            # 推定報酬でのOPE結果
            if f"{name}_est" in self.ope_results:
                print_ope_results(f"{name}ポリシー（推定報酬）", self.ope_results[f"{name}_est"], true_ite_mean)
        
        return self.ope_results
    
    def evaluate_top_n(self, ite_pred: np.ndarray = None, ns: List[int] = [500, 1000, 2000]) -> Dict:
        """
        予測ITE上位N人のATEを計算する
        
        Parameters
        ----------
        ite_pred : np.ndarray, optional
            ITE予測値の配列, by default None（dfのite_predを使用）
        ns : List[int], optional
            評価するN値のリスト, by default [500, 1000, 2000]
            
        Returns
        -------
        Dict
            各N値に対する結果の辞書
        """
        # ITE予測値（指定がなければdfから取得）
        if ite_pred is None and 'ite_pred' in self.df.columns:
            ite_pred = self.df['ite_pred'].values
        
        if ite_pred is None:
            print("ITE予測値がありません")
            return {}
        
        # 真のITE（存在する場合）
        true_ite = self.df['true_ite'].values if 'true_ite' in self.df.columns else None
        
        # 上位N人のATE計算
        top_n_results = calculate_top_n_ate(
            ite_pred, 
            self.df['treatment'].values, 
            self.df['outcome'].values, 
            ns, 
            true_ite
        )
        
        # 結果の表示
        original_treated_count = self.df['treatment'].sum()
        print_top_n_results(top_n_results, original_treated_count)
        
        return top_n_results


# 簡単な使用例
if __name__ == "__main__":
    # 分析インスタンスの作成
    #analysis = CausalAnalysis(data_path="df.csv")
    analysis = CausalAnalysis(data_path="df_balanced_group.csv")
    
    # データの前処理
    print("=== preprocess_data ===")
    analysis.preprocess_data(feature_level='original')
    
    # 傾向スコアの計算
    print("=== compute_propensity_scores ===")
    analysis.compute_propensity_scores()
    
    # ATEの推定
    print("=== estimate_ate ===")
    analysis.estimate_ate()
    
    # ポリシーの作成
    print("=== create_policies ===")
    analysis.create_policies()
    
    # ポリシーの統計情報
    print("=== calculate_policy_statistics ===")
    analysis.calculate_policy_statistics()
    
    # ATEの比較
    print("=== calculate_ate_comparison ===")
    analysis.calculate_ate_comparison()
    
    # OPE評価
    print("=== run_ope_evaluation ===")
    analysis.run_ope_evaluation()
    
    # 上位N人の評価（ITE予測値がある場合）
    print("=== evaluate_top_n ===")
    if 'ite_pred' in analysis.df.columns:
        analysis.evaluate_top_n()
