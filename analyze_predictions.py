"""
S-Learnerのアップリフト予測の詳細分析
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from causal_models import train_s_learner

def analyze_s_learner_predictions(learner_type='classification'):
    """S-Learnerの予測結果を詳細に分析する"""
    # データ読み込み
    df = pd.read_csv('df.csv')
    print("データを読み込みました")
    
    # 特徴量、処置、アウトカム
    X = df[['age_scaled', 'homeownership_scaled']]
    treatment = df['treatment'].values
    outcome = df['outcome'].values
    
    # 真のITE
    if 'true_ite' in df.columns:
        true_ite = df['true_ite'].values
    else:
        true_ite = df['y1'].values - df['y0'].values
    
    # S-Learnerをトレーニング＆予測
    print(f"S-Learner（{learner_type}）をトレーニング中...")
    model, predictions = train_s_learner(X, treatment, outcome, learner_type)
    
    # 形状変換が必要な場合
    if hasattr(predictions, 'shape') and len(predictions.shape) > 1:
        print(f"ITE予測値の形状を変換: {predictions.shape} -> ({predictions.size},)")
        predictions = predictions.flatten()
    
    # 予測結果をデータフレームとして出力
    result_df = pd.DataFrame()
    result_df['age'] = df['age']
    result_df['homeownership'] = df['homeownership']
    result_df['treatment'] = treatment
    result_df['outcome'] = outcome
    result_df['true_ite'] = true_ite
    result_df['predicted_ite'] = predictions
    
    # グループ統計（response_groupがある場合）
    if 'response_group' in df.columns:
        result_df['response_group'] = df['response_group']
        print("\n===== レスポンスグループ別のITE予測統計 =====")
        for group, group_df in result_df.groupby('response_group'):
            print(f"グループ {group}:")
            print(f"  サンプル数: {len(group_df)}")
            print(f"  予測ITE平均: {group_df['predicted_ite'].mean():.4f}")
            print(f"  真のITE平均: {group_df['true_ite'].mean():.4f}")
            print(f"  相関係数: {np.corrcoef(group_df['predicted_ite'], group_df['true_ite'])[0,1]:.4f}")
            print(f"  予測ITE標準偏差: {group_df['predicted_ite'].std():.4f}")
            print()
    
    # 上位10件と下位10件の結果を表示
    print("\n===== 最も効果が高いと予測された上位10件 =====")
    print(result_df.sort_values('predicted_ite', ascending=False).head(10))
    
    print("\n===== 最も効果が低いと予測された下位10件 =====")
    print(result_df.sort_values('predicted_ite', ascending=True).head(10))
    
    # ITEの統計情報を表示
    print("\n===== ITE予測の統計情報 =====")
    print(f"最小値: {predictions.min():.4f}")
    print(f"最大値: {predictions.max():.4f}")
    print(f"平均値: {predictions.mean():.4f}")
    print(f"標準偏差: {predictions.std():.4f}")
    print(f"中央値: {np.median(predictions):.4f}")
    
    # 予測値の分布情報
    print("\n===== ITE予測値のパーセンタイル分布 =====")
    percentiles = [0, 10, 25, 50, 75, 90, 100]
    for p in percentiles:
        print(f"{p}パーセンタイル: {np.percentile(predictions, p):.4f}")
    
    # 予測ITEと真のITEの相関を表示
    print(f"\n予測ITEと真のITEの相関係数: {np.corrcoef(predictions, true_ite)[0,1]:.4f}")
    
    # 全データをCSVファイルに出力
    result_file = f's_learner_{learner_type}_uplift_predictions.csv'
    result_df.to_csv(result_file, index=False)
    print(f"\n予測結果の詳細は {result_file} に保存されました")
    
    return result_df


if __name__ == "__main__":
    # 分類モデルでS-Learnerを分析
    print("\n===== S-Learner分類モデルの詳細分析 =====")
    result_df = analyze_s_learner_predictions(learner_type='classification')
    
    # 回帰モデルでS-Learnerを分析
    print("\n===== S-Learner回帰モデルの詳細分析 =====")
    result_df = analyze_s_learner_predictions(learner_type='regression')
