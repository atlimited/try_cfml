"""
データ読み込みと基本統計情報表示のための関数
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


def set_plot_style():
    """
    プロット用のスタイル設定を行います
    """
    # 日本語フォントの設定（macOS向け）
    plt.rcParams['font.family'] = 'Hiragino Sans GB'
    # フォールバックとしてシステムのデフォルトフォントを使用
    mpl.rcParams['axes.unicode_minus'] = False  # マイナス記号を正しく表示


def load_data(file_path: str) -> pd.DataFrame:
    """
    指定されたパスからデータを読み込みます
    
    Parameters
    ----------
    file_path : str
        データファイルのパス
        
    Returns
    -------
    pd.DataFrame
        読み込まれたデータフレーム
    """
    return pd.read_csv(file_path)


def display_basic_stats(df: pd.DataFrame, title: str = "データの基本統計情報") -> None:
    """
    データフレームの基本統計情報を表示します
    
    Parameters
    ----------
    df : pd.DataFrame
        統計情報を表示するデータフレーム
    title : str, optional
        表示するタイトル, by default "データの基本統計情報"
    """
    print(f"===== {title} =====")
    print(f"データの総数: {len(df)}件")
    print(f"トリートメント総数: {df['treatment'].sum()}件 (処置率: {df['treatment'].sum()/len(df):.2%})")
    
    # アウトカム情報
    print(f"アウトカム総数: {df['outcome'].sum():.2f} (平均: {df['outcome'].mean():.4f})")
    
    # オラクル情報（存在する場合）
    if 'y0' in df.columns and 'y1' in df.columns:
        print(f"非処置時の潜在的アウトカム総数: {df['y0'].sum():.2f} (平均: {df['y0'].mean():.4f})")
        print(f"処置時の潜在的アウトカム総数: {df['y1'].sum():.2f} (平均: {df['y1'].mean():.4f})")
        print(f"真のITE総数: {(df['y1'] - df['y0']).sum():.2f} (平均: {(df['y1'] - df['y0']).mean():.4f})")
    
    # 確率情報（存在する場合）
    if 'base_prob' in df.columns and 'treat_prob' in df.columns:
        print(f"真のITE総数(実数): {(df['treat_prob'] - df['base_prob']).sum():.2f} (平均: {(df['treat_prob'] - df['base_prob']).mean():.4f})")


def convert_to_bandit_feedback(df: pd.DataFrame) -> dict:
    """
    データフレームをOBPのBanditFeedback形式に変換します
    
    Parameters
    ----------
    df : pd.DataFrame
        変換するデータフレーム
        
    Returns
    -------
    dict
        BanditFeedback形式の辞書
    """
    bandit_feedback = {
        "n_rounds": len(df),
        "n_actions": 2,
        "action": df["treatment"].values,
        "reward": df["outcome"].values,
        "position": df.get("position", np.zeros(len(df), dtype=int)),  # 位置情報（なければ0で埋める）
        "context": df[["age", "homeownership"]].values,  # コンテキスト情報
    }
    
    # 傾向スコアが存在する場合は追加
    if "propensity_score" in df.columns:
        bandit_feedback["pscore"] = df["propensity_score"].values
    
    return bandit_feedback
