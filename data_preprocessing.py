"""
因果推論のためのデータ前処理機能
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def check_oracle_information(df):
    """データフレームに潜在的な結果変数（オラクル情報）があるか確認"""
    oracle_info = {}
    
    # 一般的なオラクル列名
    oracle_cols = ['y0', 'y1', 'ite', 'true_effect', 'true_ite']
    
    for col in oracle_cols:
        if col in df.columns:
            oracle_info[col] = True
            print(f"オラクル情報あり: {col}")
            
            # y0とy1がある場合、true_iteを計算
            if col in ['y0', 'y1'] and 'y0' in df.columns and 'y1' in df.columns and 'true_ite' not in df.columns:
                df['true_ite'] = df['y1'] - df['y0']
                print("真のITE計算: y1 - y0")
            elif col in ['ite', 'true_effect', 'true_ite']:
                df['true_ite'] = df[col]
                print(f"真のITE: {col}を使用")
        else:
            oracle_info[col] = False
    
    # 真のATEを計算（オラクル情報がある場合）
    if 'true_ite' in df.columns:
        oracle_info['true_ate'] = df['true_ite'].mean()
    
    return df, oracle_info

def create_features(df, additional_features=True):
    """特徴量の前処理と生成"""
    # 基本的なスケーリング
    scaler = StandardScaler()
    if 'age' in df.columns:
        df['age_scaled'] = scaler.fit_transform(df[['age']])
    
    if 'homeownership' in df.columns:
        df['homeownership_scaled'] = df['homeownership']
    
    # 追加の特徴量（オプション）
    if additional_features:
        # 年齢の閾値特徴量
        if 'age' in df.columns:
            df['age_over_60'] = (df['age'] > 60).astype(int)
            df['age_30_to_60'] = ((df['age'] > 30) & (df['age'] <= 60)).astype(int)
            df['age_over_40'] = (df['age'] > 40).astype(int)
        
        # データ生成で使用された潜在変数の近似
        if 'age' in df.columns and 'homeownership' in df.columns:
            df['marketing_receptivity'] = 0.01 * df['age'] + 0.5 * df['homeownership']
            df['price_sensitivity'] = -0.01 * df['age'] + 0.3 * (1 - df['homeownership'])
        
            # グループ分類条件の近似
            df['group1_score'] = df['age_over_60'] * df['marketing_receptivity']
            df['group2_score'] = df['age_30_to_60'] * df['marketing_receptivity']
            df['group3_score'] = df['age_over_40'] * df['price_sensitivity']
        
            # 非線形特徴量と交互作用
            df['age_squared'] = df['age_scaled'] ** 2
            df['age_home'] = df['age_scaled'] * df['homeownership_scaled']
            df['marketing_price_interaction'] = df['marketing_receptivity'] * df['price_sensitivity']
    
    return df

def get_feature_sets(feature_level='minimal'):
    """特徴量セットの定義を取得"""
    feature_sets = {
        'minimal': [
            'age_scaled',
            'homeownership_scaled'
        ],
        'standard': [
            'age_scaled',
            'homeownership_scaled',
            'age_squared',
            'age_home'
        ],
        'advanced': [
            'age_scaled',
            'homeownership_scaled',
            'age_squared',
            'age_home',
            'age_over_60',
            'age_30_to_60',
            'age_over_40'
        ],
        'full': [
            'age_scaled',
            'homeownership_scaled',
            'age_squared',
            'age_home',
            'age_over_60',
            'age_30_to_60',
            'age_over_40',
            'marketing_receptivity',
            'price_sensitivity',
            'group1_score',
            'group2_score',
            'group3_score',
            'marketing_price_interaction'
        ]
    }
    
    return feature_sets.get(feature_level, feature_sets['minimal'])

def prepare_data(df, feature_level='minimal', check_oracle=True):
    """データの準備（前処理、特徴量生成、オラクル情報確認）"""
    # ステップ1: オラクル情報の確認（必要な場合）
    if check_oracle:
        df, oracle_info = check_oracle_information(df)
    else:
        oracle_info = None
    
    # ステップ2: 特徴量の生成
    df = create_features(df)
    
    # ステップ3: 使用する特徴量の選択
    feature_cols = get_feature_sets(feature_level)
    
    # ステップ4: データの準備
    X = df[feature_cols]
    
    if 'treatment' in df.columns:
        treatment = df['treatment']
    else:
        treatment = None
        print("警告: 処置変数が見つかりません")
    
    if 'outcome' in df.columns:
        outcome = df['outcome']
    else:
        outcome = None
        print("警告: アウトカム変数が見つかりません")
    
    return df, X, treatment, outcome, oracle_info, feature_cols
