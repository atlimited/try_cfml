import numpy as np
import pandas as pd

def generate_simulation(N=1000, n_treated=None, seed=0, group_ratios=None):
    """
    Generate simulation data with controllable number of treated individuals.
    
    Parameters:
    - N: total number of samples
    - n_treated: desired number of treated samples. If None, treatment assigned by propensity.
    - seed: random seed for reproducibility
    - group_ratios: Optional, target proportions for groups. Default is adjusted based on features.
    
    Returns:
    - df: pandas DataFrame with columns:
        age, homeownership, base_prob, treat_prob, propensity_score,
        treatment, conversion_prob, outcome, y0, y1, response_group
        
    Response Groups:
    1: 常にoutcome=1 (y0=1, y1=1) - treatmentに関わらず常に成約
    2: treatmentがあるとoutcome=1 (y0=0, y1=1) - treatmentで成約
    3: treatmentがあるとoutcome=0 (y0=1, y1=0) - treatmentで離脱
    4: 常にoutcome=0 (y0=0, y1=0) - treatmentに関わらず成約しない
    """
    np.random.seed(seed)
    
    # 1. 特徴量の生成
    age = np.random.randint(20, 81, size=N)
    homeown = np.random.binomial(1, 0.6, size=N)
    
    # 2. 特徴量に基づいたグループ分け
    # 特徴量から反応グループへのマッピングを作成
    # 年齢と持ち家状況から反応グループを決定
    
    # グループを決定するための追加特徴量を生成
    # これは特徴量からグループを分ける境界のようなもの
    # 例えば、marketing_receptivityは顧客のマーケティング受容性を表す隠れ変数
    np.random.seed(seed + 10)
    marketing_receptivity = 0.01 * age + 0.5 * homeown + np.random.normal(0, 1, size=N)
    price_sensitivity = -0.01 * age + 0.3 * (1 - homeown) + np.random.normal(0, 1, size=N)
    
    # 反応グループを初期化
    response_group = np.zeros(N, dtype=int)
    
    # 特徴量に基づいてグループを割り当て
    # グループ1: 高年齢 + 高マーケティング受容性 -> 常に成約
    group1_mask = (age > 60) & (marketing_receptivity > 1.5)
    response_group[group1_mask] = 1
    
    # グループ2: 中年齢 + マーケティング受容性中以上 -> 処置で成約
    group2_mask = (age > 30) & (age <= 60) & (marketing_receptivity > 0) & ~group1_mask
    response_group[group2_mask] = 2
    
    # グループ3: 中～高年齢 + 高価格感度 -> 処置で離脱
    group3_mask = (age > 40) & (price_sensitivity > 1.0) & ~group1_mask & ~group2_mask
    response_group[group3_mask] = 3
    
    # グループ4: その他すべて -> 常に不成約
    response_group[(response_group == 0)] = 4
    
    # 各グループの割合をチェック
    if group_ratios is not None:
        # 目標の割合が設定されている場合は調整
        target_counts = [int(N * ratio) for ratio in group_ratios]
        for group_id in range(1, 5):
            current_count = np.sum(response_group == group_id)
            target_count = target_counts[group_id - 1]
            
            if current_count < target_count:
                # グループを増やす必要がある場合
                # 現在他のグループのものからランダムに選んでこのグループに割り当てる
                other_indices = np.where(response_group != group_id)[0]
                if len(other_indices) > 0:
                    add_count = min(target_count - current_count, len(other_indices))
                    to_change = np.random.choice(other_indices, size=add_count, replace=False)
                    response_group[to_change] = group_id
            elif current_count > target_count:
                # グループを減らす必要がある場合
                # このグループのものからランダムに選んで他のグループに割り当てる
                group_indices = np.where(response_group == group_id)[0]
                if len(group_indices) > 0:
                    remove_count = min(current_count - target_count, len(group_indices))
                    to_change = np.random.choice(group_indices, size=remove_count, replace=False)
                    # 他のグループにランダムに割り当て
                    other_groups = [g for g in range(1, 5) if g != group_id]
                    response_group[to_change] = np.random.choice(other_groups, size=remove_count)
    
    # 3. True propensity score （傾向スコア）
    logit_ps = -8 + 0.1 * age + 1.0 * homeown
    ps = 1 / (1 + np.exp(-logit_ps))
    
    # 4. Treatment assignment （処置割り当て）
    if n_treated is None:
        treatment = np.random.binomial(1, ps, size=N)
    else:
        # weighted sampling without replacement
        probs = ps / ps.sum()
        treated_idx = np.random.choice(N, size=n_treated, replace=False, p=probs)
        treatment = np.zeros(N, dtype=int)
        treatment[treated_idx] = 1
    
    # 5. グループごとの確率モデルパラメータ
    # 各グループに対して別々のパラメータでbase_probとtreat_probを計算
    base_prob = np.zeros(N)
    treat_prob = np.zeros(N)
    
    # グループ1: 常にoutcome=1 (y0=1, y1=1)
    g1_indices = np.where(response_group == 1)[0]
    g1_logit_base = -2 + 0.08 * age[g1_indices] + 0.7 * homeown[g1_indices]
    base_prob[g1_indices] = 1 / (1 + np.exp(-g1_logit_base))
    g1_logit_treat = -1.5 + 0.07 * age[g1_indices] + 0.8 * homeown[g1_indices]
    treat_prob[g1_indices] = 1 / (1 + np.exp(-g1_logit_treat))
    
    # グループ2: treatmentで成約 (y0=0, y1=1)
    g2_indices = np.where(response_group == 2)[0]
    g2_logit_base = -5 + 0.04 * age[g2_indices] + 0.3 * homeown[g2_indices]
    base_prob[g2_indices] = 1 / (1 + np.exp(-g2_logit_base))
    g2_logit_treat = -1 + 0.06 * age[g2_indices] + 0.6 * homeown[g2_indices]
    treat_prob[g2_indices] = 1 / (1 + np.exp(-g2_logit_treat))
    
    # グループ3: treatmentで離脱 (y0=1, y1=0)
    g3_indices = np.where(response_group == 3)[0]
    g3_logit_base = -1 + 0.06 * age[g3_indices] + 0.6 * homeown[g3_indices]
    base_prob[g3_indices] = 1 / (1 + np.exp(-g3_logit_base))
    g3_logit_treat = -5 + 0.03 * age[g3_indices] + 0.3 * homeown[g3_indices]
    treat_prob[g3_indices] = 1 / (1 + np.exp(-g3_logit_treat))
    
    # グループ4: 常に不成約 (y0=0, y1=0)
    g4_indices = np.where(response_group == 4)[0]
    g4_logit_base = -6 + 0.02 * age[g4_indices] + 0.2 * homeown[g4_indices]
    base_prob[g4_indices] = 1 / (1 + np.exp(-g4_logit_base))
    g4_logit_treat = -5.5 + 0.025 * age[g4_indices] + 0.25 * homeown[g4_indices]
    treat_prob[g4_indices] = 1 / (1 + np.exp(-g4_logit_treat))
    
    # 6. 確率からアウトカムを生成
    np.random.seed(seed + 1)
    y0 = (np.random.random(N) < base_prob).astype(int)
    np.random.seed(seed + 2)
    y1 = (np.random.random(N) < treat_prob).astype(int)
    
    # 7. Observed conversion probability and outcome
    conversion_prob = np.where(treatment == 1, treat_prob, base_prob)
    outcome = np.where(treatment == 1, y1, y0)
    
    # 8. DataFrame作成
    df = pd.DataFrame({
        'age': age,
        'homeownership': homeown,
        'base_prob': base_prob,
        'treat_prob': treat_prob,
        'propensity_score': ps,
        'treatment': treatment,
        'conversion_prob': conversion_prob,
        'outcome': outcome,
        'y0': y0,
        'y1': y1,
        'response_group': response_group
    })
    
    # グループごとの統計を出力
    print("Response Group Statistics:")
    for group in range(1, 5):
        count = (df['response_group'] == group).sum()
        prop = count / N
        print(f"Group {group}: {count} samples ({prop:.1%})")
    
    # グループごとのupift統計も出力
    print("\nAverage Treatment Effect by Group:")
    for group in range(1, 5):
        group_df = df[df['response_group'] == group]
        avg_y1 = group_df['y1'].mean()
        avg_y0 = group_df['y0'].mean()
        uplift = avg_y1 - avg_y0
        print(f"Group {group}: uplift = {uplift:.4f} (y1 = {avg_y1:.4f}, y0 = {avg_y0:.4f})")
    
    return df

# Example usage
# デフォルトでは特徴量に基づいてグループ分けを行う
df = generate_simulation(N=10000, n_treated=2000, seed=0)
print(df.head())
df.to_csv("df.csv", index=False)
