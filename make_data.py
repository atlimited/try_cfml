import numpy as np
import pandas as pd

def generate_simulation(N=1000, n_treated=None, seed=0):
    """
    Generate simulation data with controllable number of treated individuals.
    
    Parameters:
    - N: total number of samples
    - n_treated: desired number of treated samples. If None, treatment assigned by propensity.
    - seed: random seed for reproducibility
    
    Returns:
    - df: pandas DataFrame with columns:
        age, homeownership, base_prob, treat_prob, propensity_score,
        treatment, conversion_prob, outcome, y0, y1
    """
    np.random.seed(seed)
    # 1. Features
    age = np.random.randint(20, 81, size=N)
    homeown = np.random.binomial(1, 0.6, size=N)
    
    # 2. True propensity score
    logit_ps = -8 + 0.1 * age + 1.0 * homeown
    ps = 1 / (1 + np.exp(-logit_ps))
    
    # 3. Treatment assignment
    if n_treated is None:
        treatment = np.random.binomial(1, ps, size=N)
    else:
        # weighted sampling without replacement
        probs = ps / ps.sum()
        treated_idx = np.random.choice(N, size=n_treated, replace=False, p=probs)
        treatment = np.zeros(N, dtype=int)
        treatment[treated_idx] = 1

    # 4. Base and treated conversion probabilities
    logit_base = -4 + 0.05 * age + 0.5 * homeown
    base_prob = 1 / (1 + np.exp(-logit_base))
    logit_treated = logit_base + 1.0
    treat_prob = 1 / (1 + np.exp(-logit_treated))
    
    # 5. Potential outcomes
    y0 = (base_prob > 0.5).astype(int)
    y1 = (treat_prob > 0.5).astype(int)
    
    # 6. Observed conversion probability and outcome
    conversion_prob = np.where(treatment == 1, treat_prob, base_prob)
    outcome = (conversion_prob > 0.5).astype(int)
    
    # Assemble DataFrame
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
        'y1': y1
    })
    
    return df

# Example usage: N=1000, exactly 200 treated
#df = generate_simulation(N=1000, n_treated=200, seed=0)
df = generate_simulation(N=10000, n_treated=2000, seed=0)
print(df.head())
df.to_csv("df.csv", index=False)
