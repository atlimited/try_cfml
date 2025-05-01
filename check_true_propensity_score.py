import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("df.csv")

# 傾向スコアのヒストグラム
plt.figure()
plt.hist(df.loc[df['treatment']==0, 'propensity_score'], bins=20, density=True, alpha=0.5, label='Control')
plt.hist(df.loc[df['treatment']==1, 'propensity_score'], bins=20, density=True, alpha=0.5, label='Treated')
plt.xlabel('Propensity Score')
plt.ylabel('Density')
plt.title('Propensity Score Distribution by Treatment Assignment')
plt.legend()
plt.savefig("propensity_score.png")

plt.show()
