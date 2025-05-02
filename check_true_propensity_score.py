#import pandas as pd
#import matplotlib.pyplot as plt
#
#df = pd.read_csv("df.csv")
#
#
## 共変量のヒストグラム
#plt.figure()
#plt.hist(df.loc[df['homeownership']==0, 'propensity_score'], bins=20, density=True, alpha=0.5, label='Control')
#plt.hist(df.loc[df['homeownership']==1, 'propensity_score'], bins=20, density=True, alpha=0.5, label='Treated')
#plt.xlabel('Propensity Score')
#plt.ylabel('Density')
#plt.title('Propensity Score Distribution by Homeownership')
#plt.legend()
#plt.savefig("propensity_score_by_homeownership.png")
#
#plt.show()
#
## 共変量のヒストグラム
#plt.figure()
#plt.hist(df.loc[df['age']<60, 'propensity_score'], bins=20, density=True, alpha=0.5, label='Control')
#plt.hist(df.loc[df['age']>=60, 'propensity_score'], bins=20, density=True, alpha=0.5, label='Treated')
#plt.xlabel('Propensity Score')
#plt.ylabel('Density')
#plt.title('Propensity Score Distribution by Age')
#plt.legend()
#plt.savefig("propensity_score_by_age.png")
#
#plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from statsmodels.distributions.empirical_distribution import ECDF

# データ読み込み
df = pd.read_csv("df.csv")

# 傾向スコアのヒストグラム
plt.figure()
plt.hist(df.loc[df['treatment']==0, 'propensity_score'], bins=20, density=True, alpha=0.5, label='Control')
plt.hist(df.loc[df['treatment']==1, 'propensity_score'], bins=20, density=True, alpha=0.5, label='Treated')
plt.xlabel('Propensity Score')
plt.ylabel('Density')
plt.title('Propensity Score Distribution by Treatment Assignment')
plt.legend()
plt.savefig("propensity_score_by_treatment.png")
plt.show()

# 推定傾向スコアの再計算
X_ps = df[['age', 'homeownership']]
ps_model = LogisticRegression(solver='lbfgs', max_iter=1000)
ps_model.fit(X_ps, df['treatment'])
df['ps_hat'] = ps_model.predict_proba(X_ps)[:, 1]

# 各群の PS
ps_treated = df[df['treatment']==1]['ps_hat']
ps_control = df[df['treatment']==0]['ps_hat']

# 1. ECDF
ecdf_t = ECDF(ps_treated)
ecdf_c = ECDF(ps_control)
x = np.linspace(0, 1, 200)

plt.figure()
plt.step(x, ecdf_t(x), label='Treated', where='post')
plt.step(x, ecdf_c(x), label='Control', where='post')
plt.xlabel('Propensity Score (estimated)')
plt.ylabel('Empirical CDF')
plt.title('ECDF of Estimated Propensity Scores')
plt.legend()
plt.show()

# 2. QQ プロット
t_sorted = np.sort(ps_treated)
c_sorted = np.sort(ps_control)
m = min(len(t_sorted), len(c_sorted))

plt.figure()
plt.scatter(c_sorted[:m], t_sorted[:m], alpha=0.5)
plt.plot([0,1],[0,1],'k--')
plt.xlabel('Control PS Quantiles')
plt.ylabel('Treated PS Quantiles')
plt.title('QQ Plot of Estimated Propensity Scores')
plt.show()

# 3. ボックスプロット
plt.figure()
plt.boxplot([ps_control, ps_treated], labels=['Control','Treated'])
plt.ylabel('Estimated Propensity Score')
plt.title('Boxplot of Estimated PS by Group')
plt.show()

## 3. ボックスプロット
#plt.figure()
#plt.boxplot([ps_control, ps_treated], labels=['homeownership'])
#plt.ylabel('Estimated Propensity Score')
#plt.title('Boxplot of Estimated PS by Group')
#plt.show()


## 傾向スコアの列名を確認し、推定済みでなければ再計算
#if 'estimated_propensity_score' in df.columns:
#    df['ps_hat'] = df['estimated_propensity_score']
#else:
#    X_ps = df[['age', 'homeownership']]
#    ps_model = LogisticRegression(solver='lbfgs', max_iter=1000)
#    ps_model.fit(X_ps, df['treatment'])
#    df['ps_hat'] = ps_model.predict_proba(X_ps)[:, 1]
#
## 1. Scatter plot: Age vs PS
#plt.figure(figsize=(8, 6))
#plt.scatter(df['age'], df['ps_hat'], c=df['treatment'], cmap='bwr', alpha=0.6, edgecolor='k')
#plt.colorbar(label='Treatment (0=Control, 1=Treated)')
#plt.xlabel('Age')
#plt.ylabel('Estimated Propensity Score')
#plt.title('Scatter Plot of Estimated PS vs Age')
#plt.grid(True)
#plt.show()
#
## 2. Hexbin plot: Age vs PS density
#plt.figure(figsize=(8, 6))
#plt.hexbin(df['age'], df['ps_hat'], gridsize=30, cmap='Blues', mincnt=1)
#plt.colorbar(label='Count')
#plt.xlabel('Age')
#plt.ylabel('Estimated Propensity Score')
#plt.title('Hexbin Plot of Estimated PS vs Age')
#plt.show()