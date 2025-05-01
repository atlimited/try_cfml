# try_cfml

```
% python -V
Python 3.11.8

pip install -r requirements.txt
```

# 問題設定

## 1. ビジネス背景と目的
- **商材**：ホームセキュリティシステムの無料体験クーポン  
- **目的**：  
  1. 過去のクーポン施策（送付 vs 未送付）が「成約率（conversion）」に与えた平均的な効果（ATE）を定量化  
  2. 各ユーザーの特性に応じた個別効果（ITE）を推定し、次回以降の最適ターゲティングを設計  

## 2. シミュレーションデータの生成ルール
1. **ユーザー属性（特徴量）**  
   - `age`：20～80 歳の一様ランダム整数  
   - `homeownership`：持ち家（1）／賃貸（0）を 60% の確率でバイナリ生成  
2. **真の傾向スコア（Propensity Score; PS）**  
   \[
     \text{logit}(\Pr(T=1\mid X))
     = -8 + 0.1\cdot\text{age} + 1.0\cdot\text{homeownership}
   \]  
   \[
     \mathrm{PS} = \frac{1}{1+\exp(-\text{logit})}
   \]
3. **処置割り当て（Treatment）**  
   - デフォルト：各ユーザーを PS に従う確率でランダムにクーポン送付（T=1）  
   - オプション：PS に比例した確率でちょうど \(n_{\rm treated}\) 名だけをランダム選出  
4. **成約確率モデル**  
   - 「クーポン未送付時」のベース確率：  
     \[
       \text{logit}(\Pr(Y=1\mid T=0, X))
       = -4 + 0.05\cdot\text{age} + 0.5\cdot\text{homeownership}
     \]
   - 「クーポン送付時」はログオッズに \(+1.0\) を加算  
5. **成約アウトカム**  
   - 潜在確率を 0.5 で閾値処理し、\(y_0,y_1\in\{0,1\}\) を生成  
   - 観測アウトカム \(Y\) は、実際の \(T\) に応じて \(y_0\) or \(y_1\) を選択  
6. **出力データフレーム列**  
   - `age`, `homeownership`,  
   - `propensity_score`（真のPS）,  
   - `treatment` (0/1),  
   - `base_prob`（未処置確率）, `treat_prob`（処置後確率）,  
   - `conversion_prob`（観測時確率）, `outcome`（閾値化後）,  
   - `y0`, `y1`（真の潜在アウトカム）  

## 3. 解析タスク
1. **Propensity Score 推定**  
   - ロジスティック回帰で `age`・`homeownership` から PS モデルを学習し、`estimated_propensity_score` を算出  
2. **共通サポート確認とトリミング**  
   - 推定PSのヒストグラムを処置群／対照群で可視化  
   - \(0.05\le\widehat{PS}\le0.95\) の領域にサンプルを限定  
3. **ATE 推定（IPS 法）**  
   - 各サンプルに重み \(w_i = \frac{T_i}{\widehat{PS}_i} + \frac{1-T_i}{1-\widehat{PS}_i}\) を付与  
   - 処置群・対照群の重み付き平均差を ATE として算出  
4. **S-Learner による ITE/ATE 推定**  
   - 入力特徴に `treatment` を含む単一回帰モデル（線形 or ロジスティック）を学習  
   - \(T=1\)／\(0\) で個別予測し、差分を ITE、平均を ATE として得る  
5. **モデル評価**  
   - PS モデル：Accuracy, ROC-AUC, Brier score, Log loss  
   - S-Learner：アウトカム予測の Accuracy, ROC-AUC, Brier score, Log loss  
   - 真の ATE（\(\mathbb{E}[y_1-y_0]\)）との比較によるバイアス評価  
