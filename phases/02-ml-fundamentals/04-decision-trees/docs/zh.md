# 决策树与随机森林

> 决策树不过是一张流程图。但由它们组成的森林，是机器学习中最强大的工具之一。

**类型：** 构建
**语言：** Python
**前置条件：** 第一阶段（第 09 课信息论、第 06 课概率论）
**时间：** 约 90 分钟

## 学习目标

- 实现基尼不纯度、熵和信息增益的计算，以找到最优的决策树分裂点
- 从零构建一个带有预剪枝控制（最大深度、最小样本数）的决策树分类器
- 使用自助采样和特征随机化构建随机森林，并解释为什么它能降低方差
- 比较 MDI 特征重要性与排列重要性，并识别 MDI 在何时会产生偏差

## 问题

你有结构化数据。行是样本，列是特征，还有一列是你想预测的目标。你可以直接丢一个神经网络去处理。但对于结构化数据，基于树的模型（决策树、随机森林、梯度提升树）始终优于深度学习。在结构化数据的 Kaggle 竞赛中，是 XGBoost 和 LightGBM 占据主导地位，而非 Transformer。

为什么？树模型无需预处理即可处理混合特征类型（数值型和分类型）。它们无需特征工程即可处理非线性关系。它们是可解释的：你可以查看树的结构，确切地知道为什么做出某个预测。而随机森林，通过平均许多棵树，对于中等大小的数据集具有极强的抗过拟合能力。

本课使用递归分裂从零构建决策树，然后在此基础上构建随机森林。你将实现分裂标准背后的数学（基尼不纯度、熵、信息增益），并理解为什么弱学习器的集成能变成强学习器。

## 概念

### 决策树做了什么

决策树通过一系列是/否问题，将特征空间划分为矩形区域。

```mermaid
graph TD
    A["年龄 < 30?"] -->|是| B["收入 > 50k?"]
    A -->|否| C["信用评分 > 700?"]
    B -->|是| D["批准"]
    B -->|否| E["拒绝"]
    C -->|是| F["批准"]
    C -->|否| G["拒绝"]
```

每个内部节点对某个特征进行阈值判断。每个叶节点做出一个预测。要对新的数据点进行分类，你从根节点开始，沿着分支向下，直到到达叶子节点。

树是自顶向下构建的：在每个节点处，选择最能分离数据的特征和阈值。"最能"由分裂标准定义。

### 分裂标准：衡量不纯度

在每个节点，我们有一组样本。我们希望分裂它们，使得结果子节点尽可能"纯净"，即每个子节点主要包含一个类别。

**基尼不纯度** 衡量如果按照该节点的类别分布来标记一个随机选择的样本，它被误分类的概率。

```
Gini(S) = 1 - sum(p_k^2)

其中 p_k 是类别 k 在集合 S 中的比例。
```

对于纯节点（全部一个类别），基尼 = 0。对于 50/50 的二分类，基尼 = 0.5。越低越好。

```
示例：6 只猫，4 只狗

Gini = 1 - (0.6^2 + 0.4^2) = 1 - (0.36 + 0.16) = 0.48
```

**熵** 衡量一个节点中的信息量（混乱程度）。已在第一阶段第 09 课中介绍。

```
Entropy(S) = -sum(p_k * log2(p_k))
```

对于纯节点，熵 = 0。对于 50/50 的二分类，熵 = 1.0。越低越好。

```
示例：6 只猫，4 只狗

Entropy = -(0.6 * log2(0.6) + 0.4 * log2(0.4))
        = -(0.6 * -0.737 + 0.4 * -1.322)
        = 0.442 + 0.529
        = 0.971 比特
```

**信息增益** 是分裂后不纯度（熵或基尼）的降低量。

```
IG(S, feature, threshold) = Impurity(S) - weighted_avg(Impurity(S_left), Impurity(S_right))

其中权重是每个子节点中样本的比例。
```

每个节点的贪心算法：尝试每个特征和每个可能的阈值。选择最大化信息增益的（特征，阈值）对。

### 分裂的工作原理

对于当前节点包含 n 个特征和 m 个样本的数据集：

1. 对每个特征 j（j = 1 到 n）：
   - 按特征 j 对样本排序
   - 尝试每对连续不同值之间的中点作为阈值
   - 计算每个阈值的信息增益
2. 选择具有最高信息增益的特征和阈值
3. 将数据分裂为左侧（特征 <= 阈值）和右侧（特征 > 阈值）
4. 对每个子节点递归执行

这种贪心方法不能保证得到全局最优的树。找到最优树是 NP 难问题。但贪心分裂在实践中效果很好。

### 停止条件

没有停止条件时，树会一直生长直到每个叶节点都是纯的（每片叶子只有一个样本）。这会完美地记住训练数据，但泛化能力极差。

**预剪枝** 在树完全生长之前停止它：
- 最大深度：当树达到设定深度时停止分裂
- 每叶子最小样本数：如果节点的样本少于 k 个则停止
- 最小信息增益：如果最佳分裂对不纯度的改进低于阈值则停止
- 最大叶节点数：限制叶子的总数

**后剪枝** 先生长完整的树，然后修剪回退：
- 代价复杂度剪枝（scikit-learn 使用）：添加与叶子数量成正比的惩罚项。增加惩罚以得到更小的树
- 减少误差剪枝：如果验证误差没有增加，则移除子树

预剪枝更简单、更快速。后剪枝通常产生更好的树，因为它不会过早停止那些可能导向有用的进一步分裂。

### 用于回归的决策树

对于回归问题，叶节点的预测是该叶子中目标值的均值。分裂标准也随之改变：

**方差减少** 替代信息增益：

```
VR(S, feature, threshold) = Var(S) - weighted_avg(Var(S_left), Var(S_right))
```

选择能最大程度减少方差的分裂。树将输入空间划分为区域，并在每个区域中预测一个常量（均值）。

### 随机森林：集成的力量

单棵决策树具有高方差。数据的微小变化可能产生完全不同的树。随机森林通过对许多棵树取平均来解决这个问题。

```mermaid
graph TD
    D["训练数据"] --> B1["自助样本 1"]
    D --> B2["自助样本 2"]
    D --> B3["自助样本 3"]
    D --> BN["自助样本 N"]
    B1 --> T1["树 1<br>（随机特征子集）"]
    B2 --> T2["树 2<br>（随机特征子集）"]
    B3 --> T3["树 3<br>（随机特征子集）"]
    BN --> TN["树 N<br>（随机特征子集）"]
    T1 --> V["聚合预测<br>（多数投票或平均）"]
    T2 --> V
    T3 --> V
    TN --> V
```

两个随机性来源使树多样化：

**Bagging（自助聚合）：** 每棵树都在一个自助样本上训练，即从训练数据中有放回地随机抽取的样本。约有 63% 的原始样本出现在每个自助样本中（其余为袋外样本，可用于验证）。

**特征随机化：** 每次分裂时，仅考虑一个随机特征子集。对于分类，默认为 sqrt(n_features)；对于回归，默认为 n_features/3。这防止所有树都在同一个主导特征上分裂。

核心洞见：对许多去相关化的树取平均，能在不增加偏差的情况下降低方差。每棵单独的树可能平庸，但集成后就很强大。

### 特征重要性

随机森林自然地提供特征重要性分数。最常用的方法：

**平均不纯度减少（MDI）：** 对于每个特征，汇总在所有树和所有使用该特征的节点上的不纯度减少总量。在较早的分裂中产生更大不纯度减少的特征更重要。

```
importance(feature_j) = 对应于所有使用 feature_j 的节点的总和：
    (n_samples_at_node / n_total_samples) * impurity_decrease
```

这种方法速度快（在训练期间计算），但偏向于高基数特征和有很多可能分裂点的特征。

**排列重要性** 是另一种选择：随机打乱一个特征的值，测量模型准确率下降了多少。更可靠，但速度更慢。

### 树模型何时优于神经网络

树和森林在结构化数据上优于神经网络。原因如下：

| 因素 | 树模型 | 神经网络 |
|--------|-------|----------------|
| 混合类型（数值 + 分类） | 原生支持 | 需要编码 |
| 小数据集（< 1 万行） | 效果良好 | 过拟合 |
| 特征交互 | 通过分裂发现 | 需要架构设计 |
| 可解释性 | 完全透明 | 黑箱 |
| 训练时间 | 数分钟 | 数小时 |
| 超参数敏感度 | 低 | 高 |

当数据具有空间或序列结构（图像、文本、音频）时，神经网络胜出。对于扁平的表格特征，树模型是默认选择。

## 动手实现

### 步骤 1：基尼不纯度与熵

从零构建两种分裂标准，并验证它们对哪些分裂是好的达成一致。

```python
import math

def gini_impurity(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return 1.0 - sum((c / n) ** 2 for c in counts.values())

def entropy(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return -sum(
        (c / n) * math.log2(c / n) for c in counts.values() if c > 0
    )
```

### 步骤 2：找到最佳分裂

尝试每个特征和每个阈值。返回具有最高信息增益的那一个。

```python
def information_gain(parent_labels, left_labels, right_labels, criterion="gini"):
    measure = gini_impurity if criterion == "gini" else entropy
    n = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)
    if n_left == 0 or n_right == 0:
        return 0.0
    parent_impurity = measure(parent_labels)
    child_impurity = (
        (n_left / n) * measure(left_labels) +
        (n_right / n) * measure(right_labels)
    )
    return parent_impurity - child_impurity
```

### 步骤 3：构建 DecisionTree 类

递归分裂、预测和特征重要性跟踪。

```python
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, criterion="gini",
                 max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.tree = None
        self.feature_importances_ = None

    def fit(self, X, y):
        self.n_features = len(X[0])
        self.feature_importances_ = [0.0] * self.n_features
        self.n_samples = len(X)
        self.tree = self._build(X, y, depth=0)
        total = sum(self.feature_importances_)
        if total > 0:
            self.feature_importances_ = [
                fi / total for fi in self.feature_importances_
            ]

    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]
```

### 步骤 4：构建 RandomForest 类

自助采样、特征随机化和多数投票。

```python
class RandomForest:
    def __init__(self, n_trees=100, max_depth=None,
                 min_samples_split=2, max_features="sqrt",
                 criterion="gini"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.trees = []

    def fit(self, X, y):
        n = len(X)
        for _ in range(self.n_trees):
            indices = [random.randint(0, n - 1) for _ in range(n)]
            X_boot = [X[i] for i in indices]
            y_boot = [y[i] for i in indices]
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                criterion=self.criterion,
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def predict(self, X):
        all_preds = [tree.predict(X) for tree in self.trees]
        predictions = []
        for i in range(len(X)):
            votes = {}
            for preds in all_preds:
                v = preds[i]
                votes[v] = votes.get(v, 0) + 1
            predictions.append(max(votes, key=votes.get))
        return predictions
```

完整实现及所有辅助方法参见 `code/trees.py`。

## 使用它

使用 scikit-learn，训练一个随机森林只需三行：

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"Accuracy: {rf.score(X_test, y_test):.4f}")
print(f"Feature importances: {rf.feature_importances_}")
```

在实践中，梯度提升树（XGBoost、LightGBM、CatBoost）通常比随机森林更强，因为它们按序构建树，每棵树纠正前一棵的错误。但随机森林更难配置出错，并且几乎不需要超参数调优。

## 交付成果

本课程产出 `outputs/prompt-tree-interpreter.md` —— 一个为业务干系人解释决策树分裂的提示词。输入训练好的树结构（深度、特征、分裂阈值、准确率），它会将模型翻译成通俗易懂的规则，对特征重要性排序，标记过拟合或数据泄露，并推荐下一步行动。每当需要向不阅读代码的人解释基于树的模型时，都可以使用它。

## 练习

1. 在一个包含 3 个类别的 2D 数据集上训练单棵决策树。手动追踪分裂过程，画出矩形决策边界。比较 max_depth=2 与 max_depth=10 时的边界。
2. 为回归树实现方差减少分裂。为 200 个点生成 y = sin(x) + noise，并拟合你的回归树。绘制树的逐段常数预测与真实曲线的对比图。
3. 分别用 1、5、10、50 和 200 棵树构建随机森林。绘制训练准确率和测试准确率相对于树数量的变化曲线。观察测试准确率趋于平稳但不下降（森林抗过拟合）。
4. 在 5 个不同数据集上比较基尼不纯度与熵作为分裂标准。测量准确率和树的深度。在大多数情况下，它们产生几乎相同的结果。解释原因。
5. 实现排列重要性。在一个包含随机噪声但具有高基数的特征的数据集上，将其与 MDI 重要性对比。MDI 会给噪声特征高排名，排列重要性则不会。

## 关键术语

| 术语 | 人们怎么说 | 它实际是什么意思 |
|------|----------------|----------------------|
| 决策树 | "用于预测的流程图" | 通过学习一系列 if/else 分裂，将特征空间划分为矩形区域的模型 |
| 基尼不纯度 | "节点有多混杂" | 在节点上随机样本被误分类的概率。0 = 纯净，0.5 = 二分类最大不纯度 |
| 熵 | "节点中的混乱度" | 节点的信息量。0 = 纯净，1.0 = 二分类最大不确定性。来自信息论 |
| 信息增益 | "一次分裂有多好" | 分裂后不纯度的减少量。选择分裂的贪心标准 |
| 预剪枝 | "提前停止树的生长" | 通过设置最大深度、最小样本数或最小增益阈值来提前停止树的生长 |
| 后剪枝 | "事后修剪树" | 先生长完整的树，然后移除不能改善验证性能的子树 |
| Bagging | "在随机子集上训练" | 自助聚合。每个模型在不同的有放回随机样本上训练 |
| 随机森林 | "一堆树" | 决策树的集成，每棵树在自助样本上训练，每次分裂使用随机特征子集 |
| 特征重要性（MDI） | "哪些特征重要" | 每个特征在所有树和所有节点上贡献的不纯度减少总量 |
| 排列重要性 | "打乱然后检查" | 随机打乱特征值后模型准确率的下降量。对于噪声特征比 MDI 更可靠 |
| 方差减少 | "信息增益的回归版本" | 回归树中信息增益的对应物。选择最能减少目标变量方差的分裂 |
| 自助样本 | "带重复的随机样本" | 从原始数据集中有放回地抽取的随机样本。大小相同，但包含重复项 |

## 延伸阅读

- [Breiman: Random Forests (2001)](https://link.springer.com/article/10.1023/A:1010933404324) - 随机森林的原始论文
- [Grinsztajn 等: Why do tree-based models still outperform deep learning on tabular data? (2022)](https://arxiv.org/abs/2207.08815) - 树模型与神经网络在结构化任务上的严格对比
- [scikit-learn 决策树文档](https://scikit-learn.org/stable/modules/tree.html) - 包含可视化工具的实用指南
- [XGBoost: A Scalable Tree Boosting System (Chen & Guestrin, 2016)](https://arxiv.org/abs/1603.02754) - 主导 Kaggle 的梯度提升论文
