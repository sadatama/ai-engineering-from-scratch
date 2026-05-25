# 支持向量机

> 在两个类别之间找到最宽的街道。这就是全部思想。

**类型：** 构建
**语言：** Python
**前置条件：** 第一阶段（第 08 课优化、第 14 课范数与距离、第 18 课凸优化）
**时间：** 约 90 分钟

## 学习目标

- 使用合页损失和梯度下降在原始形式上从零实现线性 SVM
- 解释最大间隔原理，并从训练好的模型中识别支持向量
- 比较线性核、多项式核和 RBF 核，并解释核技巧如何避免显式的高维映射
- 评估由 C 参数控制的间隔宽度与分类错误之间的权衡

## 问题

你有两类数据点，需要画一条线（或超平面）将它们分隔开。无限多条线都可以完成任务。你应该选哪一条？

具有最大间隔的那一条。间隔是决策边界到每侧最近数据点的距离。更大的间隔意味着分类器更自信，对未见过数据的泛化能力更好。

这种直觉引出了支持向量机（SVM），这是机器学习中数学最优雅的算法之一。SVM 在深度学习之前是主导性的分类方法，至今仍是小数据集、高维数据以及需要原理清晰、有理论保证的模型时的最佳选择。

SVM 与第一阶段的内容直接相连：优化是凸的（第 18 课），间隔用范数测量（第 14 课），核技巧利用点积处理非线性边界，而无需在高维空间中计算。

## 概念

### 最大间隔分类器

给定线性可分的数据，标签 y_i 在 {-1, +1} 中，特征向量为 x_i，我们希望找到一个超平面 w^T x + b = 0 来分隔类别。

点 x_i 到超平面的距离为：

```
distance = |w^T x_i + b| / ||w||
```

对于正确分类的点：y_i * (w^T x_i + b) > 0。间隔是超平面到每侧最近点距离的两倍。

```mermaid
graph LR
    subgraph Margin
        direction TB
        A["w^T x + b = +1"] ~~~ B["w^T x + b = 0"] ~~~ C["w^T x + b = -1"]
    end
    D["正类点"] --> A
    E["负类点"] --> C
    B --- F["决策边界"]
```

优化问题：

```
maximize    2 / ||w||     （间隔宽度）
subject to  y_i * (w^T x_i + b) >= 1  for all i
```

等价地（最小化 ||w||^2 更容易优化）：

```
minimize    (1/2) ||w||^2
subject to  y_i * (w^T x_i + b) >= 1  for all i
```

这是一个凸二次规划问题，具有唯一的全局解。恰好位于间隔边界上的数据点（满足 y_i * (w^T x_i + b) = 1）就是支持向量。它们是唯一决定决策边界的点。移动或移除任何非支持向量的点，边界不会改变。

### 支持向量：关键的那少数几个

```mermaid
graph TD
    subgraph Classification
        SV1["支持向量（正类）<br>y(w'x+b) = 1"] --- DB["决策边界<br>w'x+b = 0"]
        DB --- SV2["支持向量（负类）<br>y(w'x+b) = 1"]
    end
    O1["其他正类点<br>（不影响边界）"] -.-> SV1
    O2["其他负类点<br>（不影响边界）"] -.-> SV2
```

大多数训练点是无足轻重的，只有支持向量才重要。这就是为什么 SVM 在预测时内存高效：你只需要存储支持向量，而不需要整个训练集。

支持向量的数量也给出了泛化误差的边界。相对于数据集大小，支持向量越少，泛化能力越好。

### 软间隔：用 C 参数处理噪声

真实数据很少能完美分隔。有些点可能在边界错误的一侧，或在间隔内部。软间隔形式通过引入松弛变量来允许违规。

```
minimize    (1/2) ||w||^2 + C * sum(xi_i)
subject to  y_i * (w^T x_i + b) >= 1 - xi_i
            xi_i >= 0  for all i
```

松弛变量 xi_i 衡量点 i 违反间隔的程度。C 控制权衡：

| C 值 | 行为 |
|---------|----------|
| 大 C | 严厉惩罚违规。窄间隔，更少的误分类。容易过拟合 |
| 小 C | 允许更多违规。宽间隔，更多的误分类。容易欠拟合 |

C 是正则化的强度，但方向相反。大 C = 更少的正则化，小 C = 更多的正则化。

### 合页损失：SVM 的损失函数

软间隔 SVM 可以重写为无约束优化问题：

```
minimize    (1/2) ||w||^2 + C * sum(max(0, 1 - y_i * (w^T x_i + b)))
```

项 max(0, 1 - y_i * f(x_i)) 就是合页损失。当点被正确分类且在间隔之外时，损失为零。当点在间隔内部或被误分类时，损失是线性的。

```
单个点的合页损失：

loss
  |
  | \
  |  \
  |   \
  |    \
  |     \_______________
  |
  +-----|-----|-------->  y * f(x)
       0     1

当 y*f(x) >= 1 时损失为零（正确分类，位于间隔之外）。
当 y*f(x) < 1 时产生线性惩罚。
```

与 Logistic 损失（逻辑回归）对比：

```
合页损失:     max(0, 1 - y*f(x))          在间隔处有硬截断
Logistic 损失:  log(1 + exp(-y*f(x)))      光滑，永远不完全为零
```

合页损失产生稀疏解（只有支持向量有非零贡献）。Logistic 损失使用所有数据点。这使得 SVM 在预测时更内存高效。

### 使用梯度下降训练线性 SVM

你可以使用梯度下降在合页损失加 L2 正则化上训练线性 SVM，而无需求解带约束的二次规划：

```
L(w, b) = (lambda/2) * ||w||^2 + (1/n) * sum(max(0, 1 - y_i * (w^T x_i + b)))

关于 w 的梯度：
  如果 y_i * (w^T x_i + b) >= 1:  dL/dw = lambda * w
  如果 y_i * (w^T x_i + b) < 1:   dL/dw = lambda * w - y_i * x_i

关于 b 的梯度：
  如果 y_i * (w^T x_i + b) >= 1:  dL/db = 0
  如果 y_i * (w^T x_i + b) < 1:   dL/db = -y_i
```

这称为原始形式。每轮（epoch）的时间复杂度为 O(n * d)，其中 n 是样本数，d 是特征数。对于大规模、稀疏、高维数据（文本分类），这很快。

### 对偶形式与核技巧

SVM 问题的拉格朗日对偶（来自第一阶段第 18 课的 KKT 条件）是：

```
maximize    sum(alpha_i) - (1/2) * sum_ij(alpha_i * alpha_j * y_i * y_j * (x_i . x_j))
subject to  0 <= alpha_i <= C
            sum(alpha_i * y_i) = 0
```

对偶形式只涉及数据点之间的点积 x_i . x_j。这是关键洞见。将每个点积替换为核函数 K(x_i, x_j)，SVM 就能学习非线性边界，而无需显式地计算变换。

```
线性核:      K(x, z) = x . z
多项式核:  K(x, z) = (x . z + c)^d
RBF（高斯）核:     K(x, z) = exp(-gamma * ||x - z||^2)
```

RBF 核将数据映射到无限维空间。在输入空间中靠近的点，核值接近 1；远离的点，核值接近 0。它可以学习任何光滑的决策边界。

```mermaid
graph LR
    subgraph "输入空间（不可分）"
        A["2D 中的数据点<br>圆形边界"]
    end
    subgraph "特征空间（可分）"
        B["高维中的数据点<br>线性边界"]
    end
    A -->|"核技巧<br>K(x,z) = phi(x).phi(z)"| B
```

核技巧在高维空间中计算点积，而无需真正进入那个空间。对于 D 维空间中的 d 次多项式核，显式特征空间有 O(D^d) 个维度。但 K(x, z) 只需 O(D) 时间即可计算。

### 用于回归的 SVM（SVR）

支持向量回归拟合一个宽度为 epsilon 的管道环绕数据。管道内的点损失为零，管道外的点被线性惩罚。

```
minimize    (1/2) ||w||^2 + C * sum(xi_i + xi_i*)
subject to  y_i - (w^T x_i + b) <= epsilon + xi_i
            (w^T x_i + b) - y_i <= epsilon + xi_i*
            xi_i, xi_i* >= 0
```

epsilon 参数控制管道宽度。更宽 = 更少的支持向量 = 更平滑的拟合。更窄 = 更多的支持向量 = 更紧密的拟合。

### 为什么 SVM 输给了深度学习（以及它们仍然胜出的场景）

SVM 从 20 世纪 90 年代末到 2010 年代初主导了机器学习。深度学习超越它们的原因包括：

| 因素 | SVM | 深度学习 |
|--------|------|---------------|
| 特征工程 | 需要特征工程 | 自动学习特征 |
| 可扩展性 | 核方法 O(n^2) 到 O(n^3) | SGD 下每轮 O(n) |
| 图像/文本/音频 | 需要手工特征 | 从原始数据学习 |
| 大数据集（>10 万） | 慢 | 扩展性好 |
| GPU 加速 | 收益有限 | 大幅加速 |

SVM 在以下场景仍然胜出：
- 小数据集（数百到数千个样本）
- 高维稀疏数据（使用 TF-IDF 特征的文本分类）
- 当你需要数学保证时（间隔边界）
- 训练时间必须极短时（线性 SVM 非常快）
- 具有清晰间隔结构的二分类问题
- 异常检测（单类 SVM）

## 动手实现

### 步骤 1：合页损失与梯度

基础部分：计算一个批次的合页损失及其梯度。

```python
def hinge_loss(X, y, w, b):
    n = len(X)
    total_loss = 0.0
    for i in range(n):
        margin = y[i] * (dot(w, X[i]) + b)
        total_loss += max(0.0, 1.0 - margin)
    return total_loss / n
```

### 步骤 2：通过梯度下降训练线性 SVM

通过最小化正则化的合页损失来训练。无需二次规划求解器。

```python
class LinearSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_epochs=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_epochs = n_epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n_features = len(X[0])
        self.w = [0.0] * n_features
        self.b = 0.0

        for epoch in range(self.n_epochs):
            for i in range(len(X)):
                margin = y[i] * (dot(self.w, X[i]) + self.b)
                if margin >= 1:
                    self.w = [wj - self.lr * self.lambda_param * wj
                              for wj in self.w]
                else:
                    self.w = [wj - self.lr * (self.lambda_param * wj - y[i] * X[i][j])
                              for j, wj in enumerate(self.w)]
                    self.b -= self.lr * (-y[i])

    def predict(self, X):
        return [1 if dot(self.w, x) + self.b >= 0 else -1 for x in X]
```

### 步骤 3：核函数

实现线性核、多项式核和 RBF 核。

```python
def linear_kernel(x, z):
    return dot(x, z)

def polynomial_kernel(x, z, degree=3, c=1.0):
    return (dot(x, z) + c) ** degree

def rbf_kernel(x, z, gamma=0.5):
    diff = [xi - zi for xi, zi in zip(x, z)]
    return math.exp(-gamma * dot(diff, diff))
```

### 步骤 4：间隔与支持向量识别

训练后，识别哪些点是支持向量并计算间隔宽度。

```python
def find_support_vectors(X, y, w, b, tol=1e-3):
    support_vectors = []
    for i in range(len(X)):
        margin = y[i] * (dot(w, X[i]) + b)
        if abs(margin - 1.0) < tol:
            support_vectors.append(i)
    return support_vectors
```

完整实现及所有演示参见 `code/svm.py`。

## 使用它

使用 scikit-learn：

```python
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale")),
])
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
print(f"Support vectors: {clf['svm'].n_support_}")
```

重要提示：在训练 SVM 之前，始终对特征进行缩放。SVM 对特征量纲敏感，因为间隔依赖于 ||w||，而未缩放的特征会扭曲几何结构。

对于大数据集，使用 `LinearSVC`（原始形式，每轮 O(n)）而非 `SVC`（对偶形式，O(n^2) 到 O(n^3)）：

```python
from sklearn.svm import LinearSVC

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(C=1.0, max_iter=10000)),
])
```

## 练习

1. 生成一个 2D 线性可分的数据集。训练你的 LinearSVM 并识别支持向量。验证支持向量是距离决策边界最近的点。
2. 在一个含噪声的数据集上，将 C 从 0.001 变化到 1000。绘制每个 C 值的决策边界。观察从宽间隔（欠拟合）到窄间隔（过拟合）的转变。
3. 创建一个类别边界为圆形（非线性）的数据集。展示线性 SVM 的失败。计算 RBF 核矩阵，并展示类别在核诱导的特征空间中变得可分了。
4. 在同一数据集上比较合页损失与 Logistic 损失。训练一个线性 SVM 和一个逻辑回归。统计每个模型的决策边界有多少训练点参与贡献（支持向量 vs 所有点）。
5. 实现 SVR（epsilon 不敏感损失）。拟合 y = sin(x) + noise。绘制预测周围的 epsilon 管道，并高亮支持向量（管道外的点）。

## 关键术语

| 术语 | 它实际是什么意思 |
|------|----------------------|
| 支持向量 | 最靠近决策边界的训练点。唯一决定超平面的点 |
| 间隔 | 决策边界与最近支持向量之间的距离。SVM 最大化这个距离 |
| 合页损失 | max(0, 1 - y*f(x))。当正确分类且在间隔之外时为零，否则产生线性惩罚 |
| C 参数 | 间隔宽度与分类错误之间的权衡。大 C = 窄间隔，小 C = 宽间隔 |
| 软间隔 | 通过松弛变量允许间隔违规的 SVM 形式，处理不可分数据 |
| 核技巧 | 在高维特征空间中计算点积，而无需显式映射到该空间 |
| 线性核 | K(x, z) = x . z。等价于标准点积，适用于线性可分数据 |
| RBF 核 | K(x, z) = exp(-gamma * \|\|x-z\|\|^2)。映射到无限维，可学习任意光滑边界 |
| 多项式核 | K(x, z) = (x . z + c)^d。映射到多项式组合的特征空间 |
| 对偶形式 | 仅依赖数据点之间点积的 SVM 问题重表述，使核方法成为可能 |
| SVR | 支持向量回归。围绕数据拟合一个 epsilon 管道，管道内的点损失为零 |
| 松弛变量 | xi_i：衡量一个点违反间隔的程度。对正确分类且位于间隔外的点为零 |
| 最大间隔 | 选择最大化到每类最近点距离的超平面的原则 |

## 延伸阅读

- [Vapnik: The Nature of Statistical Learning Theory (1995)](https://link.springer.com/book/10.1007/978-1-4757-3264-1) - SVM 和统计学习的基础性著作
- [Cortes & Vapnik: Support-vector networks (1995)](https://link.springer.com/article/10.1007/BF00994018) - SVM 原始论文
- [Platt: Sequential Minimal Optimization (1998)](https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/) - 使 SVM 训练变得实用的 SMO 算法
- [scikit-learn SVM 文档](https://scikit-learn.org/stable/modules/svm.html) - 包含实现细节的实用指南
- [LIBSVM: A Library for Support Vector Machines](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) - 大多数 SVM 实现背后的 C++ 库
