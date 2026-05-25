# 模型评估

> 模型的好坏取决于你衡量它的方式。

**类型：** 构建
**语言：** Python
**前置条件：** 第一阶段（概率与分布、机器学习统计学），第二阶段第1-8课
**时间：** 约90分钟

## 学习目标

- 从零实现K折和分层K折交叉验证，并解释为什么分层对不平衡数据至关重要
- 从零计算精确率、召回率、F1、AUC-ROC以及回归指标（MSE、RMSE、MAE、R-squared）
- 解读学习曲线以诊断模型是高偏差还是高方差
- 识别常见的评估错误，包括数据泄露、错误指标选择以及测试集污染

## 问题描述

你训练了一个模型。它在你的数据上达到了95%的准确率。它好吗？

也许好。也许不好。如果你的数据中95%属于同一个类别，一个总是预测该类别的模型就能达到95%的准确率，同时却毫无用处。如果你在训练数据上评估，95%这个数字毫无意义，因为模型只是记住了答案。如果你的数据集包含时间成分，而你在分割前进行了随机打乱，你的模型可能在用未来数据预测过去。

模型评估是大多数机器学习项目出错的地方。错误的指标让糟糕的模型看起来很好。错误的分割让模型作弊。错误的比较让你选择了更差的模型。正确地进行评估不是可选项，而是衡量一个模型是在生产环境中正常工作还是在看到真实数据时立即失败的区别。

## 概念讲解

### 训练集、验证集、测试集

```mermaid
flowchart LR
    A[完整数据集] --> B[训练集 60-70%]
    A --> C[验证集 15-20%]
    A --> D[测试集 15-20%]
    B --> E[拟合模型]
    E --> C
    C --> F[调优超参数]
    F --> E
    F --> G[最终模型]
    G --> D
    D --> H[报告性能]
```

三种分割，三个目的：

- **训练集**：模型从这些数据中学习。它在训练期间看到这些示例。
- **验证集**：用于调优超参数和在模型之间进行选择。模型从不在此数据上训练，但你的决策受其影响。
- **测试集**：恰好触碰一次，在最后用于报告最终性能。如果你看了测试性能后又回去修改模型，那它就不再是测试集了，而是变成了第二个验证集。

测试集是你保底的手段，保证报告的性能反映了模型在真正未见数据上的表现。

### K折交叉验证

对于小数据集，单次训练/验证分割浪费数据并产生嘈杂的估计。K折交叉验证使用所有数据同时进行训练和验证：

```mermaid
flowchart TB
    subgraph Fold1["第1折"]
        direction LR
        V1["验证"] --- T1a["训练"] --- T1b["训练"] --- T1c["训练"] --- T1d["训练"]
    end
    subgraph Fold2["第2折"]
        direction LR
        T2a["训练"] --- V2["验证"] --- T2b["训练"] --- T2c["训练"] --- T2d["训练"]
    end
    subgraph Fold3["第3折"]
        direction LR
        T3a["训练"] --- T3b["训练"] --- V3["验证"] --- T3c["训练"] --- T3d["训练"]
    end
    subgraph Fold4["第4折"]
        direction LR
        T4a["训练"] --- T4b["训练"] --- T4c["训练"] --- V4["验证"] --- T4d["训练"]
    end
    subgraph Fold5["第5折"]
        direction LR
        T5a["训练"] --- T5b["训练"] --- T5c["训练"] --- T5d["训练"] --- V5["验证"]
    end
    Fold1 --> R["平均得分"]
    Fold2 --> R
    Fold3 --> R
    Fold4 --> R
    Fold5 --> R
```

1. 将数据分成K个大小相等的折
2. 对每一折，在K-1折上训练，在剩余的一折上验证
3. 对K个验证分数取平均

K=5或K=10是标准选择。每个数据点恰好被用于验证一次。平均分数比任何单次分割都更稳定。

**分层K折**：在每一折中保持类别分布。如果你的数据集是70%的A类和30%的B类，每一折将有大致相同的比例。这对于不平衡数据集很重要，因为随机分割可能将所有少数类样本放入一个折中。

### 分类指标

**混淆矩阵**：基础。对于二分类：

|  | 预测为阳性 | 预测为阴性 |
|--|---|---|
| 实际阳性 | 真阳性 (TP) | 假阴性 (FN) |
| 实际阴性 | 假阳性 (FP) | 真阴性 (TN) |

从这个矩阵出发，所有其他指标：

- **准确率** = (TP + TN) / (TP + TN + FP + FN)。正确预测的比例。当类别不平衡时具有误导性。
- **精确率** = TP / (TP + FP)。在所有被预测为阳性的中，有多少真的是阳性？当假阳性的代价高昂时使用（例如垃圾邮件过滤器将正常邮件标记为垃圾邮件）。
- **召回率**（灵敏度）= TP / (TP + FN)。在所有实际阳性中，我们捕获了多少？当假阴性的代价高昂时使用（例如癌症筛查漏诊肿瘤）。
- **F1分数** = 2 * 精确率 * 召回率 / (精确率 + 召回率)。精确率和召回率的调和平均。当两者都不明显占优时平衡两者。
- **AUC-ROC**：受试者工作特征曲线下面积。在不同分类阈值下绘制真正例率 vs 假正例率。AUC = 0.5表示随机猜测，AUC = 1.0表示完美分离。与阈值无关：它衡量模型将正例排在负例前面的能力，无论你选择什么截断值。

### 回归指标

- **MSE**（均方误差）= mean((y_true - y_pred)^2)。以平方方式惩罚大误差。对离群值敏感。
- **RMSE**（均方根误差）= sqrt(MSE)。与目标变量单位相同。比MSE更容易解释。
- **MAE**（平均绝对误差）= mean(|y_true - y_pred|)。线性对待所有误差。比MSE对离群值更鲁棒。
- **R-squared** = 1 - SS_res / SS_tot，其中 SS_res = sum((y_true - y_pred)^2)，SS_tot = sum((y_true - y_mean)^2)。模型解释的方差比例。R^2 = 1.0是完美的。R^2 = 0.0表示模型不比总是预测均值更好。如果模型比均值还差，R^2可能为负。

### 学习曲线

绘制训练和验证分数随训练集大小变化的函数：

- **高偏差（欠拟合）**：两条曲线都收敛到一个低分数。增加更多数据不会改善。你需要一个更复杂的模型。
- **高方差（过拟合）**：训练分数很高，但验证分数低得多。两者之间的差距很大。增加更多数据应该有所帮助。

### 验证曲线

绘制训练和验证分数随超参数变化的函数：

- 在低复杂度：两个分数都低（欠拟合）
- 在合适的复杂度：两个分数都高且接近
- 在高复杂度：训练分数保持高位但验证分数下降（过拟合）

最优超参数值是验证分数达到峰值的地方。

### 常见评估错误

**数据泄露**：测试集的信息泄露到训练中。示例：在分割前对整个数据集拟合缩放器、在时间序列预测中包含未来数据、使用从目标派生的特征。始终先分割，然后预处理。

**类别不平衡**：99%的交易是合法的，1%是欺诈。一个总是预测"合法"的模型达到99%的准确率。请使用精确率、召回率、F1或AUC-ROC。

**错误的指标**：在应该优化召回率时优化准确率（医疗诊断），或在数据有大量离群值时优化RMSE（应使用MAE）。

**未使用分层分割**：对于不平衡数据，随机分割可能在验证折中放入极少的少数类样本，导致不稳定的估计。

**测试太频繁**：每次你看测试性能并做出调整，你就在对测试集过拟合。测试集是一次性使用的。

## 构建实现

### 第1步：训练/验证/测试分割

```python
import random
import math


def train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2, seed=42):
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test
```

### 第2步：K折和分层K折交叉验证

```python
def kfold_split(n, k=5, seed=42):
    random.seed(seed)
    indices = list(range(n))
    random.shuffle(indices)

    fold_size = n // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        val_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]
        folds.append((train_idx, val_idx))

    return folds


def stratified_kfold_split(y, k=5, seed=42):
    random.seed(seed)

    class_indices = {}
    for i, label in enumerate(y):
        class_indices.setdefault(label, []).append(i)

    for label in class_indices:
        random.shuffle(class_indices[label])

    folds = [{"train": [], "val": []} for _ in range(k)]

    for label, indices in class_indices.items():
        fold_size = len(indices) // k
        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else len(indices)
            val_part = indices[start:end]
            train_part = indices[:start] + indices[end:]
            folds[i]["val"].extend(val_part)
            folds[i]["train"].extend(train_part)

    return [(f["train"], f["val"]) for f in folds]


def cross_validate(X, y, model_fn, k=5, metric_fn=None, stratified=False):
    n = len(X)

    if stratified:
        folds = stratified_kfold_split(y, k)
    else:
        folds = kfold_split(n, k)

    scores = []
    for train_idx, val_idx in folds:
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_val = [y[i] for i in val_idx]

        model = model_fn()
        model.fit(X_train, y_train)
        predictions = [model.predict(x) for x in X_val]

        if metric_fn:
            score = metric_fn(y_val, predictions)
        else:
            score = sum(1 for yt, yp in zip(y_val, predictions) if yt == yp) / len(y_val)
        scores.append(score)

    return scores
```

### 第3步：混淆矩阵和分类指标

```python
def confusion_matrix(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return tp, tn, fp, fn


def accuracy(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0


def precision(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def roc_curve(y_true, y_scores):
    thresholds = sorted(set(y_scores), reverse=True)
    tpr_list = []
    fpr_list = []

    total_positives = sum(y_true)
    total_negatives = len(y_true) - total_positives

    for threshold in thresholds:
        y_pred = [1 if s >= threshold else 0 for s in y_scores]
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)

        tpr = tp / total_positives if total_positives > 0 else 0.0
        fpr = fp / total_negatives if total_negatives > 0 else 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return fpr_list, tpr_list, thresholds


def auc_roc(y_true, y_scores):
    fpr_list, tpr_list, _ = roc_curve(y_true, y_scores)

    pairs = sorted(zip(fpr_list, tpr_list))
    fpr_sorted = [p[0] for p in pairs]
    tpr_sorted = [p[1] for p in pairs]

    area = 0.0
    for i in range(1, len(fpr_sorted)):
        width = fpr_sorted[i] - fpr_sorted[i - 1]
        height = (tpr_sorted[i] + tpr_sorted[i - 1]) / 2
        area += width * height

    return area
```

### 第4步：回归指标

```python
def mse(y_true, y_pred):
    n = len(y_true)
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n


def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    n = len(y_true)
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n


def r_squared(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    ss_tot = sum((yt - mean_y) ** 2 for yt in y_true)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot
```

### 第5步：学习曲线

```python
def learning_curve(X, y, model_fn, metric_fn, train_sizes=None, val_ratio=0.2, seed=42):
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)

    val_size = int(n * val_ratio)
    val_idx = indices[:val_size]
    pool_idx = indices[val_size:]

    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]

    if train_sizes is None:
        train_sizes = [int(len(pool_idx) * r) for r in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]]

    train_scores = []
    val_scores = []

    for size in train_sizes:
        subset = pool_idx[:size]
        X_train = [X[i] for i in subset]
        y_train = [y[i] for i in subset]

        model = model_fn()
        model.fit(X_train, y_train)

        train_pred = [model.predict(x) for x in X_train]
        val_pred = [model.predict(x) for x in X_val]

        train_scores.append(metric_fn(y_train, train_pred))
        val_scores.append(metric_fn(y_val, val_pred))

    return train_sizes, train_scores, val_scores
```

### 第6步：用于测试的简单分类器及完整演示

```python
class SimpleLogistic:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def sigmoid(self, z):
        z = max(-500, min(500, z))
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, X, y):
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                z = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
                pred = self.sigmoid(z)
                error = yi - pred
                for j in range(n_features):
                    self.weights[j] += self.lr * error * xi[j]
                self.bias += self.lr * error

    def predict_proba(self, x):
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return self.sigmoid(z)

    def predict(self, x):
        return 1 if self.predict_proba(x) >= 0.5 else 0


class SimpleLinearRegression:
    def __init__(self, lr=0.001, epochs=200):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0
        n = len(X)

        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                pred = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
                error = yi - pred
                for j in range(n_features):
                    self.weights[j] += self.lr * error * xi[j] / n
                self.bias += self.lr * error / n

    def predict(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias


def standardize(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var) if var > 0 else 1.0
    return [(v - mean) / std for v in values], mean, std


def make_classification_data(n=300, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        x1 = random.gauss(0, 1)
        x2 = random.gauss(0, 1)
        label = 1 if (x1 + x2 + random.gauss(0, 0.5)) > 0 else 0
        X.append([x1, x2])
        y.append(label)
    return X, y


def make_regression_data(n=200, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 5)
        target = 3 * x1 + 2 * x2 + random.gauss(0, 2)
        X.append([x1, x2])
        y.append(target)
    return X, y


def make_imbalanced_data(n=300, minority_ratio=0.05, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        if random.random() < minority_ratio:
            x1 = random.gauss(3, 0.5)
            x2 = random.gauss(3, 0.5)
            label = 1
        else:
            x1 = random.gauss(0, 1)
            x2 = random.gauss(0, 1)
            label = 0
        X.append([x1, x2])
        y.append(label)
    return X, y


if __name__ == "__main__":
    X_clf, y_clf = make_classification_data(300)

    print("=== 训练/验证/测试分割 ===")
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X_clf, y_clf)
    print(f"  训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    print(f"  训练集类别分布: {sum(y_train)}/{len(y_train)} 阳性")
    print(f"  验证集类别分布: {sum(y_val)}/{len(y_val)} 阳性")

    model = SimpleLogistic(lr=0.1, epochs=200)
    model.fit(X_train, y_train)

    print("\n=== 分类指标 ===")
    y_pred = [model.predict(x) for x in X_test]
    tp, tn, fp, fn = confusion_matrix(y_test, y_pred)
    print(f"  混淆矩阵: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  准确率:  {accuracy(y_test, y_pred):.4f}")
    print(f"  精确率: {precision(y_test, y_pred):.4f}")
    print(f"  召回率:    {recall(y_test, y_pred):.4f}")
    print(f"  F1分数:  {f1_score(y_test, y_pred):.4f}")

    y_scores = [model.predict_proba(x) for x in X_test]
    auc = auc_roc(y_test, y_scores)
    print(f"  AUC-ROC:   {auc:.4f}")

    print("\n=== K折交叉验证 (K=5) ===")
    cv_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        k=5,
        metric_fn=accuracy,
    )
    mean_cv = sum(cv_scores) / len(cv_scores)
    std_cv = math.sqrt(sum((s - mean_cv) ** 2 for s in cv_scores) / len(cv_scores))
    print(f"  各折分数: {[round(s, 4) for s in cv_scores]}")
    print(f"  均值: {mean_cv:.4f} (+/- {std_cv:.4f})")

    print("\n=== 分层K折交叉验证 (K=5) ===")
    strat_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        k=5,
        metric_fn=accuracy,
        stratified=True,
    )
    strat_mean = sum(strat_scores) / len(strat_scores)
    strat_std = math.sqrt(sum((s - strat_mean) ** 2 for s in strat_scores) / len(strat_scores))
    print(f"  各折分数: {[round(s, 4) for s in strat_scores]}")
    print(f"  均值: {strat_mean:.4f} (+/- {strat_std:.4f})")

    print("\n=== 不平衡数据：为什么准确率会骗人 ===")
    X_imb, y_imb = make_imbalanced_data(300, minority_ratio=0.05)
    positives = sum(y_imb)
    print(f"  类别分布: {positives} 阳性, {len(y_imb) - positives} 阴性 ({positives/len(y_imb)*100:.1f}% 阳性)")

    always_negative = [0] * len(y_imb)
    print(f"  总是预测阴性的基线:")
    print(f"    准确率:  {accuracy(y_imb, always_negative):.4f}")
    print(f"    精确率: {precision(y_imb, always_negative):.4f}")
    print(f"    召回率:    {recall(y_imb, always_negative):.4f}")
    print(f"    F1分数:  {f1_score(y_imb, always_negative):.4f}")

    X_tr_i, y_tr_i, X_v_i, y_v_i, X_te_i, y_te_i = train_val_test_split(X_imb, y_imb)
    model_imb = SimpleLogistic(lr=0.5, epochs=500)
    model_imb.fit(X_tr_i, y_tr_i)
    y_pred_imb = [model_imb.predict(x) for x in X_te_i]
    print(f"\n  在不平衡数据上训练的模型:")
    print(f"    准确率:  {accuracy(y_te_i, y_pred_imb):.4f}")
    print(f"    精确率: {precision(y_te_i, y_pred_imb):.4f}")
    print(f"    召回率:    {recall(y_te_i, y_pred_imb):.4f}")
    print(f"    F1分数:  {f1_score(y_te_i, y_pred_imb):.4f}")

    print("\n=== 回归指标 ===")
    X_reg, y_reg = make_regression_data(200)

    col0 = [x[0] for x in X_reg]
    col1 = [x[1] for x in X_reg]
    col0_s, m0, s0 = standardize(col0)
    col1_s, m1, s1 = standardize(col1)
    X_reg_scaled = [[col0_s[i], col1_s[i]] for i in range(len(X_reg))]

    X_tr_r, y_tr_r, X_v_r, y_v_r, X_te_r, y_te_r = train_val_test_split(X_reg_scaled, y_reg)
    reg_model = SimpleLinearRegression(lr=0.01, epochs=500)
    reg_model.fit(X_tr_r, y_tr_r)
    y_pred_r = [reg_model.predict(x) for x in X_te_r]

    print(f"  MSE:       {mse(y_te_r, y_pred_r):.4f}")
    print(f"  RMSE:      {rmse(y_te_r, y_pred_r):.4f}")
    print(f"  MAE:       {mae(y_te_r, y_pred_r):.4f}")
    print(f"  R-squared: {r_squared(y_te_r, y_pred_r):.4f}")

    mean_baseline = [sum(y_tr_r) / len(y_tr_r)] * len(y_te_r)
    print(f"\n  均值基线:")
    print(f"    MSE:       {mse(y_te_r, mean_baseline):.4f}")
    print(f"    R-squared: {r_squared(y_te_r, mean_baseline):.4f}")

    print("\n=== 学习曲线 ===")
    sizes, train_sc, val_sc = learning_curve(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        metric_fn=accuracy,
    )
    print(f"  {'大小':>6} {'训练':>8} {'验证':>8}")
    for s, tr, va in zip(sizes, train_sc, val_sc):
        print(f"  {s:>6} {tr:>8.4f} {va:>8.4f}")

    print("\n=== 统计模型比较 ===")
    model_a_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=100),
        k=5, metric_fn=accuracy,
    )
    model_b_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=500),
        k=5, metric_fn=accuracy,
    )
    diffs = [a - b for a, b in zip(model_a_scores, model_b_scores)]
    mean_diff = sum(diffs) / len(diffs)
    std_diff = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / len(diffs))
    t_stat = mean_diff / (std_diff / math.sqrt(len(diffs))) if std_diff > 0 else 0.0
    print(f"  模型A（100轮）均值: {sum(model_a_scores)/len(model_a_scores):.4f}")
    print(f"  模型B（500轮）均值: {sum(model_b_scores)/len(model_b_scores):.4f}")
    print(f"  均值差异: {mean_diff:.4f}")
    print(f"  配对t统计量: {t_stat:.4f}")
    print(f"  (|t| > 2.78 自由度=4 时在p<0.05下显著)")
```

## 使用方式

使用scikit-learn，评估已内置于工作流程中：

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error, r2_score,
)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring="f1")
```

从零实现版本精确展示了交叉验证做了什么（没有魔法，只是for循环和索引跟踪），每个指标是如何计算的（只是统计TP/FP/TN/FN），以及为什么分层很重要（保留每一折中的类别比例）。库版本增加了并行性、更多评分选项以及流水线集成。

## 成果交付

本课程产出：
- `outputs/skill-evaluation.md` - 一个涵盖分类和回归模型评估策略的技能文档

## 练习题

1. 实现精确率-召回率曲线：在不同阈值下绘制精确率 vs 召回率。计算平均精确率（PR曲线下面积）。在不平衡数据集上比较PR曲线与ROC曲线，并解释每种曲线何时更有信息量。
2. 构建嵌套交叉验证循环：外层循环评估模型性能，内层循环调优超参数。使用它公平地比较两个模型，而不让验证数据泄露到评估中。
3. 实现用于模型比较的置换检验：打乱标签，重新训练，并测量性能。重复100次以构建零分布。计算观察到的模型性能在此分布下的p值。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| 过拟合 | "记住了训练数据" | 模型捕获了训练数据中的噪声，在训练数据上表现良好但在未见数据上表现差 |
| 交叉验证 | "在不同子集上测试" | 系统地轮换哪部分数据用于验证，在所有轮换中取平均结果 |
| 精确率 | "预测为阳性的中有多少正确" | TP / (TP + FP)：被预测为阳性的中实际为阳性的比例 |
| 召回率 | "我们找到了多少实际阳性" | TP / (TP + FN)：实际阳性中被正确识别的比例 |
| AUC-ROC | "模型分离类别的能力" | 在所有阈值下真正例率 vs 假正例率的曲线下面积，从0.5（随机）到1.0（完美） |
| R-squared | "解释了多大比例的方差" | 1 - (残差平方和 / 总平方和)：模型捕获的目标变量方差比例 |
| 数据泄露 | "模型作弊了" | 在训练期间使用了预测时不可用的信息，导致评估过于乐观 |
| 学习曲线 | "性能随更多数据如何变化" | 训练和验证分数随训练集大小变化的图，揭示欠拟合或过拟合 |
| 分层分割 | "保持类别比例平衡" | 分割数据使每个子集具有与完整数据集相同的各类别比例 |

## 延伸阅读

- [scikit-learn Model Selection Guide](https://scikit-learn.org/stable/model_selection.html) - 关于交叉验证、指标和超参数调优的综合参考
- [Beyond Accuracy: Precision and Recall (Google ML Crash Course)](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall) - 带有交互式示例的清晰解释
- [A Survey of Cross-Validation Procedures (Arlot & Celisse, 2010)](https://projecteuclid.org/journals/statistics-surveys/volume-4/issue-none/A-survey-of-cross-validation-procedures-for-model-selection/10.1214/09-SS054.full) - 关于不同交叉验证策略何时以及为何有效的严谨论述
