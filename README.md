# DL4CV - Deep Learning for Computer Vision

这个项目实现了多种经典的机器学习模型，用于使用 scikit-learn 的数字数据集进行数字分类。目标是比较不同的算法，包括逻辑回归、线性 SVM、K-最近邻、决策树、随机森林、SGD 分类器和感知器。

## 项目结构

```
DL4CV/
├─ data/                         # 数据目录
├─ results/
│  ├─ figures/                   # 生成的图表
│  └─ metrics/                   # 指标 JSON 文件
├─ src/
│  ├─ __init__.py                # 包初始化
│  ├─ load_data.py               # 数据加载函数
│  ├─ evaluate.py                # 模型评估函数
│  ├─ visualize.py               # 可视化函数
│  └─ models.py                  # 模型常量和映射
├─ notebooks/
│  ├─ project_overview.ipynb     # 项目概述
│  ├─ part1_logistic_regression.ipynb
│  ├─ part2_linear_svm.ipynb
│  ├─ part3_knn.ipynb
│  ├─ part4_decision_tree.ipynb
│  ├─ part5_random_forest.ipynb
│  ├─ part6_sgd.ipynb
│  ├─ part7_perceptron.ipynb
│  └─ part8_comparison.ipynb
├─ requirements.txt              # 项目依赖
└─ README.md                     # 本文件
```

## 环境设置

```bash
pip install -r requirements.txt
```

## 使用说明

### 运行单个模型

每个模型都有对应的 Jupyter Notebook 文件，位于 `notebooks/` 目录中：

1. **逻辑回归** (`part1_logistic_regression.ipynb`)
   ```bash
   jupyter notebook notebooks/part1_logistic_regression.ipynb
   ```

2. **线性 SVM** (`part2_linear_svm.ipynb`)
   ```bash
   jupyter notebook notebooks/part2_linear_svm.ipynb
   ```

3. **K-最近邻** (`part3_knn.ipynb`)
   ```bash
   jupyter notebook notebooks/part3_knn.ipynb
   ```

4. **决策树** (`part4_decision_tree.ipynb`)
   ```bash
   jupyter notebook notebooks/part4_decision_tree.ipynb
   ```

5. **随机森林** (`part5_random_forest.ipynb`)
   ```bash
   jupyter notebook notebooks/part5_random_forest.ipynb
   ```

6. **SGD 分类器** (`part6_sgd.ipynb`)
   ```bash
   jupyter notebook notebooks/part6_sgd.ipynb
   ```

7. **感知器** (`part7_perceptron.ipynb`)
   ```bash
   jupyter notebook notebooks/part7_perceptron.ipynb
   ```

### 模型比较

运行比较分析：

```bash
jupyter notebook notebooks/part8_comparison.ipynb
```

### 项目概述

查看项目整体介绍：

```bash
jupyter notebook notebooks/project_overview.ipynb
```

## 模型详情

### 逻辑回归 (Logistic Regression)
- 使用 scikit-learn 的 `LogisticRegression`
- 支持多分类问题
- 包含正则化参数调优

### 线性 SVM (Linear SVM)
- 使用 `LinearSVC` 实现
- 支持多分类
- 包含惩罚参数调优

### K-最近邻 (K-Nearest Neighbors)
- 使用 `KNeighborsClassifier`
- 包含邻居数量调优
- 支持不同距离度量

### 决策树 (Decision Tree)
- 使用 `DecisionTreeClassifier`
- 包含最大深度和最小样本分割调优
- 支持特征重要性分析

### 随机森林 (Random Forest)
- 使用 `RandomForestClassifier`
- 包含树数量和最大深度调优
- 支持特征重要性分析

### SGD 分类器 (SGD Classifier)
- 使用 `SGDClassifier`
- 支持不同的损失函数
- 包含学习率调度

### 感知器 (Perceptron)
- 使用 `Perceptron`
- 简单的线性分类器
- 包含惩罚参数调优

## 结果分析

### 查看结果

结果保存在 `results/` 目录中：

1. **指标文件** (`results/metrics/`)
   - 每个模型的评估指标（准确率、精确率、召回率、F1分数）
   - JSON 格式，便于程序化分析

2. **图表文件** (`results/figures/`)
   - 混淆矩阵
   - 学习曲线
   - 特征重要性图
   - 模型比较图

### 使用 Python 脚本

项目还提供了 Python 模块，可以直接在脚本中使用：

```python
from src.load_data import load_digits_data
from src.evaluate import evaluate_model
from src.visualize import plot_confusion_matrix

# 加载数据
X_train, X_test, y_train, y_test = load_digits_data()

# 训练和评估模型
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 评估模型
metrics = evaluate_model(model, X_test, y_test)
print(f"准确率: {metrics['accuracy']:.4f}")

# 可视化结果
plot_confusion_matrix(model, X_test, y_test)
```

## 开发指南

### 添加新模型

1. 在 `notebooks/` 目录中创建新的 notebook
2. 遵循现有的代码结构
3. 使用 `src/` 模块中的工具函数
4. 将结果保存到 `results/` 目录

### 代码结构

- `src/load_data.py`: 数据加载和预处理
- `src/evaluate.py`: 模型评估函数
- `src/visualize.py`: 可视化函数