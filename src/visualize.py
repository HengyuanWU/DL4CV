# 导入必要的库
# matplotlib.pyplot: Python 的主要绘图库，用于创建各种图表
import matplotlib.pyplot as plt
# numpy: 用于数值计算的库，提供数组操作功能
import numpy as np
# seaborn: 基于 matplotlib 的统计数据可视化库，提供更美观的图表样式
import seaborn as sns
# sklearn.metrics: scikit-learn 的评估指标模块，包含混淆矩阵等功能
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm: np.ndarray, class_names: list, out_png_path: str = None):
    """
    绘制混淆矩阵，可选保存到文件，返回figure对象供notebook显示。
    
    Args:
        cm: 混淆矩阵，numpy 数组
        class_names: 类别名称列表
        out_png_path: PNG 文件的输出路径（可选，如果提供则保存文件）
    
    Returns:
        matplotlib.figure.Figure: 创建的图形对象
    """
    # 创建一个新的图形和轴对象，设置图形大小为 8x6 英寸
    fig, ax = plt.subplots(figsize=(8, 6))
    # 使用 seaborn 绘制热力图来显示混淆矩阵
    sns.heatmap(
        cm,                    # 混淆矩阵数据
        annot=True,            # 在每个单元格中显示数值
        fmt='d',               # 数值格式为整数
        cmap='Blues',          # 使用蓝色调色板
        xticklabels=class_names,  # x 轴标签使用类别名称
        yticklabels=class_names,  # y 轴标签使用类别名称
        ax=ax                  # 明确指定在哪个axes上绘图
    )
    # 设置图表标题
    ax.set_title('Confusion Matrix')
    # 设置 x 轴标签（预测标签）
    ax.set_xlabel('Predicted Label')
    # 设置 y 轴标签（真实标签）
    ax.set_ylabel('True Label')
    # 自动调整子图参数，使图表元素不重叠
    plt.tight_layout()
    # 如果提供了输出路径，则保存图表到指定路径
    if out_png_path:
        fig.savefig(out_png_path)
    # 返回图形对象（不关闭，让notebook可以显示）
    return fig


def plot_confusion_matrix_detailed(cm: np.ndarray, class_names: list, out_png_path: str = None):
    """
    绘制混淆矩阵（使用 matplotlib 原生实现，带数值标注），可选保存到文件，返回figure对象供notebook显示。
    
    Args:
        cm: 混淆矩阵，numpy 数组
        class_names: 类别名称列表
        out_png_path: PNG 文件的输出路径（可选，如果提供则保存文件）
    
    Returns:
        matplotlib.figure.Figure: 创建的图形对象
    """
    # 创建一个新的图形，设置图形大小为 6x5 英寸
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # 使用 imshow 显示混淆矩阵
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    
    # 添加颜色条
    ax.figure.colorbar(im, ax=ax)
    
    # 设置坐标轴
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel='Predicted label',
        ylabel='True label',
        title='Confusion Matrix'
    )
    
    # 旋转 x 轴标签以避免重叠
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 在每个格子中显示数值
    # 计算阈值用于确定文字颜色（深色背景用白色文字，浅色背景用黑色文字）
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    # 自动调整子图参数，使图表元素不重叠
    plt.tight_layout()
    # 如果提供了输出路径，则保存图表到指定路径
    if out_png_path:
        plt.savefig(out_png_path, dpi=100)
    # 返回图形对象（不关闭，让notebook可以显示）
    return fig


def plot_bar(values: list, labels: list, title: str, x_label: str, out_png_path: str) -> None:
    """
    绘制并保存柱状图。
    
    Args:
        values: 柱子的值列表
        labels: x 轴标签列表
        title: 图表标题
        x_label: x 轴标签
        out_png_path: PNG 文件的输出路径（必须在 results/figures/ 目录中）
    """
    # 创建一个新的图形，设置图形大小为 10x6 英寸
    plt.figure(figsize=(10, 6))
    # 绘制柱状图，x 轴为 labels，y 轴为 values
    plt.bar(labels, values)
    # 设置图表标题
    plt.title(title)
    # 设置 x 轴标签
    plt.xlabel(x_label)
    # 设置 y 轴标签
    plt.ylabel('Value')
    # 将 x 轴标签旋转 45 度，避免长标签重叠
    plt.xticks(rotation=45)
    # 自动调整子图参数，使图表元素不重叠
    plt.tight_layout()
    # 保存图表到指定路径
    plt.savefig(out_png_path)
    # 关闭图形，释放内存
    plt.close()


def plot_random_samples(X: np.ndarray, y: np.ndarray, n_show: int = 8, random_state: int = 42) -> None:
    """
    显示随机样本图像。
    
    Args:
        X: 特征数据，numpy 数组
        y: 标签数据，numpy 数组
        n_show: 一行展示多少张图像，默认为8
        random_state: 随机种子，默认为42
    """
    from math import sqrt
    
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X), size=n_show, replace=False)

    # 尝试推断方形尺寸
    n_features = X.shape[1]
    side = int(round(sqrt(n_features)))
    is_square = (side * side == n_features)

    fig, axes = plt.subplots(1, n_show, figsize=(2.0 * n_show, 2.4))
    for ax, i in zip(axes, idx):
        ax.set_axis_off()
        if is_square:
            img = X[i].reshape(side, side)
            ax.imshow(img, cmap='gray')
            ax.set_title(f"label: {y[i]}")
        else:
            ax.text(0.5, 0.5, "非图像特征，无法还原为方形", ha='center', va='center')
    plt.suptitle("Random samples")
    plt.tight_layout()
    plt.show()


def plot_validation_curves(cv_results: dict, best_params: dict, out_dir: str = None, notebook_basename: str = None):
    """
    根据 GridSearchCV 的结果绘制验证曲线（每个超参数单独一张图），返回所有figure对象供notebook显示。
    
    Args:
        cv_results: GridSearchCV.cv_results_ 字典
        best_params: GridSearchCV.best_params_ 字典
        out_dir: 输出目录路径（可选，如果提供则保存文件）
        notebook_basename: notebook 的基础名称，用于生成文件名（保存文件时需要）
    
    Returns:
        dict: 键为参数名，值为对应的matplotlib.figure.Figure对象
    """
    from collections import defaultdict
    
    params_list = cv_results["params"]
    mean_test = cv_results["mean_test_score"]
    best = best_params
    
    def mask_close_to_best(keys_except):
        """返回与 best_params 在除 keys_except 外的键上都匹配的布尔掩码"""
        m = np.ones(len(params_list), dtype=bool)
        for i, pr in enumerate(params_list):
            for k, v in best.items():
                if k in keys_except:
                    continue
                if pr.get(k) != v:
                    m[i] = False
                    break
        return m
    
    all_keys = sorted(best.keys())
    figures = {}
    
    for key in all_keys:
        mask = mask_close_to_best(keys_except=[key])
        idx = np.where(mask)[0]
        if len(idx) == 0:
            # 如果没有完全匹配的组合，就放宽约束，只对 key 做聚合平均
            values = defaultdict(list)
            for i, pr in enumerate(params_list):
                values[pr[key]].append(mean_test[i])
            xs = list(values.keys())
            ys = [np.mean(values[x]) for x in xs]
        else:
            xs = [params_list[i][key] for i in idx]
            ys = [mean_test[i] for i in idx]
        
        # 画图：数值型 vs 类别型分别处理
        fig, ax = plt.subplots(figsize=(5.5, 4))
        if isinstance(xs[0], (int, float, np.floating)):
            # 数值参数（例如 C）
            order = np.argsort(xs)
            xs_sorted = np.array(xs)[order]
            ys_sorted = np.array(ys)[order]
            ax.plot(xs_sorted, ys_sorted, marker='o')
            if key.lower() == 'c':
                ax.set_xscale('log')
            ax.set_xlabel(key)
            ax.set_ylabel('CV mean test score')
            ax.set_title(f'Validation curve for {key}')
        else:
            # 类别参数（例如 penalty, solver, class_weight, fit_intercept）
            uniq = list(dict.fromkeys(xs))  # 保持出现顺序
            x_idx = np.arange(len(uniq))
            # 同类名可能重复（不同组合下），对相同类别取均值
            val_map = defaultdict(list)
            for x, y in zip(xs, ys):
                val_map[x].append(y)
            y_means = [np.mean(val_map[u]) for u in uniq]
            ax.plot(x_idx, y_means, marker='o')
            ax.set_xticks(x_idx)
            ax.set_xticklabels([str(u) for u in uniq], rotation=15)
            ax.set_xlabel(key)
            ax.set_ylabel('CV mean test score')
            ax.set_title(f'Validation curve for {key}')
        
        plt.tight_layout()
        
        # 如果提供了输出路径，则保存图表
        if out_dir and notebook_basename:
            out_path = f"{out_dir}/{notebook_basename}__acc_vs_{key}.png"
            plt.savefig(out_path)
            print(f"Validation curve for {key} saved to {out_path}")
        
        # 保存figure对象到字典中（不关闭，让notebook可以显示）
        figures[key] = fig
    
    return figures