# 导入必要的库
# numpy: 用于数值计算的库，提供数组操作功能
import numpy as np
# sklearn.metrics: scikit-learn 的评估指标模块，包含各种分类评估指标
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix, roc_auc_score
)
# sklearn.preprocessing: 数据预处理模块，用于标签二值化
from sklearn.preprocessing import label_binarize
# typing: Python 类型提示模块，用于指定函数参数和返回值的类型
from typing import Dict, Any, Optional


def eval_classification(model, X_test: np.ndarray, y_test: np.ndarray, target_names: list) -> Dict[str, Any]:
    """
    评估分类模型并返回性能指标。
    
    Args:
        model: 训练好的分类模型，具有 predict 方法
        X_test: 测试特征，numpy 数组
        y_test: 测试标签，numpy 数组
        target_names: 类别名称列表
        
    Returns:
        包含固定键的字典：
        - accuracy (float) 准确率
        - macro_f1 (float) 宏平均 F1 分数
        - weighted_f1 (float) 加权平均 F1 分数
        - classification_report (str) 分类报告
        - confusion_matrix (np.ndarray) 混淆矩阵
    """
    # 使用训练好的模型对测试数据进行预测
    # model.predict(X_test) 返回预测的类别标签
    y_pred = model.predict(X_test)
    
    # 计算准确率：正确预测的样本数占总样本数的比例
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    # 计算宏平均 F1 分数：对每个类别的 F1 分数取平均值，不考虑类别不平衡
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    # 计算加权平均 F1 分数：根据每个类别的样本数量加权计算 F1 分数
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # 生成详细的分类报告，包含精确率、召回率、F1 分数等指标
    class_report_str = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    # 生成混淆矩阵，显示真实标签与预测标签的对应关系
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # 返回包含所有评估指标的字典
    # 这个字典有固定的键名，便于后续使用
    return {
        'accuracy': accuracy,           # 准确率
        'precision_score': precision,   # 宏平均精确率
        'recall_score': recall,         # 宏平均召回率
        'macro_f1': macro_f1,           # 宏平均 F1 分数
        'weighted_f1': weighted_f1,     # 加权平均 F1 分数
        'classification_report': class_report_str,  # 分类报告字符串
        'confusion_matrix': conf_matrix  # 混淆矩阵数组
    }


def eval_classification_with_roc(
    model, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    target_names: list,
    print_report: bool = True
) -> Dict[str, Any]:
    """
    评估分类模型并返回增强的性能指标（包含 ROC-AUC）。
    
    Args:
        model: 训练好的分类模型，具有 predict 方法，最好有 predict_proba 或 decision_function
        X_test: 测试特征，numpy 数组
        y_test: 测试标签，numpy 数组
        target_names: 类别名称列表
        print_report: 是否打印分类报告和指标，默认为 True
        
    Returns:
        包含以下键的字典：
        - accuracy (float) 准确率
        - precision_macro (float) 宏平均精确率
        - recall_macro (float) 宏平均召回率
        - f1_macro (float) 宏平均 F1 分数
        - weighted_f1 (float) 加权平均 F1 分数
        - roc_auc (float or str) ROC-AUC 分数（二分类）或错误信息
        - roc_auc_macro_ovr (float or str) 多分类 OvR 宏平均 ROC-AUC 或错误信息
        - classification_report (str) 分类报告
        - confusion_matrix (np.ndarray) 混淆矩阵
        - y_pred (np.ndarray) 预测标签
        - y_score (np.ndarray or None) 预测概率/决策分数
    """
    # 获取预测结果
    y_pred = model.predict(X_test)
    
    # 计算基础指标
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
    }
    
    # ---- ROC-AUC（自动兼容二/多分类）-----
    classes = np.unique(y_test)
    y_score = None
    
    try:
        if len(classes) == 2:
            # 二分类情况
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(X_test)
            else:
                y_score = None
            
            if y_score is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, y_score)
            else:
                metrics['roc_auc'] = 'Model does not support probability/decision function'
        else:
            # 多分类情况，使用 OvR 宏平均
            if hasattr(model, "predict_proba") or hasattr(model, "decision_function"):
                y_bin = label_binarize(y_test, classes=classes)
                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(X_test)
                else:
                    y_score = model.decision_function(X_test)
                
                metrics['roc_auc_macro_ovr'] = roc_auc_score(
                    y_bin, y_score, average='macro', multi_class='ovr'
                )
            else:
                metrics['roc_auc_macro_ovr'] = 'Model does not support probability/decision function'
    except Exception as e:
        # AUC 计算失败不中断主流程
        if len(classes) == 2:
            metrics['roc_auc'] = f'Error: {str(e)}'
        else:
            metrics['roc_auc_macro_ovr'] = f'Error: {str(e)}'
    
    # 生成分类报告和混淆矩阵
    metrics['classification_report'] = classification_report(
        y_test, y_pred, target_names=target_names, zero_division=0
    )
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
    metrics['y_pred'] = y_pred
    metrics['y_score'] = y_score
    
    # 打印指标（如果需要）
    if print_report:
        print("=" * 50)
        print("== Evaluation Metrics ==")
        print("=" * 50)
        for k, v in metrics.items():
            if k not in ['classification_report', 'confusion_matrix', 'y_pred', 'y_score']:
                if isinstance(v, (int, float, np.floating)):
                    print(f"{k:25s}: {v:.4f}")
                else:
                    print(f"{k:25s}: {v}")
        
        print("" + "=" * 50)
        print("== Classification Report ==")
        print("=" * 50)
        print(metrics['classification_report'])
    
    return metrics
