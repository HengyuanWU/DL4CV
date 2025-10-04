"""
路径设置模块

这个模块专门为notebooks目录设计，仅负责设置Python路径。
导入src模块后，在notebook中直接导入需要的函数，以便IDE能正确追踪。
"""

import sys
import os


def setup_project_path():
    """
    设置项目路径，确保在notebooks目录中能正确导入src模块。
    
    这个方法使用绝对路径，确保在不同环境中都能正常工作。
    在调用此函数后，可以直接从src模块导入需要的函数。
    
    Example:
        >>> setup_project_path()
        >>> from load_data import load_mnist_dataset
        >>> from evaluate import eval_classification
    """
    # 获取当前文件所在目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 项目根目录是当前目录的父目录
    project_root = os.path.dirname(current_dir)
    
    # src目录路径
    src_path = os.path.join(project_root, 'src')
    
    # 将src目录添加到Python路径中（如果还没有添加）
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # 同时添加项目根目录，以防需要
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


# 自动执行路径设置
setup_project_path()
