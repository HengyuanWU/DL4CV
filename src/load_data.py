# 导入必要的库
# numpy: 用于数值计算的库，提供数组操作功能
import numpy as np
# struct: 用于处理二进制数据
import struct
# os: 用于文件路径操作
import os
# sklearn.datasets: scikit-learn 的数据集模块，包含各种内置数据集
from sklearn.datasets import load_digits
# sklearn.model_selection: 模型选择模块，包含数据集分割功能
from sklearn.model_selection import train_test_split
# sklearn.preprocessing: 数据预处理模块，包含标准化等功能
from sklearn.preprocessing import StandardScaler
# typing: Python 类型提示模块，用于指定函数参数和返回值的类型
from typing import Tuple, List


def read_idx(filename: str) -> np.ndarray:
    """
    读取 IDX 格式的二进制文件。
    
    Args:
        filename: IDX 格式文件的路径
        
    Returns:
        numpy 数组，包含从文件中读取的数据
    """
    with open(filename, 'rb') as f:
        # 读取 magic number (4 bytes)
        magic = struct.unpack(">I", f.read(4))[0]
        
        # 检查 magic number 来确定数据类型和维度
        if magic == 0x803:  # 图像文件 (3D: 样本数 x 行数 x 列数)
            num_items = struct.unpack(">I", f.read(4))[0]
            num_rows = struct.unpack(">I", f.read(4))[0]
            num_cols = struct.unpack(">I", f.read(4))[0]
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items, num_rows, num_cols)
        elif magic == 0x801:  # 标签文件 (1D: 样本数)
            num_items = struct.unpack(">I", f.read(4))[0]
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items)
        else:
            raise ValueError(f"Unsupported magic number: {magic:#x} in file {filename}")
    
    return data


def load_mnist_dataset(data_dir: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    加载 MNIST 数据集并返回训练/测试分割。
    
    Args:
        data_dir: 包含 MNIST 数据文件的目录路径，如果为 None 则自动查找
        
    Returns:
        包含 (X_train, X_test, y_train, y_test, target_names) 的元组
        其中 X_train 和 X_test 是标准化的 numpy 数组，
        y_train 和 y_test 是 numpy 数组，
        target_names 是类别名称字符串列表。
    """
    # 自动查找数据目录
    if data_dir is None:
        # 尝试从当前目录开始查找
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)  # src 的父目录是项目根目录
        data_dir = os.path.join(project_root, 'data')
    
    # 构建文件路径
    train_images_path = os.path.join(data_dir, 'train-images.idx3-ubyte')
    train_labels_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')
    test_images_path = os.path.join(data_dir, 't10k-images.idx3-ubyte')
    test_labels_path = os.path.join(data_dir, 't10k-labels.idx1-ubyte')
    
    # 检查文件是否存在
    for path in [train_images_path, train_labels_path, test_images_path, test_labels_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"MNIST data file not found: {path}")
    
    # 读取 MNIST 数据集
    print("正在加载 MNIST 数据集...")
    train_images = read_idx(train_images_path)
    train_labels = read_idx(train_labels_path)
    test_images = read_idx(test_images_path)
    test_labels = read_idx(test_labels_path)
    
    # 打印数据集信息
    print(f"训练集图像形状: {train_images.shape}")
    print(f"训练集标签形状: {train_labels.shape}")
    print(f"测试集图像形状: {test_images.shape}")
    print(f"测试集标签形状: {test_labels.shape}")
    
    # 将图像数据展平为特征向量 (28x28 -> 784)
    X_train = train_images.reshape(train_images.shape[0], -1)
    X_test = test_images.reshape(test_images.shape[0], -1)
    
    # 标签数据保持不变
    y_train = train_labels
    y_test = test_labels
    
    # 类别名称 (0-9)
    target_names = [str(i) for i in range(10)]
    
    # 创建标准化器对象，用于将特征数据标准化（均值为0，标准差为1）
    scaler = StandardScaler()
    # 对训练集进行拟合和转换：计算均值和标准差，然后标准化数据
    X_train = scaler.fit_transform(X_train.astype(np.float64))
    # 对测试集进行转换：使用训练集计算出的均值和标准差来标准化测试集
    # 注意：测试集不能重新拟合，必须使用训练集的参数
    X_test = scaler.transform(X_test.astype(np.float64))
    
    print("MNIST 数据集加载完成！")
    return X_train, X_test, y_train, y_test, target_names


def load_digits_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    加载数字数据集并返回标准化特征后的训练/测试分割。
    
    Returns:
        包含 (X_train, X_test, y_train, y_test, target_names) 的元组
        其中 X_train 和 X_test 是标准化的 numpy 数组，
        y_train 和 y_test 是 numpy 数组，
        target_names 是类别名称字符串列表。
    """
    # 加载 scikit-learn 内置的数字数据集
    # 这个数据集包含 0-9 的手写数字图像，每个图像是 8x8 像素
    digits = load_digits()
    # 提取特征数据（图像像素值）和标签数据（数字类别）
    X, y = digits.data, digits.target
    # 将类别名称转换为字符串列表，便于后续使用
    target_names = [str(name) for name in digits.target_names]
    
    # 将数据集分割为训练集和测试集
    # test_size=0.2: 20% 的数据作为测试集，80% 作为训练集
    # random_state=42: 设置随机种子，确保每次分割结果相同
    # stratify=y: 按标签分层抽样，确保训练集和测试集中各类别比例相同
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 创建标准化器对象，用于将特征数据标准化（均值为0，标准差为1）
    scaler = StandardScaler()
    # 对训练集进行拟合和转换：计算均值和标准差，然后标准化数据
    X_train = scaler.fit_transform(X_train)
    # 对测试集进行转换：使用训练集计算出的均值和标准差来标准化测试集
    # 注意：测试集不能重新拟合，必须使用训练集的参数
    X_test = scaler.transform(X_test)
    
    # 返回处理好的训练集、测试集和类别名称
    return X_train, X_test, y_train, y_test, target_names
