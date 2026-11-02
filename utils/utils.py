import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import yaml
import os

def load_config(config_path):
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        config: 配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def visualize_anomaly_map(image, anomaly_map, threshold=0.5):
    """
    可视化异常图
    
    Args:
        image: 原始图像 (PIL Image)
        anomaly_map: 异常图 (numpy array)
        threshold: 阈值
        
    Returns:
        visualization: 可视化结果
    """
    # 转换图像为numpy数组
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # 创建热力图
    heatmap = cv2.applyColorMap(
        np.uint8(255 * anomaly_map / np.max(anomaly_map)), 
        cv2.COLORMAP_JET
    )
    
    # 调整热力图大小与原图一致
    heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    
    # 融合原图和热力图
    blended = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
    
    # 二值化异常区域
    binary_mask = (anomaly_map > threshold).astype(np.uint8) * 255
    binary_mask = cv2.resize(binary_mask, (image_np.shape[1], image_np.shape[0]))
    
    return blended, binary_mask

def calculate_metrics(predictions, labels):
    """
    计算评估指标
    
    Args:
        predictions: 预测结果
        labels: 真实标签
        
    Returns:
        metrics: 评估指标字典
    """
    # 计算准确率、精确率、召回率、F1分数
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def plot_roc_curve(y_true, y_scores):
    """
    绘制ROC曲线
    
    Args:
        y_true: 真实标签
        y_scores: 预测分数
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    return roc_auc