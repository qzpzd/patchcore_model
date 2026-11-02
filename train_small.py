import torch
import yaml
import os
import argparse
from models.patchcore import PatchCore
from utils.dataset import get_data_loader
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def train_custom_small_batch(data_dir, category, config_path="./configs/config.yaml"):
    """训练函数 - 使用小批次以节省内存"""
    # 加载配置
    config = load_config(config_path)
    
    # 设置设备
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建结果保存目录
    save_path = config['training']['save_path']
    os.makedirs(save_path, exist_ok=True)
    
    # 初始化模型
    model = PatchCore(
        backbone_name=config['model']['backbone'],
        layers=config['model']['layers'],
        patch_size=config['model']['patch_size'],
        sampling_ratio=config['model']['sampling_ratio'],
        device=device
    )
    
    # 获取数据加载器 - 使用小批次大小
    print(f"Loading data from: {os.path.join(data_dir, 'BTech_Dataset_transformed', category)}")
    train_loader = get_data_loader(
        data_dir=os.path.join(data_dir, "BTech_Dataset_transformed"),
        category=category,
        split="train",
        batch_size=2,  # 减小批次大小
        img_size=(256, 256),  # 使用较小的图像尺寸
        shuffle=True,
        num_workers=0  # 避免多线程问题
    )
    
    # 检查数据加载器
    print(f"Number of batches: {len(train_loader)}")
    print(f"Training data loaded with {len(train_loader.dataset)} samples.")
    
    # 显示第一个批次的信息
    for i, (images, labels) in enumerate(train_loader):
        print(f"Batch {i}: images shape = {images.shape}, labels shape = {labels.shape}")
        break  # 只检查第一个批次
    
    # 训练模型（构建memory bank）
    print("Starting model training...")
    model.fit(train_loader)
    print("Model training completed.")
    
    # 保存模型
    model_save_path = os.path.join(save_path, f"patchcore_model_{category}_small.pth")
    print(f"Attempting to save model to {model_save_path}")
    print(f"Save path exists: {os.path.exists(save_path)}")
    model.save_model(model_save_path)
    print(f"Model saved to {model_save_path}")
    # 验证模型文件是否已创建
    if os.path.exists(model_save_path):
        print("Model file successfully created!")
    else:
        print("Warning: Model file was not created!")

if __name__ == "__main__":
    # 使用您的数据路径和类别进行训练
    data_dir = "./btad"
    category = "02"
    
    print("Starting PatchCore training with small batch size...")
    train_custom_small_batch(data_dir, category)
    print("Training completed!")