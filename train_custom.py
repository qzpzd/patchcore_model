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

def train_custom(data_dir, category, config_path="./configs/config.yaml"):
    """训练函数"""
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
    
    # 获取数据加载器
    print(f"Loading data from: {os.path.join(data_dir, 'BTech_Dataset_transformed', category)}")
    train_loader = get_data_loader(
        data_dir=os.path.join(data_dir, "BTech_Dataset_transformed"),
        category=category,
        split="train",
        batch_size=config['data']['batch_size'],
        img_size=config['data']['image_size'],
        shuffle=True
    )
    
    # 检查数据加载器
    print(f"Number of batches: {len(train_loader)}")
    print(f"Training data loaded with {len(train_loader.dataset)} samples.")
    
    # 训练模型（构建memory bank）
    print("Starting model training...")
    model.fit(train_loader)
    print("Model training completed.")
    
    # 保存模型
    model_save_path = os.path.join(save_path, f"patchcore_model_{category}.pth")
    model.save_model(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    # 使用您的数据路径和类别进行训练
    data_dir = "./data/btad"
    category = "02"
    
    print("Starting PatchCore training...")
    train_custom(data_dir, category)
    print("Training completed!")