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

def train(config, args):
    """训练函数"""
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
    # 如果指定了自定义数据路径，则使用它
    data_dir = args.data_dir if args.data_dir else "./data"  # 默认数据目录
    category = args.category if args.category else "bottle"  # 默认类别
    train_loader = get_data_loader(
        data_dir=data_dir,
        category=category,
        split="train",
        batch_size=config['data']['batch_size'],
        img_size=config['data']['image_size'],
        shuffle=True
    )
    
    # 检查数据加载器
    print(f"Number of batches: {len(train_loader)}")
    for i, (images, labels) in enumerate(train_loader):
        print(f"Batch {i}: images shape = {images.shape}, labels shape = {labels.shape}")
        break  # 只检查第一个批次
    
    print(f"Training data loaded with {len(train_loader.dataset)} samples.")
    
    # 训练模型（构建memory bank）
    print("Starting model training...")
    model.fit(train_loader)
    print("Model training completed.")
    
    # 保存模型
    model_save_path = os.path.join(save_path, "patchcore_model.pth")
    model.save_model(model_save_path)
    print(f"Model saved to {model_save_path}")

def main():
    parser = argparse.ArgumentParser(description='PatchCore Training')
    parser.add_argument('--config', type=str, default='./configs/config.yaml', 
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='./data', 
                        help='Path to data directory')
    parser.add_argument('--category', type=str, default='bottle', 
                        help='Category name')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 开始训练
    train(config, args)

if __name__ == "__main__":
    main()