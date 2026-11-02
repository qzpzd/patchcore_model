import torch
import yaml
import os
import time
from models.patchcore import PatchCore
from utils.dataset import get_data_loader
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def train_enhanced(data_dir, category, config_path="./configs/config.yaml"):
    """增强版训练函数 - 带有详细日志信息"""
    print("=== Starting Enhanced PatchCore Training ===")
    start_time = time.time()
    
    # 加载配置
    config = load_config(config_path)
    print(f"Loaded config: {config}")
    
    # 设置设备
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建结果保存目录
    save_path = config['training']['save_path']
    print(f"Save path: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    print(f"Created save directory: {os.path.exists(save_path)}")
    
    # 初始化模型
    print("Initializing PatchCore model...")
    model = PatchCore(
        backbone_name=config['model']['backbone'],
        layers=config['model']['layers'],
        patch_size=config['model']['patch_size'],
        sampling_ratio=config['model']['sampling_ratio'],
        device=device
    )
    print("Model initialized successfully.")
    
    # 获取数据加载器 - 使用小批次大小
    data_path = os.path.join(data_dir, "BTech_Dataset_transformed")
    print(f"Loading data from: {data_path}")
    print(f"Category: {category}")
    
    train_loader = get_data_loader(
        data_dir=data_path,
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
    print("Checking first batch...")
    for i, (images, labels) in enumerate(train_loader):
        print(f"Batch {i}: images shape = {images.shape}, labels shape = {labels.shape}")
        print(f"Image values range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Labels: {labels}")
        break  # 只检查第一个批次
    
    # 训练模型（构建memory bank）
    print("Starting model training...")
    training_start_time = time.time()
    model.fit(train_loader)
    training_end_time = time.time()
    print(f"Model training completed in {training_end_time - training_start_time:.2f} seconds.")
    
    # 显示模型信息
    print(f"Memory bank shape: {model.memory_bank.shape}")
    
    # 保存模型
    model_save_path = os.path.join(save_path, f"patchcore_model_{category}_enhanced.pth")
    print(f"Attempting to save model to {model_save_path}")
    print(f"Save path exists: {os.path.exists(save_path)}")
    
    model.save_model(model_save_path)
    print(f"Model save method called.")
    
    # 验证模型文件是否已创建
    if os.path.exists(model_save_path):
        print("Model file successfully created!")
        # 检查文件大小
        file_size = os.path.getsize(model_save_path)
        print(f"Model file size: {file_size} bytes")
    else:
        print("Warning: Model file was not created!")
        # 列出results目录内容
        if os.path.exists(save_path):
            print(f"Contents of {save_path}:")
            for item in os.listdir(save_path):
                print(f"  - {item}")
    
    end_time = time.time()
    print(f"=== Training Enhanced Completed in {end_time - start_time:.2f} seconds ===")

if __name__ == "__main__":
    # 使用您的数据路径和类别进行训练
    data_dir = "./btad"
    category = "02"
    
    print("Starting PatchCore enhanced training with detailed logging...")
    train_enhanced(data_dir, category)
    print("Enhanced training completed!")