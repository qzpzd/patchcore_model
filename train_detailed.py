import torch
import yaml
import os
import time
import sys
from models.patchcore import PatchCore
from utils.dataset import get_data_loader
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def train_detailed(data_dir, category, config_path="./configs/config.yaml"):
    """详细训练函数 - 带有完整的错误处理和日志"""
    print("=== Starting Detailed PatchCore Training ===")
    start_time = time.time()
    
    # 加载配置
    config = load_config(config_path)
    if config is None:
        print("Failed to load config file")
        return
    print(f"✓ Loaded config successfully")
    
    # 设置设备
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    # 创建结果保存目录
    save_path = config['training']['save_path']
    print(f"✓ Save path: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    print(f"✓ Created save directory: {os.path.exists(save_path)}")
    
    # 初始化模型
    print("Initializing PatchCore model...")
    try:
        model = PatchCore(
            backbone_name=config['model']['backbone'],
            layers=config['model']['layers'],
            patch_size=config['model']['patch_size'],
            sampling_ratio=config['model']['sampling_ratio'],
            device=device
        )
        print("✓ Model initialized successfully.")
    except Exception as e:
        print(f"✗ Error initializing model: {e}")
        return
    
    # 获取数据加载器
    data_path = os.path.join(data_dir, "BTech_Dataset_transformed")
    print(f"✓ Loading data from: {data_path}")
    print(f"✓ Category: {category}")
    
    try:
        train_loader = get_data_loader(
            data_dir=data_path,
            category=category,
            split="train",
            batch_size=2,
            img_size=(256, 256),
            shuffle=True,
            num_workers=0
        )
        print(f"✓ Number of batches: {len(train_loader)}")
        print(f"✓ Training data loaded with {len(train_loader.dataset)} samples.")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # 显示第一个批次的信息
    print("Checking first batch...")
    try:
        for i, (images, labels) in enumerate(train_loader):
            print(f"✓ Batch {i}: images shape = {images.shape}, labels shape = {labels.shape}")
            print(f"✓ Image values range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"✓ Labels: {labels}")
            break
    except Exception as e:
        print(f"✗ Error checking batch: {e}")
        return
    
    # 训练模型
    print("Starting model training...")
    training_start_time = time.time()
    
    try:
        model.fit(train_loader)
        training_end_time = time.time()
        print(f"✓ Model training completed in {training_end_time - training_start_time:.2f} seconds.")
    except Exception as e:
        print(f"✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 显示模型信息
    if model.memory_bank is not None:
        print(f"✓ Memory bank shape: {model.memory_bank.shape}")
    else:
        print("✗ Memory bank is None!")
        return
    
    # 保存模型
    model_save_path = os.path.join(save_path, f"patchcore_model_{category}_detailed.pth")
    print(f"✓ Attempting to save model to {model_save_path}")
    
    try:
        model.save_model(model_save_path)
        print("✓ Model save method called successfully.")
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        return
    
    # 验证模型文件是否已创建
    if os.path.exists(model_save_path):
        file_size = os.path.getsize(model_save_path)
        print(f"✓ Model file successfully created! Size: {file_size} bytes")
    else:
        print("✗ Model file was not created!")
        if os.path.exists(save_path):
            print(f"Contents of {save_path}:")
            for item in os.listdir(save_path):
                print(f"  - {item}")
        return
    
    end_time = time.time()
    print(f"=== Training Detailed Completed in {end_time - start_time:.2f} seconds ===")
    print("✓ Training process finished successfully!")

if __name__ == "__main__":
    data_dir = "./btad"
    category = "02"
    
    print("Starting PatchCore detailed training...")
    train_detailed(data_dir, category)
    print("Detailed training script finished.")