import torch
import yaml
import os
from models.patchcore import PatchCore
from utils.dataset import get_data_loader
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def simple_inference():
    """简单的推理测试"""
    print("=== Simple PatchCore Inference Test ===")
    
    # 加载配置
    config = load_config("./configs/config.yaml")
    
    # 设置设备
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 初始化模型
    model = PatchCore(
        backbone_name=config['model']['backbone'],
        layers=config['model']['layers'],
        patch_size=config['model']['patch_size'],
        sampling_ratio=config['model']['sampling_ratio'],
        device=device
    )
    
    # 检查模型文件是否存在
    model_path = "./results/patchcore_model_02_debug.pth"
    print(f"Looking for model at: {model_path}")
    print(f"Model file exists: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        # 加载模型权重
        print("Loading model...")
        model.load_model(model_path)
        print("Model loaded successfully.")
        print(f"Memory bank shape: {model.memory_bank.shape}")
    else:
        print("Model file not found. Please run training first.")
        return
    
    print("Inference test completed!")

if __name__ == "__main__":
    simple_inference()