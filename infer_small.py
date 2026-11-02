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

def infer_custom_small(model_path, data_dir, category, config_path="./configs/config.yaml"):
    """推理函数 - 使用小模型进行推理"""
    # 加载配置
    config = load_config(config_path)
    
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
    
    # 加载模型权重
    print(f"Loading model from: {model_path}")
    if os.path.exists(model_path):
        model.load_model(model_path)
        print("Model loaded successfully.")
    else:
        print(f"Model file not found: {model_path}")
        return
    
    # 获取测试数据加载器
    print(f"Loading test data from: {os.path.join(data_dir, 'BTech_Dataset_transformed', category)}")
    test_loader = get_data_loader(
        data_dir=os.path.join(data_dir, "BTech_Dataset_transformed"),
        category=category,
        split="test",
        batch_size=1,  # 单张图片推理
        img_size=(256, 256),
        shuffle=False,
        num_workers=0
    )
    
    # 检查测试数据加载器
    print(f"Number of test batches: {len(test_loader)}")
    print(f"Test data loaded with {len(test_loader.dataset)} samples.")
    
    # 对测试数据进行推理
    print("Starting inference...")
    model.eval()
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            print(f"Processing batch {i}: images shape = {images.shape}, labels = {labels}")
            
            # 进行推理
            anomaly_maps, anomaly_scores = model.predict(images)
            
            print(f"Anomaly maps shape: {anomaly_maps.shape}")
            print(f"Anomaly scores: {anomaly_scores}")
            
            # 只处理前几个样本作为示例
            if i >= 2:
                break
    
    print("Inference completed!")

if __name__ == "__main__":
    # 使用您的数据路径和类别进行推理
    model_path = "./results/patchcore_model_02_small.pth"
    data_dir = "./btad"
    category = "02"
    
    print("Starting PatchCore inference with small model...")
    infer_custom_small(model_path, data_dir, category)
    print("Inference completed!")