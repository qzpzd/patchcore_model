import torch
import yaml
import os
from models.patchcore import PatchCore
from utils.dataset import get_data_loader
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def visualize_anomaly_map(image, anomaly_map, score, label, save_path=None):
    """可视化异常检测结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原图
    img = image.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 异常热力图
    anomaly_map_np = anomaly_map.cpu().numpy()
    im1 = axes[1].imshow(anomaly_map_np, cmap='jet')
    axes[1].set_title(f'Anomaly Map (Score: {score:.4f})')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    # 叠加图
    axes[2].imshow(img)
    axes[2].imshow(anomaly_map_np, cmap='jet', alpha=0.5)
    axes[2].set_title(f'Overlay (Label: {label})')
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close()

def infer_custom_complete(model_path, data_dir, category, config_path="./configs/config.yaml", num_samples=5):
    """完整的推理函数"""
    print("=== Starting PatchCore Inference ===")
    
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
    
    # 创建结果目录
    results_dir = "./inference_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 对测试数据进行推理
    print("Starting inference...")
    model.eval()
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            print(f"\nProcessing sample {i+1}/{len(test_loader)}:")
            print(f"  Image shape: {images.shape}")
            print(f"  Label: {labels.item()}")
            
            # 进行推理
            anomaly_maps, anomaly_scores = model.predict(images)
            
            print(f"  Anomaly map shape: {anomaly_maps.shape}")
            print(f"  Anomaly score: {anomaly_scores.item():.4f}")
            
            # 可视化结果
            save_path = os.path.join(results_dir, f"inference_result_{i+1}.png")
            visualize_anomaly_map(
                images[0].cpu(),
                anomaly_maps[0],
                anomaly_scores.item(),
                labels.item(),
                save_path=save_path
            )
            
            # 处理指定数量的样本
            if i+1 >= num_samples:
                break
    
    print(f"\nInference completed! Results saved to {results_dir}")

if __name__ == "__main__":
    # 使用您的数据路径和类别进行推理
    model_path = "./results/patchcore_model_02_detailed.pth"
    data_dir = "./btad"
    category = "02"
    
    print("Starting PatchCore inference with complete visualization...")
    infer_custom_complete(model_path, data_dir, category, num_samples=3)
    print("Inference completed!")