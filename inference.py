import torch
import yaml
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from models.patchcore import PatchCore
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def load_model(model_path, config, retrieval_backend="torch"):
    """加载训练好的模型"""
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = PatchCore(
        backbone_name=config['model']['backbone'],
        layers=config['model']['layers'],
        patch_size=config['model']['patch_size'],
        sampling_ratio=config['model']['sampling_ratio'],
        device=device,
        retrieval_backend=retrieval_backend,
    )
    
    # 加载模型权重
    model.load_model(model_path)
    model.eval()
    
    return model, device

def preprocess_image(image_path, img_size=(256, 256)):
    """预处理输入图像"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    
    # 定义变换
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 应用变换
    image_tensor = transform(image).unsqueeze(0)  # 添加批次维度
    
    return image_tensor, image

def visualize_results(
    original_image,
    anomaly_map,
    anomaly_score,
    threshold=None,
    save_path=None,
):
    """可视化结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 异常图
    anomaly_map_np = anomaly_map.cpu().numpy()
    # 处理不同维度的异常图
    if len(anomaly_map_np.shape) == 3 and anomaly_map_np.shape[0] == 1:
        # 如果是三维张量且第一维为1，则squeeze掉第一维
        anomaly_map_np = anomaly_map_np.squeeze(0)
    elif len(anomaly_map_np.shape) == 1:
        # 如果是一维张量，reshape为方形
        length = anomaly_map_np.shape[0]
        size = int(np.sqrt(length))
        if size * size == length:
            anomaly_map_np = anomaly_map_np.reshape(size, size)
        else:
            # 如果无法reshape为方形，扩展为一个小的方形矩阵并填充
            padded_length = ((int(np.sqrt(length)) + 1) ** 2)
            padded_map = np.zeros(padded_length)
            padded_map[:length] = anomaly_map_np
            size = int(np.sqrt(padded_length))
            anomaly_map_np = padded_map.reshape(size, size)
    
    im1 = axes[1].imshow(anomaly_map_np, cmap='hot')
    axes[1].set_title(f'Anomaly Map (Score: {anomaly_score:.4f})')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    # 叠加图
    axes[2].imshow(original_image)
    axes[2].imshow(
        anomaly_map_np,
        cmap='hot',
        alpha=0.5,
        extent=(0, original_image.width, original_image.height, 0),
    )
    if threshold is None:
        status = "NO THRESHOLD"
    else:
        status = "ANOMALY" if anomaly_score > threshold else "NORMAL"
    axes[2].set_title(f'Overlay ({status})')
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Visualization saved to: {save_path}")
    elif plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)

def infer(
    config,
    model_path,
    image_path,
    threshold=None,
    save_path=None,
    retrieval_backend="torch",
):
    """推理函数"""
    # 加载模型
    model, device = load_model(model_path, config, retrieval_backend)
    
    # 预处理图像
    image_tensor, original_image = preprocess_image(
        image_path, 
        img_size=config['data']['image_size']
    )
    
    print(f"Input image loaded with shape: {image_tensor.shape}")
    
    # 推理
    anomaly_map, anomaly_score = model.predict(image_tensor)
    
    # 输出结果
    print(f"Anomaly score: {anomaly_score.item():.4f}")
    used_threshold = (
        threshold
        if threshold is not None
        else config.get("inference", {}).get("threshold")
    )
    if used_threshold is not None:
        result = "ANOMALY" if anomaly_score.item() > float(used_threshold) else "NORMAL"
        print(f"Threshold: {float(used_threshold):.4f}")
        print(f"Result: {result}")
    else:
        print("Threshold: not set; score and anomaly map are shown without classification")
    
    # 可视化结果
    if config.get('inference', {}).get('visualize', True):
        visualize_results(
            original_image, 
            anomaly_map[0], 
            anomaly_score.item(),
            None if used_threshold is None else float(used_threshold),
            save_path=save_path,
        )
    
    return anomaly_map, anomaly_score

def main():
    parser = argparse.ArgumentParser(description='PatchCore Inference')
    parser.add_argument('--config', type=str, default='./configs/config.yaml', 
                        help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Optional image-score threshold; overrides config inference.threshold',
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default=None,
        help='Optional output image path; otherwise opens an interactive window',
    )
    parser.add_argument(
        '--retrieval_backend',
        choices=['torch', 'faiss', 'auto'],
        default='torch',
        help='Exact KNN backend for Memory Bank retrieval',
    )
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 开始推理
    infer(
        config,
        args.model_path,
        args.image_path,
        threshold=args.threshold,
        save_path=args.save_path,
        retrieval_backend=args.retrieval_backend,
    )

if __name__ == "__main__":
    main()
