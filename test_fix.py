import torch
from models.patchcore import PatchCore

# 测试模型初始化
print("测试模型初始化...")
model = PatchCore(device='cpu')
print('✓ 模型初始化成功')

# 测试特征提取器
print("\n测试特征提取器...")
print(f'特征提取器键: {list(model.feature_extractor.keys())}')

# 测试前向传播
print("\n测试前向传播...")
try:
    # 创建一个测试图像
    test_image = torch.randn(1, 3, 224, 224)
    features = model.forward(test_image)
    print('✓ 前向传播成功')
    print(f'提取的特征层: {list(features.keys())}')
    
    # 检查每个特征层的形状
    for layer_name, feature in features.items():
        print(f'{layer_name}: {feature.shape}')
        
except Exception as e:
    print(f'✗ 前向传播错误: {e}')
    import traceback
    traceback.print_exc()

# 测试数据加载器
print("\n测试数据加载器...")
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from app import DataDirDataset

# 检查数据目录是否存在
data_dir = './data/btad/test'
import os
if os.path.exists(data_dir):
    print('✓ 数据目录存在')
    
    # 创建简单的测试数据集
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = DataDirDataset(data_dir, split='train', transform=transform)
    print(f'数据集样本数量: {len(dataset.samples)}')
    
    if len(dataset.samples) > 0:
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # 测试训练过程
        print("\n测试训练过程...")
        try:
            model.fit_with_progress(dataloader, None, None, len(dataloader))
            print('✓ 训练成功')
            print(f'Memory bank形状: {model.memory_bank.shape}')
        except Exception as e:
            print(f'✗ 训练错误: {e}')
            import traceback
            traceback.print_exc()
    else:
        print('数据集为空，请检查数据目录结构')
else:
    print('数据目录不存在，请先创建数据目录')