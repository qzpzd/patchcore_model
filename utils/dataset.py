import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from torchvision import transforms

class MVTecDataset(Dataset):
    def __init__(self, root_dir, category, split="train", transform=None, img_size=(256, 256)):
        """
        MVTec AD数据集加载器
        
        Args:
            root_dir: 数据集根目录
            category: 类别名称
            split: 数据集划分 ("train" 或 "test")
            transform: 图像变换
            img_size: 图像尺寸
        """
        self.root_dir = root_dir
        self.category = category
        self.split = split
        self.transform = transform
        self.img_size = img_size
        
        # 构建数据路径
        if split == "train":
            self.data_dir = os.path.join(root_dir, category, "train", "ok")  # 修改为"ok"而不是"good"
        else:
            self.data_dir = os.path.join(root_dir, category, "test")
            
        print(f"Initializing dataset with root_dir={root_dir}, category={category}, split={split}")
        print(f"Data directory: {self.data_dir}")
        print(f"Data directory exists: {os.path.exists(self.data_dir)}")
            
        # 获取图像路径
        self.image_paths = []
        self.labels = []
        
        if split == "train":
            # 训练集只包含正常样本
            print("Loading training data...")
            if os.path.exists(self.data_dir):
                for filename in os.listdir(self.data_dir):
                    if filename.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(self.data_dir, filename))
                        self.labels.append(0)  # 正常样本标签为0
                print(f"Found {len(self.image_paths)} training images")
            else:
                print(f"Training data directory does not exist: {self.data_dir}")
        else:
            # 测试集包含正常和异常样本
            print("Loading test data...")
            if os.path.exists(self.data_dir):
                for defect_type in os.listdir(self.data_dir):
                    defect_dir = os.path.join(self.data_dir, defect_type)
                    if not os.path.isdir(defect_dir):
                        continue
                        
                    label = 0 if defect_type == "good" else 1
                    
                    for filename in os.listdir(defect_dir):
                        if filename.endswith(('.png', '.jpg', '.jpeg')):
                            self.image_paths.append(os.path.join(defect_dir, filename))
                            self.labels.append(label)
                print(f"Found {len(self.image_paths)} test images")
            else:
                print(f"Test data directory does not exist: {self.data_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载图像
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # 调整图像大小
        image = image.resize(self.img_size)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        else:
            # 默认变换
            image = transforms.ToTensor()(image)
            
        label = self.labels[idx]
        
        return image, label

def get_data_loader(data_dir, category, split="train", batch_size=32, img_size=(256, 256), shuffle=True, num_workers=0):
    """
    获取数据加载器
    
    Args:
        data_dir: 数据集目录
        category: 类别
        split: 数据划分
        batch_size: 批次大小
        img_size: 图像尺寸
        shuffle: 是否打乱数据
        num_workers: 数据加载器的工作进程数
        
    Returns:
        DataLoader对象
    """
    # 定义图像变换
    if split == "train":
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    # 创建数据集
    dataset = MVTecDataset(
        root_dir=data_dir,
        category=category,
        split=split,
        transform=transform,
        img_size=img_size
    )
    
    # 创建数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,  # 使用传入的num_workers参数
        pin_memory=torch.cuda.is_available()  # 仅在GPU可用时启用pin_memory
    )
    
    return data_loader