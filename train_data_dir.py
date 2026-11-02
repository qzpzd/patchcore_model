import torch
import yaml
import os
import time
from models.patchcore import PatchCore
from utils.dataset import get_data_loader
import warnings
from PIL import Image
import torchvision.transforms as transforms
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

class DataDirDataset:
    """针对data目录结构的自定义数据集"""
    def __init__(self, data_dir, split="train", transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.samples = []
        
        # 加载数据
        self._load_samples()
    
    def _load_samples(self):
        """加载样本"""
        if self.split == "train":
            # 训练时只使用正常样本
            good_dir = os.path.join(self.data_dir, "good")
            if os.path.exists(good_dir):
                for img_file in os.listdir(good_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        img_path = os.path.join(good_dir, img_file)
                        self.samples.append((img_path, 0))  # 正常样本标签为0
        else:
            # 测试时使用所有样本
            good_dir = os.path.join(self.data_dir, "good")
            defective_dir = os.path.join(self.data_dir, "defective")
            
            if os.path.exists(good_dir):
                for img_file in os.listdir(good_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        img_path = os.path.join(good_dir, img_file)
                        self.samples.append((img_path, 0))  # 正常样本标签为0
            
            if os.path.exists(defective_dir):
                for img_file in os.listdir(defective_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        img_path = os.path.join(defective_dir, img_file)
                        self.samples.append((img_path, 1))  # 缺陷样本标签为1
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 图像转换
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

def get_data_loader_data_dir(data_dir, split="train", batch_size=32, img_size=(256, 256), shuffle=True, num_workers=0):
    """针对data目录结构的数据加载器"""
    dataset = DataDirDataset(data_dir, split=split)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader

def train_data_dir(data_dir, config_path="./configs/config.yaml"):
    """针对data目录的训练函数"""
    print("=== Starting PatchCore Training for Data Directory ===")
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
    print(f"✓ Loading data from: {data_dir}")
    
    try:
        train_loader = get_data_loader_data_dir(
            data_dir=data_dir,
            split="train",
            batch_size=2,  # 小批次
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
    model_save_path = os.path.join(save_path, "patchcore_model_data_dir.pth")
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
        return
    
    end_time = time.time()
    print(f"=== Training Completed in {end_time - start_time:.2f} seconds ===")
    print("✓ Training process finished successfully!")

if __name__ == "__main__":
    # 使用data目录下的数据进行训练
    data_dir = "./data/chilun/train"  # 注意：这里使用test目录，因为data目录下只有test数据
    
    print("Starting PatchCore training for data directory...")
    train_data_dir(data_dir)
    print("Data directory training script finished.")