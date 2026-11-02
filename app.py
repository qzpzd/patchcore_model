import streamlit as st
import os
import yaml
import torch
import time
from pathlib import Path
import sys

# 添加项目路径到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.patchcore import PatchCore
from utils.utils import load_config
import torch
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
import os

# 自定义数据集类，适配我们的数据结构
class DataDirDataset:
    def __init__(self, data_dir, split="test", transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.samples = []
        
        # 加载数据
        self._load_samples()
    
    def _load_samples(self):
        if self.split == "train":
            # 训练时只使用正常样本
            good_dir = os.path.join(self.data_dir, "good")
            if os.path.exists(good_dir):
                for img_file in os.listdir(good_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        img_path = os.path.join(good_dir, img_file)
                        self.samples.append((img_path, 0, "good"))  # 正常样本标签为0
        else:
            # 测试时使用所有样本
            good_dir = os.path.join(self.data_dir, "good")
            defective_dir = os.path.join(self.data_dir, "defective")
            
            if os.path.exists(good_dir):
                for img_file in os.listdir(good_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        img_path = os.path.join(good_dir, img_file)
                        self.samples.append((img_path, 0, "good"))  # 正常样本标签为0
            
            if os.path.exists(defective_dir):
                for img_file in os.listdir(defective_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        img_path = os.path.join(defective_dir, img_file)
                        self.samples.append((img_path, 1, "defective"))  # 缺陷样本标签为1
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, sample_type = self.samples[idx]
        
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

def get_data_loader(data_dir, split="train", batch_size=1, img_size=(256, 256), shuffle=False, num_workers=0):
    dataset = DataDirDataset(data_dir, split=split)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def select_eval_samples(dataset, num_samples, balance=True):
    """
    从数据集中选取评估样本。
    balance=True 时按 good/defective 分层抽取，避免只抽到前面的 good 导致指标全 0。
    """
    good_idx = [i for i, s in enumerate(dataset.samples) if s[1] == 0]
    def_idx = [i for i, s in enumerate(dataset.samples) if s[1] == 1]

    if not balance or (not good_idx) or (not def_idx):
        return list(range(min(num_samples, len(dataset.samples))))

    half = max(1, num_samples // 2)
    n_def = min(len(def_idx), half)
    n_good = min(len(good_idx), num_samples - n_def)
    # 若正常样本不足，用缺陷补齐
    if n_good + n_def < num_samples:
        n_def = min(len(def_idx), num_samples - n_good)

    chosen = good_idx[:n_good] + def_idx[:n_def]
    return chosen


def find_f1_optimal_threshold(y_true, y_score):
    """
    Anomalib 默认自适应阈值：在带标签集合上搜索使 F1 最大的阈值。
    判定规则：score > threshold => 异常(1)，否则正常(0)。
    """
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        # 只有单一类别时无法做 F1 最优，退回分位数
        return float(np.percentile(y_score, 95)), None

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision/recall 比 thresholds 多 1 个点
    if len(thresholds) == 0:
        return float(np.median(y_score)), None

    f1 = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best = int(np.argmax(f1))
    return float(thresholds[best]), {
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "f1": float(f1[best]),
    }


def normal_score_percentiles(normal_scores):
    """仅用正常样本分数计算分位数阈值参考。"""
    import numpy as np

    arr = np.asarray(normal_scores, dtype=float)
    if arr.size == 0:
        return {"p90": None, "p95": None, "p99": None}
    return {
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def find_normal_percentile_threshold(normal_scores, percentile=95):
    """
    部署常用：只用正常样本分数的分位数作阈值。
    例如 P95：约 5% 的正常样本会被判异常（在定阈值的正常集上）。
    """
    stats = normal_score_percentiles(normal_scores)
    key = f"p{int(percentile)}"
    if key not in stats or stats[key] is None:
        raise ValueError("正常样本为空，无法按分位数定阈值。")
    return float(stats[key]), stats


THRESHOLD_MODE_LABELS = {
    "f1_optimal": "F1最优（Anomalib 官网自适应阈值）",
    "normal_p95": "正常样本 P95（部署常用）",
    "normal_p99": "正常样本 P99（更严、更少误报）",
    "manual": "手动阈值",
}

CALIB_SOURCE_LABELS = {
    "train": "训练集 good（默认，推荐部署）",
    "calib": "单独校准集（仅正常样本）",
    "eval": "本次评估集正常样本（仅对照）",
}


def guess_train_dir_from_infer(infer_dir: str) -> str:
    """由推理目录猜测训练集：.../test -> .../train。"""
    p = Path(infer_dir)
    if p.name.lower() == "test":
        cand = p.parent / "train"
        if cand.exists():
            return str(cand)
    train_state = st.session_state.get("train_page_state") or {}
    return train_state.get("data_dir", "./data/btad/train")


def list_normal_image_paths(data_dir: str):
    """
    收集正常图路径：
    - data_dir/good/* 优先
    - 否则 data_dir 下直接放的图片
    """
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    root = Path(data_dir)
    good = root / "good"
    search = good if good.is_dir() else root
    if not search.is_dir():
        return []
    paths = [
        str(p)
        for p in sorted(search.iterdir())
        if p.is_file() and p.suffix.lower() in exts
    ]
    return paths


def score_normal_images(model, image_paths, max_samples=0, status_text=None):
    """对正常图片列表打异常分，返回 float list。max_samples<=0 表示全部。"""
    import numpy as np
    import torchvision.transforms as T

    if not image_paths:
        return []

    paths = list(image_paths)
    if max_samples and max_samples > 0:
        paths = paths[: int(max_samples)]

    transform = T.Compose(
        [
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    scores = []
    model.eval()
    with torch.no_grad():
        for i, img_path in enumerate(paths):
            if status_text is not None and (i % 10 == 0 or i + 1 == len(paths)):
                status_text.text(f"校准集打分中: {i + 1}/{len(paths)}")
            image = Image.open(img_path).convert("RGB")
            tensor = transform(image).unsqueeze(0)
            _, anomaly_scores = model.predict(tensor)
            scores.append(float(anomaly_scores[0].item()))
    return scores


def resolve_calibration_scores(
    model,
    calib_source: str,
    train_dir: str,
    calib_dir: str,
    eval_normal_scores,
    max_samples=0,
    status_text=None,
):
    """
    按来源解析用于 P95/P99 的正常分数。
    返回 (scores, meta_dict)
    """
    if calib_source == "eval":
        return list(eval_normal_scores), {
            "source": "eval",
            "path": "(本次评估集正常样本)",
            "n_images": len(eval_normal_scores),
        }

    if calib_source == "calib":
        data_dir = calib_dir
        label = "单独校准集"
    else:
        data_dir = train_dir
        label = "训练集"

    paths = list_normal_image_paths(data_dir)
    if not paths:
        raise FileNotFoundError(
            f"{label}未找到正常图片。请确认目录存在 good/ 或目录下直接有图像："
            f"{data_dir}"
        )
    scores = score_normal_images(
        model, paths, max_samples=max_samples, status_text=status_text
    )
    return scores, {
        "source": calib_source,
        "path": data_dir,
        "n_images": len(scores),
        "n_available": len(paths),
    }

def get_available_devices():
    """返回当前环境可用的计算设备列表。"""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def resolve_device(device):
    """选择设备；请求 CUDA 但不可用时回退到 CPU。"""
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


# 双模式预设：旧版工程 vs 官网 PatchCore
MODE_PRESETS = {
    "官网版 (official)": {
        "variant": "official",
        "desc": "AvgPool(3,1,1) + 插值拼接 + approx_greedy_coreset + 重加权打分(neighbors=9)",
        "sampling_ratio": 0.1,
        "num_neighbors": 9,
        "model_save_path": "./results/patchcore_official.pth",
        "coreset_method": "approx_greedy",
    },
    "旧版 (legacy)": {
        "variant": "legacy",
        "desc": "无 AvgPool + 截断拼接 + 随机 Coreset（更快）+ 最大最近邻距离打分",
        "sampling_ratio": 0.1,
        "num_neighbors": 1,
        "model_save_path": "./results/patchcore_legacy.pth",
        "coreset_method": "random",
    },
}


def apply_mode_preset(mode_label: str):
    """按所选模式刷新训练默认参数。"""
    preset = MODE_PRESETS[mode_label]
    st.session_state.algorithm_mode = mode_label
    state = st.session_state.train_page_state
    state["variant"] = preset["variant"]
    state["sampling_ratio"] = preset["sampling_ratio"]
    state["num_neighbors"] = preset["num_neighbors"]
    state["coreset_method"] = preset["coreset_method"]
    # 仅在仍是默认路径时自动切换保存名，避免覆盖用户自定义路径
    cur = state.get("model_save_path", "")
    if (not cur) or ("patchcore" in os.path.basename(cur) and cur.startswith("./results/")):
        state["model_save_path"] = preset["model_save_path"]


def main():
    st.set_page_config(
        page_title="PatchCore 异常检测系统",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 初始化session_state
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.caption(
        f"PyTorch: {torch.__version__} | CUDA可用: {torch.cuda.is_available()}"
    )

    if 'train_page_state' not in st.session_state:
        st.session_state.train_page_state = {
            'data_selection_method': "自动扫描项目数据",
            'data_dir': "./data/btad/train",
            'backbone': "wide_resnet50_2",
            'layers': ["layer2", "layer3"],
            'patch_size': 3,
            'sampling_ratio': 0.1,
            'num_neighbors': 9,
            'variant': 'official',
            'coreset_method': 'approx_greedy',
            'batch_size': 8 if torch.cuda.is_available() else 4,
            'device': default_device,
            'model_save_path': "./results/patchcore_official.pth"
        }

    # 侧边栏：算法版本切换（旧版 / 官网）
    st.sidebar.markdown("### 算法版本")
    mode_labels = list(MODE_PRESETS.keys())
    current_mode = st.session_state.get("algorithm_mode", mode_labels[0])
    if current_mode not in mode_labels:
        current_mode = mode_labels[0]
    selected_mode = st.sidebar.radio(
        "选择实现",
        mode_labels,
        index=mode_labels.index(current_mode),
        key="algorithm_mode_radio",
    )
    if selected_mode != st.session_state.get("algorithm_mode"):
        apply_mode_preset(selected_mode)
    else:
        st.session_state.algorithm_mode = selected_mode
        # 保证 variant 字段存在
        if "variant" not in st.session_state.train_page_state:
            apply_mode_preset(selected_mode)
    st.sidebar.info(MODE_PRESETS[selected_mode]["desc"])
    st.sidebar.caption("训练与推理需使用同一版本；权重文件会写入 variant 字段，推理时自动识别（缺省按官网版）。")

    # 无 CUDA 时回退到 cpu；有 CUDA 且未手动指定时优先使用 cuda
    stored_device = st.session_state.train_page_state.get('device', default_device)
    if torch.cuda.is_available() and not st.session_state.get('device_user_set', False):
        st.session_state.train_page_state['device'] = "cuda"
    else:
        st.session_state.train_page_state['device'] = resolve_device(stored_device)
    
    if 'inference_page_state' not in st.session_state:
        st.session_state.inference_page_state = {
            'model_path': "./results/patchcore_model.pth",
            'inference_data_dir': "./data/btad/test",
            'threshold_mode': "f1_optimal",  # Anomalib 官网：验证集 F1 最优
            'manual_threshold': 30.0,  # 距离分数通常远大于 0，避免误用 0 导致全判异常
            'batch_size': 1,
            'num_samples': 80,
            'high_anomaly_warn_ratio': 0.75,
            # P95/P99 默认用训练集；可切换为单独校准集
            'calib_source': 'train',
            'calib_train_dir': './data/btad/train',
            'calib_dir': './data/btad/calib',
            'calib_max_samples': 0,  # 0=全部正常图
        }
    else:
        # 兼容旧 session：补齐校准相关字段
        _inf = st.session_state.inference_page_state
        _inf.setdefault('calib_source', 'train')
        _inf.setdefault('calib_train_dir', './data/btad/train')
        _inf.setdefault('calib_dir', './data/btad/calib')
        _inf.setdefault('calib_max_samples', 0)
        _inf.setdefault('high_anomaly_warn_ratio', 0.75)
    
    if 'config_page_state' not in st.session_state:
        st.session_state.config_page_state = {
            'config_file': "configs/config.yaml"
        }
    
    st.title("🔍 PatchCore 异常检测系统")
    st.markdown("---")
    
    # 侧边栏导航
    st.sidebar.title("导航")
    page = st.sidebar.radio("选择功能", ["训练模型", "推理检测", "配置管理"])
    
    if page == "训练模型":
        train_page()
    elif page == "推理检测":
        inference_page()
    elif page == "配置管理":
        config_page()

def train_page():
    st.header("🚀 模型训练")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("训练设置")
        
        # 数据集选择 - 支持选择本地任意目录
        st.markdown("**数据集选择**")
        
        # 选项：自动扫描或手动输入
        data_selection_method = st.radio(
            "选择方式", 
            ["自动扫描项目数据", "手动输入路径", "浏览文件夹"],
            index=0 if st.session_state.train_page_state['data_selection_method'] == "自动扫描项目数据" else 
                  1 if st.session_state.train_page_state['data_selection_method'] == "手动输入路径" else 2
        )
        
        # 保存选择方式到session_state
        st.session_state.train_page_state['data_selection_method'] = data_selection_method
        
        if data_selection_method == "自动扫描项目数据":
            # 扫描data目录下的可用数据集
            data_dir_options = []
            data_base_dir = "./data"
            
            if os.path.exists(data_base_dir):
                for dataset_name in os.listdir(data_base_dir):
                    dataset_path = os.path.join(data_base_dir, dataset_name)
                    if os.path.isdir(dataset_path):
                        # 检查是否有训练数据（包含good目录）
                        train_path = os.path.join(dataset_path, "train")
                        if os.path.exists(train_path) and os.path.exists(os.path.join(train_path, "good")):
                            data_dir_options.append(train_path)
                        # 如果没有train目录，检查根目录是否有good目录
                        elif os.path.exists(os.path.join(dataset_path, "good")):
                            data_dir_options.append(dataset_path)
            
            # 如果没有找到训练数据，使用测试数据作为示例
            if not data_dir_options:
                data_dir_options = ["./data/btad/test"]
            
            # 尝试恢复之前选择的数据目录
            current_data_dir = st.session_state.train_page_state['data_dir']
            if current_data_dir in data_dir_options:
                default_index = data_dir_options.index(current_data_dir)
            else:
                default_index = 0
            
            data_dir = st.selectbox("选择数据集路径", data_dir_options, index=default_index)
            
        elif data_selection_method == "手动输入路径":
            # 手动输入路径
            data_dir = st.text_input(
                "输入数据集路径", 
                value=st.session_state.train_page_state['data_dir'],
                placeholder="例如: C:/Users/用户名/Documents/dataset"
            )
            
            # 路径验证和提示
            if data_dir:
                if os.path.exists(data_dir):
                    st.success("✅ 路径有效")
                else:
                    st.error("❌ 路径不存在，请检查路径是否正确")
            
            # 路径格式提示
            st.info("💡 提示：数据集目录应包含 'good' 子目录（正常样本）")
        
        else:  # 浏览文件夹
            st.info("💡 使用文件浏览器选择数据集目录")
            
            # 提供一个简单的文件浏览器模拟
            current_path = st.session_state.train_page_state['data_dir']
            
            # 如果当前路径不存在，使用项目根目录
            if not os.path.exists(current_path):
                current_path = os.path.dirname(os.path.abspath(__file__))
            
            # 显示当前路径
            st.text_input("当前路径", value=current_path, disabled=True)
            
            # 显示当前目录下的文件夹
            try:
                items = os.listdir(current_path)
                folders = [item for item in items if os.path.isdir(os.path.join(current_path, item))]
                
                if folders:
                    st.write("可用的文件夹:")
                    
                    # 创建两列布局
                    cols = st.columns(3)
                    
                    for i, folder in enumerate(folders[:9]):  # 最多显示9个文件夹
                        with cols[i % 3]:
                            if st.button(folder, key=f"folder_{i}"):
                                # 进入选中的文件夹
                                new_path = os.path.join(current_path, folder)
                                st.session_state.train_page_state['data_dir'] = new_path
                                st.rerun()
                else:
                    st.write("当前目录下没有文件夹")
                
                # 返回上一级按钮
                parent_path = os.path.dirname(current_path)
                if parent_path != current_path:  # 避免无限循环
                    if st.button("返回上一级"):
                        st.session_state.train_page_state['data_dir'] = parent_path
                        st.rerun()
                
                # 选择当前目录作为数据集目录
                if st.button("选择当前目录作为数据集目录"):
                    data_dir = current_path
                    st.success(f"已选择: {data_dir}")
                else:
                    data_dir = current_path
                    
            except Exception as e:
                st.error(f"无法访问目录: {e}")
                data_dir = "./data/btad/test"
        
        # 保存数据目录到session_state
        st.session_state.train_page_state['data_dir'] = data_dir
        
        # 显示数据集信息
        if st.button("检查数据集"):
            if os.path.exists(data_dir):
                good_count = len(os.listdir(os.path.join(data_dir, "good"))) if os.path.exists(os.path.join(data_dir, "good")) else 0
                defective_count = len(os.listdir(os.path.join(data_dir, "defective"))) if os.path.exists(os.path.join(data_dir, "defective")) else 0
                st.info(f"数据集信息: 正常样本 {good_count} 个, 缺陷样本 {defective_count} 个")
            else:
                st.error("数据集路径不存在！")
        
        st.markdown("---")
        
        # 模型参数
        st.markdown("**模型参数**")
        variant = st.session_state.train_page_state.get("variant", "official")
        st.success(
            f"当前模式: **{st.session_state.get('algorithm_mode', '官网版 (official)')}** "
            f"(`variant={variant}`)"
        )
        backbone = st.selectbox(
            "Backbone模型", 
            ["wide_resnet50_2", "resnet18", "resnet50"],
            index=["wide_resnet50_2", "resnet18", "resnet50"].index(st.session_state.train_page_state['backbone'])
        )
        layers = st.multiselect(
            "特征提取层", 
            ["layer1", "layer2", "layer3"], 
            default=st.session_state.train_page_state['layers']
        )
        patch_size = st.slider("Patch大小", 1, 10, st.session_state.train_page_state['patch_size'])
        sampling_ratio = st.slider(
            "Coreset采样比例",
            0.01, 0.5,
            float(st.session_state.train_page_state['sampling_ratio'])
        )
        if variant == "official":
            num_neighbors = st.number_input(
                "最近邻数 num_neighbors（官网默认 9）",
                min_value=1, max_value=50,
                value=int(st.session_state.train_page_state.get('num_neighbors', 9))
            )
            coreset_method = st.selectbox(
                "Coreset 方法",
                ["approx_greedy", "anomalib_kcenter", "random"],
                index=["approx_greedy", "anomalib_kcenter", "random"].index(
                    st.session_state.train_page_state.get("coreset_method", "approx_greedy")
                ),
            )
            st.caption("官网推荐：approx_greedy；anomalib_kcenter 更慢；random 仅调试。")
        else:
            num_neighbors = 1
            coreset_method = "random"
            st.caption("旧版固定：随机 Coreset + 简单最大距离打分（训练更快）。")
        
        # 保存模型参数到session_state
        st.session_state.train_page_state['backbone'] = backbone
        st.session_state.train_page_state['layers'] = layers
        st.session_state.train_page_state['patch_size'] = patch_size
        st.session_state.train_page_state['sampling_ratio'] = sampling_ratio
        st.session_state.train_page_state['num_neighbors'] = num_neighbors
        st.session_state.train_page_state['coreset_method'] = coreset_method
        st.session_state.train_page_state['variant'] = variant
        
        # 训练参数
        st.markdown("**训练参数**")
        batch_size = st.number_input("批次大小", 1, 64, st.session_state.train_page_state['batch_size'])
        available_devices = get_available_devices()
        preferred_device = resolve_device(st.session_state.train_page_state['device'])
        device_index = available_devices.index(preferred_device) if preferred_device in available_devices else 0
        device = st.selectbox(
            "计算设备",
            available_devices,
            index=device_index,
            key="train_device_select"
        )
        if st.session_state.get("train_device_select") != st.session_state.train_page_state.get("device"):
            st.session_state.device_user_set = True
        if not torch.cuda.is_available():
            st.caption("当前 PyTorch 为 CPU 版本，已禁用 CUDA 选项。若在 GPU 环境中仍看到此提示，请用：python -m streamlit run app.py")
        else:
            st.caption(f"检测到 GPU：{torch.cuda.get_device_name(0)}")
        
        # 保存训练参数到session_state
        st.session_state.train_page_state['batch_size'] = batch_size
        st.session_state.train_page_state['device'] = device
        
        # 模型保存路径
        st.markdown("**模型保存**")
        model_save_path = st.text_input(
            "模型保存路径", 
            value=st.session_state.train_page_state['model_save_path']
        )
        
        # 保存模型保存路径到session_state
        st.session_state.train_page_state['model_save_path'] = model_save_path
    
    with col2:
        st.subheader("训练控制")
        
        if st.button("开始训练", type="primary"):
            if not data_dir or not os.path.exists(data_dir):
                st.error("数据集路径不存在！")
                return
            if not layers:
                st.error("请至少选择一个特征提取层！")
                return
            
            try:
                with st.spinner("正在训练模型..."):
                    selected_device = resolve_device(device)
                    # 创建模型（兼容旧版/官网）
                    model = PatchCore(
                        backbone_name=backbone,
                        layers=layers,
                        patch_size=patch_size,
                        sampling_ratio=sampling_ratio,
                        num_neighbors=int(num_neighbors),
                        device=selected_device,
                        variant=variant,
                        coreset_method=coreset_method,
                    )
                    st.info(
                        f"模式: {model.variant} | 设备: {model.device} | "
                        f"coreset={model.coreset_method} | neighbors={model.num_neighbors}"
                    )
                    
                    # 加载数据
                    train_loader = get_data_loader(
                        data_dir,
                        split="train",
                        batch_size=batch_size
                    )
                    if len(train_loader.dataset) == 0:
                        st.error("训练数据集为空，请确认目录下存在 good/ 正常样本。")
                        return
                    
                    # 训练 - 添加实时进度显示
                    start_time = time.time()
                    
                    # 创建进度条和状态显示
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 计算总批次数量
                    total_batches = len(train_loader)
                    
                    # 修改模型fit方法以支持进度回调
                    model.fit_with_progress(train_loader, progress_bar, status_text, total_batches)
                    
                    training_time = time.time() - start_time
                    
                    # 保存模型
                    status_text.text("正在保存模型...")
                    progress_bar.progress(0.95)
                    save_dir = os.path.dirname(model_save_path)
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                    model.save_model(model_save_path)
                    
                    # 完成
                    progress_bar.progress(1.0)
                    status_text.text("训练完成！")
                    
                    st.success(f"训练完成！耗时: {training_time:.2f}秒")
                    st.info(f"模型已保存到: {model_save_path}")
                    st.info(f"最终设备: {model.device} | Memory Bank: {tuple(model.memory_bank.shape)}")
                    
                    # 显示训练统计信息
                    st.subheader("训练统计")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("训练样本数", len(train_loader.dataset))
                    col2.metric("总批次", total_batches)
                    col3.metric("训练耗时", f"{training_time:.1f}秒")
                    
            except Exception as e:
                st.error(f"训练失败: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

def inference_page():
    st.header("🔍 异常检测推理")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("推理设置")
        
        # 模型选择 - 动态扫描results目录下的模型文件
        st.markdown("**模型选择**")
        
        model_path_options = []
        results_dir = "./results"
        
        if os.path.exists(results_dir):
            for file_name in os.listdir(results_dir):
                if file_name.endswith('.pth') or file_name.endswith('.pt'):
                    model_path_options.append(os.path.join(results_dir, file_name))
        
        # 如果没有找到模型文件，使用默认选项
        if not model_path_options:
            model_path_options = ["./results/patchcore_model.pth"]
        
        model_path = st.selectbox("选择模型文件", model_path_options, index=0)
        
        # 检查模型文件是否存在
        if st.button("检查模型"):
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
                st.success(f"模型文件存在！大小: {file_size:.2f} MB")
            else:
                st.error("模型文件不存在！")
        
        st.markdown("---")
        
        # 推理数据选择 - 动态扫描可用的数据集
        st.markdown("**推理数据**")
        
        inference_data_options = []
        data_base_dir = "./data"
        
        if os.path.exists(data_base_dir):
            for dataset_name in os.listdir(data_base_dir):
                dataset_path = os.path.join(data_base_dir, dataset_name)
                if os.path.isdir(dataset_path):
                    # 检查是否有测试数据（包含good和/或defective目录）
                    test_path = os.path.join(dataset_path, "test")
                    if os.path.exists(test_path):
                        inference_data_options.append(test_path)
                    # 如果没有test目录，检查根目录是否有good目录
                    elif os.path.exists(os.path.join(dataset_path, "good")):
                        inference_data_options.append(dataset_path)
        
        # 如果没有找到测试数据，使用默认选项
        if not inference_data_options:
            inference_data_options = ["./data/btad/test"]
        
        inference_data_dir = st.selectbox("选择推理数据集", inference_data_options, index=0)
        
        # 检查数据集
        if st.button("检查推理数据集"):
            if os.path.exists(inference_data_dir):
                good_count = len(os.listdir(os.path.join(inference_data_dir, "good"))) if os.path.exists(os.path.join(inference_data_dir, "good")) else 0
                defective_count = len(os.listdir(os.path.join(inference_data_dir, "defective"))) if os.path.exists(os.path.join(inference_data_dir, "defective")) else 0
                st.info(f"推理数据集: 正常样本 {good_count} 个, 缺陷样本 {defective_count} 个")
            else:
                st.error("推理数据集路径不存在！")
        
        st.markdown("---")
        
        # 推理参数：兼容 Anomalib F1 自适应 + 正常样本分位数
        st.markdown("**推理参数**")
        st.caption(
            "Anomalib 官网默认：在验证集上用 **F1 最优** 自适应阈值（原始距离分数，不是 0~1）；"
            "展示时再 min-max 归一化把阈值映到约 0.5。本页同时支持 **正常样本 P95/P99**（部署更稳）。"
            "判定：score > threshold ⇒ 异常。"
        )
        mode_keys = list(THRESHOLD_MODE_LABELS.keys())
        saved_mode = st.session_state.inference_page_state.get("threshold_mode", "f1_optimal")
        if saved_mode not in mode_keys:
            saved_mode = "f1_optimal"
        threshold_mode = st.selectbox(
            "阈值模式",
            mode_keys,
            index=mode_keys.index(saved_mode),
            format_func=lambda x: THRESHOLD_MODE_LABELS[x],
        )

        st.markdown("**P95/P99 校准来源**")
        st.caption(
            "默认用 **训练集 good** 定分位数（部署推荐）；也可指定 **单独校准集**（仅正常图）。"
            "「评估集正常」仅作对照，不推荐作为正式阈值。"
        )
        calib_keys = list(CALIB_SOURCE_LABELS.keys())
        saved_calib = st.session_state.inference_page_state.get("calib_source", "train")
        if saved_calib not in calib_keys:
            saved_calib = "train"
        calib_source = st.selectbox(
            "校准数据来源",
            calib_keys,
            index=calib_keys.index(saved_calib),
            format_func=lambda x: CALIB_SOURCE_LABELS[x],
        )
        guessed_train = guess_train_dir_from_infer(inference_data_dir)
        default_train = st.session_state.inference_page_state.get("calib_train_dir") or guessed_train
        if not st.session_state.inference_page_state.get("calib_train_dir"):
            default_train = guessed_train
        calib_train_dir = st.text_input(
            "训练集目录（需含 good/）",
            value=default_train if os.path.exists(default_train) else guessed_train,
        )
        calib_dir = st.text_input(
            "单独校准集目录（仅「单独校准集」时使用；含 good/ 或直接放正常图）",
            value=st.session_state.inference_page_state.get("calib_dir", "./data/btad/calib"),
        )
        calib_max_samples = st.number_input(
            "校准最多使用正常图数量（0=全部）",
            min_value=0,
            max_value=5000,
            value=int(st.session_state.inference_page_state.get("calib_max_samples", 0)),
        )

        manual_threshold = st.number_input(
            "手动阈值（仅「手动」模式；请按页面给出的正常 P95/P99 量级设置，勿用 0）",
            min_value=0.0,
            max_value=10000.0,
            value=float(st.session_state.inference_page_state.get("manual_threshold", 30.0)),
            step=0.1,
        )
        batch_size = st.number_input("推理批次大小", 1, 64, 1)
        num_samples = st.number_input(
            "评估样本数量（分层抽取 good/defective）",
            2, 1000,
            int(st.session_state.inference_page_state.get('num_samples', 80))
        )
        st.session_state.inference_page_state['threshold_mode'] = threshold_mode
        st.session_state.inference_page_state['calib_source'] = calib_source
        st.session_state.inference_page_state['calib_train_dir'] = calib_train_dir
        st.session_state.inference_page_state['calib_dir'] = calib_dir
        st.session_state.inference_page_state['calib_max_samples'] = int(calib_max_samples)
        st.session_state.inference_page_state['manual_threshold'] = manual_threshold
        st.session_state.inference_page_state['num_samples'] = num_samples
    
    with col2:
        st.subheader("推理控制")
        
        if st.button("开始推理", type="primary"):
            if not os.path.exists(model_path):
                st.error("模型文件不存在！")
                return
            
            if not inference_data_dir or not os.path.exists(inference_data_dir):
                st.error("推理数据集路径不存在！")
                return
            
            try:
                with st.spinner("正在进行推理..."):
                    import numpy as np
                    from sklearn.metrics import roc_auc_score

                    infer_device = resolve_device("cuda" if torch.cuda.is_available() else "cpu")
                    # 初始化后由 load_model 自动恢复 variant（无 variant 字段时默认 official）
                    model = PatchCore(
                        backbone_name="wide_resnet50_2",
                        layers=["layer2", "layer3"],
                        patch_size=3,
                        sampling_ratio=0.1,
                        num_neighbors=9,
                        device=infer_device,
                        variant="official",
                    )
                    model.load_model(model_path)
                    mode_name = (
                        "官网版 (official)" if model.variant == "official" else "旧版 (legacy)"
                    )
                    st.info(
                        f"自动识别模式: **{mode_name}** | 设备: {model.device} | "
                        f"backbone={model.backbone_name} | layers={model.layers} | "
                        f"coreset={model.coreset_method}/{model.sampling_ratio} | "
                        f"neighbors={model.num_neighbors}"
                    )
                    
                    dataset = DataDirDataset(inference_data_dir, split="test")
                    if len(dataset) == 0:
                        st.error("推理数据集为空。")
                        return

                    eval_indices = select_eval_samples(dataset, int(num_samples), balance=True)
                    n_good = sum(1 for i in eval_indices if dataset.samples[i][1] == 0)
                    n_def = sum(1 for i in eval_indices if dataset.samples[i][1] == 1)
                    st.info(f"分层评估样本: 正常 {n_good} + 缺陷 {n_def} = {len(eval_indices)}")
                    if n_good == 0 or n_def == 0:
                        st.error("评估集缺少正常或缺陷样本，精确率/召回率/F1 无法计算。请增大样本数或检查数据目录。")
                        return
                    
                    results = []
                    model.eval()
                    with torch.no_grad():
                        for sample_idx in eval_indices:
                            image, label = dataset[sample_idx]
                            images = image.unsqueeze(0)
                            anomaly_maps, anomaly_scores = model.predict(images)
                            img_path, _, sample_type = dataset.samples[sample_idx]
                            results.append({
                                'image_name': os.path.basename(img_path),
                                'sample_type': sample_type,
                                'label': int(label.item()),
                                'anomaly_score': float(anomaly_scores[0].item()),
                                'anomaly_map': anomaly_maps[0]
                            })

                    y_true = [r['label'] for r in results]
                    y_score = [r['anomaly_score'] for r in results]
                    eval_normal_scores = [s for s, t in zip(y_score, y_true) if t == 0]
                    def_scores = [s for s, t in zip(y_score, y_true) if t == 1]
                    f1_thr, f1_meta = find_f1_optimal_threshold(y_true, y_score)

                    # P95/P99：默认训练集；可单独校准集；评估集仅对照
                    calib_status = st.empty()
                    calib_cache_key = (
                        os.path.abspath(model_path),
                        calib_source,
                        os.path.abspath(calib_train_dir) if calib_source == "train" else "",
                        os.path.abspath(calib_dir) if calib_source == "calib" else "",
                        int(calib_max_samples),
                        getattr(model, "variant", ""),
                        getattr(model, "num_neighbors", None),
                    )
                    try:
                        if (
                            st.session_state.get("_calib_cache_key") == calib_cache_key
                            and st.session_state.get("_calib_scores")
                        ):
                            calib_scores = list(st.session_state["_calib_scores"])
                            calib_meta = dict(st.session_state["_calib_meta"])
                            st.caption("已复用缓存的校准集分数（模型/路径未变）。")
                        else:
                            calib_scores, calib_meta = resolve_calibration_scores(
                                model,
                                calib_source=calib_source,
                                train_dir=calib_train_dir,
                                calib_dir=calib_dir,
                                eval_normal_scores=eval_normal_scores,
                                max_samples=int(calib_max_samples),
                                status_text=calib_status,
                            )
                            if calib_source != "eval":
                                st.session_state["_calib_cache_key"] = calib_cache_key
                                st.session_state["_calib_scores"] = list(calib_scores)
                                st.session_state["_calib_meta"] = dict(calib_meta)
                    except Exception as calib_err:
                        st.error(f"校准集打分失败: {calib_err}")
                        if calib_source != "eval":
                            st.warning("已回退为「本次评估集正常样本」计算 P95/P99（仅本次）。")
                            calib_scores, calib_meta = resolve_calibration_scores(
                                model,
                                calib_source="eval",
                                train_dir=calib_train_dir,
                                calib_dir=calib_dir,
                                eval_normal_scores=eval_normal_scores,
                                max_samples=0,
                                status_text=None,
                            )
                        else:
                            raise
                    finally:
                        calib_status.empty()

                    pct = normal_score_percentiles(calib_scores)
                    pct_eval = normal_score_percentiles(eval_normal_scores)

                    # 始终展示校准分位数与 F1 候选
                    st.subheader("阈值参考（校准集 P95/P99 / F1）")
                    st.caption(
                        f"校准来源: **{CALIB_SOURCE_LABELS.get(calib_meta['source'], calib_meta['source'])}** | "
                        f"路径: `{calib_meta.get('path', '')}` | "
                        f"使用正常图: {calib_meta.get('n_images', 0)}"
                        + (
                            f" / 可用 {calib_meta.get('n_available')}"
                            if calib_meta.get("n_available") is not None
                            else ""
                        )
                    )
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("校准 P90", f"{pct['p90']:.4f}" if pct["p90"] is not None else "N/A")
                    c2.metric("校准 P95", f"{pct['p95']:.4f}" if pct["p95"] is not None else "N/A")
                    c3.metric("校准 P99", f"{pct['p99']:.4f}" if pct["p99"] is not None else "N/A")
                    c4.metric("F1最优候选", f"{f1_thr:.4f}")
                    if pct_eval["p95"] is not None:
                        st.caption(
                            f"对照（评估集正常）: P95={pct_eval['p95']:.4f}, P99={pct_eval['p99']:.4f} "
                            f"（n={len(eval_normal_scores)}）。"
                            "正式 P95/P99 默认按训练集/校准集，不以评估集为准。"
                        )

                    if threshold_mode == "f1_optimal":
                        used_threshold, best_at_thr = f1_thr, f1_meta
                        st.success(
                            f"使用阈值模式: {THRESHOLD_MODE_LABELS['f1_optimal']} → **{used_threshold:.4f}**"
                        )
                        if best_at_thr is not None:
                            st.caption(
                                f"该阈值下搜索到的最佳点: "
                                f"P={best_at_thr['precision']*100:.1f}% "
                                f"R={best_at_thr['recall']*100:.1f}% "
                                f"F1={best_at_thr['f1']*100:.1f}%"
                            )
                    elif threshold_mode == "normal_p95":
                        used_threshold, _ = find_normal_percentile_threshold(calib_scores, 95)
                        st.success(
                            f"使用阈值模式: {THRESHOLD_MODE_LABELS['normal_p95']} → **{used_threshold:.4f}** "
                            f"（来自{CALIB_SOURCE_LABELS.get(calib_meta['source'], calib_meta['source'])}）"
                        )
                    elif threshold_mode == "normal_p99":
                        used_threshold, _ = find_normal_percentile_threshold(calib_scores, 99)
                        st.success(
                            f"使用阈值模式: {THRESHOLD_MODE_LABELS['normal_p99']} → **{used_threshold:.4f}** "
                            f"（来自{CALIB_SOURCE_LABELS.get(calib_meta['source'], calib_meta['source'])}）"
                        )
                    else:
                        used_threshold = float(manual_threshold)
                        st.info(
                            f"使用阈值模式: {THRESHOLD_MODE_LABELS['manual']} → **{used_threshold:.4f}**"
                        )
                        if used_threshold <= 1e-6:
                            st.warning(
                                "手动阈值为 0：PatchCore 距离分数通常远大于 0，"
                                "会导致几乎全部判为异常。请改用正常 P95/P99 或合理手动值。"
                            )

                    for result in results:
                        result['is_anomaly'] = result['anomaly_score'] > used_threshold
                    
                    st.subheader("推理结果")
                    total_samples = len(results)
                    anomaly_count = sum(1 for r in results if r['is_anomaly'])
                    normal_count = total_samples - anomaly_count
                    anomaly_ratio = anomaly_count / max(total_samples, 1)
                    warn_ratio = float(
                        st.session_state.inference_page_state.get("high_anomaly_warn_ratio", 0.75)
                    )
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("评估样本", total_samples)
                    col2.metric("预测异常", anomaly_count)
                    col3.metric("预测正常", normal_count)
                    col4.metric("真实缺陷数", n_def)

                    if anomaly_ratio >= warn_ratio:
                        st.warning(
                            f"当前阈值下预测异常占比 **{anomaly_ratio*100:.1f}%** "
                            f"（≥ 告警线 {warn_ratio*100:.0f}%）。"
                            + (
                                " F1 最优在分数重叠严重时往往会压低阈值、抬高召回，从而「大部分判异常」。"
                                if threshold_mode == "f1_optimal"
                                else " 可尝试提高阈值（如改用正常 P99），或检查正常/缺陷分数是否重叠。"
                            )
                            + f" 参考：正常 P95={pct['p95']:.4f}，P99={pct['p99']:.4f}，F1候选={f1_thr:.4f}。"
                        )
                    
                    true_positive = sum(1 for r in results if r['label'] == 1 and r['is_anomaly'])
                    false_positive = sum(1 for r in results if r['label'] == 0 and r['is_anomaly'])
                    true_negative = sum(1 for r in results if r['label'] == 0 and not r['is_anomaly'])
                    false_negative = sum(1 for r in results if r['label'] == 1 and not r['is_anomaly'])
                    
                    accuracy = (true_positive + true_negative) / total_samples
                    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
                    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
                    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                    try:
                        auroc = float(roc_auc_score(y_true, y_score))
                    except Exception:
                        auroc = float("nan")
                    
                    st.subheader("性能指标（图像级）")
                    st.caption(
                        "精确率=TP/(TP+FP)，召回率=TP/(TP+FN)，F1=2PR/(P+R)。"
                        "AUROC **与阈值无关**：对所有可能阈值画 ROC（FPR vs TPR）后求曲线下面积；"
                        "本页用 `sklearn.metrics.roc_auc_score(y_true, y_score)`。"
                    )
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("准确率", f"{accuracy*100:.1f}%")
                    m2.metric("精确率", f"{precision*100:.1f}%")
                    m3.metric("召回率", f"{recall*100:.1f}%")
                    m4.metric("F1", f"{f1*100:.1f}%")
                    m5.metric("AUROC", f"{auroc*100:.1f}%" if auroc == auroc else "N/A")

                    with st.expander("P95/P99 与 AUROC 怎么算？", expanded=False):
                        st.markdown(
                            f"""
**正常分数 P95 / P99（部署阈值）**

1. **默认**取 **训练集** `good/`（或你指定的 **单独校准集**）中正常图，用当前模型打分得到 \\(S_{{calib}}\\)。  
2. `P95 = percentile(S_calib, 95)`，`P99 = percentile(S_calib, 99)`。  
3. 阈值取 P95 时，在该校准正常集上约 **5%** 会被判异常；P99 约 **1%**。  
4. 不依赖缺陷样本。评估集上的正常分仅作对照，**不作为默认定阈依据**。  

当前校准：来源=`{calib_meta.get('source')}`，路径=`{calib_meta.get('path')}`，  
n={calib_meta.get('n_images')}，P95={pct['p95']:.4f}，P99={pct['p99']:.4f}。

**AUROC（图像级，在评估集上）**

1. 对评估集每张图有分数 \\(s\\) 与标签 \\(y\\in\\{{0,1\\}}\\)。  
2. 遍历阈值画 ROC（FPR–TPR），AUROC = 曲线下面积。  
3. 本页：`sklearn.metrics.roc_auc_score`；**与阈值无关**。  
4. 当前 AUROC={auroc*100:.1f}%（类别不全则为 N/A）。

**官网（Anomalib）阈值**

- 默认验证集 **F1 最优**；本页「F1最优」与之对齐。  
- 「正常 P95/P99」默认用训练集/校准集，是部署向的额外选项。
                            """
                        )

                    st.write("混淆矩阵")
                    st.dataframe(
                        {
                            "类别": ["真实正常 (label=0)", "真实异常 (label=1)"],
                            "预测正常": [true_negative, false_negative],
                            "预测异常": [false_positive, true_positive],
                        },
                        hide_index=True,
                    )
                    st.caption(f"TP={true_positive}, FP={false_positive}, TN={true_negative}, FN={false_negative}")

                    st.info(
                        f"评估分数范围: [{min(y_score):.3f}, {max(y_score):.3f}] | "
                        f"评估正常均值 {np.mean(eval_normal_scores):.3f} | "
                        f"缺陷均值 {np.mean(def_scores):.3f} | "
                        f"使用阈值 {used_threshold:.3f}（{THRESHOLD_MODE_LABELS.get(threshold_mode, threshold_mode)}）| "
                        f"校准 P95={pct['p95']:.3f} / P99={pct['p99']:.3f} "
                        f"（{CALIB_SOURCE_LABELS.get(calib_meta['source'], calib_meta['source'])}）"
                    )
                    
                    st.subheader("详细结果")
                    result_data = []
                    for result in results:
                        correct = (
                            (result['label'] == 1 and result['is_anomaly']) or
                            (result['label'] == 0 and not result['is_anomaly'])
                        )
                        result_data.append({
                            '图像名称': result['image_name'],
                            '样本类型': result['sample_type'],
                            '真实标签': '异常' if result['label'] == 1 else '正常',
                            '异常分数': f"{result['anomaly_score']:.4f}",
                            '检测结果': '异常' if result['is_anomaly'] else '正常',
                            '检测正确': '✓' if correct else '✗'
                        })
                    st.dataframe(result_data)
                    
                    st.subheader("异常分数分布")
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 6))
                    if eval_normal_scores:
                        ax.hist(eval_normal_scores, bins=20, alpha=0.7, label='评估集正常', color='green')
                    if def_scores:
                        ax.hist(def_scores, bins=20, alpha=0.7, label='评估集异常', color='red')
                    if calib_scores and calib_meta.get("source") != "eval":
                        ax.hist(
                            calib_scores,
                            bins=20,
                            alpha=0.35,
                            label=f"校准正常({calib_meta.get('source')})",
                            color="blue",
                        )
                    ax.axvline(x=used_threshold, color='black', linestyle='--', label=f'使用阈值 ({used_threshold:.3f})')
                    if pct["p95"] is not None:
                        ax.axvline(x=pct["p95"], color='orange', linestyle=':', label=f'校准P95 ({pct["p95"]:.3f})')
                    if pct["p99"] is not None:
                        ax.axvline(x=pct["p99"], color='purple', linestyle=':', label=f'校准P99 ({pct["p99"]:.3f})')
                    if abs(f1_thr - used_threshold) > 1e-6:
                        ax.axvline(x=f1_thr, color='gray', linestyle='-.', label=f'F1候选 ({f1_thr:.3f})')
                    ax.set_xlabel('异常分数')
                    ax.set_ylabel('样本数量')
                    ax.set_title('异常分数分布（阈值线来自校准集分位数）')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"推理失败: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

def config_page():
    st.header("⚙️ 配置管理")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # 配置文件选择
        st.subheader("配置文件管理")
        config_file_options = ["configs/config.yaml", "configs/config_data_dir.yaml"]
        config_file = st.selectbox("选择配置文件", config_file_options, index=0)
        
        # 配置文件操作
        if st.button("检查配置文件"):
            if os.path.exists(config_file):
                file_size = os.path.getsize(config_file) / 1024  # KB
                st.success(f"配置文件存在！大小: {file_size:.2f} KB")
            else:
                st.error("配置文件不存在！")
        
        # 创建新配置文件
        new_config_name = st.text_input("新配置文件名", placeholder="例如: my_config.yaml")
        if st.button("创建新配置"):
            if new_config_name:
                new_config_path = os.path.join("configs", new_config_name)
                if not new_config_name.endswith('.yaml'):
                    new_config_path += '.yaml'
                
                # 创建默认配置
                default_config = """# PatchCore Configuration

# Model settings
model:
  backbone: "wide_resnet50_2"
  layers: ["layer2", "layer3"]
  patch_size: 3
  sampling_ratio: 0.1

# Data settings
data:
  image_size: [256, 256]
  batch_size: 32
  num_workers: 4

# Training settings
training:
  device: "cuda"
  save_path: "./results"

# Inference settings
inference:
  threshold: 0.5
  visualize: True
"""
                try:
                    with open(new_config_path, 'w', encoding='utf-8') as f:
                        f.write(default_config)
                    st.success(f"配置文件 {new_config_path} 创建成功！")
                except Exception as e:
                    st.error(f"创建失败: {str(e)}")
            else:
                st.error("请输入配置文件名！")
    
    with col2:
        st.subheader("配置编辑")
        
        if os.path.exists(config_file):
            # 读取配置文件
            with open(config_file, 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            # 配置编辑器
            edited_config = st.text_area("编辑配置内容", config_content, height=300)
            
            # 配置验证和保存
            col_save, col_validate = st.columns(2)
            
            with col_save:
                if st.button("💾 保存配置"):
                    try:
                        # 验证YAML格式
                        yaml.safe_load(edited_config)
                        with open(config_file, 'w', encoding='utf-8') as f:
                            f.write(edited_config)
                        st.success("配置保存成功！")
                    except yaml.YAMLError as e:
                        st.error(f"YAML格式错误: {str(e)}")
                    except Exception as e:
                        st.error(f"保存失败: {str(e)}")
            
            with col_validate:
                if st.button("🔍 验证配置"):
                    try:
                        config = yaml.safe_load(edited_config)
                        st.success("✅ 配置格式正确！")
                        
                        # 显示配置摘要
                        st.info(f"模型: {config.get('model', {}).get('backbone', '未设置')}")
                        st.info(f"图像尺寸: {config.get('data', {}).get('image_size', '未设置')}")
                        st.info(f"设备: {config.get('training', {}).get('device', '未设置')}")
                        
                    except yaml.YAMLError as e:
                        st.error(f"❌ YAML格式错误: {str(e)}")
            
            # 配置预览
            st.subheader("配置预览")
            try:
                config = yaml.safe_load(edited_config)
                st.json(config)
            except Exception as e:
                st.error(f"配置预览错误: {str(e)}")
        else:
            st.warning("请选择或创建一个配置文件")
    
    # 配置模板
    st.markdown("---")
    st.subheader("配置模板")
    
    template_col1, template_col2 = st.columns(2)
    
    with template_col1:
        if st.button("加载默认配置模板"):
            default_config = """# PatchCore Configuration

# Model settings
model:
  backbone: "wide_resnet50_2"
  layers: ["layer2", "layer3"]
  patch_size: 3
  sampling_ratio: 0.1

# Data settings
data:
  image_size: [256, 256]
  batch_size: 32
  num_workers: 4

# Training settings
training:
  device: "cuda"
  save_path: "./results"

# Inference settings
inference:
  threshold: 0.5
  visualize: True
"""
            st.code(default_config, language='yaml')
    
    with template_col2:
        if st.button("加载轻量配置模板"):
            light_config = """# Lightweight PatchCore Configuration

# Model settings
model:
  backbone: "resnet18"
  layers: ["layer2", "layer3"]
  patch_size: 3
  sampling_ratio: 0.05

# Data settings
data:
  image_size: [224, 224]
  batch_size: 16
  num_workers: 2

# Training settings
training:
  device: "cpu"
  save_path: "./results"

# Inference settings
inference:
  threshold: 0.3
  visualize: True
"""
            st.code(light_config, language='yaml')

if __name__ == "__main__":
    main()