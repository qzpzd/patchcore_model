# PatchCore Anomaly Detection

基于PatchCore算法的无监督异常检测工具。

## 项目介绍

PatchCore是一种高效的无监督异常检测方法，特别适用于工业检测场景。该方法通过预训练的深度神经网络提取图像特征，并构建一个具有代表性的内存库(memory bank)，在推理阶段通过计算测试图像特征与内存库的距离来判断是否存在异常。

## 算法特点

1. **高精度**：在多个基准数据集上达到SOTA性能
2. **快速推理**：相比其他方法具有更快的推理速度
3. **无监督学习**：仅需正常样本进行训练
4. **可解释性强**：能够提供像素级的异常定位

## 项目结构

```
patchcore_model/
├── configs/           # 配置文件
├── models/            # 模型实现
├── utils/             # 工具函数
├── data/              # 数据集（需用户自行准备）
├── results/           # 训练结果保存目录
├── train.py           # 训练脚本
├── inference.py       # 推理脚本
├── main.py            # 主程序入口
└── requirements.txt   # 依赖包列表
```

## 环境要求

- Python >= 3.7
- PyTorch >= 1.8.0
- 其他依赖见 requirements.txt

## 安装步骤

1. 克隆项目：
```bash
git clone <project-url>
cd patchcore_model
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 数据集准备

本项目支持MVTec AD数据集格式，数据集应按以下结构组织：

```
data/
└── category/
    ├── train/
    │   └── good/
    │       ├── 000.png
    │       ├── 001.png
    │       └── ...
    └── test/
        ├── good/
        │   ├── 000.png
        │   ├── 001.png
        │   └── ...
        ├── defect_type1/
        │   ├── 000.png
        │   ├── 001.png
        │   └── ...
        └── ...
```

## 使用方法

### 训练模型

```bash
python main.py train --config ./configs/config.yaml
```

或者直接运行训练脚本：

```bash
python train.py --config ./configs/config.yaml
```

### 推理测试

单张图片推理：

```bash
python inference.py --config ./configs/config.yaml --model_path ./results/patchcore_model_data_dir.pth --image_path ./test_image.jpg --threshold 28.58 --save_path ./inference_results_single/result.png
```

不传 `--threshold` 时只输出异常分数和热力图，不做正常/异常分类。

目录推理要求目录包含 `good/` 和 `defective/`：

```text
test/
├── good/
└── defective/
```

```bash
python infer_data_dir.py --model_path ./results/patchcore_model_data_dir.pth --data_dir ./data/chilun/test
```

目录推理会同时测试良品和不良品，并在 `inference_results_data_dir/metrics/` 生成：

- `metrics.json`：AUROC、AP、精确率、召回率、F1、混淆矩阵和分数统计
- `predictions.csv`：每张图片的真实标签、异常分数和预测结果
- `evaluation_metrics.png`：分数分布、ROC、PR 曲线和混淆矩阵

限制样本数时使用 `--num_samples 100`，脚本会同时抽取良品和不良品。未传入 `--threshold` 时，脚本使用本次评估集上的 F1 最优阈值，仅适合探索；正式部署应在独立验证集上确定阈值后通过 `--threshold` 传入。

### 齿轮贴片 PatchCore 流程

流程拆分为三个独立入口，裁片、训练和推理都通过 `--tile_size` 支持不同尺寸：

- `crop_gear_tiles.py`：裁分正常图片、YOLO 标注数据集或单张推理图片
- `train_gear_patchcore.py`：使用正常图片贴片训练 PatchCore
- `infer_gear_patchcore.py`：逐贴片推理并合并为原尺寸热力图

使用 YOLO 标注数据集，从没有缺陷框相交的贴片中训练：

```powershell
python crop_gear_tiles.py labeled `
  --dataset_root D:\qzp\GDD\GDD\data\annotated_dataset `
  --workspace .\gear_512_workspace `
  --tile_size 512 `
  --overlap 0.25

python train_gear_patchcore.py `
  --workspace .\gear_512_workspace `
  --manifest train_tiles.csv `
  --tile_size 512 `
  --patches_per_tile 256 `
  --tiles_per_image 1 `
  --exclude_image D:\qzp\GDD\GDD\data\annotated_dataset\images\train\example.jpg

python infer_gear_patchcore.py `
  --workspace .\gear_512_workspace `
  --checkpoint .\gear_512_workspace\models\gear_patchcore.pth `
  --image D:\qzp\GDD\GDD\data\annotated_dataset\images\train\example.jpg `
  --tile_size 512 `
  --output_dir .\gear_512_workspace\outputs\example
```

贴片坐标、是否与缺陷框重叠、缺陷类别和缺陷距离记录在 `manifests/*_tiles.csv`。推理时，每个 512 贴片产生 64×64 异常图，再插值回 512×512，并按重叠覆盖次数平均成原始尺寸的热力图。

如果已经有独立正常图片目录，可以直接裁片并训练，不需要缺陷标注：

```powershell
python crop_gear_tiles.py normal `
  --image_dir .\data\chilun\train\good `
  --workspace .\gear_512_good_workspace `
  --tile_size 512 `
  --overlap 0.25

python train_gear_patchcore.py `
  --workspace .\gear_512_good_workspace `
  --manifest good_train_tiles.csv `
  --tile_size 512 `
  --backbone wide_resnet50_2 `
  --patches_per_tile 64 `
  --tiles_per_image 0 `
  --sampling_ratio 0.1 `
  --batch_size 2 `
  --checkpoint_name gear_512_patchcore.pth
```

`--tiles_per_image 0` 表示使用清单中的全部正常贴片。

单张图片推理时，`infer_gear_patchcore.py` 会复用已保存的同尺寸贴片；如果不存在，则自动裁剪：

```powershell
python infer_gear_patchcore.py `
  --workspace .\gear_512_good_workspace `
  --checkpoint .\gear_512_good_workspace\models\gear_512_patchcore.pth `
  --image D:\qzp\GDD\GDD\data\annotated_dataset\images\train\example.jpg `
  --tile_size 512 `
  --retrieval_backend torch `
  --output_dir .\gear_512_good_workspace\outputs\example
```

Memory Bank 检索支持以下后端：

- `torch`：默认后端，在 PyTorch 中构造完整的 patch-to-bank 距离矩阵
- `faiss`：使用 FAISS `IndexFlatL2` 精确最近邻检索
- `auto`：FAISS 可用时使用 FAISS，否则自动回退到 PyTorch

安装可选 FAISS 后端：

```powershell
python -m pip install faiss-cpu
```

FAISS 与 PyTorch 使用相同的欧氏距离。热力图每个 patch 使用 `k=1` 最近邻，官方图像级重加权使用 `k=9` 邻居支持集。当前训练 Coreset 方法为 `approx_greedy`，默认采样比例 `0.1`。

也可以先单独裁目标图片：

```powershell
python crop_gear_tiles.py single `
  --image D:\qzp\GDD\GDD\data\annotated_dataset\images\train\example.jpg `
  --workspace .\gear_512_good_workspace `
  --tile_size 512 `
  --overlap 0.25
```

更换 `--tile_size` 时必须重新裁片和训练。支持的尺寸必须大于等于 128 且能被 32 整除，例如 320、384、512、640、768。

目录推理输出：

- `merged_heatmap_overlay.png`：原图、合并热力图和叠加图
- `merged_heatmap.npy`：原始尺寸的浮点热力图数组
- `tile_scores.csv`：每个贴片的坐标、类型和异常分数
- `inference_summary.json`：贴片数、热力图尺寸、分数范围和文件路径

## 配置说明

配置文件 `configs/config.yaml` 包含以下主要参数：

- `model`: 模型相关配置
  - `backbone`: 特征提取器backbone
  - `layers`: 提取特征的层
  - `patch_size`: patch大小
  - `sampling_ratio`: 采样比例
- `data`: 数据相关配置
  - `image_size`: 输入图像尺寸
  - `batch_size`: 批次大小
  - `num_workers`: 数据加载线程数
- `training`: 训练相关配置
  - `device`: 计算设备
  - `save_path`: 模型保存路径
- `inference`: 推理相关配置
  - `threshold`: 异常阈值
  - `visualize`: 是否可视化结果

## 模型性能

在MVTec AD数据集上的典型性能表现：

| Category | Image-level AUROC | Pixel-level AUROC |
|----------|-------------------|-------------------|
| Bottle   | 98.5%             | 97.2%             |
| Cable    | 95.3%             | 92.1%             |
| Capsule  | 92.7%             | 96.8%             |
| ...      | ...               | ...               |

## 参考文献

```
@inproceedings{roth2022towards,
  title={Towards total recall in industrial anomaly detection},
  author={Roth, Karsten and Pemula, Latha and Zepeda, Joaquin and Brox, Thomas and Drummond, Tom and Favaro, Paolo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14318--14328},
  year={2022}
}
```
