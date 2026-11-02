"""
按亚马逊官网 / Anomalib 常用默认，在 MVTec AD 上训练本仓库 PatchCore。

关键对齐点:
  - 数据集: MVTec AD（非 BTAD）
  - 骨干: wide_resnet50_2（ImageNet 预训练权重）
  - layers: layer2, layer3
  - AvgPool2d(3,1,1)
  - Coreset: approx_greedy_coreset, ratio=0.1
  - num_neighbors: 9
  - 图像: resize=256, imagesize=224（亚马逊 sample_training.sh）

用法:
  python train_official_mvtec.py --category bottle
  python train_official_mvtec.py --category bottle --data_root ./datasets/MVTecAD
"""

from __future__ import annotations

import argparse
import os
import zipfile
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import numpy as np

from models.patchcore import PatchCore


MVTEC_URL = "https://www.mydrive.ch/shares/38536/383402402190eeae16475863273b541/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"


class MVTecSplit(Dataset):
    def __init__(self, root: str, category: str, split: str, image_size: int = 224):
        self.samples = []
        cat_root = Path(root) / category
        if split == "train":
            good = cat_root / "train" / "good"
            for p in sorted(good.glob("*")):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    self.samples.append((str(p), 0))
        else:
            test_root = cat_root / "test"
            for sub in sorted(test_root.iterdir()):
                if not sub.is_dir():
                    continue
                label = 0 if sub.name == "good" else 1
                for p in sorted(sub.glob("*")):
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                        self.samples.append((str(p), label))

        # 官网: resize 256 -> center/imagesize 224
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.long)


def ensure_mvtec_category(data_root: str, category: str):
    cat_dir = Path(data_root) / category
    if (cat_dir / "train" / "good").exists() and (cat_dir / "test").exists():
        print(f"已找到数据集: {cat_dir}")
        return

    print("未找到完整 MVTec 类别目录。")
    print("请手动下载 MVTec AD 并解压到:")
    print(f"  {Path(data_root).resolve()}/<category>/train/good|test/...")
    print("官方数据页: https://www.mvtec.com/company/research/datasets/mvtec-ad")
    print("也可将 bottle 等类别文件夹放到上述路径后重试。")
    raise FileNotFoundError(f"MVTec category not found: {cat_dir}")


def find_f1_threshold(y_true, y_score):
    from sklearn.metrics import precision_recall_curve

    p, r, thr = precision_recall_curve(y_true, y_score)
    if len(thr) == 0:
        return float(np.median(y_score))
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    return float(thr[int(np.argmax(f1))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--data_root", default="./datasets/MVTecAD")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sampling_ratio", type=float, default=0.1)
    parser.add_argument("--num_neighbors", type=int, default=9)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_path", default="./results/official_mvtec")
    args = parser.parse_args()

    ensure_mvtec_category(args.data_root, args.category)
    os.makedirs(args.save_path, exist_ok=True)

    print("=" * 60)
    print("Official-style PatchCore training")
    print(f" dataset   : MVTecAD/{args.category}")
    print(f" backbone  : wide_resnet50_2 (ImageNet pretrained)")
    print(f" layers    : layer2, layer3")
    print(f" coreset   : approx_greedy @ {args.sampling_ratio}")
    print(f" neighbors : {args.num_neighbors}")
    print(f" image     : resize256 -> {args.image_size}")
    print(f" device    : {args.device}")
    print("=" * 60)

    train_set = MVTecSplit(args.data_root, args.category, "train", args.image_size)
    test_set = MVTecSplit(args.data_root, args.category, "test", args.image_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    print(f"train={len(train_set)} test={len(test_set)}")

    model = PatchCore(
        backbone_name="wide_resnet50_2",
        layers=["layer2", "layer3"],
        sampling_ratio=args.sampling_ratio,
        num_neighbors=args.num_neighbors,
        device=args.device,
    )
    model.fit(train_loader)

    ckpt = os.path.join(args.save_path, f"patchcore_{args.category}.pth")
    model.save_model(ckpt)
    print(f"saved: {ckpt}")

    y_true, y_score = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            _, scores = model.predict(images)
            y_true.append(int(labels.item()))
            y_score.append(float(scores.item()))

    auroc = roc_auc_score(y_true, y_score)
    thr = find_f1_threshold(y_true, y_score)
    y_pred = [1 if s > thr else 0 for s in y_score]
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    print(f"Image AUROC: {auroc*100:.2f}%")
    print(f"F1-optimal thr: {thr:.4f}")
    print(f"Precision: {p*100:.2f}%  Recall: {r*100:.2f}%  F1: {f1*100:.2f}%")


if __name__ == "__main__":
    main()
