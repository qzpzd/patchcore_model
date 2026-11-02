"""
使用 Anomalib 官方 PatchCore + MVTec AD + ImageNet 预训练骨干训练/评估。

官网等价配置:
  backbone=wide_resnet50_2
  layers=[layer2, layer3]
  coreset_sampling_ratio=0.1
  num_neighbors=9
  ImageNet pretrained=True
  数据: MVTec AD（可自动下载）

用法:
  python train_official_anomalib.py --category bottle
  python train_official_anomalib.py --category bottle --data_root ./datasets/MVTecAD
"""

from __future__ import annotations

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Official Anomalib PatchCore on MVTec AD")
    parser.add_argument("--category", type=str, default="bottle", help="MVTec category")
    parser.add_argument(
        "--data_root",
        type=str,
        default="./datasets/MVTecAD",
        help="MVTecAD root (will download if missing)",
    )
    parser.add_argument("--max_epochs", type=int, default=1, help="PatchCore 只需 1 个 epoch")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    args = parser.parse_args()

    try:
        from anomalib.data import MVTecAD
        from anomalib.engine import Engine
        from anomalib.models import Patchcore
    except ImportError:
        print("未安装 anomalib，正在尝试安装...")
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "anomalib", "open-clip-torch", "-q"]
        )
        from anomalib.data import MVTecAD
        from anomalib.engine import Engine
        from anomalib.models import Patchcore

    os.makedirs(args.data_root, exist_ok=True)

    print("=" * 60)
    print("Official Anomalib PatchCore")
    print(f"  category     : {args.category}")
    print(f"  data_root    : {args.data_root}")
    print("  backbone     : wide_resnet50_2 (ImageNet pretrained)")
    print("  layers       : layer2, layer3")
    print("  coreset      : 0.1")
    print("  num_neighbors: 9")
    print("=" * 60)

    datamodule = MVTecAD(
        root=args.data_root,
        category=args.category,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        # Anomalib 会按需下载 MVTec
    )

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    )

    engine = Engine(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        default_root_dir=os.path.join("./results_anomalib", args.category),
    )

    print("开始训练（官方预训练骨干 + MVTec）...")
    engine.fit(model=model, datamodule=datamodule)
    print("开始测试...")
    results = engine.test(model=model, datamodule=datamodule)
    print("测试结果:")
    print(results)
    print(f"日志/权重目录: ./results_anomalib/{args.category}")


if __name__ == "__main__":
    main()
