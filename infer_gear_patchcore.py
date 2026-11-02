"""Infer a full gear image by tiles and merge tile heatmaps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from gear_patchcore_common import (
    crop_normal_images,
    load_tile_rows,
    row_tile_size,
    validate_tile_size,
    write_json,
)
from models.patchcore import PatchCore


def parse_args():
    parser = argparse.ArgumentParser(description="Infer and merge gear tiles")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--sampling_ratio", type=float, default=0.1)
    parser.add_argument(
        "--retrieval_backend",
        choices=["torch", "faiss", "auto"],
        default="torch",
        help="Exact KNN backend: torch dense search, FAISS, or automatic fallback",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def ensure_target_tiles(workspace, image_path, tile_size, overlap):
    rows = load_tile_rows(workspace, str(image_path), tile_size)
    if rows:
        return rows
    crop_normal_images(
        [image_path],
        workspace,
        f"target_{tile_size}",
        f"target_{image_path.stem}_{tile_size}_tiles.csv",
        tile_size,
        overlap,
    )
    rows = load_tile_rows(workspace, str(image_path), tile_size)
    if not rows:
        raise RuntimeError(f"Failed to prepare target tiles for {image_path}")
    return rows


def main():
    args = parse_args()
    validate_tile_size(args.tile_size)
    workspace = Path(args.workspace).resolve()
    image_path = Path(args.image).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else workspace / "outputs" / f"{image_path.stem}_tile{args.tile_size}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = ensure_target_tiles(workspace, image_path, args.tile_size, args.overlap)
    detected_sizes = {row_tile_size(row) for row in rows}
    if detected_sizes != {args.tile_size}:
        raise ValueError(f"Tile size mismatch: {detected_sizes} vs {args.tile_size}")

    first = rows[0]
    width = int(first["source_width"])
    height = int(first["source_height"])
    tile_size = args.tile_size
    device = torch.device(
        args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    )

    model = PatchCore(
        backbone_name=args.backbone,
        layers=["layer2", "layer3"],
        patch_size=3,
        sampling_ratio=args.sampling_ratio,
        num_neighbors=9,
        device=device,
        variant="official",
        coreset_method="approx_greedy",
        retrieval_backend=args.retrieval_backend,
    )
    model.load_model(args.checkpoint)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((tile_size, tile_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    heat_sum = np.zeros((height, width), dtype=np.float64)
    coverage = np.zeros((height, width), dtype=np.float64)
    tile_results = []
    for index, row in enumerate(tqdm(rows, desc="Infer tiles")):
        with Image.open(row["tile_path"]) as tile:
            tensor = transform(tile.convert("RGB")).unsqueeze(0).to(device)
        anomaly_map, image_score = model.predict(tensor)
        anomaly_map = anomaly_map[0].unsqueeze(0).unsqueeze(0)
        anomaly_map = F.interpolate(
            anomaly_map,
            size=(tile_size, tile_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0].cpu().numpy()

        x0, y0 = int(row["x0"]), int(row["y0"])
        heat_sum[y0 : y0 + tile_size, x0 : x0 + tile_size] += anomaly_map
        coverage[y0 : y0 + tile_size, x0 : x0 + tile_size] += 1.0
        tile_results.append(
            {
                "tile_index": index,
                "tile_path": row["tile_path"],
                "x0": x0,
                "y0": y0,
                "x1": row["x1"],
                "y1": row["y1"],
                "is_normal_tile": row["is_normal_tile"],
                "overlap_class_ids": row["overlap_class_ids"],
                "anomaly_score": float(image_score.item()),
            }
        )

    merged_heatmap = heat_sum / np.maximum(coverage, 1.0)
    np.save(output_dir / "merged_heatmap.npy", merged_heatmap)
    with (output_dir / "tile_scores.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tile_results[0].keys()))
        writer.writeheader()
        writer.writerows(tile_results)

    original = Image.open(image_path).convert("RGB")
    original_np = np.asarray(original)
    vmin, vmax = np.percentile(merged_heatmap, [1, 99])
    if vmax <= vmin:
        vmax = vmin + 1e-6
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    axes[0].imshow(original_np)
    axes[0].set_title("Original image")
    axes[0].axis("off")
    heat = axes[1].imshow(
        merged_heatmap, cmap="jet", vmin=vmin, vmax=vmax
    )
    axes[1].set_title(f"Merged {tile_size}-tile anomaly heatmap")
    axes[1].axis("off")
    plt.colorbar(heat, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(original_np)
    axes[2].imshow(
        merged_heatmap,
        cmap="jet",
        alpha=0.35,
        vmin=vmin,
        vmax=vmax,
    )
    axes[2].set_title("Overlay (alpha=0.35)")
    axes[2].axis("off")
    fig.tight_layout()
    result_image = output_dir / "merged_heatmap_overlay.png"
    fig.savefig(result_image, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = {
        "image": str(image_path),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "tile_size": tile_size,
        "tile_count": len(rows),
        "tile_heatmap_shape": [tile_size // 8, tile_size // 8],
        "merged_heatmap_shape": [height, width],
        "tile_score_max": max(item["anomaly_score"] for item in tile_results),
        "tile_score_mean": float(
            np.mean([item["anomaly_score"] for item in tile_results])
        ),
        "merge_method": "Uniform overlap average by coverage count",
        "retrieval_backend": model.retrieval_backend,
        "heatmap_min": float(merged_heatmap.min()),
        "heatmap_max": float(merged_heatmap.max()),
        "result_image": str(result_image),
        "raw_heatmap": str(output_dir / "merged_heatmap.npy"),
    }
    write_json(output_dir / "inference_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
