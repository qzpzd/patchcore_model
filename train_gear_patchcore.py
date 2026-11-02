"""Train PatchCore Memory Bank from cropped normal gear tiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from gear_patchcore_common import (
    read_csv,
    row_tile_size,
    select_training_tiles,
    validate_tile_size,
    write_json,
)
from models.patchcore import PatchCore


class TileDataset(Dataset):
    def __init__(self, paths, tile_size):
        self.paths = [str(path) for path in paths]
        self.transform = transforms.Compose(
            [
                transforms.Resize((tile_size, tile_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image), 0


def parse_args():
    parser = argparse.ArgumentParser(description="Train gear PatchCore")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--manifest", default="good_train_tiles.csv")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--sampling_ratio", type=float, default=0.1)
    parser.add_argument("--patches_per_tile", type=int, default=64)
    parser.add_argument(
        "--tiles_per_image",
        type=int,
        default=0,
        help="0 uses every normal tile; positive values cap tiles per source image",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--exclude_image", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_name", default="gear_patchcore.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    validate_tile_size(args.tile_size)
    workspace = Path(args.workspace).resolve()
    manifest = workspace / "manifests" / args.manifest
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    rows = read_csv(manifest)
    mismatched = [row for row in rows if row_tile_size(row) != args.tile_size]
    if mismatched:
        raise ValueError(
            f"{len(mismatched)} tiles in {manifest.name} do not match "
            f"--tile_size {args.tile_size}"
        )
    selected = select_training_tiles(
        rows, args.exclude_image, args.tiles_per_image
    )
    if not selected:
        raise RuntimeError("No normal tiles selected for training")

    source_images = {row["source_image"] for row in selected}
    print(
        f"Selected {len(selected)} normal tiles from {len(source_images)} source images"
    )

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    dataset = TileDataset([row["tile_path"] for row in selected], args.tile_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
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
    )
    model.eval()

    generator = torch.Generator().manual_seed(args.seed)
    feature_chunks = []
    for images, _ in tqdm(loader, desc="Extract tile features"):
        images = images.to(device)
        with torch.no_grad():
            embeddings, _ = model._embed(images)
            embeddings = embeddings.reshape(
                embeddings.shape[0], -1, embeddings.shape[-1]
            )
            patch_count = min(args.patches_per_tile, embeddings.shape[1])
            indices = torch.randperm(
                embeddings.shape[1], generator=generator
            )[:patch_count]
            sampled = embeddings[:, indices, :].reshape(-1, embeddings.shape[-1])
            feature_chunks.append(sampled.cpu())

    features = torch.cat(feature_chunks, dim=0)
    print(f"Features before Coreset: {tuple(features.shape)}")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model.memory_bank = model._coreset_sampling(features)

    checkpoint = workspace / "models" / args.checkpoint_name
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(checkpoint)
    summary = {
        "checkpoint": str(checkpoint),
        "manifest": str(manifest),
        "tile_size": args.tile_size,
        "backbone": args.backbone,
        "training_tiles": len(selected),
        "training_source_images": len(source_images),
        "tiles_per_image": args.tiles_per_image,
        "patches_per_tile": args.patches_per_tile,
        "features_before_coreset": int(features.shape[0]),
        "memory_bank": int(model.memory_bank.shape[0]),
        "sampling_ratio": args.sampling_ratio,
        "coreset_method": model.coreset_method,
        "excluded_image": str(Path(args.exclude_image).resolve())
        if args.exclude_image
        else None,
        "seed": args.seed,
    }
    write_json(workspace / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
