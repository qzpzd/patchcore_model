"""Crop normal, labeled, or single gear images into configurable tiles."""

from __future__ import annotations

import argparse
from pathlib import Path

from gear_patchcore_common import (
    crop_labeled_split,
    crop_normal_images,
    list_images,
    write_json,
)


def add_common_crop_args(parser):
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--jpeg_quality", type=int, default=95)


def command_normal(args):
    images = list_images(Path(args.image_dir))
    rows = crop_normal_images(
        images,
        Path(args.workspace),
        args.tile_subdir,
        args.manifest_name,
        args.tile_size,
        args.overlap,
        args.jpeg_quality,
    )
    write_json(
        Path(args.workspace) / "crop_summary.json",
        {
            "mode": "normal",
            "image_dir": str(Path(args.image_dir).resolve()),
            "images": len(images),
            "tiles": len(rows),
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "manifest": args.manifest_name,
        },
    )


def command_labeled(args):
    dataset_root = Path(args.dataset_root)
    total = 0
    for split in args.splits:
        rows = crop_labeled_split(
            dataset_root,
            Path(args.workspace),
            split,
            args.tile_size,
            args.overlap,
            args.margin,
            args.jpeg_quality,
        )
        total += len(rows)
    write_json(
        Path(args.workspace) / "crop_summary.json",
        {
            "mode": "labeled",
            "dataset_root": str(dataset_root.resolve()),
            "splits": args.splits,
            "tiles": total,
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "margin": args.margin,
        },
    )


def command_single(args):
    image = Path(args.image)
    rows = crop_normal_images(
        [image],
        Path(args.workspace),
        args.tile_subdir,
        args.manifest_name,
        args.tile_size,
        args.overlap,
        args.jpeg_quality,
    )
    write_json(
        Path(args.workspace) / "crop_summary_single.json",
        {
            "mode": "single",
            "image": str(image.resolve()),
            "tiles": len(rows),
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "manifest": args.manifest_name,
        },
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Crop gear images into tiles")
    subparsers = parser.add_subparsers(dest="command", required=True)

    normal = subparsers.add_parser("normal", help="Crop a normal/good image folder")
    normal.add_argument("--image_dir", required=True)
    add_common_crop_args(normal)
    normal.add_argument("--tile_subdir", default="good_train")
    normal.add_argument("--manifest_name", default="good_train_tiles.csv")
    normal.set_defaults(func=command_normal)

    labeled = subparsers.add_parser("labeled", help="Crop a YOLO labeled dataset")
    labeled.add_argument("--dataset_root", required=True)
    add_common_crop_args(labeled)
    labeled.add_argument("--splits", nargs="+", default=["train", "val"])
    labeled.add_argument("--margin", type=int, default=16)
    labeled.set_defaults(func=command_labeled)

    single = subparsers.add_parser("single", help="Crop one inference image")
    single.add_argument("--image", required=True)
    add_common_crop_args(single)
    single.add_argument("--tile_subdir", default="target")
    single.add_argument("--manifest_name", default="target_tiles.csv")
    single.set_defaults(func=command_single)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
