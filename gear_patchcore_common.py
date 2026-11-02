"""Shared tile and manifest utilities for the gear PatchCore pipeline."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MANIFEST_FIELDS = [
    "split",
    "source_image",
    "source_stem",
    "tile_path",
    "tile_name",
    "tile_size",
    "x0",
    "y0",
    "x1",
    "y1",
    "source_width",
    "source_height",
    "is_normal_tile",
    "overlap_class_ids",
    "defect_clearance",
]


def validate_tile_size(tile_size: int):
    if tile_size < 128 or tile_size % 32 != 0:
        raise ValueError("tile_size must be >= 128 and divisible by 32")


def stride_from_overlap(tile_size: int, overlap: float) -> int:
    validate_tile_size(tile_size)
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    stride = int(tile_size * (1 - overlap))
    if stride <= 0:
        raise ValueError("overlap produces an invalid stride")
    return stride


def tile_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length < tile_size:
        raise ValueError(f"Image side {length} is smaller than tile size {tile_size}")
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(MANIFEST_FIELDS)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_yolo_boxes(label_path: Path, width: int, height: int):
    boxes = []
    if not label_path.exists():
        return boxes
    for raw_line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
        except ValueError:
            continue
        boxes.append(
            (
                class_id,
                max(0.0, (cx - bw / 2) * width),
                max(0.0, (cy - bh / 2) * height),
                min(float(width), (cx + bw / 2) * width),
                min(float(height), (cy + bh / 2) * height),
            )
        )
    return boxes


def box_intersects_tile(box, x0, y0, tile_size, margin):
    _, bx1, by1, bx2, by2 = box
    tx1 = max(0.0, x0 - margin)
    ty1 = max(0.0, y0 - margin)
    tx2 = x0 + tile_size + margin
    ty2 = y0 + tile_size + margin
    return max(0.0, min(bx2, tx2) - max(bx1, tx1)) > 0 and max(
        0.0, min(by2, ty2) - max(by1, ty1)
    ) > 0


def defect_clearance(boxes, x0, y0, tile_size):
    if not boxes:
        return float("inf")
    tx = x0 + tile_size / 2
    ty = y0 + tile_size / 2
    distances = []
    for _, x1, y1, x2, y2 in boxes:
        nx = min(max(tx, x1), x2)
        ny = min(max(ty, y1), y2)
        distances.append(math.hypot(tx - nx, ty - ny))
    return min(distances)


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    images = sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return images


def crop_normal_images(
    image_paths,
    workspace: Path,
    tile_subdir: str,
    manifest_name: str,
    tile_size: int,
    overlap: float,
    jpeg_quality: int = 95,
) -> list[dict]:
    stride = stride_from_overlap(tile_size, overlap)
    tile_dir = workspace / "tiles" / tile_subdir
    tile_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_path in tqdm(list(image_paths), desc=f"Crop {tile_subdir}"):
        image_path = Path(image_path)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            for y0 in tile_positions(height, tile_size, stride):
                for x0 in tile_positions(width, tile_size, stride):
                    tile_path = tile_dir / (
                        f"{image_path.stem}__x{x0:04d}_y{y0:04d}.jpg"
                    )
                    if not tile_path.exists():
                        image.crop((x0, y0, x0 + tile_size, y0 + tile_size)).save(
                            tile_path,
                            quality=jpeg_quality,
                            subsampling=0,
                            optimize=True,
                        )
                    rows.append(
                        {
                            "split": tile_subdir,
                            "source_image": str(image_path.resolve()),
                            "source_stem": image_path.stem,
                            "tile_path": str(tile_path),
                            "tile_name": tile_path.name,
                            "tile_size": tile_size,
                            "x0": x0,
                            "y0": y0,
                            "x1": x0 + tile_size,
                            "y1": y0 + tile_size,
                            "source_width": width,
                            "source_height": height,
                            "is_normal_tile": 1,
                            "overlap_class_ids": "",
                            "defect_clearance": float("inf"),
                        }
                    )
    write_csv(workspace / "manifests" / manifest_name, rows)
    print(
        f"{tile_subdir}: images={len(image_paths)} tiles={len(rows)} "
        f"manifest={workspace / 'manifests' / manifest_name}"
    )
    return rows


def crop_labeled_split(
    dataset_root: Path,
    workspace: Path,
    split: str,
    tile_size: int,
    overlap: float,
    margin: int = 16,
    jpeg_quality: int = 95,
) -> list[dict]:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    images = list_images(image_dir)
    stride = stride_from_overlap(tile_size, overlap)
    tile_dir = workspace / "tiles" / split
    tile_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_path in tqdm(images, desc=f"Crop labeled {split}"):
        label_path = label_dir / f"{image_path.stem}.txt"
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            boxes = read_yolo_boxes(label_path, width, height)
            for y0 in tile_positions(height, tile_size, stride):
                for x0 in tile_positions(width, tile_size, stride):
                    overlap_ids = [
                        box[0]
                        for box in boxes
                        if box_intersects_tile(box, x0, y0, tile_size, margin)
                    ]
                    tile_path = tile_dir / (
                        f"{image_path.stem}__x{x0:04d}_y{y0:04d}.jpg"
                    )
                    if not tile_path.exists():
                        image.crop((x0, y0, x0 + tile_size, y0 + tile_size)).save(
                            tile_path,
                            quality=jpeg_quality,
                            subsampling=0,
                            optimize=True,
                        )
                    rows.append(
                        {
                            "split": split,
                            "source_image": str(image_path.resolve()),
                            "source_stem": image_path.stem,
                            "tile_path": str(tile_path),
                            "tile_name": tile_path.name,
                            "tile_size": tile_size,
                            "x0": x0,
                            "y0": y0,
                            "x1": x0 + tile_size,
                            "y1": y0 + tile_size,
                            "source_width": width,
                            "source_height": height,
                            "is_normal_tile": int(not overlap_ids),
                            "overlap_class_ids": ",".join(map(str, sorted(set(overlap_ids)))),
                            "defect_clearance": round(
                                defect_clearance(boxes, x0, y0, tile_size), 4
                            ),
                        }
                    )
    write_csv(workspace / "manifests" / f"{split}_tiles.csv", rows)
    print(
        f"{split}: images={len(images)} tiles={len(rows)} "
        f"normal={sum(int(r['is_normal_tile']) for r in rows)}"
    )
    return rows


def row_tile_size(row: dict) -> int:
    if row.get("tile_size"):
        return int(row["tile_size"])
    return int(row["x1"]) - int(row["x0"])


def select_training_tiles(rows, exclude_image, tiles_per_image):
    grouped = defaultdict(list)
    exclude_image = str(Path(exclude_image).resolve()) if exclude_image else None
    for row in rows:
        if not int(row["is_normal_tile"]):
            continue
        source = str(Path(row["source_image"]).resolve())
        if exclude_image and source == exclude_image:
            continue
        grouped[source].append(row)
    selected = []
    for source in sorted(grouped):
        candidates = sorted(
            grouped[source],
            key=lambda row: float(row["defect_clearance"]),
            reverse=True,
        )
        selected.extend(candidates if tiles_per_image <= 0 else candidates[:tiles_per_image])
    return selected


def load_tile_rows(workspace: Path, image_path: str, tile_size: int) -> list[dict]:
    source = str(Path(image_path).resolve())
    rows = []
    for manifest in sorted((workspace / "manifests").glob("*_tiles.csv")):
        for row in read_csv(manifest):
            if (
                str(Path(row["source_image"]).resolve()) == source
                and row_tile_size(row) == tile_size
            ):
                rows.append(row)
    deduplicated = {(row["x0"], row["y0"]): row for row in rows}
    return sorted(
        deduplicated.values(),
        key=lambda row: (int(row["y0"]), int(row["x0"])),
    )
