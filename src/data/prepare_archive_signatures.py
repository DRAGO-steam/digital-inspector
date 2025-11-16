import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the archive(1) dataset into a YOLO-ready signature subset."
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("archive(1)"),
        help="Root folder containing images/, train.csv, test.csv, and image_ids.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_archive_signatures"),
        help="Destination folder for YOLO images/ and labels/ splits.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of archive train images to reserve for validation (0-0.4 recommended).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for reproducible split.",
    )
    parser.add_argument(
        "--yolo-class-index",
        type=int,
        default=0,
        help="YOLO class index that should receive the archive signature annotations.",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Also export archive test.csv rows as a held-out test split.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite images if the output already exists (default skips copies).",
    )
    return parser.parse_args()


def load_image_metadata(meta_csv: Path) -> Dict[int, Dict[str, str]]:
    metadata: Dict[int, Dict[str, str]] = {}
    with meta_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = int(row["id"])
            metadata[image_id] = {
                "file_name": row["file_name"],
                "width": float(row["width"]),
                "height": float(row["height"]),
            }
    return metadata


def parse_bbox(raw_bbox: str) -> Tuple[float, float, float, float]:
    values = json.loads(raw_bbox)
    if len(values) != 4:
        raise ValueError(f"Expected 4 bbox values, received {values}")
    return tuple(float(v) for v in values)  # type: ignore[return-value]


def gather_annotations(csv_path: Path) -> Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]]:
    by_image: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = defaultdict(list)
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cat_id = int(row["category_id"])
            if cat_id != 1:
                continue  # only keep signatures
            image_id = int(row["image_id"])
            bbox = parse_bbox(row["bbox"])
            by_image[image_id].append((cat_id, bbox))
    return by_image


def ensure_split_dirs(base: Path) -> Dict[str, Tuple[Path, Path]]:
    layout: Dict[str, Tuple[Path, Path]] = {}
    for split in ("train", "val", "test"):
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        layout[split] = (img_dir, lbl_dir)
    return layout


def split_train_ids(image_ids: Iterable[int], val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    ids = list(image_ids)
    if not ids or val_fraction <= 0:
        return ids, []
    rng = random.Random(seed)
    rng.shuffle(ids)
    val_count = max(1, int(len(ids) * val_fraction))
    val_ids = ids[:val_count]
    train_ids = ids[val_count:]
    return train_ids, val_ids


def format_label_line(yolo_class: int, bbox: Tuple[float, float, float, float]) -> str:
    x, y, w, h = bbox
    cx = min(max(x + w / 2.0, 0.0), 1.0)
    cy = min(max(y + h / 2.0, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)
    return f"{yolo_class} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def export_split(
    split_name: str,
    ids: Iterable[int],
    annotations: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]],
    metadata: Dict[int, Dict[str, str]],
    dirs: Dict[str, Tuple[Path, Path]],
    images_dir: Path,
    yolo_class: int,
    overwrite: bool,
) -> Tuple[int, int]:
    exported_images = 0
    exported_boxes = 0
    img_dir, lbl_dir = dirs[split_name]
    for image_id in tqdm(list(ids), desc=f"Writing {split_name}", unit="img"):
        records = annotations.get(image_id)
        if not records:
            continue
        info = metadata.get(image_id)
        if info is None:
            tqdm.write(f"Skipping image_id={image_id}: metadata missing")
            continue
        src_img = images_dir / info["file_name"]
        if not src_img.exists():
            tqdm.write(f"Missing source image: {src_img}")
            continue
        dest_img = img_dir / info["file_name"]
        if overwrite or not dest_img.exists():
            shutil.copy2(src_img, dest_img)
        label_lines = [format_label_line(yolo_class, bbox) for _, bbox in records]
        if not label_lines:
            continue
        label_path = lbl_dir / f"{dest_img.stem}.txt"
        label_path.write_text("\n".join(label_lines), encoding="utf-8")
        exported_images += 1
        exported_boxes += len(label_lines)
    return exported_images, exported_boxes


def main() -> None:
    args = parse_args()
    archive_root = args.archive_root.resolve()
    output_dir = args.output_dir.resolve()
    images_dir = archive_root / "images"
    train_csv = archive_root / "train.csv"
    test_csv = archive_root / "test.csv"
    meta_csv = archive_root / "image_ids.csv"

    if not images_dir.exists():
        raise FileNotFoundError(f"Could not find images directory at {images_dir}")

    metadata = load_image_metadata(meta_csv)
    train_ann = gather_annotations(train_csv)
    test_ann = gather_annotations(test_csv)

    train_ids, val_ids = split_train_ids(train_ann.keys(), args.val_fraction, args.seed)

    dirs = ensure_split_dirs(output_dir)
    summary = {
        "source": str(archive_root),
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "yolo_class_index": args.yolo_class_index,
        "include_test": args.include_test,
        "splits": {},
    }

    train_stats = export_split(
        "train",
        train_ids,
        train_ann,
        metadata,
        dirs,
        images_dir,
        args.yolo_class_index,
        args.overwrite,
    )
    summary["splits"]["train"] = {"images": train_stats[0], "boxes": train_stats[1]}

    val_stats = export_split(
        "val",
        val_ids,
        train_ann,
        metadata,
        dirs,
        images_dir,
        args.yolo_class_index,
        args.overwrite,
    )
    summary["splits"]["val"] = {"images": val_stats[0], "boxes": val_stats[1]}

    if args.include_test:
        test_ids = sorted(test_ann.keys())
        test_stats = export_split(
            "test",
            test_ids,
            test_ann,
            metadata,
            dirs,
            images_dir,
            args.yolo_class_index,
            args.overwrite,
        )
        summary["splits"]["test"] = {"images": test_stats[0], "boxes": test_stats[1]}

    (output_dir / "splits_archive.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
