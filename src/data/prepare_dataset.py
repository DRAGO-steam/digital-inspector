import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
from tqdm import tqdm

TARGET_CLASSES = ["signature", "stamp", "qr"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(TARGET_CLASSES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from annotated construction PDFs.")
    parser.add_argument("--pdf-dir", type=Path, default=Path("pdfs-20251115T111116Z-1-001/pdfs"), help="Directory containing PDF files.")
    parser.add_argument("--annotations", type=Path, default=Path("selected_annotations.json"), help="Path to annotation JSON.")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"), help="Where to store YOLO-formatted dataset.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for dataset split.")
    parser.add_argument("--splits", type=str, default="0.7,0.2,0.1", help="Train, val, test split ratios (comma-separated).")
    return parser.parse_args()


def sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def ensure_dirs(base: Path) -> Dict[str, Tuple[Path, Path]]:
    dirs = {}
    for split in ("train", "val", "test"):
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        dirs[split] = (img_dir, lbl_dir)
    return dirs


def load_annotations(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def assign_splits(items: List[str], ratios: Tuple[float, float, float], seed: int) -> Dict[str, str]:
    random.Random(seed).shuffle(items)
    total = len(items)
    train_end = int(ratios[0] * total)
    val_end = train_end + int(ratios[1] * total)
    split_map = {}
    for idx, item in enumerate(items):
        if idx < train_end:
            split_map[item] = "train"
        elif idx < val_end:
            split_map[item] = "val"
        else:
            split_map[item] = "test"
    return split_map


def render_page(doc: fitz.Document, page_index: int, target_size: Dict[str, float], output_path: Path) -> Tuple[int, int]:
    page = doc[page_index]
    rect = page.rect
    scale_x = target_size["width"] / rect.width
    scale_y = target_size["height"] / rect.height
    matrix = fitz.Matrix(scale_x, scale_y)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    pix.save(output_path.as_posix())
    return pix.width, pix.height


def convert_bbox(bbox: Dict[str, float], page_width: float, page_height: float) -> Tuple[float, float, float, float]:
    cx = bbox["x"] + bbox["width"] / 2
    cy = bbox["y"] + bbox["height"] / 2
    return (cx / page_width, cy / page_height, bbox["width"] / page_width, bbox["height"] / page_height)


def write_yolo_labels(label_path: Path, annotations: List[Dict], page_width: float, page_height: float) -> None:
    lines = []
    for ann in annotations:
        for ann_id, ann_data in ann.items():
            category = ann_data["category"].lower()
            if category not in CLASS_TO_ID:
                continue
            class_id = CLASS_TO_ID[category]
            bbox = convert_bbox(ann_data["bbox"], page_width, page_height)
            line = f"{class_id} {' '.join(f'{value:.6f}' for value in bbox)}"
            lines.append(line)
    label_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    annotations = load_annotations(args.annotations)
    split_values = tuple(float(x.strip()) for x in args.splits.split(","))
    assert abs(sum(split_values) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    doc_names = list(annotations.keys())
    split_map = assign_splits(doc_names, split_values, args.seed)
    dirs = ensure_dirs(args.output_dir)

    stats = {"train": 0, "val": 0, "test": 0}
    total_pages = 0

    for doc_name, pages in tqdm(annotations.items(), desc="Processing documents"):
        pdf_path = args.pdf_dir / doc_name
        if not pdf_path.exists():
            tqdm.write(f"Warning: PDF not found for {doc_name}, skipping.")
            continue
        split = split_map[doc_name]
        img_dir, lbl_dir = dirs[split]
        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:
            tqdm.write(f"Failed to open {pdf_path}: {exc}")
            continue

        for page_key, page_data in pages.items():
            page_idx = int(page_key.split("_")[1]) - 1
            page_size = page_data["page_size"]
            page_annotations = page_data.get("annotations", [])
            image_name = f"{sanitize(doc_name)}__{page_key}.png"
            label_name = image_name.replace(".png", ".txt")
            image_path = img_dir / image_name
            label_path = lbl_dir / label_name

            width, height = render_page(doc, page_idx, page_size, image_path)
            write_yolo_labels(label_path, page_annotations, page_size["width"], page_size["height"])
            total_pages += 1
            stats[split] += 1

        doc.close()

    splits_record = {
        "seed": args.seed,
        "ratios": split_values,
        "documents": split_map,
        "page_counts": stats,
        "total_pages": total_pages,
        "classes": TARGET_CLASSES,
    }
    (args.output_dir / "splits.json").write_text(json.dumps(splits_record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
