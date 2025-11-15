import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import fitz
import numpy as np
from ultralytics import YOLO

CLASS_NAMES = ["signature", "stamp", "qr"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and visualization with a trained YOLO model.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained YOLO weights (.pt).")
    parser.add_argument("--source", type=Path, required=True, help="Image/PDF file or directory to scan.")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Directory for predictions and visualizations.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--pdf-dpi", type=int, default=200, help="DPI when rasterizing PDFs for inference.")
    return parser.parse_args()


def pdf_to_pages(pdf_path: Path, dpi: int) -> Iterable[Tuple[int, np.ndarray]]:
    doc = fitz.open(pdf_path)
    scale = dpi / 72
    matrix = fitz.Matrix(scale, scale)
    for idx, page in enumerate(doc):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        yield idx + 1, img
    doc.close()


def image_sources(source: Path, dpi: int) -> Iterable[Tuple[str, int, np.ndarray]]:
    if source.is_dir():
        for img_path in sorted(source.glob("**/*")):
            if img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                yield img_path.stem, 1, cv2.imread(str(img_path))
    elif source.suffix.lower() == ".pdf":
        for page_idx, image in pdf_to_pages(source, dpi):
            yield source.stem, page_idx, image
    else:
        yield source.stem, 1, cv2.imread(str(source))


def run_inference(args: argparse.Namespace) -> None:
    model = YOLO(str(args.weights))
    vis_dir = args.output / "visualizations"
    pred_dir = args.output / "predictions"
    vis_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    prediction_index: Dict[str, Dict] = {}

    for doc_id, page_idx, image in image_sources(args.source, args.pdf_dpi):
        if image is None:
            continue
        results = model.predict(source=image, verbose=False, conf=args.conf)
        annotated = results[0].plot()
        vis_path = vis_dir / f"{doc_id}_page-{page_idx:03d}.png"
        cv2.imwrite(str(vis_path), annotated)

        detections = []
        if results[0].boxes is not None and results[0].boxes.data is not None:
            for box in results[0].boxes.data.tolist():
                x1, y1, x2, y2, conf, cls_id = box
                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": round(float(conf), 4),
                        "category": CLASS_NAMES[int(cls_id)],
                    }
                )

        entry = prediction_index.setdefault(doc_id, {})
        entry[f"page_{page_idx}"] = {"detections": detections, "image_path": vis_path.as_posix()}

    for doc_id, payload in prediction_index.items():
        json_path = pred_dir / f"{doc_id}.json"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
