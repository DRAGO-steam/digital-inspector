"""Streamlit front-end for the Digital Inspector demo."""
import io
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from zipfile import ZipFile

import cv2
import fitz
import numpy as np
import streamlit as st
from ultralytics import YOLO

CLASS_NAMES = ["signature", "stamp", "qr"]


def pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> List[Tuple[int, np.ndarray]]:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        scale = dpi / 72
        matrix = fitz.Matrix(scale, scale)
        results = []
        for idx, page in enumerate(doc):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            results.append((idx + 1, img))
        return results


def sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name).strip("_") or "document"


@st.cache_resource(show_spinner=False)
def load_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


def run_detection(model: YOLO, image: np.ndarray, conf: float, iou: float) -> Tuple[np.ndarray, List[Dict]]:
    preds = model.predict(source=image, verbose=False, conf=conf, iou=iou)
    boxes = preds[0].boxes
    annotated = preds[0].plot()
    detections: List[Dict] = []
    if boxes is not None and boxes.data is not None:
        for x1, y1, x2, y2, score, cls_id in boxes.data.tolist():
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(score),
                    "category": CLASS_NAMES[int(cls_id)],
                }
            )
    return annotated, detections


def main() -> None:
    st.set_page_config(page_title="Digital Inspector", layout="wide")
    st.title("Digital Inspector – Signature, Stamp & QR Detection")

    default_weights = Path("runs/detect/digital_inspector_v8s/weights/best.pt")
    weights_path = st.sidebar.text_input("Weights path", value=str(default_weights))
    conf = st.sidebar.slider("Confidence threshold", min_value=0.1, max_value=0.9, value=0.35, step=0.05)
    iou = st.sidebar.slider("IoU threshold", min_value=0.1, max_value=0.95, value=0.6, step=0.05)
    dpi = st.sidebar.slider("PDF render DPI", min_value=72, max_value=400, value=200, step=10)

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs or images",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("👆 Upload construction documents to get started.")
        return

    if not Path(weights_path).exists():
        st.error("Weights file not found. Train the model first or provide a valid path.")
        return

    model = load_model(weights_path)
    results_container = st.container()
    summary_counts = defaultdict(int)
    per_page_records: List[Dict] = []

    session_root = Path("outputs/app_sessions")
    session_root.mkdir(parents=True, exist_ok=True)
    session_dir = session_root / f"session_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    session_dir.mkdir(parents=True, exist_ok=True)

    with st.spinner("Running detections..."):
        for uploaded in uploaded_files:
            doc_name = uploaded.name or "uploaded_document"
            label = sanitize(doc_name)
            doc_dir = session_dir / label
            doc_dir.mkdir(parents=True, exist_ok=True)
            doc_pages: List[Dict] = []
            doc_class_totals = defaultdict(int)

            file_bytes = uploaded.read()
            if uploaded.type == "application/pdf":
                pages = pdf_to_images(file_bytes, dpi=dpi)
            else:
                np_bytes = np.asarray(bytearray(file_bytes), dtype=np.uint8)
                image = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
                if image is None:
                    st.warning(f"Unable to decode image {doc_name}.")
                    continue
                pages = [(1, image)]

            for page_idx, page_img in pages:
                annotated, detections = run_detection(model, page_img, conf, iou)
                for det in detections:
                    summary_counts[det["category"]] += 1
                    doc_class_totals[det["category"]] += 1

                page_stub = f"page_{page_idx:03d}"
                image_path = doc_dir / f"{page_stub}.png"
                cv2.imwrite(str(image_path), annotated)

                per_page_records.append(
                    {
                        "document": doc_name,
                        "page": page_idx,
                        "detections": len(detections),
                        "image_path": image_path.as_posix(),
                    }
                )
                doc_pages.append({"page": page_idx, "detections": detections})

                with results_container:
                    st.subheader(f"{doc_name} – Page {page_idx}")
                    st.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        caption="Preview (click below for full resolution)",
                        width=500,
                    )
                    with st.expander("View full-size image & metadata", expanded=False):
                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
                        st.json({"page": page_idx, "detections": detections})

            doc_payload = {
                "document": doc_name,
                "pages": doc_pages,
                "class_totals": dict(doc_class_totals),
            }
            doc_json_path = doc_dir / f"{label}_detections.json"
            doc_json_path.write_text(json.dumps(doc_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    st.success("Detection complete.")

    if per_page_records:
        st.write("### Page Summary")
        st.dataframe(per_page_records, use_container_width=True)

    st.write("### Class Totals")
    st.table([[label, summary_counts.get(label, 0)] for label in CLASS_NAMES])

    zip_buffer = io.BytesIO()
    with ZipFile(zip_buffer, "w") as zip_file:
        for file_path in session_dir.rglob("*"):
            if file_path.is_file():
                zip_file.write(file_path, arcname=file_path.relative_to(session_dir))
    zip_buffer.seek(0)

    st.download_button(
        "Download annotated images & detections",
        data=zip_buffer,
        file_name=f"{session_dir.name}.zip",
        mime="application/zip",
    )


if __name__ == "__main__":
    main()
