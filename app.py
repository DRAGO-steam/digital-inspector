"""Streamlit front-end for the Digital Inspector demo."""
import hashlib
import io
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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
SHORT_LABELS = {"signature": "sig", "stamp": "stp", "qr": "qr"}
BOX_COLORS = {
    "signature": (0, 165, 255),  # orange
    "stamp": (0, 200, 0),        # green
    "qr": (255, 0, 0),           # blue
}
THREAD_LOCAL = threading.local()


def get_thread_model(weights_path: str) -> YOLO:
    """Return a thread-local YOLO instance so workers can run concurrently."""
    cache = getattr(THREAD_LOCAL, "models", {})
    model = cache.get(weights_path)
    if model is None:
        model = YOLO(weights_path)
        cache[weights_path] = model
        THREAD_LOCAL.models = cache
    return model


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


def load_pages(file_bytes: bytes, mime_type: str, dpi: int) -> List[Tuple[int, np.ndarray]]:
    if mime_type == "application/pdf":
        return pdf_to_images(file_bytes, dpi=dpi)
    np_bytes = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    image = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode the provided image bytes.")
    return [(1, image)]


def sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name).strip("_") or "document"


@st.cache_resource(show_spinner=False)
def load_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


def run_detection(model: YOLO, image: np.ndarray, conf: float, iou: float) -> Tuple[np.ndarray, List[Dict]]:
    preds = model.predict(source=image, verbose=False, conf=conf, iou=iou)
    boxes = preds[0].boxes
    annotated = image.copy()
    detections: List[Dict] = []
    if boxes is not None and boxes.data is not None:
        for x1, y1, x2, y2, score, cls_id in boxes.data.tolist():
            category = CLASS_NAMES[int(cls_id)]
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "category": category,
                }
            )

            color = BOX_COLORS.get(category, (0, 255, 0))
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            cv2.rectangle(annotated, pt1, pt2, color, 2)

            label = SHORT_LABELS.get(category, category)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
            text_w, text_h = text_size
            text_x, text_y = pt1
            text_y = max(text_h + baseline + 2, text_y)
            top_left = (text_x, text_y - text_h - baseline - 2)
            bottom_right = (text_x + text_w + 6, text_y)
            cv2.rectangle(annotated, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(
                annotated,
                label,
                (text_x + 3, text_y - baseline - 2),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                lineType=cv2.LINE_AA,
            )
    return annotated, detections


def get_page_preview(page_data: Dict) -> np.ndarray | None:
    annotated_rgb = page_data.get("annotated_rgb")
    if annotated_rgb is not None:
        return annotated_rgb
    image_path = page_data.get("image_path")
    if not image_path:
        return None
    bgr = cv2.imread(image_path)
    if bgr is None:
        return None
    annotated_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    page_data["annotated_rgb"] = annotated_rgb
    return annotated_rgb


def process_document(
    *,
    doc_name: str,
    file_bytes: bytes,
    mime_type: str,
    doc_dir: Path,
    weights_path: str,
    conf: float,
    iou: float,
    dpi: int,
    upload_idx: int,
) -> Dict:
    result = {
        "name": doc_name,
        "doc_dir": doc_dir,
        "upload_idx": upload_idx,
        "pages": [],
        "class_totals": {},
        "per_page": [],
        "total_detections": 0,
        "error": None,
    }

    try:
        pages = load_pages(file_bytes, mime_type, dpi)
    except Exception as exc:  # pragma: no cover - user input errors
        result["error"] = f"Failed to decode {doc_name}: {exc}"
        return result

    doc_dir.mkdir(parents=True, exist_ok=True)
    doc_class_totals = defaultdict(int)
    doc_pages: List[Dict] = []
    doc_pages_json: List[Dict] = []
    per_page_records: List[Dict] = []

    model = get_thread_model(weights_path)

    for page_idx, page_img in pages:
        annotated_bgr, detections = run_detection(model, page_img, conf, iou)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        for det in detections:
            doc_class_totals[det["category"]] += 1

        page_stub = f"page_{page_idx:03d}"
        image_path = doc_dir / f"{page_stub}.png"
        cv2.imwrite(str(image_path), annotated_bgr)

        page_info = {
            "page": page_idx,
            "detections": detections,
            "image_path": image_path.as_posix(),
            "annotated_rgb": annotated_rgb,
        }
        doc_pages.append(page_info)
        doc_pages_json.append({k: v for k, v in page_info.items() if k != "annotated_rgb"})
        per_page_records.append(
            {
                "document": doc_name,
                "page": page_idx,
                "detections": len(detections),
                "image_path": image_path.as_posix(),
            }
        )

    doc_payload = {
        "document": doc_name,
        "pages": doc_pages_json,
        "class_totals": dict(doc_class_totals),
    }
    doc_json_path = doc_dir / f"{sanitize(doc_name)}_detections.json"
    doc_json_path.write_text(json.dumps(doc_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    result.update(
        {
            "pages": sorted(doc_pages, key=lambda item: item["page"]),
            "class_totals": dict(doc_class_totals),
            "per_page": per_page_records,
            "total_detections": sum(len(page["detections"]) for page in doc_pages),
            "json_path": doc_json_path,
            "state_id": f"{upload_idx}_{sanitize(doc_name)}",
        }
    )
    return result


def process_batch(
    documents: List[Dict],
    *,
    weights_path: str,
    conf: float,
    iou: float,
    dpi: int,
    max_workers: int,
) -> Dict:
    session_root = Path("outputs/app_sessions")
    session_root.mkdir(parents=True, exist_ok=True)
    session_dir = session_root / f"session_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    session_dir.mkdir(parents=True, exist_ok=True)

    summary_counts = defaultdict(int)
    per_page_records: List[Dict] = []
    doc_results: List[Dict] = []
    errors: List[str] = []

    for doc in documents:
        doc_dir = session_dir / doc["label"]
        doc["doc_dir"] = doc_dir

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_document,
                doc_name=doc["name"],
                file_bytes=doc["bytes"],
                mime_type=doc["mime"],
                doc_dir=doc["doc_dir"],
                weights_path=weights_path,
                conf=conf,
                iou=iou,
                dpi=dpi,
                upload_idx=doc["upload_idx"],
            )
            for doc in documents
        ]

        for future in as_completed(futures):
            result = future.result()
            if result.get("error"):
                errors.append(result["error"])
                continue
            doc_results.append(result)
            per_page_records.extend(result["per_page"])
            for cls_name, count in result["class_totals"].items():
                summary_counts[cls_name] += count

    return {
        "doc_results": doc_results,
        "per_page_records": per_page_records,
        "summary_counts": dict(summary_counts),
        "errors": errors,
        "session_dir": session_dir.as_posix(),
    }


def main() -> None:
    st.set_page_config(page_title="Digital Inspector", layout="wide")
    st.title("Digital Inspector – Signature, Stamp & QR Detection")

    default_weights = Path("runs/detect/digital_inspector_v8s_archive/weights/best.pt")
    weights_path = st.sidebar.text_input("Weights path", value=str(default_weights))
    conf = st.sidebar.slider("Confidence threshold", min_value=0.1, max_value=0.9, value=0.35, step=0.05)
    iou = st.sidebar.slider("IoU threshold", min_value=0.1, max_value=0.95, value=0.6, step=0.05)
    dpi = st.sidebar.slider("PDF render DPI", min_value=72, max_value=400, value=200, step=10)
    max_workers = st.sidebar.slider(
        "Concurrent workers",
        min_value=1,
        max_value=8,
        value=2,
        help="Number of threads used to process uploads in parallel.",
    )

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

    # Warm up the primary thread to fail fast if weights are invalid.
    load_model(weights_path)

    documents: List[Dict] = []
    file_signatures: List[Tuple[str, int, str]] = []
    for order, uploaded in enumerate(uploaded_files):
        file_bytes = uploaded.getvalue()
        if not file_bytes:
            continue
        name = uploaded.name or f"document_{order+1}"
        label = f"{order:03d}_{sanitize(name)}"
        mime_type = uploaded.type or "application/octet-stream"
        file_hash = hashlib.md5(file_bytes).hexdigest()
        documents.append(
            {
                "name": name,
                "bytes": file_bytes,
                "mime": mime_type,
                "label": label,
                "upload_idx": order,
            }
        )
        file_signatures.append((name, len(file_bytes), file_hash))

    if not documents:
        st.warning("No readable files were provided.")
        return

    input_signature = {
        "weights": weights_path,
        "conf": conf,
        "iou": iou,
        "dpi": dpi,
        "workers": max_workers,
        "files": file_signatures,
    }

    if "di_results" not in st.session_state:
        st.session_state["di_results"] = None
        st.session_state["di_signature"] = None

    run_button = st.button("Run detections", type="primary")

    if run_button:
        with st.spinner("Running threaded detections..."):
            batch_output = process_batch(
                documents,
                weights_path=weights_path,
                conf=conf,
                iou=iou,
                dpi=dpi,
                max_workers=max_workers,
            )
        st.session_state["di_results"] = batch_output
        st.session_state["di_signature"] = input_signature

    results_bundle = st.session_state.get("di_results")

    if not results_bundle:
        st.info("Press **Run detections** to start processing the uploaded documents.")
        return

    stale = st.session_state.get("di_signature") != input_signature
    if stale:
        st.warning("Parameters or files changed since the last run. Click **Run detections** to refresh results.")

    if results_bundle.get("errors"):
        for err in results_bundle["errors"]:
            st.warning(err)

    doc_results = results_bundle.get("doc_results", [])
    if not doc_results:
        st.error("No successful detections to display.")
        return

    per_page_records = results_bundle.get("per_page_records", [])
    summary_counts = defaultdict(int, results_bundle.get("summary_counts", {}))
    session_dir = Path(results_bundle["session_dir"])

    st.success("Detection complete.")

    sort_choice = st.radio(
        "Sort document rows",
        options=[
            "Newest first",
            "Oldest first",
            "Name A→Z",
            "Name Z→A",
            "Most detections",
        ],
        horizontal=True,
        index=0,
    )

    def sort_docs(choice: str, documents_list: List[Dict]) -> List[Dict]:
        if choice == "Newest first":
            return sorted(documents_list, key=lambda d: d["upload_idx"], reverse=True)
        if choice == "Oldest first":
            return sorted(documents_list, key=lambda d: d["upload_idx"])
        if choice == "Name A→Z":
            return sorted(documents_list, key=lambda d: d["name"].lower())
        if choice == "Name Z→A":
            return sorted(documents_list, key=lambda d: d["name"].lower(), reverse=True)
        if choice == "Most detections":
            return sorted(documents_list, key=lambda d: d["total_detections"], reverse=True)
        return documents_list

    st.write("### Annotated documents")
    for doc in sort_docs(sort_choice, doc_results):
        pages = doc.get("pages", [])
        if not pages:
            continue
        page_count = len(pages)
        state_key = f"carousel_{doc['state_id']}"
        if state_key not in st.session_state:
            st.session_state[state_key] = 0

        st.markdown("---")
        info_col, media_col = st.columns([1, 2])

        with media_col:
            nav_cols = st.columns([1, 4, 1])
            with nav_cols[0]:
                if st.button("◀", key=f"prev_{doc['state_id']}"):
                    st.session_state[state_key] = (st.session_state[state_key] - 1) % page_count

            with nav_cols[2]:
                if st.button("▶", key=f"next_{doc['state_id']}"):
                    st.session_state[state_key] = (st.session_state[state_key] + 1) % page_count

            current_index = st.session_state[state_key]
            page_data = pages[current_index]
            page_number = page_data["page"]

            with nav_cols[1]:
                st.caption(f"Page {current_index + 1} / {page_count} (#{page_number})")
                annotated_rgb = get_page_preview(page_data)
                if annotated_rgb is None:
                    st.warning("Annotated image missing on disk.")
                else:
                    st.image(annotated_rgb, caption="Preview", width=320)
                    with st.expander("Open full preview"):
                        st.image(annotated_rgb, caption=f"{doc['name']} – page {page_number}", width="stretch")

        with info_col:
            st.subheader(doc["name"])
            st.metric("Pages", page_count)
            st.metric("Detections (doc)", doc["total_detections"])
            if doc["class_totals"]:
                for cls_name in CLASS_NAMES:
                    st.write(f"{cls_name.capitalize()}: {doc['class_totals'].get(cls_name, 0)}")
            st.caption(f"Detections on page {page_number}: {len(page_data['detections'])}")
            if page_data["detections"]:
                st.json(page_data["detections"])
            else:
                st.caption("No detections on this page")

    if per_page_records:
        st.write("### Page Summary")
        st.dataframe(per_page_records, width="stretch")

    st.write("### Class Totals")
    st.table([[label, summary_counts.get(label, 0)] for label in CLASS_NAMES])

    if not session_dir.exists():
        st.warning("Output directory no longer exists on disk; rerun detections to regenerate artifacts.")
    else:
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
