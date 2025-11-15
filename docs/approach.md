# Digital Inspector Plan

## Objectives
- Detect signatures, stamps/seals, and QR codes on scanned construction documents.
- Provide interpretable visual outputs (PDF-to-image with bounding boxes, lightweight web/CLI demo).
- Ensure training/inference pipeline is reproducible within 24-hour hackathon constraints.

## Dataset Understanding
- Input PDFs located in `pdfs-20251115T111116Z-1-001/pdfs` (A3/A4 construction docs).
- Ground-truth annotations stored in `selected_annotations.json` (target classes only) and `masked_annotations.json` (additional labels not currently needed).
- Each entry: filename → page index → page size + annotations (bbox in absolute pixel coordinates, `category` ∈ {`signature`, `stamp`, `qr`} for selected annotations).
- Need to rasterize PDFs per page before training.

## Proposed Pipeline
1. **PDF Rasterization & Dataset Split**
   - Convert every relevant PDF page to 300–400 DPI PNG using `pdf2image` (or Poppler) for consistent resolution.
   - Store under `dataset/images/{train,val,test}` with filenames `document_page-<n>.png`.
   - Normalize bounding boxes to YOLO TXT format (`class cx cy w h` in relative coordinates) using page size metadata.
   - Split by document to avoid leakage (e.g., 70/15/15) and write split manifest.

2. **Model Architecture**
   - Fine-tune `yolov8n.pt` (Ultralytics) for speed, optionally `yolov8s` if GPU available.
   - 3 custom classes: `signature`, `stamp`, `qr`.
   - Use data augmentation (HSV shift, affine, copy-paste) to handle document variance.
   - Weighted loss or class-balancing through oversampling (QR count high; signatures mid; stamps mid) as needed.

3. **Training Strategy**
   - 50–100 epochs with early stopping on validation mAP (IoU 0.5:0.95).
   - Save best + last checkpoints under `runs/detect`.
   - Track metrics via Ultralytics training logs and export to Markdown for presentation.

4. **Inference & Visualization**
   - Provide CLI script `python src/infer.py --weights runs/detect/best.pt --source pdfs/.../sample.pdf` that:
     1. Rasterizes PDF pages on the fly.
     2. Runs YOLO inference per page.
     3. Saves annotated images under `outputs/visualizations/<pdf_name>/page-#.png`.
   - Optionally stitch into PDF.

5. **Demo UX**
   - Notebook or Streamlit mini-app (`app.py`) enabling drag-and-drop image/PDF upload and showing detections.
   - Provide fallback CLI if GUI not run.

6. **Evaluation**
   - Use held-out validation/test splits; compute mAP + per-class precision/recall via Ultralytics.
   - Export JSON predictions for scoring format once provided (TBD) and include sample under `outputs/predictions/`.

7. **Documentation & Presentation**
   - `README.md` with setup, training, inference, and demo instructions (pip/conda requirements, GPU notes).
   - Slide deck outline (key findings, architecture diagram, metrics, demo screenshots) saved under `docs/presentation-outline.md` (to be created).
   - Video script guidelines referencing same flow.

## Tech Stack
- Python 3.10+
- Ultralytics YOLOv8
- pdf2image + Poppler (or fitz/PyMuPDF) for rasterization
- OpenCV / Pillow for visualization
- Streamlit (optional) for interactive demo

## Next Steps
1. Implement data prep script(s) under `src/data/` to convert JSON annotations into YOLO format.
2. Configure Ultralytics `data.yaml` file pointing to generated splits and classes.
3. Add training/inference scripts + Streamlit app.
4. Document setup, provide presentation outline, and prepare video instructions.
