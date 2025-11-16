# Digital Inspector – Signatures, Stamps & QR Detection

Computer-vision pipeline for Armeta AI's "Digital Inspector" challenge. The project converts annotated construction PDFs into a YOLOv8 dataset, fine-tunes the detector, and serves results through both CLI utilities and a Streamlit demo with visually interpretable outputs.

## Repo layout

```
.
├── app.py                         # Streamlit front-end demo
├── configs/
│   └── digital_inspector.yaml     # Ultralytics data config
├── docs/
│   ├── approach.md                # Detailed architecture + plan
│   └── presentation-outline.md    # Slide/video storytelling cues
├── outputs/                       # Visualizations & prediction JSONs (generated)
├── requirements.txt               # Python dependencies
├── src/
│   ├── data/
│   │   └── prepare_dataset.py     # PDF rasterization + YOLO label export
│   ├── infer.py                   # Batch inference + visualization writer
│   └── train.py                   # YOLOv8 fine-tuning script
├── masked_annotations.json        # Extended labels (not used directly)
├── selected_annotations.json      # Ground-truth for target classes
└── pdfs-20251115T111116Z-1-001/   # Raw PDFs from Armeta AI
```

## Setup

1. Install Python 3.10+ and (optional) GPU-enabled PyTorch.
2. Install deps:

```powershell
pip install -r requirements.txt
```

Ultralytics will pull PyTorch automatically if absent, but GPU users should install the appropriate CUDA build beforehand.

## 1️⃣ Dataset preparation

Convert PDFs + annotations into YOLO format (images + TXT labels). By default the script expects the provided folder structure; override paths as needed.

```powershell
python src/data/prepare_dataset.py \
  --pdf-dir pdfs-20251115T111116Z-1-001/pdfs \
  --annotations selected_annotations.json \
  --output-dir dataset \
  --splits 0.7,0.2,0.1
```

Key behavior:
- Rasterizes only annotated pages at their native resolution using PyMuPDF (no Poppler dependency).
- Creates `dataset/images/{train,val,test}` + matching `labels/` subfolders.
- Writes `dataset/splits.json` summarizing doc-level splits for reproducibility.

## 2️⃣ Training

Fine-tune YOLOv8 (nano by default) using Ultralytics' Python API. Customize hyperparameters through CLI flags.

```powershell
python src/train.py \
  --data configs/digital_inspector.yaml \
  --model yolov8n.pt \
  --epochs 80 \
  --imgsz 1024 \
  --batch 8 \
  --device 0
```

Outputs land in `runs/detect/digital_inspector` (best + last checkpoints, metrics, PR/mAP curves). Adjust `--model` to `yolov8s.pt` if GPU budget allows.

Need to keep training `digital_inspector_v8s` but emphasize signatures? Reuse the best checkpoint as a starting point while pointing Ultralytics to the mixed dataset config that concatenates the PDF-derived set with the archive signatures:

```powershell
python src/train.py \
  --data configs/digital_inspector_signature_mix.yaml \
  --model runs/detect/digital_inspector_v8s/weights/best.pt \
  --epochs 40 \
  --imgsz 1536 \
  --batch 8 \
  --device 0 \
  --name digital_inspector_v8s_archive
```

This keeps the original stamp/QR classes intact (the head still has three outputs) while injecting hundreds of extra signature examples, which helps push recall without catastrophic forgetting. Consider shortening `patience` (default 20) once loss plateaus to keep the warm restart snappy.

## 3️⃣ Inference & visualization

Generate annotated images plus structured JSON predictions for any folder, image, or PDF.

```powershell
python src/infer.py \
  --weights runs/detect/digital_inspector/weights/best.pt \
  --source pdfs-20251115T111116Z-1-001/pdfs/отр-1.pdf \
  --output outputs \
  --conf 0.3
```

Results:
- `outputs/visualizations/{doc}_page-XXX.png` → bounding boxes overlayed per page.
- `outputs/predictions/{doc}.json` → page-wise detection metadata for downstream scoring.

## 4️⃣ Streamlit demo (visual requirement)

Launch a lightweight UI to drag-and-drop PDFs or images (multi-file supported) and inspect detections instantly.

```powershell
streamlit run app.py
```

Features:
- Adjustable confidence, IoU, and PDF DPI controls (handy for tiny signatures/QRs).
- Batch upload: drop in several PDFs/images at once; each page is rendered with YOLOv8 detections.
- Automatic export: annotated page PNGs + JSON metadata are saved under `outputs/app_sessions/<timestamp>/` and can be downloaded as a ZIP directly from the app.
- Embedded summary tables covering per-page detection counts and per-class totals.
- Defaults to the stronger `runs/detect/digital_inspector_v8s_1536/weights/best.pt` checkpoint, but any `.pt` file can be selected via the sidebar.

## 5️⃣ Presentation & video assets

Use the outlines in `docs/` to craft the required deliverables:
- **Slide deck**: follow `docs/presentation-outline.md` (problem → data → model → demo → roadmap).
- **Video (≤3 min)**: reuse the outline's "Video Script" section for narration timing.
- **Vision slide**: highlight scaling strategy (batch processing, OCR fusion, active learning, etc.).

Suggested visuals:
- Pipeline diagram from `docs/approach.md`.
- Training curves (`runs/detect/.../results.png`).
- Before/after detection screenshots (from `outputs/visualizations`).
- Short clip of the Streamlit app running an example PDF.

## Next steps / enhancements

- Integrate automatic metric export (mAP, per-class PR) into `infer.py` for batch scoring once the official evaluation format is released.
- Add data augmentation scripts for signature scarcity (copy-paste) and synthetic QR injection.
- Package model + app into a Docker image for easier submission.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ImportError: ultralytics` | Requirements not installed | `pip install -r requirements.txt` |
| Empty detection outputs | Confidence too high or weights missing | Lower `--conf` / double-check weights path |
| PDF renders stretched | Page size metadata mismatch | Use default DPI path (no target size) in `prepare_dataset.py` or regenerate annotations |

## License & Credits

Dataset and task provided by Armeta AI (challenge terms apply). Code released for hackathon use—adapt as needed for submission.
