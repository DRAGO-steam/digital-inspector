# Digital Inspector Presentation Outline

1. **Problem & Motivation**
   - Manual inspection bottlenecks in construction compliance.
   - Need for automated detection of signatures, stamps, and QR codes.
2. **Dataset & Labeling**
   - Source PDFs (engineering drawings, permits, letters).
   - Annotation structure (per-page bounding boxes, 3 key classes).
   - Preprocessing: PDF rasterization @ native resolution, YOLO labels.
3. **Model Architecture**
   - YOLOv8n baseline + fine-tuning details.
   - Data augmentations tailored to documents (HSV, affine, copy-paste).
   - Class balancing + train/val/test strategy.
4. **Training Pipeline**
   - Dataset split, Ultralytics training config.
   - Key hyperparameters, convergence behavior, mAP curves.
   - Infrastructure (GPU spec) and optimization tricks.
5. **Demo & Results**
   - CLI inference (annotated outputs) + Streamlit web UI.
   - Example detections (screenshots) & metrics table.
   - Failure cases + mitigation ideas.
6. **Impact & Next Steps**
   - Batch processing at scale (1k docs) plan.
   - Roadmap: semi-supervised learning, multi-language OCR integration, doc-type classifiers.
   - Vision for integration into Armeta AI platform.
7. **Video Script Cheatsheet**
   - 30s intro (problem + stakes).
   - 60s architecture walkthrough.
   - 60s live demo.
   - 30s results + future vision.
