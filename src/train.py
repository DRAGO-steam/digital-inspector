import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import LOGGER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on the Digital Inspector dataset.")
    parser.add_argument("--data", type=Path, default=Path("configs/digital_inspector.yaml"), help="Path to Ultralytics data YAML.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base YOLOv8 weights to fine-tune.")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=1024, help="Input image size (pixels).")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", type=str, default="auto", help="Device spec (e.g., 0, 0,1, cpu).")
    parser.add_argument("--project", type=Path, default=Path("runs/detect"), help="Training output directory (Ultralytics project).")
    parser.add_argument("--name", type=str, default="digital_inspector", help="Run name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    model = YOLO(args.model)
    LOGGER.info("Starting training with data=%s", data_path)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        patience=20,
        optimizer="SGD",
        cos_lr=True,
        warmup_epochs=3,
        box=7.5,
        cls=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=2,
        translate=0.05,
        scale=0.4,
        shear=0.1,
        perspective=0.0,
        mixup=0.1,
        copy_paste=0.2,
    )


if __name__ == "__main__":
    main()
