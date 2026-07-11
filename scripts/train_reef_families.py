"""Fine-tune YOLOv8s on the 13-class reef fish families dataset.

Classes: Surgeonfishes, Triggerfishes, Jacks, Spadefishes, Wrasse,
Snappers, Angelfishes, Damselfishes, Parrotfishes, Tunas, Groupers,
Sharks, Moorish Idol.

Using YOLOv8s (small) instead of nano for better accuracy on a
relatively small dataset.
"""
from ultralytics import YOLO
import pathlib

DATA = pathlib.Path(__file__).resolve().parent.parent / "datasets" / "reef_families" / "data.yaml"


def main():
    model = YOLO("yolov8s.pt")

    model.train(
        data=str(DATA),
        epochs=120,
        patience=20,
        batch=16,
        imgsz=640,
        device=0,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        workers=2,
        cache="ram",
        augment=True,
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.3,
        flipud=0.3,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        project="runs/reef_families",
        name="train",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
