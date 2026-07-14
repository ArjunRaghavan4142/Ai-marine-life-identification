"""Fine-tune YOLOv8n on the 481-class fish species dataset."""
from ultralytics import YOLO
import pathlib

CHECKPOINT = pathlib.Path(r"C:\Users\arjun\runs\detect\runs\fish_species\train\weights\last.pt")
DATA = pathlib.Path(__file__).resolve().parent.parent / "datasets" / "fish_species" / "data.yaml"


def main():
    model = YOLO(str(CHECKPOINT))

    model.train(
        data=str(DATA),
        epochs=40,
        patience=15,
        batch=8,
        imgsz=480,
        device=0,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        freeze=5,
        workers=2,
        cache="ram",
        resume=True,
        augment=True,
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.3,
        flipud=0.3,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        project="runs/fish_species",
        name="train",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
