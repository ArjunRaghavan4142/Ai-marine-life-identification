"""Fine-tune FishInv.pt on the Seychelles reef fish dataset.

Splits train data 80/20 into train/val, updates data.yaml with absolute
paths, and runs YOLOv8 fine-tuning. Tuned for RTX 3050 (4GB VRAM).
"""
import pathlib
import random
import shutil
import yaml

from ultralytics import YOLO

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets" / "roboflow" / "seychelles"
WEIGHTS = ROOT / "backend" / "app" / "models" / "FishInv.pt"
OUTPUT_DIR = ROOT / "runs"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

SEED = 42
VAL_RATIO = 0.2


def split_train_val():
    """Move 20% of train images+labels into a val split."""
    if VAL_DIR.exists() and any(VAL_DIR.iterdir()):
        print(f"Val split already exists at {VAL_DIR}, skipping split.")
        return

    src_images = TRAIN_DIR / "images"
    src_labels = TRAIN_DIR / "labels"

    val_images = VAL_DIR / "images"
    val_labels = VAL_DIR / "labels"
    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)

    all_images = sorted(src_images.glob("*.*"))
    random.seed(SEED)
    random.shuffle(all_images)

    n_val = int(len(all_images) * VAL_RATIO)
    val_picks = all_images[:n_val]

    for img_path in val_picks:
        label_path = src_labels / (img_path.stem + ".txt")
        shutil.move(str(img_path), str(val_images / img_path.name))
        if label_path.exists():
            shutil.move(str(label_path), str(val_labels / label_path.name))

    print(f"Split: {len(all_images) - n_val} train, {n_val} val")


def update_data_yaml():
    """Rewrite data.yaml with absolute paths so YOLO finds the images."""
    yaml_path = DATASET_DIR / "data.yaml"
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    data["train"] = str(TRAIN_DIR / "images")
    data["val"] = str(VAL_DIR / "images")
    data.pop("test", None)

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    print(f"Updated {yaml_path}")
    print(f"  train: {data['train']}")
    print(f"  val:   {data['val']}")
    print(f"  nc:    {data['nc']}")


def train():
    model = YOLO(str(WEIGHTS))

    model.train(
        data=str(DATASET_DIR / "data.yaml"),
        epochs=80,
        patience=15,
        batch=2,
        imgsz=480,
        device=0,
        project=str(OUTPUT_DIR),
        name="seychelles_finetune2",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.0005,
        cos_lr=True,
        freeze=5,
        workers=4,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        augment=True,
        mosaic=0.5,
        mixup=0.1,
        degrees=10.0,
        flipud=0.3,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
    )


if __name__ == "__main__":
    print("=== Step 1: Split train/val ===")
    split_train_val()

    print("\n=== Step 2: Update data.yaml ===")
    update_data_yaml()

    print("\n=== Step 3: Fine-tune FishInv.pt ===")
    train()
