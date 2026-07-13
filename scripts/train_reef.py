"""Fine-tune the ReefFamilies detector on the reef_families dataset.

One invocation == one "training session". Usage:
    python scripts/train_reef.py <session_tag> <epochs>

After training, the best weights are copied into backend/app/models/ReefFamilies.pt
(the original is backed up once to ReefFamilies.orig.pt) so the pipeline picks
them up on the next backend restart.

NOTE: ReefFamilies.pt was originally trained on this same 476-image dataset, so
continued training mostly refines the fit rather than adding new capability, and
there is NO sea-turtle class here (turtle comes from MegaFauna.pt). This script
exists to satisfy the "train the model as much as you can" instruction and to
measure whether more training actually moves the two target videos.
"""
import shutil
import sys
import pathlib

from ultralytics import YOLO

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "reef_families" / "data_abs.yaml"
MODELS = ROOT / "backend" / "app" / "models"
BASE_WEIGHTS = MODELS / "ReefFamilies.pt"
ORIG_BACKUP = MODELS / "ReefFamilies.orig.pt"
RUNS = ROOT / "training_runs"


def main():
    session_tag = sys.argv[1] if len(sys.argv) > 1 else "s1"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    # Back up the original weights exactly once.
    if not ORIG_BACKUP.exists():
        shutil.copy2(BASE_WEIGHTS, ORIG_BACKUP)
        print(f"Backed up original weights -> {ORIG_BACKUP.name}")

    # Always fine-tune from the pristine original so sessions don't compound
    # overfitting on top of each other.
    print(f"=== Training session '{session_tag}' : {epochs} epochs ===")
    model = YOLO(str(ORIG_BACKUP))

    results = model.train(
        data=str(DATA),
        epochs=epochs,
        imgsz=640,
        batch=4,  # RTX 3050 Laptop has only 4GB VRAM
        device=0,
        project=str(RUNS),
        name=session_tag,
        exist_ok=True,
        patience=15,
        # Augmentation to squeeze generalization out of a small dataset.
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=5.0, translate=0.1, scale=0.4, fliplr=0.5,
        mosaic=1.0, close_mosaic=10,
        verbose=True,
    )

    best = RUNS / session_tag / "weights" / "best.pt"
    if best.exists():
        # Validate the new weights before promoting.
        val_model = YOLO(str(best))
        metrics = val_model.val(data=str(DATA), device=0, verbose=False)
        print(f"\nSession '{session_tag}' validation mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")
        shutil.copy2(best, BASE_WEIGHTS)
        print(f"Promoted best.pt -> {BASE_WEIGHTS}")
    else:
        print(f"WARNING: no best.pt at {best}")


if __name__ == "__main__":
    main()
