"""Marine organism detection + tracking.

Runs each extracted frame through both marine-detect YOLOv8 checkpoints
(FishInv: fish/invertebrates, MegaFauna: sharks/rays/turtles) using
ultralytics' built-in ByteTrack tracker instead of plain detection, so that
the same animal seen across consecutive frames keeps one track ID. That's
what makes duplicate-count reduction possible downstream.
"""
import pathlib

import cv2
from ultralytics import YOLO

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"

# `accept_conf` is marine-detect's own recommended threshold: detections at or
# above this are auto-counted. We run the tracker at a lower `review_conf` floor
# so that borderline detections (accept_conf > conf >= review_conf) aren't
# silently dropped -- they surface in the low-confidence review queue instead
# of just vanishing.
MODEL_CONFIGS = [
    {"name": "FishInv", "weights": "FishInv.pt", "accept_conf": 0.523, "review_conf": 0.30},
    {"name": "MegaFauna", "weights": "MegaFauna.pt", "accept_conf": 0.546, "review_conf": 0.30},
]


def run_detection(frames: list[dict], crops_dir: pathlib.Path) -> list[dict]:
    """Runs detection+tracking over `frames` (as produced by frames.extract_frames).

    Returns a flat list of detection dicts:
      species, confidence, track_id, frame_index, timestamp_sec, crop_path
    """
    crops_dir.mkdir(parents=True, exist_ok=True)
    detections: list[dict] = []

    for config in MODEL_CONFIGS:
        weights_path = MODELS_DIR / config["weights"]
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Missing model weights: {weights_path}. Run scripts/download_models.py first."
            )
        model = YOLO(str(weights_path))

        for frame in frames:
            result = model.track(
                source=str(frame["path"]),
                conf=config["review_conf"],
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )[0]

            if result.boxes is None or result.boxes.id is None:
                continue

            image = None
            for box in result.boxes:
                class_id = int(box.cls[0])
                species = result.names[class_id]
                confidence = float(box.conf[0])
                track_id = int(box.id[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])

                crop_path = crops_dir / f"{config['name']}_{track_id}_{frame['frame_index']}.jpg"
                if image is None:
                    image = cv2.imread(str(frame["path"]))
                crop = image[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
                if crop.size > 0:
                    cv2.imwrite(str(crop_path), crop)

                detections.append({
                    "species": species,
                    "confidence": confidence,
                    "accept_conf": config["accept_conf"],
                    "track_id": f"{config['name']}_{track_id}",
                    "frame_index": frame["frame_index"],
                    "timestamp_sec": frame["timestamp_sec"],
                    "crop_path": str(crop_path) if crop.size > 0 else None,
                })

    return detections
