"""Marine organism detection + tracking.

Runs each extracted frame through the YOLO model pipeline using
ultralytics' built-in BoT-SORT tracker so that the same animal seen
across consecutive frames keeps one track ID — duplicate-count
reduction downstream depends on this.

BoT-SORT with gmc_method=sparseOptFlow compensates for camera motion,
which is critical for handheld dive footage where the camera pans
constantly.
"""
import pathlib
import re

import cv2
import torch
from ultralytics import YOLO

MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"
TRACKER_CONFIG = pathlib.Path(__file__).resolve().parent / "botsort_divebuddy.yaml"

# CPU inference of two YOLO models + TrackTrack's optical-flow motion
# compensation is what made a 20s clip take ~5 minutes (documented in
# README). GPU inference of the same models is roughly an order of
# magnitude faster. Falls back to CPU automatically on machines without a
# CUDA-capable GPU -- same detection results either way, just faster.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# `accept_conf` is marine-detect's own recommended threshold: detections at or
# above this are auto-counted. We run the tracker at a lower `review_conf` floor
# so that borderline detections (accept_conf > conf >= review_conf) aren't
# silently dropped -- they surface in the low-confidence review queue instead
# of just vanishing.
MODEL_CONFIGS = [
    {"name": "FishSpecies", "weights": "FishSpecies.pt", "accept_conf": 0.75, "review_conf": 0.25, "specificity": 4},
    {"name": "FishInv", "weights": "FishInv.pt", "accept_conf": 0.80, "review_conf": 0.30, "specificity": 2},
    {"name": "Seychelles", "weights": "Seychelles.pt", "accept_conf": 0.80, "review_conf": 0.30, "specificity": 3},
    {"name": "ReefFamilies", "weights": "ReefFamilies.pt", "accept_conf": 0.80, "review_conf": 0.30, "specificity": 1},
    {"name": "MarineLife", "weights": "MarineLife.pt", "accept_conf": 0.75, "review_conf": 0.30, "specificity": 1,
     "class_filter": {"eel", "starfish", "crab", "jellyfish"}},
]

JUNK_CLASS_PATTERN = re.compile(r'^[A-Z0-9]{4,}-[A-Z0-9]')

GENERIC_CLASSES = {"fish", "shark", "ray", "turtle", "urchin"}

DIVER_OVERLAP_THRESH = 0.5


def _containment(inner: list, outer: list) -> float:
    """Fraction of `inner` bbox area that falls inside `outer`."""
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter / inner_area if inner_area > 0 else 0.0


def _detect_divers(frames: list[dict]) -> dict[int, list[list[int]]]:
    """Pre-pass: find diver bounding boxes using YOLOv8n person detection."""
    weights = MODELS_DIR / "yolov8n.pt"
    if not weights.exists():
        return {}
    model = YOLO(str(weights))
    diver_bboxes: dict[int, list[list[int]]] = {}

    for frame in frames:
        result = model.predict(
            source=str(frame["path"]),
            conf=0.5,
            classes=[0],
            imgsz=640,
            verbose=False,
            device=DEVICE,
        )[0]

        if result.boxes is not None and len(result.boxes) > 0:
            bboxes = []
            for box in result.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                bboxes.append([x1, y1, x2, y2])
            diver_bboxes[frame["frame_index"]] = bboxes

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return diver_bboxes


def _iou(a: list, b: list) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


PHOTO_MIN_CONF = 0.35


def run_photo_detection(image_path: pathlib.Path, crops_dir: pathlib.Path) -> list[dict]:
    """Run all models on a single image. No tracking — just detect and dedup."""
    crops_dir.mkdir(parents=True, exist_ok=True)

    diver_bboxes = []
    diver_weights = MODELS_DIR / "yolov8n.pt"
    if diver_weights.exists():
        model = YOLO(str(diver_weights))
        result = model.predict(
            source=str(image_path), conf=0.5, classes=[0],
            imgsz=640, verbose=False, device=DEVICE,
        )[0]
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                diver_bboxes.append([int(v) for v in box.xyxy[0]])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw: list[dict] = []
    for config in MODEL_CONFIGS:
        weights_path = MODELS_DIR / config["weights"]
        if not weights_path.exists():
            continue
        model = YOLO(str(weights_path))
        class_filter = config.get("class_filter")

        result = model.predict(
            source=str(image_path), conf=config["review_conf"],
            imgsz=640, verbose=False, device=DEVICE,
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        image = None
        for box in result.boxes:
            class_id = int(box.cls[0])
            species = result.names[class_id]
            if class_filter and species not in class_filter:
                continue
            if JUNK_CLASS_PATTERN.match(species):
                continue
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])

            if any(_containment([x1, y1, x2, y2], db) >= DIVER_OVERLAP_THRESH for db in diver_bboxes):
                continue

            det_id = f"{config['name']}_{len(raw)}"
            crop_path = crops_dir / f"{det_id}.jpg"
            if image is None:
                image = cv2.imread(str(image_path))
            crop = image[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
            if crop.size > 0:
                cv2.imwrite(str(crop_path), crop)

            specificity = config["specificity"]
            if species.lower() in GENERIC_CLASSES:
                specificity = 0

            raw.append({
                "species": species,
                "confidence": confidence,
                "specificity": specificity,
                "model": config["name"],
                "bbox": [x1, y1, x2, y2],
                "crop_path": str(crop_path) if crop.size > 0 else None,
            })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Cross-model dedup: same fish detected by multiple models
    keep = set(range(len(raw)))
    for i in range(len(raw)):
        for j in range(i + 1, len(raw)):
            if i not in keep or j not in keep:
                continue
            if _iou(raw[i]["bbox"], raw[j]["bbox"]) >= 0.45:
                si, sj = raw[i]["specificity"], raw[j]["specificity"]
                if si != sj:
                    loser = j if si > sj else i
                else:
                    loser = j if raw[i]["confidence"] >= raw[j]["confidence"] else i
                keep.discard(loser)

    return [raw[i] for i in sorted(keep) if raw[i]["confidence"] >= PHOTO_MIN_CONF]


def run_detection(frames: list[dict], crops_dir: pathlib.Path) -> list[dict]:
    """Runs detection+tracking over `frames` (as produced by frames.extract_frames).

    Returns a flat list of detection dicts:
      species, confidence, track_id, frame_index, timestamp_sec, crop_path
    """
    crops_dir.mkdir(parents=True, exist_ok=True)
    detections: list[dict] = []

    diver_bboxes = _detect_divers(frames)

    for config in MODEL_CONFIGS:
        weights_path = MODELS_DIR / config["weights"]
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Missing model weights: {weights_path}. Run scripts/download_models.py first."
            )
        model = YOLO(str(weights_path))
        class_filter = config.get("class_filter")

        for frame in frames:
            result = model.track(
                source=str(frame["path"]),
                conf=config["review_conf"],
                imgsz=640,
                persist=True,
                tracker=str(TRACKER_CONFIG),
                verbose=False,
                device=DEVICE,
            )[0]

            if result.boxes is None or result.boxes.id is None:
                continue

            image = None
            for box in result.boxes:
                class_id = int(box.cls[0])
                species = result.names[class_id]
                if class_filter and species not in class_filter:
                    continue
                if JUNK_CLASS_PATTERN.match(species):
                    continue
                confidence = float(box.conf[0])
                track_id = int(box.id[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])

                frame_divers = diver_bboxes.get(frame["frame_index"], [])
                if any(_containment([x1, y1, x2, y2], db) >= DIVER_OVERLAP_THRESH for db in frame_divers):
                    continue

                crop_path = crops_dir / f"{config['name']}_{track_id}_{frame['frame_index']}.jpg"
                if image is None:
                    image = cv2.imread(str(frame["path"]))
                crop = image[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
                if crop.size > 0:
                    cv2.imwrite(str(crop_path), crop)

                specificity = config["specificity"]
                if species.lower() in GENERIC_CLASSES:
                    specificity = 0

                detections.append({
                    "species": species,
                    "confidence": confidence,
                    "accept_conf": config["accept_conf"],
                    "specificity": specificity,
                    "track_id": f"{config['name']}_{track_id}",
                    "frame_index": frame["frame_index"],
                    "timestamp_sec": frame["timestamp_sec"],
                    "crop_path": str(crop_path) if crop.size > 0 else None,
                    "bbox": [x1, y1, x2, y2],
                })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return detections
