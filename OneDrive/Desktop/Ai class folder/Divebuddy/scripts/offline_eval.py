"""Offline evaluation harness for the two target videos.

Runs detection ONCE per video and caches it, then runs aggregation + Gemini
identification (with a per-crop Gemini cache) so the identification/aggregation
logic can be tuned and re-evaluated in seconds without re-detecting or
re-paying Gemini for crops already seen.

Usage:
    python scripts/offline_eval.py            # both videos, use caches
    python scripts/offline_eval.py --fresh    # re-run detection (busts det cache)
    python scripts/offline_eval.py v1         # only video 1
"""
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.pipeline.frames import extract_frames          # noqa: E402
from app.pipeline.detect import run_detection, MODEL_CONFIGS, GENERIC_CLASSES  # noqa: E402
from app.pipeline.aggregate import aggregate_tracks      # noqa: E402
from app.pipeline.identify import reidentify, regroup_yolo  # noqa: E402
from app.pipeline.vision_id import identify_species      # noqa: E402

# Map model name -> its config, so cached detections can be re-scored against
# the CURRENT specificity hierarchy without re-running detection on the GPU.
_CFG_BY_NAME = {c["name"]: c for c in MODEL_CONFIGS}


def resync_specificity(dets):
    """Recompute each cached detection's specificity from the live MODEL_CONFIGS.

    Lets a change to the specificity hierarchy be validated against cached
    detections instantly, instead of re-detecting every video.
    """
    for d in dets:
        cfg = _CFG_BY_NAME.get(d.get("model"))
        if not cfg:
            continue
        spec = cfg["specificity"]
        if d.get("species", "").lower() in GENERIC_CLASSES and not cfg.get("skip_generic_downgrade"):
            spec = 0
        d["specificity"] = spec
    return dets

API_KEY = os.environ.get("GEMINI_API_KEY", "")  # set in env for --gemini mode; never hard-code
RARITY_MAP = json.loads((BACKEND / "app" / "rarity_map.json").read_text())

CACHE = ROOT / "eval_cache"
CACHE.mkdir(exist_ok=True)
GEMINI_CACHE = CACHE / "gemini_cache.json"

VIDEOS = {
    "v1": {
        "path": r"C:\Users\arjun\OneDrive\Desktop\Test video\WhatsApp Video 2026-07-10 at 7.01.53 PM (1).mp4",
        "target": "ONLY parrotfish",
    },
    "v2": {
        "path": r"C:\Users\arjun\OneDrive\Desktop\Test video\WhatsApp Video 2026-07-10 at 7.57.07 AM.mp4",
        "target": "1 sea turtle + 2~3 angelfish",
    },
}


def cached_identify(crop_paths, api_key):
    cache = json.loads(GEMINI_CACHE.read_text()) if GEMINI_CACHE.exists() else {}
    # Drop any entries that came from a transient error so they get retried.
    cache = {k: v for k, v in cache.items() if not v.get("_error")}
    todo = [c for c in crop_paths if c and c not in cache and pathlib.Path(c).exists()]
    if todo:
        print(f"  Gemini: {len(todo)} new crops (of {len(crop_paths)} total)")
        new_ids = identify_species(todo, api_key)
        for c, i in zip(todo, new_ids):
            if not i.get("_error"):  # never cache a failed lookup
                cache[c] = i
        GEMINI_CACHE.write_text(json.dumps(cache, indent=2))
    fallback = {"common_name": "Unknown", "scientific_name": "", "group": "Unknown", "confidence": "low"}
    return [cache.get(c, fallback) for c in crop_paths]


def get_detections(key, video_path, fresh=False):
    det_cache = CACHE / f"{key}_detections.json"
    if det_cache.exists() and not fresh:
        print(f"  [cache] loading detections for {key} (re-scoring specificity)")
        return resync_specificity(json.loads(det_cache.read_text()))
    job_dir = CACHE / key
    print(f"  [detect] extracting frames for {key} ...")
    frames = extract_frames(pathlib.Path(video_path), job_dir / "frames", fps_sample=5.0)["frames"]
    print(f"  [detect] running {len(frames)} frames through detection models ...")
    dets = run_detection(frames, job_dir / "crops")  # crop_path stays absolute
    det_cache.write_text(json.dumps(dets, indent=2))
    print(f"  [detect] {len(dets)} raw detections cached")
    return dets


def evaluate(key, fresh=False, offline=False, clip=False):
    v = VIDEOS[key]
    print(f"\n{'='*64}\n{key.upper()}  target: {v['target']}\n{'='*64}")
    dets = get_detections(key, v["path"], fresh=fresh)
    result = aggregate_tracks(dets, RARITY_MAP)
    if clip:
        # Local CLIP identifier on the crops, reusing multi-crop voting.
        from app.pipeline.clip_id import classify_crops
        result = reidentify(result, pathlib.Path("."), None, RARITY_MAP, identify=classify_crops)
    elif offline:
        # No-Gemini path: names come straight from the YOLO models (group-level).
        result = regroup_yolo(result, RARITY_MAP)
    else:
        # crop paths in dets are absolute; reidentify does output_dir/crop, which
        # pathlib resolves to the absolute crop path regardless of output_dir.
        result = reidentify(result, pathlib.Path("."), API_KEY, RARITY_MAP, identify=cached_identify)

    print(f"\n  --- SPECIES SUMMARY ---")
    for s in result["species_summary"]:
        print(f"    * {s['species']}: {s['unique_count']}  (conf {s['avg_confidence']})")
    if not result["species_summary"]:
        print("    (none)")
    print(f"  --- REVIEW QUEUE ({len(result['review_queue'])}) ---")
    for r in result["review_queue"][:12]:
        print(f"    ? {r.get('species_guess')} [{r.get('gemini_group','?')}] "
              f"conf={r.get('gemini_confidence','?')} frames={r.get('frame_span','?')}")
    return result


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fresh = "--fresh" in sys.argv
    offline = "--offline" in sys.argv
    clip = "--clip" in sys.argv
    keys = args if args else ["v1", "v2"]
    results = {}
    for k in keys:
        results[k] = evaluate(k, fresh=fresh, offline=offline, clip=clip)

    print(f"\n\n{'#'*64}\nSUMMARY vs TARGETS\n{'#'*64}")
    for k in keys:
        got = ", ".join(f"{s['species']}({s['unique_count']})" for s in results[k]["species_summary"])
        print(f"\n{k.upper()}  target: {VIDEOS[k]['target']}")
        print(f"  GOT: {got or '(nothing)'}")

    (CACHE / "last_results.json").write_text(json.dumps(
        {k: results[k]["species_summary"] for k in keys}, indent=2))


if __name__ == "__main__":
    main()
