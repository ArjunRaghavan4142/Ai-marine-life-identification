"""Validation harness for the two target dive videos.

Instruction from user:
  - Video 1 (7.01.53 PM) should identify ONLY parrotfish
  - Video 2 (7.57.07 AM) should identify 1 sea turtle + 2~3 angelfish

Run after every training session to check the pipeline against these targets.
"""
import sys
import time
import json

import requests

API = "http://localhost:8000"

VIDEOS = [
    {
        "label": "VIDEO 1 (7.01.53 PM)",
        "path": r"C:\Users\arjun\OneDrive\Desktop\Test video\WhatsApp Video 2026-07-10 at 7.01.53 PM (1).mp4",
        "target": "ONLY parrotfish",
    },
    {
        "label": "VIDEO 2 (7.57.07 AM)",
        "path": r"C:\Users\arjun\OneDrive\Desktop\Test video\WhatsApp Video 2026-07-10 at 7.57.07 AM.mp4",
        "target": "1 sea turtle + 2~3 angelfish",
    },
]


def wait_for_server():
    for attempt in range(20):
        try:
            requests.get(f"{API}/", timeout=2)
            return True
        except Exception:
            print(f"  waiting for backend... ({attempt+1})")
            time.sleep(3)
    return False


def run_video(video):
    print(f"\n{'='*60}")
    print(f"{video['label']}  -- target: {video['target']}")
    print(f"{'='*60}")

    with open(video["path"], "rb") as f:
        resp = requests.post(
            f"{API}/upload",
            files={"file": (video["path"].split("\\")[-1], f, "video/mp4")},
            data={"region": "global"},
        )
    job_id = resp.json()["job_id"]
    print(f"  job_id: {job_id}")

    while True:
        status = requests.get(f"{API}/jobs/{job_id}").json()
        if status.get("error"):
            print(f"  ERROR: {status['error']}")
            return None
        if status["status"] == "done":
            break
        time.sleep(5)

    result = requests.get(f"{API}/jobs/{job_id}/results").json()
    print(f"  identification_method: {result.get('identification_method', 'n/a')}")
    print(f"  --- SPECIES DETECTED ({len(result['species_summary'])}) ---")
    for s in result["species_summary"]:
        print(f"    * {s['species']}: {s['unique_count']}  (conf {s['avg_confidence']}, {s['rarity']})")
    if not result["species_summary"]:
        print("    (none)")
    print(f"  --- REVIEW QUEUE ({len(result['review_queue'])}) ---")
    for r in result["review_queue"]:
        guess = r.get("species_guess", "?")
        gc = r.get("gemini_confidence", "?")
        print(f"    ? {guess}  (gemini_conf {gc}, ts {r.get('timestamp_sec','?')}s)")
    return result


def main():
    if not wait_for_server():
        print("Backend never came up.")
        sys.exit(1)
    results = {}
    for v in VIDEOS:
        results[v["label"]] = run_video(v)

    print(f"\n\n{'#'*60}")
    print("SUMMARY vs TARGETS")
    print(f"{'#'*60}")
    for v in VIDEOS:
        r = results[v["label"]]
        got = ", ".join(f"{s['species']}({s['unique_count']})" for s in r["species_summary"]) if r else "FAILED"
        print(f"\n{v['label']}")
        print(f"  TARGET: {v['target']}")
        print(f"  GOT:    {got or '(nothing)'}")


if __name__ == "__main__":
    main()
