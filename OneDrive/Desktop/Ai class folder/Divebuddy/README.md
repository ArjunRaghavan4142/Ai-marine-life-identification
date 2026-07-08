# DiveBuddy MVP

AI pipeline that turns an uploaded dive video into a biodiversity report:
frame extraction → marine species detection/tracking → duplicate-count
reduction → low-confidence review flagging → CSV/PDF report.

Detection runs on [marine-detect](https://github.com/Orange-OpenSource/marine-detect)'s
pretrained YOLOv8 checkpoints (18 reef species/taxa across two models), with
duplicate suppression via TrackTrack (bundled in `ultralytics`, tuned in
`backend/app/pipeline/tracktrack_divebuddy.yaml`). See `datasets/README.md`
for the FishNet/LILA fine-tuning path once real dive footage surfaces
species these models miss.

## Performance: GPU vs CPU

Frames are sampled at 5 fps because anything sparser measurably misses real
sightings (verified against real dive footage — see git history on
`backend/app/main.py`), and TrackTrack needs continuous-enough motion between
frames to keep an animal's identity stable across a moving camera. That's a
lot of YOLO inference calls per video, so hardware matters a lot here:

- **CPU only**: ~5 minutes of processing per 20 seconds of footage. A real
  20-60 minute dive would take **hours**. Fine for short clips, not viable
  for full dive logs.
- **NVIDIA GPU** (`device="cuda"` in `backend/app/pipeline/detect.py`, picked
  automatically when available): the same 20s clip measured at **~1 minute**
  — roughly 5x faster, verified with identical detection output on both. A
  full dive is still minutes rather than seconds, but no longer hours.

There's still no smarter adaptive sampling (e.g. skipping visually-static
stretches) — that's the next lever if GPU still isn't fast enough for very
long videos.

## Run it

```
# one-time setup
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.venv\Scripts\python.exe scripts\download_models.py

# if you have an NVIDIA GPU, swap in the CUDA build of torch (~5x faster --
# see "Performance" above). Check your GPU/driver first with `nvidia-smi`;
# cu128 works with any reasonably recent driver.
.venv\Scripts\python.exe -m pip uninstall -y torch torchvision
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# start the API
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --app-dir backend

# open frontend/index.html in a browser
```

## Layout

The backend (`backend/`) is a standalone, UI-agnostic script — it doesn't know
or care what frontend talks to it. `frontend/index.html` is just the current
UI wired to it. Swapping in a new UI design means pointing its JS at the API
contract below; it does not touch anything in `backend/`. This is exactly how
`frontend/index.html` itself got built — a separately-designed HTML mockup
had its fetch calls pointed at these same endpoints.

- `backend/app/main.py` — FastAPI app: upload endpoint, job status/results, report downloads
- `backend/app/pipeline/` — frame extraction, detection+tracking, aggregation, report generation
- `backend/app/rarity_map.json` — placeholder common/uncommon/rare tag per species (not from real survey data yet)
- `frontend/index.html` — current UI, wired to the API contract below
- `scripts/download_models.py` — fetches the marine-detect model weights
- `datasets/README.md` — FishNet/LILA notes for future fine-tuning

## API contract

Base URL: `http://localhost:8000`. CORS is wide open (`allow_origins=["*"]`),
so any UI — a local HTML file, a dev server on a different port, whatever —
can call these directly with no proxy/config needed.

**`POST /upload`** — multipart form, field name `file`, must end in `.mp4`
(case-insensitive). Saves the video and kicks off processing in the
background; returns immediately.
```json
{"job_id": "1cec1ad8648e", "status": "processing", "filename": "dive.mp4"}
```

**`GET /jobs/{job_id}`** — poll this until `status` is `"done"` or `"error"`.
```json
{"job_id": "1cec1ad8648e", "status": "done", "error": null}
```

**`GET /jobs/{job_id}/results`** — 409 until the job is done.
```json
{
  "species_summary": [
    {"species": "fish", "unique_count": 3, "avg_confidence": 0.75, "rarity": "common"}
  ],
  "review_queue": [
    {"track_id": "FishInv_19", "species_guess": "fish", "max_confidence": 0.38,
     "timestamp_sec": 12.0, "crop_path": "1cec.../crops/FishInv_19_60.jpg"}
  ]
}
```
`crop_path` is servable directly at `GET /outputs/{crop_path}` (static mount).

**`GET /jobs/{job_id}/report.csv`** / **`GET /jobs/{job_id}/report.pdf`** —
file downloads, 409 until done.

**`GET /jobs`** — full job history, most recent first. Powers Home /
Previous Dives.
```json
{"jobs": [{"job_id": "...", "filename": "...", "status": "done",
  "uploaded_at": "2026-07-08T16:35:44+00:00", "species_count": 2,
  "total_sightings": 4, "flagged_count": 1}]}
```

**`GET /species`** — the 18-class catalog, tallied against every completed
job. Powers Fish Dex.
```json
{"species": [{"species": "fish", "seen": true, "total_sightings": 3}]}
```

Job history is in-memory only — it resets if the backend restarts. There's
no auth, no rate limiting, and no per-user scoping (every job is global);
none of that was needed for a single-user MVP, but a next UI that assumes
multiple accounts would need those built first.
