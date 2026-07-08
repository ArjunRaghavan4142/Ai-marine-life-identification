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

## Known limitation: processing time doesn't scale to full-length dives

Frames are sampled at 5 fps because anything sparser measurably misses real
sightings (verified against real dive footage — see git history on
`backend/app/main.py`), and TrackTrack needs continuous-enough motion between
frames to keep an animal's identity stable across a moving camera. On CPU,
that's roughly 5 minutes of processing per 20 seconds of footage with both
detection models running. A real 20-60 minute dive video would take **hours**,
not minutes. This is fine for short clips/highlights; it is not yet viable
for full dive logs. Fixing this for real means either GPU inference or a
smarter adaptive sampling strategy — both are follow-up work, not done here.

## Run it

```
# one-time setup
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.venv\Scripts\python.exe scripts\download_models.py

# start the API
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --app-dir backend

# open frontend/index.html in a browser
```

## Layout

- `backend/app/main.py` — FastAPI app: upload endpoint, job status/results, report downloads
- `backend/app/pipeline/` — frame extraction, detection+tracking, aggregation, report generation
- `backend/app/rarity_map.json` — placeholder common/uncommon/rare tag per species (not from real survey data yet)
- `frontend/index.html` — upload UI + results table + review queue + report downloads
- `scripts/download_models.py` — fetches the marine-detect model weights
- `datasets/README.md` — FishNet/LILA notes for future fine-tuning
