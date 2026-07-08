# DiveBuddy MVP

AI pipeline that turns an uploaded dive video into a biodiversity report:
frame extraction → marine species detection/tracking → duplicate-count
reduction → low-confidence review flagging → CSV/PDF report.

Detection runs on [marine-detect](https://github.com/Orange-OpenSource/marine-detect)'s
pretrained YOLOv8 checkpoints (18 reef species/taxa across two models), with
duplicate suppression via ByteTrack (bundled in `ultralytics`). See
`datasets/README.md` for the FishNet/LILA fine-tuning path once real dive
footage surfaces species these models miss.

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
