# DiveBuddy — AI Marine Life Identification

An app that turns dive footage into a biodiversity report. Divers and
researchers upload a video (or photo); DiveBuddy detects the marine animals,
removes duplicate sightings of the same individual, identifies each one, and
produces a species summary plus CSV/PDF reports.

## Pipeline

```
video ─▶ frame extraction  ─▶ detection + tracking ─▶ aggregation ─▶ identification ─▶ report
        (OpenCV 5 fps +       (YOLO ensemble +        (dedup, unique   (BioCLIP local,  (CSV/PDF)
         underwater enhance)   BoT-SORT tracker)       counting)        region-aware)
```

1. **Frame extraction** — OpenCV samples frames, then an underwater colour/
   contrast restoration pass (`enhance.py`) corrects the blue cast before
   detection, lifting both detection and ID confidence on murky footage.
2. **Detection + tracking** — a YOLO model ensemble finds animals; BoT-SORT
   keeps one track ID per individual across frames (so one fish isn't counted
   once per frame).
3. **Aggregation** — counts unique tracks, merges fragmented/overlapping
   tracks, and routes weak detections to a review queue.
4. **Identification** — crops of each animal are named (see below).
5. **Report** — per-species counts, confidence, rarity, and a low-confidence
   review list, exported as CSV and PDF.

### Detection models
A trimmed ensemble of **specialist + core** YOLO detectors (see
`backend/app/pipeline/detect.py`):
- Specialists: **Lionfish**, **Corals**, **MegaFauna** (turtle/shark/ray)
- Core fish: **FishSpecies**, **Seychelles**, **FishInv**

Cross-model dedup keeps the most specific detection when two models overlap
(the `specificity` field). MegaFauna is given top priority because it is the
only turtle/shark/ray detector and must not be overwritten by a fish model's
overlapping guess.

Broad/generalist models (ReefFamilies, MultiClass, MarineLife) were disabled —
they were redundant with the species models and mostly produced mislabels.

## Species identification — current approach

**Detection answers *where* an animal is; a vision model answers *what* it is.**
The YOLO detectors' own class labels are unreliable at species level (they call
reef fish "grouper/shark/etc."), so naming is done by a separate vision model on
the crops, collapsed to a coarse **group** (Parrotfish, Angelfish, Sea Turtle,
Moorish Idol, …) with score-weighted multi-crop voting.

Naming runs as a graceful **engine chain** (`main.identify_with_fallback`) so no
single dead dependency can ever blank a report:

1. **BioCLIP — local, primary** (`bioclip_id.py`). A tree-of-life CLIP model that
   runs on-device, so naming needs no API token, quota, or network. It zero-shot
   matches each crop against a curated reef label set, **restricted to the dive
   region** so it can't pick a species that doesn't live there. Model weights
   (~1GB) auto-download from HuggingFace on first run, then run offline.
2. **iNaturalist Vision — fallback** (`inat_id.py`, auto-refresh in
   `inat_auth.py`). A strong second opinion when a token is configured.
3. **Offline YOLO grouping — last resort** (`identify.regroup_yolo`). Rebuilds the
   summary from the detectors' own labels so the report is never all-"Unknown".

**Precision gate** (`identify.py`): a detection is only *asserted* as a species
when the ID is high/medium confidence; a confident detection with an uncertain
name is routed to the review queue instead of being reported (and possibly wrong).

### Fine-tuning on your own footage (highest accuracy lever)
`scripts/build_bioclip_prototypes.py` builds a prototype classifier from labelled
crops — one folder per species under `backend/datasets/reef_labeled/`. It needs
only a few hundred crops, no GPU training, and the API auto-loads the result on
restart. This anchors naming to your own water, lighting, and camera. A full
fine-tune on the **FishNet** dataset plus our own labelled crops is the later
step if DiveBuddy is ever published (note: FishNet is fish-only, so MegaFauna
must be kept for sea turtles).

Alternative engines kept in-tree and swappable in `main.py`: **Gemini Vision**
(`vision_id.py`, ~20 requests/day free tier) and an older **local CLIP**
(`clip_id.py`).

## Setup

```bash
# backend
cd backend
pip install -r requirements.txt        # fastapi, uvicorn, ultralytics, torch, opencv, reportlab, requests, open_clip_torch
uvicorn app.main:app --host 0.0.0.0 --port 8000

# frontend (separate shell)
python -m http.server 5500 --directory frontend
```

Model weights (`backend/app/models/*.pt`) are gitignored (large binaries).

### iNaturalist auto-refresh (recommended — token never expires)
1. Register an OAuth app: https://www.inaturalist.org/oauth/applications/new
   (Redirect URI `urn:ietf:wg:oauth:2.0:oob`) → get Client ID + Secret.
2. `cp backend/.inat_env.example backend/.inat_env` and fill in Client ID,
   Client Secret, and your iNat username + password. **`.inat_env` is
   gitignored — never commit it.**
3. Restart the backend. It now self-refreshes its token.

Without `.inat_env`, set a static `INAT_TOKEN` env var (from
inaturalist.org/users/api_token) — but that expires ~24h.

## Repo notes
- Secrets (API tokens/keys, `.inat_env`) must never be committed.
- `datasets/`, model weights, uploads, and outputs are gitignored.
