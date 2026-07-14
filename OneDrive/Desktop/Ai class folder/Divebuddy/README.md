# DiveBuddy — AI Marine Life Identification

An app that turns dive footage into a biodiversity report. Divers and
researchers upload a video (or photo); DiveBuddy detects the marine animals,
removes duplicate sightings of the same individual, identifies each one, and
produces a species summary plus CSV/PDF reports.

## Pipeline

```
video ─▶ frame extraction ─▶ detection + tracking ─▶ aggregation ─▶ identification ─▶ report
        (OpenCV, 5 fps)      (YOLO ensemble +        (dedup, unique   (iNaturalist    (CSV/PDF)
                              BoT-SORT tracker)       counting)        Vision)
```

1. **Frame extraction** — OpenCV samples frames from the upload.
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

## Species identification — current approach & decision

**Detection answers *where* an animal is; a vision model answers *what* it is.**
The YOLO detectors' own class labels are unreliable at species level (they call
reef fish "grouper/shark/etc."), so naming is done by a separate vision model on
the crops, collapsed to a coarse **group** (Parrotfish, Angelfish, Sea Turtle,
Moorish Idol, …) with score-weighted multi-crop voting.

**Current engine: iNaturalist Vision** (`backend/app/pipeline/inat_id.py`).
- Trained on the world's largest citizen-science organism dataset — real
  underwater photos in messy conditions, so it generalizes well to blurry dive
  crops where local models (CLIP) and the detectors' own labels failed.
- Results scoped to reef-relevant taxa (fish/reptiles/inverts) to drop
  spurious mammal/bird guesses.
- **Auto-refreshing auth** (`inat_auth.py`): with OAuth credentials configured
  it mints and refreshes its own API token, so it never expires. Falls back to
  a static `INAT_TOKEN` otherwise. See "Setup" below.

Alternative engines are kept in-tree and can be swapped in `main.py`:
- **Gemini Vision** (`vision_id.py`) — very accurate, but the free tier is
  capped at ~20 requests/day/model.
- **Offline / YOLO-label** (`identify.regroup_yolo`) — fully offline and
  instant, but mislabels reef fish (models' limitation).
- **Local CLIP** (`clip_id.py`) — offline; too weak on these crops (kept for
  reference).

### Why iNaturalist now, and when to switch to FishNet

**Decision (for private long-term display & testing, app *not* published): use
iNaturalist auto-refresh.** It is already accurate on real footage, needs no
training, and is low-maintenance. iNaturalist's rate-limit / ToS concerns apply
to a *published* app with many users hammering their API — not to private,
low-volume demo and testing use.

**Planned upgrade path — FishNet, later, only if we publish:** if DiveBuddy is
ever published (or needs to run fully offline / independent of iNaturalist), the
right move is to **fine-tune our own model on the FishNet dataset *plus* labeled
crops from our own dive footage**, kept alongside turtle/megafauna detection.
Notes for when we do this:
- FishNet is **fish-only** — it has no sea turtles, so MegaFauna (or turtle
  training data) must be retained for the turtle.
- FishNet is curated reference photos; our footage is blurry dive video, so
  domain adaptation on our own labeled crops is needed for it to transfer.

This is deliberately deferred — not needed for current display/testing.

## Setup

```bash
# backend
cd backend
pip install -r requirements.txt        # fastapi, uvicorn, ultralytics, torch, opencv, reportlab, requests
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
