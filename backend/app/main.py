import datetime
import json
import os
import pathlib
import shutil
import traceback
import uuid

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.pipeline.aggregate import aggregate_tracks, common_name
from app.pipeline.detect import run_detection, run_photo_detection
from app.pipeline.frames import extract_frames
from app.pipeline.report import generate_csv, generate_pdf

app = FastAPI(title="DiveBuddy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
RARITY_MAP_PATH = pathlib.Path(__file__).resolve().parent / "rarity_map.json"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

RARITY_MAP = json.loads(RARITY_MAP_PATH.read_text())

# Serves crop thumbnails for the low-confidence review queue. Job output dirs
# are named with a random uuid4 hex, so there's nothing guessable/sensitive
# being exposed by mounting the whole outputs tree read-only.
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# In-memory job store -- fine for a single-process MVP, would move to a real
# DB/queue (e.g. Redis + Celery) before this needs to survive a restart or
# scale past one worker.
JOBS: dict[str, dict] = {}


@app.get("/")
def root():
    return {"message": "DiveBuddy API is running"}


REGIONS = {
    "indo-pacific": "Indo-Pacific (best coverage: Seychelles species-level + reef families)",
    "caribbean": "Caribbean (family-level coverage, limited species ID)",
    "red-sea": "Red Sea (partial species overlap with Indo-Pacific models)",
    "mediterranean": "Mediterranean (family-level coverage only)",
    "temperate": "Temperate / Cold Water (limited coverage)",
    "other": "Other / Unknown region",
}


@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    region: str = Form("indo-pacific"),
):
    if not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files are accepted")

    job_id = uuid.uuid4().hex[:12]
    video_path = UPLOAD_DIR / f"{job_id}.mp4"

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    JOBS[job_id] = {
        "status": "processing",
        "video_name": file.filename,
        "uploaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "region": region,
        "error": None,
        "result": None,
    }
    background_tasks.add_task(process_video, job_id, video_path)

    return {"job_id": job_id, "status": "processing", "filename": file.filename}


@app.get("/regions")
def list_regions():
    return {"regions": [{"id": k, "description": v} for k, v in REGIONS.items()]}


def process_video(job_id: str, video_path: pathlib.Path) -> None:
    job_dir = OUTPUT_DIR / job_id
    try:
        # 5 fps balances two failure modes found while testing on real dive
        # footage: too sparse (1-2 fps) and TrackTrack simply stops finding new
        # animals in the back half of a clip as the camera pans to new reef --
        # verified on a 20s test clip where 2 fps found nothing new after 7s
        # despite the clip running to 20s, while 5 fps kept finding new fish
        # through the very end. Denser sampling is ~5x the inference cost, but
        # under-counting real sightings is worse than mild over-counting for a
        # biodiversity survey tool. On CPU this does not scale to full-length
        # (20-60 min) dive videos -- expect hours, not minutes -- see README.
        extraction = extract_frames(video_path, job_dir / "frames", fps_sample=5.0)
        frames = extraction["frames"]
        detections = run_detection(frames, job_dir / "crops")

        # Store crop paths relative to OUTPUT_DIR so they can be served directly
        # through the /outputs static mount instead of leaking absolute filesystem paths.
        for detection in detections:
            if detection["crop_path"] is not None:
                detection["crop_path"] = pathlib.Path(detection["crop_path"]).relative_to(OUTPUT_DIR).as_posix()

        result = aggregate_tracks(detections, RARITY_MAP)

        (job_dir / "detections.json").write_text(json.dumps(detections, indent=2))
        generate_csv(result, job_dir / "report.csv")
        generate_pdf(result, job_dir / "report.pdf", JOBS[job_id]["video_name"], job_id)

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = result
    except Exception as exc:  # noqa: BLE001 - surface any pipeline failure to the API
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(exc)
        traceback.print_exc()


PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@app.post("/identify")
async def identify_photo(file: UploadFile = File(...)):
    ext = pathlib.Path(file.filename).suffix.lower()
    if ext not in PHOTO_EXTS:
        raise HTTPException(status_code=400, detail=f"Accepted formats: {', '.join(PHOTO_EXTS)}")

    photo_id = uuid.uuid4().hex[:12]
    photo_dir = OUTPUT_DIR / f"photo_{photo_id}"
    photo_dir.mkdir(parents=True, exist_ok=True)

    image_path = photo_dir / f"input{ext}"
    with open(image_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        detections = run_photo_detection(image_path, photo_dir / "crops")

        for d in detections:
            d["species_common"] = common_name(d["species"])
            d["rarity"] = RARITY_MAP.get(d["species"], "unknown")
            if d["crop_path"] is not None:
                d["crop_path"] = pathlib.Path(d["crop_path"]).relative_to(OUTPUT_DIR).as_posix()

        input_rel = image_path.relative_to(OUTPUT_DIR).as_posix()
        return {
            "photo_id": photo_id,
            "input_image": input_rel,
            "detections": detections,
            "total": len(detections),
        }
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


def get_job_or_404(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return job


@app.get("/jobs")
def list_jobs():
    """Job history for the Home / Previous Dives screens, most recent first."""
    jobs = []
    for job_id, job in JOBS.items():
        result = job.get("result")
        jobs.append({
            "job_id": job_id,
            "filename": job["video_name"],
            "status": job["status"],
            "uploaded_at": job["uploaded_at"],
            "species_count": len(result["species_summary"]) if result else None,
            "total_sightings": sum(s["unique_count"] for s in result["species_summary"]) if result else None,
            "flagged_count": len(result["review_queue"]) if result else None,
        })
    jobs.sort(key=lambda j: j["uploaded_at"], reverse=True)
    return {"jobs": jobs}


@app.get("/species")
def species_catalog():
    """Known species catalog (marine-detect's 18 classes) for the Fish Dex
    screen, marked seen/unseen and tallied against every completed job."""
    seen_counts: dict[str, int] = {}
    for job in JOBS.values():
        if job["status"] != "done":
            continue
        for row in job["result"]["species_summary"]:
            seen_counts[row["species"]] = seen_counts.get(row["species"], 0) + row["unique_count"]

    catalog = [
        {"species": common_name(species), "seen": species in seen_counts, "total_sightings": seen_counts.get(species, 0)}
        for species in RARITY_MAP
    ]
    catalog.sort(key=lambda s: (-s["seen"], s["species"]))
    return {"species": catalog}


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    job = get_job_or_404(job_id)
    region = job.get("region", "indo-pacific")
    return {
        "job_id": job_id,
        "status": job["status"],
        "error": job["error"],
        "region": region,
        "region_info": REGIONS.get(region, ""),
    }


@app.get("/jobs/{job_id}/results")
def get_job_results(job_id: str):
    job = get_job_or_404(job_id)
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job is {job['status']}, not ready yet")
    return job["result"]


@app.get("/jobs/{job_id}/report.csv")
def get_job_csv(job_id: str):
    job = get_job_or_404(job_id)
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job is {job['status']}, not ready yet")
    return FileResponse(OUTPUT_DIR / job_id / "report.csv", filename=f"divebuddy_report_{job_id}.csv")


@app.get("/jobs/{job_id}/report.pdf")
def get_job_pdf(job_id: str):
    job = get_job_or_404(job_id)
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job is {job['status']}, not ready yet")
    return FileResponse(OUTPUT_DIR / job_id / "report.pdf", filename=f"divebuddy_report_{job_id}.pdf")


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    species: str
    correct: bool
    corrected_species: str | None = None


@app.post("/jobs/{job_id}/feedback")
async def submit_feedback(job_id: str, req: FeedbackRequest):
    get_job_or_404(job_id)
    feedback_dir = OUTPUT_DIR / job_id
    feedback_dir.mkdir(parents=True, exist_ok=True)
    feedback_path = feedback_dir / "feedback.json"

    existing = []
    if feedback_path.exists():
        existing = json.loads(feedback_path.read_text())

    existing.append({
        "species": req.species,
        "correct": req.correct,
        "corrected_species": req.corrected_species,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    })

    feedback_path.write_text(json.dumps(existing, indent=2))
    return {"status": "saved", "total_feedback": len(existing)}


@app.get("/jobs/{job_id}/feedback")
async def get_feedback(job_id: str):
    get_job_or_404(job_id)
    feedback_path = OUTPUT_DIR / job_id / "feedback.json"
    if not feedback_path.exists():
        return {"feedback": []}
    return {"feedback": json.loads(feedback_path.read_text())}


# ---------------------------------------------------------------------------
# AI Chatbot (built-in, no external API needed)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


def _smart_reply(msg: str, result: dict, video_name: str) -> str:
    msg_lower = msg.lower().strip()
    summary = result["species_summary"]
    review = result["review_queue"]
    total_species = len(summary)
    total_count = sum(s["unique_count"] for s in summary)
    rare = [s for s in summary if s["rarity"] == "rare"]
    uncommon = [s for s in summary if s["rarity"] == "uncommon"]
    schools = [s for s in summary if s.get("is_school")]

    if any(w in msg_lower for w in ["what did you find", "what was found", "summary", "results", "what did i see", "overview", "tell me"]):
        if total_species == 0:
            return (
                f"From your video '{video_name}', no species were confidently identified "
                f"(need 5+ high-confidence detections to auto-count). "
                f"However, {len(review)} detection(s) are in the review queue for you to check manually."
            )
        lines = [f"From your video '{video_name}', I found {total_count} individual(s) across {total_species} species:"]
        for s in summary:
            lines.append(f"  - {s['species']}: {s['unique_count']} (confidence: {s['avg_confidence']:.0%}, {s['rarity']})")
        if review:
            lines.append(f"\nPlus {len(review)} detection(s) flagged for manual review.")
        return "\n".join(lines)

    if any(w in msg_lower for w in ["rare", "special", "unusual", "exciting"]):
        if rare:
            names = ", ".join(s["species"] for s in rare)
            return f"Great spot! You found rare species: {names}. These are uncommon sightings worth noting in your dive log."
        if uncommon:
            names = ", ".join(s["species"] for s in uncommon)
            return f"No rare species this time, but you did see some uncommon ones: {names}."
        return "No rare or uncommon species were detected in this dive. Keep diving — rare sightings happen when you least expect them!"

    if any(w in msg_lower for w in ["school", "group", "shoal", "many"]):
        if schools:
            names = ", ".join(s["species"] for s in schools)
            return f"Yes! Schools were detected: {names}. When 5+ individuals of the same species appear in a single frame, DiveBuddy flags it as a school."
        return "No schools were detected in this video. Schools are flagged when 5+ individuals of the same species appear together in the same frame."

    if any(w in msg_lower for w in ["review", "flagged", "unsure", "low confidence", "queue"]):
        if not review:
            return "Nothing was flagged for review — every detection was confident enough to auto-count. Nice clean results!"
        lines = [f"{len(review)} detection(s) need your review:"]
        for r in review:
            lines.append(f"  - {r['species_guess']} at {r['timestamp_sec']}s (confidence: {r['max_confidence']:.0%})")
        lines.append("\nUse the Correct/Wrong buttons to provide feedback — this helps improve future accuracy.")
        return "\n".join(lines)

    if any(w in msg_lower for w in ["how", "pipeline", "model", "detect", "work", "yolo", "confidence"]):
        return (
            "DiveBuddy uses a 6-model YOLO detection pipeline:\n"
            "  1. FishSpecies — 481 species\n"
            "  2. FishInv — 15 classes (fish families + 3 named species)\n"
            "  3. Seychelles — 72 species (best for Indo-Pacific)\n"
            "  4. ReefFamilies — 13 reef fish families\n"
            "  5. MarineLife — jellyfish, crabs, starfish, eels, shells\n"
            "  6. MegaFauna — sharks, rays, sea turtles\n\n"
            "Each model runs on every frame. A tracker (BoT-SORT) follows the same fish across frames "
            "so it counts once, not once per frame. Detections above 85% confidence are auto-counted; "
            "below that goes to the review queue. Cross-model dedup ensures the same fish isn't double-counted "
            "when multiple models detect it, with more specific models (species-level) winning over generic ones. "
            "When 5+ individuals of a species appear together, they're logged as one school rather than individual fish."
        )

    if any(w in msg_lower for w in ["count", "how many", "total", "number"]):
        if total_species == 0:
            return f"No species were auto-counted. {len(review)} detection(s) are in the review queue for manual verification."
        return f"Total: {total_count} individual(s) across {total_species} species. Plus {len(review)} in the review queue."

    if any(w in msg_lower for w in ["best", "most", "common", "top"]):
        if summary:
            top = summary[0]
            return f"The most detected species was {top['species']} with {top['unique_count']} individual(s) at {top['avg_confidence']:.0%} average confidence ({top['rarity']})."
        return "No species were confidently detected in this video."

    if any(w in msg_lower for w in ["hi", "hello", "hey"]):
        return f"Hey! I'm DiveBuddy's assistant. I can help you understand your dive results — ask me about what was found, rare species, how the detection works, or anything about your dive."

    if any(w in msg_lower for w in ["thank", "thanks", "cheers"]):
        return "Happy diving! Let me know if you have any other questions about your results."

    return (
        f"I can help with your dive results! Try asking:\n"
        f"  - \"What did you find?\" — full summary\n"
        f"  - \"Any rare species?\" — highlight special sightings\n"
        f"  - \"What's in the review queue?\" — flagged detections\n"
        f"  - \"How does detection work?\" — pipeline explanation\n"
        f"  - \"How many fish total?\" — counts"
    )


@app.post("/jobs/{job_id}/chat")
async def chat(job_id: str, req: ChatRequest):
    job = get_job_or_404(job_id)
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job is {job['status']}, not ready yet")

    reply = _smart_reply(req.message, job["result"], job["video_name"])
    return {"reply": reply}
