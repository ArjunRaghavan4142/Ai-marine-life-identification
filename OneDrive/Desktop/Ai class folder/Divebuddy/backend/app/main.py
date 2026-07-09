import datetime
import json
import pathlib
import shutil
import traceback
import uuid

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.pipeline.aggregate import aggregate_tracks, common_name
from app.pipeline.detect import run_detection
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


@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
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
        "error": None,
        "result": None,
    }
    background_tasks.add_task(process_video, job_id, video_path)

    return {"job_id": job_id, "status": "processing", "filename": file.filename}


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
        frames = extract_frames(video_path, job_dir / "frames", fps_sample=5.0)
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
    return {"job_id": job_id, "status": job["status"], "error": job["error"]}


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
