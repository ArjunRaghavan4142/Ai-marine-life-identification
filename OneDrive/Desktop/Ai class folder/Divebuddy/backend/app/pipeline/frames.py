"""Frame extraction: pulls roughly one frame per second out of a dive video."""
import pathlib
import cv2

def extract_frames(
    video_path: pathlib.Path,
    out_dir: pathlib.Path,
    fps_sample: float = 2.0
) -> dict:
    """
    Extracts frames at `fps_sample` frames/sec and writes them as JPEGs.

    Returns:
        {
            "frames":        list of {"path", "frame_index", "timestamp_sec"},
            "duration_sec":  total video duration,
            "video_fps":     original video frame rate,
            "total_frames":  total frames in video,
            "sampled_count": number of frames extracted,
        }
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec  = round(total_frames / video_fps, 2)
    frame_interval = max(int(round(video_fps / fps_sample)), 1)

    frames      = []
    frame_index = 0
    saved_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % frame_interval == 0:
            timestamp_sec = round(frame_index / video_fps, 2)
            frame_path    = out_dir / f"frame_{saved_index:05d}.jpg"
            cv2.imwrite(
                str(frame_path), frame,
                [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            frames.append({
                "path":          frame_path,
                "frame_index":   saved_index,
                "timestamp_sec": timestamp_sec,
            })
            saved_index += 1

        if frame_index % (frame_interval * 60) == 0 and frame_index > 0:
            mins = int((frame_index / video_fps) / 60)
            print(f"  Extracting... {mins}min processed, {saved_index} frames saved")

        frame_index += 1

    cap.release()
    print(f"  Done — {saved_index} frames from {duration_sec}s video")

    return {
        "frames":        frames,
        "duration_sec":  duration_sec,
        "video_fps":     video_fps,
        "total_frames":  total_frames,
        "sampled_count": saved_index,
    }
