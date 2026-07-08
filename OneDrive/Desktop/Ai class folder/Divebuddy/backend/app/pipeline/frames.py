"""Frame extraction: pulls roughly one frame per second out of a dive video."""
import pathlib

import cv2


def extract_frames(video_path: pathlib.Path, out_dir: pathlib.Path, fps_sample: float = 1.0) -> list[dict]:
    """Extracts frames at `fps_sample` frames/sec and writes them as JPEGs.

    Returns a list of {"path": ..., "frame_index": ..., "timestamp_sec": ...}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(int(round(video_fps / fps_sample)), 1)

    frames = []
    frame_index = 0
    saved_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % frame_interval == 0:
            timestamp_sec = frame_index / video_fps
            frame_path = out_dir / f"frame_{saved_index:05d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frames.append({
                "path": frame_path,
                "frame_index": saved_index,
                "timestamp_sec": round(timestamp_sec, 2),
            })
            saved_index += 1
        frame_index += 1

    cap.release()
    return frames
