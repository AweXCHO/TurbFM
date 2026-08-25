from __future__ import annotations

from pathlib import Path

import numpy as np


def read_video_frames(path: str | Path, frame_count: int = 15, image_size: int = 224) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"opencv-python is required to read AVI files: {exc}") from exc

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    frames = []
    try:
        for idx in range(frame_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame.shape[0] != image_size or frame.shape[1] != image_size:
                frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {path}")
    while len(frames) < frame_count:
        frames.append(frames[-1].copy())
    arr = np.stack(frames[:frame_count]).astype("float32") / 255.0
    return arr.transpose(0, 3, 1, 2)
