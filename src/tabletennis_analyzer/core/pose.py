from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import pandas as pd


TABLE_TENNIS_LANDMARKS = {
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
}


@dataclass(frozen=True)
class PoseExtractionConfig:
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    max_frames: int | None = None


def read_video_metadata(video_path: str | Path) -> dict[str, Any]:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": frame_count / fps if fps else 0.0,
        "width": width,
        "height": height,
    }


def extract_pose_landmarks(
    video_path: str | Path,
    config: PoseExtractionConfig | None = None,
) -> pd.DataFrame:
    import mediapipe as mp

    config = config or PoseExtractionConfig()
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    pose_module = mp.solutions.pose
    rows: list[dict[str, float | int | str]] = []

    with pose_module.Pose(
        model_complexity=config.model_complexity,
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    ) as pose:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if config.max_frames is not None and frame_index >= config.max_frames:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)
            time_s = frame_index / fps

            if result.pose_landmarks:
                for landmark_id, landmark in enumerate(result.pose_landmarks.landmark):
                    name = pose_module.PoseLandmark(landmark_id).name
                    if name not in TABLE_TENNIS_LANDMARKS:
                        continue
                    rows.append(
                        {
                            "time_s": time_s,
                            "frame_index": frame_index,
                            "landmark": name,
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z,
                            "visibility": landmark.visibility,
                        }
                    )

            frame_index += 1

    capture.release()
    return pd.DataFrame(rows)
