from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from tabletennis_analyzer.core.comparison import compare_motion, summarize_differences
from tabletennis_analyzer.core.export import export_dataframe, export_json
from tabletennis_analyzer.core.kinematics import add_motion_magnitudes, resample_by_normalized_time, smooth_and_derive
from tabletennis_analyzer.core.pose import PoseExtractionConfig, extract_pose_landmarks, read_video_metadata


@dataclass(frozen=True)
class VideoMotionAnalysisConfig:
    label: str = "video"
    max_frames: int | None = None
    smooth_window_length: int = 9
    smooth_polyorder: int = 2


@dataclass(frozen=True)
class VideoMotionAnalysisResult:
    video_path: Path
    output_dir: Path
    positions_csv: Path
    kinematics_csv: Path
    metadata_json: Path
    joint_kinematics_dir: Path
    joint_kinematics_csvs: dict[str, Path]
    frame_count: int
    detected_frames: int
    landmark_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "video_path": str(self.video_path),
            "output_dir": str(self.output_dir),
            "positions_csv": str(self.positions_csv),
            "kinematics_csv": str(self.kinematics_csv),
            "metadata_json": str(self.metadata_json),
            "joint_kinematics_dir": str(self.joint_kinematics_dir),
            "joint_kinematics_csvs": {key: str(value) for key, value in self.joint_kinematics_csvs.items()},
            "frame_count": self.frame_count,
            "detected_frames": self.detected_frames,
            "landmark_count": self.landmark_count,
        }


@dataclass(frozen=True)
class PairedMotionAnalysisResult:
    user: VideoMotionAnalysisResult
    coach: VideoMotionAnalysisResult
    summary_json: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "user": self.user.as_dict(),
            "coach": self.coach.as_dict(),
            "summary_json": str(self.summary_json),
        }


@dataclass(frozen=True)
class DifferenceAnalysisResult:
    paired_motion: PairedMotionAnalysisResult
    difference_csv: Path
    difference_summary_json: Path
    joint_difference_dir: Path
    joint_difference_csvs: dict[str, Path]
    summary: dict[str, Any]
    llm_analysis_text: str | None = None
    llm_analysis_txt: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "paired_motion": self.paired_motion.as_dict(),
            "difference_csv": str(self.difference_csv),
            "difference_summary_json": str(self.difference_summary_json),
            "joint_difference_dir": str(self.joint_difference_dir),
            "joint_difference_csvs": {key: str(value) for key, value in self.joint_difference_csvs.items()},
            "llm_analysis_txt": str(self.llm_analysis_txt) if self.llm_analysis_txt else None,
        }


def analyze_video_motion(
    video_path: str | Path,
    output_dir: str | Path,
    config: VideoMotionAnalysisConfig | None = None,
) -> VideoMotionAnalysisResult:
    config = config or VideoMotionAnalysisConfig()
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if video_path.suffix.lower() != ".mp4":
        raise ValueError("Only .mp4 input is enabled for this first milestone.")

    metadata = read_video_metadata(video_path)
    pose_config = PoseExtractionConfig(max_frames=config.max_frames)
    positions = extract_pose_landmarks(video_path, pose_config)

    if positions.empty:
        raise ValueError("No pose landmarks were detected in this video.")

    kinematics = smooth_and_derive(
        positions,
        window_length=config.smooth_window_length,
        polyorder=config.smooth_polyorder,
    )
    kinematics = add_motion_magnitudes(kinematics)

    positions_csv = export_dataframe(positions, output_dir, f"{config.label}_positions.csv")
    kinematics_csv = export_dataframe(kinematics, output_dir, f"{config.label}_kinematics.csv")
    joint_kinematics_dir = output_dir / "joints"
    joint_kinematics_csvs = _export_joint_kinematics(kinematics, joint_kinematics_dir, config.label)

    detected_frames = int(positions["frame_index"].nunique())
    landmark_count = int(positions["landmark"].nunique())
    landmarks = sorted(str(landmark) for landmark in positions["landmark"].unique())
    metadata_payload: dict[str, Any] = {
        "input": str(video_path),
        "video": metadata,
        "analysis": {
            "label": config.label,
            "frame_count": int(metadata.get("frame_count") or 0),
            "processed_frames": int(config.max_frames or metadata.get("frame_count") or 0),
            "detected_frames": detected_frames,
            "landmark_count": landmark_count,
            "landmarks": landmarks,
            "coordinates": ["x", "y", "z"],
            "derived_columns": [
                "x_smooth",
                "y_smooth",
                "z_smooth",
                "x_velocity",
                "y_velocity",
                "z_velocity",
                "speed",
                "x_acceleration",
                "y_acceleration",
                "z_acceleration",
                "acceleration_magnitude",
            ],
            "calculation": "velocity and acceleration are numerical derivatives of smoothed landmark positions over time_s",
            "physical_units_caveat": "MediaPipe x/y/z coordinates are normalized/relative values. Derived velocity and acceleration are relative image-space quantities, not calibrated m/s or m/s^2.",
            "row_grain": "one row per video frame and body landmark",
        },
    }
    metadata_json = export_json(metadata_payload, output_dir, f"{config.label}_metadata.json")

    return VideoMotionAnalysisResult(
        video_path=video_path,
        output_dir=output_dir,
        positions_csv=positions_csv,
        kinematics_csv=kinematics_csv,
        metadata_json=metadata_json,
        joint_kinematics_dir=joint_kinematics_dir,
        joint_kinematics_csvs=joint_kinematics_csvs,
        frame_count=int(metadata.get("frame_count") or 0),
        detected_frames=detected_frames,
        landmark_count=landmark_count,
    )


def analyze_user_and_coach_motion(
    user_video: str | Path,
    coach_video: str | Path,
    output_dir: str | Path,
    max_frames: int | None = None,
) -> PairedMotionAnalysisResult:
    output_dir = Path(output_dir)
    user_result = analyze_video_motion(
        user_video,
        output_dir / "user",
        VideoMotionAnalysisConfig(label="user", max_frames=max_frames),
    )
    coach_result = analyze_video_motion(
        coach_video,
        output_dir / "coach",
        VideoMotionAnalysisConfig(label="coach", max_frames=max_frames),
    )
    summary_json = export_json(
        {
            "user": user_result.as_dict(),
            "coach": coach_result.as_dict(),
            "note": "Each kinematics CSV contains one row per frame and body joint. Velocity and acceleration columns are per-joint derivatives.",
            "physical_units_caveat": "Velocity and acceleration are relative MediaPipe-coordinate quantities unless camera calibration and scale conversion are added.",
        },
        output_dir,
        "user_coach_motion_summary.json",
    )
    return PairedMotionAnalysisResult(user=user_result, coach=coach_result, summary_json=summary_json)


def analyze_user_coach_differences(
    user_video: str | Path,
    coach_video: str | Path,
    output_dir: str | Path,
    max_frames: int | None = None,
    samples: int = 101,
    api_key: str | None = None,
    model: str = "gpt-5.5",
) -> DifferenceAnalysisResult:
    from tabletennis_analyzer.llm.openai_client import analyze_motion_differences

    output_dir = Path(output_dir)
    paired = analyze_user_and_coach_motion(user_video, coach_video, output_dir, max_frames=max_frames)

    user_kinematics = pd.read_csv(paired.user.kinematics_csv)
    coach_kinematics = pd.read_csv(paired.coach.kinematics_csv)
    user_resampled = resample_by_normalized_time(user_kinematics, samples=samples)
    coach_resampled = resample_by_normalized_time(coach_kinematics, samples=samples)
    diff = compare_motion(user_resampled, coach_resampled)

    difference_csv = export_dataframe(diff, output_dir, "velocity_acceleration_difference_functions.csv")
    joint_difference_dir = output_dir / "difference_joints"
    joint_difference_csvs = _export_joint_differences(diff, joint_difference_dir)
    summary = summarize_differences(diff)
    summary["artifacts"] = {
        "difference_csv": str(difference_csv),
        "joint_difference_dir": str(joint_difference_dir),
        "user_kinematics_csv": str(paired.user.kinematics_csv),
        "coach_kinematics_csv": str(paired.coach.kinematics_csv),
    }
    summary["physical_units_caveat"] = (
        "All speed/acceleration values are relative quantities derived from MediaPipe normalized coordinates, "
        "not real-world m/s or m/s^2."
    )
    difference_summary_json = export_json(summary, output_dir, "velocity_acceleration_difference_summary.json")

    llm_analysis_text: str | None = None
    llm_analysis_txt: Path | None = None
    if api_key and api_key.strip():
        llm_analysis_text = analyze_motion_differences(api_key, model, summary)
        llm_analysis_txt = output_dir / "gpt_motion_analysis.txt"
        llm_analysis_txt.parent.mkdir(parents=True, exist_ok=True)
        llm_analysis_txt.write_text(llm_analysis_text, encoding="utf-8")

    return DifferenceAnalysisResult(
        paired_motion=paired,
        difference_csv=difference_csv,
        difference_summary_json=difference_summary_json,
        joint_difference_dir=joint_difference_dir,
        joint_difference_csvs=joint_difference_csvs,
        summary=summary,
        llm_analysis_text=llm_analysis_text,
        llm_analysis_txt=llm_analysis_txt,
    )


def _export_joint_kinematics(kinematics, output_dir: Path, label: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for landmark, group in kinematics.sort_values(["landmark", "time_s"]).groupby("landmark"):
        landmark_name = str(landmark)
        path = export_dataframe(group, output_dir, f"{label}_{landmark_name}_kinematics.csv")
        paths[landmark_name] = path
    return paths


def _export_joint_differences(diff_frame: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if diff_frame.empty:
        return paths
    for landmark, group in diff_frame.sort_values(["landmark", "time_norm"]).groupby("landmark"):
        landmark_name = str(landmark)
        path = export_dataframe(group, output_dir, f"{landmark_name}_difference_functions.csv")
        paths[landmark_name] = path
    return paths
