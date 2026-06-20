from __future__ import annotations

import numpy as np
import pandas as pd


KINEMATIC_SUFFIXES = ("smooth", "velocity", "acceleration")
COORDINATES = ("x", "y", "z")
SCALAR_KINEMATICS = ("speed", "acceleration_magnitude")


def compare_motion(user_frame: pd.DataFrame, coach_frame: pd.DataFrame) -> pd.DataFrame:
    if user_frame.empty or coach_frame.empty:
        return pd.DataFrame()

    merge_keys = ["landmark", "time_norm"]
    merged = user_frame.merge(coach_frame, on=merge_keys, suffixes=("_user", "_coach"))
    result = merged[merge_keys].copy()

    for coord in COORDINATES:
        for suffix in KINEMATIC_SUFFIXES:
            base = f"{coord}_{suffix}"
            user_col = f"{base}_user"
            coach_col = f"{base}_coach"
            if user_col in merged and coach_col in merged:
                result[f"{base}_diff"] = merged[user_col] - merged[coach_col]
                result[f"{base}_abs_diff"] = result[f"{base}_diff"].abs()

    for base in SCALAR_KINEMATICS:
        user_col = f"{base}_user"
        coach_col = f"{base}_coach"
        if user_col in merged and coach_col in merged:
            result[f"{base}_diff"] = merged[user_col] - merged[coach_col]
            result[f"{base}_abs_diff"] = result[f"{base}_diff"].abs()

    _add_vector_angle_difference(result, merged, "velocity")
    _add_vector_angle_difference(result, merged, "acceleration")

    return result


def summarize_differences(diff_frame: pd.DataFrame, top_n: int = 8) -> dict[str, object]:
    if diff_frame.empty:
        return {"status": "empty", "landmarks": []}

    metric_columns = [column for column in diff_frame.columns if column.endswith("_abs_diff")]
    landmark_summaries = []

    for landmark, group in diff_frame.groupby("landmark"):
        metrics = {
            column: {
                "mean": float(group[column].mean()),
                "max": float(group[column].max()),
            }
            for column in metric_columns
        }
        priority_score = _priority_score(group)
        peak_time_norm = _peak_time_norm(group)
        landmark_summaries.append(
            {
                "landmark": landmark,
                "priority_score": priority_score,
                "peak_time_norm": peak_time_norm,
                "metrics": metrics,
            }
        )

    landmark_summaries.sort(key=lambda item: float(item["priority_score"]), reverse=True)

    return {
        "status": "ok",
        "landmarks": landmark_summaries,
        "top_landmarks": landmark_summaries[:top_n],
        "metric_columns": metric_columns,
        "diff_definition": "diff = user - coach on normalized time axis",
        "time_axis": "time_norm ranges from 0.0 to 1.0 for each stroke/video",
        "coordinate_caveat": "MediaPipe coordinates are normalized/relative image-space values, not calibrated meters.",
        "angle_metrics": "velocity_angle_abs_diff_deg and acceleration_angle_abs_diff_deg compare vector direction, not speed magnitude.",
    }


def _priority_score(group: pd.DataFrame) -> float:
    preferred = [
        "speed_abs_diff",
        "acceleration_magnitude_abs_diff",
        "x_velocity_abs_diff",
        "y_velocity_abs_diff",
        "z_velocity_abs_diff",
        "x_acceleration_abs_diff",
        "y_acceleration_abs_diff",
        "z_acceleration_abs_diff",
        "velocity_angle_abs_diff_deg",
        "acceleration_angle_abs_diff_deg",
    ]
    columns = [column for column in preferred if column in group]
    if not columns:
        columns = [column for column in group.columns if column.endswith("_abs_diff")]
    if not columns:
        return 0.0
    return float(group[columns].mean().mean())


def _peak_time_norm(group: pd.DataFrame) -> float | None:
    columns = [column for column in ("speed_abs_diff", "acceleration_magnitude_abs_diff") if column in group]
    if not columns:
        return None
    scores = group[columns].sum(axis=1)
    if scores.empty:
        return None
    return float(group.loc[scores.idxmax(), "time_norm"])


def _add_vector_angle_difference(result: pd.DataFrame, merged: pd.DataFrame, suffix: str) -> None:
    user_columns = [f"{coord}_{suffix}_user" for coord in COORDINATES]
    coach_columns = [f"{coord}_{suffix}_coach" for coord in COORDINATES]
    if not all(column in merged for column in user_columns + coach_columns):
        return

    user_vectors = merged[user_columns].to_numpy(dtype=float)
    coach_vectors = merged[coach_columns].to_numpy(dtype=float)
    user_norm = np.linalg.norm(user_vectors, axis=1)
    coach_norm = np.linalg.norm(coach_vectors, axis=1)
    denom = user_norm * coach_norm

    angles = np.full(len(merged), np.nan, dtype=float)
    valid = denom > 1e-12
    if valid.any():
        cos_values = np.sum(user_vectors[valid] * coach_vectors[valid], axis=1) / denom[valid]
        angles[valid] = np.degrees(np.arccos(np.clip(cos_values, -1.0, 1.0)))

    result[f"{suffix}_angle_diff_deg"] = angles
    result[f"{suffix}_angle_abs_diff_deg"] = np.abs(angles)
