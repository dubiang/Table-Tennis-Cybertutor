from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


COORDINATES = ("x", "y", "z")


def add_normalized_time(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    result = frame.copy()
    min_t = float(result["time_s"].min())
    max_t = float(result["time_s"].max())
    span = max(max_t - min_t, 1e-9)
    result["time_norm"] = (result["time_s"] - min_t) / span
    return result


def smooth_and_derive(frame: pd.DataFrame, window_length: int = 9, polyorder: int = 2) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    result_parts: list[pd.DataFrame] = []
    for landmark, group in frame.sort_values(["landmark", "time_s"]).groupby("landmark"):
        group = group.copy()
        times = group["time_s"].to_numpy(dtype=float)

        for coord in COORDINATES:
            values = group[coord].to_numpy(dtype=float)
            smoothed = _smooth(values, window_length=window_length, polyorder=polyorder)
            velocity = _gradient(smoothed, times)
            acceleration = _gradient(velocity, times)
            group[f"{coord}_smooth"] = smoothed
            group[f"{coord}_velocity"] = velocity
            group[f"{coord}_acceleration"] = acceleration

        result_parts.append(group)

    return pd.concat(result_parts, ignore_index=True)


def add_motion_magnitudes(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    result = frame.copy()
    velocity_columns = [f"{coord}_velocity" for coord in COORDINATES]
    acceleration_columns = [f"{coord}_acceleration" for coord in COORDINATES]

    if all(column in result for column in velocity_columns):
        result["speed"] = np.sqrt(sum(result[column] ** 2 for column in velocity_columns))
    if all(column in result for column in acceleration_columns):
        result["acceleration_magnitude"] = np.sqrt(sum(result[column] ** 2 for column in acceleration_columns))

    return result


def resample_by_normalized_time(frame: pd.DataFrame, samples: int = 101) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    frame = add_normalized_time(frame)
    grid = np.linspace(0.0, 1.0, samples)
    value_columns = [
        column
        for column in frame.columns
        if column in COORDINATES
        or column in ("speed", "acceleration_magnitude")
        or column.endswith("_smooth")
        or column.endswith("_velocity")
        or column.endswith("_acceleration")
    ]

    rows: list[dict[str, float | str]] = []
    for landmark, group in frame.sort_values("time_norm").groupby("landmark"):
        group = group.drop_duplicates(subset=["time_norm"])
        source_t = group["time_norm"].to_numpy(dtype=float)
        if len(source_t) < 2:
            continue

        for target_t in grid:
            row: dict[str, float | str] = {"landmark": landmark, "time_norm": float(target_t)}
            for column in value_columns:
                row[column] = float(np.interp(target_t, source_t, group[column].to_numpy(dtype=float)))
            rows.append(row)

    return pd.DataFrame(rows)


def _smooth(values: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    if len(values) < 5:
        return values

    usable_window = min(window_length, len(values) if len(values) % 2 else len(values) - 1)
    usable_window = max(usable_window, polyorder + 2)
    if usable_window % 2 == 0:
        usable_window -= 1

    if usable_window <= polyorder or usable_window < 3:
        return values

    return savgol_filter(values, window_length=usable_window, polyorder=polyorder)


def _gradient(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values)
    return np.gradient(values, times, edge_order=1)
