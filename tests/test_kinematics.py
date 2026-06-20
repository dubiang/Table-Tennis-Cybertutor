import pandas as pd

from tabletennis_analyzer.core.kinematics import add_motion_magnitudes, add_normalized_time, resample_by_normalized_time, smooth_and_derive


def test_add_normalized_time_spans_zero_to_one():
    frame = pd.DataFrame(
        [
            {"time_s": 2.0, "landmark": "RIGHT_WRIST", "x": 0.0, "y": 0.0, "z": 0.0},
            {"time_s": 4.0, "landmark": "RIGHT_WRIST", "x": 1.0, "y": 1.0, "z": 1.0},
        ]
    )

    result = add_normalized_time(frame)

    assert result["time_norm"].tolist() == [0.0, 1.0]


def test_resample_by_normalized_time_returns_requested_samples():
    frame = pd.DataFrame(
        [
            {"time_s": 0.0, "landmark": "RIGHT_WRIST", "x": 0.0, "y": 0.0, "z": 0.0},
            {"time_s": 1.0, "landmark": "RIGHT_WRIST", "x": 1.0, "y": 1.0, "z": 1.0},
        ]
    )

    result = resample_by_normalized_time(frame, samples=5)

    assert len(result) == 5
    assert result["x"].tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_smooth_and_derive_adds_velocity_acceleration_and_magnitudes():
    frame = pd.DataFrame(
        [
            {"time_s": 0.0, "frame_index": 0, "landmark": "RIGHT_WRIST", "x": 0.0, "y": 0.0, "z": 0.0},
            {"time_s": 0.5, "frame_index": 1, "landmark": "RIGHT_WRIST", "x": 0.5, "y": 0.0, "z": 0.0},
            {"time_s": 1.0, "frame_index": 2, "landmark": "RIGHT_WRIST", "x": 1.0, "y": 0.0, "z": 0.0},
        ]
    )

    result = add_motion_magnitudes(smooth_and_derive(frame))

    assert "x_velocity" in result
    assert "x_acceleration" in result
    assert "speed" in result
    assert "acceleration_magnitude" in result
    assert result["speed"].round(6).tolist() == [1.0, 1.0, 1.0]
