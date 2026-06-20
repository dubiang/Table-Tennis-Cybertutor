import pandas as pd

from tabletennis_analyzer.core.comparison import compare_motion, summarize_differences


def test_compare_motion_includes_speed_and_acceleration_magnitude_differences():
    user = pd.DataFrame(
        [
            {"landmark": "RIGHT_WRIST", "time_norm": 0.0, "speed": 2.0, "acceleration_magnitude": 4.0},
            {"landmark": "RIGHT_WRIST", "time_norm": 1.0, "speed": 3.0, "acceleration_magnitude": 6.0},
        ]
    )
    coach = pd.DataFrame(
        [
            {"landmark": "RIGHT_WRIST", "time_norm": 0.0, "speed": 1.5, "acceleration_magnitude": 3.0},
            {"landmark": "RIGHT_WRIST", "time_norm": 1.0, "speed": 2.5, "acceleration_magnitude": 7.0},
        ]
    )

    diff = compare_motion(user, coach)
    summary = summarize_differences(diff)

    assert diff["speed_diff"].tolist() == [0.5, 0.5]
    assert diff["acceleration_magnitude_diff"].tolist() == [1.0, -1.0]
    assert summary["top_landmarks"][0]["landmark"] == "RIGHT_WRIST"


def test_compare_motion_includes_vector_angle_difference():
    user = pd.DataFrame(
        [
            {
                "landmark": "RIGHT_WRIST",
                "time_norm": 0.0,
                "x_velocity": 1.0,
                "y_velocity": 0.0,
                "z_velocity": 0.0,
                "x_acceleration": 0.0,
                "y_acceleration": 1.0,
                "z_acceleration": 0.0,
            }
        ]
    )
    coach = pd.DataFrame(
        [
            {
                "landmark": "RIGHT_WRIST",
                "time_norm": 0.0,
                "x_velocity": -1.0,
                "y_velocity": 0.0,
                "z_velocity": 0.0,
                "x_acceleration": 0.0,
                "y_acceleration": 1.0,
                "z_acceleration": 0.0,
            }
        ]
    )

    diff = compare_motion(user, coach)

    assert round(float(diff.loc[0, "velocity_angle_abs_diff_deg"]), 6) == 180.0
    assert round(float(diff.loc[0, "acceleration_angle_abs_diff_deg"]), 6) == 0.0
