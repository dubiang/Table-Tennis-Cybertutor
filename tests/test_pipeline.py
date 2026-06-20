from pathlib import Path

import pandas as pd

from tabletennis_analyzer.core.pipeline import VideoMotionAnalysisConfig, analyze_user_and_coach_motion, analyze_user_coach_differences, analyze_video_motion


def test_analyze_video_motion_exports_positions_kinematics_and_metadata(tmp_path, monkeypatch):
    video = tmp_path / "stroke.mp4"
    video.write_bytes(b"fake mp4 for pipeline test")

    def fake_metadata(path: Path):
        return {"fps": 2.0, "frame_count": 3, "duration_s": 1.5, "width": 640, "height": 480}

    def fake_extract(path: Path, config):
        return pd.DataFrame(
            [
                {"time_s": 0.0, "frame_index": 0, "landmark": "RIGHT_WRIST", "x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.9},
                {"time_s": 0.5, "frame_index": 1, "landmark": "RIGHT_WRIST", "x": 0.5, "y": 0.0, "z": 0.0, "visibility": 0.9},
                {"time_s": 1.0, "frame_index": 2, "landmark": "RIGHT_WRIST", "x": 1.0, "y": 0.0, "z": 0.0, "visibility": 0.9},
            ]
        )

    monkeypatch.setattr("tabletennis_analyzer.core.pipeline.read_video_metadata", fake_metadata)
    monkeypatch.setattr("tabletennis_analyzer.core.pipeline.extract_pose_landmarks", fake_extract)

    result = analyze_video_motion(video, tmp_path / "out", VideoMotionAnalysisConfig(label="stroke"))

    assert result.positions_csv.exists()
    assert result.kinematics_csv.exists()
    assert result.metadata_json.exists()
    assert result.detected_frames == 3
    exported = pd.read_csv(result.kinematics_csv)
    assert "speed" in exported
    assert "acceleration_magnitude" in exported
    assert result.joint_kinematics_csvs["RIGHT_WRIST"].exists()


def test_analyze_user_and_coach_motion_exports_two_sets_of_joint_functions(tmp_path, monkeypatch):
    user_video = tmp_path / "user.mp4"
    coach_video = tmp_path / "coach.mp4"
    user_video.write_bytes(b"user")
    coach_video.write_bytes(b"coach")

    def fake_metadata(path: Path):
        return {"fps": 2.0, "frame_count": 3, "duration_s": 1.5, "width": 640, "height": 480}

    def fake_extract(path: Path, config):
        offset = 0.0 if path.name == "user.mp4" else 0.2
        return pd.DataFrame(
            [
                {"time_s": 0.0, "frame_index": 0, "landmark": "RIGHT_WRIST", "x": 0.0 + offset, "y": 0.0, "z": 0.0, "visibility": 0.9},
                {"time_s": 0.5, "frame_index": 1, "landmark": "RIGHT_WRIST", "x": 0.5 + offset, "y": 0.0, "z": 0.0, "visibility": 0.9},
                {"time_s": 1.0, "frame_index": 2, "landmark": "RIGHT_WRIST", "x": 1.0 + offset, "y": 0.0, "z": 0.0, "visibility": 0.9},
                {"time_s": 0.0, "frame_index": 0, "landmark": "RIGHT_ELBOW", "x": 0.0 + offset, "y": 0.5, "z": 0.0, "visibility": 0.9},
                {"time_s": 0.5, "frame_index": 1, "landmark": "RIGHT_ELBOW", "x": 0.2 + offset, "y": 0.5, "z": 0.0, "visibility": 0.9},
                {"time_s": 1.0, "frame_index": 2, "landmark": "RIGHT_ELBOW", "x": 0.4 + offset, "y": 0.5, "z": 0.0, "visibility": 0.9},
            ]
        )

    monkeypatch.setattr("tabletennis_analyzer.core.pipeline.read_video_metadata", fake_metadata)
    monkeypatch.setattr("tabletennis_analyzer.core.pipeline.extract_pose_landmarks", fake_extract)

    result = analyze_user_and_coach_motion(user_video, coach_video, tmp_path / "out")

    assert result.user.kinematics_csv.exists()
    assert result.coach.kinematics_csv.exists()
    assert result.summary_json.exists()
    assert set(result.user.joint_kinematics_csvs) == {"RIGHT_ELBOW", "RIGHT_WRIST"}
    assert set(result.coach.joint_kinematics_csvs) == {"RIGHT_ELBOW", "RIGHT_WRIST"}


def test_analyze_user_coach_differences_exports_diff_functions_and_llm_text(tmp_path, monkeypatch):
    user_video = tmp_path / "user.mp4"
    coach_video = tmp_path / "coach.mp4"
    user_video.write_bytes(b"user")
    coach_video.write_bytes(b"coach")

    def fake_metadata(path: Path):
        return {"fps": 2.0, "frame_count": 3, "duration_s": 1.5, "width": 640, "height": 480}

    def fake_extract(path: Path, config):
        offset = 0.0 if path.name == "user.mp4" else 0.2
        return pd.DataFrame(
            [
                {"time_s": 0.0, "frame_index": 0, "landmark": "RIGHT_WRIST", "x": 0.0 + offset, "y": 0.0, "z": 0.0, "visibility": 0.9},
                {"time_s": 0.5, "frame_index": 1, "landmark": "RIGHT_WRIST", "x": 0.5 + offset, "y": 0.0, "z": 0.0, "visibility": 0.9},
                {"time_s": 1.0, "frame_index": 2, "landmark": "RIGHT_WRIST", "x": 1.2 + offset, "y": 0.0, "z": 0.0, "visibility": 0.9},
            ]
        )

    def fake_llm(api_key: str, model: str, summary: dict):
        assert api_key == "test-key"
        assert model == "gpt-5.5"
        assert summary["status"] == "ok"
        return "需要提高右手腕加速稳定性。"

    monkeypatch.setattr("tabletennis_analyzer.core.pipeline.read_video_metadata", fake_metadata)
    monkeypatch.setattr("tabletennis_analyzer.core.pipeline.extract_pose_landmarks", fake_extract)
    monkeypatch.setattr("tabletennis_analyzer.llm.openai_client.analyze_motion_differences", fake_llm)

    result = analyze_user_coach_differences(
        user_video,
        coach_video,
        tmp_path / "out",
        samples=5,
        api_key="test-key",
        model="gpt-5.5",
    )

    diff = pd.read_csv(result.difference_csv)
    assert "speed_diff" in diff
    assert "acceleration_magnitude_diff" in diff
    assert result.difference_summary_json.exists()
    assert result.joint_difference_csvs["RIGHT_WRIST"].exists()
    assert result.llm_analysis_txt is not None
    assert result.llm_analysis_txt.read_text(encoding="utf-8") == "需要提高右手腕加速稳定性。"
