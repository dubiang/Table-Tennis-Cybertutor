from __future__ import annotations

import argparse
from pathlib import Path

from tabletennis_analyzer.core.pipeline import (
    VideoMotionAnalysisConfig,
    analyze_user_and_coach_motion,
    analyze_user_coach_differences,
    analyze_video_motion,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export table-tennis pose and per-joint kinematics from MP4 videos.")
    subparsers = parser.add_subparsers(dest="command")

    single_parser = subparsers.add_parser("single", help="Analyze one MP4.")
    single_parser.add_argument("video", type=Path, help="Input MP4 video path.")
    single_parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for exported files.")
    single_parser.add_argument("--label", default="video", help="Filename prefix for exported files.")
    single_parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for smoke tests.")

    pair_parser = subparsers.add_parser("pair", help="Analyze user and coach MP4 videos.")
    pair_parser.add_argument("--user-video", required=True, type=Path, help="User MP4 video path.")
    pair_parser.add_argument("--coach-video", required=True, type=Path, help="Coach/reference MP4 video path.")
    pair_parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "user_coach_motion", help="Directory for exported files.")
    pair_parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for smoke tests.")

    diff_parser = subparsers.add_parser("diff", help="Analyze user-vs-coach speed and acceleration differences.")
    diff_parser.add_argument("--user-video", required=True, type=Path, help="User MP4 video path.")
    diff_parser.add_argument("--coach-video", required=True, type=Path, help="Coach/reference MP4 video path.")
    diff_parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "difference_analysis", help="Directory for exported files.")
    diff_parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for smoke tests.")
    diff_parser.add_argument("--samples", type=int, default=101, help="Number of normalized-time samples for difference functions.")
    diff_parser.add_argument("--api-key", default="", help="OpenAI API key. Prefer entering it in the UI for normal use.")
    diff_parser.add_argument("--model", default="gpt-5.5", help="Model name.")

    args = parser.parse_args()

    if args.command == "pair":
        result = analyze_user_and_coach_motion(
            args.user_video,
            args.coach_video,
            args.output_dir,
            max_frames=args.max_frames,
        )
        print("User and coach motion analysis complete")
        print(f"User kinematics: {result.user.kinematics_csv}")
        print(f"User joint CSV dir: {result.user.joint_kinematics_dir}")
        print(f"Coach kinematics: {result.coach.kinematics_csv}")
        print(f"Coach joint CSV dir: {result.coach.joint_kinematics_dir}")
        print(f"Summary: {result.summary_json}")
        return 0

    if args.command == "diff":
        result = analyze_user_coach_differences(
            args.user_video,
            args.coach_video,
            args.output_dir,
            max_frames=args.max_frames,
            samples=args.samples,
            api_key=args.api_key,
            model=args.model,
        )
        print("Velocity and acceleration difference analysis complete")
        print(f"Difference functions: {result.difference_csv}")
        print(f"Joint difference CSV dir: {result.joint_difference_dir}")
        print(f"Difference summary: {result.difference_summary_json}")
        if result.llm_analysis_txt:
            print(f"GPT analysis: {result.llm_analysis_txt}")
        else:
            print("GPT analysis skipped because no API key was provided.")
        return 0

    if args.command is None:
        parser.print_help()
        return 2

    result = analyze_video_motion(args.video, args.output_dir, VideoMotionAnalysisConfig(label=args.label, max_frames=args.max_frames))
    print("Analysis complete")
    print(f"Positions: {result.positions_csv}")
    print(f"Kinematics: {result.kinematics_csv}")
    print(f"Joint CSV dir: {result.joint_kinematics_dir}")
    print(f"Metadata: {result.metadata_json}")
    print(f"Detected frames: {result.detected_frames}")
    print(f"Landmarks: {result.landmark_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
