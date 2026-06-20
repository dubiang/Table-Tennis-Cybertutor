# Table Tennis Motion Analyzer

Windows desktop prototype for comparing a user's table-tennis motion video with a coach/reference video.

The application extracts body landmarks from both videos, exports time-based joint position functions, derives velocity and acceleration functions, computes user-vs-coach difference functions, and sends the summarized differences to an OpenAI-compatible model using an API key entered by the user in the UI.

## Current Structure

```text
tabletennis/
  docs/
    architecture.md
  scripts/
    check_environment.ps1
    setup_windows.ps1
    run_app.ps1
  src/
    tabletennis_analyzer/
      app.py
      core/
        comparison.py
        kinematics.py
        pose.py
        export.py
      llm/
        openai_client.py
      ui/
        main_window.py
  tests/
    test_kinematics.py
  pyproject.toml
  requirements.txt
```

## Development Environment

Recommended:

- Windows 10/11
- Python 3.11.x
- FFmpeg available in `PATH` for robust video handling
- A virtual environment in `.venv`

This machine currently does not expose `python`, `py`, `winget`, or `ffmpeg` in `PATH`, so install Python 3.11 and FFmpeg first if they are not already installed.

After Python is available:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_app.ps1
```

Analyze one MP4 without opening the UI:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\analyze_video.ps1 -Video path\to\stroke.mp4 -Label user_stroke
```

Analyze a user MP4 and a coach/reference MP4 together:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\analyze_user_coach.ps1 -UserVideo path\to\user.mp4 -CoachVideo path\to\coach.mp4
```

Analyze user-vs-coach velocity and acceleration differences, then call GPT when an API key is provided:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\analyze_differences.ps1 -UserVideo path\to\user.mp4 -CoachVideo path\to\coach.mp4 -ApiKey "<your-api-key>" -Model gpt-5.5
```

This exports:

- `*_positions.csv`: landmark position functions over `time_s`
- `*_kinematics.csv`: smoothed position, relative velocity, relative acceleration, speed, and acceleration magnitude
- `*_metadata.json`: video and analysis metadata
- `joints\*_kinematics.csv`: one CSV per body joint, with that joint's position, velocity, and acceleration functions
- `velocity_acceleration_difference_functions.csv`: time-related speed and acceleration difference functions
- `difference_joints\*_difference_functions.csv`: one difference-function CSV per joint
- `velocity_acceleration_difference_summary.json`: compact summary sent to GPT
- `gpt_motion_analysis.txt`: GPT analysis, only when an API key is supplied

Note: MediaPipe coordinates are normalized/relative image-space values. The exported velocity and acceleration values are relative motion quantities, not calibrated real-world m/s or m/s^2.

If you only want to check prerequisites:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check_environment.ps1
```

Build the Windows app:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_exe.ps1 -Clean
```

The generated app is:

```text
dist\TableTennisAnalyzer\TableTennisAnalyzer.exe
```

## Security Note

The OpenAI API key is never stored in this repository. The UI asks the user to enter it at runtime and passes it only to the request client for that session.

## First Milestone

1. Load two videos.
2. Extract pose landmarks with MediaPipe.
3. Normalize and align landmark time series.
4. Export position, velocity, acceleration, and difference CSV files.
5. Ask the user-entered model to analyze the difference summary.
