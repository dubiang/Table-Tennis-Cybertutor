# Architecture

## Product Goal

The user uploads two table-tennis videos:

- User motion video
- Coach/reference motion video

The software extracts human pose landmarks from both videos, represents each important joint as a time-dependent position function, calculates velocity and acceleration functions, computes user-vs-coach difference functions, and asks a large model to explain the likely technical differences.

## Module Boundaries

### UI Layer

`tabletennis_analyzer.ui.main_window`

- Lets the user select two videos.
- Lets the user enter an API key and model name at runtime.
- Starts analysis and displays progress/results.
- Does not contain computer vision or LLM business logic.

### Pose Extraction

`tabletennis_analyzer.core.pose`

- Reads video frames with OpenCV.
- Runs MediaPipe Pose.
- Emits a tidy table:
  - `time_s`
  - `frame_index`
  - `landmark`
  - `x`
  - `y`
  - `z`
  - `visibility`

### Kinematics

`tabletennis_analyzer.core.kinematics`

- Converts landmark positions into smoothed time series.
- Calculates velocity and acceleration with numerical derivatives.
- Keeps calculations per landmark and coordinate.

### Comparison

`tabletennis_analyzer.core.comparison`

- Aligns the user and coach time axes.
- Computes position, velocity, and acceleration differences.
- Produces compact metrics for the LLM prompt.

The first implementation uses shared normalized time interpolation. A later milestone can add Dynamic Time Warping for better phase alignment between strokes.

### Export

`tabletennis_analyzer.core.export`

- Writes CSV and JSON analysis artifacts.
- Keeps raw computed data separate from the model explanation.

### LLM Analysis

`tabletennis_analyzer.llm.openai_client`

- Receives the API key from the UI at runtime.
- Sends a concise difference summary to the selected model.
- Does not persist API keys.

## Data Flow

```text
User video   -> pose extraction -> kinematics -> comparison -> exports
Coach video  -> pose extraction -> kinematics -> comparison -> exports
                                                 comparison summary -> LLM -> explanation
```

## Important Joints For Table Tennis

Initial joint set:

- Nose
- Left/right shoulder
- Left/right elbow
- Left/right wrist
- Left/right hip
- Left/right knee
- Left/right ankle

Priority analysis areas:

- Hitting arm wrist speed and acceleration
- Elbow angle timing
- Shoulder rotation and stability
- Hip and knee weight transfer
- Body center movement
- Stroke rhythm compared with coach reference

## Environment Plan

Use Python because the pose-estimation and scientific-computing ecosystem is mature on Windows:

- PySide6: Windows desktop UI
- OpenCV: video decoding and frame access
- MediaPipe: human pose landmarks
- NumPy/Pandas/SciPy: time-series math
- OpenAI SDK: model call using runtime API key
- PyInstaller later: packaging as `.exe`

