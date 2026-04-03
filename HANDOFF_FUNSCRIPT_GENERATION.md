# Handoff: Funscript Generation Algorithm Improvement

## Context

Scripture is a semi-automated funscript generator for 2D video. The GUI (scene splitting, axis annotation, project persistence) is working well. The core problem is that **the funscript generation algorithm produces far too few actions** — 69 actions for a 25-minute video, when hundreds to thousands are needed.

## Current Architecture

```
scripture/
├── motion_tracker.py   # Optical flow tracking along user-defined axis
├── stroke_extract.py   # Peak/valley detection on the position signal
├── funscript.py        # JSON output in funscript format
├── scene.py            # Scene dataclass + split logic
├── gui.py              # PyQt6 GUI (not your concern)
└── project.py          # Save/load
```

## What the User Wants

A funscript maps timestamps to positions (0–100). The video shows repeated stroking motions. The user defines a **tip** and **base** point on the anatomy. A **contact point** (where a hand or mouth touches) moves between tip and base. The `pos` value should be:
- 100 when contact is at the tip
- 0 when contact is at the base
- Linear interpolation in between

The funscript should capture **stroke turnaround points** (peaks and valleys), not every frame. Typical good funscripts have one action per turnaround — when the hand/mouth changes direction. For fast repetitive motion, that might be every 300-500ms. For slow scenes, less frequent.

## What Currently Exists

### motion_tracker.py
- Uses **Farneback dense optical flow** between consecutive frames
- Projects the flow field onto the user-defined tip→base axis direction
- Averages the projected motion within an ROI (rectangle around the axis)
- Accumulates displacement over time to get a position signal

### stroke_extract.py
- Applies Savitzky-Golay smoothing
- Uses `scipy.signal.find_peaks` to find peaks and valleys
- Outputs only the extrema as funscript actions

## Why It Fails

1. **No anatomical understanding.** The system measures bulk pixel motion in the ROI. It has no concept of hands, mouths, or contact points. Fast background motion, camera shake, and irrelevant movement all contribute noise that drowns the stroke signal.

2. **Global normalization.** The position signal is normalized 0–1 over the entire scene, so drift and noise compress the actual stroke range.

3. **Over-smoothing + under-detection.** The peak detection finds too few peaks because the signal-to-noise ratio is too low after averaging motion across a large ROI.

## What Needs to Change

The user originally described a system that would:
1. Identify the anchor from various angles
2. Identify tip and base locations (user provides these via the GUI — already done)
3. **Identify the contact point** — the furthest point toward the base currently being touched by a hand or mouth
4. Project the contact point onto the tip→base axis to compute `pos`

This is fundamentally a **computer vision / object detection problem**, not a motion statistics problem. Approaches to consider:

### Option A: Per-Frame Contact Detection
Use a vision model (or classical CV) to detect the contact point per frame. Could involve:
- Skin segmentation to find the hand
- Edge/contour detection along the axis
- Tracking the "occlusion boundary" where the hand covers the redacted

### Option B: Improved Motion Tracking
Keep optical flow but dramatically improve signal extraction:
- Narrower ROI (tight around the redacted)
- Track specific features near the contact point rather than bulk motion
- Use Lucas-Kanade sparse optical flow on keypoints along the axis
- Better drift correction

### Option C: Hybrid
- Use optical flow for temporal coherence
- Sample key frames for per-frame analysis to anchor the signal
- Use a vision LLM (Claude Vision API) on sampled frames to validate/calibrate positions

## Test Data

- `first_attempt.funscript` in the project root contains the 69-action output from the first run
- The user has a `.scripture` project file in `sessions/` with scene splits and axis annotations
- Video is ~25 minutes, ~44,520 frames at 30fps

## Constraints

- Processing time matters — the first run took "a very long time" (likely hours). The user wants a tighter feedback loop.
- The user has an OSR2 device that needs minimum motor speed — too few actions means the device barely moves.
- Good funscripts capture stroke turnaround points minimally. Not every frame, but every direction change.
- The GUI already handles scene splitting, axis definition, and per-frame progress reporting. The algorithm just needs to produce better `actions` lists.

## Files to Modify

- `scripture/motion_tracker.py` — the core tracking algorithm
- `scripture/stroke_extract.py` — peak/valley extraction
- Possibly new modules for contact detection

## Interface Contract

The GUI calls:
```python
result = track_motion(video_path, axis, start_frame, end_frame, on_frame=callback)
actions = extract_strokes(result.positions, result.timestamps_ms, fps=fps)
```

`actions` must be a `list[dict]` where each dict has `{"at": int_ms, "pos": int_0_to_100}`.

The `on_frame` callback is called with each frame index for progress reporting.
