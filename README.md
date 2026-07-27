# Pickleball Court-Coverage Heatmap

A computer vision pipeline that turns overhead match footage into a movement heatmap
showing where players actually spend their time on court.

![Heatmap](heatmap.gif)

## Pipeline

```
video → frame sampling → YOLO11x person detection → position accumulation
      → Gaussian smoothing → colour mapping → alpha-blended overlay → H.264 re-encode
```

Each detected player contributes a filled circle at their centre point to a float
accumulator the size of the frame. Repeated positions stack up and turn hot; untouched
areas stay at zero. The accumulator is blurred, normalised, mapped through a JET colour
scale, and blended over the original frame with alpha proportional to heat — so the
overlay appears only where players have been, not across the whole image.

## Design decisions

**Building the heatmap by hand instead of using the library's.** Ultralytics ships a
`solutions.Heatmap` helper, and it was the obvious starting point. It doesn't accept an
`imgsz` argument, and diagnostics showed `imgsz=1920` was required to detect players at
all from an overhead angle, where each person occupies a small fraction of the frame.
Rather than accept worse detection to keep the convenience, I called YOLO directly and
built the accumulation, smoothing, and blending in NumPy and OpenCV.

**Frame sampling, with the assumption stated.** Detection at 1920px on every frame is
the expensive part of the pipeline. Source footage is 23 FPS and the pipeline processes
one frame in three, for an effective 7 FPS. The justification is that a player moves
very little in 0.13 seconds, and a heatmap is a cumulative structure — sampling changes
the density of contributions, not their distribution. The output is visually
indistinguishable from full-rate processing at roughly a third of the compute.

**yolo11x rather than a smaller variant.** The largest model in the family, chosen
because overhead framing makes people small and easily missed. This is batch processing,
not live analysis, so the latency cost is acceptable — the trade would go the other way
for a real-time application.

**Re-encoding with FFmpeg.** OpenCV writes with the `mp4v` codec, which most browsers
(including Colab's player) won't display. The final step re-encodes to H.264 so the
output is viewable anywhere rather than only in a desktop video player.

## Known limitations

- **No player tracking.** Detections are aggregated across all players, so the heatmap
  shows combined court coverage rather than per-player patterns. Adding a tracker
  (ByteTrack, DeepSORT) would give each player their own map.
- **Image-space, not court-space.** Positions are in pixel coordinates, so the result is
  tied to the camera angle. A homography onto court dimensions would make coverage
  measurable in metres and comparable across recordings.
- **A player's centre point is not their footprint.** Heat is placed at the centre of the
  bounding box, which sits at torso height rather than at the feet where the player
  actually stands.
- Fixed detection threshold (`conf=0.30`) and fixed circle radius, both tuned by eye for
  this footage rather than derived.

## Running it

```bash
pip install -r requirements.txt
```

Open the notebook in Colab, upload a video when prompted, and run the cells in order. A
GPU runtime is strongly recommended — yolo11x at 1920px is slow on CPU.

## Stack

Python · Ultralytics YOLO11x · OpenCV · NumPy · FFmpeg
