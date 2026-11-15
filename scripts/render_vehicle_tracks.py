"""
render_vehicle_tracks.py — Render vehicle tracking visualizations to MP4 video.

Reads consecutive frames from an input video alongside matching records from
a tracks.jsonl file (as produced by track_vehicle_detections.py), renders each
observed bounding box with a consistent colour per track ID, and writes an
annotated MP4 video file.

Visual annotations:
- Bounding box with consistent deterministic colour per track ID.
- Label badge: "ID {track_id} | {class_name} {confidence:.2f}".
- Optional image-space trails (--trail-length): drawn beneath boxes/labels connecting
  consecutive observed box centers for active tracks without gap crossing.
- Status banner: "Frame {frame_index:04d} | Temporary IDs; capture timing unverified."
  (extended with "Image-space trails; not road trajectories." when trails are enabled).

Behaviour & Guards:
- Preserves input video frame dimensions and container FPS; rejects invalid/non-finite FPS or dimensions.
- Refuses to overwrite an existing output file.
- Validates consecutive matching frame indices between video and tracks JSONL.
- Refuses missing video frames if EOF occurs before tracks JSONL is exhausted.
- Stops processing immediately after the final track record, leaving the remaining video unprocessed.
- Releases all OpenCV and file resources on success and error.

Licence notice for code:
------------------------
Copyright (c) 2026 Mohsin Ali. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Distinct Consistent Colour Palette
# ---------------------------------------------------------------------------

# 24 vibrant, distinct BGR colours for consistent track ID visualization
TRACK_PALETTE: list[tuple[int, int, int]] = [
    (0, 215, 255),   # amber / gold
    (255, 128, 0),   # azure
    (0, 255, 128),   # spring green
    (0, 165, 255),   # orange
    (255, 0, 128),   # deep pink
    (128, 0, 255),   # violet / purple
    (0, 255, 255),   # yellow
    (255, 255, 0),   # cyan
    (50, 205, 50),   # lime green
    (238, 130, 238), # light violet
    (0, 69, 255),    # red-orange
    (205, 90, 106),  # slate blue
    (30, 105, 210),  # chocolate
    (180, 105, 255), # hot pink
    (60, 20, 220),   # crimson
    (255, 191, 0),   # deep sky blue
    (144, 238, 144), # light green
    (75, 0, 130),    # indigo
    (240, 230, 140), # khaki
    (0, 128, 128),   # teal
    (218, 112, 214), # orchid
    (0, 140, 255),   # dark orange
    (127, 255, 212), # aquamarine
    (220, 20, 60),   # crimson red
]


def get_track_color(track_id: int) -> tuple[int, int, int]:
    """Return a consistent, distinct BGR colour for a given track ID."""
    idx = (abs(track_id) - 1) % len(TRACK_PALETTE)
    return TRACK_PALETTE[idx]


# ---------------------------------------------------------------------------
# Visual Annotation Helpers
# ---------------------------------------------------------------------------

def add_frame_banner(
    frame,
    frame_index: int,
    caveat_text: str = "Temporary IDs; capture timing unverified.",
):
    """Overlay a compact status banner on the top-left of the frame."""
    text = f"Frame {frame_index:04d} | {caveat_text}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    pad_x, pad_y = 8, 6
    x1, y1 = 10, 10
    x2 = x1 + tw + 2 * pad_x
    y2 = y1 + th + 2 * pad_y + baseline

    # Opaque dark background rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (25, 25, 25), -1)
    # White text with anti-aliasing
    text_x = x1 + pad_x
    text_y = y1 + pad_y + th
    cv2.putText(
        frame,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (230, 230, 230),
        thickness,
        cv2.LINE_AA,
    )
    return frame


def annotate_tracks(
    frame,
    tracks: list[dict],
    trails: dict[int, list[tuple[int, int]]] | None = None,
):
    """Draw bounding boxes, optional trails, and ID / class / confidence labels on a copy of the frame."""
    annotated = frame.copy()
    h_frame, w_frame = annotated.shape[:2]

    # Draw trails first (beneath boxes/labels) for currently observed IDs only
    if trails:
        for trk in tracks:
            track_id = int(trk["track_id"])
            pts = trails.get(track_id)
            if pts and len(pts) > 1:
                color = get_track_color(track_id)
                for i in range(len(pts) - 1):
                    cv2.line(annotated, pts[i], pts[i + 1], color, 2, cv2.LINE_AA)

    for trk in tracks:
        track_id = int(trk["track_id"])
        class_name = str(trk.get("class_name", "vehicle"))
        confidence = float(trk.get("confidence", 0.0))
        box = trk["box"]

        bx, by, bw, bh = [int(round(v)) for v in box]
        color = get_track_color(track_id)

        # Bounding box
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), color, 2)

        # Label badge
        label = f"ID {track_id} | {class_name} {confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        badge_y1 = max(0, by - th - 6)
        badge_y2 = by
        badge_x2 = min(w_frame, bx + tw + 6)
        cv2.rectangle(annotated, (bx, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.putText(
            annotated,
            label,
            (bx + 3, badge_y2 - baseline - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render vehicle tracking visualizations to MP4 video using input video "
            "and offline tracks.jsonl records."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input video file (required).",
    )
    parser.add_argument(
        "--tracks",
        type=Path,
        required=True,
        help="Path to input tracks JSONL file (required).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output annotated MP4 video file (required; refuses overwrite).",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=0,
        help=(
            "Maximum consecutive observed box-center points per ID to draw as trail "
            "(default: 0 = disabled)."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main Workflow
# ---------------------------------------------------------------------------

def run_render(args: argparse.Namespace) -> int:
    """Execute video tracks rendering pipeline."""
    # --- Guard 1: Refuse existing output file -------------------------------
    if args.output.exists():
        print(
            f"ERROR: Output file already exists: '{args.output}'. "
            "Refusing to overwrite.",
            file=sys.stderr,
        )
        return 1

    # --- Guard 1b: Validate trail length parameter -------------------------
    if args.trail_length < 0:
        print(
            f"ERROR: --trail-length must be a non-negative integer; got {args.trail_length}.",
            file=sys.stderr,
        )
        return 1

    # --- Guard 2: Input video must exist ------------------------------------
    if not args.input.exists():
        print(f"ERROR: Input video file not found: '{args.input}'.", file=sys.stderr)
        return 1

    if not args.input.is_file():
        print(f"ERROR: Input path exists but is not a regular file: '{args.input}'.", file=sys.stderr)
        return 1

    # --- Guard 3: Tracks JSONL must exist -----------------------------------
    if not args.tracks.exists():
        print(f"ERROR: Tracks file not found: '{args.tracks}'.", file=sys.stderr)
        return 1

    if not args.tracks.is_file():
        print(f"ERROR: Tracks path exists but is not a regular file: '{args.tracks}'.", file=sys.stderr)
        return 1

    # --- Open Input Video ---------------------------------------------------
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print(f"ERROR: OpenCV could not open video: '{args.input}'.", file=sys.stderr)
        return 2

    writer = None

    try:
        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        container_fps = cap.get(cv2.CAP_PROP_FPS)

        # --- Guard 4: Reject non-positive dimensions ------------------------
        if w_orig <= 0 or h_orig <= 0:
            print(
                f"ERROR: Input video reported non-positive frame dimensions: {w_orig}x{h_orig} px.",
                file=sys.stderr,
            )
            return 2

        # --- Guard 5: Reject invalid or non-finite container FPS ------------
        if not (math.isfinite(container_fps) and container_fps > 0):
            print(
                f"ERROR: Input video reported invalid or non-finite container FPS: {container_fps}.",
                file=sys.stderr,
            )
            return 2

        fps = float(container_fps)

        # Ensure output parent directory exists
        args.output.parent.mkdir(parents=True, exist_ok=True)

        # Initialize VideoWriter preserving dimensions and FPS
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output), fourcc, fps, (w_orig, h_orig))
        if not writer.isOpened():
            print(
                f"ERROR: OpenCV could not open VideoWriter for '{args.output}'. "
                "Check codec support.",
                file=sys.stderr,
            )
            return 2

        banner_caveat = (
            "Temporary IDs; capture timing unverified. Image-space trails; not road trajectories."
            if args.trail_length > 0
            else "Temporary IDs; capture timing unverified."
        )
        active_trails: dict[int, list[tuple[int, int]]] = {}

        print(f"Input video         : {args.input.resolve()}")
        print(f"Tracks JSONL        : {args.tracks.resolve()}")
        print(f"Output video        : {args.output.resolve()}")
        print(f"Resolution & FPS    : {w_orig}x{h_orig} px @ {fps:.2f} FPS")
        print(f"Trail length        : {args.trail_length}")
        print()

        expected_frame_idx: int | None = None
        rendered_frames_count = 0
        render_failed = False
        failure_message = ""

        with open(args.tracks, "r", encoding="utf-8") as tracks_fh:
            for line_num, raw_line in enumerate(tracks_fh, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                # 1. Parse JSON line
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    render_failed = True
                    failure_message = f"Line {line_num} in '{args.tracks}' is invalid JSON: {exc}"
                    break

                if not isinstance(record, dict) or "frame_index" not in record or "tracks" not in record:
                    render_failed = True
                    failure_message = (
                        f"Line {line_num} missing required 'frame_index' or 'tracks' field."
                    )
                    break

                track_frame_idx = record["frame_index"]
                if not isinstance(track_frame_idx, int) or track_frame_idx < 0:
                    render_failed = True
                    failure_message = (
                        f"Line {line_num}: 'frame_index' must be a non-negative integer; got {track_frame_idx}."
                    )
                    break

                # 2. Validate consecutive matching frame indices
                if expected_frame_idx is None:
                    expected_frame_idx = track_frame_idx
                    if expected_frame_idx > 0:
                        if not cap.set(cv2.CAP_PROP_POS_FRAMES, expected_frame_idx):
                            render_failed = True
                            failure_message = (
                                f"Could not seek video to starting frame index {expected_frame_idx}."
                            )
                            break
                elif track_frame_idx != expected_frame_idx:
                    render_failed = True
                    failure_message = (
                        f"Line {line_num}: Non-consecutive track frame index; "
                        f"expected {expected_frame_idx}, got {track_frame_idx}."
                    )
                    break

                # 3. Read corresponding video frame (refuse missing frames)
                ok, frame = cap.read()
                if not ok or frame is None:
                    render_failed = True
                    failure_message = (
                        f"Video ended prematurely at frame index {expected_frame_idx}; "
                        f"track record exists at line {line_num} (frame {track_frame_idx}), "
                        "but the video frame could not be read."
                    )
                    break

                tracks_list = record["tracks"]
                if not isinstance(tracks_list, list):
                    render_failed = True
                    failure_message = f"Line {line_num}: 'tracks' field must be a list."
                    break

                # 4. Update trail histories when enabled
                if args.trail_length > 0:
                    current_observed_ids = set()
                    for trk in tracks_list:
                        track_id = int(trk["track_id"])
                        current_observed_ids.add(track_id)
                        box = trk["box"]
                        bx, by, bw, bh = [float(v) for v in box]
                        cx = int(round(bx + bw / 2.0))
                        cy = int(round(by + bh / 2.0))
                        center_pt = (cx, cy)

                        if track_id in active_trails:
                            active_trails[track_id].append(center_pt)
                            if len(active_trails[track_id]) > args.trail_length:
                                active_trails[track_id] = active_trails[track_id][-args.trail_length:]
                        else:
                            active_trails[track_id] = [center_pt]

                    # Clear an ID's history whenever it is absent from a frame
                    for tid in list(active_trails.keys()):
                        if tid not in current_observed_ids:
                            del active_trails[tid]

                # 5. Annotate tracks and add frame banner
                annotated = annotate_tracks(
                    frame,
                    tracks_list,
                    trails=active_trails if args.trail_length > 0 else None,
                )
                annotated = add_frame_banner(annotated, track_frame_idx, caveat_text=banner_caveat)

                # 6. Write frame to output video
                writer.write(annotated)

                rendered_frames_count += 1
                expected_frame_idx += 1

                # Progress reporting every 30 frames
                if rendered_frames_count % 30 == 0:
                    print(f"Rendered frame {rendered_frames_count:4d} …")

        # --- Handle failures ------------------------------------------------
        if render_failed:
            print(f"ERROR: {failure_message}", file=sys.stderr)
            print(
                f"Partial output preserved: '{args.output.resolve()}' "
                f"({rendered_frames_count} frames written).",
                file=sys.stderr,
            )
            return 2

        if rendered_frames_count == 0:
            print(
                f"ERROR: Zero track records were rendered from '{args.tracks}'.",
                file=sys.stderr,
            )
            return 2

        # --- Completion Summary ---------------------------------------------
        print()
        print("=== Vehicle Tracks Rendering Complete ===")
        print(f"Rendered frames     : {rendered_frames_count}")
        print(f"Output video        : {args.output.resolve()}")

    finally:
        if writer is not None:
            writer.release()
        cap.release()

    return 0


def main() -> None:
    args = parse_args()
    sys.exit(run_render(args))


if __name__ == "__main__":
    main()
