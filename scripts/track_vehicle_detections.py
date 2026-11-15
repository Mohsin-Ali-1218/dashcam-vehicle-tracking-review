"""
track_vehicle_detections.py — Simple offline vehicle tracking baseline.

Consumes per-frame vehicle bounding box detections from an existing detections.jsonl
file (as produced by detect_vehicle_video.py) and tracks vehicles across consecutive
frames using frame-to-frame bounding-box IoU matching.

Matching characteristics:
- Matches candidate (track, detection) pairs greedily in descending IoU order.
- Enforces one detection per track and one track per detection.
- Matches across vehicle classes to tolerate detector label fluctuations (e.g. car/truck ambiguity).
- Unmatched detections are initialized as new tracks with incrementing integer IDs.
- Unmatched tracks are retained for up to 5 consecutive missed frames and retired on the 6th.
- Outputs only observed detections with assigned track IDs (no interpolation or invented boxes).

Limitations & Caveats:
- This baseline lacks motion prediction (such as Kalman filtering).
- This baseline lacks appearance re-identification (ReID) feature embedding.
- This baseline lacks camera-motion / ego-motion compensation.
- Consequently, track IDs can switch or fragment during occlusion, rapid camera motion,
  sharp turns, or brief vehicle overlap.

Licence notice for adapted code:
--------------------------------
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
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# IoU Computation
# ---------------------------------------------------------------------------

def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """
    Compute Intersection-over-Union (IoU) between two bounding boxes.

    Parameters
    ----------
    box_a, box_b : list[float]
        Bounding boxes formatted as [x, y, width, height].

    Returns
    -------
    float
        Intersection-over-Union score in [0.0, 1.0].
    """
    xa1, ya1, wa, ha = box_a
    xb1, yb1, wb, hb = box_b

    xa2 = xa1 + wa
    ya2 = ya1 + ha
    xb2 = xb1 + wb
    yb2 = yb1 + hb

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    area_a = wa * ha
    area_b = wb * hb
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0

    return float(inter_area / union_area)


# ---------------------------------------------------------------------------
# Track State
# ---------------------------------------------------------------------------

class Track:
    """Represents an active or historic vehicle track state."""

    def __init__(self, track_id: int, det: dict, frame_idx: int) -> None:
        self.track_id = track_id
        self.box = [float(v) for v in det["box"]]
        self.class_id = int(det["class_id"])
        self.class_name = str(det["class_name"])
        self.confidence = float(det["confidence"])
        self.missed_count = 0
        self.total_hits = 1
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.initial_class = str(det["class_name"])

    def update(self, det: dict, frame_idx: int) -> None:
        """Update track with a newly matched detection."""
        self.box = [float(v) for v in det["box"]]
        self.class_id = int(det["class_id"])
        self.class_name = str(det["class_name"])
        self.confidence = float(det["confidence"])
        self.missed_count = 0
        self.total_hits += 1
        self.last_frame = frame_idx


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline frame-to-frame vehicle tracking baseline using bounding-box "
            "IoU matching on an existing detections.jsonl log."
        ),
        epilog=(
            "Note: This baseline lacks motion prediction, visual re-identification, "
            "and camera-motion compensation; track IDs can switch."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input detections JSONL file (required).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to new output directory (required; refuses existing directory).",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.3,
        help="Minimum bounding-box IoU threshold for matching (default: 0.3).",
    )
    parser.add_argument(
        "--max-missed",
        type=int,
        default=5,
        help="Maximum missed frames before retiring a track (default: 5; removed on the 6th).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def run_tracking(args: argparse.Namespace) -> int:
    """Execute offline tracking workflow."""
    # --- Guard 1: Require new output directory and refuse existing one ------
    if args.output_dir.exists():
        print(
            f"ERROR: Output directory already exists: '{args.output_dir}'. "
            "A new, non-existent directory path is required.",
            file=sys.stderr,
        )
        return 1

    # --- Guard 2: Input detections file must exist --------------------------
    if not args.input.exists():
        print(f"ERROR: Input detections file not found: '{args.input}'.", file=sys.stderr)
        return 1

    if not args.input.is_file():
        print(f"ERROR: Input path exists but is not a regular file: '{args.input}'.", file=sys.stderr)
        return 1

    # --- Guard 3: Validate thresholds and arguments -------------------------
    if not (np.isfinite(args.iou_thresh) and 0.0 <= args.iou_thresh <= 1.0):
        print(
            f"ERROR: --iou-thresh must be a finite float in [0.0, 1.0]; got {args.iou_thresh}.",
            file=sys.stderr,
        )
        return 1

    if args.max_missed < 0:
        print(
            f"ERROR: --max-missed must be a non-negative integer; got {args.max_missed}.",
            file=sys.stderr,
        )
        return 1

    # --- Create output directory --------------------------------------------
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        print(f"ERROR: Failed to create output directory '{args.output_dir}': {exc}", file=sys.stderr)
        return 1

    tracks_out_path = args.output_dir / "tracks.jsonl"
    summary_out_path = args.output_dir / "summary.json"

    # Open output tracks file
    try:
        tracks_fh = open(tracks_out_path, "w", encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: Could not open '{tracks_out_path}' for writing: {exc}", file=sys.stderr)
        return 1

    # Tracking state
    active_tracks: list[Track] = []
    all_tracks: list[Track] = []
    next_track_id = 0
    expected_frame_idx: int | None = None
    processed_frames_count = 0
    tracking_failed = False
    failure_message = ""

    try:
        with open(args.input, "r", encoding="utf-8") as in_fh:
            for line_num, raw_line in enumerate(in_fh, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                # 1. Parse JSON line
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    tracking_failed = True
                    failure_message = f"Line {line_num} in '{args.input}' is invalid JSON: {exc}"
                    break

                if not isinstance(record, dict) or "frame_index" not in record or "detections" not in record:
                    tracking_failed = True
                    failure_message = (
                        f"Line {line_num} missing required 'frame_index' or 'detections' field."
                    )
                    break

                frame_idx = record["frame_index"]
                if not isinstance(frame_idx, int) or frame_idx < 0:
                    tracking_failed = True
                    failure_message = (
                        f"Line {line_num}: 'frame_index' must be a non-negative integer; got {frame_idx}."
                    )
                    break

                # 2. Validate consecutive frame indices
                if expected_frame_idx is None:
                    expected_frame_idx = frame_idx
                elif frame_idx != expected_frame_idx:
                    tracking_failed = True
                    failure_message = (
                        f"Line {line_num}: Non-consecutive frame index; "
                        f"expected {expected_frame_idx}, got {frame_idx}."
                    )
                    break
                expected_frame_idx = frame_idx + 1

                raw_dets = record["detections"]
                if not isinstance(raw_dets, list):
                    tracking_failed = True
                    failure_message = f"Line {line_num}: 'detections' must be a list."
                    break

                # 3. Validate detections and positive-area boxes
                current_detections: list[dict] = []
                det_validation_error = False
                for d_i, det in enumerate(raw_dets):
                    if not isinstance(det, dict):
                        tracking_failed = True
                        failure_message = f"Line {line_num}, detection {d_i}: not a dictionary."
                        det_validation_error = True
                        break

                    for field in ("class_id", "class_name", "confidence", "box"):
                        if field not in det:
                            tracking_failed = True
                            failure_message = f"Line {line_num}, detection {d_i}: missing field '{field}'."
                            det_validation_error = True
                            break
                    if det_validation_error:
                        break

                    conf = det["confidence"]
                    if not (isinstance(conf, (int, float)) and np.isfinite(conf) and 0.0 <= conf <= 1.0):
                        tracking_failed = True
                        failure_message = (
                            f"Line {line_num}, detection {d_i}: invalid confidence {conf}; "
                            "must be a finite float in [0.0, 1.0]."
                        )
                        det_validation_error = True
                        break

                    box = det["box"]
                    if (
                        not isinstance(box, list)
                        or len(box) != 4
                        or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in box)
                    ):
                        tracking_failed = True
                        failure_message = (
                            f"Line {line_num}, detection {d_i}: box must be 4 finite numbers [x, y, w, h]; got {box}."
                        )
                        det_validation_error = True
                        break

                    bw, bh = float(box[2]), float(box[3])
                    if bw <= 0 or bh <= 0:
                        tracking_failed = True
                        failure_message = (
                            f"Line {line_num}, detection {d_i}: box width and height must be positive; "
                            f"got width={bw}, height={bh}."
                        )
                        det_validation_error = True
                        break

                    current_detections.append(det)

                if det_validation_error:
                    break

                # 4. Greedy frame-to-frame IoU matching across classes
                candidates: list[tuple[int, int, float]] = []
                for t_idx, track in enumerate(active_tracks):
                    for d_idx, det in enumerate(current_detections):
                        iou = compute_iou(track.box, det["box"])
                        if iou >= args.iou_thresh:
                            candidates.append((t_idx, d_idx, iou))

                # Sort by IoU in descending order
                candidates.sort(key=lambda item: item[2], reverse=True)

                matched_track_indices: set[int] = set()
                matched_det_indices: set[int] = set()
                matched_pairs: list[tuple[int, int]] = []

                for t_idx, d_idx, _iou in candidates:
                    if t_idx not in matched_track_indices and d_idx not in matched_det_indices:
                        matched_track_indices.add(t_idx)
                        matched_det_indices.add(d_idx)
                        matched_pairs.append((t_idx, d_idx))

                det_to_track_id: dict[int, int] = {}

                # Update matched tracks
                for t_idx, d_idx in matched_pairs:
                    track = active_tracks[t_idx]
                    track.update(current_detections[d_idx], frame_idx)
                    det_to_track_id[d_idx] = track.track_id

                # Update unmatched tracks (retain up to max_missed, drop on max_missed + 1)
                surviving_active_tracks: list[Track] = []
                for t_idx, track in enumerate(active_tracks):
                    if t_idx not in matched_track_indices:
                        track.missed_count += 1
                        if track.missed_count <= args.max_missed:
                            surviving_active_tracks.append(track)
                    else:
                        surviving_active_tracks.append(track)

                # Initialize new tracks for unmatched detections
                for d_idx, det in enumerate(current_detections):
                    if d_idx not in matched_det_indices:
                        next_track_id += 1
                        new_track = Track(next_track_id, det, frame_idx)
                        all_tracks.append(new_track)
                        surviving_active_tracks.append(new_track)
                        det_to_track_id[d_idx] = new_track.track_id

                active_tracks = surviving_active_tracks

                # 5. Output observed detections with assigned track IDs (includes empty lists)
                frame_tracks = []
                for d_idx, det in enumerate(current_detections):
                    tid = det_to_track_id[d_idx]
                    frame_tracks.append({
                        "track_id": tid,
                        "class_id": det["class_id"],
                        "class_name": det["class_name"],
                        "confidence": det["confidence"],
                        "box": det["box"],
                    })

                out_record = {
                    "frame_index": frame_idx,
                    "tracks": frame_tracks,
                }
                try:
                    tracks_fh.write(json.dumps(out_record) + "\n")
                except OSError as exc:
                    tracking_failed = True
                    failure_message = f"Writing error on '{tracks_out_path}' at frame {frame_idx}: {exc}"
                    break

                processed_frames_count += 1

    finally:
        tracks_fh.close()

    # --- Failure handling ---------------------------------------------------
    if tracking_failed:
        print(f"ERROR: {failure_message}", file=sys.stderr)
        print(
            f"Partial outputs preserved in '{args.output_dir}': "
            f"'{tracks_out_path.name}' ({processed_frames_count} frames written).",
            file=sys.stderr,
        )
        print("Summary file will not be written due to failure.", file=sys.stderr)
        return 2

    if processed_frames_count == 0:
        print(f"ERROR: Zero frames processed from '{args.input}'.", file=sys.stderr)
        print(
            f"Partial outputs preserved in '{args.output_dir}': '{tracks_out_path.name}' (0 frames).",
            file=sys.stderr,
        )
        print("Summary file will not be written due to failure.", file=sys.stderr)
        return 2

    # --- Compute track summary statistics -----------------------------------
    track_lengths = [t.total_hits for t in all_tracks]
    class_dist: dict[str, int] = {}
    for t in all_tracks:
        class_dist[t.initial_class] = class_dist.get(t.initial_class, 0) + 1

    summary = {
        "input_file": str(args.input.resolve()),
        "output_file": str(tracks_out_path.resolve()),
        "settings": {
            "iou_thresh": args.iou_thresh,
            "max_missed_frames": args.max_missed,
            "cross_class_matching": True,
        },
        "processed_frames_count": processed_frames_count,
        "total_unique_tracks": len(all_tracks),
        "active_tracks_at_end": len(active_tracks),
        "track_length_frames": {
            "min": int(np.min(track_lengths)) if track_lengths else 0,
            "max": int(np.max(track_lengths)) if track_lengths else 0,
            "mean": round(float(np.mean(track_lengths)), 2) if track_lengths else 0.0,
            "median": round(float(np.median(track_lengths)), 2) if track_lengths else 0.0,
        },
        "class_distribution_at_creation": class_dist,
        "limitations": (
            "This baseline lacks motion prediction (Kalman filtering), visual "
            "re-identification (ReID), and camera-motion compensation; track IDs can switch."
        ),
    }

    try:
        with open(summary_out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except OSError as exc:
        print(f"ERROR: Failed to write summary JSON to '{summary_out_path}': {exc}", file=sys.stderr)
        return 1

    # --- Console Report -----------------------------------------------------
    print("=== Vehicle Tracking Summary ===")
    print(f"Processed frames     : {processed_frames_count}")
    print(f"Total unique tracks  : {len(all_tracks)}")
    print(f"Active tracks at end : {len(active_tracks)}")
    for name, count in sorted(class_dist.items()):
        print(f"  - Initial {name:10s}: {count}")
    if track_lengths:
        mean_len = np.mean(track_lengths)
        med_len = np.median(track_lengths)
        print(f"Track length frames  : min={np.min(track_lengths)}, max={np.max(track_lengths)}, mean={mean_len:.1f}, median={med_len:.1f}")
    print(f"Outputs written to   : {args.output_dir.resolve()}")
    print(f"  - Tracks JSONL     : {tracks_out_path.name}")
    print(f"  - Summary JSON     : {summary_out_path.name}")
    print("Limitations          : Lacks motion prediction, re-ID, and camera-motion compensation; IDs can switch.")

    return 0


def main() -> None:
    args = parse_args()
    sys.exit(run_tracking(args))


if __name__ == "__main__":
    main()
