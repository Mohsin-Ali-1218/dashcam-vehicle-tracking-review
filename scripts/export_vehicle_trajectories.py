"""
export_vehicle_trajectories.py — Export vehicle tracks JSONL to image-space trajectory CSV.

Converts frame-by-frame tracking output (tracks.jsonl) into an inspectable
CSV representation of observed vehicle detections in image coordinates.

Scope & Constraints:
- Pure Python standard library only (no external dependencies).
- Strictly image-space bounding-box observations; does NOT compute or include
  timestamps, metric distances, real-world speeds, GPS positions, or conflict scores.
- Bounding-box center and bottom-center pixel coordinates describe box geometry in
  the image plane, not physical road-contact points or ground locations.
- Preserves natural missed-frame gaps; does NOT interpolate, smooth, or invent boxes.
- Validates input integrity BEFORE writing any output:
  - Rejects existing destination files (refuses overwrite).
  - Validates strictly consecutive frame indices.
  - Validates finite numeric bounding box coordinates with positive dimensions.
  - Validates that no duplicate track ID exists within any single frame.

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

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CSV_FIELDNAMES = [
    "frame_index",
    "track_id",
    "class_name",
    "confidence",
    "x",
    "y",
    "width",
    "height",
    "center_x",
    "center_y",
    "bottom_center_x",
    "bottom_center_y",
]


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export vehicle tracks JSONL into an inspectable image-space trajectory CSV.",
        epilog=(
            "Note: Pixel points describe bounding boxes in 2D image coordinates, "
            "not measured road-contact positions or real-world metric coordinates."
        ),
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
        help="Path to output trajectory CSV file (required; refuses existing file).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core Export & Validation Logic
# ---------------------------------------------------------------------------

def export_trajectories(tracks_path: Path, output_path: Path) -> int:
    """
    Validate and export tracks.jsonl into an image-space trajectory CSV.
    Returns 0 on success, non-zero on error.
    """
    # 1. Enforce output file refusal: must not exist prior to execution
    if output_path.exists():
        print(f"ERROR: Output file already exists: {output_path}", file=sys.stderr)
        print("Refusing to overwrite existing output file.", file=sys.stderr)
        return 1

    # 2. Verify input tracks file exists
    if not tracks_path.is_file():
        print(f"ERROR: Input tracks file not found: {tracks_path}", file=sys.stderr)
        return 1

    print(f"Input tracks file   : {tracks_path.resolve()}")
    print(f"Output CSV file    : {output_path.resolve()}")
    print()

    # In-memory buffer to accumulate validated rows before writing
    validated_rows: list[list[Any]] = []
    unique_track_ids: set[int] = set()

    expected_frame_idx: int | None = None
    total_frames_count = 0

    # 3. Read and validate all records before creating or writing to output
    with tracks_path.open("r", encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue

            # Parse JSON line
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"ERROR: Line {line_num} in '{tracks_path}' is invalid JSON: {exc}",
                    file=sys.stderr,
                )
                return 2

            if not isinstance(record, dict) or "frame_index" not in record or "tracks" not in record:
                print(
                    f"ERROR: Line {line_num}: record missing required 'frame_index' or 'tracks' field.",
                    file=sys.stderr,
                )
                return 2

            frame_idx = record["frame_index"]
            if not isinstance(frame_idx, int) or frame_idx < 0:
                print(
                    f"ERROR: Line {line_num}: 'frame_index' must be a non-negative integer; got {frame_idx}.",
                    file=sys.stderr,
                )
                return 2

            # Validate strictly consecutive frame sequence
            if expected_frame_idx is None:
                expected_frame_idx = frame_idx
            elif frame_idx != expected_frame_idx:
                print(
                    f"ERROR: Line {line_num}: non-consecutive frame_index; "
                    f"expected {expected_frame_idx}, got {frame_idx}.",
                    file=sys.stderr,
                )
                return 2

            expected_frame_idx += 1
            total_frames_count += 1

            tracks_list = record["tracks"]
            if not isinstance(tracks_list, list):
                print(
                    f"ERROR: Line {line_num} (frame {frame_idx}): 'tracks' must be a list.",
                    file=sys.stderr,
                )
                return 2

            # Validate no duplicate track IDs within this frame
            seen_frame_track_ids: set[int] = set()

            for t_idx, item in enumerate(tracks_list):
                if not isinstance(item, dict):
                    print(
                        f"ERROR: Line {line_num} (frame {frame_idx}), item {t_idx}: track entry must be a dict.",
                        file=sys.stderr,
                    )
                    return 2

                track_id = item.get("track_id")
                if not isinstance(track_id, int):
                    print(
                        f"ERROR: Line {line_num} (frame {frame_idx}), item {t_idx}: 'track_id' must be an integer; got {track_id}.",
                        file=sys.stderr,
                    )
                    return 2

                if track_id in seen_frame_track_ids:
                    print(
                        f"ERROR: Line {line_num} (frame {frame_idx}): duplicate track_id {track_id} within the same frame.",
                        file=sys.stderr,
                    )
                    return 2
                seen_frame_track_ids.add(track_id)
                unique_track_ids.add(track_id)

                class_name = item.get("class_name")
                if not isinstance(class_name, str) or not class_name:
                    print(
                        f"ERROR: Line {line_num} (frame {frame_idx}), track {track_id}: 'class_name' must be a non-empty string.",
                        file=sys.stderr,
                    )
                    return 2

                conf = item.get("confidence")
                if not (isinstance(conf, (int, float)) and math.isfinite(conf) and 0.0 <= conf <= 1.0):
                    print(
                        f"ERROR: Line {line_num} (frame {frame_idx}), track {track_id}: "
                        f"'confidence' must be a finite float in [0.0, 1.0]; got {conf}.",
                        file=sys.stderr,
                    )
                    return 2

                box = item.get("box")
                if not isinstance(box, list) or len(box) != 4:
                    print(
                        f"ERROR: Line {line_num} (frame {frame_idx}), track {track_id}: "
                        f"'box' must be a list of 4 numbers [x, y, w, h]; got {box}.",
                        file=sys.stderr,
                    )
                    return 2

                if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box):
                    print(
                        f"ERROR: Line {line_num} (frame {frame_idx}), track {track_id}: "
                        f"box contains non-finite or non-numeric values: {box}.",
                        file=sys.stderr,
                    )
                    return 2

                x = float(box[0])
                y = float(box[1])
                width = float(box[2])
                height = float(box[3])

                if width <= 0 or height <= 0:
                    print(
                        f"ERROR: Line {line_num} (frame {frame_idx}), track {track_id}: "
                        f"box dimensions must be strictly positive; got width={width}, height={height}.",
                        file=sys.stderr,
                    )
                    return 2

                # Derived image-space pixel coordinates (rounded to 4 decimal places to prevent float representation artifacts)
                center_x = round(x + width / 2.0, 4)
                center_y = round(y + height / 2.0, 4)
                bottom_center_x = round(x + width / 2.0, 4)
                bottom_center_y = round(y + height, 4)

                validated_rows.append([
                    frame_idx,
                    track_id,
                    class_name,
                    conf,
                    x,
                    y,
                    width,
                    height,
                    center_x,
                    center_y,
                    bottom_center_x,
                    bottom_center_y,
                ])

    if total_frames_count == 0:
        print(f"ERROR: Input tracks file '{tracks_path}' contained no frame records.", file=sys.stderr)
        return 2

    print("✓ All validation checks passed:")
    print(f"  Processed frame records  : {total_frames_count:,}")
    print(f"  Observed detections      : {len(validated_rows):,}")
    print(f"  Unique track IDs         : {len(unique_track_ids):,}")
    print()

    # 4. Write output CSV only after all validations pass
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with output_path.open("x", newline="", encoding="utf-8") as out_fh:
            writer = csv.writer(out_fh)
            writer.writerow(CSV_FIELDNAMES)
            writer.writerows(validated_rows)
    except FileExistsError:
        print(f"ERROR: Destination output file already exists: {output_path}", file=sys.stderr)
        print("Refusing to overwrite existing destination file.", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"ERROR: Failed writing output CSV '{output_path}': {exc}", file=sys.stderr)
        return 3

    print(f"✓ Saved image-space trajectory CSV: {output_path} ({len(validated_rows):,} rows)")
    return 0


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    return export_trajectories(
        tracks_path=args.tracks,
        output_path=args.output,
    )


if __name__ == "__main__":
    sys.exit(main())
