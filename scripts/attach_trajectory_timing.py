"""
attach_trajectory_timing.py — Attach validated daylight sensor timing to image-space trajectories.

Merges retrieved daylight frame timing arrays (frame_times) with image-space
trajectory observations (trajectories.csv) based on the reviewed 1:1 frame-alignment
evidence (offset 0 uniquely minimal MAD within a +/-2-frame search window).

Scope & Constraints:
- Pure Python standard library and installed NumPy only.
- Reads existing trajectory CSV without modifying original columns or values.
- Loads frame_times with allow_pickle=False and enforces:
  - Exact shape (1200,) and count 1200.
  - Real numeric dtype (rejects object, string, boolean, and complex).
  - All finite values (no NaN or Inf).
  - Strictly increasing timestamps (strictly monotonic diffs > 0).
- Validates all input data in memory BEFORE creating output:
  - Validates integer frame indices bounded to [0, 1199].
  - Rejects duplicate (frame_index, track_id) rows.
  - Preserves natural observation gaps without interpolation or extrapolation.
- Appends three new columns:
  - source_frame_time_boot_s: frame_times[frame_index]
  - elapsed_from_source_start_s: frame_times[frame_index] - frame_times[0]
  - alignment_basis: 'same_index_supported_by_local_image_similarity'
- Creates output exclusively using mode "x" (fails if destination already exists).
- Prints summary: row count, unique ID count, observed frame range, elapsed time range.
- Important Caveats:
  1. Boot-relative timestamps are elapsed seconds since device boot, NOT UTC wall-clock time.
  2. Frame-to-timestamp mapping relies on local image-similarity evidence (MAD offset 0 across +/-2 frames),
     not embedded container timestamps or source telemetry packet headers.
  3. Camera pose / ego-vehicle motion does NOT measure or locate tracked-vehicle positions.
  4. Does NOT calculate metric speeds, 3D world coordinates, or traffic conflict metrics.

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
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_FRAME_COUNT = 1200
ALIGNMENT_BASIS_VALUE = "same_index_supported_by_local_image_similarity"

NEW_TIMING_COLUMNS = [
    "source_frame_time_boot_s",
    "elapsed_from_source_start_s",
    "alignment_basis",
]


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach validated daylight frame timing arrays to 2D vehicle trajectory CSV.\n\n"
            "Caveats:\n"
            "- Boot-relative timestamps are elapsed seconds since logger boot, NOT UTC wall-clock time.\n"
            "- Mapping relies on local image-similarity evidence within a +/-2-frame window.\n"
            "- Camera pose / ego-vehicle motion does not locate tracked vehicles.\n"
            "- Does not compute speeds, metric distances, or conflict scores."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--trajectories",
        type=Path,
        required=True,
        help="Path to input 2D image-space trajectories CSV (e.g. data/preview/daylight-tracking-1200-v1/trajectories.csv).",
    )
    parser.add_argument(
        "--frame-times",
        type=Path,
        required=True,
        help="Path to daylight frame_times array (e.g. data/raw/daylight-2018-07-29-segment-7-pose/frame_times).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for timed trajectory CSV. Must not already exist (exclusive creation).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Array Loading & Validation
# ---------------------------------------------------------------------------

def load_and_validate_frame_times(path: Path) -> np.ndarray:
    """Load frame_times.npy with allow_pickle=False and validate shape, dtype, and monotonicity.

    Raises:
        FileNotFoundError: If the array file does not exist.
        ValueError: If shape, dtype, finiteness, or monotonicity requirements are violated.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Frame times array not found: {path}")

    arr = np.load(path, allow_pickle=False)

    if not isinstance(arr, np.ndarray):
        raise ValueError(f"Loaded object from {path} is not a numpy.ndarray: {type(arr)}")

    # Reject object, string, boolean, and complex dtypes; require real numeric
    if arr.dtype.kind in ("O", "b", "c", "U", "S") or arr.dtype.kind not in ("f", "i", "u"):
        raise ValueError(
            f"Array '{path.name}' has non-real-numeric dtype '{arr.dtype}' (kind '{arr.dtype.kind}'). "
            "Object, string, boolean, and complex dtypes are rejected."
        )

    # Require exact shape (1200,)
    expected_shape = (EXPECTED_FRAME_COUNT,)
    if arr.shape != expected_shape:
        raise ValueError(
            f"Shape/count mismatch for '{path.name}': "
            f"actual shape {arr.shape} (count {arr.size}) != expected shape {expected_shape} (count {EXPECTED_FRAME_COUNT}). "
            "Refusing to reshape, truncate, or interpolate."
        )

    # Require all finite values
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite values (NaN/Inf) detected in '{path.name}'.")

    # Require strictly increasing timestamps
    diffs = np.diff(arr)
    if not np.all(diffs > 0):
        min_diff = float(np.min(diffs))
        raise ValueError(
            f"Array '{path.name}' is not strictly monotonic. Minimum interval between consecutive frames: {min_diff:.8f} s."
        )

    return arr


# ---------------------------------------------------------------------------
# Trajectory Loading, Validation & Augmentation
# ---------------------------------------------------------------------------

def augment_trajectories_with_timing(
    csv_path: Path,
    frame_times: np.ndarray,
) -> tuple[list[str], list[list[str]], dict[str, Any]]:
    """Read existing trajectory CSV, validate all rows, and append timing fields.

    Preserves original columns and values verbatim. Does not interpolate gaps.

    Returns:
        tuple of (output_header, augmented_rows, summary_stats)
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Trajectories CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            original_header = next(reader)
        except StopIteration:
            raise ValueError(f"Trajectories CSV is empty: {csv_path}")

        if "frame_index" not in original_header:
            raise ValueError(f"Trajectories CSV missing required 'frame_index' column: {csv_path}")
        if "track_id" not in original_header:
            raise ValueError(f"Trajectories CSV missing required 'track_id' column: {csv_path}")

        # Reject duplicate CSV header names
        if len(original_header) != len(set(original_header)):
            seen_headers: set[str] = set()
            duplicates: list[str] = []
            for col in original_header:
                if col in seen_headers and col not in duplicates:
                    duplicates.append(col)
                seen_headers.add(col)
            raise ValueError(
                f"Trajectories CSV contains duplicate header column names: {duplicates}"
            )

        # Reject any input already containing one of NEW_TIMING_COLUMNS
        conflicting_cols = [col for col in NEW_TIMING_COLUMNS if col in original_header]
        if conflicting_cols:
            raise ValueError(
                f"Trajectories CSV already contains timing column(s): {conflicting_cols}. "
                "Refusing to re-augment an existing timed CSV."
            )

        frame_idx_col = original_header.index("frame_index")
        track_id_col = original_header.index("track_id")

        augmented_rows: list[list[str]] = []
        seen_pairs: set[tuple[int, int]] = set()
        unique_track_ids: set[int] = set()

        observed_frame_indices: list[int] = []
        observed_elapsed_times: list[float] = []

        start_time_boot_s = float(frame_times[0])

        for line_num, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(original_header):
                raise ValueError(
                    f"Line {line_num}: Column count mismatch (expected {len(original_header)}, got {len(row)})."
                )

            frame_idx_str = row[frame_idx_col].strip()
            track_id_str = row[track_id_col].strip()

            try:
                frame_idx = int(frame_idx_str)
            except ValueError:
                raise ValueError(
                    f"Line {line_num}: 'frame_index' must be an integer; got '{frame_idx_str}'."
                )

            if str(frame_idx) != frame_idx_str:
                raise ValueError(
                    f"Line {line_num}: 'frame_index' contains non-integer representation: '{frame_idx_str}'."
                )

            if not (0 <= frame_idx < EXPECTED_FRAME_COUNT):
                raise ValueError(
                    f"Line {line_num}: 'frame_index' {frame_idx} out of valid bounds [0, {EXPECTED_FRAME_COUNT - 1}]."
                )

            try:
                track_id = int(track_id_str)
            except ValueError:
                raise ValueError(
                    f"Line {line_num}: 'track_id' must be an integer; got '{track_id_str}'."
                )

            if str(track_id) != track_id_str:
                raise ValueError(
                    f"Line {line_num}: 'track_id' contains non-integer representation: '{track_id_str}'."
                )

            if track_id <= 0:
                raise ValueError(
                    f"Line {line_num}: 'track_id' must be a positive integer; got {track_id}."
                )

            pair = (frame_idx, track_id)
            if pair in seen_pairs:
                raise ValueError(
                    f"Line {line_num}: Duplicate (frame_index, track_id) pair ({frame_idx}, {track_id}) detected."
                )
            seen_pairs.add(pair)
            unique_track_ids.add(track_id)

            # Compute timing fields
            source_frame_time = float(frame_times[frame_idx])
            elapsed_time = source_frame_time - start_time_boot_s

            observed_frame_indices.append(frame_idx)
            observed_elapsed_times.append(elapsed_time)

            # Format timing fields with 6 decimal places (microsecond resolution)
            source_time_str = f"{source_frame_time:.6f}"
            elapsed_time_str = f"{elapsed_time:.6f}"

            # Append new timing columns without modifying original values
            new_row = list(row) + [source_time_str, elapsed_time_str, ALIGNMENT_BASIS_VALUE]
            augmented_rows.append(new_row)

        if not augmented_rows:
            raise ValueError(f"Trajectories CSV contains no observation rows: {csv_path}")

    output_header = list(original_header) + NEW_TIMING_COLUMNS

    summary: dict[str, Any] = {
        "total_rows": len(augmented_rows),
        "unique_track_ids_count": len(unique_track_ids),
        "min_frame_index": min(observed_frame_indices),
        "max_frame_index": max(observed_frame_indices),
        "min_elapsed_s": min(observed_elapsed_times),
        "max_elapsed_s": max(observed_elapsed_times),
    }

    return output_header, augmented_rows, summary


# ---------------------------------------------------------------------------
# Output Writing
# ---------------------------------------------------------------------------

def write_timed_trajectories(
    output_path: Path,
    header: list[str],
    rows: list[list[str]],
) -> None:
    """Write augmented trajectory data to destination file using exclusive creation mode 'x'.

    Raises:
        FileExistsError: If destination already exists.
    """
    if output_path.exists():
        raise FileExistsError(
            f"Destination file already exists: {output_path}. Refusing to overwrite."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    print("================================================================================")
    print("Attach Trajectory Timing")
    print("================================================================================")
    print(f"Input trajectories CSV : {args.trajectories}")
    print(f"Frame times array      : {args.frame_times}")
    print(f"Destination output CSV : {args.output}")

    # Check output pre-existence before doing work
    if args.output.exists():
        print(f"ERROR: Destination already exists: {args.output}. Refusing overwrite.", file=sys.stderr)
        return 3

    # 1. Load and validate frame_times array
    print("\nLoading and validating frame_times array...")
    try:
        frame_times = load_and_validate_frame_times(args.frame_times)
    except Exception as e:
        print(f"ERROR loading frame_times: {e}", file=sys.stderr)
        return 4

    span_s = float(frame_times[-1] - frame_times[0])
    diffs = np.diff(frame_times)
    print(f"  Frame count          : {len(frame_times)} (shape: {frame_times.shape})")
    print(f"  Dtype                : {frame_times.dtype}")
    print(f"  All finite           : True")
    print(f"  Strictly monotonic   : True")
    print(f"  Total timespan       : {span_s:.6f} s")
    print(f"  Interval (min/med/max): {float(np.min(diffs)):.6f} / {float(np.median(diffs)):.6f} / {float(np.max(diffs)):.6f} s")

    # 2. Read, validate, and augment trajectory observations
    print("\nReading and validating trajectory observations...")
    try:
        header, rows, summary = augment_trajectories_with_timing(args.trajectories, frame_times)
    except Exception as e:
        print(f"ERROR validating trajectories: {e}", file=sys.stderr)
        return 5

    # 3. Write augmented trajectories with exclusive creation mode 'x'
    print(f"\nWriting {len(rows)} timed trajectory records to {args.output} (mode 'x')...")
    try:
        write_timed_trajectories(args.output, header, rows)
    except FileExistsError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"ERROR writing output file: {e}", file=sys.stderr)
        return 6

    # 4. Print validation summary & caveats
    print("\n================================================================================")
    print("Timing Attachment Summary")
    print("================================================================================")
    print(f"Total rows written       : {summary['total_rows']}")
    print(f"Unique track IDs         : {summary['unique_track_ids_count']}")
    if summary["min_frame_index"] is not None and summary["max_frame_index"] is not None:
        frame_span = summary["max_frame_index"] - summary["min_frame_index"] + 1
        print(
            f"Observed frame range     : {summary['min_frame_index']} .. {summary['max_frame_index']} "
            f"({frame_span} frames in span)"
        )
    else:
        print("Observed frame range     : None")

    if summary["min_elapsed_s"] is not None and summary["max_elapsed_s"] is not None:
        elapsed_span = summary["max_elapsed_s"] - summary["min_elapsed_s"]
        print(
            f"Elapsed time range (s)   : {summary['min_elapsed_s']:.6f} s .. {summary['max_elapsed_s']:.6f} s "
            f"(span: {elapsed_span:.6f} s)"
        )
    else:
        print("Elapsed time range (s)   : None")

    print(f"Alignment basis          : {ALIGNMENT_BASIS_VALUE}")

    print("\n--- Scope & Evidence Caveats ---")
    print("1. Boot-relative time : source_frame_time_boot_s values are elapsed seconds since logger")
    print("                        boot, NOT UTC wall-clock time.")
    print("2. Alignment evidence : Mapping assumes 1:1 same-index correspondence based on reviewed")
    print("                        local image-similarity evidence (mean MAD 0.3012, min margin 0.6753")
    print("                        over nearest candidate within +/-2 frames). Timestamps were not")
    print("                        embedded in the MP4 container or joined directly to track records.")
    print("3. Sensor attribution : Camera pose and frame timing describe ego-vehicle logger motion,")
    print("                        NOT tracked-vehicle road positions.")
    print("4. Analytical scope   : Pixel coordinates describe 2D image bounding boxes. Speeds, metric")
    print("                        distances, and traffic conflict scores are NOT computed.")
    print("================================================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
