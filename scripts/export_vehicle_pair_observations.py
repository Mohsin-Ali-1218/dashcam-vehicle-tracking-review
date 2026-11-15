"""
export_vehicle_pair_observations.py — Export same-frame image-space vehicle pair relationships.

Reads a timed vehicle trajectories CSV (trajectories-timed.csv) and exports pairwise
image-space bounding-box relationships for vehicles detected within the same frame.
Emits each unordered pair of different IDs once per frame, ordered by track ID.

Purpose:
- Export same-frame image-space geometric relationships for later manual review.
- Strictly NOT conflict classification, collision risk, or safety assessment.

Scope & Constraints:
- Pure Python standard library only (no external dependencies).
- Validates all input data in memory BEFORE creating or writing to output:
  - Input CSV must be non-empty with unique header column names.
  - Required columns must be present: frame_index, track_id, class_name, confidence,
    x, y, width, height, elapsed_from_source_start_s, alignment_basis.
  - Requires valid non-negative integer frame_index (>= 0).
  - Requires valid positive integer track_id (> 0).
  - Rejects duplicate (frame_index, track_id) rows across the input file.
  - Validates that confidence, coordinates, dimensions, and timestamps are finite numbers.
  - Validates strictly positive bounding box dimensions (width > 0, height > 0).
  - Enforces consistent timing (elapsed_from_source_start_s) and alignment_basis across
    all observations within each frame.
  - Requires alignment_basis to equal 'same_index_supported_by_local_image_similarity'
    (rejects other values before writing).
- Refuses overwrite using exclusive creation mode "x" (fails if output file already exists).
- Uses all input observations without confidence filtering; frames with fewer than two observations emit no pair rows.
- Pairing rules:
  - For each frame, emits each unordered pair of different track IDs once, ordered by ID
    (track_id_a < track_id_b).
  - No interpolation or cross-frame track pairing.
- Geometric definitions:
  - Axis gap (horizontal and vertical): 0.0 when 1D box intervals overlap or touch;
    otherwise positive separation distance.
  - IoU (Intersection over Union): computed from original bounding boxes.
  - Center distance: Euclidean distance in pixels between box geometric centers.

Important Caveats:
1. Observation only  : Emits image-space geometric observations for human review; does NOT
                       perform conflict classification, risk assessment, or safety scoring.
2. Temporary IDs     : Track IDs represent temporary tracker associations, NOT verified
                       unique physical vehicles.
3. 2D image space    : All measurements (gaps, IoU, center distances) are in 2D image pixel
                       coordinates and do not reflect 3D world geometry, roadway lanes, or
                       physical depth.
4. Non-goals         : Does NOT infer lanes, physical proximity, collisions, near-misses,
                       vehicle speeds, Time-To-Collision (TTC), or traffic safety.

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
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_ALIGNMENT_BASIS = "same_index_supported_by_local_image_similarity"

REQUIRED_INPUT_COLUMNS = [
    "frame_index",
    "track_id",
    "class_name",
    "confidence",
    "x",
    "y",
    "width",
    "height",
    "elapsed_from_source_start_s",
    "alignment_basis",
]

OUTPUT_FIELDNAMES = [
    "frame_index",
    "elapsed_from_source_start_s",
    "alignment_basis",
    "track_id_a",
    "track_id_b",
    "class_name_a",
    "class_name_b",
    "confidence_a",
    "confidence_b",
    "box_iou",
    "horizontal_box_gap_px",
    "vertical_box_gap_px",
    "center_distance_px",
]


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class Observation(NamedTuple):
    track_id: int
    class_name: str
    confidence_str: str
    confidence: float
    x: float
    y: float
    width: float
    height: float


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export same-frame image-space vehicle pair relationships from timed trajectory CSV.\n\n"
            "Caveats:\n"
            "- Exports 2D image-space geometric relationships for manual review, NOT conflict classification.\n"
            "- Track IDs are temporary tracking associations, NOT verified unique physical vehicles.\n"
            "- Does NOT infer lanes, physical proximity, collisions, speeds, TTC, or safety risk.\n"
            "- Uses all input observations without confidence filtering; frames with fewer than two observations emit no pair rows."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input timed trajectories CSV (e.g. data/preview/daylight-tracking-1200-v1/trajectories-timed.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for vehicle pair observations CSV. Must not already exist (mode 'x').",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data Loading & Validation
# ---------------------------------------------------------------------------

def load_and_validate_trajectories(
    csv_path: Path,
) -> tuple[dict[int, list[Observation]], dict[int, tuple[str, float, str]], int]:
    """Read and validate timed trajectory CSV.

    Validates:
    - Non-empty CSV with unique header column names.
    - Presence of all required columns.
    - Valid non-negative integer frame_index (>= 0).
    - Valid positive integer track_id (> 0).
    - No duplicate (frame_index, track_id) rows across the entire file.
    - Finite numerical values for confidence, coordinates, dimensions, and timestamps.
    - Strictly positive box dimensions (width > 0, height > 0).
    - Requires alignment_basis to equal 'same_index_supported_by_local_image_similarity'.
    - Consistent timing (elapsed_from_source_start_s) within each frame.

    Returns:
        tuple of (frame_observations, frame_metadata, total_observations)
        where:
        - frame_observations maps frame_index -> list of Observation
        - frame_metadata maps frame_index -> (elapsed_from_source_start_s_str, elapsed_float, alignment_basis)
        - total_observations is total valid observation rows
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input trajectories CSV not found: {csv_path}")

    frame_observations: dict[int, list[Observation]] = defaultdict(list)
    frame_metadata: dict[int, tuple[str, float, str]] = {}
    seen_frame_id_pairs: set[tuple[int, int]] = set()
    total_observations = 0

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"Input CSV is completely empty: {csv_path}")

        # Reject duplicate column names in header
        if len(header) != len(set(header)):
            seen_headers: set[str] = set()
            duplicates: list[str] = []
            for col in header:
                if col in seen_headers and col not in duplicates:
                    duplicates.append(col)
                seen_headers.add(col)
            raise ValueError(f"Input CSV contains duplicate header columns: {duplicates}")

        # Check required columns
        for req_col in REQUIRED_INPUT_COLUMNS:
            if req_col not in header:
                raise ValueError(f"Input CSV missing required column '{req_col}': {csv_path}")

        frame_idx_col = header.index("frame_index")
        track_id_col = header.index("track_id")
        class_name_col = header.index("class_name")
        conf_col = header.index("confidence")
        x_col = header.index("x")
        y_col = header.index("y")
        width_col = header.index("width")
        height_col = header.index("height")
        elapsed_col = header.index("elapsed_from_source_start_s")
        basis_col = header.index("alignment_basis")

        for line_num, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(
                    f"Line {line_num}: Column count mismatch (expected {len(header)}, got {len(row)})."
                )

            total_observations += 1

            # Validate frame_index
            frame_idx_str = row[frame_idx_col].strip()
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
            if frame_idx < 0:
                raise ValueError(
                    f"Line {line_num}: 'frame_index' must be non-negative; got {frame_idx}."
                )

            # Validate track_id
            track_id_str = row[track_id_col].strip()
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

            # Validate duplicate (frame_index, track_id)
            pair = (frame_idx, track_id)
            if pair in seen_frame_id_pairs:
                raise ValueError(
                    f"Line {line_num}: Duplicate (frame_index, track_id) pair ({frame_idx}, {track_id}) detected."
                )
            seen_frame_id_pairs.add(pair)

            # Validate class_name
            class_name = row[class_name_col].strip()
            if not class_name:
                raise ValueError(
                    f"Line {line_num}: 'class_name' must be a non-empty string."
                )

            # Validate confidence
            conf_str = row[conf_col].strip()
            try:
                conf = float(conf_str)
            except ValueError:
                raise ValueError(
                    f"Line {line_num}: 'confidence' must be numeric float; got '{conf_str}'."
                )
            if not math.isfinite(conf):
                raise ValueError(
                    f"Line {line_num}: 'confidence' is non-finite: '{conf_str}'."
                )
            if not (0.0 <= conf <= 1.0):
                raise ValueError(
                    f"Line {line_num}: 'confidence' out of valid range [0.0, 1.0]: {conf}."
                )

            # Validate box coordinates and dimensions
            coords: dict[str, float] = {}
            for col_name, col_idx in [
                ("x", x_col),
                ("y", y_col),
                ("width", width_col),
                ("height", height_col),
            ]:
                val_str = row[col_idx].strip()
                try:
                    val = float(val_str)
                except ValueError:
                    raise ValueError(
                        f"Line {line_num}: '{col_name}' must be numeric float; got '{val_str}'."
                    )
                if not math.isfinite(val):
                    raise ValueError(
                        f"Line {line_num}: '{col_name}' is non-finite: '{val_str}'."
                    )
                coords[col_name] = val

            if coords["width"] <= 0.0 or coords["height"] <= 0.0:
                raise ValueError(
                    f"Line {line_num}: Box dimensions must be strictly positive; "
                    f"got width={coords['width']}, height={coords['height']}."
                )

            # Validate elapsed timestamp
            elapsed_str = row[elapsed_col].strip()
            try:
                elapsed_val = float(elapsed_str)
            except ValueError:
                raise ValueError(
                    f"Line {line_num}: 'elapsed_from_source_start_s' must be numeric float; got '{elapsed_str}'."
                )
            if not math.isfinite(elapsed_val):
                raise ValueError(
                    f"Line {line_num}: 'elapsed_from_source_start_s' is non-finite: '{elapsed_str}'."
                )

            # Validate alignment_basis
            basis = row[basis_col].strip()
            if basis != EXPECTED_ALIGNMENT_BASIS:
                raise ValueError(
                    f"Line {line_num}: Invalid alignment_basis '{basis}'. "
                    f"Expected qualified basis '{EXPECTED_ALIGNMENT_BASIS}'."
                )

            # Validate frame-level consistency for timing and alignment_basis
            if frame_idx in frame_metadata:
                prev_elapsed_str, prev_elapsed_val, prev_basis = frame_metadata[frame_idx]
                if not math.isclose(elapsed_val, prev_elapsed_val, rel_tol=0.0, abs_tol=1e-7):
                    raise ValueError(
                        f"Line {line_num}: Inconsistent 'elapsed_from_source_start_s' in frame {frame_idx}: "
                        f"got {elapsed_str} ({elapsed_val}), previously {prev_elapsed_str} ({prev_elapsed_val})."
                    )
                if basis != prev_basis:
                    raise ValueError(
                        f"Line {line_num}: Inconsistent 'alignment_basis' in frame {frame_idx}: "
                        f"got '{basis}', previously '{prev_basis}'."
                    )
            else:
                frame_metadata[frame_idx] = (elapsed_str, elapsed_val, basis)

            obs = Observation(
                track_id=track_id,
                class_name=class_name,
                confidence_str=conf_str,
                confidence=conf,
                x=coords["x"],
                y=coords["y"],
                width=coords["width"],
                height=coords["height"],
            )
            frame_observations[frame_idx].append(obs)

    if total_observations == 0:
        raise ValueError(f"Input CSV contains no observation rows: {csv_path}")

    return frame_observations, frame_metadata, total_observations


# ---------------------------------------------------------------------------
# Geometric Computations (Image Space)
# ---------------------------------------------------------------------------

def compute_interval_gap(start_a: float, length_a: float, start_b: float, length_b: float) -> float:
    """Compute 1D separation between intervals [start_a, start_a + length_a] and [start_b, start_b + length_b].

    Gap on an axis is zero when the box intervals overlap or touch; otherwise it is their separation distance.
    """
    end_a = start_a + length_a
    end_b = start_b + length_b
    gap = max(start_a, start_b) - min(end_a, end_b)
    if gap <= 0.0:
        return 0.0
    return round(gap, 4)


def compute_box_iou(
    x_a: float, y_a: float, w_a: float, h_a: float,
    x_b: float, y_b: float, w_b: float, h_b: float,
) -> float:
    """Compute Intersection over Union (IoU) of two 2D bounding boxes in image space."""
    inter_x1 = max(x_a, x_b)
    inter_y1 = max(y_a, y_b)
    inter_x2 = min(x_a + w_a, x_b + w_b)
    inter_y2 = min(y_a + h_a, y_b + h_b)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    area_a = w_a * h_a
    area_b = w_b * h_b
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0

    return round(inter_area / union_area, 4)


def compute_center_distance_px(
    x_a: float, y_a: float, w_a: float, h_a: float,
    x_b: float, y_b: float, w_b: float, h_b: float,
) -> float:
    """Compute Euclidean distance between box geometric centers in image space."""
    center_x_a = x_a + w_a / 2.0
    center_y_a = y_a + h_a / 2.0
    center_x_b = x_b + w_b / 2.0
    center_y_b = y_b + h_b / 2.0
    return round(math.hypot(center_x_a - center_x_b, center_y_a - center_y_b), 4)


# ---------------------------------------------------------------------------
# Pair Observation Generation
# ---------------------------------------------------------------------------

def generate_pair_observations(
    frame_observations: dict[int, list[Observation]],
    frame_metadata: dict[int, tuple[str, float, str]],
) -> list[list[Any]]:
    """Generate image-space pair observations for each frame.

    Emits each unordered pair of different IDs once per frame, ordered by track ID.
    Uses all input observations without confidence filtering; frames with fewer than two observations emit no pair rows.
    """
    pair_rows: list[list[Any]] = []

    for frame_idx in sorted(frame_observations.keys()):
        obs_list = frame_observations[frame_idx]
        if len(obs_list) < 2:
            continue

        elapsed_str, _, alignment_basis = frame_metadata[frame_idx]

        # Sort observations by track_id ascending
        sorted_obs = sorted(obs_list, key=lambda o: o.track_id)

        for i in range(len(sorted_obs)):
            obs_a = sorted_obs[i]
            for j in range(i + 1, len(sorted_obs)):
                obs_b = sorted_obs[j]

                # track_id_a < track_id_b is guaranteed by sorted order and unique track IDs
                iou = compute_box_iou(
                    obs_a.x, obs_a.y, obs_a.width, obs_a.height,
                    obs_b.x, obs_b.y, obs_b.width, obs_b.height,
                )
                h_gap = compute_interval_gap(obs_a.x, obs_a.width, obs_b.x, obs_b.width)
                v_gap = compute_interval_gap(obs_a.y, obs_a.height, obs_b.y, obs_b.height)
                c_dist = compute_center_distance_px(
                    obs_a.x, obs_a.y, obs_a.width, obs_a.height,
                    obs_b.x, obs_b.y, obs_b.width, obs_b.height,
                )

                pair_rows.append([
                    frame_idx,
                    elapsed_str,
                    alignment_basis,
                    obs_a.track_id,
                    obs_b.track_id,
                    obs_a.class_name,
                    obs_b.class_name,
                    obs_a.confidence_str,
                    obs_b.confidence_str,
                    iou,
                    h_gap,
                    v_gap,
                    c_dist,
                ])

    return pair_rows


# ---------------------------------------------------------------------------
# Output Writing
# ---------------------------------------------------------------------------

def write_pair_observations_csv(output_path: Path, rows: list[list[Any]]) -> None:
    """Write pair observations to CSV file exclusively using mode 'x'.

    Fails with FileExistsError if destination already exists.
    """
    if output_path.exists():
        raise FileExistsError(
            f"Destination output file already exists: {output_path}. "
            "Refusing to overwrite existing file (mode 'x')."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_FIELDNAMES)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # Pre-check output existence before expensive parsing
    if args.output.exists():
        print(
            f"ERROR: Destination output file already exists: {args.output}\n"
            "Refusing to overwrite existing file (mode 'x').",
            file=sys.stderr,
        )
        return 1

    if not args.input.is_file():
        print(f"ERROR: Input trajectories CSV not found: {args.input}", file=sys.stderr)
        return 2

    print("\nReading and validating timed trajectories CSV...")
    try:
        frame_observations, frame_metadata, total_observations = load_and_validate_trajectories(args.input)
    except Exception as e:
        print(f"ERROR reading input CSV: {e}", file=sys.stderr)
        return 3

    print(f"  Total observations   : {total_observations:,}")
    print(f"  Observed frames      : {len(frame_observations):,}")

    print("\nGenerating same-frame image-space pair observations...")
    try:
        pair_rows = generate_pair_observations(frame_observations, frame_metadata)
    except Exception as e:
        print(f"ERROR generating pair observations: {e}", file=sys.stderr)
        return 4

    print(f"  Total pair rows      : {len(pair_rows):,}")

    print(f"\nWriting pair observations to {args.output} (mode 'x')...")
    try:
        write_pair_observations_csv(args.output, pair_rows)
    except FileExistsError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR writing output file: {e}", file=sys.stderr)
        return 5

    print(f"✓ Successfully wrote {len(pair_rows):,} vehicle pair observations to {args.output}")

    print("\n================================================================================")
    print("Vehicle Pair Observations Summary")
    print("================================================================================")
    print(f"Input trajectories file       : {args.input.resolve()}")
    print(f"Output pair observations CSV  : {args.output.resolve()}")
    print(f"Total input observations      : {total_observations:,}")
    print(f"Frames with >= 1 observation  : {len(frame_observations):,}")
    frames_with_pairs = sum(1 for obs in frame_observations.values() if len(obs) >= 2)
    print(f"Frames with >= 2 observations : {frames_with_pairs:,}")
    print(f"Total vehicle pairs emitted   : {len(pair_rows):,}")
    print("\n--- Scope & Evidence Caveats ---")
    print("1. Purpose           : Image-space geometric relationships for manual review.")
    print("                       NOT conflict classification, near-misses, or risk scoring.")
    print("2. Temporary IDs     : Track IDs represent temporary tracker associations, NOT verified")
    print("                       unique physical vehicles.")
    print("3. 2D Image Space    : Measurements are in 2D pixel coordinates and do not reflect 3D")
    print("                       world geometry, roadway lanes, or physical proximity.")
    print("4. Non-goals         : Does NOT infer lanes, physical proximity, collisions, speeds,")
    print("                       Time-To-Collision (TTC), or safety risk.")
    print("5. Completeness      : Uses all input observations without confidence filtering; frames with fewer than two observations emit no pair rows.")
    print("================================================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
