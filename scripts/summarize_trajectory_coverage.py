"""
summarize_trajectory_coverage.py — Summarize per-track temporal coverage, gaps, and class changes.

Reads a timed trajectory CSV (trajectories-timed.csv) and produces a per-track summary CSV
with one row per temporary track ID, detailing observation counts, frame and timestamp spans,
within-track missed-frame gaps, and class label transitions.

Scope & Constraints:
- Pure Python standard library only (no external dependencies).
- Validates all input data in memory BEFORE creating output:
  - Validates non-empty input CSV with unique header columns.
  - Requires valid positive integer track IDs and integer frame indices.
  - Rejects duplicate (frame_index, track_id) observation rows.
  - Requires the existing qualified alignment_basis ('same_index_supported_by_local_image_similarity').
  - Validates finite, strictly increasing timestamps per track ID.
- Gap & Span Definitions:
  - A within-ID gap occurs between successive observations when frame_index difference > 1.
  - Gap size counts missing frames (diff - 1), NOT the raw index difference.
  - unobserved_frames_within_span sums all missing frames across gaps within an ID's span.
  - largest_gap_frames is the maximum missing frames in any single gap (0 if no gaps).
  - endpoint_span_s is last minus first timestamp; it does NOT mean continuous visibility.
  - class_change_count counts changes between successive observed class labels.
- Creates output exclusively using mode "x" (refuses overwrite if destination exists).
- Important Caveats:
  1. Track IDs represent temporary tracker associations, NOT verified unique physical vehicles.
  2. Gaps and observation spans describe tracker persistence, NOT ground-truth vehicle presence or accuracy.
  3. Does NOT calculate unique vehicle counts, tracking accuracy (MOTA/HOTA), metric speeds, or conflict scores.

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
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_ALIGNMENT_BASIS = "same_index_supported_by_local_image_similarity"

REQUIRED_INPUT_COLUMNS = [
    "frame_index",
    "track_id",
    "class_name",
    "elapsed_from_source_start_s",
    "alignment_basis",
]

OUTPUT_FIELDNAMES = [
    "track_id",
    "observation_count",
    "first_frame",
    "last_frame",
    "first_elapsed_s",
    "last_elapsed_s",
    "endpoint_span_s",
    "within_id_gap_count",
    "unobserved_frames_within_span",
    "largest_gap_frames",
    "class_change_count",
]


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize per-track temporal coverage, missed-frame gaps, and class changes "
            "from a timed vehicle trajectory CSV.\n\n"
            "Caveats:\n"
            "- Track IDs are temporary tracking associations, NOT verified unique physical vehicles.\n"
            "- Endpoint span is last minus first timestamp; it does NOT mean continuous visibility.\n"
            "- Gaps describe tracker observation continuity, NOT detector accuracy or vehicle counts.\n"
            "- Does NOT compute metric speeds, distances, or conflict scores."
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
        help="Destination path for per-track coverage summary CSV. Must not already exist (exclusive creation).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data Loading & Validation
# ---------------------------------------------------------------------------

def load_and_validate_timed_trajectories(
    csv_path: Path,
) -> tuple[dict[int, list[tuple[int, float, str]]], int]:
    """Read and validate timed trajectory CSV.

    Validates:
    - Non-empty CSV with unique header columns.
    - All required columns present.
    - Expected alignment_basis value on every row.
    - Integer frame_index (>= 0) and positive integer track_id (> 0).
    - No duplicate (frame_index, track_id) pairs across the entire file.
    - Finite elapsed timestamps.

    Returns:
        tuple of (tracks_dict, total_observation_count)
        where tracks_dict maps track_id -> list of (frame_index, elapsed_s, class_name)
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input trajectories CSV not found: {csv_path}")

    tracks: dict[int, list[tuple[int, float, str]]] = defaultdict(list)
    seen_pairs: set[tuple[int, int]] = set()
    total_rows = 0

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"Input CSV is completely empty: {csv_path}")

        # Check for duplicate column names in header
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
        elapsed_col = header.index("elapsed_from_source_start_s")
        basis_col = header.index("alignment_basis")

        for line_num, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(
                    f"Line {line_num}: Column count mismatch (expected {len(header)}, got {len(row)})."
                )

            total_rows += 1

            # Validate alignment_basis
            basis = row[basis_col].strip()
            if basis != EXPECTED_ALIGNMENT_BASIS:
                raise ValueError(
                    f"Line {line_num}: Invalid alignment_basis '{basis}'. "
                    f"Expected qualified basis '{EXPECTED_ALIGNMENT_BASIS}'."
                )

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
            if pair in seen_pairs:
                raise ValueError(
                    f"Line {line_num}: Duplicate (frame_index, track_id) pair ({frame_idx}, {track_id}) detected."
                )
            seen_pairs.add(pair)

            # Validate class_name
            class_name = row[class_name_col].strip()
            if not class_name:
                raise ValueError(
                    f"Line {line_num}: 'class_name' must be a non-empty string."
                )

            # Validate elapsed timestamp
            elapsed_str = row[elapsed_col].strip()
            try:
                elapsed_s = float(elapsed_str)
            except ValueError:
                raise ValueError(
                    f"Line {line_num}: 'elapsed_from_source_start_s' must be a numeric float; got '{elapsed_str}'."
                )
            if not math.isfinite(elapsed_s):
                raise ValueError(
                    f"Line {line_num}: 'elapsed_from_source_start_s' is non-finite: '{elapsed_str}'."
                )

            tracks[track_id].append((frame_idx, elapsed_s, class_name))

    if total_rows == 0:
        raise ValueError(f"Input CSV contains no observation rows: {csv_path}")

    return tracks, total_rows


# ---------------------------------------------------------------------------
# Per-Track Coverage Summary Computation
# ---------------------------------------------------------------------------

def compute_track_summaries(
    tracks: dict[int, list[tuple[int, float, str]]],
) -> list[list[Any]]:
    """Compute per-track coverage, gaps, endpoint spans, and class changes.

    Validates that per-ID timestamps are strictly increasing along sorted frame indices.

    Returns:
        List of summary rows sorted by track_id ascending.
    """
    summary_rows: list[list[Any]] = []

    # Sort track IDs in ascending numerical order
    for track_id in sorted(tracks.keys()):
        obs = tracks[track_id]
        # Sort observations by frame_index (guaranteed unique by earlier pair check)
        obs.sort(key=lambda x: x[0])

        obs_count = len(obs)
        first_frame = obs[0][0]
        last_frame = obs[-1][0]
        first_elapsed = obs[0][1]
        last_elapsed = obs[-1][1]
        endpoint_span = last_elapsed - first_elapsed

        within_id_gap_count = 0
        unobserved_frames_within_span = 0
        largest_gap_frames = 0
        class_change_count = 0

        # Validate strictly increasing timestamps and compute gaps & class changes
        for i in range(1, obs_count):
            prev_frame, prev_elapsed, prev_class = obs[i - 1]
            curr_frame, curr_elapsed, curr_class = obs[i]

            # Require strictly increasing timestamps along observed frames
            if curr_elapsed <= prev_elapsed:
                raise ValueError(
                    f"Track ID {track_id}: Timestamp did not strictly increase from "
                    f"frame {prev_frame} ({prev_elapsed:.6f}s) to frame {curr_frame} ({curr_elapsed:.6f}s)."
                )

            # Check for gap: frame difference > 1
            frame_diff = curr_frame - prev_frame
            if frame_diff > 1:
                gap_size = frame_diff - 1
                within_id_gap_count += 1
                unobserved_frames_within_span += gap_size
                if gap_size > largest_gap_frames:
                    largest_gap_frames = gap_size

            # Check for class change
            if curr_class != prev_class:
                class_change_count += 1

        summary_rows.append([
            track_id,
            obs_count,
            first_frame,
            last_frame,
            f"{first_elapsed:.6f}",
            f"{last_elapsed:.6f}",
            f"{endpoint_span:.6f}",
            within_id_gap_count,
            unobserved_frames_within_span,
            largest_gap_frames,
            class_change_count,
        ])

    return summary_rows


# ---------------------------------------------------------------------------
# Output Writing
# ---------------------------------------------------------------------------

def write_coverage_csv(
    output_path: Path,
    summary_rows: list[list[Any]],
) -> None:
    """Write coverage summary CSV using exclusive creation mode 'x'.

    Raises:
        FileExistsError: If destination already exists.
    """
    if output_path.exists():
        raise FileExistsError(
            f"Destination file already exists: {output_path}. Refusing overwrite."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_FIELDNAMES)
        writer.writerows(summary_rows)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    print("================================================================================")
    print("Summarize Trajectory Coverage")
    print("================================================================================")
    print(f"Input timed CSV        : {args.input}")
    print(f"Destination output CSV : {args.output}")

    # Check output pre-existence before doing work
    if args.output.exists():
        print(f"ERROR: Destination already exists: {args.output}. Refusing overwrite.", file=sys.stderr)
        return 3

    # 1. Load and validate input CSV
    print("\nReading and validating timed trajectory CSV...")
    try:
        tracks, total_observations = load_and_validate_timed_trajectories(args.input)
    except Exception as e:
        print(f"ERROR reading input CSV: {e}", file=sys.stderr)
        return 4

    print(f"  Total observations   : {total_observations}")
    print(f"  Unique track IDs     : {len(tracks)}")

    # 2. Compute coverage summaries
    print("\nComputing per-track coverage, gaps, and class transitions...")
    try:
        summary_rows = compute_track_summaries(tracks)
    except Exception as e:
        print(f"ERROR computing summaries: {e}", file=sys.stderr)
        return 5

    # 3. Write summary CSV with exclusive creation mode 'x'
    print(f"\nWriting {len(summary_rows)} track summary rows to {args.output} (mode 'x')...")
    try:
        write_coverage_csv(args.output, summary_rows)
    except FileExistsError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"ERROR writing output file: {e}", file=sys.stderr)
        return 6

    # 4. Compute aggregate statistics for printed summary
    ids_with_gaps = sum(1 for row in summary_rows if row[7] > 0)
    total_within_id_gaps = sum(row[7] for row in summary_rows)
    total_unobserved_frames = sum(row[8] for row in summary_rows)
    max_single_gap = max((row[9] for row in summary_rows), default=0)
    ids_with_class_changes = sum(1 for row in summary_rows if row[10] > 0)
    total_class_changes = sum(row[10] for row in summary_rows)

    print("\n================================================================================")
    print("Coverage Summary Report")
    print("================================================================================")
    print(f"Total track IDs summarized    : {len(summary_rows)}")
    print(f"Total observations            : {total_observations}")
    print(f"IDs with internal gaps        : {ids_with_gaps} / {len(summary_rows)} ({ids_with_gaps / len(summary_rows) * 100:.2f}%)")
    print(f"Total within-ID gaps          : {total_within_id_gaps}")
    print(f"Total unobserved frames in gaps: {total_unobserved_frames}")
    print(f"Largest single gap across IDs : {max_single_gap} frames")
    print(f"IDs with class label changes  : {ids_with_class_changes} / {len(summary_rows)} ({ids_with_class_changes / len(summary_rows) * 100:.2f}%)")
    print(f"Total class label changes     : {total_class_changes}")

    print("\n--- Scope & Evidence Caveats ---")
    print("1. Temporary IDs     : Track IDs represent temporary tracker associations, NOT verified")
    print("                       unique physical vehicles. ID count alone does not establish vehicle count")
    print("                       or fragmentation; entries, exits, false detections and association breaks")
    print("                       require review.")
    print("2. Endpoint span     : endpoint_span_s is simply (last_elapsed - first_elapsed); it does NOT")
    print("                       imply continuous vehicle visibility throughout that timespan.")
    print("3. Gaps vs Accuracy  : Missed-frame gaps describe tracker continuity. They do not constitute")
    print("                       ground-truth accuracy (MOTA/HOTA) or detector failure measurements.")
    print("4. Non-goals         : Does NOT compute metric distances, speeds, 3D positions, or conflict scores.")
    print("================================================================================")

    return 0


if __name__ == "__main__":
    sys.exit(main())
