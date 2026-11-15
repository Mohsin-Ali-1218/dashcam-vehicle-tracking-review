"""
build_candidate_review.py — Offline interactive candidate vehicle pair review builder.

Creates a self-contained offline HTML tool for human image-space review of candidate
vehicle pairs from video footage, tracking results, and pairwise geometric observations.

CLI Inputs:
  --video        : Path to unannotated input video file (e.g. MP4).
  --tracks       : Path to tracks JSONL file (e.g. tracks.jsonl).
  --pairs        : Path to vehicle pair observations CSV (e.g. pair-observations.csv).
  --start-frame  : Starting frame index (zero-based inclusive).
  --end-frame    : Ending frame index (zero-based inclusive, max 60-frame interval).
  --ids          : Exactly two distinct positive temporary IDs (e.g. --ids 4 40).
  --output       : Destination path for self-contained HTML file (refuses overwrite).

Constraints & Guarantees:
- Uses existing OpenCV and Python standard library only; no framework, server, CDN, or network requests.
- Decodes sequentially from frame zero through the selected end frame to preserve exact indexing.
- Retains only the selected interval [start_frame, end_frame].
- Validates consecutive track frame indices, track entry structure, and required candidate evidence.
- Reuses annotate_tracks from render_vehicle_tracks.py for selected-ID bounding boxes and trails.
- Trails begin at the selected interval, reset whenever an ID is absent, and never interpolate.
- Displays frame controls (prev/next/slider/first/last), original/annotated image toggle,
  selected-ID presence, and exact saved pair evidence for each frame.
- Displays "No saved pair observation" when absent on a frame; does not invent zero-valued evidence
  or recalculate pair metrics.
- Includes a human review form with explicit reviewed frame range, assessment (same vehicle /
  distinct vehicles / uncertain), uncertainty notes, and supporting observations (starts unreviewed).
- Download button exports review JSON with metadata and ISO review timestamp, explicitly noting
  that download does not automatically update the repository.
- Prominently labeled: Manual image-space review, temporary IDs, no physical-conflict conclusion.
- Refuses to overwrite existing HTML output files.

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

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2

# Import tracking visualization utilities from render_vehicle_tracks
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.render_vehicle_tracks import annotate_tracks, get_track_color


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_INTERVAL_FRAMES = 60
REVIEW_SCHEMA_VERSION = "candidate_review_v1"
EXPECTED_ALIGNMENT_BASIS = "same_index_supported_by_local_image_similarity"

REQUIRED_PAIR_COLUMNS = [
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
# CLI Argument Parsing & Input Normalization
# ---------------------------------------------------------------------------

def parse_ids_argument(raw_ids: list[Any]) -> tuple[int, int]:
    """Parse and validate exactly two distinct positive integer track IDs."""
    tokens: list[str] = []
    for item in raw_ids:
        for part in str(item).replace(",", " ").split():
            tokens.append(part)

    if len(tokens) != 2:
        raise ValueError(
            f"Expected exactly two distinct positive temporary IDs for --ids; got {len(tokens)}: {raw_ids}"
        )

    try:
        id_a = int(tokens[0])
        id_b = int(tokens[1])
    except ValueError:
        raise ValueError(f"--ids must be integers; got: {tokens}")

    if id_a <= 0 or id_b <= 0:
        raise ValueError(f"--ids must be positive integers (> 0); got: ({id_a}, {id_b})")

    if id_a == id_b:
        raise ValueError(f"--ids must be two distinct IDs; got duplicate: {id_a}")

    return id_a, id_b


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an offline self-contained HTML tool for manual candidate vehicle pair review.\n\n"
            "Caveats:\n"
            "- Manual 2D image-space review of temporary tracking associations.\n"
            "- Does NOT perform physical conflict classification, 3D proximity, or safety assessment."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to unannotated input video file (e.g. daylight-2018-07-29-segment-7.mp4).",
    )
    parser.add_argument(
        "--tracks",
        type=Path,
        required=True,
        help="Path to tracks JSONL file (e.g. tracks.jsonl).",
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        required=True,
        help="Path to vehicle pair observations CSV (e.g. pair-observations.csv).",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        required=True,
        help="Starting frame index (zero-based inclusive, non-negative).",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        required=True,
        help="Ending frame index (zero-based inclusive, max 60-frame interval).",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        required=True,
        help="Exactly two distinct positive temporary IDs (e.g. --ids 4 40).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for self-contained HTML file (refuses overwrite).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Input Validation & Sequential Video Extraction
# ---------------------------------------------------------------------------

def validate_frame_bounds(start_frame: int, end_frame: int) -> int:
    """Validate zero-based inclusive frame bounds and maximum interval length."""
    if start_frame < 0:
        raise ValueError(f"--start-frame must be non-negative (>= 0); got {start_frame}")
    if end_frame < start_frame:
        raise ValueError(
            f"--end-frame ({end_frame}) cannot be smaller than --start-frame ({start_frame})"
        )
    num_frames = end_frame - start_frame + 1
    if num_frames > MAX_INTERVAL_FRAMES:
        raise ValueError(
            f"Requested frame interval spans {num_frames} frames ({start_frame}..{end_frame}), "
            f"exceeding the maximum allowed interval of {MAX_INTERVAL_FRAMES} frames."
        )
    return num_frames


def decode_selected_video_frames(
    video_path: Path,
    start_frame: int,
    end_frame: int,
) -> tuple[dict[int, Any], dict[str, Any]]:
    """Sequentially decode video from frame 0 through end_frame, retaining only [start_frame, end_frame].

    Preserves exact frame indexing without seeking drift. Fails clearly on missing frames.
    """
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video file not found: '{video_path}'")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video file: '{video_path}'")

    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        container_fps = cap.get(cv2.CAP_PROP_FPS)

        if w <= 0 or h <= 0:
            raise ValueError(f"Input video reported invalid frame dimensions: {w}x{h} px.")
        if not (math.isfinite(container_fps) and container_fps > 0):
            raise ValueError(f"Input video reported invalid container FPS: {container_fps}.")

        retained_frames: dict[int, Any] = {}
        current_idx = 0

        # Sequential decode from frame 0 up through end_frame
        while current_idx <= end_frame:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"Video ended prematurely at frame index {current_idx}; "
                    f"expected frames through end frame {end_frame}."
                )
            if current_idx >= start_frame:
                retained_frames[current_idx] = frame
            current_idx += 1

        video_meta = {
            "width": w,
            "height": h,
            "fps": float(container_fps),
            "video_path": str(video_path.resolve()),
        }
        return retained_frames, video_meta
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Tracks & Pair Evidence Loading & Validation
# ---------------------------------------------------------------------------

def load_and_validate_tracks(
    tracks_path: Path,
    start_frame: int,
    end_frame: int,
) -> dict[int, list[dict[str, Any]]]:
    """Read and validate consecutive matching track frame indices and record structures from tracks.jsonl.

    Validations:
    - Consecutive integer frame indices without gaps up through end_frame.
    - Each track entry must be a dictionary.
    - Positive integer track_id (rejects booleans).
    - No duplicate track IDs within the same frame.
    - Four finite bounding box coordinates with strictly positive width and height.
    - Non-empty string class_name.
    - Finite float confidence within [0.0, 1.0] (rejects booleans).
    """
    if not tracks_path.is_file():
        raise FileNotFoundError(f"Tracks JSONL file not found: '{tracks_path}'")

    tracks_by_frame: dict[int, list[dict[str, Any]]] = {}
    expected_frame_idx: int | None = None

    with tracks_path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Line {line_num} in '{tracks_path}' is invalid JSON: {exc}"
                )

            if not isinstance(record, dict) or "frame_index" not in record or "tracks" not in record:
                raise ValueError(
                    f"Line {line_num} in '{tracks_path}' missing required 'frame_index' or 'tracks' field."
                )

            track_frame_idx = record["frame_index"]
            if isinstance(track_frame_idx, bool) or not isinstance(track_frame_idx, int) or track_frame_idx < 0:
                raise ValueError(
                    f"Line {line_num} in '{tracks_path}': 'frame_index' must be non-negative integer; "
                    f"got {track_frame_idx!r}."
                )

            # Check consecutive frame indices
            if expected_frame_idx is None:
                expected_frame_idx = track_frame_idx
                if expected_frame_idx > start_frame:
                    raise ValueError(
                        f"Tracks file begins at frame {expected_frame_idx}, which is after requested "
                        f"start frame {start_frame}."
                    )
            elif track_frame_idx != expected_frame_idx:
                raise ValueError(
                    f"Line {line_num} in '{tracks_path}': Non-consecutive track frame index; "
                    f"expected {expected_frame_idx}, got {track_frame_idx}."
                )

            tracks_list = record["tracks"]
            if not isinstance(tracks_list, list):
                raise ValueError(f"Line {line_num} in '{tracks_path}': 'tracks' field must be a list.")

            # Validate each track entry in the record
            seen_ids_in_frame: set[int] = set()
            for trk_idx, trk in enumerate(tracks_list):
                if not isinstance(trk, dict):
                    raise ValueError(
                        f"Line {line_num} in '{tracks_path}': Track entry {trk_idx} must be a dictionary; got {trk!r}."
                    )

                # Validate track_id: integer positive ID, reject booleans
                tid = trk.get("track_id")
                if isinstance(tid, bool) or not isinstance(tid, int) or tid <= 0:
                    raise ValueError(
                        f"Line {line_num} in '{tracks_path}': Track entry {trk_idx} 'track_id' must be a positive integer "
                        f"(not boolean); got {tid!r}."
                    )

                # Reject duplicate IDs within a frame
                if tid in seen_ids_in_frame:
                    raise ValueError(
                        f"Line {line_num} in '{tracks_path}': Duplicate track ID {tid} in frame {track_frame_idx}."
                    )
                seen_ids_in_frame.add(tid)

                # Validate class_name: non-empty string
                class_name = trk.get("class_name")
                if not isinstance(class_name, str) or not class_name.strip():
                    raise ValueError(
                        f"Line {line_num} in '{tracks_path}': Track {tid} 'class_name' must be a non-empty string; got {class_name!r}."
                    )

                # Validate confidence: finite float in [0.0, 1.0] (rejects booleans)
                conf = trk.get("confidence")
                if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not math.isfinite(conf) or not (0.0 <= conf <= 1.0):
                    raise ValueError(
                        f"Line {line_num} in '{tracks_path}': Track {tid} 'confidence' must be a finite number in [0.0, 1.0]; got {conf!r}."
                    )

                # Validate box: 4 finite coordinates with strictly positive width and height
                box = trk.get("box")
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    raise ValueError(
                        f"Line {line_num} in '{tracks_path}': Track {tid} 'box' must be a list of 4 numbers; got {box!r}."
                    )
                for coord_idx, coord in enumerate(box):
                    if isinstance(coord, bool) or not isinstance(coord, (int, float)) or not math.isfinite(coord):
                        raise ValueError(
                            f"Line {line_num} in '{tracks_path}': Track {tid} box coordinate {coord_idx} must be finite number; got {coord!r}."
                        )
                bx, by, bw, bh = [float(v) for v in box]
                if bw <= 0.0 or bh <= 0.0:
                    raise ValueError(
                        f"Line {line_num} in '{tracks_path}': Track {tid} box dimensions must be strictly positive (> 0); "
                        f"got width={bw}, height={bh}."
                    )

            if start_frame <= track_frame_idx <= end_frame:
                tracks_by_frame[track_frame_idx] = tracks_list

            expected_frame_idx += 1
            if track_frame_idx >= end_frame:
                break

    # Validate complete coverage of requested interval
    for f_idx in range(start_frame, end_frame + 1):
        if f_idx not in tracks_by_frame:
            raise ValueError(
                f"Tracks file '{tracks_path}' did not contain required frame index {f_idx} "
                f"(interval spans {start_frame}..{end_frame})."
            )

    return tracks_by_frame


def load_and_validate_pairs(
    pairs_path: Path,
    start_frame: int,
    end_frame: int,
    target_ids: tuple[int, int],
) -> tuple[dict[int, dict[str, Any]], dict[int, tuple[str, float, str]]]:
    """Read and validate vehicle pair observations CSV.

    Validations:
    - Rejects duplicate header column names.
    - Requires all REQUIRED_PAIR_COLUMNS.
    - Requires positive ordered IDs (track_id_a < track_id_b).
    - Rejects duplicate selected-pair rows for the same frame.
    - Requires finite numeric values for gaps, distances, confidences, IoU, and timestamps.
    - Requires confidence and IoU within [0.0, 1.0].
    - Requires non-negative gaps, distance, and elapsed timestamp (>= 0.0).
    - Requires alignment_basis equal to 'same_index_supported_by_local_image_similarity'.
    - Rejects inconsistent timing/alignment across rows within selected frames.
    - Preserves exact original CSV strings without toFixed rounding.
    - Fails clearly on invalid timestamps with no fallback to 0.0.
    """
    if not pairs_path.is_file():
        raise FileNotFoundError(f"Pair observations CSV file not found: '{pairs_path}'")

    pair_evidence_by_frame: dict[int, dict[str, Any]] = {}
    frame_timing: dict[int, tuple[str, float, str]] = {}
    target_pair = (min(target_ids), max(target_ids))

    with pairs_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"Pair observations CSV is empty: '{pairs_path}'")

        # Reject duplicate CSV header column names
        if len(header) != len(set(header)):
            seen_headers: set[str] = set()
            duplicates: list[str] = []
            for col in header:
                if col in seen_headers and col not in duplicates:
                    duplicates.append(col)
                seen_headers.add(col)
            raise ValueError(f"Pair observations CSV contains duplicate header columns: {duplicates}")

        # Check required columns
        for req_col in REQUIRED_PAIR_COLUMNS:
            if req_col not in header:
                raise ValueError(f"Pair observations CSV missing required column '{req_col}': '{pairs_path}'")

        col_map = {col: header.index(col) for col in REQUIRED_PAIR_COLUMNS}

        for line_num, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(
                    f"Line {line_num} in '{pairs_path}': Column count mismatch "
                    f"(expected {len(header)}, got {len(row)})."
                )

            # Frame index
            f_idx_str = row[col_map["frame_index"]].strip()
            try:
                f_idx = int(f_idx_str)
            except ValueError:
                raise ValueError(f"Line {line_num} in '{pairs_path}': Invalid integer 'frame_index': '{f_idx_str}'.")
            if str(f_idx) != f_idx_str:
                raise ValueError(f"Line {line_num} in '{pairs_path}': 'frame_index' contains non-integer string: '{f_idx_str}'.")
            if f_idx < 0:
                raise ValueError(f"Line {line_num} in '{pairs_path}': 'frame_index' must be non-negative (>= 0); got {f_idx}.")

            # Track IDs: require positive ordered IDs (track_id_a < track_id_b)
            t_a_str = row[col_map["track_id_a"]].strip()
            t_b_str = row[col_map["track_id_b"]].strip()
            try:
                t_a = int(t_a_str)
                t_b = int(t_b_str)
            except ValueError:
                raise ValueError(f"Line {line_num} in '{pairs_path}': Non-integer track IDs: '{t_a_str}', '{t_b_str}'.")
            if str(t_a) != t_a_str or str(t_b) != t_b_str:
                raise ValueError(
                    f"Line {line_num} in '{pairs_path}': Track IDs contain non-integer representations: "
                    f"track_id_a='{t_a_str}', track_id_b='{t_b_str}'."
                )
            if t_a <= 0 or t_b <= 0:
                raise ValueError(
                    f"Line {line_num} in '{pairs_path}': Track IDs must be positive integers (> 0); "
                    f"got track_id_a={t_a}, track_id_b={t_b}."
                )
            if t_a >= t_b:
                raise ValueError(
                    f"Line {line_num} in '{pairs_path}': Track IDs must be strictly ordered (track_id_a < track_id_b); "
                    f"got track_id_a={t_a}, track_id_b={t_b}."
                )

            # Timing & Alignment basis: require valid finite timestamp and expected alignment basis
            elapsed_str = row[col_map["elapsed_from_source_start_s"]].strip()
            try:
                elapsed_val = float(elapsed_str)
            except ValueError as exc:
                raise ValueError(
                    f"Line {line_num} in '{pairs_path}': 'elapsed_from_source_start_s' must be numeric float; "
                    f"got '{elapsed_str}'."
                ) from exc
            if not math.isfinite(elapsed_val):
                raise ValueError(f"Line {line_num} in '{pairs_path}': 'elapsed_from_source_start_s' is non-finite: '{elapsed_str}'.")
            if elapsed_val < 0.0:
                raise ValueError(f"Line {line_num} in '{pairs_path}': 'elapsed_from_source_start_s' must be non-negative; got {elapsed_val}.")

            basis = row[col_map["alignment_basis"]].strip()
            if basis != EXPECTED_ALIGNMENT_BASIS:
                raise ValueError(
                    f"Line {line_num} in '{pairs_path}': Invalid alignment_basis '{basis}'. "
                    f"Expected qualified basis '{EXPECTED_ALIGNMENT_BASIS}'."
                )

            # Check and record consistent frame timing within review window
            if start_frame <= f_idx <= end_frame:
                if f_idx in frame_timing:
                    prev_elapsed_str, prev_elapsed_val, prev_basis = frame_timing[f_idx]
                    if not math.isclose(elapsed_val, prev_elapsed_val, rel_tol=0.0, abs_tol=1e-7):
                        raise ValueError(
                            f"Line {line_num} in '{pairs_path}': Inconsistent 'elapsed_from_source_start_s' in frame {f_idx}: "
                            f"got {elapsed_str} ({elapsed_val}), previously {prev_elapsed_str} ({prev_elapsed_val})."
                        )
                    if basis != prev_basis:
                        raise ValueError(
                            f"Line {line_num} in '{pairs_path}': Inconsistent 'alignment_basis' in frame {f_idx}: "
                            f"got '{basis}', previously '{prev_basis}'."
                        )
                else:
                    frame_timing[f_idx] = (elapsed_str, elapsed_val, basis)

            # Check candidate pair row
            row_pair = (t_a, t_b)
            if row_pair == target_pair and start_frame <= f_idx <= end_frame:
                # Reject duplicate selected-pair rows for the same frame
                if f_idx in pair_evidence_by_frame:
                    raise ValueError(
                        f"Line {line_num} in '{pairs_path}': Duplicate pair observation row detected for "
                        f"frame {f_idx} and IDs ({t_a}, {t_b})."
                    )

                # Classes
                class_a = row[col_map["class_name_a"]].strip()
                class_b = row[col_map["class_name_b"]].strip()
                if not class_a or not class_b:
                    raise ValueError(f"Line {line_num} in '{pairs_path}': Class names must be non-empty strings.")

                # Numerical fields
                conf_a_str = row[col_map["confidence_a"]].strip()
                conf_b_str = row[col_map["confidence_b"]].strip()
                box_iou_str = row[col_map["box_iou"]].strip()
                h_gap_str = row[col_map["horizontal_box_gap_px"]].strip()
                v_gap_str = row[col_map["vertical_box_gap_px"]].strip()
                c_dist_str = row[col_map["center_distance_px"]].strip()

                for field_name, field_str in [
                    ("confidence_a", conf_a_str),
                    ("confidence_b", conf_b_str),
                    ("box_iou", box_iou_str),
                    ("horizontal_box_gap_px", h_gap_str),
                    ("vertical_box_gap_px", v_gap_str),
                    ("center_distance_px", c_dist_str),
                ]:
                    try:
                        num_f = float(field_str)
                    except ValueError as exc:
                        raise ValueError(f"Line {line_num} in '{pairs_path}': '{field_name}' must be numeric float; got '{field_str}'.") from exc
                    if not math.isfinite(num_f):
                        raise ValueError(f"Line {line_num} in '{pairs_path}': '{field_name}' is non-finite: '{field_str}'.")

                conf_a = float(conf_a_str)
                conf_b = float(conf_b_str)
                box_iou = float(box_iou_str)
                h_gap = float(h_gap_str)
                v_gap = float(v_gap_str)
                c_dist = float(c_dist_str)

                if not (0.0 <= conf_a <= 1.0):
                    raise ValueError(f"Line {line_num} in '{pairs_path}': 'confidence_a' out of valid range [0.0, 1.0]: {conf_a}.")
                if not (0.0 <= conf_b <= 1.0):
                    raise ValueError(f"Line {line_num} in '{pairs_path}': 'confidence_b' out of valid range [0.0, 1.0]: {conf_b}.")
                if not (0.0 <= box_iou <= 1.0):
                    raise ValueError(f"Line {line_num} in '{pairs_path}': 'box_iou' out of valid range [0.0, 1.0]: {box_iou}.")
                if h_gap < 0.0:
                    raise ValueError(f"Line {line_num} in '{pairs_path}': 'horizontal_box_gap_px' must be non-negative (>= 0); got {h_gap}.")
                if v_gap < 0.0:
                    raise ValueError(f"Line {line_num} in '{pairs_path}': 'vertical_box_gap_px' must be non-negative (>= 0); got {v_gap}.")
                if c_dist < 0.0:
                    raise ValueError(f"Line {line_num} in '{pairs_path}': 'center_distance_px' must be non-negative (>= 0); got {c_dist}.")

                pair_evidence_by_frame[f_idx] = {
                    "frame_index": f_idx,
                    "frame_index_str": f_idx_str,
                    "elapsed_from_source_start_s": elapsed_val,
                    "elapsed_from_source_start_s_str": elapsed_str,
                    "alignment_basis": basis,
                    "track_id_a": t_a,
                    "track_id_a_str": t_a_str,
                    "track_id_b": t_b,
                    "track_id_b_str": t_b_str,
                    "class_name_a": class_a,
                    "class_name_b": class_b,
                    "confidence_a": conf_a,
                    "confidence_a_str": conf_a_str,
                    "confidence_b": conf_b,
                    "confidence_b_str": conf_b_str,
                    "box_iou": box_iou,
                    "box_iou_str": box_iou_str,
                    "horizontal_box_gap_px": h_gap,
                    "horizontal_box_gap_px_str": h_gap_str,
                    "vertical_box_gap_px": v_gap,
                    "vertical_box_gap_px_str": v_gap_str,
                    "center_distance_px": c_dist,
                    "center_distance_px_str": c_dist_str,
                }

    return pair_evidence_by_frame, frame_timing


def validate_pair_and_tracks_agreement(
    pair_evidence_by_frame: dict[int, dict[str, Any]],
    tracks_by_frame: dict[int, list[dict[str, Any]]],
) -> None:
    """For each saved candidate pair row, confirm both IDs are observed in that frame
    and that their classes and confidences agree with the tracking records.

    Does not recalculate pair geometry.
    """
    for f_idx, pair_ev in sorted(pair_evidence_by_frame.items()):
        if f_idx not in tracks_by_frame:
            raise ValueError(
                f"Saved pair observation exists for frame {f_idx}, but frame {f_idx} is missing from tracks."
            )

        frame_tracks = {int(t["track_id"]): t for t in tracks_by_frame[f_idx]}
        t_a = pair_ev["track_id_a"]
        t_b = pair_ev["track_id_b"]

        if t_a not in frame_tracks:
            raise ValueError(
                f"Frame {f_idx}: Saved pair observation includes track ID {t_a}, "
                f"but ID {t_a} was not observed in tracks for frame {f_idx}."
            )
        if t_b not in frame_tracks:
            raise ValueError(
                f"Frame {f_idx}: Saved pair observation includes track ID {t_b}, "
                f"but ID {t_b} was not observed in tracks for frame {f_idx}."
            )

        trk_a = frame_tracks[t_a]
        trk_b = frame_tracks[t_b]

        # Confirm class agreement
        trk_class_a = str(trk_a["class_name"]).strip()
        pair_class_a = str(pair_ev["class_name_a"]).strip()
        if trk_class_a != pair_class_a:
            raise ValueError(
                f"Frame {f_idx}: Class mismatch for track ID {t_a}: "
                f"tracks has '{trk_class_a}', pair observation has '{pair_class_a}'."
            )

        trk_class_b = str(trk_b["class_name"]).strip()
        pair_class_b = str(pair_ev["class_name_b"]).strip()
        if trk_class_b != pair_class_b:
            raise ValueError(
                f"Frame {f_idx}: Class mismatch for track ID {t_b}: "
                f"tracks has '{trk_class_b}', pair observation has '{pair_class_b}'."
            )

        # Confirm confidence agreement
        trk_conf_a = float(trk_a["confidence"])
        pair_conf_a = float(pair_ev["confidence_a"])
        if not math.isclose(trk_conf_a, pair_conf_a, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(
                f"Frame {f_idx}: Confidence mismatch for track ID {t_a}: "
                f"tracks has {trk_conf_a}, pair observation has {pair_conf_a}."
            )

        trk_conf_b = float(trk_b["confidence"])
        pair_conf_b = float(pair_ev["confidence_b"])
        if not math.isclose(trk_conf_b, pair_conf_b, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(
                f"Frame {f_idx}: Confidence mismatch for track ID {t_b}: "
                f"tracks has {trk_conf_b}, pair observation has {pair_conf_b}."
            )


# ---------------------------------------------------------------------------
# Annotation & Frame Processing
# ---------------------------------------------------------------------------

def process_review_frames(
    retained_frames: dict[int, Any],
    tracks_by_frame: dict[int, list[dict[str, Any]]],
    pair_evidence_by_frame: dict[int, dict[str, Any]],
    frame_timing: dict[int, tuple[str, float, str]],
    start_frame: int,
    end_frame: int,
    selected_ids: tuple[int, int],
) -> list[dict[str, Any]]:
    """Annotate frames, compute image-space trails, and encode JPEG Base64 data."""
    pair_id_a = min(selected_ids)
    pair_id_b = max(selected_ids)
    selected_ids_set = {pair_id_a, pair_id_b}

    active_trails: dict[int, list[tuple[int, int]]] = {}
    frames_data: list[dict[str, Any]] = []

    for f_idx in range(start_frame, end_frame + 1):
        raw_frame = retained_frames[f_idx]
        all_tracks = tracks_by_frame[f_idx]

        # Filter tracks to selected IDs only
        selected_tracks = [trk for trk in all_tracks if int(trk["track_id"]) in selected_ids_set]
        observed_ids_in_frame = set()

        presence_by_id: dict[str, dict[str, Any]] = {}

        for trk in selected_tracks:
            tid = int(trk["track_id"])
            observed_ids_in_frame.add(tid)
            box = trk["box"]
            bx, by, bw, bh = [float(v) for v in box]
            cx = int(round(bx + bw / 2.0))
            cy = int(round(by + bh / 2.0))

            # Trails begin at the selected interval, never interpolate, and reset when absent
            if tid in active_trails:
                active_trails[tid].append((cx, cy))
            else:
                active_trails[tid] = [(cx, cy)]

            presence_by_id[str(tid)] = {
                "present": True,
                "track_id": tid,
                "role": "Pair ID A" if tid == pair_id_a else "Pair ID B",
                "class_name": str(trk["class_name"]),
                "confidence": float(trk["confidence"]),
                "confidence_str": str(trk["confidence"]),
                "box": [round(bx, 1), round(by, 1), round(bw, 1), round(bh, 1)],
                "center": [cx, cy],
            }

        # Clear trail history whenever an ID is absent from a frame
        for tid in (pair_id_a, pair_id_b):
            if tid not in observed_ids_in_frame:
                if tid in active_trails:
                    del active_trails[tid]
                presence_by_id[str(tid)] = {
                    "present": False,
                    "track_id": tid,
                    "role": "Pair ID A" if tid == pair_id_a else "Pair ID B",
                }

        # Reuse annotate_tracks from render_vehicle_tracks.py
        # Generate annotations for Both IDs, ID A only, and ID B only to allow
        # the reviewer to inspect each candidate ID separately when boxes/labels/trails overlap.
        tracks_a = [trk for trk in selected_tracks if int(trk["track_id"]) == pair_id_a]
        tracks_b = [trk for trk in selected_tracks if int(trk["track_id"]) == pair_id_b]

        annotated_frame_both = annotate_tracks(raw_frame, selected_tracks, trails=active_trails)
        annotated_frame_a = annotate_tracks(raw_frame, tracks_a, trails=active_trails)
        annotated_frame_b = annotate_tracks(raw_frame, tracks_b, trails=active_trails)

        # Encode original and annotated frames as JPEG base64 strings
        ok_orig, buf_orig = cv2.imencode(".jpg", raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok_orig:
            raise RuntimeError(f"OpenCV failed to encode original JPEG for frame {f_idx}")
        ok_annot_both, buf_annot_both = cv2.imencode(".jpg", annotated_frame_both, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok_annot_both:
            raise RuntimeError(f"OpenCV failed to encode annotated 'both' JPEG for frame {f_idx}")
        ok_annot_a, buf_annot_a = cv2.imencode(".jpg", annotated_frame_a, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok_annot_a:
            raise RuntimeError(f"OpenCV failed to encode annotated 'ID A' JPEG for frame {f_idx}")
        ok_annot_b, buf_annot_b = cv2.imencode(".jpg", annotated_frame_b, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok_annot_b:
            raise RuntimeError(f"OpenCV failed to encode annotated 'ID B' JPEG for frame {f_idx}")

        orig_b64 = "data:image/jpeg;base64," + base64.b64encode(buf_orig).decode("ascii")
        annot_both_b64 = "data:image/jpeg;base64," + base64.b64encode(buf_annot_both).decode("ascii")
        annot_a_b64 = "data:image/jpeg;base64," + base64.b64encode(buf_annot_a).decode("ascii")
        annot_b_b64 = "data:image/jpeg;base64," + base64.b64encode(buf_annot_b).decode("ascii")

        # Saved pair observation (None when absent)
        saved_pair = pair_evidence_by_frame.get(f_idx, None)

        elapsed_s = None
        elapsed_str = None
        alignment_basis = None
        if saved_pair:
            elapsed_s = saved_pair["elapsed_from_source_start_s"]
            elapsed_str = saved_pair["elapsed_from_source_start_s_str"]
            alignment_basis = saved_pair["alignment_basis"]
        elif f_idx in frame_timing:
            elapsed_str, elapsed_s, alignment_basis = frame_timing[f_idx]

        frames_data.append({
            "frame_index": f_idx,
            "elapsed_from_source_start_s": elapsed_s,
            "elapsed_from_source_start_s_str": elapsed_str,
            "alignment_basis": alignment_basis,
            "orig_img": orig_b64,
            "annot_img": annot_both_b64,
            "annot_both_img": annot_both_b64,
            "annot_a_img": annot_a_b64,
            "annot_b_img": annot_b_b64,
            "presence": presence_by_id,
            "pair_evidence": saved_pair,
        })

    return frames_data


# ---------------------------------------------------------------------------
# Self-Contained Offline HTML Generator
# ---------------------------------------------------------------------------

def bgr_to_css_hex(bgr: tuple[int, int, int]) -> str:
    """Convert an OpenCV BGR tuple to a CSS hex string #RRGGBB."""
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def build_offline_html(
    meta_payload: dict[str, Any],
    frames_payload: list[dict[str, Any]],
) -> str:
    """Generate self-contained offline HTML file with embedded images, evidence, and plain JS."""
    full_payload = {
        "meta": meta_payload,
        "frames": frames_payload,
    }

    # Safely escape embedded JSON for inline script injection, rejecting NaN/Infinity
    safe_json_data = (
        json.dumps(full_payload, ensure_ascii=False, allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )

    id_a = meta_payload["pair_ids"]["track_id_a"]
    id_b = meta_payload["pair_ids"]["track_id_b"]
    cli_ids = meta_payload["selected_ids"]
    color_a = meta_payload["track_colors"][str(id_a)]
    color_b = meta_payload["track_colors"][str(id_b)]
    start_frame = meta_payload["start_frame"]
    end_frame = meta_payload["end_frame"]

    order_note = (
        f"CLI IDs: [{cli_ids[0]}, {cli_ids[1]}] &rarr; Ordered Pair: ID A = {id_a}, ID B = {id_b}"
        if (cli_ids[0] != id_a)
        else f"Ordered Pair: ID A = {id_a}, ID B = {id_b}"
    )

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Candidate Pair Review — IDs {id_a} &amp; {id_b} (Frames {start_frame}–{end_frame})</title>
  <style>
    /* CSS Design System: Clean, high-contrast dark theme */
    :root {{
      --bg-base: #0b0f17;
      --bg-panel: #131a26;
      --bg-panel-subtle: #1a2332;
      --bg-input: #0f1622;
      --border-color: #273449;
      --border-accent: #3b82f6;
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --color-ida: {color_a};
      --color-idb: {color_b};
      --accent-green: #10b981;
      --accent-amber: #f59e0b;
      --accent-rose: #f43f5e;
      --accent-blue: #38bdf8;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg-base);
      color: var(--text-main);
      line-height: 1.45;
      padding: 16px;
      font-size: 14px;
    }}

    .app-container {{
      max-width: 1440px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}

    /* Header & Scope Banner */
    header.review-header {{
      background-color: var(--bg-panel);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 16px 20px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}

    .header-top {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }}

    .header-title {{
      font-size: 18px;
      font-weight: 700;
      letter-spacing: -0.01em;
      color: #ffffff;
      display: flex;
      align-items: center;
      gap: 10px;
    }}

    .badges-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}

    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 3px 8px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      border-radius: 4px;
      border: 1px solid transparent;
    }}

    .badge-scope {{
      background-color: rgba(245, 158, 11, 0.15);
      color: #fbbf24;
      border-color: rgba(245, 158, 11, 0.3);
    }}

    .badge-offline {{
      background-color: rgba(56, 189, 248, 0.15);
      color: var(--accent-blue);
      border-color: rgba(56, 189, 248, 0.3);
    }}

    .badge-ida {{
      background-color: rgba(255, 255, 255, 0.05);
      border-color: var(--color-ida);
      color: var(--color-ida);
    }}

    .badge-idb {{
      background-color: rgba(255, 255, 255, 0.05);
      border-color: var(--color-idb);
      color: var(--color-idb);
    }}

    .header-caveat-banner {{
      background-color: rgba(245, 158, 11, 0.08);
      border-left: 3px solid #f59e0b;
      padding: 8px 12px;
      font-size: 12px;
      color: #cbd5e1;
      border-radius: 0 4px 4px 0;
    }}

    .header-meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
      font-size: 12px;
      color: var(--text-muted);
      border-top: 1px solid rgba(255, 255, 255, 0.07);
      padding-top: 10px;
    }}

    .header-meta-grid strong {{
      color: var(--text-main);
    }}

    /* Main Review Layout: Split 2 columns */
    .main-layout {{
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(360px, 1fr);
      gap: 16px;
      align-items: start;
    }}

    @media (max-width: 1080px) {{
      .main-layout {{
        grid-template-columns: 1fr;
      }}
    }}

    /* Left Column: Frame Viewer */
    .viewer-panel {{
      background-color: var(--bg-panel);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}

    .toolbar-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
    }}

    .toggle-group {{
      display: inline-flex;
      background-color: var(--bg-base);
      border: 1px solid var(--border-color);
      border-radius: 6px;
      padding: 2px;
    }}

    .toggle-btn {{
      background: none;
      border: none;
      color: var(--text-muted);
      padding: 5px 12px;
      font-size: 12px;
      font-weight: 600;
      border-radius: 4px;
      cursor: pointer;
      transition: all 0.15s ease;
    }}

    .toggle-btn.active {{
      background-color: var(--border-accent);
      color: #ffffff;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }}

    .view-selectors-wrap {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}

    .toggle-btn#select-ida.active {{
      background-color: rgba(255, 255, 255, 0.12);
      border: 1px solid var(--color-ida);
      color: var(--color-ida);
    }}

    .toggle-btn#select-idb.active {{
      background-color: rgba(255, 255, 255, 0.12);
      border: 1px solid var(--color-idb);
      color: var(--color-idb);
    }}

    .frame-status-badge {{
      font-size: 13px;
      font-weight: 600;
      color: #e2e8f0;
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .frame-status-badge .elapsed {{
      font-size: 12px;
      font-weight: 400;
      color: var(--text-muted);
    }}

    .image-viewport {{
      position: relative;
      background-color: #000000;
      border-radius: 6px;
      overflow: hidden;
      border: 1px solid var(--border-color);
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 400px;
    }}

    .image-viewport img {{
      display: block;
      width: 100%;
      height: auto;
      max-height: 68vh;
      object-fit: contain;
    }}

    /* Playback & Scrubber Controls */
    .controls-panel {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      background-color: var(--bg-panel-subtle);
      border: 1px solid var(--border-color);
      border-radius: 6px;
      padding: 10px 14px;
    }}

    .slider-row {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}

    .slider-row input[type="range"] {{
      flex: 1;
      height: 6px;
      accent-color: var(--border-accent);
      background: var(--bg-input);
      border-radius: 3px;
      cursor: pointer;
    }}

    .nav-btn-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
    }}

    .nav-btn-group {{
      display: flex;
      gap: 6px;
    }}

    .btn {{
      background-color: var(--bg-input);
      color: var(--text-main);
      border: 1px solid var(--border-color);
      border-radius: 4px;
      padding: 5px 12px;
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      transition: all 0.15s ease;
    }}

    .btn:hover:not(:disabled) {{
      background-color: var(--bg-panel);
      border-color: #475569;
    }}

    .btn:disabled {{
      opacity: 0.4;
      cursor: not-allowed;
    }}

    .btn-primary {{
      background-color: var(--border-accent);
      border-color: var(--border-accent);
      color: #ffffff;
      font-weight: 600;
    }}

    .btn-primary:hover:not(:disabled) {{
      background-color: #2563eb;
    }}

    .jump-group {{
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      color: var(--text-muted);
    }}

    .jump-group select {{
      background-color: var(--bg-input);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 12px;
    }}

    .keyboard-hint {{
      font-size: 11px;
      color: var(--text-dim);
      text-align: right;
    }}

    /* Right Column: Evidence & Review Form */
    .review-sidebar {{
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}

    .card {{
      background-color: var(--bg-panel);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}

    .card-title {{
      font-size: 13px;
      font-weight: 700;
      color: #ffffff;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding-bottom: 6px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }}

    /* Presence Cards */
    .presence-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}

    .presence-box {{
      background-color: var(--bg-panel-subtle);
      border: 1px solid var(--border-color);
      border-radius: 6px;
      padding: 10px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-size: 12px;
    }}

    .presence-box.box-ida {{
      border-top: 3px solid var(--color-ida);
    }}

    .presence-box.box-idb {{
      border-top: 3px solid var(--color-idb);
    }}

    .presence-box-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-weight: 700;
    }}

    .status-tag {{
      display: inline-block;
      padding: 2px 6px;
      font-size: 10px;
      font-weight: 700;
      border-radius: 3px;
      text-transform: uppercase;
    }}

    .status-present {{
      background-color: rgba(16, 185, 129, 0.15);
      color: var(--accent-green);
      border: 1px solid rgba(16, 185, 129, 0.3);
    }}

    .status-absent {{
      background-color: rgba(100, 116, 139, 0.15);
      color: var(--text-muted);
      border: 1px solid rgba(100, 116, 139, 0.3);
    }}

    .meta-list {{
      display: flex;
      flex-direction: column;
      gap: 3px;
      color: var(--text-muted);
      font-size: 11px;
    }}

    .meta-list span strong {{
      color: #cbd5e1;
    }}

    /* Saved Pair Evidence Display */
    .evidence-container {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}

    .evidence-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}

    .evidence-table th, .evidence-table td {{
      padding: 6px 8px;
      text-align: left;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }}

    .evidence-table th {{
      color: var(--text-muted);
      font-weight: 500;
      width: 50%;
    }}

    .evidence-table td {{
      font-weight: 600;
      color: #ffffff;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }}

    .no-pair-box {{
      background-color: rgba(100, 116, 139, 0.08);
      border: 1px dashed #475569;
      border-radius: 6px;
      padding: 18px 14px;
      text-align: center;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}

    .no-pair-title {{
      font-size: 13px;
      font-weight: 700;
      color: #cbd5e1;
    }}

    .no-pair-sub {{
      font-size: 11px;
      color: var(--text-dim);
    }}

    /* Human Review Form */
    .form-group {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}

    .form-group label {{
      font-size: 12px;
      font-weight: 600;
      color: #cbd5e1;
    }}

    .range-inputs-row {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    .range-inputs-row input[type="number"] {{
      width: 84px;
      background-color: var(--bg-input);
      border: 1px solid var(--border-color);
      color: #ffffff;
      padding: 6px 8px;
      border-radius: 4px;
      font-size: 12px;
    }}

    .assessment-radios {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      background-color: var(--bg-panel-subtle);
      border: 1px solid var(--border-color);
      border-radius: 6px;
      padding: 8px 10px;
    }}

    .radio-label {{
      display: flex;
      align-items: center;
      gap: 8px;
      cursor: pointer;
      font-size: 12px;
      color: var(--text-main);
    }}

    .radio-label input[type="radio"] {{
      accent-color: var(--border-accent);
      cursor: pointer;
    }}

    textarea.form-textarea {{
      background-color: var(--bg-input);
      border: 1px solid var(--border-color);
      color: #ffffff;
      border-radius: 4px;
      padding: 8px;
      font-size: 12px;
      font-family: inherit;
      resize: vertical;
      min-height: 60px;
    }}

    textarea.form-textarea:focus, input:focus, select:focus {{
      outline: none;
      border-color: var(--border-accent);
    }}

    .download-section {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 4px;
    }}

    .download-notice {{
      font-size: 11px;
      color: #cbd5e1;
      background-color: rgba(56, 189, 248, 0.08);
      border-left: 3px solid var(--accent-blue);
      padding: 8px 10px;
      border-radius: 0 4px 4px 0;
      line-height: 1.4;
    }}

    .validation-alert {{
      display: none;
      font-size: 11px;
      color: #fecdd3;
      background-color: rgba(244, 63, 94, 0.15);
      border: 1px solid rgba(244, 63, 94, 0.4);
      padding: 6px 8px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <div class="app-container">

    <!-- Header & Scope Caveats -->
    <header class="review-header">
      <div class="header-top">
        <h1 class="header-title">
          <span>Candidate Vehicle Pair Review</span>
          <span class="badge badge-scope">Manual Image-Space Review</span>
        </h1>
        <div class="badges-row">
          <span class="badge badge-offline">Offline Self-Contained Tool</span>
          <span class="badge badge-ida">Track ID {id_a} (Pair ID A)</span>
          <span class="badge badge-idb">Track ID {id_b} (Pair ID B)</span>
        </div>
      </div>

      <div class="header-caveat-banner">
        <strong>Scope Notice:</strong> Manual 2D image-space review of temporary tracker IDs ({id_a} and {id_b}).
        No physical conflict conclusion, road lane inference, 3D proximity, speed, or collision risk is established.
      </div>

      <div class="header-meta-grid">
        <div><strong>Frame Interval:</strong> {start_frame}–{end_frame} ({len(frames_payload)} frames)</div>
        <div><strong>{order_note}</strong></div>
        <div><strong>Video Source:</strong> {html.escape(Path(meta_payload["video_path"]).name)}</div>
        <div><strong>Tracks Source:</strong> {html.escape(Path(meta_payload["tracks_path"]).name)}</div>
        <div><strong>Pair Observations:</strong> {html.escape(Path(meta_payload["pairs_path"]).name)}</div>
      </div>
    </header>

    <!-- Main 2-Column Review Layout -->
    <main class="main-layout">

      <!-- Left Column: Frame Viewer -->
      <section class="viewer-panel">
        <div class="toolbar-row">
          <div class="view-selectors-wrap">
            <div class="toggle-group" role="group" aria-label="Image View Toggle">
              <button id="toggle-annotated" class="toggle-btn active" type="button">Annotated</button>
              <button id="toggle-original" class="toggle-btn" type="button">Original Frame</button>
            </div>

            <div class="toggle-group" id="id-selector-group" role="group" aria-label="Display Selector">
              <button id="select-both" class="toggle-btn active" type="button" title="Show annotations for both IDs ({id_a} and {id_b})">Both IDs</button>
              <button id="select-ida" class="toggle-btn" type="button" data-track-id="{id_a}" title="Show annotations for ID A ({id_a}) only">ID A only</button>
              <button id="select-idb" class="toggle-btn" type="button" data-track-id="{id_b}" title="Show annotations for ID B ({id_b}) only">ID B only</button>
            </div>
          </div>

          <div class="frame-status-badge">
            <span id="frame-indicator">Frame {start_frame}</span>
            <span id="elapsed-indicator" class="elapsed">(0.000s)</span>
          </div>
        </div>

        <div class="image-viewport">
          <img id="main-frame-img" src="" alt="Candidate review video frame">
        </div>

        <!-- Controls & Scrubber -->
        <div class="controls-panel">
          <div class="slider-row">
            <input id="frame-scrubber" type="range" min="{start_frame}" max="{end_frame}" value="{start_frame}" step="1">
          </div>

          <div class="nav-btn-row">
            <div class="nav-btn-group">
              <button id="btn-first" class="btn" title="First Frame (|&lt;)">|&lt;</button>
              <button id="btn-prev" class="btn" title="Previous Frame (&lt;)">&lt; Prev</button>
              <button id="btn-next" class="btn" title="Next Frame (&gt;)">Next &gt;</button>
              <button id="btn-last" class="btn" title="Last Frame (&gt;|)">&gt;|</button>
            </div>

            <div class="jump-group">
              <label for="jump-select">Jump to:</label>
              <select id="jump-select"></select>
            </div>

            <div class="keyboard-hint">
              Shortcuts: &larr; / &rarr; navigate &bull; Space toggles original &bull; 1: Both, 2: ID A, 3: ID B
            </div>
          </div>
        </div>
      </section>

      <!-- Right Column: Evidence & Human Review Form -->
      <aside class="review-sidebar">

        <!-- Selected-ID Presence Card -->
        <div class="card">
          <div class="card-title">
            <span>Selected-ID Presence</span>
            <span style="font-size: 11px; font-weight: normal; color: var(--text-muted);">Current Frame</span>
          </div>

          <div class="presence-grid">
            <!-- ID A (Pair ID A) -->
            <div class="presence-box box-ida">
              <div class="presence-box-header">
                <span style="color: var(--color-ida);">ID {id_a} <span style="font-size: 11px; font-weight: normal;">(Pair ID A)</span></span>
                <span id="presence-tag-a" class="status-tag status-absent">ABSENT</span>
              </div>
              <div id="meta-list-a" class="meta-list">
                <span>Class: <strong>—</strong></span>
                <span>Conf: <strong>—</strong></span>
                <span>Box: <strong>—</strong></span>
              </div>
            </div>

            <!-- ID B (Pair ID B) -->
            <div class="presence-box box-idb">
              <div class="presence-box-header">
                <span style="color: var(--color-idb);">ID {id_b} <span style="font-size: 11px; font-weight: normal;">(Pair ID B)</span></span>
                <span id="presence-tag-b" class="status-tag status-absent">ABSENT</span>
              </div>
              <div id="meta-list-b" class="meta-list">
                <span>Class: <strong>—</strong></span>
                <span>Conf: <strong>—</strong></span>
                <span>Box: <strong>—</strong></span>
              </div>
            </div>
          </div>
        </div>

        <!-- Saved Pair Evidence Card -->
        <div class="card">
          <div class="card-title">
            <span>Saved Pair Evidence</span>
            <span id="evidence-status-tag" style="font-size: 11px; font-weight: normal; color: var(--text-muted);">Frame Observation</span>
          </div>

          <div id="evidence-body" class="evidence-container">
            <!-- Populated dynamically by updateFrame() -->
          </div>
        </div>

        <!-- Human Review Assessment Form -->
        <div class="card">
          <div class="card-title">
            <span>Human Review Assessment</span>
            <span style="font-size: 11px; color: var(--accent-amber); font-weight: 600;">Starts Unreviewed</span>
          </div>

          <form id="review-form" onsubmit="return false;" style="display: flex; flex-direction: column; gap: 12px;">

            <!-- Explicit Reviewed Frame Range -->
            <div class="form-group">
              <label>Reviewed Frame Range (zero-based inclusive):</label>
              <div class="range-inputs-row">
                <input id="rev-start-frame" type="number" min="{start_frame}" max="{end_frame}" value="{start_frame}">
                <span style="color: var(--text-muted);">&ndash;</span>
                <input id="rev-end-frame" type="number" min="{start_frame}" max="{end_frame}" value="{end_frame}">
                <button id="btn-set-full-range" class="btn" type="button" style="font-size: 11px; padding: 4px 8px;">Full Interval</button>
                <button id="btn-set-current-frame" class="btn" type="button" style="font-size: 11px; padding: 4px 8px;">Current</button>
              </div>
            </div>

            <!-- Assessment Radio Selection -->
            <div class="form-group">
              <label>Assessment:</label>
              <div class="assessment-radios">
                <label class="radio-label">
                  <input type="radio" name="assessment" value="same_vehicle">
                  <span><strong>Same vehicle</strong> (e.g. duplicate detector boxes on single vehicle)</span>
                </label>
                <label class="radio-label">
                  <input type="radio" name="assessment" value="distinct_vehicles">
                  <span><strong>Distinct vehicles</strong> (e.g. separate vehicles with perspective overlap)</span>
                </label>
                <label class="radio-label">
                  <input type="radio" name="assessment" value="uncertain">
                  <span><strong>Uncertain</strong> (e.g. visual occlusion, ambiguous association, distant)</span>
                </label>
              </div>
            </div>

            <!-- Uncertainty Notes -->
            <div class="form-group">
              <label for="rev-uncertainty">Uncertainty Notes:</label>
              <textarea id="rev-uncertainty" class="form-textarea" placeholder="Document any ambiguity, occlusions, visual limits, or reasons for uncertainty..."></textarea>
            </div>

            <!-- Supporting Observations -->
            <div class="form-group">
              <label for="rev-supporting">Supporting Observations:</label>
              <textarea id="rev-supporting" class="form-textarea" placeholder="Document supporting visual evidence (e.g. box overlap on chassis, lane markings, motion continuity)..."></textarea>
            </div>

            <div id="validation-alert" class="validation-alert"></div>

            <!-- Download Button & Notice -->
            <div class="download-section">
              <button id="btn-download-json" class="btn btn-primary" type="button">Download Review JSON</button>
              <div class="download-notice">
                <strong>Notice:</strong> Clicking "Download Review JSON" saves the assessment locally to your computer.
                This offline review does not automatically update or commit files to the repository.
              </div>
            </div>

          </form>
        </div>

      </aside>

    </main>
  </div>

  <!-- Embedded Review Payload -->
  <script id="review-payload" type="application/json">
    {safe_json_data}
  </script>

  <!-- Plain JavaScript Logic -->
  <script>
    (function() {{
      "use strict";

      // Parse embedded data safely
      const payloadEl = document.getElementById("review-payload");
      if (!payloadEl) return;
      const data = JSON.parse(payloadEl.textContent);

      const meta = data.meta;
      const frames = data.frames;
      const frameMap = new Map();
      frames.forEach((f, idx) => frameMap.set(f.frame_index, {{ frame: f, index: idx }}));

      const idA = meta.pair_ids.track_id_a;
      const idB = meta.pair_ids.track_id_b;

      // DOM Elements
      const mainImg = document.getElementById("main-frame-img");
      const toggleAnnotated = document.getElementById("toggle-annotated");
      const toggleOriginal = document.getElementById("toggle-original");
      const selectBoth = document.getElementById("select-both");
      const selectIdA = document.getElementById("select-ida");
      const selectIdB = document.getElementById("select-idb");
      const frameIndicator = document.getElementById("frame-indicator");
      const elapsedIndicator = document.getElementById("elapsed-indicator");
      const scrubber = document.getElementById("frame-scrubber");
      const jumpSelect = document.getElementById("jump-select");

      const btnFirst = document.getElementById("btn-first");
      const btnPrev = document.getElementById("btn-prev");
      const btnNext = document.getElementById("btn-next");
      const btnLast = document.getElementById("btn-last");

      const presenceTagA = document.getElementById("presence-tag-a");
      const presenceTagB = document.getElementById("presence-tag-b");
      const metaListA = document.getElementById("meta-list-a");
      const metaListB = document.getElementById("meta-list-b");
      const evidenceBody = document.getElementById("evidence-body");
      const evidenceStatusTag = document.getElementById("evidence-status-tag");

      // Form Elements
      const revStartInput = document.getElementById("rev-start-frame");
      const revEndInput = document.getElementById("rev-end-frame");
      const btnSetFullRange = document.getElementById("btn-set-full-range");
      const btnSetCurrentFrame = document.getElementById("btn-set-current-frame");
      const btnDownloadJson = document.getElementById("btn-download-json");
      const validationAlert = document.getElementById("validation-alert");
      const uncertaintyInput = document.getElementById("rev-uncertainty");
      const supportingInput = document.getElementById("rev-supporting");

      // State
      let currentFrameIdx = meta.start_frame;
      let viewMode = "annotated"; // "annotated" or "original"
      let idDisplayMode = "both"; // "both", "ida", or "idb"

      // Initialize Jump Dropdown
      frames.forEach(f => {{
        const opt = document.createElement("option");
        opt.value = f.frame_index;
        const hasEvidence = f.pair_evidence !== null;
        opt.textContent = `Frame ${{f.frame_index}} ${{hasEvidence ? '(Pair Obs)' : ''}}`;
        jumpSelect.appendChild(opt);
      }});

      function updateFrame(fIdx) {{
        currentFrameIdx = fIdx;
        const item = frameMap.get(fIdx);
        if (!item) return;
        const f = item.frame;

        // Update image based on viewMode and idDisplayMode
        if (viewMode === "original") {{
          mainImg.src = f.orig_img;
        }} else if (idDisplayMode === "ida") {{
          mainImg.src = f.annot_a_img;
        }} else if (idDisplayMode === "idb") {{
          mainImg.src = f.annot_b_img;
        }} else {{
          mainImg.src = f.annot_both_img || f.annot_img;
        }}

        // Update toolbar
        frameIndicator.textContent = `Frame ${{f.frame_index}} / ${{meta.end_frame}} (Step ${{item.index + 1}}/${{frames.length}})`;
        if (f.elapsed_from_source_start_s_str) {{
          elapsedIndicator.textContent = `(${{f.elapsed_from_source_start_s_str}}s)`;
        }} else if (f.elapsed_from_source_start_s !== null && f.elapsed_from_source_start_s !== undefined) {{
          elapsedIndicator.textContent = `(${{f.elapsed_from_source_start_s}}s)`;
        }} else {{
          elapsedIndicator.textContent = "";
        }}

        scrubber.value = f.frame_index;
        jumpSelect.value = f.frame_index;

        // Nav button states
        btnFirst.disabled = (fIdx <= meta.start_frame);
        btnPrev.disabled = (fIdx <= meta.start_frame);
        btnNext.disabled = (fIdx >= meta.end_frame);
        btnLast.disabled = (fIdx >= meta.end_frame);

        // Update ID Presence
        const pA = f.presence[String(idA)] || {{ present: false }};
        const pB = f.presence[String(idB)] || {{ present: false }};

        if (pA.present) {{
          presenceTagA.className = "status-tag status-present";
          presenceTagA.textContent = "PRESENT";
          metaListA.innerHTML = `
            <span>Role: <strong>${{escapeHtml(pA.role || "Pair ID A")}}</strong></span>
            <span>Class: <strong>${{escapeHtml(pA.class_name)}}</strong></span>
            <span>Conf: <strong>${{escapeHtml(pA.confidence_str)}}</strong></span>
            <span>Box: <strong>[${{pA.box.join(", ")}}]</strong></span>
          `;
        }} else {{
          presenceTagA.className = "status-tag status-absent";
          presenceTagA.textContent = "ABSENT";
          metaListA.innerHTML = `
            <span>Role: <strong>${{escapeHtml(pA.role || "Pair ID A")}}</strong></span>
            <span>Class: <strong>—</strong></span>
            <span>Conf: <strong>—</strong></span>
            <span>Box: <strong>—</strong></span>
          `;
        }}

        if (pB.present) {{
          presenceTagB.className = "status-tag status-present";
          presenceTagB.textContent = "PRESENT";
          metaListB.innerHTML = `
            <span>Role: <strong>${{escapeHtml(pB.role || "Pair ID B")}}</strong></span>
            <span>Class: <strong>${{escapeHtml(pB.class_name)}}</strong></span>
            <span>Conf: <strong>${{escapeHtml(pB.confidence_str)}}</strong></span>
            <span>Box: <strong>[${{pB.box.join(", ")}}]</strong></span>
          `;
        }} else {{
          presenceTagB.className = "status-tag status-absent";
          presenceTagB.textContent = "ABSENT";
          metaListB.innerHTML = `
            <span>Role: <strong>${{escapeHtml(pB.role || "Pair ID B")}}</strong></span>
            <span>Class: <strong>—</strong></span>
            <span>Conf: <strong>—</strong></span>
            <span>Box: <strong>—</strong></span>
          `;
        }}

        // Update Saved Pair Evidence with exact CSV strings and explicit ID labeling
        const ev = f.pair_evidence;
        if (ev) {{
          evidenceStatusTag.textContent = "Saved Evidence Present";
          evidenceStatusTag.style.color = "var(--accent-green)";

          evidenceBody.innerHTML = `
            <table class="evidence-table">
              <tbody>
                <tr>
                  <th>Track ID A (Ordered):</th>
                  <td style="color: var(--color-ida);">ID ${{escapeHtml(ev.track_id_a_str)}}</td>
                </tr>
                <tr>
                  <th>Track ID B (Ordered):</th>
                  <td style="color: var(--color-idb);">ID ${{escapeHtml(ev.track_id_b_str)}}</td>
                </tr>
                <tr>
                  <th>Box IoU:</th>
                  <td style="color: var(--accent-amber); font-size: 13px;">${{escapeHtml(ev.box_iou_str)}}</td>
                </tr>
                <tr>
                  <th>Center Distance:</th>
                  <td>${{escapeHtml(ev.center_distance_px_str)}} px</td>
                </tr>
                <tr>
                  <th>Horizontal Box Gap:</th>
                  <td>${{escapeHtml(ev.horizontal_box_gap_px_str)}} px</td>
                </tr>
                <tr>
                  <th>Vertical Box Gap:</th>
                  <td>${{escapeHtml(ev.vertical_box_gap_px_str)}} px</td>
                </tr>
                <tr>
                  <th>Elapsed Time:</th>
                  <td>${{escapeHtml(ev.elapsed_from_source_start_s_str)}} s</td>
                </tr>
                <tr>
                  <th>Class (Pair ID A = ${{escapeHtml(ev.track_id_a_str)}}):</th>
                  <td>${{escapeHtml(ev.class_name_a)}}</td>
                </tr>
                <tr>
                  <th>Class (Pair ID B = ${{escapeHtml(ev.track_id_b_str)}}):</th>
                  <td>${{escapeHtml(ev.class_name_b)}}</td>
                </tr>
                <tr>
                  <th>Confidence (Pair ID A = ${{escapeHtml(ev.track_id_a_str)}}):</th>
                  <td>${{escapeHtml(ev.confidence_a_str)}}</td>
                </tr>
                <tr>
                  <th>Confidence (Pair ID B = ${{escapeHtml(ev.track_id_b_str)}}):</th>
                  <td>${{escapeHtml(ev.confidence_b_str)}}</td>
                </tr>
                <tr>
                  <th>Alignment Basis:</th>
                  <td style="font-size: 11px; word-break: break-all;">${{escapeHtml(ev.alignment_basis)}}</td>
                </tr>
              </tbody>
            </table>
          `;
        }} else {{
          evidenceStatusTag.textContent = "No saved pair observation";
          evidenceStatusTag.style.color = "var(--text-dim)";

          evidenceBody.innerHTML = `
            <div class="no-pair-box">
              <div class="no-pair-title">No saved pair observation</div>
              <div class="no-pair-sub">Neither pair observation nor recalculation is applied for this frame.</div>
            </div>
          `;
        }}
      }}

      function escapeHtml(str) {{
        if (str === null || str === undefined) return "";
        return String(str)
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;")
          .replace(/'/g, "&#039;");
      }}

      // Image Mode & Display Selector
      function setIdDisplayMode(mode) {{
        idDisplayMode = mode;
        selectBoth.classList.toggle("active", mode === "both");
        selectIdA.classList.toggle("active", mode === "ida");
        selectIdB.classList.toggle("active", mode === "idb");

        // Selecting an ID display mode activates annotated view
        if (viewMode !== "annotated") {{
          viewMode = "annotated";
          toggleAnnotated.classList.add("active");
          toggleOriginal.classList.remove("active");
        }}
        updateFrame(currentFrameIdx);
      }}

      selectBoth.addEventListener("click", () => setIdDisplayMode("both"));
      selectIdA.addEventListener("click", () => setIdDisplayMode("ida"));
      selectIdB.addEventListener("click", () => setIdDisplayMode("idb"));

      toggleAnnotated.addEventListener("click", () => {{
        viewMode = "annotated";
        toggleAnnotated.classList.add("active");
        toggleOriginal.classList.remove("active");
        updateFrame(currentFrameIdx);
      }});

      toggleOriginal.addEventListener("click", () => {{
        viewMode = "original";
        toggleOriginal.classList.add("active");
        toggleAnnotated.classList.remove("active");
        updateFrame(currentFrameIdx);
      }});

      // Navigation Event Listeners
      btnFirst.addEventListener("click", () => updateFrame(meta.start_frame));
      btnLast.addEventListener("click", () => updateFrame(meta.end_frame));

      btnPrev.addEventListener("click", () => {{
        if (currentFrameIdx > meta.start_frame) {{
          updateFrame(currentFrameIdx - 1);
        }}
      }});

      btnNext.addEventListener("click", () => {{
        if (currentFrameIdx < meta.end_frame) {{
          updateFrame(currentFrameIdx + 1);
        }}
      }});

      scrubber.addEventListener("input", (e) => {{
        updateFrame(parseInt(e.target.value, 10));
      }});

      jumpSelect.addEventListener("change", (e) => {{
        updateFrame(parseInt(e.target.value, 10));
      }});

      // Keyboard Controls
      window.addEventListener("keydown", (e) => {{
        // Ignore if user is typing in form controls
        if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) {{
          return;
        }}

        if (e.key === "ArrowLeft") {{
          e.preventDefault();
          if (currentFrameIdx > meta.start_frame) updateFrame(currentFrameIdx - 1);
        }} else if (e.key === "ArrowRight") {{
          e.preventDefault();
          if (currentFrameIdx < meta.end_frame) updateFrame(currentFrameIdx + 1);
        }} else if (e.key === " " || e.code === "Space") {{
          e.preventDefault();
          if (viewMode === "annotated") {{
            toggleOriginal.click();
          }} else {{
            toggleAnnotated.click();
          }}
        }} else if (e.key === "1") {{
          e.preventDefault();
          selectBoth.click();
        }} else if (e.key === "2") {{
          e.preventDefault();
          selectIdA.click();
        }} else if (e.key === "3") {{
          e.preventDefault();
          selectIdB.click();
        }} else if (e.key === "Home") {{
          e.preventDefault();
          updateFrame(meta.start_frame);
        }} else if (e.key === "End") {{
          e.preventDefault();
          updateFrame(meta.end_frame);
        }}
      }});

      // Form Quick-Set Buttons
      btnSetFullRange.addEventListener("click", () => {{
        revStartInput.value = meta.start_frame;
        revEndInput.value = meta.end_frame;
      }});

      btnSetCurrentFrame.addEventListener("click", () => {{
        revStartInput.value = currentFrameIdx;
        revEndInput.value = currentFrameIdx;
      }});

      // Download Review JSON
      btnDownloadJson.addEventListener("click", () => {{
        validationAlert.style.display = "none";
        validationAlert.textContent = "";

        const selectedRadio = document.querySelector('input[name="assessment"]:checked');
        if (!selectedRadio) {{
          validationAlert.textContent = "Please select an assessment conclusion (same vehicle / distinct vehicles / uncertain) before downloading.";
          validationAlert.style.display = "block";
          return;
        }}

        // Strict non-empty integer validation (replaces parseInt)
        const rawStart = revStartInput.value.trim();
        const rawEnd = revEndInput.value.trim();

        if (!rawStart || !rawEnd || !/^-?\\d+$/.test(rawStart) || !/^-?\\d+$/.test(rawEnd)) {{
          validationAlert.textContent = "Please enter valid, non-empty integers for the reviewed frame range.";
          validationAlert.style.display = "block";
          return;
        }}

        const rStart = Number(rawStart);
        const rEnd = Number(rawEnd);

        if (!Number.isSafeInteger(rStart) || !Number.isSafeInteger(rEnd)) {{
          validationAlert.textContent = "Reviewed frame bounds must be safe integers.";
          validationAlert.style.display = "block";
          return;
        }}

        // Enforce: meta.start_frame <= reviewed start <= reviewed end <= meta.end_frame
        if (rStart < meta.start_frame || rStart > rEnd || rEnd > meta.end_frame) {{
          validationAlert.textContent = `Reviewed frame range must satisfy: ${{meta.start_frame}} <= reviewed start <= reviewed end <= ${{meta.end_frame}} (got ${{rStart}}..${{rEnd}}).`;
          validationAlert.style.display = "block";
          return;
        }}

        const assessmentValue = selectedRadio.value;
        const uncertaintyNotes = uncertaintyInput.value.trim();
        const supportingObs = supportingInput.value.trim();
        const reviewTimestamp = new Date().toISOString();

        const reviewExport = {{
          schema_version: meta.schema_version,
          review_timestamp: reviewTimestamp,
          input_paths: {{
            video: meta.video_path,
            tracks: meta.tracks_path,
            pairs: meta.pairs_path
          }},
          selected_interval: {{
            start_frame: meta.start_frame,
            end_frame: meta.end_frame
          }},
          selected_ids: meta.selected_ids,
          pair_ids: {{
            track_id_a: meta.pair_ids.track_id_a,
            track_id_b: meta.pair_ids.track_id_b
          }},
          reviewed_range: {{
            start_frame: rStart,
            end_frame: rEnd
          }},
          assessment: assessmentValue,
          uncertainty_notes: uncertaintyNotes,
          supporting_observations: supportingObs,
          notes: {{
            uncertainty_notes: uncertaintyNotes,
            supporting_observations: supportingObs
          }},
          caveats: [
            "Manual image-space review of temporary tracking associations.",
            "Track IDs do not represent verified unique physical vehicles.",
            "All bounding box observations are 2D image coordinates only.",
            "No roadway lanes, physical proximity, speeds, Time-To-Collision (TTC), or physical conflict conclusion."
          ]
        }};

        const jsonBlob = new Blob([JSON.stringify(reviewExport, null, 2)], {{ type: "application/json" }});
        const url = URL.createObjectURL(jsonBlob);
        const downloadAnchor = document.createElement("a");
        downloadAnchor.href = url;
        downloadAnchor.download = `candidate_review_ids_${{idA}}_${{idB}}_frames_${{rStart}}_${{rEnd}}.json`;
        document.body.appendChild(downloadAnchor);
        downloadAnchor.click();
        document.body.removeChild(downloadAnchor);
        URL.revokeObjectURL(url);
      }});

      // Initial Frame Render
      updateFrame(meta.start_frame);

    }})();
  </script>
</body>
</html>
"""
    return html_template


# ---------------------------------------------------------------------------
# Main Execution Workflow
# ---------------------------------------------------------------------------

def run_build_review(args: argparse.Namespace) -> int:
    """Execute validation and generation of offline candidate review HTML."""
    # Guard 1: Refuse existing output file
    if args.output.exists():
        print(
            f"ERROR: Destination HTML output file already exists: '{args.output}'. "
            "Refusing to overwrite existing file.",
            file=sys.stderr,
        )
        return 1

    # Guard 2: Parse and validate IDs
    try:
        id_a, id_b = parse_ids_argument(args.ids)
    except ValueError as exc:
        print(f"ERROR: Invalid --ids: {exc}", file=sys.stderr)
        return 1

    # Guard 3: Validate frame bounds and interval limit
    try:
        num_frames = validate_frame_bounds(args.start_frame, args.end_frame)
    except ValueError as exc:
        print(f"ERROR: Invalid frame bounds: {exc}", file=sys.stderr)
        return 1

    pair_id_a = min(id_a, id_b)
    pair_id_b = max(id_a, id_b)

    print("=== Candidate Vehicle Pair Review Builder ===")
    print(f"Input video      : {args.video.resolve()}")
    print(f"Tracks JSONL     : {args.tracks.resolve()}")
    print(f"Pairs CSV        : {args.pairs.resolve()}")
    print(f"Selected frames  : {args.start_frame}..{args.end_frame} ({num_frames} frames inclusive)")
    print(f"Target IDs       : {id_a} and {id_b} (Ordered Pair: ID A = {pair_id_a}, ID B = {pair_id_b})")
    print(f"Output file      : {args.output.resolve()}")
    print()

    # Step 1: Sequential video decoding from frame 0 up through end_frame
    print(f"Decoding video sequentially from frame 0 through {args.end_frame}...")
    try:
        retained_frames, video_meta = decode_selected_video_frames(
            args.video, args.start_frame, args.end_frame
        )
    except Exception as exc:
        print(f"ERROR reading video frames: {exc}", file=sys.stderr)
        return 2

    # Step 2: Load and validate consecutive tracks JSONL and track dictionary structure
    print(f"Validating consecutive track records in '{args.tracks.name}'...")
    try:
        tracks_by_frame = load_and_validate_tracks(
            args.tracks, args.start_frame, args.end_frame
        )
    except Exception as exc:
        print(f"ERROR reading tracks JSONL: {exc}", file=sys.stderr)
        return 2

    # Verify both IDs appear in tracks within interval
    observed_track_ids = set()
    for f_idx in range(args.start_frame, args.end_frame + 1):
        for trk in tracks_by_frame[f_idx]:
            observed_track_ids.add(int(trk["track_id"]))

    if pair_id_a not in observed_track_ids:
        print(
            f"ERROR: Track ID {pair_id_a} (Pair ID A) was never observed in tracks within "
            f"interval [{args.start_frame}, {args.end_frame}].",
            file=sys.stderr,
        )
        return 2
    if pair_id_b not in observed_track_ids:
        print(
            f"ERROR: Track ID {pair_id_b} (Pair ID B) was never observed in tracks within "
            f"interval [{args.start_frame}, {args.end_frame}].",
            file=sys.stderr,
        )
        return 2

    # Step 3: Load and validate saved pair observations
    print(f"Validating saved pair observations in '{args.pairs.name}'...")
    try:
        pair_evidence_by_frame, frame_timing = load_and_validate_pairs(
            args.pairs, args.start_frame, args.end_frame, (pair_id_a, pair_id_b)
        )
    except Exception as exc:
        print(f"ERROR reading pair observations CSV: {exc}", file=sys.stderr)
        return 2

    if not pair_evidence_by_frame:
        print(
            f"ERROR: No saved pair observation found for IDs {pair_id_a} and {pair_id_b} within selected "
            f"frame interval [{args.start_frame}, {args.end_frame}]. Required candidate pair evidence is missing.",
            file=sys.stderr,
        )
        return 2

    print(f"✓ Found {len(pair_evidence_by_frame)} saved pair observation(s) across selected frames.")

    # Step 3b: Confirm agreement between saved pair rows and tracks (without recalculating geometry)
    print("Validating agreement between saved pair observations and track records...")
    try:
        validate_pair_and_tracks_agreement(pair_evidence_by_frame, tracks_by_frame)
    except Exception as exc:
        print(f"ERROR verifying pair observations against tracks: {exc}", file=sys.stderr)
        return 2

    print("✓ Confirmed both IDs, classes, and confidences match between pair observations and tracks.")

    # Step 4: Annotate tracks, maintain trails, and package frame payload
    print("Processing frame visual annotations and image encoding...")
    try:
        frames_payload = process_review_frames(
            retained_frames,
            tracks_by_frame,
            pair_evidence_by_frame,
            frame_timing,
            args.start_frame,
            args.end_frame,
            (pair_id_a, pair_id_b),
        )
    except Exception as exc:
        print(f"ERROR during frame annotation and encoding: {exc}", file=sys.stderr)
        return 2

    # Build color palette mapping for selected IDs
    color_a_css = bgr_to_css_hex(get_track_color(pair_id_a))
    color_b_css = bgr_to_css_hex(get_track_color(pair_id_b))

    meta_payload = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "video_path": str(args.video.resolve()),
        "tracks_path": str(args.tracks.resolve()),
        "pairs_path": str(args.pairs.resolve()),
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "selected_ids": [id_a, id_b],
        "pair_ids": {
            "track_id_a": pair_id_a,
            "track_id_b": pair_id_b,
        },
        "track_colors": {
            str(pair_id_a): color_a_css,
            str(pair_id_b): color_b_css,
        },
        "width": video_meta["width"],
        "height": video_meta["height"],
        "fps": video_meta["fps"],
    }

    # Step 5: Render HTML and write exclusively (refuse overwrite)
    print("Generating self-contained offline HTML content...")
    html_content = build_offline_html(meta_payload, frames_payload)

    # Ensure parent output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(args.output, "x", encoding="utf-8") as out_f:
            out_f.write(html_content)
    except FileExistsError:
        print(
            f"ERROR: Destination HTML file already exists: '{args.output}'. "
            "Refusing to overwrite existing file.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"ERROR writing HTML file '{args.output}': {exc}", file=sys.stderr)
        return 2

    file_size_kb = args.output.stat().st_size / 1024.0
    print()
    print("=== Candidate Review Generation Complete ===")
    print(f"Output HTML file : {args.output.resolve()} ({file_size_kb:.1f} KB)")
    print(f"Embedded frames  : {len(frames_payload)} frames ({args.start_frame}..{args.end_frame})")
    print(f"Target IDs       : {id_a} and {id_b} (ID A = {pair_id_a}, ID B = {pair_id_b})")
    print(f"Saved pair rows  : {len(pair_evidence_by_frame)}")
    print("Scope Caveat     : Manual image-space review; temporary IDs; no physical-conflict conclusion.")
    return 0


def main() -> None:
    args = parse_args()
    sys.exit(run_build_review(args))


if __name__ == "__main__":
    main()
