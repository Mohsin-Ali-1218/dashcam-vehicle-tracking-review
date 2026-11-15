"""
inspect_daylight_frame_alignment.py — Compare decoded image content between HEVC and MP4.

Sequentially decodes the raw daylight HEVC file and the unannotated preview MP4,
resizes frames to 160x120 grayscale, and compares pixel content across a local
window of offsets (-2, -1, 0, +1, +2) using mean absolute difference (MAD).

Scope & Boundaries:
- Uses installed OpenCV and NumPy only.
- Bounded decoding: decodes sequentially up to 1,201 frames per file, requiring
  exactly 1,200 frames each and matching original frame dimensions.
- Bounded memory: frames resized to 160x120 8-bit grayscale for comparison.
- Evaluates same-index differences, best candidate offsets, ties, and margins.
- Important Caveats:
  - This is local image-similarity evidence within a +/-2-frame window; it is NOT
    proof against every possible conversion alteration or non-local artifact.
  - Lossy re-encoding (e.g. H.264 compression, color-space conversion) naturally
    produces non-zero pixel differences even under exact 1:1 frame alignment.
  - Do NOT infer capture timing, sensor frame rate, or camera calibration from
    image pixel comparisons.
- Read-only execution: writes no media files or reports; modifies no existing files.
- Releases all OpenCV video capture resources reliably in finally blocks.

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
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------

DEFAULT_HEVC_PATH = Path("data/raw/daylight-2018-07-29-segment-7.hevc")
DEFAULT_MP4_PATH = Path("data/preview/daylight-2018-07-29-segment-7.mp4")

EXPECTED_FRAME_COUNT = 1200
MAX_DECODE_FRAMES = 1201  # Bounded ceiling to detect over-length inputs
RESIZE_WIDTH = 160
RESIZE_HEIGHT = 120
OFFSETS = (-2, -1, 0, 1, 2)
TIE_TOLERANCE = 1e-5


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare decoded image content between daylight HEVC and MP4 "
            "across local offsets (-2 to +2) using OpenCV and NumPy."
        ),
        epilog=(
            "Note: This tool provides local image-similarity evidence within a "
            "+/-2-frame window, not proof against every possible conversion alteration. "
            "Do not infer capture timing or calibration from pixel comparisons."
        ),
    )
    parser.add_argument(
        "--hevc",
        type=Path,
        default=DEFAULT_HEVC_PATH,
        help=f"Path to input daylight HEVC file (default: {DEFAULT_HEVC_PATH}).",
    )
    parser.add_argument(
        "--mp4",
        type=Path,
        default=DEFAULT_MP4_PATH,
        help=f"Path to input daylight MP4 file (default: {DEFAULT_MP4_PATH}).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Video Decoding & Preprocessing
# ---------------------------------------------------------------------------

def decode_and_preprocess(
    video_path: Path,
    label: str,
    expected_count: int = EXPECTED_FRAME_COUNT,
    max_decode: int = MAX_DECODE_FRAMES,
    target_size: tuple[int, int] = (RESIZE_WIDTH, RESIZE_HEIGHT),
) -> tuple[list[np.ndarray], tuple[int, int]]:
    """
    Sequentially decode up to max_decode frames, convert to grayscale, and resize.
    Returns (list_of_160x120_uint8_frames, (orig_width, orig_height)).
    """
    if not video_path.is_file():
        raise FileNotFoundError(f"{label} file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {label} video: {video_path}")

    frames: list[np.ndarray] = []
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        for idx in range(max_decode):
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if idx == 0 and (orig_w <= 0 or orig_h <= 0):
                orig_h, orig_w = frame.shape[:2]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
            frames.append(resized)
    finally:
        cap.release()

    actual_count = len(frames)
    if actual_count != expected_count:
        raise ValueError(
            f"{label} frame count mismatch: decoded {actual_count} frames, "
            f"expected exactly {expected_count} (decode bound was {max_decode})."
        )

    return frames, (orig_w, orig_h)


# ---------------------------------------------------------------------------
# Alignment Comparison Logic
# ---------------------------------------------------------------------------

def compute_alignment_records(
    hevc_frames: list[np.ndarray],
    mp4_frames: list[np.ndarray],
    offsets: tuple[int, ...] = OFFSETS,
) -> list[dict[str, Any]]:
    """
    For each MP4 frame, calculate mean absolute pixel difference against HEVC frames
    at specified offsets (-2, -1, 0, +1, +2) where available.
    """
    num_frames = len(mp4_frames)
    records: list[dict[str, Any]] = []

    for i in range(num_frames):
        mp4_f = mp4_frames[i].astype(np.float32)
        diffs: dict[int, float] = {}

        for off in offsets:
            j = i + off
            if 0 <= j < len(hevc_frames):
                hevc_f = hevc_frames[j].astype(np.float32)
                mad = float(np.mean(np.abs(mp4_f - hevc_f)))
                diffs[off] = mad

        same_diff = diffs[0]
        min_diff = min(diffs.values())

        # Determine all candidate offsets achieving minimum difference within tolerance
        best_offsets = [off for off, val in diffs.items() if abs(val - min_diff) <= TIE_TOLERANCE]

        is_zero_best = (best_offsets == [0])
        is_zero_tied = (0 in best_offsets and len(best_offsets) > 1)

        # Alternative candidate differences (excluding offset 0)
        alt_diffs = {off: val for off, val in diffs.items() if off != 0}
        if alt_diffs:
            best_alt_off = min(alt_diffs, key=alt_diffs.get)  # type: ignore[arg-type]
            best_alt_diff = alt_diffs[best_alt_off]
            margin = best_alt_diff - same_diff
        else:
            best_alt_off = 0
            best_alt_diff = same_diff
            margin = 0.0

        records.append({
            "frame_index": i,
            "same_diff": same_diff,
            "min_diff": min_diff,
            "best_offsets": best_offsets,
            "is_zero_best": is_zero_best,
            "is_zero_tied": is_zero_tied,
            "best_alt_offset": best_alt_off,
            "best_alt_diff": best_alt_diff,
            "margin": margin,
            "all_diffs": diffs,
        })

    return records


# ---------------------------------------------------------------------------
# Reporting & Output
# ---------------------------------------------------------------------------

def print_alignment_report(
    hevc_path: Path,
    mp4_path: Path,
    hevc_dim: tuple[int, int],
    mp4_dim: tuple[int, int],
    records: list[dict[str, Any]],
) -> None:
    """Print a concise summary and the 20 weakest same-index matches."""
    total_frames = len(records)
    same_diffs = [r["same_diff"] for r in records]
    margins = [r["margin"] for r in records]

    strictly_best_zero_count = sum(1 for r in records if r["is_zero_best"])
    tied_zero_count = sum(1 for r in records if r["is_zero_tied"])
    other_strictly_best_count = total_frames - strictly_best_zero_count - tied_zero_count

    # Tally how often each offset achieves the minimal difference
    offset_win_counts = {off: 0 for off in OFFSETS}
    for r in records:
        for off in r["best_offsets"]:
            offset_win_counts[off] += 1

    print("================================================================================")
    print("Daylight Frame Alignment Inspection Report (HEVC vs. MP4)")
    print("================================================================================")
    print(f"HEVC source stream : {hevc_path.resolve()}")
    print(f"  Decoded frames   : {total_frames} (bound checked: exactly {total_frames})")
    print(f"  Dimensions       : {hevc_dim[0]}x{hevc_dim[1]} px")
    print(f"MP4 preview video  : {mp4_path.resolve()}")
    print(f"  Decoded frames   : {total_frames} (bound checked: exactly {total_frames})")
    print(f"  Dimensions       : {mp4_dim[0]}x{mp4_dim[1]} px")
    print("Comparison format  : 160x120 grayscale, Mean Absolute Pixel Difference (MAD)")
    print("Search window      : offsets in {-2, -1, 0, +1, +2}")
    print()

    print("--- Scope & Evidence Caveats ---")
    print("1. Local Evidence Only: This comparison provides local image-similarity evidence")
    print("   within a +/-2-frame window; it is NOT proof against every possible conversion")
    print("   alteration or non-local artifact.")
    print("2. Lossy Compression: Lossy H.264 re-encoding (quantization, color-space transform)")
    print("   inherently causes non-zero pixel differences even under exact 1:1 frame alignment.")
    print("3. Timing Attribution: Do NOT infer source capture timing, sensor frame rate, or")
    print("   camera calibration from image pixel comparisons.")
    print()

    print("--- Alignment Match Summary ---")
    print(f"Total frames evaluated             : {total_frames:,}")
    pct_best_zero = (strictly_best_zero_count / total_frames) * 100.0
    print(f"Same-index (offset 0) strictly best: {strictly_best_zero_count:,} ({pct_best_zero:.2f}%)")
    pct_tied_zero = (tied_zero_count / total_frames) * 100.0
    print(f"Same-index (offset 0) tied for best: {tied_zero_count:,} ({pct_tied_zero:.2f}%)")
    pct_other = (other_strictly_best_count / total_frames) * 100.0
    print(f"Alternative offset strictly better : {other_strictly_best_count:,} ({pct_other:.2f}%)")
    print()

    print("--- Offset Candidate Win Frequencies (including ties) ---")
    for off in OFFSETS:
        sign = "+" if off > 0 else ""
        cnt = offset_win_counts[off]
        pct = (cnt / total_frames) * 100.0
        print(f"  Offset {sign}{off:2d} : {cnt:5d} frames ({pct:6.2f}%)")
    print()

    print("--- Same-Index Pixel Difference (MAD, range [0, 255]) ---")
    print(f"  Min difference    : {np.min(same_diffs):.4f}")
    print(f"  Median difference : {np.median(same_diffs):.4f}")
    print(f"  Mean difference   : {np.mean(same_diffs):.4f}")
    print(f"  Max difference    : {np.max(same_diffs):.4f}")
    print()

    print("--- Separation Margin (Nearest Alternative Diff - Same-Index Diff) ---")
    print(f"  Min margin        : {np.min(margins):.4f}")
    print(f"  Median margin     : {np.median(margins):.4f}")
    print(f"  Mean margin       : {np.mean(margins):.4f}")
    print(f"  Max margin        : {np.max(margins):.4f}")
    print()

    # Identify the 20 weakest same-index matches (largest same-index MAD)
    weakest_matches = sorted(records, key=lambda r: r["same_diff"], reverse=True)[:20]

    print("--- 20 Weakest Same-Index Matches (Highest Pixel Discrepancy) ---")
    print("Frame | Same MAD (0) | Best Offset(s) | Best Alt (Off) | Margin (Alt - Same)")
    print("------+--------------+----------------+----------------+--------------------")
    for r in weakest_matches:
        idx = r["frame_index"]
        same_d = r["same_diff"]
        best_offs = ",".join(f"{o:+d}" if o != 0 else "0" for o in r["best_offsets"])
        alt_o = f"{r['best_alt_offset']:+d}"
        alt_d = r["best_alt_diff"]
        margin = r["margin"]
        print(
            f"{idx:5d} | {same_d:12.4f} | {best_offs:14s} | "
            f"{alt_d:7.4f} ({alt_o:2s}) | {margin:+18.4f}"
        )
    print("================================================================================")


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def inspect_daylight_frame_alignment(
    hevc_path: Path = DEFAULT_HEVC_PATH,
    mp4_path: Path = DEFAULT_MP4_PATH,
) -> int:
    """
    Execute daylight HEVC vs. MP4 frame alignment inspection.
    Returns 0 on success, non-zero on failure.
    """
    print(f"Decoding HEVC: {hevc_path} ...")
    try:
        hevc_frames, hevc_dim = decode_and_preprocess(hevc_path, label="HEVC")
    except Exception as exc:
        print(f"ERROR: Failed decoding HEVC video: {exc}", file=sys.stderr)
        return 1

    print(f"Decoding MP4 : {mp4_path} ...")
    try:
        mp4_frames, mp4_dim = decode_and_preprocess(mp4_path, label="MP4")
    except Exception as exc:
        print(f"ERROR: Failed decoding MP4 video: {exc}", file=sys.stderr)
        return 1

    # Validate matching original dimensions
    if hevc_dim != mp4_dim:
        print(
            f"ERROR: Frame dimension mismatch: HEVC is {hevc_dim[0]}x{hevc_dim[1]} px, "
            f"MP4 is {mp4_dim[0]}x{mp4_dim[1]} px.",
            file=sys.stderr,
        )
        return 2

    print("Comparing frame content across offsets {-2, -1, 0, +1, +2} ...")
    records = compute_alignment_records(hevc_frames, mp4_frames, offsets=OFFSETS)

    print()
    print_alignment_report(hevc_path, mp4_path, hevc_dim, mp4_dim, records)
    return 0


def main() -> int:
    args = parse_args()
    return inspect_daylight_frame_alignment(
        hevc_path=args.hevc,
        mp4_path=args.mp4,
    )


if __name__ == "__main__":
    sys.exit(main())
