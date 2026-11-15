"""
fetch_daylight_pose.py — Bounded retrieval of daylight segment global_pose arrays.

Inspects raw_data/Chunk_1.zip on Hugging Face using RangeReader and retrieves
only the 5 global_pose array members for daylight segment:
  b0c9d2329ad1606b|2018-07-29--11-17-20/7/

Safety & budget limits:
- 2,097,152 bytes (2 MiB) cumulative HTTP response bytes across directory and members.
- 32 maximum HTTP Range requests.
- 20.0 seconds per-request timeout.
- No retries or automatic budget increases.
- RangeReader rejects non-206 responses and validates range boundaries.
- Member validation: unencrypted, <= 128 KiB uncompressed, CRC32 verified via zipfile.
- Arrays loaded with numpy.load(..., allow_pickle=False), requiring real numeric
  ndarrays (rejecting object, string, boolean and complex dtypes) and exact shapes:
    frame_times: (1200,)
    frame_gps_times: (1200, 2)
    frame_positions: (1200, 3)
    frame_orientations: (1200, 4)
    frame_velocities: (1200, 3)
  On shape/count mismatch, reports actual vs expected and stops without saving
  (no reshaping, truncation, or interpolation).
- Finite-value and timestamp-monotonicity validation.
- Compares local HEVC byte size and CRC32 with archive video.hevc central directory entry
  WITHOUT downloading the video body.
- Manifest includes provenance caveat on decoded-frame correspondence and calibration.
- Destination data/raw/daylight-2018-07-29-segment-7-pose/ must not already exist.

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
import hashlib
import io
import json
import sys
import zlib
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

# Import RangeReader and compression table from existing local script
sys.path.insert(0, str(Path(__file__).parent))
from inspect_remote_zip import COMPRESSION_NAMES, RangeReader  # noqa: E402

# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------

ARCHIVE_URL = (
    "https://huggingface.co/datasets/commaai/comma2k19"
    "/resolve/main/raw_data/Chunk_1.zip"
)

SEGMENT_PREFIX = "Chunk_1/b0c9d2329ad1606b|2018-07-29--11-17-20/7/"

POSE_MEMBERS = (
    "frame_times",
    "frame_gps_times",
    "frame_positions",
    "frame_orientations",
    "frame_velocities",
)

EXPECTED_SHAPES: dict[str, tuple[int, ...]] = {
    "frame_times": (1200,),
    "frame_gps_times": (1200, 2),
    "frame_positions": (1200, 3),
    "frame_orientations": (1200, 4),
    "frame_velocities": (1200, 3),
}

VIDEO_MEMBER_NAME = "video.hevc"

DEFAULT_DEST_DIR = Path("data/raw/daylight-2018-07-29-segment-7-pose")
DEFAULT_LOCAL_HEVC = Path("data/raw/daylight-2018-07-29-segment-7.hevc")

MAX_BYTES = 2_097_152  # 2 MiB cumulative HTTP response bytes
MAX_REQUESTS = 32      # Max Range requests
TIMEOUT = 20.0         # Seconds per request
MAX_MEMBER_SIZE = 128 * 1024  # 128 KiB uncompressed limit per pose member


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_file_crc32_and_size(path: Path) -> tuple[int, int]:
    """Compute CRC-32 and file size in bytes for a local file."""
    crc = 0
    size = 0
    with path.open("rb") as f:
        while chunk := f.read(65536):
            crc = zlib.crc32(chunk, crc)
            size += len(chunk)
    return crc, size


def analyze_array(name: str, arr: np.ndarray) -> dict[str, Any]:
    """Inspect shape, dtype, and finite values for an array."""
    summary: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "all_finite": bool(np.all(np.isfinite(arr))),
    }

    if name == "frame_times":
        count = int(arr.size)
        summary["count"] = count
        if count > 1:
            diffs = np.diff(arr)
            strictly_monotonic = bool(np.all(diffs > 0))
            span = float(arr[-1] - arr[0])
            min_interval = float(np.min(diffs))
            median_interval = float(np.median(diffs))
            max_interval = float(np.max(diffs))
        else:
            strictly_monotonic = True
            span = 0.0
            min_interval = 0.0
            median_interval = 0.0
            max_interval = 0.0

        summary["strictly_monotonic"] = strictly_monotonic
        summary["span_seconds"] = span
        summary["interval_min_seconds"] = min_interval
        summary["interval_median_seconds"] = median_interval
        summary["interval_max_seconds"] = max_interval

    return summary


# ---------------------------------------------------------------------------
# Core Retrieval & Validation Routine
# ---------------------------------------------------------------------------

def fetch_daylight_pose(
    archive_url: str = ARCHIVE_URL,
    dest_dir: Path = DEFAULT_DEST_DIR,
    local_hevc_path: Path = DEFAULT_LOCAL_HEVC,
    max_bytes: int = MAX_BYTES,
    max_requests: int = MAX_REQUESTS,
    timeout: float = TIMEOUT,
) -> int:
    """
    Retrieve and validate pose members from Chunk_1.zip.
    Returns 0 on success, non-zero on error.
    """
    print(f"Archive URL        : {archive_url}")
    print(f"Target segment     : {SEGMENT_PREFIX}")
    print(f"Destination dir    : {dest_dir}")
    print(f"Budget limits      : {max_bytes:,} B, {max_requests} requests, {timeout:.1f}s timeout")
    print()

    # 1. Enforce destination refusal: must not exist prior to run
    if dest_dir.exists():
        print(f"ERROR: Destination directory already exists: {dest_dir}", file=sys.stderr)
        print("Refusing to overwrite existing destination.", file=sys.stderr)
        return 1

    # 2. Check local daylight HEVC exists for comparison
    if not local_hevc_path.is_file():
        print(f"ERROR: Local daylight HEVC file not found: {local_hevc_path}", file=sys.stderr)
        return 1

    # Initialize RangeReader with exact limits
    reader = RangeReader(
        url=archive_url,
        max_bytes=max_bytes,
        max_requests=max_requests,
        timeout=timeout,
    )

    try:
        zf = zipfile.ZipFile(reader)
    except zipfile.BadZipFile as exc:
        print(f"ERROR: zipfile could not read archive central directory: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"ERROR: RangeReader failure during directory read: {exc}", file=sys.stderr)
        return 3

    with zf:
        # Build map of members in segment
        segment_entries: dict[str, zipfile.ZipInfo] = {}
        for info in zf.infolist():
            if info.filename.startswith(SEGMENT_PREFIX):
                sub_path = info.filename[len(SEGMENT_PREFIX):]
                segment_entries[sub_path] = info

        # 3. Verify video.hevc central directory entry and compare with local file
        video_sub_path = VIDEO_MEMBER_NAME
        if video_sub_path not in segment_entries:
            print(f"ERROR: Video entry '{SEGMENT_PREFIX}{video_sub_path}' not found in archive directory.", file=sys.stderr)
            return 4

        video_info = segment_entries[video_sub_path]
        print(f"Found archive video entry: {video_info.filename}")
        print(f"  Archive uncompressed size : {video_info.file_size:,} bytes")
        print(f"  Archive CRC-32            : 0x{video_info.CRC:08x}")

        local_crc, local_size = compute_file_crc32_and_size(local_hevc_path)
        print(f"Local daylight HEVC ({local_hevc_path}):")
        print(f"  Local size                : {local_size:,} bytes")
        print(f"  Local CRC-32              : 0x{local_crc:08x}")

        if local_size != video_info.file_size:
            print(f"ERROR: Local HEVC size ({local_size}) does not match archive video.hevc ({video_info.file_size})", file=sys.stderr)
            return 5

        if local_crc != video_info.CRC:
            print(f"ERROR: Local HEVC CRC-32 (0x{local_crc:08x}) does not match archive video.hevc (0x{video_info.CRC:08x})", file=sys.stderr)
            return 5

        print("  ✓ Local HEVC size and CRC-32 exactly match archive video.hevc entry.")
        print()

        # 4. Validate and retrieve each pose member
        pose_results: dict[str, dict[str, Any]] = {}
        retrieved_bytes: dict[str, bytes] = {}

        for member_name in POSE_MEMBERS:
            sub_path = f"global_pose/{member_name}"
            if sub_path not in segment_entries:
                print(f"ERROR: Required pose member '{SEGMENT_PREFIX}{sub_path}' not found in archive directory.", file=sys.stderr)
                return 6

            info = segment_entries[sub_path]

            # Require exact path, unencrypted, and size bounds
            if bool(info.flag_bits & 0x1):
                print(f"ERROR: Member '{info.filename}' is encrypted. Refusing extraction.", file=sys.stderr)
                return 7

            if info.file_size > MAX_MEMBER_SIZE:
                print(f"ERROR: Member '{info.filename}' uncompressed size {info.file_size} exceeds {MAX_MEMBER_SIZE} limit.", file=sys.stderr)
                return 8

            print(f"Reading '{member_name}':")
            print(f"  Archive path      : {info.filename}")
            print(f"  Compressed size   : {info.compress_size:,} bytes")
            print(f"  Uncompressed size : {info.file_size:,} bytes")
            print(f"  Compression       : {COMPRESSION_NAMES.get(info.compress_type, str(info.compress_type))}")
            print(f"  Expected CRC-32   : 0x{info.CRC:08x}")

            # Read via zipfile to trigger ZipFile's CRC validation
            try:
                with zf.open(info) as member_file:
                    raw_data = member_file.read()
            except zipfile.BadZipFile as exc:
                print(f"ERROR: CRC or decompression error on member '{info.filename}': {exc}", file=sys.stderr)
                return 9
            except RuntimeError as exc:
                print(f"ERROR: RangeReader budget/network failure reading member '{info.filename}': {exc}", file=sys.stderr)
                return 10

            # Double-check CRC & length in memory
            computed_crc = zlib.crc32(raw_data)
            if computed_crc != info.CRC:
                print(f"ERROR: Computed CRC-32 0x{computed_crc:08x} does not match 0x{info.CRC:08x} for '{member_name}'.", file=sys.stderr)
                return 11

            if len(raw_data) != info.file_size:
                print(f"ERROR: Decompressed length {len(raw_data)} does not match expected {info.file_size} for '{member_name}'.", file=sys.stderr)
                return 12

            sha256 = hashlib.sha256(raw_data).hexdigest()

            # Load array using numpy.load with allow_pickle=False
            try:
                buf = io.BytesIO(raw_data)
                arr = np.load(buf, allow_pickle=False)
            except Exception as exc:
                print(f"ERROR: Failed to load numpy array from '{member_name}': {exc}", file=sys.stderr)
                return 13

            # Require real numeric NumPy ndarray
            if not isinstance(arr, np.ndarray):
                print(f"ERROR: Loaded object for '{member_name}' is not a numpy.ndarray: {type(arr)}", file=sys.stderr)
                return 13

            # Reject object, string, boolean and complex dtypes
            if arr.dtype.kind in ("O", "b", "c", "U", "S") or arr.dtype.kind not in ("f", "i", "u"):
                print(
                    f"ERROR: Array '{member_name}' has non-real-numeric dtype '{arr.dtype}' (kind '{arr.dtype.kind}'). "
                    "Object, string, boolean and complex dtypes are rejected.",
                    file=sys.stderr,
                )
                return 14

            # Require exact shape and count; stop without saving on mismatch (no reshape/truncate/interpolate)
            expected_shape = EXPECTED_SHAPES.get(member_name)
            if expected_shape is None:
                print(f"ERROR: Unknown pose member '{member_name}'.", file=sys.stderr)
                return 15

            if arr.shape != expected_shape:
                print(
                    f"ERROR: Shape/count mismatch for '{member_name}': "
                    f"actual shape {arr.shape} (count {arr.shape[0]}) != expected shape {expected_shape} (count {expected_shape[0]}).",
                    file=sys.stderr,
                )
                print(
                    "Stopping without saving; refusing to reshape, truncate, or interpolate.",
                    file=sys.stderr,
                )
                return 15

            array_summary = analyze_array(member_name, arr)
            if not array_summary["all_finite"]:
                print(f"ERROR: Non-finite values (NaN/Inf) detected in '{member_name}'.", file=sys.stderr)
                return 16

            if member_name == "frame_times":
                if not array_summary["strictly_monotonic"]:
                    print("ERROR: frame_times array is not strictly monotonic.", file=sys.stderr)
                    return 17
                print(f"  Array summary     : count={array_summary['count']}, shape={array_summary['shape']}, dtype={array_summary['dtype']}")
                print(f"  Timing validation : monotonic=True, span={array_summary['span_seconds']:.6f}s")
                print(f"                      intervals: min={array_summary['interval_min_seconds']:.6f}s, "
                      f"median={array_summary['interval_median_seconds']:.6f}s, "
                      f"max={array_summary['interval_max_seconds']:.6f}s")
            else:
                print(f"  Array summary     : shape={array_summary['shape']}, dtype={array_summary['dtype']}, all_finite=True")

            print(f"  SHA-256           : {sha256}")
            print()

            retrieved_bytes[member_name] = raw_data
            pose_results[member_name] = {
                "archive_path": info.filename,
                "compressed_size_bytes": info.compress_size,
                "uncompressed_size_bytes": info.file_size,
                "compression_method": COMPRESSION_NAMES.get(info.compress_type, str(info.compress_type)),
                "zip_crc32": f"0x{info.CRC:08x}",
                "computed_crc32": f"0x{computed_crc:08x}",
                "sha256": sha256,
                "array_summary": array_summary,
            }

    # 5. All validations passed; save files and manifest to destination directory
    stats = reader.stats()
    print("--- Transfer Statistics ---")
    print(f"Range requests made    : {stats['requests_made']} / {max_requests}")
    print(f"Response bytes fetched : {stats['bytes_fetched']:,} / {max_bytes:,} B")
    print(f"Archive total size     : {stats['archive_size']:,} bytes")
    print()

    manifest: dict[str, Any] = {
        "archive_url": archive_url,
        "segment": SEGMENT_PREFIX,
        "caveat": (
            "matching archive segment and HEVC size/CRC establish sample provenance, "
            "but exact decoded-frame correspondence and calibration remain unverified."
        ),
        "video_hevc_validation": {
            "archive_path": f"{SEGMENT_PREFIX}{VIDEO_MEMBER_NAME}",
            "archive_size_bytes": video_info.file_size,
            "archive_crc32": f"0x{video_info.CRC:08x}",
            "local_path": str(local_hevc_path),
            "local_size_bytes": local_size,
            "local_crc32": f"0x{local_crc:08x}",
            "verified_match": True,
        },
        "members": pose_results,
        "transfer_statistics": {
            "requests_made": stats["requests_made"],
            "max_requests_limit": max_requests,
            "bytes_fetched": stats["bytes_fetched"],
            "max_bytes_budget": max_bytes,
            "per_request_timeout_seconds": timeout,
            "archive_size_bytes": stats["archive_size"],
        },
    }

    try:
        dest_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"ERROR: Destination directory appeared during execution: {dest_dir}", file=sys.stderr)
        return 18

    # Write each original raw array file
    for name, data in retrieved_bytes.items():
        out_file = dest_dir / name
        out_file.write_bytes(data)
        print(f"Saved: {out_file} ({len(data):,} bytes)")

    # Write manifest
    manifest_path = dest_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved: {manifest_path}")
    print()
    print("✓ Successfully retrieved, verified, and saved daylight pose arrays.")
    return 0


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and validate daylight segment global_pose arrays from Chunk_1.zip."
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=DEFAULT_DEST_DIR,
        help=f"Destination directory (default: {DEFAULT_DEST_DIR})",
    )
    parser.add_argument(
        "--local-hevc",
        type=Path,
        default=DEFAULT_LOCAL_HEVC,
        help=f"Path to local daylight HEVC for CRC/size comparison (default: {DEFAULT_LOCAL_HEVC})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return fetch_daylight_pose(
        dest_dir=args.dest_dir,
        local_hevc_path=args.local_hevc,
    )


if __name__ == "__main__":
    sys.exit(main())
