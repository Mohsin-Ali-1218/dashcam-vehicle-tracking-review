"""
inspect_video.py — Project D video inspection utility.

Reports basic file and container-level properties of a video file using OpenCV.
FPS and duration are container/metadata values assigned during encoding and may
not reflect the actual capture frame rate of the source footage.  Our preview
footage was re-encoded at an assigned 20 FPS.
"""

import argparse
import sys
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a video file and report file size, OpenCV open status, "
            "container-reported width, height, FPS, frame count, and nominal duration."
        ),
    )
    parser.add_argument(
        "video_path",
        type=Path,
        help="Path to the video file to inspect.",
    )
    return parser.parse_args()


def inspect_video(video_path: Path) -> int:
    """
    Inspect *video_path* and print a summary of its properties.

    Returns
    -------
    int
        0 on success, non-zero on any error.
    """
    # --- File existence check ---------------------------------------------------
    if not video_path.exists():
        print(f"ERROR: File not found: {video_path}", file=sys.stderr)
        return 1

    if not video_path.is_file():
        print(f"ERROR: Path exists but is not a regular file: {video_path}", file=sys.stderr)
        return 1

    # --- File size ---------------------------------------------------------------
    file_size_bytes = video_path.stat().st_size
    print(f"File              : {video_path.resolve()}")
    print(f"File size (bytes) : {file_size_bytes:,}")

    # --- OpenCV open -------------------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            print("OpenCV opened     : NO", file=sys.stderr)
            print(
                "ERROR: OpenCV could not open the video file. "
                "The file may be corrupt or the codec unsupported.",
                file=sys.stderr,
            )
            return 2

        print("OpenCV opened     : YES")

        # --- Container-reported properties ---------------------------------------
        width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps         = cap.get(cv2.CAP_PROP_FPS)          # container value
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Width (px)        : {width}")
        print(f"Height (px)       : {height}")
        print(
            f"FPS               : {fps}"
            "  [container value — not verified source capture timing]"
        )
        print(f"Frame count       : {frame_count}")

        # --- Nominal duration ----------------------------------------------------
        if fps > 0 and frame_count > 0:
            nominal_duration_s = frame_count / fps
            mm = int(nominal_duration_s // 60)
            ss = nominal_duration_s % 60
            print(
                f"Nominal duration  : {nominal_duration_s:.3f} s  ({mm}m {ss:.3f}s)"
                "  [container value — not verified source capture timing]"
            )
        else:
            print(
                "Nominal duration  : N/A "
                f"(fps={fps}, frame_count={frame_count} — cannot compute)"
            )

    finally:
        cap.release()

    return 0


def main() -> None:
    args = parse_args()
    sys.exit(inspect_video(args.video_path))


if __name__ == "__main__":
    main()
