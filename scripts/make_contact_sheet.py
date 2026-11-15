"""
make_contact_sheet.py — Project D contact sheet generator.

Extracts six frames from the preview video at fixed indices, arranges them in a
2-column × 3-row grid, labels each with its frame index, and writes a JPEG
contact sheet.

Notes
-----
- FPS reported by the container (20.0) was assigned during re-encoding and does
  not verify the original capture timing; the heading reflects this.
- The output file will NOT be overwritten if it already exists.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_INPUT_VIDEO  = Path("data/preview/traffic-preview.mp4")
DEFAULT_OUTPUT_IMAGE = Path("data/preview/contact-sheet.jpg")
INPUT_VIDEO          = DEFAULT_INPUT_VIDEO
OUTPUT_IMAGE         = DEFAULT_OUTPUT_IMAGE

FRAME_INDICES   = [0, 240, 480, 720, 960, 1199]
GRID_COLS       = 2
GRID_ROWS       = 3
THUMB_MAX_WIDTH = 582          # px — aspect ratio preserved

HEADING_TEXT    = "Preview samples \u2014 capture timing unverified."
HEADING_HEIGHT  = 48           # px reserved above the grid
LABEL_HEIGHT    = 24           # px reserved below each thumbnail
FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_HEAD = 0.65
FONT_SCALE_LBL  = 0.50
FONT_THICKNESS  = 1
BG_COLOR        = (30, 30, 30)    # dark grey  (B, G, R)
TEXT_COLOR      = (220, 220, 220) # near-white (B, G, R)
JPEG_QUALITY    = 92


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resize_to_width(frame: np.ndarray, target_width: int) -> np.ndarray:
    """Return *frame* resized so its width equals *target_width*, preserving AR."""
    h, w = frame.shape[:2]
    if w == target_width:
        return frame
    scale  = target_width / w
    new_h  = max(1, int(round(h * scale)))
    return cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)


def draw_label(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    """Draw *text* centred inside the rectangle (x, y, w, h) on *canvas*."""
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE_LBL, FONT_THICKNESS)
    tx = x + (w - tw) // 2
    ty = y + (h + th) // 2
    cv2.putText(canvas, text, (tx, ty), FONT, FONT_SCALE_LBL,
                TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)


def draw_heading(canvas: np.ndarray, text: str, total_width: int) -> None:
    """Draw *text* centred horizontally in the heading band."""
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE_HEAD, FONT_THICKNESS)
    tx = (total_width - tw) // 2
    ty = (HEADING_HEIGHT + th) // 2
    cv2.putText(canvas, text, (tx, ty), FONT, FONT_SCALE_HEAD,
                TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def extract_frames(video_path: Path, indices: list[int]) -> list[tuple[int, np.ndarray]]:
    """
    Open *video_path* and seek to each index in *indices*.

    Returns
    -------
    list of (frame_index, frame_bgr) tuples, in the order they appear in *indices*.

    Raises
    ------
    SystemExit
        On open failure or any read error, with a descriptive message on stderr.
    """
    cap = cv2.VideoCapture(str(video_path))
    frames: list[tuple[int, np.ndarray]] = []
    try:
        if not cap.isOpened():
            print(
                f"ERROR: OpenCV could not open '{video_path}'. "
                "The file may be missing, corrupt, or the codec unsupported.",
                file=sys.stderr,
            )
            sys.exit(2)

        for idx in indices:
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, idx):
                print(
                    f"ERROR: Could not seek to frame {idx} in '{video_path}'.",
                    file=sys.stderr,
                )
                sys.exit(3)

            ok, frame = cap.read()
            if not ok or frame is None:
                print(
                    f"ERROR: Could not read frame {idx} from '{video_path}'. "
                    "The video may be truncated.",
                    file=sys.stderr,
                )
                sys.exit(3)

            frames.append((idx, frame))
    finally:
        cap.release()

    return frames


def build_contact_sheet(
    frames: list[tuple[int, np.ndarray]],
) -> np.ndarray:
    """
    Arrange *frames* in a GRID_ROWS × GRID_COLS grid and return the composite image.

    Each thumbnail is limited to THUMB_MAX_WIDTH pixels wide (aspect preserved).
    A heading band sits above the grid; a label band sits below each thumbnail.
    """
    # Resize all thumbnails first so we know their common height
    thumbs = [(idx, resize_to_width(f, THUMB_MAX_WIDTH)) for idx, f in frames]
    thumb_h = thumbs[0][1].shape[0]   # all frames share the same AR → same height
    thumb_w = THUMB_MAX_WIDTH

    cell_h     = thumb_h + LABEL_HEIGHT
    sheet_w    = thumb_w * GRID_COLS
    sheet_h    = HEADING_HEIGHT + cell_h * GRID_ROWS

    sheet = np.full((sheet_h, sheet_w, 3), BG_COLOR, dtype=np.uint8)

    draw_heading(sheet, HEADING_TEXT, sheet_w)

    for i, (idx, thumb) in enumerate(thumbs):
        row = i // GRID_COLS
        col = i  % GRID_COLS

        x = col * thumb_w
        y = HEADING_HEIGHT + row * cell_h

        # Place thumbnail
        sheet[y : y + thumb_h, x : x + thumb_w] = thumb

        # Place label band below thumbnail
        label_y = y + thumb_h
        draw_label(sheet, f"frame {idx}", x, label_y, thumb_w, LABEL_HEIGHT)

    return sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract six frames from a preview video, arrange them in a "
            "2-column × 3-row grid, and write a JPEG contact sheet."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_VIDEO,
        help=f"Path to input video file (default: {DEFAULT_INPUT_VIDEO})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_IMAGE,
        help=f"Path to output JPEG contact sheet (default: {DEFAULT_OUTPUT_IMAGE})",
    )
    return parser.parse_args()


def make_contact_sheet(
    input_video: Path = DEFAULT_INPUT_VIDEO,
    output_image: Path = DEFAULT_OUTPUT_IMAGE,
) -> int:
    """Entry point; returns an exit code."""

    # --- Guard: output must not already exist ----------------------------------
    if output_image.exists():
        print(
            f"ERROR: Output file already exists: '{output_image}'. "
            "Remove it manually before running this script.",
            file=sys.stderr,
        )
        return 1

    # --- Guard: input must exist -----------------------------------------------
    if not input_video.exists():
        print(f"ERROR: Input video not found: '{input_video}'.", file=sys.stderr)
        return 1

    if not input_video.is_file():
        print(
            f"ERROR: Input path exists but is not a regular file: '{input_video}'.",
            file=sys.stderr,
        )
        return 1

    # --- Extract frames ---------------------------------------------------------
    print(f"Opening '{input_video}' …")
    frames = extract_frames(input_video, FRAME_INDICES)
    print(f"Extracted {len(frames)} frames: {[idx for idx, _ in frames]}")

    # --- Build sheet ------------------------------------------------------------
    print("Building contact sheet …")
    sheet = build_contact_sheet(frames)
    print(f"Sheet dimensions  : {sheet.shape[1]} × {sheet.shape[0]} px (W × H)")

    # --- Write output -----------------------------------------------------------
    output_image.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_image), sheet, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        print(
            f"ERROR: cv2.imwrite failed — could not write '{output_image}'. "
            "Check directory permissions.",
            file=sys.stderr,
        )
        return 4

    print(f"Saved             : {output_image.resolve()}")
    return 0


def main() -> None:
    args = parse_args()
    sys.exit(make_contact_sheet(input_video=args.input, output_image=args.output))


if __name__ == "__main__":
    main()
