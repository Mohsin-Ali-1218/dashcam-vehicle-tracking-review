"""
detect_vehicle_video.py — Video-level CPU vehicle detection baseline.

Processes consecutive frames from an input video using OpenCV DNN and YOLOX-S on CPU.
Reuses preprocessing, postprocessing (with class-agnostic NMS), and bounding box
annotation functions from detect_vehicle_frame.py to avoid code duplication.

Outputs written to a newly created output directory:
- annotated.mp4: Video with vehicle bounding box labels, frame indices, and
  timing caveat banners, preserving input dimensions and container FPS.
- detections.jsonl: Per-frame records containing frame index, detections list,
  and inference timing (including empty detection lists).
- summary.json: Audit file recording model SHA-256, settings, processed frame
  count, inference timing statistics, and processing loop duration.

Adapted from and reuses components of the official OpenCV Model Zoo implementation:
- OpenCV Zoo YOLOX directory: https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox
- Reference script: https://github.com/opencv/opencv_zoo/blob/main/models/object_detection_yolox/yolox.py
- Upstream YOLOX: https://github.com/Megvii-BaseDetection/YOLOX

Licence notice for adapted code:
--------------------------------
Copyright (c) 2021 Megvii, Inc. All rights reserved.
Copyright (c) 2022 OpenCV. All rights reserved.

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
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Import shared detector components from detect_vehicle_frame
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))
from detect_vehicle_frame import (  # noqa: E402
    INPUT_SIZE,
    PAD_VALUE,
    STRIDES,
    VEHICLE_CLASSES,
    annotate_frame,
    postprocess,
    preprocess_frame,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_sha256(filepath: Path) -> str:
    """Compute the SHA-256 hex digest of a local file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def add_frame_banner(
    frame: np.ndarray,
    frame_index: int,
    caveat_text: str = "Preview timing; capture timing unverified.",
) -> np.ndarray:
    """
    Overlay a compact status banner on the top-left of the frame showing the
    frame index and timing caveat.
    """
    text = f"Frame {frame_index:04d} | {caveat_text}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    pad_x, pad_y = 8, 6
    x1, y1 = 10, 10
    x2 = x1 + tw + 2 * pad_x
    y2 = y1 + th + 2 * pad_y + baseline

    # Opaque dark background rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (25, 25, 25), -1)
    # White text with anti-aliasing
    text_x = x1 + pad_x
    text_y = y1 + pad_y + th
    cv2.putText(
        frame,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (230, 230, 230),
        thickness,
        cv2.LINE_AA,
    )
    return frame


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process consecutive video frames with YOLOX-S vehicle detection on CPU, "
            "writing an annotated MP4 video, JSONL detections log, and summary JSON."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to local YOLOX ONNX model weights (required; auto-downloads disabled).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input video file (required).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to new output directory (required; refuses existing directory).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=120,
        help="Maximum consecutive frames to process from frame 0 (default: 120; must be a positive integer).",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.35,
        help="Confidence score threshold for candidate boxes (default: 0.35).",
    )
    parser.add_argument(
        "--nms-thresh",
        type=float,
        default=0.45,
        help="Non-maximum suppression IoU threshold (default: 0.45).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main Workflow
# ---------------------------------------------------------------------------

def run_video_detection(args: argparse.Namespace) -> int:
    """Run sequential video vehicle detection pipeline."""
    # --- Guard 1: Require new output directory and refuse existing one ------
    if args.output_dir.exists():
        print(
            f"ERROR: Output directory already exists: '{args.output_dir}'. "
            "A new, non-existent directory path is required.",
            file=sys.stderr,
        )
        return 1

    # --- Guard 2: Model weights must exist locally --------------------------
    if not args.model.exists():
        print(
            f"ERROR: Model file not found at '{args.model}'. "
            "Local weights are required; automatic downloads are prohibited.",
            file=sys.stderr,
        )
        return 1

    if not args.model.is_file():
        print(
            f"ERROR: Model path exists but is not a regular file: '{args.model}'.",
            file=sys.stderr,
        )
        return 1

    # --- Guard 3: Input video must exist ------------------------------------
    if not args.input.exists():
        print(f"ERROR: Input video file not found: '{args.input}'.", file=sys.stderr)
        return 1

    if not args.input.is_file():
        print(
            f"ERROR: Input path exists but is not a regular file: '{args.input}'.",
            file=sys.stderr,
        )
        return 1

    # --- Guard 4: Validate max-frames (must be positive) --------------------
    if args.max_frames <= 0:
        print(
            f"ERROR: --max-frames must be a positive integer; got {args.max_frames}.",
            file=sys.stderr,
        )
        return 1

    # --- Guard 5: Threshold validation (finite, in [0.0, 1.0]) --------------
    if not (np.isfinite(args.conf_thresh) and 0.0 <= args.conf_thresh <= 1.0):
        print(
            f"ERROR: Confidence threshold must be a finite float in [0.0, 1.0]; "
            f"got {args.conf_thresh}.",
            file=sys.stderr,
        )
        return 1

    if not (np.isfinite(args.nms_thresh) and 0.0 <= args.nms_thresh <= 1.0):
        print(
            f"ERROR: NMS threshold must be a finite float in [0.0, 1.0]; "
            f"got {args.nms_thresh}.",
            file=sys.stderr,
        )
        return 1

    # --- Compute model SHA-256 for audit record -----------------------------
    model_sha256 = compute_sha256(args.model)

    # --- Load model once via OpenCV DNN on CPU ------------------------------
    try:
        net = cv2.dnn.readNetFromONNX(str(args.model))
    except cv2.error as exc:
        print(f"ERROR: Failed to load ONNX model via OpenCV DNN: {exc}", file=sys.stderr)
        return 3

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # --- Open Input Video ---------------------------------------------------
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print(f"ERROR: OpenCV could not open video: '{args.input}'.", file=sys.stderr)
        return 2

    try:
        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        container_fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # --- Guard 6: Reject non-positive dimensions ------------------------
        if w_orig <= 0 or h_orig <= 0:
            print(
                f"ERROR: Input video reported non-positive frame dimensions: {w_orig}x{h_orig} px.",
                file=sys.stderr,
            )
            return 2

        # --- Guard 7: Reject invalid or non-finite container FPS ------------
        if not (np.isfinite(container_fps) and container_fps > 0):
            print(
                f"ERROR: Input video reported invalid or non-finite container FPS: {container_fps}.",
                file=sys.stderr,
            )
            return 2

        fps = float(container_fps)

        # Target frame count expected if video does not fail early
        expected_frames = (
            min(args.max_frames, total_video_frames)
            if total_video_frames > 0
            else args.max_frames
        )

        # --- Create output directory ----------------------------------------
        try:
            args.output_dir.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            print(f"ERROR: Failed to create output directory '{args.output_dir}': {exc}", file=sys.stderr)
            return 1

        # Output file paths
        video_out_path = args.output_dir / "annotated.mp4"
        jsonl_out_path = args.output_dir / "detections.jsonl"
        summary_out_path = args.output_dir / "summary.json"

        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (w_orig, h_orig))
        if not writer.isOpened():
            print(
                f"ERROR: OpenCV could not open VideoWriter for '{video_out_path}'. "
                "Check codec support.",
                file=sys.stderr,
            )
            writer.release()
            return 2

        # Initialize JSONL file
        try:
            jsonl_fh = open(jsonl_out_path, "w", encoding="utf-8")
        except OSError as exc:
            print(f"ERROR: Could not open '{jsonl_out_path}' for writing: {exc}", file=sys.stderr)
            writer.release()
            return 1

        print(f"Input video             : {args.input.resolve()}")
        print(f"Resolution & FPS        : {w_orig}x{h_orig} px @ {fps:.2f} FPS")
        print(f"Model file              : {args.model.resolve()}")
        print(f"Model SHA-256           : {model_sha256}")
        print(f"Target frame limit      : {args.max_frames}")
        print(f"Output directory        : {args.output_dir.resolve()}")
        print()

        frame_idx = 0
        inference_times: list[float] = []
        class_counts = {name: 0 for name in VEHICLE_CLASSES.values()}
        total_detections_count = 0
        loop_failed = False
        failure_message = ""

        t_loop_start = time.perf_counter()

        try:
            while frame_idx < args.max_frames:
                try:
                    ok, frame = cap.read()
                except cv2.error as exc:
                    loop_failed = True
                    failure_message = f"OpenCV read error at frame {frame_idx}: {exc}"
                    break

                if not ok or frame is None:
                    # End of file reached or read decode failure
                    break

                try:
                    # 1. Preprocess
                    blob, scale = preprocess_frame(frame, target_size=INPUT_SIZE, pad_value=PAD_VALUE)
                    net.setInput(blob)

                    # 2. Forward pass with timing
                    t0 = time.perf_counter()
                    raw_outputs = net.forward()
                    infer_duration = time.perf_counter() - t0
                    inference_times.append(infer_duration)

                    # Verify output shape on first frame
                    if frame_idx == 0:
                        if raw_outputs.shape not in ((1, 8400, 85), (8400, 85)):
                            loop_failed = True
                            failure_message = (
                                f"Unexpected model output shape {raw_outputs.shape}; "
                                "expected (1, 8400, 85) or (8400, 85)."
                            )
                            break

                    # 3. Postprocess with class-agnostic NMS
                    detections = postprocess(
                        raw_outputs=raw_outputs,
                        scale=scale,
                        orig_shape=(h_orig, w_orig),
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                    )
                except (cv2.error, ValueError, Exception) as exc:
                    loop_failed = True
                    failure_message = f"Inference/postprocessing error on frame {frame_idx}: {exc}"
                    break

                total_detections_count += len(detections)
                for det in detections:
                    class_counts[det["class_name"]] += 1

                # 4. Write per-frame record to detections.jsonl (includes empty lists)
                try:
                    record = {
                        "frame_index": frame_idx,
                        "detections": detections,
                        "inference_duration_s": round(infer_duration, 6),
                    }
                    jsonl_fh.write(json.dumps(record) + "\n")
                except OSError as exc:
                    loop_failed = True
                    failure_message = f"File writing error on '{jsonl_out_path}' at frame {frame_idx}: {exc}"
                    break

                # 5. Annotate frame with bounding boxes and caveat banner, then write
                try:
                    annotated = annotate_frame(frame, detections)
                    annotated = add_frame_banner(annotated, frame_idx)
                    writer.write(annotated)
                except cv2.error as exc:
                    loop_failed = True
                    failure_message = f"Video writing error on '{video_out_path}' at frame {frame_idx}: {exc}"
                    break

                frame_idx += 1

                # Progress reporting every 30 frames
                if frame_idx % 30 == 0:
                    print(
                        f"Progress: frame {frame_idx:4d}/{args.max_frames} processed "
                        f"(infer: {infer_duration * 1000:6.1f} ms) …"
                    )

        finally:
            jsonl_fh.close()
            writer.release()

        loop_duration = time.perf_counter() - t_loop_start

        # --- Handle failures during processing loop -------------------------
        if loop_failed:
            print(f"ERROR: {failure_message}", file=sys.stderr)
            print(
                f"Partial outputs preserved in '{args.output_dir}': "
                f"'{video_out_path.name}' ({frame_idx} frames written), "
                f"'{jsonl_out_path.name}' ({frame_idx} records written).",
                file=sys.stderr,
            )
            print("Summary file will not be written due to failure.", file=sys.stderr)
            return 3

        # --- Handle zero processed frames -----------------------------------
        if frame_idx == 0:
            print(
                f"ERROR: Zero frames were processed from '{args.input}'.",
                file=sys.stderr,
            )
            print(
                f"Partial outputs preserved in '{args.output_dir}': "
                f"'{video_out_path.name}' (0 frames), "
                f"'{jsonl_out_path.name}' (0 records).",
                file=sys.stderr,
            )
            print("Summary file will not be written due to failure.", file=sys.stderr)
            return 2

        # --- Check premature termination when frame count is reported -------
        if total_video_frames > 0 and frame_idx < expected_frames:
            print(
                f"ERROR: Premature termination: processed {frame_idx} frame(s), "
                f"expected {expected_frames} (video reported {total_video_frames} frames, "
                f"requested max_frames {args.max_frames}).",
                file=sys.stderr,
            )
            print(
                f"Partial outputs preserved in '{args.output_dir}': "
                f"'{video_out_path.name}' ({frame_idx} frames written), "
                f"'{jsonl_out_path.name}' ({frame_idx} records written).",
                file=sys.stderr,
            )
            print("Summary file will not be written due to failure.", file=sys.stderr)
            return 2

        # --- Report unknown frame count caveat if stopped before limit ------
        frame_count_caveat = None
        if total_video_frames <= 0 and frame_idx < args.max_frames:
            frame_count_caveat = (
                f"Reading stopped at frame {frame_idx} before reaching requested max_frames "
                f"({args.max_frames}). Total frame count is unknown; EOF versus decode "
                "failure cannot be distinguished."
            )
            print(f"NOTE: {frame_count_caveat}")

        # --- Write summary.json on success ----------------------------------
        summary = {
            "model": {
                "path": str(args.model.resolve()),
                "sha256": model_sha256,
                "input_size": list(INPUT_SIZE),
            },
            "settings": {
                "conf_thresh": args.conf_thresh,
                "nms_thresh": args.nms_thresh,
                "max_frames_requested": args.max_frames,
                "classes": VEHICLE_CLASSES,
            },
            "input_video": {
                "path": str(args.input.resolve()),
                "resolution": {"width": w_orig, "height": h_orig},
                "container_fps": round(container_fps, 3),
                "reported_frame_count": total_video_frames,
            },
            "output_video": {
                "path": str(video_out_path.resolve()),
                "resolution": {"width": w_orig, "height": h_orig},
                "fps": round(fps, 3),
            },
            "processed_frame_count": frame_idx,
            "total_detections": total_detections_count,
            "detection_counts_by_class": class_counts,
            "inference_timing_ms": {
                "min": round(min(inference_times) * 1000, 2) if inference_times else 0.0,
                "max": round(max(inference_times) * 1000, 2) if inference_times else 0.0,
                "mean": round(float(np.mean(inference_times)) * 1000, 2) if inference_times else 0.0,
                "median": round(float(np.median(inference_times)) * 1000, 2) if inference_times else 0.0,
                "std": round(float(np.std(inference_times)) * 1000, 2) if inference_times else 0.0,
            },
            "processing_loop_duration_s": round(loop_duration, 3),
        }
        if frame_count_caveat:
            summary["input_video"]["frame_count_note"] = frame_count_caveat

        try:
            with open(summary_out_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except OSError as exc:
            print(
                f"ERROR: Failed to write summary JSON to '{summary_out_path}': {exc}",
                file=sys.stderr,
            )
            print(
                f"Partial outputs preserved in '{args.output_dir}': "
                f"'{video_out_path.name}', '{jsonl_out_path.name}'.",
                file=sys.stderr,
            )
            return 1

        # --- Console Report -------------------------------------------------
        print()
        print("=== Video Detection Summary ===")
        print(f"Processed frames        : {frame_idx}")
        print(f"Total detections        : {total_detections_count}")
        for name, count in class_counts.items():
            print(f"  - {name:10s}          : {count}")
        if inference_times:
            mean_ms = np.mean(inference_times) * 1000
            med_ms = np.median(inference_times) * 1000
            print(f"Inference mean / median : {mean_ms:.2f} ms / {med_ms:.2f} ms")
        print(f"Processing loop duration: {loop_duration:.2f} s")
        print(f"Outputs written to      : {args.output_dir.resolve()}")
        print(f"  - Video               : {video_out_path.name}")
        print(f"  - Detections JSONL    : {jsonl_out_path.name}")
        print(f"  - Summary JSON        : {summary_out_path.name}")

    finally:
        cap.release()

    return 0


def main() -> None:
    args = parse_args()
    sys.exit(run_video_detection(args))


if __name__ == "__main__":
    main()
