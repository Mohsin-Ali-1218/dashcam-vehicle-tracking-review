"""
detect_vehicle_frame.py — Single-frame CPU vehicle detection baseline.

Extracts a single specified frame from a video file, preprocesses it according
to the official OpenCV Zoo YOLOX specifications (640x640 RGB float32, unscaled
0-255 values, top-left placement with bottom/right padding of 114), runs inference
via the OpenCV DNN module on CPU, decodes grid/stride outputs, applies class-agnostic
NMS across retained vehicle classes (car, motorcycle, bus, truck), clips bounding
boxes to frame bounds, and writes an annotated output image.

Adapted from the official OpenCV Model Zoo implementation:
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
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants & Configurations
# ---------------------------------------------------------------------------

INPUT_SIZE = (640, 640)
STRIDES = [8, 16, 32]
PAD_VALUE = 114

# Vehicle classes from MS COCO 80-class dataset (0-indexed)
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Distinct BGR colours for vehicle visualization
CLASS_COLORS = {
    2: (0, 215, 255),  # car: gold / amber
    3: (255, 128, 0),  # motorcycle: azure
    5: (0, 165, 255),  # bus: orange
    7: (0, 0, 255),    # truck: red
}


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run single-frame vehicle detection on dashcam video footage using "
            "an OpenCV Zoo YOLOX ONNX model on CPU."
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
        "--frame-index",
        type=int,
        required=True,
        help="Non-negative integer index of the frame to inspect (required).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output annotated image file (required; refuses overwrite).",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.35,
        help="Confidence score threshold for filtering candidate boxes (default: 0.35).",
    )
    parser.add_argument(
        "--nms-thresh",
        type=float,
        default=0.45,
        help="IoU threshold for non-maximum suppression (default: 0.45).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_frame(
    frame: np.ndarray,
    target_size: tuple[int, int] = INPUT_SIZE,
    pad_value: int = PAD_VALUE,
) -> tuple[np.ndarray, float]:
    """
    Preprocess image frame according to official OpenCV Zoo YOLOX specifications.

    - Uniform aspect-ratio scaling to fit within target_size (640x640).
    - Top-left placement inside the canvas, leaving bottom and right padding.
    - Constant padding with pad_value (114).
    - Color conversion from BGR to RGB.
    - Unscaled 0-255 float32 4D blob (1, 3, 640, 640).

    Returns
    -------
    tuple[np.ndarray, float]
        (blob, scale) where scale = min(target_h / orig_h, target_w / orig_w).
    """
    h_orig, w_orig = frame.shape[:2]
    target_h, target_w = target_size

    scale = min(target_h / h_orig, target_w / w_orig)
    w_new = int(w_orig * scale)
    h_new = int(h_orig * scale)

    # Resize preserving aspect ratio
    resized = cv2.resize(frame, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

    # Top-left placement with bottom/right padding of 114
    padded = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    padded[0:h_new, 0:w_new] = resized

    # Build 4D blob: float32, unscaled [0, 255], RGB channel order
    blob = cv2.dnn.blobFromImage(
        padded,
        scalefactor=1.0,
        size=target_size,
        mean=(0.0, 0.0, 0.0),
        swapRB=True,
        crop=False,
    )
    return blob, scale


# ---------------------------------------------------------------------------
# Grid & Stride Generation
# ---------------------------------------------------------------------------

def generate_grids_and_strides(
    strides: list[int] = STRIDES,
    input_size: tuple[int, int] = INPUT_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate anchor-free grid coordinates and expanded stride vectors.

    For input size 640x640 and strides [8, 16, 32]:
    - Stride 8 : 80x80 = 6400 cells
    - Stride 16: 40x40 = 1600 cells
    - Stride 32: 20x20 =  400 cells
    Total      : 8400 grid points

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        grids (8400, 2), expanded_strides (8400, 1)
    """
    grids = []
    expanded_strides = []
    for stride in strides:
        hsize = input_size[0] // stride
        wsize = input_size[1] // stride
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), axis=2).reshape(-1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((grid.shape[0], 1), stride, dtype=np.float32))

    grids = np.concatenate(grids, axis=0).astype(np.float32)
    expanded_strides = np.concatenate(expanded_strides, axis=0).astype(np.float32)
    return grids, expanded_strides


# ---------------------------------------------------------------------------
# Postprocessing & Decoding
# ---------------------------------------------------------------------------

def postprocess(
    raw_outputs: np.ndarray,
    scale: float,
    orig_shape: tuple[int, int],
    conf_thresh: float,
    nms_thresh: float,
) -> list[dict]:
    """
    Decode YOLOX grid/stride outputs, compute scores, and apply class-agnostic NMS.

    Parameters
    ----------
    raw_outputs : np.ndarray
        Network output tensor of shape (1, 8400, 85) or (8400, 85).
    scale : float
        Uniform scaling factor applied during preprocessing.
    orig_shape : tuple[int, int]
        (height, width) of the original video frame.
    conf_thresh : float
        Minimum score threshold (objectness * class_probability).
    nms_thresh : float
        Intersection-over-Union (IoU) threshold for NMS.

    Returns
    -------
    list[dict]
        List of detections, each containing:
        - class_id (int)
        - class_name (str)
        - confidence (float)
        - box (list of int: [x, y, w, h]) in original frame coordinates.
    """
    if raw_outputs.shape not in ((1, 8400, 85), (8400, 85)):
        raise ValueError(
            f"Unexpected model output shape {raw_outputs.shape}; "
            "expected (1, 8400, 85) or (8400, 85)."
        )

    outputs = raw_outputs[0] if raw_outputs.ndim == 3 else raw_outputs
    h_orig, w_orig = orig_shape

    grids, expanded_strides = generate_grids_and_strides(STRIDES, INPUT_SIZE)

    # Box center coordinates (cx, cy) and dimensions (w, h) in 640x640 space
    boxes_xy = (outputs[:, :2] + grids) * expanded_strides
    boxes_wh = np.exp(outputs[:, 2:4]) * expanded_strides

    # Objectness and class probabilities
    obj_conf = outputs[:, 4:5]
    class_probs = outputs[:, 5:]

    # Multiply objectness by class probability across all 80 classes
    class_scores = obj_conf * class_probs

    # 1. Choose each prediction's highest-scoring class across all 80 classes
    best_class_ids = np.argmax(class_scores, axis=1)
    best_scores = class_scores[np.arange(len(class_scores)), best_class_ids]

    # 2. Map both box corners back to original coordinates
    x1 = (boxes_xy[:, 0] - boxes_wh[:, 0] / 2.0) / scale
    y1 = (boxes_xy[:, 1] - boxes_wh[:, 1] / 2.0) / scale
    x2 = (boxes_xy[:, 0] + boxes_wh[:, 0] / 2.0) / scale
    y2 = (boxes_xy[:, 1] + boxes_wh[:, 1] / 2.0) / scale

    # Clip corners to original image bounds
    x1_clipped = np.clip(x1, 0.0, float(w_orig))
    y1_clipped = np.clip(y1, 0.0, float(h_orig))
    x2_clipped = np.clip(x2, 0.0, float(w_orig))
    y2_clipped = np.clip(y2, 0.0, float(h_orig))

    # Filter candidate predictions whose highest-scoring class is in VEHICLE_CLASSES
    # and whose confidence exceeds conf_thresh
    is_vehicle = np.isin(best_class_ids, list(VEHICLE_CLASSES.keys()))
    cand_mask = is_vehicle & (best_scores > conf_thresh)
    if not np.any(cand_mask):
        return []

    c_x1 = x1_clipped[cand_mask]
    c_y1 = y1_clipped[cand_mask]
    c_x2 = x2_clipped[cand_mask]
    c_y2 = y2_clipped[cand_mask]
    c_scores = best_scores[cand_mask]
    c_class_ids = best_class_ids[cand_mask]

    # Discard zero- or negative-area boxes
    valid_area = (c_x2 > c_x1) & (c_y2 > c_y1)
    if not np.any(valid_area):
        return []

    c_x1 = c_x1[valid_area]
    c_y1 = c_y1[valid_area]
    c_x2 = c_x2[valid_area]
    c_y2 = c_y2[valid_area]
    c_scores = c_scores[valid_area]
    c_class_ids = c_class_ids[valid_area]

    # Derive [x, y, width, height] from clipped corners
    boxes_list = []
    valid_scores = []
    retained_class_ids = []
    for bx1, by1, bx2, by2, score, cls_id in zip(
        c_x1, c_y1, c_x2, c_y2, c_scores, c_class_ids
    ):
        ix1 = int(round(bx1))
        iy1 = int(round(by1))
        ix2 = int(round(bx2))
        iy2 = int(round(by2))
        bw = ix2 - ix1
        bh = iy2 - iy1
        if bw > 0 and bh > 0:
            boxes_list.append([ix1, iy1, bw, bh])
            valid_scores.append(float(score))
            retained_class_ids.append(int(cls_id))

    if not boxes_list:
        return []

    # Single class-agnostic NMS pass across all retained vehicle candidates
    indices = cv2.dnn.NMSBoxes(boxes_list, valid_scores, conf_thresh, nms_thresh)
    detections: list[dict] = []
    if len(indices) > 0:
        for idx in np.array(indices).flatten():
            cid = retained_class_ids[idx]
            detections.append({
                "class_id": cid,
                "class_name": VEHICLE_CLASSES[cid],
                "confidence": valid_scores[idx],
                "box": boxes_list[idx],
            })

    return detections


# ---------------------------------------------------------------------------
# Annotation & Visualization
# ---------------------------------------------------------------------------

def annotate_frame(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes and class/confidence labels onto a copy of the frame."""
    annotated = frame.copy()
    for det in detections:
        x, y, w, h = det["box"]
        cls_id = det["class_id"]
        label = f"{det['class_name']}: {det['confidence']:.2f}"
        color = CLASS_COLORS.get(cls_id, (0, 255, 0))

        # Bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        # Label badge with background fill for readability
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        badge_y1 = max(0, y - th - 6)
        badge_y2 = y
        badge_x2 = min(annotated.shape[1], x + tw + 6)
        cv2.rectangle(annotated, (x, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.putText(
            annotated,
            label,
            (x + 3, badge_y2 - baseline - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------

def run_detection(args: argparse.Namespace) -> int:
    """Run single-frame vehicle detection workflow."""
    # --- Guard 1: Refuse output overwrite -----------------------------------
    if args.output.exists():
        print(
            f"ERROR: Output file already exists: '{args.output}'. "
            "Refusing to overwrite.",
            file=sys.stderr,
        )
        return 1

    # --- Guard 2: Model weights must exist locally (no auto-download) --------
    if not args.model.exists():
        print(
            f"ERROR: Model weights not found at '{args.model}'. "
            "A local model file is required; automatic downloads are prohibited.",
            file=sys.stderr,
        )
        return 1

    if not args.model.is_file():
        print(
            f"ERROR: Model path exists but is not a regular file: '{args.model}'.",
            file=sys.stderr,
        )
        return 1

    # --- Guard 3: Video input file check ------------------------------------
    if not args.input.exists():
        print(f"ERROR: Input video file not found: '{args.input}'.", file=sys.stderr)
        return 1

    if not args.input.is_file():
        print(
            f"ERROR: Input path exists but is not a regular file: '{args.input}'.",
            file=sys.stderr,
        )
        return 1

    if args.frame_index < 0:
        print(
            f"ERROR: Frame index must be non-negative; got {args.frame_index}.",
            file=sys.stderr,
        )
        return 1

    # --- Guard 4: Threshold validation (finite, in [0.0, 1.0]) --------------
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

    # --- Open Video & Extract Frame -----------------------------------------
    cap = cv2.VideoCapture(str(args.input))
    try:
        if not cap.isOpened():
            print(f"ERROR: OpenCV could not open video: '{args.input}'.", file=sys.stderr)
            return 2

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0 and args.frame_index >= total_frames:
            print(
                f"ERROR: Requested frame index {args.frame_index} exceeds total frame "
                f"count of {total_frames} in '{args.input}'.",
                file=sys.stderr,
            )
            return 2

        if not cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index):
            print(
                f"ERROR: Could not seek to frame index {args.frame_index} in '{args.input}'.",
                file=sys.stderr,
            )
            return 2

        ok, frame = cap.read()
        if not ok or frame is None:
            print(
                f"ERROR: Could not read frame at index {args.frame_index} from '{args.input}'.",
                file=sys.stderr,
            )
            return 2
    finally:
        cap.release()

    h_orig, w_orig = frame.shape[:2]

    # --- Preprocess ---------------------------------------------------------
    blob, scale = preprocess_frame(frame, target_size=INPUT_SIZE, pad_value=PAD_VALUE)

    # --- Load Network & Run Inference on CPU --------------------------------
    try:
        net = cv2.dnn.readNetFromONNX(str(args.model))
    except cv2.error as exc:
        print(f"ERROR: Failed to load ONNX model via OpenCV DNN: {exc}", file=sys.stderr)
        return 3

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    net.setInput(blob)

    t0 = time.perf_counter()
    raw_outputs = net.forward()
    inference_duration_s = time.perf_counter() - t0

    # --- Guard 5: Output shape validation -----------------------------------
    if raw_outputs.shape not in ((1, 8400, 85), (8400, 85)):
        print(
            f"ERROR: Unexpected model output shape {raw_outputs.shape}; "
            "expected (1, 8400, 85) or (8400, 85).",
            file=sys.stderr,
        )
        return 3

    # --- Postprocess & NMS --------------------------------------------------
    detections = postprocess(
        raw_outputs=raw_outputs,
        scale=scale,
        orig_shape=(h_orig, w_orig),
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh,
    )

    # --- Annotate & Save Output ---------------------------------------------
    annotated = annotate_frame(frame, detections)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(args.output), annotated)
    if not ok:
        print(f"ERROR: cv2.imwrite failed to save '{args.output}'.", file=sys.stderr)
        return 4

    # --- Report Results -----------------------------------------------------
    class_counts = {name: 0 for name in VEHICLE_CLASSES.values()}
    for det in detections:
        class_counts[det["class_name"]] += 1

    print(f"Input video         : {args.input.resolve()}")
    print(f"Frame index         : {args.frame_index} ({w_orig}x{h_orig} px)")
    print(f"Model file          : {args.model.resolve()}")
    print(f"Inference duration  : {inference_duration_s * 1000:.2f} ms ({inference_duration_s:.4f} s)")
    print(f"Detection count     : {len(detections)}")
    for name, count in class_counts.items():
        print(f"  - {name:10s}        : {count}")
    print(f"Annotated output    : {args.output.resolve()}")

    return 0


def main() -> None:
    args = parse_args()
    sys.exit(run_detection(args))


if __name__ == "__main__":
    main()
