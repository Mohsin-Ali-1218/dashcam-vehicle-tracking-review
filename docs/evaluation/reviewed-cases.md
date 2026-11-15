# Reviewed Cases: Manual Tracking Continuity Evaluation

This report documents the read-only evaluation of tracking continuity across two selected manual visibility references against decoded daylight video tracks ([data/preview/daylight-tracking-1200-v1/tracks.jsonl](../../data/preview/daylight-tracking-1200-v1/tracks.jsonl)), alongside the verified clip-wide trajectory coverage summary ([data/preview/daylight-tracking-1200-v1/trajectory-coverage.csv](../../data/preview/daylight-tracking-1200-v1/trajectory-coverage.csv)).

---

## 1. Frame Record Integrity & Scope

- **Record Existence**: All 52 labelled frame records (41 frames in the SUV case + 11 frames in the distant vehicle case) exist in `tracks.jsonl`.
- **Detection Completeness Caveat**: Non-empty records in `tracks.jsonl` do not mean every visible vehicle was detected; other active tracks were detected in each frame while specific vehicle bodies experienced detection dropouts.
- **Evaluation Scope**: These selected intervals support the reviewed dropout and exit findings; they do **not** represent clip-wide tracking accuracy, systematic ID-switch counts, or general tracker validation.
- **Tracker Thresholds**: Matching thresholds are kept unchanged (IoU threshold 0.30, 5-frame missed-detection memory window before track retirement).

---

## 2. Selected Case Evaluations

### Case 1: SUV Right Boundary Exit (Track ID 4)

- **Reference File**: [docs/evaluation/suv-right-200-240.csv](suv-right-200-240.csv)
- **Explicit Mapping**: `suv_right_01` → Track ID 4
- **Labelled Visibility Counts**:
  - `partially_visible`: 23 frames (frames 200–222)
  - `outside_view`: 18 frames (frames 223–240)
  - `visible`: 0 frames
- **Observation Breakdown**:
  - Mapped ID 4 observed while partially visible: **21 frames** (frames 200–220)
  - Mapped ID 4 absent while partially visible: **2 frames** (`221, 222`)
  - Mapped ID 4 observed while labelled outside view: **0 frames** (no observations across frames 223–240)
- **Finding**: The vehicle was tracked continuously through frame 220. Partial vehicle body remained visible at the right image edge during frames 221–222 without detector output (boundary dropout). No ID 4 observations occurred in labelled outside-view frames 223–240; this does not assess all detections or ID assignments.

### Case 2: Distant Vehicle Detection Dropout & Recovery (Track ID 3)

- **Reference File**: [docs/evaluation/distant-vehicle-404-414.csv](distant-vehicle-404-414.csv)
- **Explicit Mapping**: `distant_vehicle_01` → Track ID 3
- **Labelled Visibility Counts**:
  - `visible`: 11 frames (frames 404–414)
  - `partially_visible`: 0 frames
  - `outside_view`: 0 frames (outside-view behavior was not assessed because no such frames were labelled)
- **Observation Breakdown**:
  - Mapped ID 3 observed while visible: **6 frames** (frames 404, 405, 406, 412, 413, 414)
  - Mapped ID 3 absent while visible: **5 frames** (`407, 408, 409, 410, 411`)
  - Track ID 3 resumes: frame **412**
  - Mapped ID 3 observed while labelled outside view: **0 frames** (no outside-view frames in reference)
- **Finding**: The vehicle remained visible in the original video throughout frames 404–414. The detector produced no bounding box across the 5-frame interval 407–411 (detection dropout). Because the tracker retains unmatched tracks for up to 5 missed frames, ID 3 was maintained in memory and successfully recovered on frame 412 with no observed fragmentation or ID reassignment in frames 404–414.

---

## 3. Verified Full-Clip Trajectory Coverage Summary

Summary statistics produced by [scripts/summarize_trajectory_coverage.py](../../scripts/summarize_trajectory_coverage.py) from [data/preview/daylight-tracking-1200-v1/trajectories-timed.csv](../../data/preview/daylight-tracking-1200-v1/trajectories-timed.csv):

| Metric | Verified Value |
|---|---|
| Total temporary track IDs | 146 |
| Total observations | 3,324 |
| IDs with internal gaps | 28 (19.18%) |
| Total within-ID gaps | 44 |
| Total unobserved frames within spans | 88 |
| Maximum single gap across IDs | 5 frames |
| IDs with class label changes | 8 (5.48%) |
| Total class label changes | 25 |

---

## 4. Methodological Caveats & Next Steps

1. **Selected Cases vs. Full Ground Truth**: Manual reviews of boundary exits and detection dropouts provide qualitative evidence on tracker mechanics under specific conditions. They do not constitute full-clip ground truth, formal precision/recall, MOTA/HOTA, or systematic ID-switch counts.
2. **Tracker Thresholds**: The 5-frame retention window enabled recovery in Case 2 while ID 4 ceased being output after frame 220; retirement timing follows the inspected lifecycle code and was not directly logged. Thresholds remain unchanged.
3. **Status**: Selected-case tracking assessment is completed. Broader labelled tracking evaluation and potential-conflict analysis remain pending.

---

## 5. Additional Gap-Boundary Visual Review

- **ID 107**: Original frames 1012–1017 reviewed consecutively; vehicle remains visible during absent-ID frames 1014–1015; same ID resumes at 1016.
- **ID 14**: Frames 171–176 reviewed; vehicle partly occluded by nearer sedan around absent-ID frames 173–174. Physical correspondence is less certain; do not count confirmed fragmentation or an ID switch.
- **ID 67**: Frames 430–434 reviewed; vehicle remains visible at absent-ID frame 432; same ID resumes at 433.
- **ID 72**: Frames 516–524 reviewed; yellow vehicle remains partly visible at left boundary during absent-ID frames 518 and 522; same ID resumes at 519 and 523.

These are additional visual checks against saved records, not new frame-label CSVs or systematic identity metrics. Original two-case reference counts are preserved. Tracker thresholds remain unchanged.
