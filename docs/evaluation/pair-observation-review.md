# Pair Observation Review: Same-Frame Image-Space Relationships

This report documents the verification and selected-frame visual review of same-frame vehicle pair observations exported by [scripts/export_vehicle_pair_observations.py](../../scripts/export_vehicle_pair_observations.py) to [data/preview/daylight-tracking-1200-v1/pair-observations.csv](../../data/preview/daylight-tracking-1200-v1/pair-observations.csv), derived from [data/preview/daylight-tracking-1200-v1/trajectories-timed.csv](../../data/preview/daylight-tracking-1200-v1/trajectories-timed.csv).

---

## 1. Verified Export Counts & Data Integrity

- **Output File**: [data/preview/daylight-tracking-1200-v1/pair-observations.csv](../../data/preview/daylight-tracking-1200-v1/pair-observations.csv)
- **Pair Observations**: **5,974** same-frame pair rows across **757** frames with $\ge 2$ detections.
- **Verification Result**: Assistant comparison found no missing or duplicate pairs, and zero field or geometry mismatches against the timed trajectory records.
- **Positive Box Overlaps**: **673** pair rows have positive rounded bounding-box IoU (`box_iou > 0.0`).
- **Critical Distinction**: Positive bounding-box IoU values are **not** conflict events or collision indicators. In a 2D monocular camera projection, overlapping bounding boxes arise frequently from perspective projection, occlusions, and detection artifacts without implying physical roadway proximity or conflict.

---

## 2. Selected-Frame Visual Review Findings

Visual review of sampled frames exhibiting positive bounding-box IoU highlights three distinct phenomena:

### Finding 1: Duplicate Detection on Single Vehicle
- **Frame Index**: `148`
- **Track IDs**: `4` / `40`
- **Box IoU**: `0.2755`
- **Visual Finding**: Duplicate bounding boxes assigned to a single SUV body moving along the right roadway edge. The detector emitted two overlapping detections for one physical vehicle, which the tracker initialized as separate temporary tracks (IDs 4 and 40). This is a detector duplication artifact, not two interacting vehicles.

### Finding 2: Occluded Vehicle Cluster with Uncertain Association
- **Frame Index**: `375`
- **Track IDs**: `44` / `57`
- **Box IoU**: `0.4045`
- **Visual Finding**: A visually occluded vehicle cluster in traffic where bounding boxes overlap significantly. Vehicle association is uncertain; dense visual occlusion prevents verified 2D separation. This represents tracking ambiguity in heavy visual clutter rather than a physical conflict.

### Finding 3: Optical Overlap in Projection Behind Highway Median
- **Frame Index**: `1095`
- **Track IDs**: `117` / `119`
- **Box IoU**: `0.2237`
- **Visual Finding**: Distinct physical vehicles traveling in opposing or separated lanes behind the median barrier overlap visually in 2D image space due to camera perspective and sightline alignment. The vehicles are physically separated by the median infrastructure; the overlap is purely a line-of-sight projection artifact.

---

## 3. Scope & Methodological Caveats

1. **Selected-Frame Review vs. Event Validation**: These three cases represent selected-frame visual spot checks of geometric relationships. They do **not** constitute continuous event validation or conflict classification.
2. **2D Image Space Limits**: Bounding-box IoU, axis gaps, and pixel center distances describe 2D image coordinates only. They do not measure 3D world proximity, metric distances, vehicle speeds, Time-To-Collision (TTC), or traffic safety.
3. **No Filtering Applied**: In accordance with the exporter specification, the export uses all input observations without confidence filtering; frames with fewer than two observations emit no pair rows.
4. **Settings & Status**: Tracker and export settings are kept unchanged. Conflict analysis remains pending.
