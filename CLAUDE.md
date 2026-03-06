# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Data preparation** (XML annotations → TXT labels):
```bash
python dataprep_xml2txt.py
```

**Training** (requires pre-trained weights at `./weights/yolov3.weights`):
```bash
python detection_train.py
```

**Detection/Testing** (outputs to `./data/detection/`):
```bash
python detection_test.py
```

**Evaluation** (mAP, outputs to `./mAP/`):
```bash
python mAP/result.py
```

**Tracking** (image or video input, configured in `core/config.py`):
```bash
python tracking.py
```

## Environment Requirements

- Python 3.7
- CUDA 10.0 + cuDNN 7.4
- TensorFlow 2.0
- Install dependencies: `pip3 install -r ./docs/requirements.txt`

## Architecture

### Data Flow

1. Raw XML annotations (`./data/dataset/dogcat/Annotation_merge/`) are converted to TXT format by `dataprep_xml2txt.py`, producing `Label_train.txt` and `Label_test.txt` in `./data/dataset/dogcat/`.
2. `core/dataset.py` reads these label files and feeds batches to the model during training.
3. The model operates on images resized/padded to 416×416 (letterbox via `utils.image_preporcess`).

### Detection Pipeline (`core/`)

- **`backbone.py`** — Darknet53 feature extractor; returns 3 route tensors at different scales.
- **`yolov3.py`** — Assembles the full YOLOv3 head on top of Darknet53; outputs 3 feature maps (`conv_sbbox`, `conv_mbbox`, `conv_lbbox`) for small/medium/large objects. `decode()` converts raw grid outputs to absolute (x, y, w, h, conf, prob). Loss uses GIoU + focal confidence + sigmoid class probability.
- **`common.py`** — Shared building blocks: `convolutional` (Conv+BN+LeakyReLU), `upsample`, `residual_block`.
- **`utils.py`** — Weight loading from Darknet `.weights` format, NMS, box post-processing, distance calculation, and drawing/tracking helpers.

### Distance Measurement

`utils.cal_distance(u, v)` computes real-world ground distance from a pixel coordinate using a full camera projection model (rotation matrices Rx/Ry/Rz + translation). Camera intrinsics/extrinsics are configured under `cfg.DISTANCE` in `core/config.py`. Default values are placeholders — they must be calibrated for the actual camera used.

### Tracking Pipeline

`tracking.py` instantiates two independent `Tracker` objects from `core/KalmanFilterTracker.py`:
- `tracker15` for **cats** (COCO class ID 15)
- `tracker16` for **dogs** (COCO class ID 16)

`KalmanFilterTracker.py` provides three classes:
- `KalmanFilter` — 2D position Kalman filter (predict/correct cycle).
- `Track` — Wraps a `KalmanFilter` instance with a track ID, skip counter, and trace history.
- `Tracker` — Manages a list of `Track` objects; uses the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) to associate detections to existing tracks each frame. Tracks are pruned after `MAX_FRAMES_TO_SKIP` consecutive missed detections.

`Tracker.Update()` flow per frame:
1. Build N×M Euclidean distance cost matrix (tracks × detections).
2. Run Hungarian algorithm to get optimal assignment.
3. Reject assignments where distance exceeds `dist_thresh`; increment `skipped_frames` for all unmatched tracks.
4. Delete stale tracks (indices iterated in reverse to avoid index shift).
5. Create new tracks for unmatched detections via `_create_track()`.
6. Run KF predict/correct for each remaining track; trim `trace` to `max_trace_length`.

### Configuration (`core/config.py`)

All paths and hyperparameters are centralized using `easydict`. Key sections:
- `cfg.DATASET` — raw/converted data paths, train/test split ratio.
- `cfg.YOLO` — class names file, anchors file, strides `[8, 16, 32]`.
- `cfg.TRAIN` — batch size, epochs, LR schedule (warmup + cosine decay), weight paths, `TRAINING_FROM_SCRATCH` flag.
- `cfg.TEST` — score/IOU thresholds, output paths.
- `cfg.TRACKING` — `INPUT_TYPE` (`"image"` or `"video"`), distance/skip/trace-length thresholds.
- `cfg.DISTANCE` — camera intrinsics and extrinsics for distance measurement.

### Weight Loading

- `.weights` (Darknet binary format) → loaded via `utils.load_weights()` which manually maps weights to TF layers in order.
- `.weights` (TF SavedModel/checkpoint) → loaded via `model.load_weights()`.
- The code checks `if '.weights' in path` to select the loader.
