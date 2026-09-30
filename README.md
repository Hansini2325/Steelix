# Steelix - Intelligent Steel Surface Defect Detection & Traceability AI System

A real-time surface defect detection system for steel sheets, built on YOLOv8 and deployed through a Streamlit dashboard. The system detects six classes of surface defects from the NEU-DET dataset, supports both image and video inference, stores every inspection in a structured SQLite database, and generates a Digital Steel Passport for each inspected batch.

---

## Table of Contents

- [Overview](#overview)
- [Screenshots](#screenshots)
- [Project Structure](#project-structure)
- [Defect Classes](#defect-classes)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Dashboard](#dashboard)
- [Deployment & Export](#deployment--export)
- [Testing](#testing)
- [Configuration](#configuration)
- [Modules](#modules)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## Overview

This project was built around the NEU Surface Defect Database, which contains grayscale images of hot-rolled steel strips grouped into six defect categories. The pipeline covers everything from raw dataset conversion to YOLO-format preparation, model training, real-time inference, and a full analytics dashboard.

The Streamlit frontend (Steelix) provides:

- Single image upload and batch folder inference
- Live webcam feed with bounding box overlays
- Per-inspection analytics — class distribution, spatial heatmaps, temporal trends, and recurrence detection
- A Digital Steel Passport that summarizes each inspection session and stores it persistently
- ONNX and TensorRT export paths for production deployment

The system is designed as a research and demonstration tool. The inspection decision layer (PASS / FLAGGED / REVIEW) is a confidence-threshold heuristic and is **not** a certified quality-control system.

---

## Screenshots

**Dashboard — Home / Live Inference**
<img width="959" height="506" alt="Screenshot 2026-09-30 211025" src="https://github.com/user-attachments/assets/f27c4ead-5857-4096-90c5-c9772d294455" />
<img width="956" height="503" alt="Screenshot 2026-09-30 211511" src="https://github.com/user-attachments/assets/82f2bf03-0052-4284-b42b-7f7e0fe47d62" />

**Defect Analytics — Heatmap View**
<img width="953" height="497" alt="Screenshot 2026-09-30 211553" src="https://github.com/user-attachments/assets/1d3b8529-9dbe-4d51-8bb2-78f3e9bfd520" />

**Digital Steel Passport**
<img width="956" height="501" alt="Screenshot 2026-09-30 211527" src="https://github.com/user-attachments/assets/d21fefc7-9a37-43d4-aa8a-c23acc753fa8" />

**Training Metrics**
<img width="956" height="365" alt="Screenshot 2026-09-30 211607" src="https://github.com/user-attachments/assets/870e7d3e-bb5e-4089-a83b-6f39e9d39aee" />


---

## Project Structure

```
steel-defect-detection/
│
├── configs/
│   ├── dataset.yaml          # Dataset paths, class names, split ratios
│   ├── train.yaml            # YOLOv8 training hyperparameters
│   ├── inference.yaml        # Inference thresholds, output paths, visualization
│   ├── inspection_rules.yaml # Pass/flag/review decision rules
│   └── tracking.yaml         # Object tracking settings
│
├── data/
│   ├── raw/                  # Place the original NEU-DET download here
│   ├── interim/              # Intermediate files during dataset conversion
│   └── processed/            # Final YOLO-format dataset (auto-generated)
│
├── demo/
│   ├── app.py                # Streamlit dashboard entry point
│   ├── logo.png
│   └── favicon.ico
│
├── models/
│   └── best.pt               # Trained weights (generated after training)
│
├── notebooks/                # Jupyter notebooks for exploration
│
├── results/
│   ├── training/             # Training logs, plots, checkpoints
│   └── predictions/          # Saved annotated outputs
│
├── scripts/
│   ├── prepare_dataset.py    # Converts NEU-DET to YOLO format
│   ├── train.py              # Launches training
│   ├── evaluate.py           # Runs evaluation on the test split
│   ├── predict.py            # Runs inference on an image or folder
│   ├── realtime.py           # Webcam inference (OpenCV window)
│   ├── benchmark.py          # FPS / latency benchmarking
│   └── export.py             # Exports to ONNX or TensorRT
│
├── src/
│   ├── analytics/            # Heatmaps, temporal trends, recurrence detection
│   ├── database/             # SQLite models and repository layer
│   ├── deployment/           # ONNX export, TensorRT export, benchmarking
│   ├── evaluation/           # mAP, precision, recall computation
│   ├── inference/            # Ultralytics and ONNX Runtime predictors
│   ├── passport/             # Digital Steel Passport generation and export
│   ├── tracking/             # Multi-object tracking across video frames
│   ├── training/             # Trainer wrapper around Ultralytics
│   └── utils/                # Logger, hardware detection, visualization helpers
│
├── tests/                    # Pytest test suite
├── requirements.txt
├── pyproject.toml
└── pytest.ini
```

---

## Defect Classes

The model is trained to detect these six defect types from the NEU-DET dataset:

| ID | Class Name       | Description                                                  |
|----|------------------|--------------------------------------------------------------|
| 0  | crazing          | Network of fine surface cracks                               |
| 1  | inclusion        | Embedded foreign particles or slag                           |
| 2  | patches          | Irregular discolored regions on the surface                  |
| 3  | pitted_surface   | Small cavities or pits distributed across the surface        |
| 4  | rolled_in_scale  | Scale pressed into the steel during rolling                  |
| 5  | scratches        | Linear surface scratches from mechanical contact             |

---

## Requirements

- Python 3.10 or 3.11
- pip
- CUDA-capable GPU (optional but strongly recommended for training)

Core dependencies:

```
torch >= 2.1.0
torchvision >= 0.16.0
ultralytics >= 8.0.200
opencv-python >= 4.8.0
streamlit >= 1.28.0
plotly >= 5.17.0
onnx >= 1.14.0
onnxruntime >= 1.16.0
pandas >= 2.0.0
scikit-learn >= 1.3.0
```

See `requirements.txt` for the full pinned list.

---

## Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/yourusername/steel-defect-detection.git
cd steel-defect-detection

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

If you want GPU-accelerated ONNX inference, replace `onnxruntime` with `onnxruntime-gpu` in `requirements.txt` before installing.

---

## Dataset Preparation

Download the NEU-DET dataset and place it under `data/raw/`. The prepare script auto-detects the layout (images + XML annotations, classification folders, or flat YOLO-format) and converts everything to the YOLO directory structure expected by Ultralytics.

```bash
python scripts/prepare_dataset.py
```

After this completes, `data/processed/neu_det.yaml` will be written. That file is referenced by the training config automatically.

The default split is 80 / 10 / 10 (train / val / test). You can adjust ratios in `configs/dataset.yaml`.

---

## Training

```bash
python scripts/train.py
```

Training uses the settings in `configs/train.yaml`. Key defaults:

| Parameter       | Default     |
|-----------------|-------------|
| Base model      | yolov8n.pt  |
| Epochs          | 100         |
| Image size      | 640         |
| Batch size      | 16          |
| Optimizer       | AdamW       |
| Early stopping  | 20 epochs   |
| Mixed precision | Enabled     |

Checkpoints and plots are saved to `results/training/neu_det_experiment/`. The best weights are copied to `models/best.pt` on completion.

To switch to a larger model, edit `model` in `configs/train.yaml`:

```yaml
model: yolov8s.pt   # or yolov8m.pt for higher accuracy
```

---

## Inference

**Single image:**

```bash
python scripts/predict.py --source path/to/image.jpg
```

**Folder of images:**

```bash
python scripts/predict.py --source path/to/folder/
```

**Video file:**

```bash
python scripts/predict.py --source path/to/video.mp4
```

**Webcam (live feed):**

```bash
python scripts/realtime.py
```

Annotated outputs are saved to `results/predictions/` by default. Confidence and IoU thresholds can be overridden via command-line flags or by editing `configs/inference.yaml`.

---

## Dashboard

Start the Streamlit dashboard:

```bash
streamlit run demo/app.py
```

The dashboard opens at `http://localhost:8501` and includes:

- **Live Inference** — upload an image or enable webcam mode for real-time detection
- **Batch Processing** — run inference on a folder of images at once
- **Analytics** — spatial heatmaps, class distribution charts, temporal trends across sessions, and recurrence pattern detection
- **Inspection History** — browse all past inspections stored in the local SQLite database
- **Digital Steel Passport** — view and export a full inspection report for any recorded session

The database file is created automatically under `storage/` on first run.

---

## Deployment & Export

**Export to ONNX:**

```bash
python scripts/export.py --format onnx
```

**Export to TensorRT** (requires TensorRT installed separately):

```bash
python scripts/export.py --format tensorrt
```

**Run latency benchmark:**

```bash
python scripts/benchmark.py
```

ONNX models can be used directly in the predictor by pointing `weights` in `configs/inference.yaml` to the `.onnx` file instead of `.pt`. The `ONNXPredictor` class in `src/inference/predictor.py` handles preprocessing, NMS, and postprocessing independently of Ultralytics.

---

## Testing

```bash
pytest
```

The test suite covers analytics, database operations, passport generation, the predictor classes, utility functions, and visualization helpers. Coverage report:

```bash
pytest --cov=src --cov-report=term-missing
```

---

## Configuration

All runtime behavior is controlled through YAML files in `configs/`. You rarely need to touch Python files to adjust thresholds or paths.

| File                    | Controls                                                      |
|-------------------------|---------------------------------------------------------------|
| `dataset.yaml`          | Dataset paths, class names, split ratios                      |
| `train.yaml`            | Model, epochs, batch size, augmentation, optimizer            |
| `inference.yaml`        | Confidence threshold, IoU, output folder, visualization style |
| `inspection_rules.yaml` | Pass / flag / review decision thresholds                      |
| `tracking.yaml`         | Tracker type, track buffer length, match threshold            |

---

## Modules

### `src/inference`

Two predictor classes: `UltralyticsPredictor` (uses the full Ultralytics stack, supports `.pt` weights) and `ONNXPredictor` (uses ONNX Runtime, no Ultralytics dependency at inference time). Both return the same output dict format so they can be swapped without changing calling code.

### `src/analytics`

- `defect_analytics.py` — aggregates per-frame detections into session-level statistics  
- `heatmap.py` — discretizes bounding box coordinates into a spatial grid for heatmap rendering  
- `recurrence_detector.py` — identifies defect classes or regions that appear repeatedly across consecutive frames  
- `temporal_analysis.py` — tracks detection counts and confidence over time  
- `spatial_analysis.py` — computes which image region (grid cell) is most frequently affected  

### `src/database`

SQLite-backed persistence using a hand-written repository pattern (no ORM). `models.py` defines dataclasses for `Inspection` and `ModelMetadata`. `repository.py` handles all queries.

### `src/passport`

`PassportManager` creates an inspection record at the start of a session, collects frame-level data during inference, and finalizes the record with a status of PASS, FLAGGED, or REVIEW. `ReportGenerator` formats the stored data into a printable report. `ExportManager` handles file output.

### `src/deployment`

ONNX and TensorRT export wrappers, plus a benchmarking harness in `benchmark.py` that runs a configurable number of warmup and timed iterations and reports mean latency and throughput.

---

## Known Limitations

- The inspection decision layer (PASS / FLAGGED / REVIEW) is based on simple confidence thresholds and defect count heuristics. It has not been validated against any industrial standard and should not be used as-is in a production quality control line.
- The NEU-DET dataset contains only grayscale images of hot-rolled steel strips. Performance on other steel types, surface finishes, or imaging conditions will vary and may require fine-tuning.
- Webcam inference speed depends on your hardware. On CPU, frame rates may be too low for practical real-time use. A CUDA-capable GPU is recommended.
- TensorRT export requires a compatible NVIDIA GPU, the CUDA toolkit, and TensorRT installed separately. The export script will raise a clear error if these are not present.


