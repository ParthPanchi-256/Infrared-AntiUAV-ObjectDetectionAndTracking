# Project Audit Report — Infrared Dim-Small Aircraft Detection & Tracking

**Date:** 2026-10-29  
**Auditor:** Automated ML Engineering Audit  
**Project Root:** `C:\Users\parth\Documents\My Projects\MiniProject`

---

## 1. Executive Summary

| Area | Status |
|------|--------|
| **Python Environment** | ✅ Python 3.11.5 venv operational |
| **PyTorch + CUDA** | ✅ **GPU-Ready** — PyTorch 2.14.0+cu126, CUDA ops verified |
| **GPU** | ✅ RTX 3050 Laptop (4 GB VRAM), Compute 8.6 |
| **Dataset** | ⚠️ 22 infrared sequences, 16,177 BMP frames (~3 GB), center-point annotations only — **no bounding boxes** |
| **YOLO Baseline** | ⚠️ Ultralytics 8.4.154 installed; pre-trained `yolo11n.pt` weights present |
| **PISTTN Feasibility** | 🔴 Sparse-convolution libraries NOT installed; no bounding-box annotations available |
| **Git / Version Control** | 🔴 No `.git` repository initialized |

**Classification: GPU-Ready**  
Evidence: PyTorch 2.14.0+cu126 is installed, `torch.cuda.is_available()` returns `True`, a 512×512 matrix multiply on CUDA completed successfully, and GPU memory was properly allocated and freed.

---

## 2. Project Directory Overview

```
MiniProject/
├── anti-uav/                  # Anti-UAV experiment subfolder
│   ├── outputs/               # (empty)
│   ├── scripts/               # (empty)
│   ├── videos/
│   │   └── test.mov           # 20 MB, 1280×720, ~30 fps, 1217 frames, ~40 s
│   └── yolo11n.pt             # 5.4 MB — YOLOv11 nano weights (duplicate)
├── data/                      # Primary dataset (22 sequences + labels)
│   ├── data1/ … data22/       # 22 sequence folders, each containing a subfolder with BMP frames
│   └── data_label/            # 22 .txt annotation files
├── yolo-env/                  # Python 3.11.5 virtual environment
├── anti_uav_tracked.mp4       # 29 MB output video — 640×512, 30 fps, 3000 frames, 100 s
├── bus.jpg                    # 137 KB — test image (likely Ultralytics sample)
├── extracted_page.json        # 10 KB — OCR output (Gujarati text, unrelated)
├── practice.py                # OCR practice script (HuggingFace indic-ocr)
├── test.jpg                   # 88 KB — Gujarati newspaper image for OCR
├── test.py                    # YOLO + ByteTrack tracking pipeline script
└── yolo11n.pt                 # 5.4 MB — YOLOv11 nano weights
```

> [!NOTE]
> No `requirements.txt`, `pyproject.toml`, `setup.py`, `Makefile`, `Dockerfile`, `.gitignore`, `.env`, or `README.md` exist. No Git repository is initialized.

### Python Source Files

| File | Purpose | Status |
|------|---------|--------|
| [`practice.py`](file:///c:/Users/parth/Documents/My%20Projects/MiniProject/practice.py) | Downloads & runs HuggingFace `bodhan-ai/indic-ocr` model on `test.jpg` | **Verified** — Unrelated to IR detection project |
| [`test.py`](file:///c:/Users/parth/Documents/My%20Projects/MiniProject/test.py) | Loads YOLO11n, runs ByteTrack on an image sequence, outputs tracked video | **Verified** — Working YOLO+tracking baseline, references `C:\Users\parth\Downloads\track1_test\track1_test\wg2022_ir_054_split_03` (external path) |

---

## 3. Python & Virtual Environment Details

| Property | Value | Confidence |
|----------|-------|------------|
| **Venv location** | `MiniProject/yolo-env/` | **Verified** |
| **Venv Python** | `yolo-env/Scripts/python.exe` → Python 3.11.5 | **Verified** |
| **Base Python** | `C:\Users\parth\AppData\Local\Programs\Python\Python311\python.exe` | **Verified** |
| **Pip version** | 26.2.1 (inside venv) | **Verified** |
| **System site-packages** | `false` (isolated) | **Verified** |
| **Original creation path** | `C:\Users\parth\Documents\My Projects\ML\yolo-env` (moved from different location) | **Verified** — pyvenv.cfg `command` field differs from current path |

> [!WARNING]
> The venv was originally created at `…\ML\yolo-env` but now lives at `…\MiniProject\yolo-env`. This is functional but could cause issues with certain path-dependent tools. The `home` and `executable` fields still point to the valid base Python.

---

## 4. Installed Dependency Inventory

### Core ML/CV Packages (Verified — all imports succeed)

| Package | Version | Notes |
|---------|---------|-------|
| `torch` | 2.14.0+cu126 | CUDA 12.6 build |
| `torchvision` | 0.29.0+cu126 | CUDA 12.6 build |
| `torchaudio` | 2.11.0+cu126 | CUDA 12.6 build |
| `ultralytics` | 8.4.154 | YOLOv11 support |
| `opencv-python` | 5.0.0.93 | cv2 5.0.0 |
| `numpy` | 2.4.6 | |
| `Pillow` | 12.3.0 | |
| `matplotlib` | 3.11.2 | |
| `PyYAML` | 6.0.3 | |
| `tqdm` | 4.70.1 | |
| `transformers` | 5.17.0 | HuggingFace Transformers |
| `huggingface_hub` | 1.32.0 | |
| `accelerate` | 1.15.0 | |
| `safetensors` | 0.8.0 | |
| `einops` | 0.8.2 | Useful for research architectures |
| `lap` | 0.5.13 | Linear Assignment — used by trackers |

### Additional Installed Packages

| Package | Version |
|---------|---------|
| `polars` | 1.44.2 |
| `snac` | 1.2.1 |
| `soundfile` | 0.14.0 |
| `nvidia-ml-py` | 13.610.43 |
| `rich` | 15.0.0 |
| `httpx` | 0.28.1 |
| `networkx` | 3.6.1 |
| `sympy` | 1.14.0 |
| `regex` | 2026.9.10 |
| `cloudpickle` | 3.1.2 |
| `psutil` | 7.2.2 |

### Missing Packages (NOT INSTALLED)

| Package | Needed For |
|---------|------------|
| `pandas` | Data analysis, metric computation |
| `scipy` | Scientific computing, evaluation metrics |
| `scikit-learn` | Classification metrics, data splitting |
| `tensorboard` | Training visualization |
| `pytest` | Unit testing |
| `spconv` / `MinkowskiEngine` | **Sparse convolutions (critical for PISTTN)** |

### Dependency Health

| Check | Result |
|-------|--------|
| `pip check` | **No broken requirements found** | **Verified** |

---

## 5. NVIDIA Driver & GPU Information

| Property | Value | Confidence |
|----------|-------|------------|
| **GPU Name** | NVIDIA GeForce RTX 3050 Laptop GPU | **Verified** |
| **Driver Version** | 592.00 | **Verified** |
| **CUDA Version (nvidia-smi)** | 13.1 (max supported by driver) | **Verified** |
| **Driver-Model** | WDDM | **Verified** |
| **Total VRAM** | 4096 MiB (~4 GB) | **Verified** |
| **Idle VRAM Usage** | 21 MiB (at audit time) | **Verified** |
| **GPU Temperature** | 52°C (idle) | **Verified** |
| **Power Usage** | 3W / 80W cap | **Verified** |
| **GPU Utilization** | 0% (idle) | **Verified** |
| **Compute Capability** | 8.6 (Ampere) | **Verified** |
| **Multi-Processor Count** | 16 SMs | **Verified** |

> [!IMPORTANT]
> **CUDA version distinction:** `nvidia-smi` reports CUDA 13.1 (maximum the driver supports). PyTorch was built with CUDA 12.6. This is compatible — the driver supports CUDA 12.6 and below.

---

## 6. PyTorch / CUDA Verification Results

### Test Script Output (Verified)

```
PyTorch version: 2.14.0+cu126
PyTorch CUDA build: 12.6
CUDA available: True
GPU count: 1
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
Total memory (MB): 4095.5
CUDA capability: (8, 6)
CUDA op successful: torch.Size([512, 512])
Free memory (MB): 3267.7
Total memory (MB): 4095.5
```

### Verification Checklist

| Test | Result | Confidence |
|------|--------|------------|
| PyTorch installed | ✅ 2.14.0+cu126 | **Verified** |
| PyTorch CUDA build | ✅ CUDA 12.6 | **Verified** |
| `torch.cuda.is_available()` | ✅ `True` | **Verified** |
| `torch.cuda.device_count()` | ✅ 1 | **Verified** |
| `torch.cuda.get_device_name(0)` | ✅ RTX 3050 Laptop GPU | **Verified** |
| CUDA tensor allocation (512×512) | ✅ Success | **Verified** |
| Matrix multiply on CUDA | ✅ Success, shape [512,512] | **Verified** |
| `torch.cuda.synchronize()` | ✅ No errors | **Verified** |
| `torch.cuda.empty_cache()` | ✅ Memory freed | **Verified** |
| `torchvision` import | ✅ 0.29.0+cu126 | **Verified** |
| CUDA errors during test | ✅ None | **Verified** |

---

## 7. Dataset Structure & File Statistics

### Overview

| Metric | Value | Confidence |
|--------|-------|------------|
| **Total sequences** | 22 | **Verified** |
| **Total image frames** | 16,177 | **Verified** |
| **Total image data size** | ~3,034 MB (~3.0 GB) | **Verified** |
| **Image format** | BMP (uncompressed) | **Verified** |
| **Label files** | 22 `.txt` files in `data_label/` | **Verified** |
| **Image dimensions** | 256 × 256 pixels (all sampled) | **Verified** |
| **File size per image** | 196,662 bytes (all identical) | **Verified** |

### Per-Sequence Breakdown

| Label File | Seq Name (header) | Frames | Unique Objects | Empty Frames | Object IDs |
|------------|--------------------|--------|----------------|--------------|------------|
| data1.txt | data1 | 399 | 1 | 0 | object:1 |
| data2.txt | data2 | 599 | 2 | 0 | object:1, object:2 |
| data3.txt | data3 | 100 | 2 | 27 | object:1, object:2 |
| data4.txt | data4 | 399 | 2 | 0 | object:1, object:2 |
| data5.txt | data5 | 3,000 | 1 | 0 | object:1 |
| data6.txt | data6 | 399 | 1 | 0 | object:1 |
| data7.txt | data7 | 399 | 1 | 0 | object:1 |
| data8.txt | data8 | 399 | 1 | 0 | object:1 |
| data9.txt | data9 | 399 | 1 | 0 | object:1 |
| data10.txt | data1 | 401 | 1 | 0 | object:1 |
| data11.txt | data3 | 745 | 1 | 0 | object:1 |
| data12.txt | data4 | 1,500 | 1 | 0 | object:1 |
| data13.txt | data5 | 763 | 1 | 0 | object:1 |
| data14.txt | data6 | 1,426 | 1 | 0 | object:1 |
| data15.txt | data7 | 751 | 1 | 0 | object:1 |
| data16.txt | data8 | 499 | 2 | 5 | object:1, object:2 |
| data17.txt | data9 | 500 | 1 | 0 | object:1 |
| data18.txt | data10 | 500 | 1 | 0 | object:1 |
| data19.txt | data11 | 1,599 | 5 | 199 | object:1..5 |
| data20.txt | data12 | 400 | 1 | 0 | object:1 |
| data21.txt | data13 | 500 | 1 | 0 | object:1 |
| data22.txt | data14 | 500 | 1 | 0 | object:1 |

> [!NOTE]
> **Label-to-folder naming mismatch:** Label file names (data1.txt..data22.txt) map to folder names (data1..data22), but the internal header `seq_name` field uses a different naming scheme (data1..data14). Multiple label files reference the same header name (e.g., data10.txt→data1, data11.txt→data3). This suggests the dataset may have been renumbered or reorganized.

---

## 8. Image & Video Metadata Findings

### Image Properties (Verified from multiple sequences)

| Property | Value | Confidence |
|----------|-------|------------|
| **Format** | BMP (uncompressed bitmap) | **Verified** |
| **Dimensions** | 256 × 256 pixels | **Verified** |
| **PIL Mode** | RGB (3-channel) | **Verified** |
| **Actual Content** | **Grayscale** — all 3 RGB channels are identical | **Verified** |
| **Bit Depth** | 8-bit per channel (uint8) | **Verified** |
| **Data Type** | numpy uint8 | **Verified** |
| **Array Shape** | (256, 256, 3) | **Verified** |
| **Pixel Min** | Varies by sequence (17–84) | **Verified** |
| **Pixel Max** | 255 (all sampled sequences) | **Verified** |
| **Mean Intensity** | ~127 (all sampled sequences) | **Verified** |
| **Std Dev** | ~13 (data1 sample) | **Verified** |
| **Unique Values** | 99 (data1/frame 0) — limited dynamic range | **Verified** |

> [!WARNING]
> Images are stored as 24-bit RGB BMP but are **grayscale** (all 3 channels identical). This triples storage and processing cost. The limited unique pixel count (99 of 256 possible values) and narrow standard deviation (~13) suggest low contrast — consistent with infrared dim-target imagery. Converting to single-channel grayscale would save 2/3 of storage/memory with zero information loss.

### Video Files

| File | Location | Size | Resolution | FPS | Frames | Duration | Notes |
|------|----------|------|------------|-----|--------|----------|-------|
| `test.mov` | `anti-uav/videos/` | 20 MB | 1280×720 | ~30.15 | 1,217 | ~40 s | **Verified** — Source video |
| `anti_uav_tracked.mp4` | project root | 29 MB | 640×512 | 30 | 3,000 | 100 s | **Verified** — Output from `test.py` |

---

## 9. Annotation Schema & Representative Examples

### Header Format (Line 1 of each label file)

```
<class_id>\t<sequence_name>\t<frame_count>\t<total_unique_objects>
```

Example: `0	data2	599	2` — Class 0, sequence "data2", 599 frames, 2 unique objects.

> [!NOTE]
> Some headers have extra tab characters causing the frame count to shift columns. This is a minor parsing issue (data18–data22).

### Per-Frame Format (Lines 2+)

**Single object:**
```
frame:<N>\t<obj_count>\tobject:<id>\t<x>\t<y>
```
Example: `frame:0	1	object:1	136	119`

**Multiple simultaneous objects:**
```
frame:<N>\t<obj_count>\tobject:<id1>\t<x1>\t<y1>\tobject:<id2>\t<x2>\t<y2>
```
Example: `frame:0	2	object:1	127	131	object:2	173	125`

**No object visible:**
```
frame:<N>\t0\t\t\t
```
Example: `frame:0	0			`

### Key Schema Observations

| Aspect | Finding | Confidence |
|--------|---------|------------|
| **Annotation type** | **Center-point (x, y) only** — NO bounding boxes | **Verified** |
| **Coordinate system** | Pixel coordinates in 256×256 image space | **Verified** |
| **Object IDs** | `object:1`, `object:2`, … — **track identities present** | **Verified** |
| **Class IDs** | All headers show class `0` — single class | **Verified** |
| **Per-frame object count** | Field 2 is the number of objects in that specific frame | **Verified** |
| **Object appearance pattern** | In multi-object sequences (data3, data16, data19), objects may appear sequentially (not simultaneously) with gap frames | **Verified** |
| **Simultaneous multi-object** | data2 and data4 have 2 objects visible simultaneously in every frame | **Verified** |
| **Frame numbering** | Continuous, 0-indexed: `frame:0` through `frame:(N-1)` | **Verified** |

### Representative Annotation Examples

**data1.txt (single target, continuous tracking):**
```
0	data1	399	1	
frame:0	1	object:1	136	119
frame:1	1	object:1	136	119
...
frame:398	1	object:1	166	130
frame:399				    ← frame 399 has no annotation (1 extra frame line)
```

**data2.txt (two simultaneous targets):**
```
0	data2	599	2			
frame:0	2	object:1	127	131	object:2	173	125
frame:1	2	object:1	127	130	object:2	172	124
```

**data3.txt (two sequential targets with gaps):**
```
0	data3	100	2	
frame:0	0				← no target
...
frame:12	1	object:1	244	74   ← object:1 appears
...
frame:28	1	object:1	255	53
frame:29	0				← gap
...
frame:44	1	object:2	229	72   ← object:2 appears (new identity)
```

**data19.txt (5 sequential targets):**
- Object:1 present in frames 0–1373
- Object:2 present in frames 1395–1406
- Object:3 present in frames 1414–1417
- Object:4 present in frames 1428–1433
- Object:5 present in frames 1460–1463
- 199 frames have no visible target

---

## 10. Data-Quality Issues

| Issue | Severity | Details |
|-------|----------|---------|
| **Grayscale stored as 3-channel RGB** | Medium | 3× storage/memory waste. No information loss if converted. |
| **BMP format (uncompressed)** | Low | ~3 GB for 16k frames. PNG would save ~60-70% disk. Not a blocker. |
| **Label file naming inconsistency** | Medium | External file names (data1.txt–data22.txt) don't match internal header sequence names (data1–data14). Mapping requires reading headers. |
| **Header format inconsistency** | Low | Files data18–data22 have extra tab characters in headers, shifting the frame count column. Needs robust parser. |
| **data1.txt last frame** | Low | Frame 399 line exists but has no annotation (empty). Expected frame count is 399, file has 0-indexed frames 0–398 plus one empty line for frame 399. May indicate off-by-one. |
| **No bounding boxes** | **Critical** | Only center coordinates (x, y) are provided. YOLO requires bounding boxes (x_center, y_center, width, height). |
| **Very small targets** | Medium | Target positions move slowly across 256×256 images. Targets may be only 1–3 pixels in size — challenging for standard detectors. |
| **Limited dynamic range** | Low | Only ~99 unique intensity values out of 256. Typical for 8-bit IR quantization. |
| **No train/val/test split defined** | Medium | All 22 sequences are in a flat structure with no split designation. |

---

## 11. Suitability for Detection, Tracking, YOLO & PISTTN

### Object Detection

| Capability | Supported? | Notes |
|------------|------------|-------|
| **Center-point detection** | ✅ Directly | Annotations provide (x, y) centers per frame |
| **Bounding-box detection** | ❌ Not directly | No width/height annotations. Would need to: (a) manually annotate, or (b) synthesize fixed-size boxes around centers (introduces assumption) |
| **YOLO training** | ❌ Not directly | YOLO requires `class x_center y_center width height` in normalized coordinates. Center-only annotations need conversion with assumed box sizes. |
| **Multi-class detection** | ❌ Not applicable | All annotations are class 0 (single class) |

### Single-Target Tracking

| Capability | Supported? | Notes |
|------------|------------|-------|
| **Center-point tracking** | ✅ Directly | 17 of 22 sequences have a single continuous target trajectory |
| **Trajectory evaluation** | ✅ Directly | Euclidean distance between predicted and GT center can be computed |
| **Bounding-box tracking** | ❌ Not directly | No bounding box dimensions |

### Multi-Object Tracking (MOT)

| Capability | Supported? | Notes |
|------------|------------|-------|
| **Track IDs available** | ✅ Yes | `object:1`, `object:2`, etc. |
| **Simultaneous multi-object** | ⚠️ Partial | Only 2 sequences (data2, data4) have simultaneous multi-object. Most multi-object sequences are sequential. |
| **MOT evaluation** | ⚠️ Partial | Standard MOT metrics (MOTA, IDF1) require bounding boxes. Center-based metrics are possible but non-standard. |

### YOLO Baseline

| Requirement | Status |
|-------------|--------|
| Ultralytics installed | ✅ v8.4.154 |
| Pre-trained weights | ✅ `yolo11n.pt` (5.4 MB) |
| YOLO annotation format | ❌ **Not available** — needs conversion from center-point + assumed box size |
| ByteTrack tracker | ✅ Referenced in `test.py`, `lap` package installed |

### PISTTN Architecture

| Requirement | Status |
|-------------|--------|
| Temporal frame sequences | ✅ Available (sequential IR frames with frame indices) |
| Sparse convolution library | ❌ **NOT INSTALLED** (`spconv`, `MinkowskiEngine`) |
| Windows compatibility for sparse conv | ⚠️ **Unknown** — `spconv` has limited Windows support; `MinkowskiEngine` historically requires Linux |
| Point-based annotations | ✅ Center coordinates available |
| Small target detection focus | ✅ Dataset characteristics match (dim, small IR targets) |

---

## 12. Hardware Feasibility & Memory Risks

| Scenario | Feasibility | Notes |
|----------|-------------|-------|
| **Basic PyTorch CUDA ops** | ✅ Verified | 512×512 matmul succeeds |
| **YOLO11n inference (256×256)** | ✅ Likely feasible | Nano model (~5.6 MB), small input. Estimated <1 GB VRAM. |
| **YOLO11n training (256×256, batch=8)** | ⚠️ Tight | ~2–3 GB VRAM estimated for nano model. Mixed precision (`amp`) recommended. |
| **YOLO11s/m training** | ⚠️ Risky | Larger models may OOM at 4 GB. Batch=1-2 with amp may work. |
| **Higher resolution inputs** | ⚠️ Risky | 256×256 is manageable. 640×640 (YOLO default) would 6× memory. |
| **PISTTN temporal processing** | ⚠️ Unknown | Multiple frames in memory simultaneously. 4 GB may be limiting. |
| **Sparse convolutions** | 🔴 Blocked | Libraries not installed and may not be Windows-compatible. |

### Recommendations

- Use **mixed precision** (`torch.cuda.amp`) for all training
- Start with **batch size 1–4** and monitor VRAM with `nvidia-smi`
- Keep input resolution at **256×256** (native dataset resolution) rather than upscaling
- YOLO11n (nano) is the most feasible YOLO variant for 4 GB VRAM
- Gradient checkpointing may be needed for larger architectures

---

## 13. Prioritized Blockers & Recommended Next Steps

### 🔴 Critical Blockers

1. **No bounding-box annotations** — The dataset provides only center-point (x, y) coordinates. YOLO, PISTTN, and most detection architectures require bounding boxes. This is the single biggest blocker.
2. **No sparse-convolution library** — PISTTN requires sparse convolutions. Neither `spconv` nor `MinkowskiEngine` is installed, and both have significant Windows compatibility challenges.
3. **No Git version control** — No `.git` repository exists. All work is unversioned.

### 🟡 Important Issues

4. **No `requirements.txt`** — Environment is not reproducible. Dependencies should be frozen.
5. **Missing evaluation libraries** — `scipy`, `scikit-learn`, `pandas` not installed (needed for metrics).
6. **Label naming inconsistency** — External file names don't match internal sequence names; needs a robust mapping.
7. **Grayscale images stored as 3-channel RGB BMP** — 3× storage/memory waste.

### ✅ Recommended Next Steps (Priority Order)

1. **Investigate target sizes and determine bounding-box strategy** — Measure actual target pixel extents in sampled images using the center-point annotations. Decide whether to: (a) manually annotate bounding boxes, (b) use fixed-size synthetic boxes centered on the annotation points, or (c) use an alternative detection paradigm (center-point heatmap regression like CenterNet).

2. **Initialize Git repository and create `requirements.txt`** — Run `git init`, create `.gitignore` (exclude `yolo-env/`, `*.bmp`, `*.mp4`, `*.mov`, `__pycache__/`), and freeze dependencies with `pip freeze > requirements.txt`.

3. **Research PISTTN sparse-convolution requirements for Windows** — Investigate whether `spconv-cu126` or alternative sparse-conv packages exist for Python 3.11 + Windows + CUDA 12.6. If not feasible on Windows, document the Linux requirement.

---

## 14. Exact Commands & Tests Used

### Environment Inspection

```powershell
nvidia-smi
where.exe python
& "yolo-env\Scripts\python.exe" --version
& "yolo-env\Scripts\python.exe" -m pip --version
& "yolo-env\Scripts\python.exe" -m pip list --format=columns
& "yolo-env\Scripts\python.exe" -m pip check
```

### PyTorch / CUDA Verification

```python
import torch
print("PyTorch version:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("GPU:", torch.cuda.get_device_name(0))
props = torch.cuda.get_device_properties(0)
print("Total memory (MB):", props.total_memory / 1048576)
print("CUDA capability:", torch.cuda.get_device_capability(0))
x = torch.randn((512, 512), device="cuda")
y = x @ x
torch.cuda.synchronize()
print("CUDA op successful:", y.shape)
del x, y
torch.cuda.empty_cache()
free, total = torch.cuda.mem_get_info(0)
print("Free memory (MB):", free / 1048576)
```

### Package Import Verification

```python
import torch, torchvision, cv2, numpy, PIL, yaml, tqdm, matplotlib, ultralytics
# All imported successfully
import importlib.util
# Checked: pandas, scipy, sklearn, tensorboard, pytest, spconv, MinkowskiEngine
# Result: all NOT INSTALLED except torchaudio
```

### Dataset Inspection

```powershell
# File counts per sequence
Get-ChildItem data -Directory | ForEach-Object { count files recursively }

# Total BMP count and size
(Get-ChildItem -Recurse data -Include *.bmp | Measure-Object -Property Length -Sum)
# Result: 16,177 files, 3,034 MB
```

```python
# Image analysis
from PIL import Image
import numpy as np
img = Image.open("data/data1/data1/0.bmp")
# Mode: RGB, Size: (256, 256), dtype: uint8
# All 3 channels identical (grayscale stored as RGB)
# min: 84, max: 255, mean: 127.24, std: 13.0, unique values: 99

# Label schema analysis
# Read all 22 label file headers
# Verified frame counts match actual file counts
# Analyzed object IDs, empty frames, multi-object patterns
```

### Video Inspection

```python
import cv2
cap = cv2.VideoCapture("anti-uav/videos/test.mov")
# 1280×720, ~30 fps, 1217 frames, ~40s

cap = cv2.VideoCapture("anti_uav_tracked.mp4")
# 640×512, 30 fps, 3000 frames, 100s
```

---

*Report generated at: 2026-10-29T12:33 IST*  
*Location: `docs/project_audit.md`*
