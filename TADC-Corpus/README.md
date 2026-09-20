# TADC-Corpus

**TADC-Corpus** is an end-to-end framework for **unsupervised object discovery, vision-language semantic calibration, and iterative pseudo-label refinement**.

The framework combines **DINOv3**, **Qwen-VL**, and **YOLO-based self-training** into a unified three-stage pipeline:

> **Class-agnostic object discovery → Vision-language semantic calibration → Iterative pseudo-label refinement**

The primary objective is to construct high-quality object detection training corpora and models with minimal or no manual bounding-box annotations.

---

## Overview

Conventional object detection systems typically require substantial manual bounding-box annotations. TADC-Corpus generates, calibrates, and progressively refines pseudo-labels through three complementary stages.

```text
                         Input Images
                              │
                              ▼
              ┌──────────────────────────────┐
              │ Stage 1: DINOv3              │
              │ Class-Agnostic Object        │
              │ Discovery                    │
              └──────────────────────────────┘
                              │
                              ▼
                    Candidate Proposals
                              │
                              ▼
              ┌──────────────────────────────┐
              │ Stage 2: Qwen-VL             │
              │ Semantic Calibration         │
              │                              │
              │ • Object verification        │
              │ • Category assignment        │
              │ • Background filtering       │
              │ • Box refinement             │
              └──────────────────────────────┘
                              │
                              ▼
                 Initial YOLO Pseudo-Labels
                              │
                              ▼
              ┌──────────────────────────────┐
              │ Stage 3: YOLO                │
              │ Iterative Self-Training      │
              │                              │
              │ Round 1 → Round 2 → Round 3  │
              │                              │
              │ Adaptive pseudo-label        │
              │ fusion and refinement        │
              └──────────────────────────────┘
                              │
                              ▼
                    Final Detection Model
```

---

## Key Features

* **Class-agnostic object discovery** powered by multi-layer DINOv3 attention and feature representations.
* **Graph-based attention diffusion** (AttnWalk) and PCA saliency fusion for robust objectness estimation without supervision.
* **Vision-language semantic calibration** via Qwen-VL to assign categories and refine bounding boxes.
* **Adaptive Otsu thresholding and temporal decay** to dynamically fuse historical pseudo-labels with new detector predictions.
* **Iterative self-training** for progressive label denoising and corpus expansion.
* **Modular design** allowing each stage to be executed standalone or sequentially.

---

# Pipeline Architecture

### Stage 1 — DINOv3 Class-Agnostic Object Discovery
1. Multi-layer intermediate feature maps and RoPE-aligned attention tokens are extracted from DINOv3.
2. A graph random-walk segmenter (`AttnWalkSegmenter`) computes foreground saliency and binary masks.
3. Connected components generate bounding boxes, followed by Non-Maximum Suppression (NMS).
4. **Output format (`outputs/stage1/json/*.bbox.json`)**:
   ```json
   {
     "image_name": "image_000001",
     "image_size": [640, 480],
     "num_instances": 1,
     "instances": [
       {
         "bbox": [120, 80, 240, 180],
         "area": 43200,
         "centroid": [240.0, 170.0],
         "cluster_id": 0,
         "bbox_xyxy": [120, 80, 360, 260],
         "bbox_normalized": [0.1875, 0.166667, 0.375, 0.375],
         "bbox_xyxy_normalized": [0.1875, 0.166667, 0.5625, 0.541667],
         "id": 1
       }
     ]
   }
   ```

### Stage 2 — Qwen-VL Semantic Calibration
1. Stage 1 candidate boxes are normalized and formatted into structured prompts.
2. Qwen-VL verifies object presence, rejects background candidates, splits overlapping entities, and assigns target categories.
3. Coordinates are converted to the standard YOLO normalized format (`class_id x_center y_center width height`).
4. **Output format (`outputs/stage2/yolo_labels/*.txt`)**:
   ```text
   0 0.375000 0.354167 0.375000 0.375000
   ```

### Stage 3 — Iterative YOLO Self-Training
1. Initial pseudo-labels from Stage 2 are converted into 6-column format `(class_id, x, y, w, h, conf)`.
2. A YOLO detector (e.g. `yolov8s.pt`) trains on the current pseudo-labels.
3. The trained detector predicts on the training corpus.
4. An adaptive Otsu-based fusion strategy (`pseudo_label_merge.py`) filters and merges historical and newly predicted boxes.
5. Repeats across configured rounds (default: 3 rounds) to yield the final weights under `outputs/stage3/unsup_round3/weights/best.pt`.

---

# Repository Structure

```text
TADC-Corpus/
│
├── configs/
│   └── pipeline.yaml               # Master pipeline configuration
│
├── stage1_dinov3/
│   ├── __init__.py
│   ├── attnwalk.py                 # Graph random-walk segmentation algorithm
│   ├── dino_attention.py           # DINOv3 attention hooking and RoPE handling
│   └── run_stage1.py               # Stage 1 execution entry
│
├── stage2_qwen/
│   ├── __init__.py
│   ├── label_converter.py          # Format conversions between Qwen and YOLO
│   ├── qwen_client.py              # DashScope multi-modal client wrapper
│   └── run_stage2.py               # Stage 2 execution entry
│
├── stage3_yolo/
│   ├── __init__.py
│   ├── pseudo_label_merge.py       # Class-aware Otsu thresholding & label fusion
│   └── run_stage3.py               # Stage 3 multi-round self-training loop
│
├── utils/
│   ├── __init__.py
│   └── config.py                   # YAML loader utility
│
├── scripts/
│   └── setup_external_repos.py     # Clones external DINOv3 dependency
│
├── requirements/
│   ├── common.txt                  # Common Python libraries
│   ├── cu126.txt                   # CUDA 12.6 PyTorch specification
│   └── cpu.txt                     # CPU-only fallback specification
│
├── run_pipeline.py                 # Main pipeline CLI entry point
├── environment.yml                 # Conda environment definition (GPU / CUDA 12.6)
├── environment-cpu.yml             # Conda environment definition (CPU)
├── requirements.txt                # Pip requirements entry point
├── .env.example                    # DashScope API key template
├── .gitignore                      # Git ignore patterns
└── README.md
```

---

# Installation & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/sybiuff24bafcoder/TADC-Corpus.git
cd TADC-Corpus
```

### 2. Set Up the Conda Environment

**Option A: Using Conda (Recommended)**
```bash
# Create and activate environment directly from environment.yml (PyTorch 2.7.1 + CUDA 12.6)
conda env create -f environment.yml
conda activate tadc-corpus
```

**Option B: Manual pip installation inside a fresh environment**
```bash
conda create -n tadc-corpus python=3.10 -y
conda activate tadc-corpus

# Install PyTorch with CUDA 12.6 support
pip install torch==2.7.1 torchvision==0.22.1 --extra-index-url https://download.pytorch.org/whl/cu126

# Install remaining dependencies
pip install -r requirements.txt
```

---

# External Dependencies & API Setup

### 1. DINOv3 Setup
Run the automated setup script to clone the official DINOv3 repository and prepare the weight folder:
```bash
python scripts/setup_external_repos.py
```

Place your pretrained DINOv3 checkpoint under `external/dinov3/weights/`:
```text
external/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

### 2. DashScope API Configuration (for Stage 2)
Stage 2 calls Qwen-VL through Alibaba Cloud DashScope:
```bash
cp .env.example .env
```
Open `.env` and set your API key:
```bash
DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
```
Or export it in your shell:
```bash
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
```

---

# Dataset Preparation

Ensure your dataset follows the standard YOLO layout:

```text
datasets/
└── your_dataset/
    ├── images/
    │   ├── train/                  # Required for training & pseudo-labeling
    │   │   ├── 000001.jpg
    │   │   └── ...
    │   └── val/                    # Required for YOLO validation
    │       ├── 000002.jpg
    │       └── ...
    └── labels/
        └── val/                    # Ground-truth annotations for validation
            ├── 000002.txt
            └── ...
```

---

# Configuration

Configure all pipeline settings in `configs/pipeline.yaml`:

```yaml
paths:
  image_dir: "./datasets/your_dataset/images/train"
  dinov3_repo: "./external/dinov3"
  dinov3_checkpoint: "./external/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
  output_root: "./outputs"
  stage2_yolo_labels: "./outputs/stage2/yolo_labels"

dataset:
  # Class index mapping corresponds to list ordering (0 -> aeroplane, 1 -> bicycle, ...)
  classes:
    - aeroplane
    - bicycle
    - bird
    - boat

stage1:
  model_name: "dinov3_vitl16"
  n_layers_fusion: 4
  input_size: 512
  k_neighbors: 15
  beta: 0.30
  max_iter: 10
  softmax_scale: 5.0
  slicing_enabled: true
  overlap_ratio: 0.20
  crop_ratio: 0.60
  min_crop_size: 100
  bbox_area_threshold: 50
  nms_iou_threshold: 0.45

stage2:
  model: "qwen3-vl-flash"
  target_long_edge: 1024
  max_workers: 4
  max_retries: 2
  retry_interval: 3.0
  class_prompts: {}

stage3:
  dataset_root: "./datasets/your_dataset"
  work_dir: "./outputs/stage3"
  base_model: "yolov8s.pt"           # Pretrained weights recommended
  num_rounds: 3
  image_size: 640
  batch_size: 32
  workers: 8
  iou_merge_thr: 0.50
  lmm_init_conf: 1.0
  prediction_conf: 0.05
  decay_rate: 0.80

runtime:
  device: "auto"                    # 'auto', 'cuda', or 'cpu'
  enable_visualization: true
```

---

# Usage

The main driver script is `run_pipeline.py`. Note that the default mode is `--stage 1-2`. To run all stages, specify `--stage all`.

### 1. Run Complete Pipeline (Stages 1 → 2 → 3)
```bash
python run_pipeline.py --config configs/pipeline.yaml --stage all
```

### 2. Run Stage 1 Only (DINOv3 Proposal Discovery)
```bash
python run_pipeline.py --config configs/pipeline.yaml --stage 1
```
*Outputs stored in `outputs/stage1/json/`.*

### 3. Run Stage 2 Only (Qwen-VL Semantic Calibration)
```bash
python run_pipeline.py --config configs/pipeline.yaml --stage 2
```
*Outputs stored in `outputs/stage2/yolo_labels/`.*

### 4. Run Stage 1 + Stage 2 (Generate Initial Pseudo-Label Corpus)
```bash
python run_pipeline.py --config configs/pipeline.yaml --stage 1-2
```

### 5. Run Stage 3 Only (YOLO Iterative Self-Training)
```bash
python run_pipeline.py --config configs/pipeline.yaml --stage 3
```
*Final detector weights stored at `outputs/stage3/unsup_round3/weights/best.pt`.*

---

# Citation

If you use TADC-Corpus in your research, please cite our work:

```bibtex
@unpublished{tadc_corpus2026,
  title     = {From Generic Priors to Task-Adaptive Detection Corpora: Schema-Constrained Construction and Revision},
  author    = {Cai, Kewei and Ji, Xiuwen and Yang, Zhipeng and Feng, Guanbo and Liu, Ying and Han, Jibin},
  year      = {2026},
  note      = {Under review}
}

```

---

# License & Acknowledgements

*This repository is released under the MIT License. See the `LICENSE` file for full terms.
* This work relies on open-source contributions from [DINOv3](https://github.com/facebookresearch/dinov3), [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), and [Ultralytics](https://github.com/ultralytics/ultralytics).
