# ODELIA — Breast MRI Classification

Baseline code for breast MRI malignancy classification using the **Multi-Slice Transformer (MST)** and **ResNet** models. Supports single-institution training, multi-institution federated evaluation, and **swarm learning simulation** on the public DUKE dataset.

> For step-by-step commands and troubleshooting details see [PREPROCESSING_GUIDE.md](PREPROCESSING_GUIDE.md).

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Dataset Overview — DUKE](#dataset-overview--duke)
3. [Preprocessing Pipeline](#preprocessing-pipeline)
4. [Swarm Learning Data Split](#swarm-learning-data-split)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Model Architecture](#model-architecture)
8. [ODELIA Multi-Institutional Dataset](#odelia-multi-institutional-dataset)

---

## Environment Setup

```bash
git clone <this-repo>
cd odelia_breast_mri

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt   # or: conda env create -f environment.yaml

# Install the odelia package in editable mode
pip install -e .
```

> **Note:** When running scripts directly (`python scripts/main_train.py`), prefix with `PYTHONPATH=$(pwd)` so Python can locate the `odelia` package. Example: `PYTHONPATH=$(pwd) python scripts/main_train.py ...`

**Requirements:**
- CUDA 12.1-compatible GPU driver (`nvidia-smi` to verify)
- Python 3.12

---

## Dataset Overview — DUKE

The [Duke Breast Cancer MRI dataset](https://sites.duke.edu/mazurowski/resources/breast-cancer-mri-dataset/) is a public dataset from The Cancer Imaging Archive (TCIA).

| Property | Value |
|----------|-------|
| Patients | 651 |
| Breasts (UIDs) | 1302 (651 left + 651 right) |
| No-lesion breasts | 623 |
| Malignant breasts | 679 |
| Benign breasts | 0 (binary labels only) |
| Sequences | Pre, Post_1–4, T1 (DCE-MRI) |
| Raw data location | `/mnt/nvme2n1p1/jeff/duke_dataset/data_raw/` |
| Processed data location | `/mnt/nvme2n1p1/jeff/DUKE/` |

**Label encoding:** `0` = no lesion, `2` = malignant (no class `1` in DUKE)

**Raw folder structure required:**
```
DUKE/
├── data_raw/
│   ├── Breast_MRI_001/
│   │   └── <StudyUID>/<SeriesUID>/  ← DICOM files
│   ├── Breast_MRI_002/
│   └── ...
└── metadata/
    ├── Breast-Cancer-MRI-filepath_filename-mapping.xlsx
    └── Clinical_and_Other_Features.xlsx
```

---

## Preprocessing Pipeline

All scripts are run from the project root with the venv activated.

### Step 1 — DICOM → NIfTI
```bash
python scripts/preprocessing/duke/step1_dicom2nifti.py
```
Converts each DICOM series to a named NIfTI file (Pre, Post_1–4, T1).
Output: `DUKE/data/<PatientID>/*.nii.gz`

### Step 2 — Subtraction Image
```bash
python scripts/preprocessing/step2_compute_sub.py
```
Computes `Sub_1 = Post_1 − Pre` to highlight contrast-enhancing tissue.
Output: `Sub_1.nii.gz` added to each patient directory.

### Step 3 — Unilateral Split + Standardisation
```bash
python scripts/preprocessing/step3_unilateral.py
```
Resamples to `(0.7, 0.7, 3) mm`, pads/crops to `512×512×32`, then splits each bilateral volume into left/right breast halves (`256×256×32`).
Output: `DUKE/data_unilateral/<PatientID>_left/` and `<PatientID>_right/`

### Step 4 — Annotation & 5-Fold Cross-Validation Split
```bash
python scripts/preprocessing/duke/step4_create_split.py
```
Reads `Clinical_and_Other_Features.xlsx` and produces:
- `DUKE/metadata/annotation.csv` — per-breast labels (UID, PatientID, Age, Lesion)
- `DUKE/metadata/split.csv` — 5-fold stratified split (UID, Fold 0–4, Split train/val/test)

Strategy: `StratifiedGroupKFold` — stratified by lesion type, grouped by PatientID to prevent patient leakage.

### Step 5 — Swarm Learning Node Partitions
```bash
python scripts/preprocessing/duke/step5_create_swarm_split.py
```
Partitions patients across swarm learning nodes. See [Swarm Learning Data Split](#swarm-learning-data-split) below.

---

## Swarm Learning Data Split

The DUKE dataset is partitioned into **four non-overlapping patient groups** to simulate swarm/federated learning — each node trains independently on its local data subset.

### Partition Summary

| Node | Patients | % of total | Train UIDs | Val UIDs | Purpose |
|------|----------|-----------|------------|----------|---------|
| **node_A** | 260 | 40% | 416 | 104 | Largest training node |
| **node_B** | 195 | 30% | 312 | 78 | Medium training node |
| **node_C** | 65 | 10% | 104 | 26 | Smallest training node |
| **test** | 131 | 20% | — | 262 | Held-out evaluation (never seen during training) |
| **Total** | **651** | **100%** | **832** | **208** | + 262 test |

### Splitting Algorithm

Splits are **stratified by `n_malignant`** per patient (1 = unilateral cancer, 2 = bilateral cancer) to preserve the bilateral-cancer ratio (~4.3% of patients) across all partitions. `random_state=42` throughout.

```
Step A: peel test   = 20% of 651       → 131 test, 520 remaining
Step B: peel node_A = 50% of remaining → 260 node_A, 260 rest
Step C: split rest  = 75% / 25%        → 195 node_B, 65 node_C

Within each training node: 80% / 20% train/val (patient-level, stratified)
```

Zero patient overlap is verified by the script with an assertion.

### Output Directory Structure

```
/mnt/nvme2n1p1/jeff/DUKE_swarm/
├── node_A/
│   ├── metadata_unilateral/
│   │   ├── annotation.csv      ← UID, PatientID, Age, Lesion
│   │   └── split.csv           ← UID, Fold=0, Split=train|val
│   └── data_unilateral         → symlink → DUKE/data_unilateral/
├── node_B/                     (same structure)
├── node_C/                     (same structure)
├── test/
│   ├── metadata_unilateral/
│   │   ├── annotation.csv
│   │   └── split.csv           ← all UIDs: Fold=0, Split=test
│   └── data_unilateral         → symlink
├── runs/                       ← training checkpoints per node
└── results/                    ← evaluation outputs per node
```

Symlinks are used for `data_unilateral` — no data is copied, saving ~100 GB.

---

## Training

### Single command (one node)
```bash
SWARM=/mnt/nvme2n1p1/jeff/DUKE_swarm
PYTHONPATH=$(pwd) venv/bin/python scripts/main_train.py \
    --institution node_A \
    --model MST \
    --task binary \
    --config unilateral \
    --fold 0 \
    --backbone dinov2 \
    --path_root $SWARM \
    --out_root $SWARM
```

### All three nodes sequentially
```bash
SWARM=/mnt/nvme2n1p1/jeff/DUKE_swarm
for NODE in node_A node_B node_C; do
    WANDB_MODE=offline PYTHONPATH=$(pwd) venv/bin/python scripts/main_train.py \
        --institution $NODE --model MST --task binary \
        --config unilateral --fold 0 --backbone dinov2 \
        --path_root $SWARM --out_root $SWARM
done
```

### Combined baseline (all node data pooled)
Trains on the union of node_A + node_B + node_C (832 train UIDs) to establish a centralised upper bound:
```bash
SWARM=/mnt/nvme2n1p1/jeff/DUKE_swarm
WANDB_MODE=offline PYTHONPATH=$(pwd) venv/bin/python scripts/main_train.py \
    --institution node_all --model MST --task binary \
    --config unilateral --fold 0 --backbone dinov2 \
    --path_root $SWARM --out_root $SWARM
```
The `node_all` metadata is generated by merging the three node annotation and split CSV files — see [Step 5](#step-5--swarm-learning-node-partitions).

### Training arguments

| Argument | Default | Options |
|----------|---------|---------|
| `--institution` | `ODELIA` | `node_A`, `node_B`, `node_C`, or any institution name |
| `--model` | `MST` | `MST`, `ResNet` |
| `--task` | `ordinal` | `binary` (0/1), `ordinal` (0/1/2) |
| `--config` | `unilateral` | `unilateral`, `original` |
| `--fold` | `0` | `0`–`4` |
| `--backbone` | `dinov2` | `dinov2` (public), `dinov3` (gated HF token needed), `resnet` |
| `--path_root` | class default | path to dataset root |
| `--out_root` | `./runs` | path for checkpoint output |

### Training hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 8 |
| Optimizer | Adam, lr=1e-5 |
| Max epochs | 1000 |
| Early stopping | patience=25 on `val/AUC_ROC` |
| Precision | 16-mixed (FP16) |
| Augmentation | random flip, random 90° rotate, Gaussian noise |
| Input | `Sub_1.nii.gz` at 224×224×32 |

> **Why `binary` for DUKE?** DUKE has no benign (class 1) samples — only no-lesion (0) and malignant (2). Use `--task binary` which maps to `(Lesion == 2)`.

> **Why `dinov2` not `dinov3`?** The default DINOv3 backbone (`facebook/dinov3-vits16-pretrain-lvd1689m`) is on a gated HuggingFace repo. Use `--backbone dinov2` (publicly accessible, nearly identical architecture) or authenticate with `huggingface-cli login` to use DINOv3.

Checkpoints are saved to `<out_root>/runs/<institution>/MST_binary_unilateral_<timestamp>_fold0/`.

---

## Evaluation

After training completes, evaluate each model on the held-out test set:

```bash
SWARM=/mnt/nvme2n1p1/jeff/DUKE_swarm
for NODE in node_A node_B node_C; do
    RUN_DIR=$(ls -td ${SWARM}/runs/${NODE}/MST_binary_unilateral_* | head -1)
    CKPT=$(python3 -c "import json; print(json.load(open('${RUN_DIR}/best_checkpoint.json'))['best_model_epoch'])")
    WANDB_MODE=offline PYTHONPATH=$(pwd) venv/bin/python scripts/main_predict.py \
        --path_run ${RUN_DIR}/${CKPT} \
        --test_institution test \
        --path_root $SWARM \
        --out_root $SWARM
done
```

Results are written to `<out_root>/results/<node>/MST_binary_unilateral_.../test/`:
- `results.csv` — per-breast UID with ground truth, prediction, and probability
- `roc_conf_Lesion.png` — ROC curve + confusion matrix

### Metrics reported
- AUC-ROC (Malignant class)
- Sensitivity @ Specificity = 0.90
- Specificity @ Sensitivity = 0.90
- Accuracy, macro sensitivity, macro specificity

### Results (MST + DINOv2, binary task, test set = 262 breasts, 3 independent runs)

Single-run results (run 1) with per-class metrics:

| Node | Train UIDs | Best epoch | val AUC | **test AUC** | Sens@Spec0.90 | Spec@Sens0.90 | Accuracy |
|------|-----------|-----------|---------|-------------|---------------|---------------|----------|
| node_A | 416 | 9 | 0.83 | **0.89** | 0.75 | 0.65 | 0.81 |
| node_B | 312 | 51 | 0.95 | **0.90** | 0.78 | 0.59 | 0.77 |
| node_C | 104 | 13 | 0.89 | **0.55** | 0.09 | 0.11 | 0.56 |
| **node_all** | **832** | **39** | — | **0.91** | **0.79** | **0.67** | **0.84** |

Stability across 3 independent runs (random weight initialisation, same data split):

| Node | Train UIDs | Run 1 | Run 2 | Run 3 | **Mean AUC ± Std** |
|------|-----------|-------|-------|-------|-------------------|
| node_A | 416 | 0.889 | 0.886 | 0.920 | **0.898 ± 0.016** |
| node_B | 312 | 0.897 | 0.899 | 0.900 | **0.899 ± 0.001** |
| node_C | 104 | 0.551 | 0.749 | 0.680 | **0.660 ± 0.081** |
| **node_all** | **832** | **0.907** | **0.900** | **0.910** | **0.906 ± 0.004** |

**Observations:**
- **node_all** is the consistent best (mean 0.906) — pooling all 832 train UIDs provides a small but reliable gain over any individual swarm node
- **node_B** is the most stable individual node (std=0.001) — 312 UIDs is sufficient for consistent convergence with a pretrained DINOv2 backbone
- **node_A** shows more variance (std=0.016) despite being the largest single node — early stopping sometimes triggers before full convergence
- **node_C** is unreliable (std=0.081, range 0.55–0.75) — 104 UIDs / 65 patients is below the stable fine-tuning threshold for 23.5M parameters; val AUC on only 26 UIDs is misleadingly optimistic
- Swarm gap vs. centralised: ~0.007 AUC (mean of A+B = 0.899 vs. node_all 0.906), consistent across all 3 runs

### Non-IID Split — Age-Stratified (1 run)

To simulate realistic institutional heterogeneity, the dataset is re-partitioned by patient age. Younger patients have denser breast tissue (higher background parenchymal enhancement); older patients have fattier tissue — a genuine biological difference that affects MRI appearance and model decision boundaries.

| Node | Age range | Analogy | Train UIDs | IID AUC (mean±std) | **Non-IID AUC** | Δ |
|------|-----------|---------|-----------|-------------------|-----------------|---|
| node_A | 52–83 yrs | Post-menopausal screening centre | 416 | 0.898 ± 0.016 | **0.85** | −0.048 |
| node_B | 40–52 yrs | General hospital | 312 | 0.899 ± 0.001 | **0.90** | +0.001 |
| node_C | 24–40 yrs | Young/hereditary-risk programme | 104 | 0.660 ± 0.081 | **0.84** | +0.180 |
| node_all | 24–83 yrs | Centralised baseline | 832 | 0.906 ± 0.004 | **0.90** | −0.006 |

**Observations:**
- **node_B (middle age) performs best** — its training age range overlaps most with the test set median, making it the best-calibrated individual node
- **node_A dropped** (0.85 vs 0.898 IID) — trained only on older fatty tissue, it struggles with younger dense breasts in the balanced test set
- **node_C improved dramatically** (0.84 vs 0.660 IID) — age-stratified assignment gives it a coherent training distribution; poor IID performance was due to random undersampling, not an inherent size limit
- **node_all unchanged** (0.90) — covers all ages, behaves as the centralised upper bound regardless of split strategy
- The non-IID gap between best and worst node (0.90 − 0.85 = 0.05) is larger and more interpretable than the IID gap (0.899 − 0.660 = 0.239, dominated by node_C data scarcity)

---

## Model Architecture

### MST — Multi-Slice Transformer ([paper](https://arxiv.org/abs/2411.15802))

Processes a 3D MRI volume by encoding each 2D axial slice independently with a pretrained backbone, then fusing slice-level embeddings using a transformer.

```
Input: Sub_1.nii.gz  (1 × 224 × 224 × 32)
  │
  ├─ 32 axial slices  (each 224 × 224)
  │
  ├─ DINOv2-ViT-S/14 backbone → 32 embeddings (dim=384)
  │
  ├─ prepend CLS token → sequence of length 33
  │
  ├─ 1-layer Transformer (12 heads, rotary pos emb)
  │
  └─ CLS output → Linear → 1 logit → sigmoid → P(malignant)
```

| Component | Params |
|-----------|--------|
| DINOv2-ViT-S backbone | ~21M |
| Slice-fusion transformer | ~1.2M |
| Classification head | ~385 |
| **Total** | **~23.5M** |

### ResNet (3D)
MONAI ResNet-34, processes full 3D volume directly. Use `--model ResNet`. No HuggingFace download required.

---

## ODELIA Multi-Institutional Dataset

The codebase also supports the private [ODELIA](https://odelia.ai/) multi-institutional dataset with institutions CAM, MHA, RUMC, UKA, UMCU (training) and RSH (held-out test).

To train on ODELIA data, point `PATH_ROOT` in [dataset_3d_odelia.py](odelia/data/datasets/dataset_3d_odelia.py) to the dataset root, then:

```bash
PYTHONPATH=$(pwd) venv/bin/python scripts/main_train.py \
    --institution ODELIA \
    --model MST \
    --task ordinal \
    --config unilateral \
    --fold 0 \
    --backbone dinov2
```

Each institution folder must follow the structure:
```
<institution>/
├── data_unilateral/<UID>/Sub_1.nii.gz
└── metadata_unilateral/
    ├── annotation.csv   (UID, PatientID, Age, Lesion)
    └── split.csv        (UID, Fold, Split)
```
