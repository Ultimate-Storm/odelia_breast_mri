# ODELIA Breast MRI — Preprocessing & Swarm Training Guide

## Environment

| Item | Value |
|------|-------|
| Project root | `/home/jeff/Projects/odelia_breast_mri` |
| Virtual environment | `./venv` |
| Raw data | `/mnt/nvme2n1p1/jeff/duke_dataset/` |
| Processed data | `/mnt/nvme2n1p1/jeff/DUKE/` |
| Swarm learning outputs | `/mnt/nvme2n1p1/jeff/DUKE_swarm/` |

### Activate environment
```bash
cd /home/jeff/Projects/odelia_breast_mri
source venv/bin/activate
```

---

## Full Pipeline Overview

```
Step 1  DICOM → NIfTI              step1_dicom2nifti.py
Step 1b Rename sequences           step1b_rename.py
Step 2  Subtraction images         step2_compute_sub.py
Step 3  Unilateral split + resize  step3_unilateral.py
Step 4  Annotation + 5-fold split  step4_create_split.py
Step 5  Swarm node partitions      step5_create_swarm_split.py
Step 6  Train (one model per node) main_train.py
Step 7  Evaluate on test set       main_predict.py
```

---

## Step 1 — Convert DICOM to NIfTI

Converts each patient's raw DICOM series into NIfTI (`.nii.gz`) files, named by sequence type (Pre, Post_1, Post_2, …, Post_4, T1).

**Input:** `/mnt/nvme2n1p1/jeff/duke_dataset/data_raw/`
**Output:** `/mnt/nvme2n1p1/jeff/DUKE/data/`
**Requires:** `Breast-Cancer-MRI-filepath_filename-mapping.xlsx` in `/mnt/nvme2n1p1/jeff/DUKE/metadata/`

```bash
python scripts/preprocessing/duke/step1_dicom2nifti.py
```

Each patient directory ends up with:
```
DUKE/data/<PatientID>/
├── Pre.nii.gz
├── Post_1.nii.gz
├── Post_2.nii.gz
├── Post_3.nii.gz
├── Post_4.nii.gz
└── T1.nii.gz
```

---

## Step 2 — Compute Subtraction Images

Creates `Sub_1.nii.gz = Post_1 − Pre` for each patient. The subtraction image highlights contrast-enhancing (potentially malignant) tissue by removing background signal.

**Input:** `/mnt/nvme2n1p1/jeff/DUKE/data/` (requires `Pre.nii.gz`, `Post_1.nii.gz`)
**Output:** `Sub_1.nii.gz` added to each patient directory

```bash
python scripts/preprocessing/step2_compute_sub.py
```

Uses multiprocessing by default. After this step each patient directory has `Sub_1.nii.gz` alongside the original volumes.

---

## Step 3 — Unilateral Split + Standardised Resampling

Splits each bilateral MRI volume into separate left and right breast volumes and standardises the spatial resolution and field of view. This is the most compute-intensive preprocessing step.

**Input:** `/mnt/nvme2n1p1/jeff/DUKE/data/`
**Output:** `/mnt/nvme2n1p1/jeff/DUKE/data_unilateral/`

```bash
python scripts/preprocessing/step3_unilateral.py
```

**What it does per patient:**
1. Resamples to `(0.7, 0.7, 3) mm` voxel spacing
2. Pads/crops to `512 × 512 × 32` voxels
3. Applies intensity-guided height crop to centre the breast (target height 256)
4. Splits the volume down the midline → `<PatientID>_left/` and `<PatientID>_right/`
5. Saves all sequences (Pre, Post_1–4, Sub_1, T1) for each side

Output directories named `<PatientID>_left` and `<PatientID>_right`, each containing all `.nii.gz` sequences at `256 × 256 × 32`.

---

## Step 4 — Annotation & 5-Fold Split

Reads the clinical metadata spreadsheet to build a per-breast annotation table and a 5-fold stratified cross-validation split.

**Input:** `/mnt/nvme2n1p1/jeff/DUKE/metadata/Clinical_and_Other_Features.xlsx`
**Output:**
- `/mnt/nvme2n1p1/jeff/DUKE/metadata/annotation.csv`
- `/mnt/nvme2n1p1/jeff/DUKE/metadata/split.csv`

```bash
python scripts/preprocessing/duke/step4_create_split.py
```

**annotation.csv** — one row per breast (UID):

| Column | Description |
|--------|-------------|
| UID | `<PatientID>_left` or `<PatientID>_right` |
| PatientID | Source patient ID |
| Age | Age in days (absolute value) |
| Lesion | `0` = no lesion, `2` = malignant |

**split.csv** — one row per (UID, fold):

| Column | Description |
|--------|-------------|
| UID | Breast UID |
| Fold | Cross-validation fold index (0–4) |
| Split | `train`, `val`, or `test` |

**Dataset statistics:**
- 651 patients, 1302 breasts
- 623 no-lesion breasts, 679 malignant breasts
- 5-fold `StratifiedGroupKFold` — stratified by lesion type, grouped by PatientID (no patient leakage across splits)

---

## Step 5 — Swarm Learning Node Partitions

Divides the 651 patients into four non-overlapping partitions that simulate swarm/federated learning nodes. No data is copied — each node directory symlinks to the shared `data_unilateral/` folder.

**Input:** `/mnt/nvme2n1p1/jeff/DUKE/metadata/annotation.csv`
**Output:** `/mnt/nvme2n1p1/jeff/DUKE_swarm/`

```bash
python scripts/preprocessing/duke/step5_create_swarm_split.py
```

**Partition proportions** (stratified by bilateral vs. unilateral cancer, `random_state=42`):

| Node | Patients | % | Train UIDs | Val UIDs | Purpose |
|------|----------|---|------------|----------|---------|
| node_A | 260 | 40% | 416 | 104 | Largest training node |
| node_B | 195 | 30% | 312 | 78 | Medium training node |
| node_C | 65 | 10% | 104 | 26 | Smallest training node |
| test | 131 | 20% | — | 262 | Held-out evaluation only |

**Splitting algorithm** (sequential stratified peeling):
```
Step A: test   = 20% of 651         → 131 test, 520 pool
Step B: node_A = 50% of pool        → 260 node_A, 260 rest
Step C: node_B = 75% of rest        → 195 node_B, 65 node_C
Within each training node: 80/20 train/val (patient-level, stratified)
```

**Output structure per node:**
```
DUKE_swarm/
├── node_A/
│   ├── metadata_unilateral/
│   │   ├── annotation.csv    (UID, PatientID, Age, Lesion)
│   │   └── split.csv         (UID, Fold=0, Split=train|val)
│   └── data_unilateral  →  symlink → DUKE/data_unilateral/
├── node_B/  (same structure)
├── node_C/  (same structure)
└── test/
    ├── metadata_unilateral/
    │   ├── annotation.csv
    │   └── split.csv         (all UIDs: Fold=0, Split=test)
    └── data_unilateral  →  symlink
```

The `metadata_unilateral/` directory name is what `ODELIA_Dataset3D` expects for `config="unilateral"`.

---

## Step 6 — Training (One Model per Node)

Trains an MST (Multi-Slice Transformer) model independently on each swarm node. The model is trained for binary classification: no malignant lesion (0) vs. malignant (1).

**Why `binary` and not `ordinal`:** DUKE has no benign (class 1) cases — only no-lesion (0) and malignant (2). Ordinal regression requires all intermediate classes to be present during training; binary classification is the correct formulation here.

**Model — MST (Multi-Slice Transformer):**
- Backbone: DINOv3-ViT-S/16 (pretrained on natural images, fine-tuned on MRI)
- Slice fusion: 1-layer transformer over the 32 axial slices
- Input per sample: `Sub_1.nii.gz` at `224 × 224 × 32`, normalised per-volume
- Output: binary logit → sigmoid → probability of malignancy

**Training hyperparameters (from `main_train.py`):**

| Parameter | Value |
|-----------|-------|
| Batch size | 8 |
| Optimizer | Adam |
| Max epochs | 1000 |
| Early stopping | patience=25 on `val/AUC_ROC` |
| Precision | 16-mixed (FP16) |
| Augmentation | random flip, random 90° rotate, Gaussian noise |
| `log_every_n_steps` | `min(50, steps_per_epoch)` |

**Run training (sequentially — each waits for the previous to finish):**

```bash
cd /home/jeff/Projects/odelia_breast_mri
PYTHON=venv/bin/python
SWARM=/mnt/nvme2n1p1/jeff/DUKE_swarm

# Node A  (~416 train UIDs, ~52 steps/epoch)
WANDB_MODE=offline PYTHONPATH=$(pwd) $PYTHON scripts/main_train.py \
    --institution node_A --model MST --task binary \
    --config unilateral --fold 0 --backbone dinov2 \
    --path_root $SWARM --out_root $SWARM

# Node B  (~312 train UIDs, ~39 steps/epoch)
WANDB_MODE=offline PYTHONPATH=$(pwd) $PYTHON scripts/main_train.py \
    --institution node_B --model MST --task binary \
    --config unilateral --fold 0 --backbone dinov2 \
    --path_root $SWARM --out_root $SWARM

# Node C  (~104 train UIDs, ~13 steps/epoch)
WANDB_MODE=offline PYTHONPATH=$(pwd) $PYTHON scripts/main_train.py \
    --institution node_C --model MST --task binary \
    --config unilateral --fold 0 --backbone dinov2 \
    --path_root $SWARM --out_root $SWARM
```

> **Note:** `PYTHONPATH=$(pwd)` is required because running `python scripts/main_train.py` sets `sys.path[0]` to the `scripts/` directory rather than the project root, making `import odelia` fail. Alternatively, activate the venv and run `pip install -e .` once to install the package properly.

**Checkpoints** are saved to:
```
DUKE_swarm/runs/<node>/MST_binary_unilateral_<timestamp>_fold0/
├── best_checkpoint.json    # filename of the best epoch checkpoint
├── epoch=X-step=Y.ckpt     # best checkpoint
└── last.ckpt               # last epoch checkpoint
```

**WandB logs** are written offline to `wandb/` in the project root. To sync later:
```bash
wandb sync wandb/offline-run-*/
```

---

## Step 7 — Evaluation on the Test Set

Runs inference with each trained model on the held-out 20% test set and computes AUC-ROC, sensitivity, and specificity.

```bash
cd /home/jeff/Projects/odelia_breast_mri
source venv/bin/activate
SWARM=/mnt/nvme2n1p1/jeff/DUKE_swarm

for NODE in node_A node_B node_C; do
    RUN_DIR=$(ls -td ${SWARM}/runs/${NODE}/MST_binary_unilateral_* | head -1)
    CKPT=$(python3 -c "import json; print(json.load(open('${RUN_DIR}/best_checkpoint.json'))['best_model_epoch'])")
    python scripts/main_predict.py \
        --path_run ${RUN_DIR}/${CKPT} \
        --test_institution test \
        --path_root $SWARM \
        --out_root $SWARM
done
```

**Results** are written to:
```
DUKE_swarm/results/<node>/MST_binary_unilateral_<timestamp>_fold0/test/
├── results.csv         # per-UID: GT, NN prediction, NN probability
└── roc_conf_Lesion.png # ROC curve + confusion matrix
```

**Metrics reported per model:**
- AUC-ROC for the Malignant class
- Sensitivity @ Specificity=0.90
- Specificity @ Sensitivity=0.90
- Accuracy, macro sensitivity, macro specificity

### Step 7b — Train and evaluate combined node (node_all)

To establish a centralised baseline, merge all three nodes' data into a single `node_all` partition and train one model:

```bash
cd /home/jeff/Projects/odelia_breast_mri
source venv/bin/activate
SWARM=/mnt/nvme2n1p1/jeff/DUKE_swarm

# Train on combined data (node_A + node_B + node_C = 832 train UIDs)
WANDB_MODE=offline PYTHONPATH=$(pwd) venv/bin/python scripts/main_train.py \
    --institution node_all --model MST --task binary \
    --config unilateral --fold 0 --backbone dinov2 \
    --path_root $SWARM --out_root $SWARM

# Evaluate on held-out test set
RUN_DIR=$(ls -td ${SWARM}/runs/node_all/MST_binary_unilateral_* | head -1)
CKPT=$(python3 -c "import json; print(json.load(open('${RUN_DIR}/best_checkpoint.json'))['best_model_epoch'])")
WANDB_MODE=offline PYTHONPATH=$(pwd) venv/bin/python scripts/main_predict.py \
    --path_run ${RUN_DIR}/${CKPT} \
    --test_institution test \
    --path_root $SWARM \
    --out_root $SWARM
```

The `node_all` metadata was created by concatenating annotation and split CSVs from node_A, node_B, and node_C (preserving their existing train/val assignments).

### Actual Results (MST + DINOv2, binary task, test set = 262 breasts)

| Node | Train UIDs | Best epoch | val AUC | **test AUC** | Sens@Spec0.90 | Spec@Sens0.90 | Accuracy |
|------|-----------|-----------|---------|-------------|---------------|---------------|----------|
| node_A | 416 | 9 | 0.83 | **0.89** | 0.75 | 0.65 | 0.81 |
| node_B | 312 | 51 | 0.95 | **0.90** | 0.78 | 0.59 | 0.77 |
| node_C | 104 | 13 | 0.89 | **0.55** | 0.09 | 0.11 | 0.56 |
| **node_all** | **832** | **39** | — | **0.91** | **0.79** | **0.67** | **0.84** |

**Observations:**
- **node_all (0.91)** is the centralised upper bound — pooling all 832 train UIDs yields a modest but consistent gain over any individual node
- node_A and node_B match within noise (0.89/0.90) despite a 33% size difference — DINOv2 pretraining transfers well even from smaller datasets
- node_C (104 UIDs / 65 patients) collapses to near chance (0.55) — too few samples to fine-tune 23.5M parameters; its val AUC (0.89 on 26 UIDs) was misleadingly optimistic
- The swarm gap vs. centralised: ~0.01 AUC (0.895 avg of A+B vs. 0.91 node_all), typical for federated learning on balanced partitions

---

## Installed Dependencies

| Package | Version |
|---------|---------|
| PyTorch | 2.5.1 (CUDA 12.1) |
| PyTorch Lightning | 2.4.0 |
| TorchIO | 0.20.23 |
| MONAI | 1.5.2 |
| SimpleITK | 2.5.3 |
| PyDICOM | 3.0.2 |
| NumPy / Pandas / Scikit-learn | latest compatible |

---

## Troubleshooting

### Step 1
- Ensure `Breast-Cancer-MRI-filepath_filename-mapping.xlsx` is in the metadata folder
- DICOM files must exist under `data_raw/` subdirectories

### Step 2
- Verify step 1 completed: each patient dir should have `Pre.nii.gz` and `Post_1.nii.gz`
- Check disk space (subtraction images are typically smaller than source volumes)

### Step 3
- If a patient fails, it is skipped silently by multiprocessing; re-run single-CPU option to surface errors:
  ```python
  # In step3_unilateral.py, comment out Pool() block and uncomment:
  for path_dir in tqdm(path_patients):
      preprocess(path_dir)
  ```

### Step 5
- Symlinks use absolute paths; moving `DUKE_swarm/` to another machine requires re-running step 5 with updated paths

### Step 6 — Training
- **`val/AUC_ROC` not improving:** For node_C the validation set is small (~26 UIDs). Occasional AUC plateaus are expected; early stopping handles this automatically.
- **GPU OOM:** Reduce batch size in `main_train.py` (line `batch_size = 8`)
- **WandB login prompt:** Use `WANDB_MODE=offline` prefix or `WANDB_MODE=disabled` to skip entirely
- **`log_every_n_steps` warning:** Already fixed — the value is now `min(50, steps_per_epoch)`
- **`GatedRepoError` / 401 Unauthorized (DINOv3):** The default DINOv3 backbone (`facebook/dinov3-vits16-pretrain-lvd1689m`) is on a gated HuggingFace repo. Use `--backbone dinov2` (DINOv2 with registers, publicly accessible, similar architecture) or `--backbone resnet` (torchvision ResNet-34 2D backbone, no HuggingFace required). To use DINOv3: run `huggingface-cli login` and accept the model terms at huggingface.co first.

### Step 7 — Evaluation
- `results.csv` `GT` and `NN` columns are stored as Python list strings; use `ast.literal_eval` to parse
- If the run directory has multiple checkpoints, `best_checkpoint.json` always points to the best `val/AUC_ROC` epoch
