# Models and Data — Quick Reference

## Datasets

### DUKE (public)

| Item | Path |
|------|------|
| Raw DICOM | `/mnt/nvme2n1p1/jeff/duke_dataset/data_raw/` |
| NIfTI volumes | `/mnt/nvme2n1p1/jeff/DUKE/data/` |
| Preprocessed unilateral | `/mnt/nvme2n1p1/jeff/DUKE/data_unilateral/` |
| Metadata & annotation | `/mnt/nvme2n1p1/jeff/DUKE/metadata/` |

**651 patients, 1302 UIDs** (left + right breast per patient).  
Labels: `0` = no lesion, `2` = malignant (no benign class).  
Every patient has at least one malignant breast — DUKE is a cancer cohort.

Preprocessed volumes: `<PatientID>_left/Sub_1.nii.gz` and `<PatientID>_right/Sub_1.nii.gz` at 224×224×32 voxels, (0.7, 0.7, 3) mm spacing.

---

### DUKE Swarm Partitions

Root: `/mnt/nvme2n1p1/jeff/DUKE_swarm/`

Each node has its own metadata but **shares the physical NIfTI files** via symlink — no data is copied.

```
DUKE_swarm/
├── node_A/metadata_unilateral/{annotation,split}.csv   ← IID or non-IID assignment
├── node_B/metadata_unilateral/{annotation,split}.csv
├── node_C/metadata_unilateral/{annotation,split}.csv
├── node_all/metadata_unilateral/{annotation,split}.csv  ← union of A+B+C
├── test/metadata_unilateral/{annotation,split}.csv      ← held-out 20%
├── node_*/data_unilateral  →  symlink → DUKE/data_unilateral/
├── runs/     ← model checkpoints
└── results/  ← evaluation outputs
```

Two split strategies have been applied (re-running `step5_create_swarm_split.py` overwrites the CSVs):

| Strategy | node_A | node_B | node_C | Basis |
|----------|--------|--------|--------|-------|
| **IID** | 40%, ~52% mal | 30%, ~52% mal | 10%, ~52% mal | Random stratified by n_malignant |
| **Non-IID (age)** | 40%, 52–83 yrs | 30%, 40–52 yrs | 10%, 24–40 yrs | Sorted by patient age |

The **test set (20%, 131 patients, 262 UIDs)** is identical in both strategies — randomly stratified, all ages.

---

### ODELIA Challenge 2025 (private, multi-institutional)

Root: `/mnt/nvme2n1p1/jeff/ODELIA_Challenge_2025/`

| Institution | Role | Path |
|-------------|------|------|
| CAM | Training | `.../CAM/` |
| MHA | Training | `.../MHA/` |
| RUMC | Training | `.../RUMC/` |
| UKA | Training | `.../UKA/` |
| UMCU | Training | `.../UMCU/` |
| RSH | OOD test | `.../RSH/` |

Each institution: `<inst>/data_unilateral/<UID>/Sub_1.nii.gz` + `metadata_unilateral/{annotation,split}.csv`.  
Downloaded from `ODELIA-AI/ODELIA-Challenge-2025` on HuggingFace (~2.6 GB, 1482 UIDs).

---

## Models

### Architecture — MST (Multi-Slice Transformer)

```
Input: Sub_1.nii.gz  (1 × 224 × 224 × 32)
  │
  ├─ 32 axial slices  (each 224 × 224)
  ├─ DINOv2-ViT-S/14 backbone  →  32 embeddings (dim=384)   [~21M params]
  ├─ prepend CLS token  →  sequence length 33
  ├─ 1-layer Transformer (12 heads, rotary pos emb)          [~1.2M params]
  └─ CLS → Linear → 1 logit → sigmoid → P(malignant)        [~385 params]
                                                              Total: ~23.5M
```

- Backbone: `facebook/dinov2-with-registers-small` (public HuggingFace)
- Task: binary classification — `0` = no malignancy, `1` = malignant
- Input: subtraction image `Sub_1 = Post_1 − Pre`

---

### Checkpoint Locations

All checkpoints: `/mnt/nvme2n1p1/jeff/DUKE_swarm/runs/<node>/<run_dir>/`

Each run directory contains:
```
best_checkpoint.json      ← {"best_model_epoch": "epoch=X-step=Y.ckpt"}
epoch=X-step=Y.ckpt       ← best val/AUC_ROC checkpoint  ← use this for inference
last.ckpt                 ← final epoch
```

#### IID split — 3 independent runs

| Node | Run 1 | Run 2 | Run 3 |
|------|-------|-------|-------|
| node_A | `MST_binary_unilateral_2026_04_14_215628_fold0` | `MST_binary_unilateral_2026_04_16_161821_fold0` | `MST_binary_unilateral_2026_04_17_161237_fold0` |
| node_B | `MST_binary_unilateral_2026_04_14_221009_fold0` | `MST_binary_unilateral_2026_04_17_151546_fold0` | `MST_binary_unilateral_2026_04_17_163350_fold0` |
| node_C | `MST_binary_unilateral_2026_04_14_223444_fold0` | `MST_binary_unilateral_2026_04_17_154154_fold0` | `MST_binary_unilateral_2026_04_17_165451_fold0` |
| node_all | `MST_binary_unilateral_2026_04_15_094319_fold0` | `MST_binary_unilateral_2026_04_17_155107_fold0` | `MST_binary_unilateral_2026_04_17_170100_fold0` |

#### Non-IID split (age-stratified) — 1 run

| Node | Run dir |
|------|---------|
| node_A | `MST_binary_unilateral_2026_04_18_083216_fold0` |
| node_B | `MST_binary_unilateral_2026_04_18_223813_fold0` |
| node_C | `MST_binary_unilateral_2026_04_18_225101_fold0` |
| node_all | `MST_binary_unilateral_2026_04_18_225947_fold0` |

---

## Training

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam, lr=1e-5 |
| Batch size | 8 |
| Max epochs | 1000 |
| Early stopping | patience=25 on `val/AUC_ROC` |
| Precision | FP16 mixed |
| Augmentation | random flip, random 90° rotate, Gaussian noise |
| Backbone | DINOv2-with-registers-small (frozen during training) |

### Training command

```bash
cd /home/jeff/Projects/odelia_breast_mri
source venv/bin/activate
SWARM=/mnt/nvme2n1p1/jeff/DUKE_swarm

WANDB_MODE=offline PYTHONPATH=$(pwd) venv/bin/python scripts/main_train.py \
    --institution node_A \      # node_A | node_B | node_C | node_all
    --model MST \
    --task binary \
    --config unilateral \
    --fold 0 \
    --backbone dinov2 \
    --path_root $SWARM \
    --out_root $SWARM
```

Checkpoints saved to `$SWARM/runs/<institution>/MST_binary_unilateral_<timestamp>_fold0/`.

### Evaluation command

```bash
RUN_DIR=/mnt/nvme2n1p1/jeff/DUKE_swarm/runs/node_A/<run_dir>
CKPT=$(PYTHONPATH=$(pwd) venv/bin/python -c \
  "import json; print(json.load(open('${RUN_DIR}/best_checkpoint.json'))['best_model_epoch'])")

WANDB_MODE=offline PYTHONPATH=$(pwd) venv/bin/python scripts/main_predict.py \
    --path_run ${RUN_DIR}/${CKPT} \
    --test_institution test \          # or: CAM,MHA,RUMC,UKA,UMCU  or: RSH
    --path_root $SWARM \
    --out_root $SWARM
```

Results written to `$SWARM/results/<node>/<run_dir>/<test_institution>/results.csv`.

---

## Results Summary

Full metrics (AUC, Accuracy, Sens@Spec0.90, Spec@Sens0.90) for all runs across all evaluation sets:

```
/mnt/nvme2n1p1/jeff/DUKE_swarm/results/DUKE_swarm_results.xlsx
```

| Sheet | Contents |
|-------|----------|
| IID Results (3 runs) | DUKE internal test — 3 runs + mean ± std |
| Non-IID Results (age split) | DUKE internal test — age-stratified run |
| IID vs Non-IID Comparison | AUC delta per node |
| ODELIA Standard (...) | External eval on CAM,MHA,RUMC,UKA,UMCU — 3 runs + mean ± std |
| ODELIA OOD (RSH) | External OOD eval on RSH — 3 runs + mean ± std |
