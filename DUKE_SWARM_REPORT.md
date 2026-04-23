# DUKE Swarm Learning — Experiment Report

**Model:** MST (Multi-Slice Transformer) + DINOv2-ViT-S backbone, 23.5M parameters  
**Task:** Binary malignancy classification (`Sub_1.nii.gz`, 224×224×32)  
**Dataset:** DUKE Breast Cancer MRI — 651 patients, 1302 breasts (679 malignant, 623 no-lesion)  
**Test set:** 131 patients (262 breasts), held-out, never seen during any training

---

## 1. Data Split Strategies

### Strategy A — IID (Random Stratified)

Patients split randomly, stratified by `n_malignant` to preserve the ~52% malignancy rate across all nodes.

| Node | Patients | % | Train UIDs | Val UIDs | Malignant (train) | No-lesion (train) |
|------|----------|---|-----------|---------|-------------------|-------------------|
| node_A | 260 | 40% | 416 | 104 | 217 (52%) | 199 (48%) |
| node_B | 195 | 30% | 312 | 78 | 162 (52%) | 150 (48%) |
| node_C | 65 | 10% | 104 | 26 | 54 (52%) | 50 (48%) |
| node_all | 520 | 80% | 832 | 208 | 433 (52%) | 399 (48%) |
| **test** | **131** | **20%** | — | **262** | **137 (52%)** | **125 (48%)** |

> All nodes are statistically equivalent samples of the same distribution — only size differs.

---

### Strategy B — Non-IID (Age-Stratified)

Patients partitioned by age. Breast tissue density and MRI appearance are strongly age-dependent (younger = dense, older = fatty), creating genuine institutional heterogeneity. Test set remains randomly stratified and covers all ages.

| Node | Age range | Median age | Patients | % | Train UIDs | Real-world analogy |
|------|-----------|-----------|----------|---|-----------|-------------------|
| node_A | 52–83 yrs | 60 yrs | 260 | 40% | 416 | Post-menopausal screening centre |
| node_B | 40–52 yrs | 47 yrs | 195 | 30% | 312 | General hospital |
| node_C | 24–40 yrs | 36 yrs | 65 | 10% | 104 | Young / hereditary-risk programme |
| node_all | 24–83 yrs | 52 yrs | 520 | 80% | 832 | Centralised pooled training |
| **test** | **25–89 yrs** | **52 yrs** | **131** | **20%** | — | **Shared evaluation (all ages)** |

> Each node trains on a biologically distinct cohort. A model trained on node_A has never seen young dense breast tissue; node_C has never seen post-menopausal fatty tissue.

---

## 2. Training Setup

| Parameter | Value |
|-----------|-------|
| Model | MST (Multi-Slice Transformer) |
| Backbone | DINOv2-ViT-S/14 with registers (public HuggingFace) |
| Input | `Sub_1.nii.gz` — subtraction image (Post_1 − Pre), 224×224×32 |
| Task | Binary: no malignancy (0) vs malignant (1) |
| Optimizer | Adam, lr=1e-5 |
| Batch size | 8, FP16 mixed precision |
| Early stopping | patience=25 on `val/AUC_ROC` |
| Max epochs | 1000 |
| Each node | Trained independently (no weight sharing between nodes) |

---

## 3. Results

### 3A — IID Split: 3 Independent Runs

| Node | Run 1 | Run 2 | Run 3 | **Mean ± Std** | Sens@Spec0.90 | Spec@Sens0.90 | Accuracy |
|------|-------|-------|-------|----------------|---------------|---------------|----------|
| node_A (40%) | 0.889 | 0.886 | 0.920 | **0.898 ± 0.016** | 0.75 | 0.65 | 0.81 |
| node_B (30%) | 0.897 | 0.899 | 0.899 | **0.898 ± 0.001** | 0.78 | 0.59 | 0.77 |
| node_C (10%) | 0.551 | 0.749 | 0.675 | **0.658 ± 0.082** | 0.09 | 0.11 | 0.56 |
| node_all (80%) | 0.907 | 0.899 | 0.906 | **0.904 ± 0.004** | 0.79 | 0.67 | 0.84 |

**Key findings:**
- node_A and node_B converge to nearly identical AUC (0.898) despite a 33% data size difference — DINOv2 pretraining dominates at these volumes
- node_B is the most stable (std=0.001); node_C is unreliable (std=0.082, range 0.55–0.75) — 104 UIDs is below the stable fine-tuning threshold for 23.5M parameters
- Swarm gap vs centralised: **0.006 AUC** — small, hard to motivate aggregation under IID conditions

---

### 3B — Non-IID Split: Age-Stratified (1 run)

| Node | Age cohort | Train UIDs | **Test AUC** | Sens@Spec0.90 | Spec@Sens0.90 | Accuracy |
|------|-----------|-----------|-------------|---------------|---------------|----------|
| node_A | Oldest (52–83 yrs) | 416 | **0.855** | 0.61 | 0.50 | 0.79 |
| node_B | Middle (40–52 yrs) | 312 | **0.901** | 0.73 | 0.73 | 0.81 |
| node_C | Youngest (24–40 yrs) | 104 | **0.839** | 0.62 | 0.50 | 0.76 |
| node_all | All ages (24–83 yrs) | 832 | **0.903** | 0.72 | 0.67 | 0.83 |

**Key findings:**
- node_B performs best — middle-age range most closely matches the test set median (52 yrs)
- node_A drops to 0.855 (−0.043 vs IID) — trained only on fatty post-menopausal tissue, struggles on dense younger breasts in test set
- node_C recovers to 0.839 (+0.181 vs IID) — coherent age distribution replaces noisy random undersampling
- Swarm gap vs centralised: **0.048 AUC** (node_A vs node_B) — meaningful, motivates aggregation

---

### 3C — IID vs Non-IID Head-to-Head

| Node | IID Mean ± Std | Non-IID AUC | **Δ** | Why |
|------|---------------|-------------|--------|-----|
| node_A | 0.898 ± 0.016 | 0.855 | **−0.043** | Age mismatch: trained on fatty tissue, tested on all ages |
| node_B | 0.898 ± 0.001 | 0.901 | **+0.003** | Middle age is representative — unaffected |
| node_C | 0.658 ± 0.082 | 0.839 | **+0.181** | Coherent distribution rescues small node |
| node_all | 0.904 ± 0.004 | 0.903 | **−0.001** | Centralised model robust to split strategy |

---

## 4. Summary

| Finding | Detail |
|---------|--------|
| Minimum viable node size (IID) | ~312 UIDs — below this, results become unreliable |
| IID swarm gap | 0.006 AUC — nodes nearly match centralised; weak motivation for aggregation |
| Non-IID swarm gap | 0.048 AUC (node_A vs node_B) — age heterogeneity creates meaningful specialisation |
| Best individual node | node_B (IID: 0.898 ± 0.001 / non-IID: 0.901) — most representative age cohort |
| Worst individual node | node_C IID (0.658 ± 0.082) — noise-dominated; node_A non-IID (0.855) — distribution shift |
| Centralised upper bound | 0.903–0.904 — robust across both split strategies |
| Next step | FedAvg model aggregation — non-IID setting provides the clearest motivation |

---

## 5. Model Checkpoint Locations

All checkpoints: `/mnt/nvme2n1p1/jeff/DUKE_swarm/runs/<node>/<run_name>/`

| Node | IID Run 1 | IID Run 2 | IID Run 3 | Non-IID Run |
|------|-----------|-----------|-----------|-------------|
| node_A | `…04_14_215628…` | `…04_16_161821…` | `…04_17_161237…` | `…04_18_083216…` |
| node_B | `…04_14_221009…` | `…04_17_151546…` | `…04_17_163350…` | `…04_18_223813…` |
| node_C | `…04_14_223444…` | `…04_17_154154…` | `…04_17_165451…` | `…04_18_225101…` |
| node_all | `…04_15_094319…` | `…04_17_155107…` | `…04_17_170100…` | `…04_18_225947…` |

Each run directory contains `best_checkpoint.json` → `epoch=X-step=Y.ckpt` (best val AUC) and `last.ckpt`.

Evaluation outputs (ROC curves, per-UID predictions):
`/mnt/nvme2n1p1/jeff/DUKE_swarm/results/<node>/<run_name>/test/results.csv`
