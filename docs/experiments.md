# DUKE Swarm Learning — Experiment Log

All experiments use **MST + DINOv2-ViT-S/14** (23.5M params), binary task, `Sub_1.nii.gz` input at 224×224×32, early stopping patience=25 on `val/AUC_ROC`.

Test set: **262 breasts / 131 patients** (20% held-out, never seen during training).

---

## Experiment 1 — IID Split (3 runs)

**Split strategy:** Patients randomly assigned to nodes, stratified by `n_malignant` → every node gets ~52% malignant UIDs (IID).

| Node | Patients | Age range | Train UIDs | Malignant rate |
|------|----------|-----------|-----------|----------------|
| node_A | 260 (40%) | mixed | 416 | ~52% |
| node_B | 195 (30%) | mixed | 312 | ~52% |
| node_C | 65 (10%) | mixed | 104 | ~52% |
| node_all | 520 (80%) | mixed | 832 | ~52% |
| test | 131 (20%) | mixed | — | ~52% |

**Results — test AUC across 3 independent runs:**

| Node | Run 1 | Run 2 | Run 3 | Mean ± Std |
|------|-------|-------|-------|------------|
| node_A | 0.889 | 0.886 | 0.920 | **0.898 ± 0.016** |
| node_B | 0.897 | 0.899 | 0.900 | **0.899 ± 0.001** |
| node_C | 0.551 | 0.749 | 0.680 | **0.660 ± 0.081** |
| node_all | 0.907 | 0.900 | 0.910 | **0.906 ± 0.004** |

**Key findings:**
- node_A and node_B converge similarly despite a 33% size difference — DINOv2 pretraining dominates at these data sizes
- node_C is highly unstable (std=0.081) — 104 UIDs / 65 patients is below the stable fine-tuning threshold
- node_all (centralised baseline) gives only a marginal gain (~0.007 AUC) over best individual node

**Limitation:** IID splits produce nearly identical per-node distributions, masking the real challenge of federated learning across heterogeneous institutions.

---

## Experiment 2 — Non-IID Age-Stratified Split (1 run)

**Split strategy:** Patients sorted by age and carved into non-overlapping age bands. Age drives breast tissue density (young = dense, old = fatty), creating genuine heterogeneity between nodes. Test set remains randomly stratified (all ages).

| Node | Patients | Age range | Median age | Train UIDs | Real-world analogy |
|------|----------|-----------|------------|-----------|-------------------|
| node_A | 260 (40%) | 52–83 yrs | 60 yrs | 416 | Post-menopausal screening centre |
| node_B | 195 (30%) | 40–52 yrs | 47 yrs | 312 | General hospital |
| node_C | 65 (10%) | 24–40 yrs | 36 yrs | 104 | Young/hereditary-risk programme |
| node_all | 520 (80%) | 24–83 yrs | 52 yrs | 832 | Centralised baseline |
| test | 131 (20%) | 25–89 yrs | 52 yrs | — | Shared evaluation |

**Results — test AUC (1 run) vs IID mean:**

| Node | Non-IID AUC | IID Mean | Δ | Interpretation |
|------|-------------|----------|---|----------------|
| node_A | **0.85** | 0.898 | −0.048 | Struggles with young dense breasts in test set |
| node_B | **0.90** | 0.899 | +0.001 | Middle age overlaps most with test median — best calibrated |
| node_C | **0.84** | 0.660 | +0.180 | Coherent age distribution; IID was hurt by random undersampling |
| node_all | **0.90** | 0.906 | −0.006 | Full age coverage — robust regardless of split strategy |

**Key findings:**
- Non-IID gap between best and worst node: **0.05 AUC** (0.90 − 0.85), larger and more interpretable than IID
- node_A shows the clearest age-mismatch penalty
- node_C's dramatic improvement over IID reveals that its IID instability was due to poor random samples, not an inherent data-scarcity limit
- node_all remains the reliable upper bound (~0.90) confirming the benefit of full-population coverage

---

## External Evaluation — ODELIA Challenge 2025

Both IID and non-IID DUKE models were evaluated on the external [ODELIA-AI/ODELIA-Challenge-2025](https://huggingface.co/datasets/ODELIA-AI/ODELIA-Challenge-2025) dataset (downloaded to `/mnt/nvme2n1p1/jeff/ODELIA_Challenge_2025`).

| Evaluation target | Institutions | Split | Samples |
|-------------------|-------------|-------|---------|
| Standard centres | CAM, MHA, RUMC, UKA, UMCU | test | 260 |
| OOD | RSH | all | 200 |

See detailed results in:
- [docs/odelia_challenge_iid_evaluation.md](odelia_challenge_iid_evaluation.md) — all 12 IID models (3 runs × 4 nodes)
- [docs/odelia_challenge_noniid_evaluation.md](odelia_challenge_noniid_evaluation.md) — latest non-IID models
- [docs/validation_summary.md](validation_summary.md) — consolidated master table

---

## Checkpoints

All model checkpoints are stored under `/mnt/nvme2n1p1/jeff/DUKE_swarm/runs/`:

```
runs/
├── node_A/
│   ├── MST_binary_unilateral_2026_04_14_215628_fold0/   # IID run 1
│   ├── MST_binary_unilateral_2026_04_16_161821_fold0/   # IID run 2
│   ├── MST_binary_unilateral_2026_04_17_043336_fold0/   # IID run 3
│   └── MST_binary_unilateral_2026_04_18_083216_fold0/   # non-IID run
├── node_B/  (same structure)
├── node_C/  (same structure)
└── node_all/ (same structure)
```

Each run dir contains `best_checkpoint.json` pointing to the best `val/AUC_ROC` epoch checkpoint.
