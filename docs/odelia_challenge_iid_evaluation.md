# ODELIA Challenge External Evaluation Report

Date: 2026-04-18

## Objective

This report summarizes the external validation of DUKE-trained breast MRI models on the `ODELIA-AI/ODELIA-Challenge-2025` dataset from Hugging Face.

The goal was to answer a simple question:

How well do the DUKE-trained models transfer to a different multi-center dataset, and how does performance vary across local-node models versus the pooled `node_all` model?

## Scope

This report covers the **12 models** from the **original DUKE mixed split**:

- `node_A`: 3 repetitions
- `node_B`: 3 repetitions
- `node_C`: 3 repetitions
- `node_all`: 3 repetitions

These are the same repetitions behind the internal DUKE evaluation table:

| Node | DUKE Run 1 | DUKE Run 2 | DUKE Run 3 | DUKE Mean ± Std |
| --- | ---: | ---: | ---: | ---: |
| `node_A` | 0.889 | 0.886 | 0.920 | 0.898 ± 0.016 |
| `node_B` | 0.897 | 0.899 | 0.900 | 0.899 ± 0.001 |
| `node_C` | 0.551 | 0.749 | 0.680 | 0.660 ± 0.081 |
| `node_all` | 0.907 | 0.900 | 0.910 | 0.906 ± 0.004 |

Important: this report does **not** cover the newer age-stratified non-IID split. It evaluates the older "mixed" DUKE models only.

## Model and Inference Setup

- Model family: `MST`
- Task: binary classification (`malignant` vs `non-malignant`)
- Input configuration: `unilateral`
- Backbone: `DINOv2 with registers (small)`
- Input image used for external evaluation: `Sub_1.nii.gz`

Only `Sub_1` was downloaded from the ODELIA Challenge dataset, because that is the only sequence consumed by the current DUKE MST checkpoints.

## External Dataset

Source:

- Hugging Face dataset: `ODELIA-AI/ODELIA-Challenge-2025`
- Configuration: `unilateral`

Downloaded institutions:

- `CAM`
- `MHA`
- `RSH`
- `RUMC`
- `UKA`
- `UMCU`

Downloaded local path:

- `/mnt/nvme2n1p1/jeff/ODELIA_Challenge_2025`

Downloaded content:

- `1482` unilateral `Sub_1.nii.gz` volumes
- local size approximately `2.6G`

## Evaluation Protocol

Two external evaluations were run for each of the 12 DUKE models:

1. **Standard ODELIA centers**
   - institutions: `CAM,MHA,RUMC,UKA,UMCU`
   - split used: `Split == test`
   - total samples: `260`

2. **RSH OOD evaluation**
   - institution: `RSH`
   - split used: **all available rows**
   - total samples: `200`

Why `RSH` used all rows:

- the current `main_predict.py` has a special case for `RSH`, where it loads `split=None`
- this matches the earlier code assumption that `RSH` is an out-of-distribution institution used separately from standard train/val/test centers

This means the `RSH` results below are useful as an OOD stress test, but they are **not** strictly the same as evaluating only `RSH test`.

## Exact Models Evaluated

### `node_A`

- `MST_binary_unilateral_2026_04_14_215628_fold0`
- `MST_binary_unilateral_2026_04_16_161821_fold0`
- `MST_binary_unilateral_2026_04_17_161237_fold0`

### `node_B`

- `MST_binary_unilateral_2026_04_14_221009_fold0`
- `MST_binary_unilateral_2026_04_17_151546_fold0`
- `MST_binary_unilateral_2026_04_17_163350_fold0`

### `node_C`

- `MST_binary_unilateral_2026_04_14_223444_fold0`
- `MST_binary_unilateral_2026_04_17_154154_fold0`
- `MST_binary_unilateral_2026_04_17_165451_fold0`

### `node_all`

- `MST_binary_unilateral_2026_04_15_094319_fold0`
- `MST_binary_unilateral_2026_04_17_155107_fold0`
- `MST_binary_unilateral_2026_04_17_170100_fold0`

## Results: Standard ODELIA Centers

Evaluation target: `CAM,MHA,RUMC,UKA,UMCU` test split only, `n=260`

| Node | Run 1 | Run 2 | Run 3 | Mean ± Std |
| --- | ---: | ---: | ---: | ---: |
| `node_A` | 0.848 | 0.854 | 0.875 | 0.859 ± 0.014 |
| `node_B` | 0.848 | 0.839 | 0.837 | 0.841 ± 0.006 |
| `node_C` | 0.577 | 0.784 | 0.655 | 0.672 ± 0.105 |
| `node_all` | 0.873 | 0.872 | 0.879 | 0.875 ± 0.004 |

Mean accuracy over the same runs:

| Node | Mean Accuracy |
| --- | ---: |
| `node_A` | 0.738 |
| `node_B` | 0.718 |
| `node_C` | 0.412 |
| `node_all` | 0.844 |

## Results: RSH OOD

Evaluation target: `RSH` all available rows, `n=200`

| Node | Run 1 | Run 2 | Run 3 | Mean ± Std |
| --- | ---: | ---: | ---: | ---: |
| `node_A` | 0.679 | 0.722 | 0.717 | 0.706 ± 0.024 |
| `node_B` | 0.690 | 0.710 | 0.706 | 0.702 ± 0.011 |
| `node_C` | 0.495 | 0.641 | 0.547 | 0.561 ± 0.074 |
| `node_all` | 0.779 | 0.798 | 0.765 | 0.781 ± 0.017 |

Mean accuracy over the same runs:

| Node | Mean Accuracy |
| --- | ---: |
| `node_A` | 0.745 |
| `node_B` | 0.738 |
| `node_C` | 0.568 |
| `node_all` | 0.775 |

## Internal vs External Comparison

| Node | DUKE Internal Mean ± Std | ODELIA Standard Mean ± Std | RSH OOD Mean ± Std |
| --- | ---: | ---: | ---: |
| `node_A` | 0.898 ± 0.016 | 0.859 ± 0.014 | 0.706 ± 0.024 |
| `node_B` | 0.899 ± 0.001 | 0.841 ± 0.006 | 0.702 ± 0.011 |
| `node_C` | 0.660 ± 0.081 | 0.672 ± 0.105 | 0.561 ± 0.074 |
| `node_all` | 0.906 ± 0.004 | 0.875 ± 0.004 | 0.781 ± 0.017 |

## Interpretation

### 1. `node_all` is the best and most stable external model

Across both external targets, `node_all` remained the strongest performer:

- standard ODELIA: `0.875 ± 0.004`
- `RSH`: `0.781 ± 0.017`

This is consistent with the DUKE internal study: pooling data across all DUKE nodes produces the most robust representation.

### 2. `node_A` transfers slightly better than `node_B`

Internally, `node_A` and `node_B` were almost tied on DUKE:

- `node_A`: `0.898 ± 0.016`
- `node_B`: `0.899 ± 0.001`

Externally, `node_A` transferred somewhat better:

- standard ODELIA: `0.859` vs `0.841`
- `RSH`: `0.706` vs `0.702`

The gap is modest, but consistent enough to suggest that the data seen by `node_A` may have induced slightly more transferable features than `node_B`.

### 3. `node_C` remains unstable

`node_C` is still the weakest node by far:

- standard ODELIA: `0.672 ± 0.105`
- `RSH`: `0.561 ± 0.074`

This mirrors the original DUKE findings: the smallest node has the largest variance and the poorest robustness.

### 4. OOD shift at `RSH` is real

Every node drops on `RSH` relative to the standard ODELIA centers:

- `node_A`: `0.859 -> 0.706`
- `node_B`: `0.841 -> 0.702`
- `node_C`: `0.672 -> 0.561`
- `node_all`: `0.875 -> 0.781`

This supports the intended role of `RSH` as an out-of-distribution evaluation site.

### 5. External ranking follows the expected pooled-data trend

The external ranking is still:

`node_all` > `node_A` ≈ `node_B` >> `node_C`

So the central conclusion from DUKE persists on ODELIA:

- pooled training gives the strongest model
- the weakest node remains brittle
- external transfer preserves the broad ordering of model quality

## Artifacts and Logs

External evaluation root:

- `/mnt/nvme2n1p1/jeff/ODELIA_Challenge_eval_all12`

Batch log:

- `/mnt/nvme2n1p1/jeff/ODELIA_Challenge_eval_all12/eval_all12.log`

Per-model prediction outputs:

- `/mnt/nvme2n1p1/jeff/ODELIA_Challenge_eval_all12/results/...`

Batch script used:

- `/home/jeff/Projects/odelia_breast_mri/scripts/run_all_old_duke_models_on_odelia_challenge.py`

Dataset download script used:

- `/home/jeff/Projects/odelia_breast_mri/scripts/download_odelia_challenge.py`

## Limitations

1. `RSH` was evaluated on all available rows, not only `Split == test`.
2. Only the `Sub_1` sequence was downloaded and evaluated.
3. These are DUKE-trained models transferred directly to ODELIA without domain adaptation.
4. The original DUKE node metadata CSVs for the old split were overwritten later by the non-IID split generation, although the checkpoints and internal result files are still preserved.

## Recommended Next Steps

1. Re-run `RSH` in strict `Split == test` mode for challenge-style comparability.
2. Evaluate the newer age-stratified non-IID models on ODELIA and compare them against the mixed-split models.
3. Build a simple ensemble of the three `node_all` models and measure whether external AUROC improves over the single-model mean.
4. If swarm training is the next objective, compare:
   - best local node
   - `node_all`
   - future swarm/federated aggregated model

## Bottom Line

The DUKE-trained pooled model generalizes reasonably well to the ODELIA Challenge dataset:

- `0.875 ± 0.004` AUROC on the standard external ODELIA centers
- `0.781 ± 0.017` AUROC on `RSH`

Among the local models, `node_A` transfers best, `node_B` is close behind, and `node_C` remains unstable.

The overall conclusion is that **pooling data during DUKE training produces the most reliable external model**, and the ODELIA Challenge evaluation preserves the same broad ranking observed in the internal DUKE experiments.
