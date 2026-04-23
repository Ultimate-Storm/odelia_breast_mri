# Validation Summary: DUKE Internal and ODELIA External Results

Date: 2026-04-19

## Overview

This report consolidates the main validation results produced so far for the DUKE breast MRI experiments and their external transfer to the ODELIA Challenge dataset.

It combines:

- **IID training results**
  - internal DUKE validation/testing
  - external ODELIA evaluation
- **Non-IID training results**
  - internal DUKE validation/testing
  - external ODELIA evaluation

The goal is to keep the full picture in one place.

## Evaluation Targets

### Internal DUKE

- held-out DUKE test split
- sample count: `262`

### External ODELIA Standard Centers

- institutions: `CAM,MHA,RUMC,UKA,UMCU`
- split: `Split == test`
- sample count: `260`

### External RSH OOD

- institution: `RSH`
- current evaluation mode: **all available rows**
- sample count: `200`

Important note:

- `RSH` was evaluated using all available rows because the current `main_predict.py` treats `RSH` as a special OOD institution and loads `split=None`
- these `RSH` numbers are therefore useful for OOD comparison, but they are not the same as strict `RSH test` only

## Model Setup

All reported models use:

- architecture: `MST`
- task: `binary`
- config: `unilateral`
- backbone: `DINOv2 with registers (small)`
- inference image: `Sub_1.nii.gz`

## Master Summary

This is the highest-level combined view.

### IID Summary

These are the 3-run means from the original DUKE mixed split.

| Regime | Node | DUKE Internal AUROC | ODELIA Standard AUROC | RSH AUROC |
| --- | --- | ---: | ---: | ---: |
| IID | `node_A` | 0.898 ± 0.016 | 0.859 ± 0.014 | 0.706 ± 0.024 |
| IID | `node_B` | 0.899 ± 0.001 | 0.841 ± 0.006 | 0.702 ± 0.011 |
| IID | `node_C` | 0.660 ± 0.081 | 0.672 ± 0.105 | 0.561 ± 0.074 |
| IID | `node_all` | 0.906 ± 0.004 | 0.875 ± 0.004 | 0.781 ± 0.017 |

### Non-IID Summary

These are the **latest valid non-IID checkpoints per node** as deployed externally.

| Regime | Node | DUKE Internal AUROC | ODELIA Standard AUROC | RSH AUROC |
| --- | --- | ---: | ---: | ---: |
| Non-IID | `node_A` | 0.855 | 0.853 | 0.726 |
| Non-IID | `node_B` | 0.901 | 0.798 | 0.673 |
| Non-IID | `node_C` | 0.839 | 0.785 | 0.581 |
| Non-IID | `node_all` | 0.903 | 0.866 | 0.735 |

## IID Results

### IID Internal DUKE

Original 3-run internal DUKE summary:

| Node | Run 1 | Run 2 | Run 3 | Mean ± Std |
| --- | ---: | ---: | ---: | ---: |
| `node_A` | 0.889 | 0.886 | 0.920 | 0.898 ± 0.016 |
| `node_B` | 0.897 | 0.899 | 0.900 | 0.899 ± 0.001 |
| `node_C` | 0.551 | 0.749 | 0.680 | 0.660 ± 0.081 |
| `node_all` | 0.907 | 0.900 | 0.910 | 0.906 ± 0.004 |

Mean internal DUKE accuracy across the same 3 IID runs:

| Node | Mean Accuracy |
| --- | ---: |
| `node_A` | 0.814 |
| `node_B` | 0.810 |
| `node_C` | 0.598 |
| `node_all` | 0.849 |

### IID External ODELIA Standard

Evaluation target: `CAM,MHA,RUMC,UKA,UMCU`, `n=260`

| Node | Run 1 | Run 2 | Run 3 | Mean ± Std |
| --- | ---: | ---: | ---: | ---: |
| `node_A` | 0.848 | 0.854 | 0.875 | 0.859 ± 0.014 |
| `node_B` | 0.848 | 0.839 | 0.837 | 0.841 ± 0.006 |
| `node_C` | 0.577 | 0.784 | 0.655 | 0.672 ± 0.105 |
| `node_all` | 0.873 | 0.872 | 0.879 | 0.875 ± 0.004 |

Mean ODELIA standard accuracy across the same 3 IID runs:

| Node | Mean Accuracy |
| --- | ---: |
| `node_A` | 0.738 |
| `node_B` | 0.718 |
| `node_C` | 0.412 |
| `node_all` | 0.844 |

### IID External RSH OOD

Evaluation target: `RSH`, `n=200`

| Node | Run 1 | Run 2 | Run 3 | Mean ± Std |
| --- | ---: | ---: | ---: | ---: |
| `node_A` | 0.679 | 0.722 | 0.717 | 0.706 ± 0.024 |
| `node_B` | 0.690 | 0.710 | 0.706 | 0.702 ± 0.011 |
| `node_C` | 0.495 | 0.641 | 0.547 | 0.561 ± 0.074 |
| `node_all` | 0.779 | 0.798 | 0.765 | 0.781 ± 0.017 |

Mean RSH accuracy across the same 3 IID runs:

| Node | Mean Accuracy |
| --- | ---: |
| `node_A` | 0.745 |
| `node_B` | 0.738 |
| `node_C` | 0.568 |
| `node_all` | 0.775 |

### IID Checkpoint List

#### `node_A`

- `MST_binary_unilateral_2026_04_14_215628_fold0`
- `MST_binary_unilateral_2026_04_16_161821_fold0`
- `MST_binary_unilateral_2026_04_17_161237_fold0`

#### `node_B`

- `MST_binary_unilateral_2026_04_14_221009_fold0`
- `MST_binary_unilateral_2026_04_17_151546_fold0`
- `MST_binary_unilateral_2026_04_17_163350_fold0`

#### `node_C`

- `MST_binary_unilateral_2026_04_14_223444_fold0`
- `MST_binary_unilateral_2026_04_17_154154_fold0`
- `MST_binary_unilateral_2026_04_17_165451_fold0`

#### `node_all`

- `MST_binary_unilateral_2026_04_15_094319_fold0`
- `MST_binary_unilateral_2026_04_17_155107_fold0`
- `MST_binary_unilateral_2026_04_17_170100_fold0`

## Non-IID Results

### Non-IID Internal DUKE

These values correspond to the latest valid non-IID checkpoints deployed externally.

| Node | DUKE Internal AUROC | DUKE Internal Accuracy |
| --- | ---: | ---: |
| `node_A` | 0.855 | 0.790 |
| `node_B` | 0.901 | 0.809 |
| `node_C` | 0.839 | 0.756 |
| `node_all` | 0.903 | 0.828 |

### Non-IID External ODELIA Standard

Evaluation target: `CAM,MHA,RUMC,UKA,UMCU`, `n=260`

| Node | AUROC | Accuracy |
| --- | ---: | ---: |
| `node_A` | 0.853 | 0.758 |
| `node_B` | 0.798 | 0.619 |
| `node_C` | 0.785 | 0.765 |
| `node_all` | 0.866 | 0.831 |

### Non-IID External RSH OOD

Evaluation target: `RSH`, `n=200`

| Node | AUROC | Accuracy |
| --- | ---: | ---: |
| `node_A` | 0.726 | 0.745 |
| `node_B` | 0.673 | 0.645 |
| `node_C` | 0.581 | 0.625 |
| `node_all` | 0.735 | 0.790 |

### Non-IID Checkpoint List

These are the latest valid non-IID checkpoints used for deployment:

| Node | Run Directory | Best Checkpoint |
| --- | --- | --- |
| `node_A` | `MST_binary_unilateral_2026_04_18_083216_fold0` | `epoch=12-step=676.ckpt` |
| `node_B` | `MST_binary_unilateral_2026_04_18_223813_fold0` | `epoch=31-step=1248.ckpt` |
| `node_C` | `MST_binary_unilateral_2026_04_18_225101_fold0` | `epoch=55-step=728.ckpt` |
| `node_all` | `MST_binary_unilateral_2026_04_18_225947_fold0` | `epoch=14-step=1560.ckpt` |

## Interpretation

### 1. `node_all` is the best model in both regimes

Across IID and non-IID training, the pooled model stays strongest:

- IID ODELIA standard: `0.875 ± 0.004`
- IID RSH: `0.781 ± 0.017`
- Non-IID ODELIA standard: `0.866`
- Non-IID RSH: `0.735`

This is the most consistent pattern in the project.

### 2. IID beats the latest non-IID snapshot externally

Comparing the current summaries:

| Node | IID ODELIA Standard | Non-IID ODELIA Standard |
| --- | ---: | ---: |
| `node_A` | 0.859 ± 0.014 | 0.853 |
| `node_B` | 0.841 ± 0.006 | 0.798 |
| `node_C` | 0.672 ± 0.105 | 0.785 |
| `node_all` | 0.875 ± 0.004 | 0.866 |

For `node_A`, `node_B`, and `node_all`, the IID models are currently stronger on standard ODELIA.

### 3. `node_C` is the exception

`node_C` improves materially under the latest non-IID snapshot:

- IID ODELIA standard mean: `0.672`
- Non-IID ODELIA standard: `0.785`

But `node_C` is still weak on `RSH`:

- IID RSH mean: `0.561`
- Non-IID RSH: `0.581`

So the gain is mainly on the standard centers, not on the hardest OOD site.

### 4. `RSH` remains the hardest target

Every training regime and node drops on `RSH` relative to the standard ODELIA institutions.

That remains true for both IID and non-IID training and confirms `RSH` as the strongest domain-shift test.

## Recommended Deployment Choice

If the goal is immediate deployment on ODELIA, the best options from the current evidence are:

1. **Best overall**: IID `node_all`
   - standard ODELIA: `0.875 ± 0.004`
   - RSH: `0.781 ± 0.017`

2. **Best non-IID option**: Non-IID `node_all`
   - standard ODELIA: `0.866`
   - RSH: `0.735`

3. **Best single local-node model**
   - IID setting: `node_A`
   - Non-IID setting: also `node_A`

## Artifact Locations

### IID external evaluation

- root: `/mnt/nvme2n1p1/jeff/ODELIA_Challenge_eval_all12`
- log: `/mnt/nvme2n1p1/jeff/ODELIA_Challenge_eval_all12/eval_all12.log`

### Non-IID external evaluation

- root: `/mnt/nvme2n1p1/jeff/ODELIA_Challenge_eval_noniid_latest`
- log: `/mnt/nvme2n1p1/jeff/ODELIA_Challenge_eval_noniid_latest/eval_noniid.log`

### Reports

- IID external report: [ODELIA_CHALLENGE_EVALUATION_REPORT.md](/home/jeff/Projects/odelia_breast_mri/ODELIA_CHALLENGE_EVALUATION_REPORT.md:1)
- Non-IID external report: [ODELIA_CHALLENGE_NONIID_EVALUATION_REPORT.md](/home/jeff/Projects/odelia_breast_mri/ODELIA_CHALLENGE_NONIID_EVALUATION_REPORT.md:1)
- Consolidated report: [README_VALIDATION_SUMMARY.md](/home/jeff/Projects/odelia_breast_mri/README_VALIDATION_SUMMARY.md:1)

## Caveats

1. `RSH` was evaluated on all available rows, not only `Split == test`.
2. Only `Sub_1` was used for inference.
3. The IID section reports 3-run means, while the non-IID section reports the latest valid checkpoint per node.
4. The latest non-IID checkpoints were generated at slightly different times, so they form a practical deployment snapshot rather than a synchronized four-model batch.

## Bottom Line

The single best-performing model family remains the pooled `node_all` model.

From the current evidence:

- **best external result overall**: IID `node_all`
- **best non-IID deployment model**: Non-IID `node_all`
- **best local-node transfer model**: `node_A`

If you want one model to carry forward right now, choose **`node_all`**. If you want a local-node reference model alongside it, choose **`node_A`**.
