# ODELIA Challenge Evaluation Report for Latest Non-IID DUKE Models

Date: 2026-04-19

## Objective

This report summarizes the deployment of the **latest non-IID DUKE-trained models** onto the downloaded `ODELIA-AI/ODELIA-Challenge-2025` dataset for external evaluation.

The aim was to evaluate whether the most recent non-IID local models generalize to:

- the standard ODELIA challenge institutions
- the OOD institution `RSH`

## Models Evaluated

The latest valid non-IID checkpoint for each node was selected from:

- `/mnt/nvme2n1p1/jeff/DUKE_swarm/runs/`

These are the exact run directories used:

| Node | Run Directory | Best Checkpoint |
| --- | --- | --- |
| `node_A` | `MST_binary_unilateral_2026_04_18_083216_fold0` | `epoch=12-step=676.ckpt` |
| `node_B` | `MST_binary_unilateral_2026_04_18_223813_fold0` | `epoch=31-step=1248.ckpt` |
| `node_C` | `MST_binary_unilateral_2026_04_18_225101_fold0` | `epoch=55-step=728.ckpt` |
| `node_all` | `MST_binary_unilateral_2026_04_18_225947_fold0` | `epoch=14-step=1560.ckpt` |

Important note:

- these four checkpoints are the **latest valid non-IID runs per node**
- their timestamps are not perfectly aligned because some nodes were rerun later on April 18
- this report therefore reflects the **latest deployable non-IID snapshot**, not a single synchronized four-model training wave started at the same minute

## Model Configuration

- Architecture: `MST`
- Task: `binary`
- Config: `unilateral`
- Backbone: `DINOv2 with registers (small)`
- Input modality used at inference: `Sub_1.nii.gz`

## External Dataset

Dataset:

- Hugging Face: `ODELIA-AI/ODELIA-Challenge-2025`
- Configuration: `unilateral`

Local download root:

- `/mnt/nvme2n1p1/jeff/ODELIA_Challenge_2025`

Downloaded institutions:

- `CAM`
- `MHA`
- `RSH`
- `RUMC`
- `UKA`
- `UMCU`

Only `Sub_1` was downloaded for inference because the current DUKE MST checkpoints only consume `Sub_1`.

## Evaluation Protocol

Each of the 4 non-IID models was evaluated on two targets:

### 1. Standard ODELIA Centers

- institutions: `CAM,MHA,RUMC,UKA,UMCU`
- split: `Split == test`
- total samples: `260`

### 2. `RSH` OOD Evaluation

- institution: `RSH`
- split used by current script: **all available rows**
- total samples: `200`

This follows the existing project convention in `main_predict.py`, where `RSH` is treated as a special OOD institution and loaded with `split=None`.

## Internal DUKE Non-IID Baseline

For context, the same four models were already evaluated on the DUKE held-out internal test set (`n=262`).

| Node | DUKE Internal AUROC | DUKE Internal Accuracy |
| --- | ---: | ---: |
| `node_A` | 0.855 | 0.790 |
| `node_B` | 0.901 | 0.809 |
| `node_C` | 0.839 | 0.756 |
| `node_all` | 0.903 | 0.828 |

## External Results

### Standard ODELIA Centers

Evaluation target: `CAM,MHA,RUMC,UKA,UMCU`, `n=260`

| Node | AUROC | Accuracy |
| --- | ---: | ---: |
| `node_A` | 0.853 | 0.758 |
| `node_B` | 0.798 | 0.619 |
| `node_C` | 0.785 | 0.765 |
| `node_all` | 0.866 | 0.831 |

### `RSH` OOD

Evaluation target: `RSH`, all rows, `n=200`

| Node | AUROC | Accuracy |
| --- | ---: | ---: |
| `node_A` | 0.726 | 0.745 |
| `node_B` | 0.673 | 0.645 |
| `node_C` | 0.581 | 0.625 |
| `node_all` | 0.735 | 0.790 |

## Internal vs External Comparison

| Node | DUKE Internal AUROC | ODELIA Standard AUROC | RSH AUROC |
| --- | ---: | ---: | ---: |
| `node_A` | 0.855 | 0.853 | 0.726 |
| `node_B` | 0.901 | 0.798 | 0.673 |
| `node_C` | 0.839 | 0.785 | 0.581 |
| `node_all` | 0.903 | 0.866 | 0.735 |

## Interpretation

### 1. `node_all` remains the strongest external model

The pooled non-IID model is still the best performer on both external targets:

- standard ODELIA: `0.866`
- `RSH`: `0.735`

This matches the broader project trend that pooling training data produces the most robust representation.

### 2. `node_A` transfers very well to the standard ODELIA centers

`node_A` is almost unchanged when moving from DUKE internal to standard ODELIA:

- DUKE internal: `0.855`
- ODELIA standard: `0.853`

That suggests the newest `node_A` non-IID model generalizes well to the main ODELIA institutions.

### 3. `node_B` suffers the largest external drop

`node_B` shows the strongest internal-to-external degradation:

- DUKE internal: `0.901`
- ODELIA standard: `0.798`
- `RSH`: `0.673`

This implies that the latest `node_B` non-IID training produced a model that fits DUKE well but transfers less effectively than `node_A` or `node_all`.

### 4. `node_C` is weaker on OOD despite a respectable standard-center result

`node_C` achieves:

- `0.785` on standard ODELIA centers
- `0.581` on `RSH`

So it is not catastrophic on the standard centers, but it still breaks down substantially under stronger domain shift.

### 5. `RSH` remains clearly harder than the standard ODELIA centers

Every model drops on `RSH`:

- `node_A`: `0.853 -> 0.726`
- `node_B`: `0.798 -> 0.673`
- `node_C`: `0.785 -> 0.581`
- `node_all`: `0.866 -> 0.735`

This confirms that `RSH` behaves as a more challenging out-of-distribution site.

## Practical Ranking

For deployment on ODELIA, the ranking from this non-IID snapshot is:

### Standard ODELIA centers

`node_all` > `node_A` > `node_B` ≈ `node_C`

### `RSH`

`node_all` > `node_A` > `node_B` > `node_C`

## Artifacts

Evaluation output root:

- `/mnt/nvme2n1p1/jeff/ODELIA_Challenge_eval_noniid_latest`

Main log:

- `/mnt/nvme2n1p1/jeff/ODELIA_Challenge_eval_noniid_latest/eval_noniid.log`

Per-model outputs:

- `/mnt/nvme2n1p1/jeff/ODELIA_Challenge_eval_noniid_latest/results/...`

Batch runner used:

- `/home/jeff/Projects/odelia_breast_mri/scripts/run_latest_noniid_models_on_odelia_challenge.py`

## Limitations

1. `RSH` was evaluated on **all available rows**, not only `Split == test`.
2. Only `Sub_1` was used for inference.
3. The four deployed non-IID checkpoints are the latest valid runs per node, but they were generated at slightly different times.
4. This report evaluates one non-IID snapshot per node, not repeated mean ± std non-IID external performance.

## Bottom Line

The latest non-IID DUKE models transfer to ODELIA as follows:

- `node_all`: strongest external model (`0.866` standard, `0.735` RSH)
- `node_A`: best local-node transfer (`0.853` standard, `0.726` RSH)
- `node_B`: best internal DUKE score but noticeably weaker external transfer
- `node_C`: acceptable on standard ODELIA, weak on `RSH`

If the immediate goal is deployment on ODELIA, the best choice from this non-IID set is clearly **`node_all`**, with **`node_A`** as the strongest single local-node model.
