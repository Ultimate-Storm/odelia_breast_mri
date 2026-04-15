"""
Create swarm learning data partitions for the DUKE breast MRI dataset.

Splits 651 patients into 4 partitions:
  node_A: 40%  (~260 patients)
  node_B: 30%  (~195 patients)
  node_C: 10%  (~65 patients)
  test:   20%  (~131 patients — held-out evaluation set)

For node_A/B/C: further 80/20 train/val split (patient-level, stratified).
For test: all UIDs get Split='test', Fold=0.

Outputs:
  /mnt/nvme2n1p1/jeff/DUKE_swarm/{node}/metadata_unilateral/{annotation,split}.csv
  /mnt/nvme2n1p1/jeff/DUKE_swarm/{node}/data_unilateral  ->  symlink to shared data
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ── Configuration ──────────────────────────────────────────────────────────
SRC_ANNOTATION = Path("/mnt/nvme2n1p1/jeff/DUKE/metadata/annotation.csv")
SRC_DATA       = Path("/mnt/nvme2n1p1/jeff/DUKE/data_unilateral")
OUT_ROOT       = Path("/mnt/nvme2n1p1/jeff/DUKE_swarm")
RANDOM_STATE   = 42

# ── Load source annotation ─────────────────────────────────────────────────
df_anno = pd.read_csv(SRC_ANNOTATION, dtype={"UID": str, "PatientID": str})

# ── Build patient-level stratification variable ────────────────────────────
# n_malignant: number of malignant breasts per patient (1 = unilateral, 2 = bilateral)
# Stratifying on this preserves the bilateral-cancer ratio across all splits.
patient_stats = (
    df_anno.groupby("PatientID")["Lesion"]
    .apply(lambda x: int((x == 2).sum()))
    .reset_index()
    .rename(columns={"Lesion": "n_malignant"})
)
patients = patient_stats["PatientID"].to_numpy()
strat    = patient_stats["n_malignant"].to_numpy()
pid_to_nmal = dict(zip(patient_stats["PatientID"], patient_stats["n_malignant"]))

# ── Step A: peel test set (20% of total) ──────────────────────────────────
p_pool, p_test = train_test_split(
    patients, test_size=0.20, stratify=strat, random_state=RANDOM_STATE
)

# ── Step B: peel node_A (50% of pool = 40% of total) ──────────────────────
strat_pool = np.array([pid_to_nmal[p] for p in p_pool])
p_rest, p_A = train_test_split(
    p_pool, test_size=0.50, stratify=strat_pool, random_state=RANDOM_STATE
)

# ── Step C: split remainder into node_B (75%) and node_C (25%) ────────────
# node_B = 75% of rest = 30% of total; node_C = 25% of rest = 10% of total
strat_rest = np.array([pid_to_nmal[p] for p in p_rest])
p_B, p_C = train_test_split(
    p_rest, test_size=0.25, stratify=strat_rest, random_state=RANDOM_STATE
)


def make_trainval_split(node_patients):
    """Return (train_patients, val_patients) with stratified 80/20 split."""
    strat_node = np.array([pid_to_nmal[p] for p in node_patients])
    p_tr, p_val = train_test_split(
        node_patients, test_size=0.20,
        stratify=strat_node, random_state=RANDOM_STATE
    )
    return p_tr, p_val


def write_node_metadata(node_name, train_pids, val_pids):
    all_pids = np.concatenate([train_pids, val_pids])
    df_node_anno = df_anno[df_anno["PatientID"].isin(all_pids)].copy()

    train_uids = set(df_anno[df_anno["PatientID"].isin(train_pids)]["UID"])
    val_uids   = set(df_anno[df_anno["PatientID"].isin(val_pids)]["UID"])

    df_split = pd.DataFrame([
        {"UID": uid, "Fold": 0, "Split": "train" if uid in train_uids else "val"}
        for uid in df_node_anno["UID"].values
    ])

    meta_dir = OUT_ROOT / node_name / "metadata_unilateral"
    meta_dir.mkdir(parents=True, exist_ok=True)
    df_node_anno.to_csv(meta_dir / "annotation.csv", index=False)
    df_split.to_csv(meta_dir / "split.csv", index=False)

    link = OUT_ROOT / node_name / "data_unilateral"
    if not link.exists():
        link.symlink_to(SRC_DATA, target_is_directory=True)

    n_train_uids = len(train_uids)
    n_val_uids   = len(val_uids)
    print(f"{node_name}: {len(train_pids)} train patients ({n_train_uids} UIDs), "
          f"{len(val_pids)} val patients ({n_val_uids} UIDs)")


def write_test_metadata(test_pids):
    df_test_anno = df_anno[df_anno["PatientID"].isin(test_pids)].copy()
    df_split = pd.DataFrame([
        {"UID": uid, "Fold": 0, "Split": "test"}
        for uid in df_test_anno["UID"].values
    ])

    meta_dir = OUT_ROOT / "test" / "metadata_unilateral"
    meta_dir.mkdir(parents=True, exist_ok=True)
    df_test_anno.to_csv(meta_dir / "annotation.csv", index=False)
    df_split.to_csv(meta_dir / "split.csv", index=False)

    link = OUT_ROOT / "test" / "data_unilateral"
    if not link.exists():
        link.symlink_to(SRC_DATA, target_is_directory=True)

    print(f"test:   {len(test_pids)} patients ({len(df_test_anno)} UIDs)")


if __name__ == "__main__":
    node_splits = {
        "node_A": make_trainval_split(p_A),
        "node_B": make_trainval_split(p_B),
        "node_C": make_trainval_split(p_C),
    }

    for node_name, (p_tr, p_val) in node_splits.items():
        write_node_metadata(node_name, p_tr, p_val)

    write_test_metadata(p_test)

    # Verify no patient overlap between partitions
    all_sets = [set(p_A), set(p_B), set(p_C), set(p_test)]
    for i, s1 in enumerate(all_sets):
        for j, s2 in enumerate(all_sets):
            if i < j:
                overlap = s1 & s2
                assert len(overlap) == 0, f"Overlap between partition {i} and {j}: {overlap}"

    print("\nSummary:")
    print(f"  node_A : {len(p_A):3d} patients ({len(p_A)/651*100:.1f}%)")
    print(f"  node_B : {len(p_B):3d} patients ({len(p_B)/651*100:.1f}%)")
    print(f"  node_C : {len(p_C):3d} patients ({len(p_C)/651*100:.1f}%)")
    print(f"  test   : {len(p_test):3d} patients ({len(p_test)/651*100:.1f}%)")
    print(f"  total  : {len(p_A)+len(p_B)+len(p_C)+len(p_test)} patients")
    print(f"\nNo patient overlap between partitions. Output: {OUT_ROOT}")
