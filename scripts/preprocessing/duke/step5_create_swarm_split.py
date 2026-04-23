"""
Create swarm learning data partitions for the DUKE breast MRI dataset.

Non-IID strategy: nodes are stratified by patient age to simulate realistic
institutional heterogeneity. Younger patients have denser breast tissue and
different MRI appearance from older patients, so models trained on different
age cohorts genuinely diverge in decision boundaries.

  node_A: 40% of patients — oldest cohort  (oldest 50% of pool)
  node_B: 30% of patients — middle-age cohort
  node_C: 10% of patients — youngest cohort (youngest 12.5% of pool)
  test:   20% of patients — held-out, randomly stratified (all ages)

Real-world analogy:
  node_A ~ general/post-menopausal screening centre (fatty breast tissue)
  node_B ~ mixed general hospital
  node_C ~ hereditary-risk / young-patient programme (dense breast tissue)

Test set is stratified by n_malignant (random) to remain representative of
the full population. Within each training node: 80/20 train/val split,
stratified by n_malignant.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ── Configuration ─────────────────────────────────────────────────────────────
SRC_ANNOTATION = Path("/mnt/nvme2n1p1/jeff/DUKE/metadata/annotation.csv")
SRC_DATA       = Path("/mnt/nvme2n1p1/jeff/DUKE/data_unilateral")
OUT_ROOT       = Path("/mnt/nvme2n1p1/jeff/DUKE_swarm")
RANDOM_STATE   = 42

# ── Load and build patient-level stats ────────────────────────────────────────
df_anno = pd.read_csv(SRC_ANNOTATION, dtype={"UID": str, "PatientID": str})

patient_stats = (
    df_anno.groupby("PatientID")
    .agg(
        n_malignant=("Lesion", lambda x: int((x == 2).sum())),
        age_days=("Age", "median"),
    )
    .reset_index()
)

patients    = patient_stats["PatientID"].to_numpy()
strat_nmal  = patient_stats["n_malignant"].to_numpy()
pid_to_nmal = dict(zip(patient_stats["PatientID"], patient_stats["n_malignant"]))
pid_to_age  = dict(zip(patient_stats["PatientID"], patient_stats["age_days"]))

# ── Step A: peel test set (20%, stratified by n_malignant — keep balanced) ────
p_pool, p_test = train_test_split(
    patients, test_size=0.20, stratify=strat_nmal, random_state=RANDOM_STATE
)

# ── Step B: assign pool nodes by age (non-IID) ────────────────────────────────
# Sort pool patients youngest → oldest, then carve out three age bands:
#   bottom 12.5% of pool (=10% of total) → node_C (youngest)
#   top    50.0% of pool (=40% of total) → node_A (oldest)
#   middle 37.5% of pool (=30% of total) → node_B
pool_ages      = np.array([pid_to_age[p] for p in p_pool])
age_order      = np.argsort(pool_ages)           # ascending: youngest first
p_pool_sorted  = p_pool[age_order]
n_pool         = len(p_pool_sorted)

n_C = round(n_pool * 0.125)          # youngest 12.5 %
n_A = round(n_pool * 0.500)          # oldest   50.0 %
n_B = n_pool - n_A - n_C             # middle   37.5 %

p_C = p_pool_sorted[:n_C]            # youngest
p_B = p_pool_sorted[n_C : n_C + n_B] # middle
p_A = p_pool_sorted[n_C + n_B :]     # oldest


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_trainval_split(node_patients):
    """80/20 train/val split, stratified by n_malignant where possible."""
    strat = np.array([pid_to_nmal[p] for p in node_patients])
    counts = pd.Series(strat).value_counts()
    can_stratify = (counts >= 2).all() and len(counts) > 1
    p_tr, p_val = train_test_split(
        node_patients, test_size=0.20,
        stratify=strat if can_stratify else None,
        random_state=RANDOM_STATE
    )
    return p_tr, p_val


def write_node_metadata(node_name, train_pids, val_pids):
    all_pids     = np.concatenate([train_pids, val_pids])
    df_node_anno = df_anno[df_anno["PatientID"].isin(all_pids)].copy()
    train_uids   = set(df_anno[df_anno["PatientID"].isin(train_pids)]["UID"])
    val_uids     = set(df_anno[df_anno["PatientID"].isin(val_pids)]["UID"])

    df_split = pd.DataFrame([
        {"UID": uid, "Fold": 0, "Split": "train" if uid in train_uids else "val"}
        for uid in df_node_anno["UID"].values
    ])

    meta_dir = OUT_ROOT / node_name / "metadata_unilateral"
    meta_dir.mkdir(parents=True, exist_ok=True)
    df_node_anno.to_csv(meta_dir / "annotation.csv", index=False)
    df_split.to_csv(meta_dir / "split.csv", index=False)

    link = OUT_ROOT / node_name / "data_unilateral"
    if link.is_symlink():
        link.unlink()
    if not link.exists():
        link.symlink_to(SRC_DATA, target_is_directory=True)

    train_anno = df_anno[df_anno["PatientID"].isin(train_pids)]
    mal_rate   = (train_anno["Lesion"] == 2).mean()
    ages       = [pid_to_age[p] for p in all_pids]
    print(
        f"{node_name:10s}: {len(all_pids):3d} patients | "
        f"{len(train_uids):3d} train UIDs | "
        f"mal rate {mal_rate:.0%} | "
        f"age {np.min(ages)/365:.0f}–{np.max(ages)/365:.0f} yrs "
        f"(median {np.median(ages)/365:.0f})"
    )


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
    if link.is_symlink():
        link.unlink()
    if not link.exists():
        link.symlink_to(SRC_DATA, target_is_directory=True)

    ages     = [pid_to_age[p] for p in test_pids]
    mal_rate = (df_test_anno["Lesion"] == 2).mean()
    print(
        f"{'test':10s}: {len(test_pids):3d} patients | "
        f"{len(df_test_anno):3d} test UIDs  | "
        f"mal rate {mal_rate:.0%} | "
        f"age {np.min(ages)/365:.0f}–{np.max(ages)/365:.0f} yrs "
        f"(median {np.median(ages)/365:.0f})"
    )


if __name__ == "__main__":
    node_splits = {
        "node_A": make_trainval_split(p_A),
        "node_B": make_trainval_split(p_B),
        "node_C": make_trainval_split(p_C),
    }
    for node_name, (p_tr, p_val) in node_splits.items():
        write_node_metadata(node_name, p_tr, p_val)

    write_test_metadata(p_test)

    # node_all: union of all three training nodes (covers full age range)
    p_all_tr  = np.concatenate([node_splits[n][0] for n in ["node_A", "node_B", "node_C"]])
    p_all_val = np.concatenate([node_splits[n][1] for n in ["node_A", "node_B", "node_C"]])
    write_node_metadata("node_all", p_all_tr, p_all_val)

    # Verify zero patient overlap between partitions
    all_sets = [set(p_A), set(p_B), set(p_C), set(p_test)]
    for i, s1 in enumerate(all_sets):
        for j, s2 in enumerate(all_sets):
            if i < j:
                assert len(s1 & s2) == 0, f"Patient overlap between partition {i} and {j}"

    print("\nSummary (non-IID by patient age):")
    print(f"  node_A (oldest  ): {len(p_A):3d} patients ({len(p_A)/651*100:.1f}%)")
    print(f"  node_B (middle  ): {len(p_B):3d} patients ({len(p_B)/651*100:.1f}%)")
    print(f"  node_C (youngest): {len(p_C):3d} patients ({len(p_C)/651*100:.1f}%)")
    print(f"  test             : {len(p_test):3d} patients ({len(p_test)/651*100:.1f}%)")
    print(f"  total            : {len(p_A)+len(p_B)+len(p_C)+len(p_test)} patients")
    print(f"\nNo patient overlap verified. Output: {OUT_ROOT}")
