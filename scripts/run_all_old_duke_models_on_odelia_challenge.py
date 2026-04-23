#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path("/home/jeff/Projects/odelia_breast_mri")
PYTHON = ROOT / "venv/bin/python"
CHALLENGE_ROOT = Path("/mnt/nvme2n1p1/jeff/ODELIA_Challenge_2025")
OUT_ROOT = Path("/mnt/nvme2n1p1/jeff/ODELIA_Challenge_eval_all12")
LOG_FILE = OUT_ROOT / "eval_all12.log"
STANDARD_SITES = "CAM,MHA,RUMC,UKA,UMCU"

# These are the 12 "mixed" DUKE models from the original split strategy:
# 3 repetitions each for node_A, node_B, node_C, and node_all.
MODEL_RUN_DIRS = [
    Path("/mnt/nvme2n1p1/jeff/DUKE_swarm/runs/node_A/MST_binary_unilateral_2026_04_14_215628_fold0"),
    Path("/mnt/nvme2n1p1/jeff/DUKE_swarm/runs/node_A/MST_binary_unilateral_2026_04_16_161821_fold0"),
    Path("/mnt/nvme2n1p1/jeff/DUKE_swarm/runs/node_A/MST_binary_unilateral_2026_04_17_161237_fold0"),
    Path("/mnt/nvme2n1p1/jeff/DUKE_swarm/runs/node_B/MST_binary_unilateral_2026_04_14_221009_fold0"),
    Path("/mnt/nvme2n1p1/jeff/DUKE_swarm/runs/node_B/MST_binary_unilateral_2026_04_17_151546_fold0"),
    Path("/mnt/nvme2n1p1/jeff/DUKE_swarm/runs/node_B/MST_binary_unilateral_2026_04_17_163350_fold0"),
    Path("/mnt/nvme2n1p1/jeff/DUKE_swarm/runs/node_C/MST_binary_unilateral_2026_04_14_223444_fold0"),
    Path("/mnt/nvme2n1p1/jeff/DUKE_swarm/runs/node_C/MST_binary_unilateral_2026_04_17_154154_fold0"),
    Path("/mnt/nvme2n1p1/jeff/DUKE_swarm/runs/node_C/MST_binary_unilateral_2026_04_17_165451_fold0"),
    Path("/mnt/nvme2n1p1/jeff/DUKE_swarm/runs/node_all/MST_binary_unilateral_2026_04_15_094319_fold0"),
    Path("/mnt/nvme2n1p1/jeff/DUKE_swarm/runs/node_all/MST_binary_unilateral_2026_04_17_155107_fold0"),
    Path("/mnt/nvme2n1p1/jeff/DUKE_swarm/runs/node_all/MST_binary_unilateral_2026_04_17_170100_fold0"),
]


def log(message: str, fh) -> None:
    print(message, flush=True)
    fh.write(message + "\n")
    fh.flush()


def best_checkpoint(run_dir: Path) -> Path:
    payload = json.loads((run_dir / "best_checkpoint.json").read_text())
    return run_dir / payload["best_model_epoch"]


def run_eval(path_ckpt: Path, test_institution: str, fh) -> None:
    cmd = [
        str(PYTHON),
        "scripts/main_predict.py",
        "--path_run",
        str(path_ckpt),
        "--test_institution",
        test_institution,
        "--path_root",
        str(CHALLENGE_ROOT),
        "--out_root",
        str(OUT_ROOT),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    print(proc.stdout, end="", flush=True)
    fh.write(proc.stdout)
    fh.flush()


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("w", encoding="utf-8") as fh:
        log("=== ODELIA all-12 evaluation started ===", fh)
        for run_dir in MODEL_RUN_DIRS:
            ckpt = best_checkpoint(run_dir)
            run_name = run_dir.name

            log("", fh)
            log(f">>> {run_name} :: {STANDARD_SITES}", fh)
            run_eval(ckpt, STANDARD_SITES, fh)

            log("", fh)
            log(f">>> {run_name} :: RSH", fh)
            run_eval(ckpt, "RSH", fh)

        log("", fh)
        log("=== ODELIA all-12 evaluation complete ===", fh)


if __name__ == "__main__":
    main()
