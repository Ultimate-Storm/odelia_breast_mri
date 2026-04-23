#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torchio as tio
from datasets import load_dataset
from tqdm import tqdm


DIR_CONFIG = {
    "default": {
        "data": "data",
        "metadata": "metadata",
    },
    "unilateral": {
        "data": "data_unilateral",
        "metadata": "metadata_unilateral",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ODELIA Challenge data from Hugging Face into the folder layout expected by this repo."
    )
    parser.add_argument("--repo_id", default="ODELIA-AI/ODELIA-Challenge-2025")
    parser.add_argument("--config", default="unilateral", choices=sorted(DIR_CONFIG))
    parser.add_argument(
        "--output_root",
        default="/mnt/nvme2n1p1/jeff/ODELIA_Challenge_2025",
        help="Destination root. Creates institution/data(_unilateral) and metadata(_unilateral) folders under this path.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="Hugging Face token with access to the gated dataset. Defaults to HF_TOKEN/HUGGINGFACE_HUB_TOKEN.",
    )
    parser.add_argument(
        "--institutions",
        nargs="*",
        default=None,
        help="Optional list of institutions to download. Default: all institutions in the dataset.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Optional split filter such as train validation test. Default: all splits exposed by the dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite NIfTI files even if they already exist.",
    )
    parser.add_argument(
        "--image_names",
        nargs="*",
        default=None,
        help="Optional list of image names to save, e.g. Sub_1. Default: save every available image.",
    )
    parser.add_argument(
        "--flush_every",
        type=int,
        default=25,
        help="Write metadata CSVs every N processed items so interrupted downloads remain resumable.",
    )
    return parser.parse_args()


def save_item(item: dict, output_root: Path, config: str, overwrite: bool, image_names: list[str] | None) -> dict:
    uid = item["UID"]
    institution = item["Institution"]
    data_dir = DIR_CONFIG[config]["data"]
    path_folder = output_root / institution / data_dir / uid
    path_folder.mkdir(parents=True, exist_ok=True)

    image_keys = sorted(key for key in item if key.startswith("Image_"))
    row = {}
    for key, value in item.items():
        if key.startswith("Image_") or key.startswith("Affine_"):
            continue
        row[key] = value

    for image_key in image_keys:
        image_name = image_key.split("Image_", 1)[1]
        if image_names is not None and image_name not in image_names:
            continue
        affine_key = f"Affine_{image_name}"
        image_data = item.get(image_key)
        image_affine = item.get(affine_key)
        if image_data is None or image_affine is None:
            continue

        out_file = path_folder / f"{image_name}.nii.gz"
        if out_file.exists() and not overwrite:
            continue

        tensor = np.array(image_data, dtype=np.int16)
        affine = np.array(image_affine, dtype=np.float64)
        tio.ScalarImage(tensor=tensor, affine=affine).save(out_file)

    return row


def write_metadata(rows_by_institution: dict[str, list[dict]], output_root: Path, config: str) -> None:
    metadata_dir_name = DIR_CONFIG[config]["metadata"]
    for institution, rows in sorted(rows_by_institution.items()):
        if not rows:
            continue

        df = pd.DataFrame(rows)
        path_metadata = output_root / institution / metadata_dir_name
        path_metadata.mkdir(parents=True, exist_ok=True)

        split_cols = [col for col in ["UID", "Split", "Fold"] if col in df.columns]
        annotation_drop = [col for col in ["Institution", "Split", "Fold"] if col in df.columns]

        df_anno = df.drop(columns=annotation_drop)
        df_anno.to_csv(path_metadata / "annotation.csv", index=False)

        if split_cols:
            df_split = df[split_cols].copy()
            df_split.to_csv(path_metadata / "split.csv", index=False)


def main() -> None:
    args = parse_args()
    if not args.token:
        raise SystemExit(
            "No Hugging Face token provided. Pass --token or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN."
        )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        args.repo_id,
        name=args.config,
        streaming=True,
        token=args.token,
    )

    rows_by_institution: dict[str, list[dict]] = defaultdict(list)
    split_names = list(dataset.keys()) if args.splits is None else args.splits
    processed = 0

    for split_name in split_names:
        if split_name not in dataset:
            raise ValueError(f"Requested split '{split_name}' not found. Available: {list(dataset.keys())}")

        iterator = dataset[split_name]
        progress = tqdm(iterator, desc=f"Downloading {split_name}", unit="item")
        for item in progress:
            institution = item["Institution"]
            if args.institutions and institution not in args.institutions:
                continue

            row = save_item(
                item,
                output_root=output_root,
                config=args.config,
                overwrite=args.overwrite,
                image_names=args.image_names,
            )
            rows_by_institution[institution].append(row)
            processed += 1
            if args.flush_every > 0 and processed % args.flush_every == 0:
                write_metadata(rows_by_institution, output_root=output_root, config=args.config)

    write_metadata(rows_by_institution, output_root=output_root, config=args.config)

    print(f"Downloaded dataset successfully to {output_root}")
    print("Institutions:", ", ".join(sorted(rows_by_institution)))


if __name__ == "__main__":
    main()
