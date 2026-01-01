
"""
create_subset.py

Creates a deterministic, stratified subset from the
AI vs Human Generated Dataset (Kaggle).

This script is run ONCE to generate the dataset artifact
used by all experiments.

Reproducibility:
- Stratified by label
- Fixed random seed
- Scripted (no manual selection)
"""

import argparse
import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a stratified subset from the full AI-vs-Human dataset"
    )

    parser.add_argument(
        "--full_data_root",
        type=Path,
        required=True,
        help="Path to full dataset root (contains train.csv and train_data/)",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data"),
        help="Output directory for subset (default: ./data)",
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split (default: 42)",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    full_root = args.full_data_root
    out_root = args.output_dir
    train_ratio = args.train_ratio
    seed = args.seed

    csv_path = full_root / "train.csv"
    img_root = full_root / "train_data"
    out_img_root = out_root / "train_subset_data"

    # Safety checks
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing train.csv at {csv_path}")
    if not img_root.exists():
        raise FileNotFoundError(f"Missing train_data/ at {img_root}")

    out_root.mkdir(exist_ok=True)
    out_img_root.mkdir(exist_ok=True)

    # Load metadata
    df = pd.read_csv(csv_path)

    # Human-readable labels (kept from original pipeline)
    label_map = {
        0: "Human-Created Image",
        1: "AI-Generated Image",
    }
    df["label_name"] = df["label"].map(label_map)

    # Stratified split
    train_df, eval_df = train_test_split(
        df,
        test_size=1 - train_ratio,
        stratify=df["label"],
        random_state=seed,
    )

    subset_df = pd.concat([train_df, eval_df]).reset_index(drop=True)

    # Save CSV artifact
    subset_csv = out_root / "train_subset.csv"
    subset_df.to_csv(subset_csv, index=False)

    # Copy images
    for fname in subset_df["file_name"]:
        fname = fname.strip()
        if fname.startswith("train_data/"):
            fname = fname.replace("train_data/", "", 1)

        src = img_root / fname
        dst = out_img_root / fname
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    # Summary
    print("Subset creation completed successfully.")
    print(f"Train samples: {len(train_df)}")
    print(f"Eval samples : {len(eval_df)}")
    print(f"Total        : {len(subset_df)}")
    print(f"Saved to     : {out_root.resolve()}")

if __name__ == "__main__":
    main()
