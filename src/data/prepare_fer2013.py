# Prepare FER2013 train/val/test CSV splits with fixed defaults (no argparse).
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
SRC_ROOT = os.path.abspath(os.path.join(HERE, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

CANDIDATES = [
    os.path.join(PROJECT_ROOT, "src", "data", "fer2013.csv"),
]

EXISTING = [p for p in CANDIDATES if os.path.exists(p)]
if not EXISTING:
    raise FileNotFoundError(
        "fer2013.csv not found. Put it at 'src/data/fer2013.csv'."
    )

CSV_PATH = max(EXISTING, key=os.path.getsize)  # pick the largest
print(f"[prepare] Using CSV: {CSV_PATH} ({os.path.getsize(CSV_PATH):,} bytes)")

OUT_DIR = os.path.join(PROJECT_ROOT, "datasets")
VAL_RATIO = 0.10
SEED = 1337


def main():
    try:
        df = pd.read_csv(
            CSV_PATH,
            dtype={"emotion": "int64", "pixels": "string", "Usage": "string"},
            low_memory=False,
        )
    except Exception:
        df = pd.read_csv(
            CSV_PATH,
            dtype={"emotion": "int64", "pixels": "string", "Usage": "string"},
            low_memory=False,
            engine="python",
        )

    required = {"emotion", "pixels", "Usage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[prepare] Missing columns in CSV: {missing}")

    total = len(df)
    vc = df["Usage"].value_counts(dropna=False)
    print(f"[prepare] Total rows: {total:,}")
    for k in ["Training", "PublicTest", "PrivateTest"]:
        print(f"[prepare] {k:>11}: {int(vc.get(k, 0)):,}")

    # Sanity check for the number of rows in the CSV
    # the one we use has 35,887 rows
    if total < 35000:
        print("[prepare][WARN] CSV has fewer than 35k rows. "
              "This usually means you're loading a partial or truncated copy.")

    train_df = df[df["Usage"] == "Training"].reset_index(drop=True)
    test_pub = df[df["Usage"] == "PublicTest"].reset_index(drop=True)
    test_pri = df[df["Usage"] == "PrivateTest"].reset_index(drop=True)

    # Create validation split from the training partition
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(train_df))
    rng.shuffle(idx)
    cut = int(len(idx) * (1 - VAL_RATIO))
    tr = train_df.iloc[idx[:cut]].reset_index(drop=True)
    va = train_df.iloc[idx[cut:]].reset_index(drop=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    tr.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    va.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
    test_pub.to_csv(os.path.join(OUT_DIR, "test_public.csv"), index=False)
    test_pri.to_csv(os.path.join(OUT_DIR, "test_private.csv"), index=False)
    print(f"[prepare] Wrote: train.csv, val.csv, test_public.csv, test_private.csv → {OUT_DIR}")


if __name__ == '__main__':
    main()
