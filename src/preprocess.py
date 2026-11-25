"""Produce transformed copies of a labeled-data folder.

Contract (so build_experiment works on the output unchanged):
  in : a labeled-data folder (sessions with frames + CollectedData_dani.h5/.csv)
  out: a new labeled-data folder inside the same data/, e.g. data/labeled-data_flipped

Rule: frames and labels stay consistent.
  - Pixel-only transforms (CLAHE, brightness): coordinates unchanged.
  - Geometric transforms (flip, rotate, crop): apply the SAME mapping to the
    keypoint coordinates, so labels still match the frames.

Transforms chain: run one output folder into the next to combine them, e.g.
  canonicalize_orientation("data/labeled-data",         "data/labeled-data_flipped")
  clahe(                   "data/labeled-data_flipped",  "data/labeled-data_flipped_clahe")
"""

import shutil
from pathlib import Path

import cv2
import pandas as pd


def _labeled_folders(src):
    return [p for p in sorted(Path(src).iterdir())
            if p.is_dir() and list(p.glob("CollectedData_*.h5"))]


def _orientation(df):
    ax = df.loc[:, (slice(None), "eye-anterior", "x")].mean().mean()
    px = df.loc[:, (slice(None), "eye-posterior", "x")].mean().mean()
    return "left" if ax < px else "right"


def canonicalize_orientation(src, dst, target="left"):
    """Flip sessions not already facing `target` so all share one orientation.

    Horizontal mirror of the frames + x -> W-1-x on the labels. Labels keep
    their names (anterior stays anterior; the point moves with the pixels).
    Sessions already at `target` are copied unchanged.
    """
    dst = Path(dst)
    for folder in _labeled_folders(src):
        out = dst / folder.name
        out.mkdir(parents=True, exist_ok=True)
        df = pd.read_hdf(next(folder.glob("CollectedData_*.h5")))

        if _orientation(df) == target:
            for f in folder.iterdir():
                shutil.copy2(f, out / f.name)
            continue

        w = cv2.imread(str(next(folder.glob("img*.png")))).shape[1]
        for img in folder.glob("img*.png"):
            cv2.imwrite(str(out / img.name), cv2.flip(cv2.imread(str(img)), 1))
        x_cols = [c for c in df.columns if c[-1] == "x"]
        df[x_cols] = w - 1 - df[x_cols]
        df.to_hdf(out / "CollectedData_dani.h5", key="df_with_missing", mode="w")
        df.to_csv(out / "CollectedData_dani.csv")
        print(f"flipped {folder.name}")
    print("done")


def clahe(src, dst, clip_limit=2.0, tile=8):
    """Contrast Limited Adaptive Histogram Equalization on every frame.

    Pixel-only: equalizes local contrast to reduce lighting/glare differences
    between sessions. Coordinates are unchanged, so labels are copied as-is.
    """
    dst = Path(dst)
    op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    for folder in _labeled_folders(src):
        out = dst / folder.name
        out.mkdir(parents=True, exist_ok=True)
        for f in folder.iterdir():
            if f.suffix == ".png":
                gray = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2GRAY)
                eq = op.apply(gray)
                cv2.imwrite(str(out / f.name), cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR))
            else:
                shutil.copy2(f, out / f.name)   # labels unchanged
    print("done")