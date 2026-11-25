"""Visualization helpers for the pupil/eye DeepLabCut project.

Works on either a DLC project folder (has labeled-data/ and
evaluation-results-pytorch/) or a raw labeled-data folder (sessions directly
inside). Predictions and training stats take a shuffle number, since a project
holds one per scenario (1=hard, 2=easy, 3=mixed).

  show_ground_truth(path, session, image=None)                 -> dots
  show_predictions(project, session, image=None, shuffle=1)    -> filled triangles
  compare(project, session, image=None, shuffle=1)             -> both, side by side
  load_training_stats(project, shuffle=1) / per_keypoint_error(project, session, shuffle=1)
  plot_training_curves(stats, evals, per_keypoint=None)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

KEYPOINTS = ["eye-anterior", "eye-posterior", "eye-dorsal", "eye-ventral",
             "pupil-anterior", "pupil-posterior", "pupil-dorsal", "pupil-ventral"]

_COLORS = {bp: plt.cm.rainbow(i / (len(KEYPOINTS) - 1)) for i, bp in enumerate(KEYPOINTS)}


# ----- internal helpers -------------------------------------------------------

def _imread(path):
    path = os.path.abspath(path)
    if os.name == "nt" and len(path) > 250:
        path = "\\\\?\\" + path
    return np.asarray(Image.open(path))


def _session_dir(path, session):
    p = Path(path)
    for cand in (p / "labeled-data" / session, p / session):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"session '{session}' not found under {path}")


def _load_labels(path, session):
    h5 = next(_session_dir(path, session).glob("CollectedData_*.h5"))
    return pd.read_hdf(h5)


def _eval_h5(project, shuffle):
    cands = [p for p in (Path(project) / "evaluation-results-pytorch").rglob("*.h5")
             if f"shuffle{shuffle}_" in p.name]
    if not cands:
        raise FileNotFoundError(f"no evaluation results for shuffle {shuffle} under {project}")
    return sorted(cands)[0]


def _load_predictions(project, shuffle=1):
    return pd.read_hdf(_eval_h5(project, shuffle))


def _coord_level(df):
    for i in range(df.columns.nlevels):
        if set(df.columns.get_level_values(i)) <= {"x", "y", "likelihood"}:
            return i
    raise ValueError("no coord level found")


def _bp_level(df):
    for i in range(df.columns.nlevels):
        if set(df.columns.get_level_values(i)) & set(KEYPOINTS):
            return i
    raise ValueError("no bodypart level found")


def _point(df, idx, bodypart):
    row = df.loc[idx]
    cl, bl = _coord_level(df), _bp_level(df)
    sel = row[row.index.get_level_values(bl) == bodypart]

    def g(c):
        m = sel.index.get_level_values(cl) == c
        return float(sel[m].iloc[0]) if m.any() else None

    return g("x"), g("y"), g("likelihood")


def _row(df, session, image):
    names = [ix[-1] for ix in df.index if ix[1] == session]
    image = image or names[0]
    idx = next(ix for ix in df.index if ix[1] == session and ix[-1] == image)
    return idx, image


def _scatter(ax, df, idx, marker, size, pcutoff=None):
    for bp in KEYPOINTS:
        x, y, p = _point(df, idx, bp)
        if x is None:
            continue
        faded = pcutoff is not None and p is not None and p < pcutoff
        ax.scatter(x, y, s=size, marker=marker,
                   facecolors="none" if faded else [_COLORS[bp]],
                   edgecolors=[_COLORS[bp]] if faded else "black",
                   linewidths=1.6 if faded else 0.4, alpha=0.9)


def _legend(ax):
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=_COLORS[bp],
                      markersize=8, label=bp) for bp in KEYPOINTS]
    ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.6)


# ----- single-image views -----------------------------------------------------

def show_ground_truth(path, session, image=None, ax=None):
    """Human labels (dots). `path` may be a project or a raw labeled-data folder."""
    df = _load_labels(path, session)
    idx, image = _row(df, session, image)
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(_imread(_session_dir(path, session) / image), cmap="gray")
    _scatter(ax, df, idx, marker="o", size=45)
    ax.set_title(f"Ground truth — {session}/{image}", fontsize=9)
    ax.axis("off")
    if own:
        _legend(ax); plt.tight_layout(); plt.show()
    return ax


def show_predictions(project, session, image=None, shuffle=1, pcutoff=0.6, ax=None):
    """Model predictions (filled triangles) for a given shuffle; faded if below pcutoff."""
    preds = _load_predictions(project, shuffle)
    idx, image = _row(preds, session, image)
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(_imread(_session_dir(project, session) / image), cmap="gray")
    _scatter(ax, preds, idx, marker="^", size=60, pcutoff=pcutoff)
    ax.set_title(f"Predictions (shuffle {shuffle}) — {session}/{image}", fontsize=9)
    ax.axis("off")
    if own:
        _legend(ax); plt.tight_layout(); plt.show()
    return ax


def compare(project, session, image=None, shuffle=1, pcutoff=0.6):
    """Ground truth vs predictions, side by side, for one frame of one shuffle."""
    preds = _load_predictions(project, shuffle)
    _, image = _row(preds, session, image)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
    show_ground_truth(project, session, image, ax=a1)
    show_predictions(project, session, image, shuffle=shuffle, pcutoff=pcutoff, ax=a2)
    _legend(a2)
    plt.tight_layout(); plt.show()


def show_all_variants(session, data_root="./data"):
    """Show the ground-truth frame for `session` in every data/labeled-data* variant."""
    variants = sorted(p for p in Path(data_root).glob("labeled-data*") if p.is_dir())
    for v in variants:
        if (v / session).exists():
            print(v.name)
            show_ground_truth(v, session)


# ----- training curves --------------------------------------------------------

def load_training_stats(project, shuffle=1):
    """Return (stats, evals) for a shuffle: full per-epoch log and the evaluated rows."""
    csv = None
    for c in (Path(project) / "dlc-models-pytorch").rglob("learning_stats.csv"):
        if c.parent.parent.name.endswith(f"shuffle{shuffle}"):
            csv = c; break
    if csv is None:
        raise FileNotFoundError(f"no learning_stats for shuffle {shuffle} under {project}")
    stats = pd.read_csv(csv)
    return stats, stats.dropna(subset=["metrics/test.rmse"])


def per_keypoint_error(project, session, shuffle=1):
    """Mean pixel error per bodypart, predictions vs ground truth, on one session."""
    preds, truth = _load_predictions(project, shuffle), _load_labels(project, session)
    out = {}
    for bp in KEYPOINTS:
        errs = []
        for idx in [i for i in preds.index if i[1] == session]:
            tx, ty, _ = _point(truth, idx, bp)
            px, py, _ = _point(preds, idx, bp)
            if None not in (tx, ty, px, py):
                errs.append(((tx - px) ** 2 + (ty - py) ** 2) ** 0.5)
        out[bp] = float(np.mean(errs)) if errs else float("nan")
    return out


def plot_training_curves(stats, evals, per_keypoint=None, test_session="110535"):
    """The 4-panel figure. Pass values from load_training_stats / per_keypoint_error."""
    best = evals.loc[evals["metrics/test.rmse"].idxmin()]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(stats["step"], stats["losses/train.total_loss"], label="Train")
    ax.plot(evals["step"], evals["losses/eval.total_loss"], marker="o", label=f"Test ({test_session})")
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss — train vs test (divergence = overfitting)"); ax.legend()

    ax = axes[0, 1]
    ax.plot(evals["step"], evals["metrics/test.rmse"], marker="o", label="RMSE (all)")
    ax.plot(evals["step"], evals["metrics/test.rmse_pcutoff"], marker="s", label="RMSE (confident)")
    ax.axvline(best["step"], color="grey", ls="--", label=f"Best epoch ({int(best['step'])})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Error (px)")
    ax.set_title(f"Test position error on {test_session}"); ax.legend()

    ax = axes[1, 0]
    ax.plot(evals["step"], evals["metrics/test.mAP"], marker="o", label="mAP (precision)")
    ax.plot(evals["step"], evals["metrics/test.mAR"], marker="s", label="mAR (recall)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score (0–100)")
    ax.set_title(f"Detection quality on {test_session}"); ax.legend()

    ax = axes[1, 1]
    if per_keypoint:
        ax.barh(list(per_keypoint), list(per_keypoint.values())); ax.invert_yaxis()
        ax.set_xlabel("Test error (px)"); ax.set_title("Per-keypoint test error")
    else:
        ax.axis("off"); ax.set_title("Per-keypoint error (pass per_keypoint=...)")

    plt.tight_layout(); plt.show()
