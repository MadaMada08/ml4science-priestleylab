"""Visualization helpers for the pupil/eye DeepLabCut project.

Ground truth and predictions are read from a DLC project folder:
  <project>/labeled-data/<session>/CollectedData_*.h5   (human labels + frames)
  <project>/evaluation-results-pytorch/**/**.h5          (model predictions)

Public functions:
  show_ground_truth(project, session, image=None)   -> dots
  show_predictions(project, session, image=None)    -> filled triangles
  compare(project, session, image=None)             -> both, side by side
  load_training_stats(project) / per_keypoint_error(project, session)
  plot_training_curves(stats, evals, per_keypoint=None)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

KEYPOINTS = [
    "eye-anterior",
    "eye-posterior",
    "eye-dorsal",
    "eye-ventral",
    "pupil-anterior",
    "pupil-posterior",
    "pupil-dorsal",
    "pupil-ventral",
]

_COLORS = {
    bp: plt.cm.rainbow(i / (len(KEYPOINTS) - 1)) for i, bp in enumerate(KEYPOINTS)
}


# ----- internal helpers -------------------------------------------------------


def _imread(path):
    path = os.path.abspath(path)
    if os.name == "nt" and len(path) > 250:  # Windows MAX_PATH escape
        path = "\\\\?\\" + path
    return np.asarray(Image.open(path))


def _load_labels(project, session):
    h5 = next((Path(project) / "labeled-data" / session).glob("CollectedData_*.h5"))
    return pd.read_hdf(h5)


def _load_predictions(project):
    h5 = next((Path(project) / "evaluation-results-pytorch").rglob("*.h5"))
    return pd.read_hdf(h5)


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


def _images_of(df, session):
    return [idx[-1] for idx in df.index if idx[1] == session]


def _scatter(ax, df, idx, marker, size, pcutoff=None):
    for bp in KEYPOINTS:
        x, y, p = _point(df, idx, bp)
        if x is None:
            continue
        faded = pcutoff is not None and p is not None and p < pcutoff
        ax.scatter(
            x,
            y,
            s=size,
            marker=marker,
            color=[_COLORS[bp]],
            alpha=0.25 if faded else 1.0,
            edgecolors="black",
            linewidths=0.4,
        )


def _legend(ax):
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=_COLORS[bp],
            markersize=8,
            label=bp,
        )
        for bp in KEYPOINTS
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.6)


# ----- single-image views -----------------------------------------------------


def show_ground_truth(project, session, image=None, ax=None):
    """Human labels (dots) on one frame; defaults to the first frame of the session."""
    df = _load_labels(project, session)
    image = image or _images_of(df, session)[0]
    idx = ("labeled-data", session, image)
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(_imread(Path(project) / "labeled-data" / session / image), cmap="gray")
    _scatter(ax, df, idx, marker="o", size=45)
    ax.set_title(f"Ground truth — {session}/{image}", fontsize=9)
    ax.axis("off")
    if own:
        _legend(ax)
        plt.tight_layout()
        plt.show()
    return ax


def show_predictions(project, session, image=None, pcutoff=0.6, ax=None):
    """Model predictions (filled triangles) on one frame; faded if below pcutoff."""
    preds = _load_predictions(project)
    image = image or _images_of(preds, session)[0]
    idx = ("labeled-data", session, image)
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(_imread(Path(project) / "labeled-data" / session / image), cmap="gray")
    _scatter(ax, preds, idx, marker="^", size=60, pcutoff=pcutoff)
    ax.set_title(f"Predictions — {session}/{image}", fontsize=9)
    ax.axis("off")
    if own:
        _legend(ax)
        plt.tight_layout()
        plt.show()
    return ax


def compare(project, session, image=None, pcutoff=0.6):
    """Ground truth vs predictions, side by side, for one frame."""
    preds = _load_predictions(project)
    image = image or _images_of(preds, session)[0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
    show_ground_truth(project, session, image, ax=a1)
    show_predictions(project, session, image, pcutoff=pcutoff, ax=a2)
    _legend(a2)
    plt.tight_layout()
    plt.show()


# ----- training curves --------------------------------------------------------


def load_training_stats(project):
    """Return (stats, evals): full per-epoch log and the rows where test was measured."""
    csv = next((Path(project) / "dlc-models-pytorch").rglob("learning_stats.csv"))
    stats = pd.read_csv(csv)
    evals = stats.dropna(subset=["metrics/test.rmse"])
    return stats, evals


def per_keypoint_error(project, session):
    """Mean pixel error per bodypart, predictions vs ground truth, on one session."""
    preds, truth = _load_predictions(project), _load_labels(project, session)
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
    """The 4-panel baseline figure. Pass values from load_training_stats / per_keypoint_error."""
    best = evals.loc[evals["metrics/test.rmse"].idxmin()]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(stats["step"], stats["losses/train.total_loss"], label="Train (7 sessions)")
    ax.plot(
        evals["step"],
        evals["losses/eval.total_loss"],
        marker="o",
        label=f"Test ({test_session})",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss — train vs test (divergence = overfitting)")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(evals["step"], evals["metrics/test.rmse"], marker="o", label="RMSE (all)")
    ax.plot(
        evals["step"],
        evals["metrics/test.rmse_pcutoff"],
        marker="s",
        label="RMSE (confident)",
    )
    ax.axvline(
        best["step"], color="grey", ls="--", label=f"Best epoch ({int(best['step'])})"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error (px)")
    ax.set_title(f"Test position error on {test_session}")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(
        evals["step"], evals["metrics/test.mAP"], marker="o", label="mAP (precision)"
    )
    ax.plot(evals["step"], evals["metrics/test.mAR"], marker="s", label="mAR (recall)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score (0–100)")
    ax.set_title(f"Detection quality on {test_session}")
    ax.legend()

    ax = axes[1, 1]
    if per_keypoint:
        ax.barh(list(per_keypoint), list(per_keypoint.values()))
        ax.invert_yaxis()
        ax.set_xlabel("Test error (px)")
        ax.set_title("Per-keypoint test error")
    else:
        ax.axis("off")
        ax.set_title("Per-keypoint error (pass per_keypoint=...)")

    plt.tight_layout()
    plt.show()
