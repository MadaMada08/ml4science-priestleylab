"""Score keypoint predictions against the labels, the same way for every experiment."""

from __future__ import annotations

import json
import math
from collections.abc import Hashable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .data import BODYPARTS, eye_width

PCUTOFF = 0.6
PCK_THRESHOLD = 0.1  # fraction of the eye width
FAR_THRESHOLD = 0.25  # fraction of the eye width
PREDICTION_COORDS = ["x", "y", "likelihood"]


def find_predictions(project: Path, shuffle: int, snapshot: int) -> Path:
    """Return the predictions file DLC's evaluation wrote for one shuffle and snapshot.

    When the last epoch is also the best one, DLC saves the final model as snapshot-best-<epoch>,
    so both names are accepted.
    """
    eval_dir = Path(project) / "evaluation-results-pytorch" / "iteration-0"
    matches = [
        path
        for uid in (f"{snapshot:03d}", f"best-{snapshot:03d}")
        for path in eval_dir.glob(f"*shuffle{shuffle}/*_snapshot_{uid}.h5")
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one predictions file for shuffle {shuffle}, snapshot {snapshot}, "
            f"found {len(matches)} in {eval_dir}."
        )
    return matches[0]


def load_predictions(h5_path: Path) -> pd.DataFrame:
    """Load DLC predictions, indexed by (session, image) with (bodypart, x/y/likelihood) columns."""
    raw = pd.read_hdf(h5_path)
    if not isinstance(raw, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame in {h5_path}, got {type(raw).__name__}.")
    extra_levels: list[Hashable] = [
        level for level in ("scorer", "individuals") if level in raw.columns.names
    ]
    predictions = raw.droplevel(extra_levels, axis=1)
    predictions = predictions.reindex(
        columns=pd.MultiIndex.from_product(
            [BODYPARTS, PREDICTION_COORDS], names=["bodyparts", "coords"]
        )
    )
    predictions.index = pd.MultiIndex.from_arrays(
        [
            [str(idx[-2]) for idx in raw.index],
            [str(idx[-1]) for idx in raw.index],
        ],
        names=["session", "image"],
    )
    return predictions.astype(float)


def keypoint_table(
    predictions: pd.DataFrame, labels: pd.DataFrame, pcutoff: float = PCUTOFF
) -> pd.DataFrame:
    """One row per (frame, keypoint): error in pixels and in eye widths, likelihood, visibility.

    Only frames that have both a prediction and a label row are kept. Errors are normalized by
    the session's median eye width, so frames with a missing eye corner are still scored.
    """
    frames = predictions.index.intersection(labels.index)
    preds = predictions.loc[frames]
    truth = labels.loc[frames]
    sessions = frames.get_level_values("session")
    session_width = eye_width(labels).groupby(level="session").median()
    width = session_width.reindex(sessions).to_numpy()

    parts = []
    for bodypart in BODYPARTS:
        dx = (preds[(bodypart, "x")] - truth[(bodypart, "x")]).to_numpy()
        dy = (preds[(bodypart, "y")] - truth[(bodypart, "y")]).to_numpy()
        error = np.hypot(dx, dy)
        parts.append(
            pd.DataFrame(
                {
                    "session": sessions,
                    "image": frames.get_level_values("image"),
                    "bodypart": bodypart,
                    "error_px": error,
                    "error_norm": error / width,
                    "likelihood": preds[(bodypart, "likelihood")].to_numpy(),
                    "visible": truth[(bodypart, "x")].notna().to_numpy(),
                }
            )
        )
    table = pd.concat(parts, ignore_index=True)
    table["detected"] = table["likelihood"] >= pcutoff
    return table


def _mean(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else math.nan


def score(table: pd.DataFrame) -> dict[str, float | int]:
    """Summary scores of a keypoint table (or any subset of its rows)."""
    visible = table[table["visible"]]
    detected = visible[visible["detected"]]
    hidden = table[~table["visible"]]
    return {
        "keypoints": len(visible),
        "mean_error_px": _mean(visible["error_px"]),
        "median_error_px": (
            float(visible["error_px"].median()) if len(visible) else math.nan
        ),
        "mean_error_norm": _mean(visible["error_norm"]),
        "pck": _mean(visible["error_norm"] < PCK_THRESHOLD),
        "detection_rate": _mean(visible["detected"]),
        "detected_error_px": _mean(detected["error_px"]),
        "confident_wrong_rate": _mean(detected["error_norm"] > FAR_THRESHOLD),
        "false_detection_rate": _mean(hidden["detected"]),
    }


def score_by(table: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Summary scores per group, e.g. by=["session"] or by=["session", "bodypart"]."""
    records = []
    for key, group in table.groupby(by, sort=True):
        keys = key if isinstance(key, tuple) else (key,)
        records.append({**dict(zip(by, keys)), **score(group)})
    scores = pd.DataFrame(records).set_index(by)
    if "bodypart" in by and len(by) == 1:
        scores = scores.reindex(BODYPARTS)
    return scores


def orientation_accuracy(predictions: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """Per session, the share of frames where the predicted pupil anterior/posterior order is right."""
    frames = predictions.index.intersection(labels.index)
    preds = predictions.loc[frames]
    truth = labels.loc[frames]
    pred_left = preds[("pupil-anterior", "x")] < preds[("pupil-posterior", "x")]
    true_left = truth[("pupil-anterior", "x")] < truth[("pupil-posterior", "x")]
    labeled = (
        truth[("pupil-anterior", "x")].notna() & truth[("pupil-posterior", "x")].notna()
    )
    correct = (pred_left == true_left)[labeled]
    return correct.groupby(level="session").mean().rename("orientation_accuracy")


def evaluate_run(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    test_sessions: Iterable[str],
    pcutoff: float = PCUTOFF,
) -> dict:
    """All scores of one experiment, ready to save as metrics.json."""
    test_sessions = sorted(test_sessions)
    table = keypoint_table(predictions, labels, pcutoff)
    is_test = table["session"].isin(test_sessions)
    orientation = orientation_accuracy(predictions, labels)
    return {
        "pcutoff": pcutoff,
        "test_sessions": test_sessions,
        "test": score(table[is_test]),
        "train_frames_seen_in_training": score(table[~is_test]),
        "test_per_keypoint": score_by(table[is_test], ["bodypart"]).to_dict(
            orient="index"
        ),
        "per_session": score_by(table, ["session"]).to_dict(orient="index"),
        "test_orientation_accuracy": _mean(orientation.reindex(test_sessions).dropna()),
    }


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, float | np.floating):
        return None if math.isnan(value) else round(float(value), 4)
    return value


def save_metrics(metrics: dict, path: Path) -> None:
    """Write metrics as JSON; NaN (no data) becomes null."""
    Path(path).write_text(json.dumps(_json_ready(metrics), indent=2), encoding="utf-8")


def compare_runs(runs: dict[str, dict]) -> pd.DataFrame:
    """One row per experiment with its test scores, for side-by-side comparison."""
    rows = {
        name: {
            "test_sessions": ", ".join(metrics["test_sessions"]),
            **metrics["test"],
            "orientation_accuracy": metrics["test_orientation_accuracy"],
        }
        for name, metrics in runs.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index")
