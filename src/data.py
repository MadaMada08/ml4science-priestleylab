"""Load the labeled eye/pupil keypoints and the per-session metadata."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

BODYPARTS = [
    "eye-anterior",
    "eye-posterior",
    "eye-dorsal",
    "eye-ventral",
    "pupil-anterior",
    "pupil-posterior",
    "pupil-dorsal",
    "pupil-ventral",
]
COORDS = ["x", "y"]
DEFAULT_SCORER = "dani"


def load_session_labels(
    session_dir: Path, scorer: str = DEFAULT_SCORER
) -> pd.DataFrame:
    """Load one session's labels, indexed by (session, image) with (bodypart, coord) columns.

    Images present in the folder but missing from the label file are added as all-NaN rows:
    in this dataset they are frames where the eye is not visible.
    """
    session_dir = Path(session_dir)
    session = session_dir.name
    raw = pd.read_hdf(session_dir / f"CollectedData_{scorer}.h5")
    if not isinstance(raw, pd.DataFrame):
        raise TypeError(
            f"Expected a DataFrame in {session_dir}, got {type(raw).__name__}."
        )
    labels = raw.droplevel("scorer", axis=1)
    # reindex would silently turn a file with other bodypart names (e.g. the lab's original
    # eye-left/eye-right export) into all-NaN labels, so check first.
    missing = set(BODYPARTS) - set(labels.columns.get_level_values("bodyparts"))
    if missing:
        raise ValueError(
            f"{session_dir / f'CollectedData_{scorer}.h5'} lacks bodyparts {sorted(missing)}; "
            "convert it to anatomical names first (python -m src.canonicalize)."
        )
    labels = labels.reindex(
        columns=pd.MultiIndex.from_product(
            [BODYPARTS, COORDS], names=["bodyparts", "coords"]
        )
    )
    images = [str(idx[-1]) for idx in labels.index]
    labels.index = pd.MultiIndex.from_arrays(
        [[session] * len(images), images], names=["session", "image"]
    )

    unlabeled = sorted({p.name for p in session_dir.glob("*.png")} - set(images))
    if unlabeled:
        empty = pd.DataFrame(
            np.nan,
            index=pd.MultiIndex.from_product(
                [[session], unlabeled], names=["session", "image"]
            ),
            columns=labels.columns,
        )
        labels = pd.concat([labels, empty])

    return labels.sort_index().astype(float)


def load_labels(labeled_data_dir: Path, scorer: str = DEFAULT_SCORER) -> pd.DataFrame:
    """Load every labeled session under `labeled-data/`; sessions without a label file are skipped."""
    sessions = []
    for session_dir in sorted(Path(labeled_data_dir).iterdir()):
        if not session_dir.is_dir():
            continue
        if not (session_dir / f"CollectedData_{scorer}.h5").exists():
            warnings.warn(
                f"Session {session_dir.name} has no CollectedData_{scorer}.h5, skipped.",
                stacklevel=2,
            )
            continue
        sessions.append(load_session_labels(session_dir, scorer))
    return pd.concat(sessions)


def load_metadata(xlsx_path: Path) -> pd.DataFrame:
    """Load the lab's session table, indexed by session id."""
    raw = pd.read_excel(xlsx_path, dtype=str)
    metadata = pd.DataFrame(
        {
            "mouse": raw["Mouse"],
            "date": pd.to_datetime(raw["Date"], format="%Y%m%d"),
            "difficulty": raw["Diff"],
            "setup": raw["Setup"],
            "orientation": raw["Orient"],
            "comments": raw["Comments"],
            "labeled": raw["Labeled"].str.lower().eq("true"),
            "video_url": raw["URL"],
        }
    )
    metadata.index = pd.Index(raw["ID"].str.zfill(6), name="session")
    return metadata


def eye_width(labels: pd.DataFrame) -> pd.Series:
    """Anterior-posterior eye distance per frame, in pixels; the natural scale for normalizing errors."""
    dx = labels[("eye-anterior", "x")] - labels[("eye-posterior", "x")]
    dy = labels[("eye-anterior", "y")] - labels[("eye-posterior", "y")]
    return pd.Series(np.hypot(dx, dy), index=labels.index, name="eye_width_px")


def session_summary(labels: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """One row per labeled session: frame counts, eye size and the lab's metadata."""
    sessions = labels.index.get_level_values("session")
    anterior_left = labels[("eye-anterior", "x")] < labels[("eye-posterior", "x")]
    summary = pd.DataFrame(
        {
            "frames": labels.groupby(level="session").size(),
            "eye_not_visible": labels.isna().all(axis=1).groupby(sessions).sum(),
            "frames_with_missing_keypoints": labels.isna()
            .any(axis=1)
            .groupby(sessions)
            .sum(),
            "eye_width_px": eye_width(labels).groupby(sessions).mean().round(1),
            "anterior_side": anterior_left.groupby(sessions)
            .mean()
            .gt(0.5)
            .map({True: "left", False: "right"}),
        }
    )
    return summary.join(metadata[["mouse", "setup", "difficulty", "comments"]])


# Read off the frames by eye: bright-pupil IR illumination makes the pupil white.
PUPIL_POLARITY = {
    "094124": "dark",
    "110535": "bright",
    "115300": "bright",
    "131707": "dark",
    "135059": "dark",
    "171101": "dark",
    "172731": "bright",
    "214411": "bright",
}
LARGE_EYE_PX = 125  # the two camera setups give ~95-100 px and ~150-170 px eye widths


def session_conditions(labels: pd.DataFrame) -> pd.DataFrame:
    """One row per session: which way the eye faces, eye size class and pupil polarity."""
    sessions = labels.index.get_level_values("session")
    anterior_left = labels[("eye-anterior", "x")] < labels[("eye-posterior", "x")]
    width = eye_width(labels).groupby(sessions).median()
    conditions = pd.DataFrame(
        {
            "anterior_side": anterior_left.groupby(sessions)
            .mean()
            .gt(0.5)
            .map({True: "left", False: "right"}),
            "eye_size": width.gt(LARGE_EYE_PX).map({True: "large", False: "small"}),
            "pupil": pd.Series(PUPIL_POLARITY),
        }
    )
    return conditions.loc[width.index]


def unseen_condition_sessions(conditions: pd.DataFrame) -> list[str]:
    """Sessions whose combination of conditions no other session has.

    Held out, these are the sessions a model has to generalize to; report them separately,
    since averaging over all sessions hides their errors.
    """
    combo = conditions.apply(tuple, axis=1)
    return sorted(combo[~combo.duplicated(keep=False)].index)


def load_dataset(
    data_dir: Path, scorer: str = DEFAULT_SCORER
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load labels and metadata from a `data/` folder holding `labeled-data/` and `metadata.xlsx`."""
    data_dir = Path(data_dir)
    return load_labels(data_dir / "labeled-data", scorer), load_metadata(
        data_dir / "metadata.xlsx"
    )
