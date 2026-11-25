"""Convert a copy of the lab's labeled-data export to the layout the rest of `src/` expects.

The export mixes three label files per session: `CollectedData_dani.h5` with image-side names
(eye-left, eye-right, ...), `CollectedData_dani_renamed.h5` with anatomical names, and, for
171101, `CollectedData_ml_project.h5`. This writes one `CollectedData_dani.h5` per session with
anatomical names and scorer "dani", image paths pointing at the session's own folder, known
anterior/posterior swaps fixed, and copies the frames.
The source folder is never modified.

    python -m src.canonicalize ~/Downloads/labeled-data data/labeled-data
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from .data import BODYPARTS, DEFAULT_SCORER

# Side of the image the nose is on, checked by eye. 135059 was exported with anterior/posterior
# the wrong way round; stating the true side (rather than "swap") keeps the fix idempotent.
KNOWN_ANTERIOR_SIDE = {"135059": "left"}
SWAP = {
    "eye-anterior": "eye-posterior",
    "eye-posterior": "eye-anterior",
    "pupil-anterior": "pupil-posterior",
    "pupil-posterior": "pupil-anterior",
}


def _anatomical_label_file(session_dir: Path) -> Path | None:
    """Return the session's label file that uses anatomical bodypart names, if any."""
    candidates = [session_dir / f"CollectedData_{DEFAULT_SCORER}_renamed.h5"]
    candidates += sorted(session_dir.glob("CollectedData_*.h5"))
    for path in candidates:
        if not path.exists():
            continue
        names = set(pd.read_hdf(path).columns.get_level_values("bodyparts"))
        if set(BODYPARTS) <= names:
            return path
    return None


def canonical_labels(session_dir: Path) -> pd.DataFrame | None:
    """Return the session's labels with anatomical names, scorer "dani" and known swaps fixed."""
    path = _anatomical_label_file(session_dir)
    if path is None:
        return None
    labels = pd.read_hdf(path).droplevel("scorer", axis=1)
    session = session_dir.name
    if session in KNOWN_ANTERIOR_SIDE:
        anterior_left = (
            labels[("eye-anterior", "x")] < labels[("eye-posterior", "x")]
        ).mean() > 0.5
        if anterior_left != (KNOWN_ANTERIOR_SIDE[session] == "left"):
            labels = labels.rename(columns=SWAP, level="bodyparts")
    labels = labels[pd.MultiIndex.from_product([BODYPARTS, ["x", "y"]])]
    # DLC loads frames from the stored path; the 110535 export points at a "110535_2" folder.
    labels.index = pd.MultiIndex.from_tuples(
        [("labeled-data", session, str(idx[-1])) for idx in labels.index]
    )
    labels.columns = pd.MultiIndex.from_tuples(
        [(DEFAULT_SCORER, *col) for col in labels.columns],
        names=["scorer", "bodyparts", "coords"],
    )
    return labels


def canonicalize(source: Path, destination: Path) -> list[str]:
    """Write a canonical copy of every session under `source` into `destination`.

    Returns:
        The sessions that had no anatomical label file (frames are still copied).
    """
    unlabeled = []
    for session_dir in sorted(p for p in Path(source).iterdir() if p.is_dir()):
        out = Path(destination) / session_dir.name
        out.mkdir(parents=True, exist_ok=True)
        for frame in session_dir.glob("*.png"):
            shutil.copy2(frame, out / frame.name)
        labels = canonical_labels(session_dir)
        if labels is None:
            unlabeled.append(session_dir.name)
            continue
        target = out / f"CollectedData_{DEFAULT_SCORER}.h5"
        labels.to_hdf(target, key="df_with_missing", mode="w")
        labels.to_csv(target.with_suffix(".csv"))
    return unlabeled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        parser.error("destination must differ from source; the export is kept as is.")
    unlabeled = canonicalize(args.source, args.destination)
    if unlabeled:
        print(f"No anatomical labels for: {', '.join(unlabeled)} (frames copied).")


if __name__ == "__main__":
    main()
