"""Lighting augmentation, and offline copies of labeled sessions for DLC.

DLC's training config has no gamma, contrast or exposure augmentation (only hist_eq, motion blur,
noise, covering, elastic, affine, hflip). Rather than patching DLC, we write lighting-altered
copies of the labeled frames as extra labeled-data folders with the same labels; they are used
for training only (see `src.shuffles`).

The copies live in their own DLC project (`create_photometric_project`): DLC merges every folder
listed in config.yaml into each new training set and stores splits as row numbers into that
merge, so adding folders to a project that has shuffles would silently shift their splits.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from deeplabcut.utils import auxiliaryfunctions

COPY_SUFFIX = "_photo"


def photometric_variant(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a uint8 image with random gamma, gain, contrast, overexposure, blur and noise.

    The ranges span what the sessions show, from dim dark-pupil frames (mean ~35/255) to
    bright-pupil frames saturated around the eye (110535), while keeping the eye visible: wider
    ranges turned whole frames black or white, which would teach DLC labels on an invisible eye.
    """
    x = image.astype(np.float32) / 255.0
    x = x ** rng.uniform(0.5, 1.6)
    x = x * rng.uniform(0.7, 2.2)
    mean = x.mean()
    x = (x - mean) * rng.uniform(0.75, 1.4) + mean
    if rng.random() < 0.3:
        x = x * rng.uniform(1.3, 2.0)
    if rng.random() < 0.3:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 1.5))
    x = x + rng.normal(0, rng.uniform(0, 0.03), x.shape).astype(np.float32)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def copy_name(session: str, copy: int) -> str:
    return f"{session}{COPY_SUFFIX}{copy}"


def base_session(folder: str) -> str:
    """Return the session a labeled-data folder belongs to (itself, or the session it copies)."""
    return folder.split(COPY_SUFFIX)[0]


def write_photometric_copies(
    config_path: Path, sessions: list[str], copies: int, seed: int = 0
) -> list[str]:
    """Write `copies` lighting-altered copies of each session's labeled frames into the project.

    Each copy is a new labeled-data folder (e.g. `094124_photo0`) holding the altered frames and the
    session's labels, registered in config.yaml's video_sets so DLC merges it. Existing copies are
    left as they are.

    Returns:
        The names of all copy folders for these sessions.
    """
    cfg = auxiliaryfunctions.read_config(str(config_path))
    project = Path(config_path).parent
    if any(project.glob("training-datasets/iteration-*/*/Documentation_data-*.pickle")):
        raise RuntimeError(
            f"{project} already has shuffles; new labeled-data folders would shift their splits. "
            "Use create_photometric_project to work in a copy."
        )
    labeled = project / "labeled-data"
    scorer = cfg["scorer"]
    rng = np.random.default_rng(seed)
    folders = []
    for session in sessions:
        labels = pd.read_hdf(labeled / session / f"CollectedData_{scorer}.h5")
        for copy in range(copies):
            folder = copy_name(session, copy)
            folders.append(folder)
            out = labeled / folder
            if (out / f"CollectedData_{scorer}.h5").exists():
                continue
            out.mkdir(exist_ok=True)
            for image in labels.index.get_level_values(-1):
                frame = cv2.imread(str(labeled / session / image))
                cv2.imwrite(str(out / image), photometric_variant(frame, rng))
            copied = labels.copy()
            copied.index = pd.MultiIndex.from_tuples(
                [(idx[0], folder, idx[-1]) for idx in labels.index]
            )
            copied.to_hdf(
                out / f"CollectedData_{scorer}.h5", key="df_with_missing", mode="w"
            )
            copied.to_csv(out / f"CollectedData_{scorer}.csv")
    _register_folders(config_path, folders)
    return folders


def _folder_of(video: str) -> str:
    """Labeled-data folder of a video_sets entry; entries may be Windows paths."""
    return Path(video.replace("\\", "/")).stem


def create_photometric_project(
    source_project: Path, target_project: Path, copies: int, seed: int = 0
) -> Path:
    """Copy a DLC project's config and labeled data (not its shuffles) and add photometric copies.

    Returns:
        The new project's config.yaml path.
    """
    source_project, target_project = Path(source_project), Path(target_project)
    cfg = auxiliaryfunctions.read_config(str(source_project / "config.yaml"))
    cfg["project_path"] = str(target_project.resolve())
    video_sets = {}
    for video, settings in cfg["video_sets"].items():
        session = _folder_of(video)
        if base_session(session) != session:
            continue
        shutil.copytree(
            source_project / "labeled-data" / session,
            target_project / "labeled-data" / session,
            dirs_exist_ok=True,
        )
        video_sets[f"{cfg['project_path']}/videos/{session}.mp4"] = settings
    cfg["video_sets"] = video_sets
    sessions = [_folder_of(video) for video in video_sets]
    config_path = target_project / "config.yaml"
    auxiliaryfunctions.write_config(str(config_path), cfg)
    write_photometric_copies(config_path, sessions, copies, seed)
    return config_path


def _register_folders(config_path: Path, folders: list[str]) -> None:
    """Add labeled-data folders to config.yaml's video_sets (DLC merges only the folders listed there)."""
    cfg = auxiliaryfunctions.read_config(str(config_path))
    known = {_folder_of(video) for video in cfg["video_sets"]}
    template = next(iter(cfg["video_sets"].values()))
    for folder in folders:
        if folder not in known:
            cfg["video_sets"][f"{cfg['project_path']}/videos/{folder}.mp4"] = dict(
                template
            )
    auxiliaryfunctions.write_config(str(config_path), cfg)
