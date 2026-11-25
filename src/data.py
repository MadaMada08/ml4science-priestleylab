"""Build a DeepLabCut project from a labeled-data folder.

`labeled_dir` is the path to a labeled-data folder whose subfolders are
sessions, e.g. data/labeled-data or a preprocessed variant
data/labeled-data_flipped. All variants live inside the single data/ folder.
"""

import shutil
from pathlib import Path

import cv2
import deeplabcut
from deeplabcut.utils import auxiliaryfunctions

BODYPARTS = ["eye-anterior", "eye-posterior", "eye-dorsal", "eye-ventral",
             "pupil-anterior", "pupil-posterior", "pupil-dorsal", "pupil-ventral"]


def _sessions(labeled_dir):
    root = Path(labeled_dir)
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and list(p.glob("CollectedData_*.h5")))


def _bootstrap_video(labeled_dir, session):
    frames = sorted((Path(labeled_dir) / session).glob("img*.png"))
    h, w = cv2.imread(str(frames[0])).shape[:2]
    out = Path(labeled_dir).parent / "bootstrap.avi"
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"MJPG"), 10, (w, h))
    for f in frames:
        vw.write(cv2.imread(str(f)))
    vw.release()
    return str(out)


def _existing_project(work_dir, name):
    matches = sorted(Path(work_dir).glob(f"{name}-*"))
    return matches[0] if matches else None


def build_project(labeled_dir, name, work_dir="./dlc", frame_size=(1280, 1024)):
    """Create a DLC project named <name> from the labels in labeled_dir. Returns config_path."""
    existing = _existing_project(work_dir, name)
    if existing:
        raise FileExistsError(
            f"Project '{name}' already exists at {existing}. "
            f"delete it or choose a different name.")

    labeled_dir = Path(labeled_dir)
    sessions = _sessions(labeled_dir)
    boot = _bootstrap_video(labeled_dir, sessions[0])

    config_path = deeplabcut.create_new_project(
        name, "dani", [boot], working_directory=str(work_dir), copy_videos=False)

    cfg = auxiliaryfunctions.read_config(config_path)
    w, h = frame_size
    proj = cfg["project_path"]
    cfg["bodyparts"] = BODYPARTS
    cfg["video_sets"] = {f"{proj}/videos/{s}.mp4": {"crop": f"0, {w}, 0, {h}"} for s in sessions}
    cfg["pcutoff"] = 0.6
    cfg["default_net_type"] = "resnet_50"
    auxiliaryfunctions.write_config(config_path, cfg)

    for s in sessions:
        src, dst = labeled_dir / s, Path(proj) / "labeled-data" / s
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            shutil.copy2(f, dst / f.name)

    Path(boot).unlink(missing_ok=True)
    return config_path
