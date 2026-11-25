"""Build a DeepLabCut project from a labeled-data folder.

Reconstructs a DLC project around existing labels: makes a bootstrap video
(DLC needs one video to create a project), configures it to the lab's
settings, and copies the labeled-data in. One call per data variant.
"""

import shutil
from pathlib import Path

import cv2
import deeplabcut
from deeplabcut.utils import auxiliaryfunctions

BODYPARTS = ["eye-anterior", "eye-posterior", "eye-dorsal", "eye-ventral",
             "pupil-anterior", "pupil-posterior", "pupil-dorsal", "pupil-ventral"]


def labeled_sessions(data_root):
    root = Path(data_root) / "labeled-data"
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and list(p.glob("CollectedData_*.h5")))


def _bootstrap_video(data_root, session):
    frames = sorted((Path(data_root) / "labeled-data" / session).glob("img*.png"))
    h, w = cv2.imread(str(frames[0])).shape[:2]
    out = Path(data_root) / "bootstrap.avi"
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"MJPG"), 10, (w, h))
    for f in frames:
        vw.write(cv2.imread(str(f)))
    vw.release()
    return str(out)


def build_project(data_root, name, work_dir="./dlc", frame_size=(1280, 1024)):
    """Create a DLC project named <name> from the labels in data_root. Returns config_path."""
    data_root = Path(data_root)
    sessions = labeled_sessions(data_root)
    boot = _bootstrap_video(data_root, sessions[0])

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
        src, dst = data_root / "labeled-data" / s, Path(proj) / "labeled-data" / s
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            shutil.copy2(f, dst / f.name)

    Path(boot).unlink(missing_ok=True)
    return config_path
