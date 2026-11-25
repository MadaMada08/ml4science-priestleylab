"""Build and run a 3-scenario experiment (hard / easy / mixed).

An experiment = one DLC project (one set of frames) + three shuffles,
one per scenario. build_experiment sets it up; check_experiment verifies
the splits BEFORE training; train_all / evaluate_all run them.
"""

import pickle
from collections import Counter
from pathlib import Path

import pandas as pd
import deeplabcut
from deeplabcut.utils import auxiliaryfunctions

from . import data as _data
from . import splits as _splits


def _merged_df(config_path):
    cfg = auxiliaryfunctions.read_config(config_path)
    folder = Path(cfg["project_path"]) / auxiliaryfunctions.get_training_set_folder(cfg)
    return pd.read_hdf(next(folder.glob("CollectedData_*.h5")))


def _create_shuffle(config_path, shuffle, train, test):
    try:
        deeplabcut.create_training_dataset(
            config_path, Shuffles=[shuffle],
            trainIndices=[train], testIndices=[test], net_type="resnet_50")
    except ValueError as e:
        if "already exists" not in str(e):
            raise


def build_experiment(labeled_dir, name, work_dir="./dlc"):
    """Build project <name> from labeled_dir and create the 3 scenario shuffles. Returns config_path."""
    config_path = _data.build_project(labeled_dir, name, work_dir)

    deeplabcut.create_training_dataset(config_path, Shuffles=[99])  # dummy merge → learn frame order
    df = _merged_df(config_path)
    total = len(df)

    plans = {sh: (lbl, fn(df)) for sh, (lbl, fn) in _splits.SCENARIOS.items()}

    cfg = auxiliaryfunctions.read_config(config_path)
    cfg["TrainingFraction"] = sorted({round(len(tr) / total, 2) for _, (tr, te) in plans.values()} | {0.95})
    auxiliaryfunctions.write_config(config_path, cfg)

    for sh, (lbl, (tr, te)) in plans.items():
        _create_shuffle(config_path, sh, tr, te)
    return config_path


def check_experiment(config_path):
    """Print the train/test split of each scenario shuffle — run this before training."""
    order = list(_merged_df(config_path).index)
    cfg = auxiliaryfunctions.read_config(config_path)
    folder = Path(cfg["project_path"]) / auxiliaryfunctions.get_training_set_folder(cfg)
    for sh, (lbl, _) in _splits.SCENARIOS.items():
        pkl = next(folder.glob(f"Documentation_data-*shuffle{sh}.pickle"))
        with open(pkl, "rb") as f:
            _, train_idx, test_idx, _ = pickle.load(f)
        sessions = dict(Counter(order[i][1] for i in test_idx))
        print(f"shuffle {sh} [{lbl}]: train={len(train_idx)} test={len(test_idx)} | test sessions: {sessions}")


def _trainset_index(config_path, shuffle):
    cfg = auxiliaryfunctions.read_config(config_path)
    folder = Path(cfg["project_path"]) / auxiliaryfunctions.get_training_set_folder(cfg)
    pkl = next(folder.glob(f"Documentation_data-*shuffle{shuffle}.pickle"))
    frac = float(pkl.stem.split("_")[-1].replace(f"shuffle{shuffle}", "")) / 100
    return cfg["TrainingFraction"].index(round(frac, 2))


def train_all(config_path, max_iters=50000, **kw):
    """Train all three scenario shuffles (the long run)."""
    for sh in _splits.SCENARIOS:
        tsi = _trainset_index(config_path, sh)
        print(f"=== training shuffle {sh} ({_splits.SCENARIOS[sh][0]}), trainingsetindex={tsi} ===")
        deeplabcut.train_network(config_path, shuffle=sh, trainingsetindex=tsi,
                                 max_iters=max_iters, **kw)


def evaluate_all(config_path, plotting=False):
    """Evaluate all three scenario shuffles. plotting=True also writes DLC's
    per-frame overlays (needs Windows long paths enabled)."""
    for sh in _splits.SCENARIOS:
        tsi = _trainset_index(config_path, sh)
        deeplabcut.evaluate_network(config_path, Shuffles=[sh], trainingsetindex=tsi,
                                    plotting=plotting, per_keypoint_evaluation=True)