"""Create, configure, train and evaluate leave-one-session-out DLC shuffles.

Moved from experiments_edouard.ipynb so every notebook shares one implementation. Changes:
- the split is computed from a fresh merge of the labels (done in a temporary folder, so nothing
  in the project changes), instead of from a merged file read once when the notebook starts;
- lighting-altered copies of a session (`src.photometric`) count as that session: copies of the
  held-out session are never trained on;
- DLC's `snapshot-best-<epoch>` name, used when the last epoch is also the best one, is handled.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import deeplabcut
import pandas as pd
from deeplabcut.generate_training_dataset.trainingsetmanipulation import (
    merge_annotateddatasets,
)
from deeplabcut.utils import auxiliaryfunctions

from .photometric import base_session


class DlcProject:
    """A DLC project folder; every DLC call goes through its config.yaml."""

    def __init__(self, project: Path) -> None:
        self.project = Path(project)
        self.config_path = str(self.project / "config.yaml")
        cfg = auxiliaryfunctions.read_config(self.config_path)
        if cfg["project_path"] != str(self.project.resolve()):
            cfg["project_path"] = str(self.project.resolve())
            auxiliaryfunctions.write_config(self.config_path, cfg)
        self.models_dir = (
            self.project / "dlc-models-pytorch" / f"iteration-{cfg['iteration']}"
        )
        self.eval_dir = (
            self.project
            / "evaluation-results-pytorch"
            / f"iteration-{cfg['iteration']}"
        )

    def frame_folders(self) -> list[str]:
        """Return the labeled-data folder of each row of the labels as DLC will merge them.

        DLC stores a shuffle's split as row numbers into this merge, so the order must match the
        one `create_training_dataset` produces; both come from the same deterministic merge.
        """
        cfg = auxiliaryfunctions.read_config(self.config_path)
        with tempfile.TemporaryDirectory() as tmp:
            merged = merge_annotateddatasets(cfg, Path(tmp))
        return [str(idx[1]) for idx in merged.index]

    def frame_counts(self) -> pd.Series:
        return pd.Series(self.frame_folders()).value_counts().sort_index()

    def split_indices(
        self, test_session: str, with_copies: bool = True
    ) -> tuple[list[int], list[int]]:
        """Return the (train, test) row indices of the merged labels, holding out one session.

        Test rows are the session's own frames; its lighting-altered copies are in neither set.
        With `with_copies=False` no copy is trained on, so a baseline and its augmented version can
        share one project (and one merged label file).
        """
        folders = self.frame_folders()
        test = [i for i, folder in enumerate(folders) if folder == test_session]
        train = [
            i
            for i, folder in enumerate(folders)
            if base_session(folder) != test_session
            and (with_copies or base_session(folder) == folder)
        ]
        if not test:
            raise ValueError(f"No labeled frames for session {test_session}.")
        return train, test

    def shuffle_dir(self, shuffle: int) -> Path | None:
        """Return a shuffle's model folder, or None if the shuffle doesn't exist yet."""
        matches = list(self.models_dir.glob(f"*shuffle{shuffle}"))
        return matches[0] if matches else None

    def create_shuffle(
        self, shuffle: int, test_session: str, with_copies: bool = True
    ) -> None:
        """Create a shuffle holding out one session, unless the shuffle already exists."""
        if self.shuffle_dir(shuffle) is not None:
            print(f"Shuffle {shuffle} already exists, skipped.")
            return
        train, test = self.split_indices(test_session, with_copies)
        print(
            f"Shuffle {shuffle}: test on {test_session} | train frames: {len(train)} "
            f"| test frames: {len(test)}"
        )
        # DLC names a shuffle after its train fraction and requires that fraction in config.yaml's
        # TrainingFraction list.
        fraction = round(len(train) / (len(train) + len(test)), 2)
        cfg = auxiliaryfunctions.read_config(self.config_path)
        if fraction not in cfg["TrainingFraction"]:
            cfg["TrainingFraction"].append(fraction)
            auxiliaryfunctions.write_config(self.config_path, cfg)
        deeplabcut.create_training_dataset(
            self.config_path,
            Shuffles=[shuffle],
            trainIndices=[train],
            testIndices=[test],
            net_type="resnet_50",
        )

    def trainingset_index(self, shuffle: int) -> int:
        """Position of a shuffle's train fraction in config.yaml's TrainingFraction list, as DLC's functions expect."""
        folder = self.shuffle_dir(shuffle)
        assert folder is not None, f"Shuffle {shuffle} doesn't exist."
        percent = int(folder.name.split("trainset")[1].split("shuffle")[0])
        fractions = auxiliaryfunctions.read_config(self.config_path)["TrainingFraction"]
        return [round(100 * f) for f in fractions].index(percent)

    def configure_shuffle(
        self,
        shuffle: int,
        train: dict,
        inference: dict,
        epochs: int,
        lr_milestones: list[int] | None = None,
    ) -> None:
        """Overwrite chosen training settings in a shuffle's pytorch_config.yaml.

        Args:
            lr_milestones: epochs at which the learning rate drops (DLC's default is [90, 120]).
                They count epochs, so a run over a 3x larger training set needs them divided
                by 3 to follow the same schedule in gradient steps.
        """
        folder = self.shuffle_dir(shuffle)
        assert folder is not None, f"Shuffle {shuffle} doesn't exist."
        path = folder / "train" / "pytorch_config.yaml"
        pcfg = auxiliaryfunctions.read_plainconfig(path)
        pcfg["data"]["train"].update(train)
        pcfg["data"]["inference"].update(inference)
        pcfg["train_settings"]["epochs"] = epochs
        if lr_milestones is not None:
            pcfg["runner"]["scheduler"]["params"]["milestones"] = lr_milestones
        milestones = pcfg["runner"]["scheduler"]["params"]["milestones"]
        auxiliaryfunctions.write_plainconfig(path, pcfg)
        print(
            f"Shuffle {shuffle} | train: {train} | inference: {inference} | epochs: {epochs} "
            f"| lr drops at epochs {milestones}"
        )

    def final_snapshot(self, shuffle: int, epochs: int) -> str | None:
        """Name of the snapshot saved at the last epoch, or None if training hasn't finished.

        When the last epoch is also the best one, DLC saves it only as snapshot-best-<epoch>.
        """
        folder = self.shuffle_dir(shuffle)
        assert folder is not None, f"Shuffle {shuffle} doesn't exist."
        for name in (f"snapshot-{epochs:03d}", f"snapshot-best-{epochs:03d}"):
            if (folder / "train" / f"{name}.pt").exists():
                return name
        return None

    def train_shuffle(self, shuffle: int, epochs: int) -> None:
        """Train a shuffle, unless its final snapshot already exists."""
        if self.final_snapshot(shuffle, epochs) is not None:
            print(f"Shuffle {shuffle} already trained, skipped.")
            return
        deeplabcut.train_network(
            self.config_path,
            shuffle=shuffle,
            trainingsetindex=self.trainingset_index(shuffle),
        )

    def evaluate_shuffle(self, shuffle: int, epochs: int, force: bool = False) -> None:
        """Evaluate a shuffle's final snapshot, unless already done (never a 'best' one picked with the test score).

        DLC applies hist_eq with probability 0.5, at inference too: with hist_eq in the inference
        settings, re-evaluating can change the numbers, so use force=True on purpose only.
        """
        snapshot = self.final_snapshot(shuffle, epochs)
        assert (
            snapshot is not None
        ), f"Shuffle {shuffle} has no snapshot for epoch {epochs}."
        uid = snapshot.removeprefix("snapshot-")
        done = list(
            self.eval_dir.glob(f"*shuffle{shuffle}/*_snapshot_{uid}-results.csv")
        )
        if done and not force:
            print(f"Shuffle {shuffle} already evaluated, skipped.")
            return
        deeplabcut.evaluate_network(
            self.config_path,
            Shuffles=[shuffle],
            trainingsetindex=self.trainingset_index(shuffle),
            snapshots_to_evaluate=[snapshot],
            per_keypoint_evaluation=True,
        )
