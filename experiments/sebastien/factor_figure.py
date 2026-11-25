"""Plot one eye crop per session, titled with its conditions (facing side, eye size, pupil)."""

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data import (
    load_labels,
    session_conditions,
    unseen_condition_sessions,
)  # noqa: E402

LABELED = Path("data/labeled-data")
ORDER = ["094124", "131707", "135059", "110535", "171101", "115300", "172731", "214411"]


def main() -> None:
    labels = load_labels(LABELED)
    labels = labels[labels.notna().all(axis=1)]
    conditions = session_conditions(labels)
    unseen = unseen_condition_sessions(conditions)
    fig, axes = plt.subplots(2, 4, figsize=(12, 7))
    for ax, session in zip(axes.ravel(), ORDER):
        (_, image), row = next(labels.loc[[session]].iloc[5:6].iterrows())
        frame = cv2.imread(str(LABELED / session / image), cv2.IMREAD_GRAYSCALE)
        kp = row.to_numpy().reshape(8, 2)
        center = (kp[0] + kp[1]) / 2
        side = 2.6 * np.linalg.norm(kp[0] - kp[1])
        x0, y0 = (np.maximum(center - side / 2, 0)).astype(int)
        ax.imshow(
            frame[y0 : y0 + int(side), x0 : x0 + int(side)],
            cmap="gray",
            vmin=0,
            vmax=255,
        )
        ax.scatter(kp[:4, 0] - x0, kp[:4, 1] - y0, s=14, c="#2bd96b")
        ax.scatter(kp[4:, 0] - x0, kp[4:, 1] - y0, s=14, c="#ff5a5a")
        ax.scatter(
            kp[0, 0] - x0,
            kp[0, 1] - y0,
            s=70,
            facecolors="none",
            edgecolors="yellow",
            lw=1.5,
        )
        c = conditions.loc[session]
        highlight = session in unseen
        ax.set_title(
            f"{session}\nfaces {c['anterior_side']} · {c['eye_size']} eye · {c['pupil']} pupil",
            fontsize=9,
            color="#c0392b" if highlight else "black",
            fontweight="bold" if highlight else "normal",
        )
        ax.axis("off")
    fig.suptitle(
        "Sessions differ in facing side, eye size and pupil polarity\n"
        "red = combination absent from the other 7 sessions · yellow ring = eye-anterior label",
        fontsize=11,
    )
    plt.subplots_adjust(hspace=0.3, top=0.86, bottom=0.02, left=0.02, right=0.98)
    plt.savefig("results/sebastien/session_factors.png", dpi=130)


if __name__ == "__main__":
    main()
