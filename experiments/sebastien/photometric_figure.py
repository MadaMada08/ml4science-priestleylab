"""Plot random lighting-altered copies from src/photometric.py next to the original frames."""

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.photometric import photometric_variant  # noqa: E402

# (session, frame, eye center) covering the dark-pupil, bright-pupil and overexposed looks
EXAMPLES = [
    ("094124", "img010418.png", (286, 381)),
    ("172731", "img001254.png", (782, 544)),
    ("110535", "img025189.png", (290, 394)),
]


def main() -> None:
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(EXAMPLES), 6, figsize=(14, 7.2))
    for row, (session, image, (cx, cy)) in zip(axes, EXAMPLES):
        frame = cv2.imread(f"data/labeled-data/{session}/{image}", cv2.IMREAD_GRAYSCALE)
        panels = [frame] + [photometric_variant(frame, rng) for _ in range(5)]
        for k, (ax, panel) in enumerate(zip(row, panels)):
            ax.imshow(
                panel[cy - 150 : cy + 150, cx - 190 : cx + 190],
                cmap="gray",
                vmin=0,
                vmax=255,
            )
            ax.set_title(f"{session} original" if k == 0 else f"copy {k}", fontsize=9)
            ax.axis("off")
    fig.suptitle(
        "src/photometric.py: random lighting-altered copies (crops around the eye; DLC sees the full frame)",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig("results/sebastien/photometric_examples.png", dpi=105)


if __name__ == "__main__":
    main()
