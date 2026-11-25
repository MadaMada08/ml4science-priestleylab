"""Collect the per-fold JSON results into tables and a figure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(sys.argv[1] if len(sys.argv) > 1 else "results/sebastien")
ORDER = ["none", "hflip", "hflip+photo", "hflip+photo+invert"]
UNSEEN = {
    "110535",
    "171101",
}  # src.data.unseen_condition_sessions on the labeled sessions


def load(kind: str = "oracle", e2e: bool = False, seed: int | None = 0) -> pd.DataFrame:
    """Load one result kind ("oracle" or "end_to_end") of every run; seed=None keeps all seeds."""
    rows = []
    for path in sorted(RESULTS.glob("*.json")):
        if ("__e2e" in path.stem) != e2e:
            continue
        run_seed = int(path.stem.rsplit("seed", 1)[1])
        if seed is not None and run_seed != seed:
            continue
        r = json.loads(path.read_text())
        if kind not in r:
            continue
        rows.append(
            {
                "session": r["test_session"],
                "condition": r["condition"],
                "seed": run_seed,
                **r[kind],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = load("oracle")
    df["condition"] = pd.Categorical(df["condition"], ORDER, ordered=True)
    for metric in ["median_px", "pck10", "orientation"]:
        table = df.pivot_table(
            index="session", columns="condition", values=metric, observed=True
        )
        table.loc["mean (all 8)"] = table.mean()
        table.loc["mean (unseen combos)"] = table.loc[
            sorted(UNSEEN & set(table.index))
        ].mean()
        print(f"\n== {metric} ==")
        print(table.round(3).to_string())
        table.round(3).to_csv(RESULTS / f"table_{metric}.csv")

    sessions = sorted(df["session"].unique(), key=lambda s: (s not in UNSEEN, s))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    colors = ["#9aa5b1", "#5b8def", "#2e9e6b", "#c97b2a"]
    for ax, metric, label in [
        (axes[0], "median_px", "median error (px, full resolution) — lower is better"),
        (axes[1], "pck10", "PCK@0.1 eye width — higher is better"),
    ]:
        width = 0.2
        for k, (cond, color) in enumerate(zip(ORDER, colors)):
            sub = df[df["condition"] == cond].set_index("session").reindex(sessions)
            ax.bar(
                [i + (k - 1.5) * width for i in range(len(sessions))],
                sub[metric],
                width,
                label=cond,
                color=color,
            )
        ax.set_xticks(range(len(sessions)))
        ax.set_xticklabels(
            [s + ("*" if s in UNSEEN else "") for s in sessions], rotation=0
        )
        ax.set_title(label, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Leave-one-session-out, eye-crop keypoint model (* = factor combination unseen in training)",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(RESULTS / "conditions.png", dpi=130)

    seeds = load("oracle", seed=None)
    multi = seeds.groupby(["session", "condition"]).filter(lambda g: len(g) > 1)
    if len(multi):
        print("\n== seed check ==")
        print(
            multi.pivot_table(
                index=["session", "condition"],
                columns="seed",
                values=["median_px", "pck10"],
            )
            .round(3)
            .to_string()
        )

    cols = [
        "session",
        "condition",
        "median_px",
        "pck10",
        "orientation",
        "stage1_center_err_median_px",
        "stage1_center_err_max_px",
    ]
    e2e = load("end_to_end", e2e=True)
    if len(e2e):
        oracle = load("oracle", e2e=True).set_index("session")[["median_px", "pck10"]]
        out = e2e[cols].set_index("session").join(oracle, rsuffix="_oracle_crop")
        print("\n== end to end (crops from the stage-1 eye detector) ==")
        print(out.round(3).to_string())
        out.round(3).to_csv(RESULTS / "table_end_to_end.csv")


if __name__ == "__main__":
    main()
