"""Two-stage (top-down) eye/pupil keypoint experiments that run on CPU.

Stage 1 finds the eye in a downscaled full frame; stage 2 predicts the 8 keypoints inside a crop
whose side is set by the eye width, so the eye has the same size for the network in every session.
Every experiment is leave-one-session-out; errors are in full-resolution pixels and in eye widths.

Models are small and trained from scratch (pretrained weights could not be downloaded), so absolute
numbers are not comparable with DLC's; the comparison between augmentation conditions is the point.

    python experiments/sebastien/eyecrop.py --epochs 40
    python experiments/sebastien/eyecrop.py --sessions 110535 171101 --conditions hflip+photo --stage1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data import BODYPARTS, load_labels  # noqa: E402

EYE = slice(0, 4)
CROP_SCALE = 2.0  # crop side, in eye widths
CROP_PX = 128
FRAME_DOWN = 4  # stage 1 sees the frame downscaled 4x (320 x 256)

CONDITIONS = {
    "none": dict(hflip=False, photometric=False, invert=False),
    "hflip": dict(hflip=True, photometric=False, invert=False),
    "hflip+photo": dict(hflip=True, photometric=True, invert=False),
    "hflip+photo+invert": dict(hflip=True, photometric=True, invert=True),
}

Data = dict[str, np.ndarray]
Sample = tuple[np.ndarray, np.ndarray]
Crop = tuple[np.ndarray, np.ndarray, np.ndarray]


def load_dataset(labeled_dir: Path, cache: Path) -> Data:
    """Load grayscale frames, (n, 8, 2) keypoints and session ids of all labeled frames (cached)."""
    if cache.exists():
        with np.load(cache) as d:
            return {k: d[k] for k in d.files}
    labels = load_labels(labeled_dir)
    labels = labels[labels.notna().any(axis=1)]
    kp = labels[[(bp, c) for bp in BODYPARTS for c in ("x", "y")]].to_numpy(np.float32)
    frames = [
        cv2.imread(str(labeled_dir / session / image), cv2.IMREAD_GRAYSCALE)
        for session, image in labels.index
    ]
    data = {
        "frames": np.stack(frames),
        "kp": kp.reshape(-1, 8, 2),
        "session": labels.index.get_level_values("session").to_numpy(str),
    }
    np.savez(cache, **data)
    return data


def eye_box(kp: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the eye center and width from the 4 eye keypoints (missing ones are ignored)."""
    eye = kp[EYE]
    center = np.nanmean(eye, 0)
    width = float(np.hypot(*(eye[0] - eye[1])))
    if not np.isfinite(width):
        width = 2 * float(np.nanmax(np.linalg.norm(eye - center, axis=1)))
    return center, width


def photometric(img: np.ndarray, rng: np.random.Generator, invert: bool) -> np.ndarray:
    """Apply random gamma, contrast, brightness, overexposure, blur, noise and optional inversion.

    Wider than src.photometric: crops are standardized afterwards, which undoes the extreme
    brightness shifts that would hide the eye in DLC's unstandardized input.
    """
    x = img.astype(np.float32) / 255.0
    x = x ** rng.uniform(0.4, 2.5)
    x = (x - x.mean()) * rng.uniform(0.5, 1.8) + x.mean() + rng.uniform(-0.25, 0.25)
    if rng.random() < 0.3:
        x = np.minimum(x * rng.uniform(1.5, 4.0), 1.0)
    if rng.random() < 0.3:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.5, 1.5))
    x = np.clip(
        x + rng.normal(0, rng.uniform(0, 0.05), x.shape).astype(np.float32), 0, 1
    )
    if invert and rng.random() < 0.5:
        x = 1.0 - x
    return (x * 255).astype(np.uint8)


def standardize(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-3)


def warp(
    frame: np.ndarray,
    kp: np.ndarray,
    center: np.ndarray,
    side: float,
    out: int,
    rng: np.random.Generator | None = None,
    cond: dict | None = None,
    jitter: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Crop:
    """Crop a `side`-px square around `center`, resized to `out` px, augmented when `rng` is given.

    Args:
        jitter: maximum shift (fraction of `side`), log-scale change and rotation (degrees).

    Returns:
        The crop, the keypoints in crop coordinates and the 2x3 frame-to-crop affine matrix.
    """
    offset, scale, angle, flip = np.zeros(2), 1.0, 0.0, False
    if rng is not None:
        shift, log_scale, rotation = jitter
        offset = rng.uniform(-shift, shift, 2) * side
        scale = math.exp(rng.uniform(-log_scale, log_scale))
        angle = rng.uniform(-rotation, rotation)
        flip = bool(cond["hflip"]) and rng.random() < 0.5
    zoom = out / (side * scale)
    M = cv2.getRotationMatrix2D(tuple(map(float, center + offset)), angle, zoom)
    M[:, 2] += out / 2 - (center + offset)
    interpolation = cv2.INTER_AREA if zoom < 1 else cv2.INTER_LINEAR
    img = cv2.warpAffine(
        frame, M, (out, out), flags=interpolation, borderMode=cv2.BORDER_REFLECT
    )
    pts = kp @ M[:, :2].T + M[:, 2]
    if flip:  # labels are anatomical, so no left/right keypoint swap
        img = img[:, ::-1].copy()
        pts[:, 0] = out - 1 - pts[:, 0]
    if rng is not None and cond["photometric"]:
        img = photometric(img, rng, cond["invert"])
    return img, pts, M


def to_frame(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Map crop coordinates back to frame coordinates."""
    Minv = cv2.invertAffineTransform(M)
    return pts @ Minv[:, :2].T + Minv[:, 2]


def conv_block(cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class KeypointNet(nn.Module):
    """Small encoder-decoder; a soft-argmax over a stride-4 heatmap gives sub-pixel coordinates."""

    def __init__(self, n_keypoints: int, width: int = 24) -> None:
        super().__init__()
        w = width
        self.d1 = conv_block(1, w, 2)
        self.d2 = conv_block(w, 2 * w, 2)
        self.d3 = conv_block(2 * w, 4 * w, 2)
        self.d4 = conv_block(4 * w, 8 * w, 2)
        self.u3 = conv_block(8 * w + 4 * w, 4 * w, 1)
        self.u2 = conv_block(4 * w + 2 * w, 2 * w, 1)
        self.head = nn.Conv2d(2 * w, n_keypoints, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (batch, keypoints, 2) coordinates in input pixels and the heatmap peak heights."""
        x2 = self.d2(self.d1(x))
        x3 = self.d3(x2)
        y = self.u3(torch.cat([F.interpolate(self.d4(x3), size=x3.shape[-2:]), x3], 1))
        y = self.u2(torch.cat([F.interpolate(y, size=x2.shape[-2:]), x2], 1))
        logits = self.head(y)
        b, k, h, w = logits.shape
        prob = logits.flatten(2).softmax(-1).view(b, k, h, w)
        ys = (torch.arange(h, dtype=torch.float32) + 0.5) * 4 - 0.5
        xs = (torch.arange(w, dtype=torch.float32) + 0.5) * 4 - 0.5
        coords = torch.stack(
            [(prob.sum(2) * xs).sum(-1), (prob.sum(3) * ys).sum(-1)], -1
        )
        return coords, prob.flatten(2).max(-1).values


def make_batch(samples: list[Sample]) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.from_numpy(np.stack([standardize(img) for img, _ in samples]))[
        :, None
    ]
    points = torch.from_numpy(np.stack([pts for _, pts in samples]).astype(np.float32))
    return images, points


def train(
    model: KeypointNet,
    sample: Callable[[int], Sample],
    n_train: int,
    epochs: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
    scale: float,
) -> KeypointNet:
    """Train with AdamW and a one-cycle schedule; `scale` normalizes coordinates in the loss."""
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = epochs * math.ceil(n_train / args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, args.lr, total_steps=steps, pct_start=0.15
    )
    model.train()
    for _ in range(epochs):
        order = rng.permutation(n_train)
        for start in range(0, n_train, args.bs):
            images, points = make_batch(
                [sample(j) for j in order[start : start + args.bs]]
            )
            pred, _ = model(images)
            labeled = torch.isfinite(points)
            loss = F.smooth_l1_loss(
                pred[labeled] / scale, points[labeled] / scale, beta=0.02
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return model.eval()


@torch.no_grad()
def predict(model: KeypointNet, samples: list[Sample]) -> np.ndarray:
    batches = [samples[i : i + 16] for i in range(0, len(samples), 16)]
    return np.concatenate([model(make_batch(batch)[0])[0].numpy() for batch in batches])


def stage1_sample(
    data: Data, i: int, rng: np.random.Generator | None = None, cond: dict | None = None
) -> Crop:
    """Downscaled full frame with the 4 eye corners as targets."""
    frame, kp = data["frames"][i], data["kp"][i][EYE]
    h, w = frame.shape
    # a w x w square from the top edge; the rows below the frame are cut off afterwards
    center = np.array([w / 2, w / 2])
    img, pts, M = warp(
        frame, kp, center, w, w // FRAME_DOWN, rng, cond, (0.08, 0.35, 10)
    )
    return img[: h // FRAME_DOWN], pts, M


def stage2_sample(
    data: Data,
    i: int,
    rng: np.random.Generator | None = None,
    cond: dict | None = None,
    center: np.ndarray | None = None,
    width: float | None = None,
) -> Crop:
    """Eye crop of frame `i`, around the labeled eye unless `center` and `width` are given."""
    frame, kp = data["frames"][i], data["kp"][i]
    if center is None:
        center, width = eye_box(kp)
    # training jitter stands in for an imperfect stage 1
    return warp(
        frame, kp, center, CROP_SCALE * width, CROP_PX, rng, cond, (0.12, 0.2, 15)
    )


def scores(pred: np.ndarray, kp: np.ndarray, widths: np.ndarray) -> dict[str, float]:
    """Summarize frame-coordinate predictions against the labels."""
    err = np.linalg.norm(pred - kp, axis=-1)
    norm = err / widths[:, None]
    labeled = np.isfinite(err)
    sides = np.isfinite(kp[:, 0, 0]) & np.isfinite(kp[:, 1, 0])
    true_left = kp[sides, 0, 0] < kp[sides, 1, 0]
    pred_left = pred[sides, 0, 0] < pred[sides, 1, 0]
    return {
        "median_px": float(np.median(err[labeled])),
        "mean_px": float(np.mean(err[labeled])),
        "mean_norm": float(np.mean(norm[labeled])),
        "pck10": float(np.mean(norm[labeled] < 0.10)),
        "pck05": float(np.mean(norm[labeled] < 0.05)),
        "eye_median_px": float(np.nanmedian(err[:, :4])),
        "pupil_median_px": float(np.nanmedian(err[:, 4:])),
        "orientation": float(np.mean(true_left == pred_left)),
    }


def run_fold(
    data: Data,
    test_session: str,
    cond_name: str,
    args: argparse.Namespace,
    stage1: KeypointNet | None = None,
) -> dict:
    """Train stage 2 without `test_session`; score it on true eye boxes, and on stage-1 boxes if given."""
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cond = CONDITIONS[cond_name]
    train_idx = np.flatnonzero(data["session"] != test_session)
    test_idx = np.flatnonzero(data["session"] == test_session)
    kp = data["kp"][test_idx]
    widths = np.array([eye_box(k)[1] for k in kp])

    def sample(j: int) -> Sample:
        img, pts, _ = stage2_sample(data, train_idx[j], rng, cond)
        return img, pts

    start = time.time()
    model = train(
        KeypointNet(8, args.width),
        sample,
        len(train_idx),
        args.epochs,
        args,
        rng,
        CROP_PX,
    )
    result = {
        "test_session": test_session,
        "condition": cond_name,
        "train_s": time.time() - start,
    }

    crops = [stage2_sample(data, i) for i in test_idx]
    pred = predict(model, [(img, pts) for img, pts, _ in crops])
    full = np.stack([to_frame(p, M) for p, (_, _, M) in zip(pred, crops)])
    result["oracle"] = scores(full, kp, widths)

    if stage1 is not None:
        frames = [stage1_sample(data, i) for i in test_idx]
        corners = predict(stage1, [(img, pts) for img, pts, _ in frames])
        crops, center_err = [], []
        for (_, _, M1), p, i in zip(frames, corners, test_idx):
            eye = np.concatenate([to_frame(p, M1), np.full((4, 2), np.nan)])
            center, width = eye_box(eye)
            center_err.append(np.linalg.norm(center - eye_box(data["kp"][i])[0]))
            crops.append(stage2_sample(data, i, center=center, width=width))
        pred = predict(model, [(img, pts) for img, pts, _ in crops])
        full = np.stack([to_frame(p, M) for p, (_, _, M) in zip(pred, crops)])
        result["end_to_end"] = scores(full, kp, widths) | {
            "stage1_center_err_median_px": float(np.median(center_err)),
            "stage1_center_err_max_px": float(np.max(center_err)),
        }
    return result


def train_stage1(
    data: Data, test_session: str, args: argparse.Namespace
) -> KeypointNet:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed + 1)
    cond = CONDITIONS[args.stage1_condition]
    train_idx = np.flatnonzero(data["session"] != test_session)

    def sample(j: int) -> Sample:
        img, pts, _ = stage1_sample(data, train_idx[j], rng, cond)
        return img, pts

    model = KeypointNet(4, args.width)
    return train(model, sample, len(train_idx), args.stage1_epochs, args, rng, 320)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/labeled-data"))
    parser.add_argument("--out", type=Path, default=Path("results/sebastien"))
    parser.add_argument("--sessions", nargs="*")
    parser.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument(
        "--stage1", action="store_true", help="also score crops from a trained stage 1"
    )
    parser.add_argument("--stage1-epochs", type=int, default=40)
    parser.add_argument("--stage1-condition", default="hflip+photo")
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    data = load_dataset(args.data, args.data.parent / "cache_sebastien.npz")
    for session in args.sessions or sorted(set(data["session"])):
        stage1 = train_stage1(data, session, args) if args.stage1 else None
        for cond in args.conditions:
            tag = "__e2e" if args.stage1 else ""
            path = args.out / f"{session}__{cond}{tag}__seed{args.seed}.json"
            if path.exists():
                continue
            result = run_fold(data, session, cond, args, stage1)
            path.write_text(json.dumps(result, indent=2))
            o = result["oracle"]
            line = (
                f"[{session}] {cond:20s} median {o['median_px']:.1f}px "
                f"pck10 {o['pck10']:.2f} orientation {o['orientation']:.2f}"
            )
            if "end_to_end" in result:
                e = result["end_to_end"]
                line += f" | stage-1 crops: median {e['median_px']:.1f}px pck10 {e['pck10']:.2f}"
            print(f"{line} ({result['train_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
