# notes - sebastien

Branched from `personal/edouard`. What is built on his work vs new is listed at the bottom.

## 1. why do held-out sessions fail?

Eye crop per session: `results/sebastien/session_factors.png` (`experiments/sebastien/factor_figure.py`).
Sessions differ along 3 conditions, now in `src/data.session_conditions`:

| session | faces | eye width | pupil |
|---|---|---|---|
| 094124, 131707, 135059 | left | ~155 px | dark |
| **110535** | left | ~152 px | **bright** |
| 115300, 172731, 214411 | right | ~97 px | bright |
| **171101** | right | ~99 px | **dark** |

- two camera setups (eye left of frame and big / right and small) and two illuminations (bright-pupil
  IR makes the pupil white)
- **110535 and 171101 are the only sessions whose combination is missing from the other 7**
  (`src/data.unseen_condition_sessions` finds them from the labels)
- in training "bright pupil" always comes with "faces right, small eye": a shortcut the model can learn
- these are exactly the 2 sessions where the DLC baseline breaks (220 px, 32 px); 135059 has 2 look-alike
  sessions and is fine (8 px)
- explains why hflip works so well: it creates the missing combinations
- 150130 (unlabeled) is a white mouse on a light background: yet another combination

## 2. CPU study (not DLC): augmentation conditions, leave-one-session-out on all 8 sessions

`experiments/sebastien/eyecrop.py`. Small model trained from scratch on eye crops (crop side = 2 eye widths,
so eye size is the same for every session). No pretrained weights reachable from the sandbox -> absolute numbers
are not comparable to DLC; the comparison between conditions is what matters.

Median error, full-res px, seed 0 (`results/sebastien/table_*.csv`, `conditions.png`):

| held out | none | hflip | hflip+photo | hflip+photo+invert |
|---|---|---|---|---|
| **110535** | 30.1 | 22.0 | **15.0** | 14.4 |
| **171101** | 19.8 | 7.8 | **5.9** | 8.9 |
| 094124 | 8.5 | 7.8 | 7.8 | 8.7 |
| 115300 | 5.9 | 5.7 | 8.2 | 5.7 |
| 131707 | 9.4 | 7.8 | 8.9 | 7.3 |
| 135059 | 8.3 | 10.4 | 10.3 | 11.5 |
| 172731 | 4.6 | 4.6 | 5.5 | 5.0 |
| 214411 | 3.9 | 5.1 | 5.3 | 5.5 |
| mean, unseen combos | 25.0 | 14.9 | **10.5** | 11.6 |
| PCK@0.1, unseen combos | 13% | 46% | **68%** | 55% |
| ant/post orientation, unseen | 14% | 100% | 100% | 100% |

photo = random gamma, contrast, brightness, overexposure, blur, noise; invert = also flip intensities.

- without flip this model also swaps anterior/posterior on both unseen sessions -> the shortcut is not a DLC quirk
- flip fixes orientation; lighting aug is what helps 110535 further (22 -> 15 px, PCK 26 -> 52%)
- on the 6 seen-combination sessions all conditions are within ~±2 px; aug costs a bit on 135059 (8.3 -> 10.3)
- intensity inversion: no gain -> dropped

seed 1 on the unseen sessions: 110535 41.9 / 20.5 / 14.8 px, 171101 16.0 / 9.2 / 8.2 px (none / hflip / +photo).
Same ordering; seed noise ~1-3 px, so the none -> hflip -> +photo gains hold, photo vs photo+invert doesn't.

end to end (crop box from a trained eye detector instead of the labels, hflip+photo):
110535 14.9 px (vs 15.0 with the true box), 171101 9.8 px (vs 5.9). Detector center is only 7 px off on 171101,
so probably the box *size* is off; not checked. Two-stage is not a win yet.

The JSON results were produced before eyecrop.py was switched to `src.data.load_labels`; re-running
110535 and 171101 (hflip+photo, seed 0) with the current code gives identical numbers (15.007 px and 5.923 px).

## 3. DLC follow-up, ready to run (not run: needs a GPU)

`experiments_sebastien.ipynb`: Edouard's hist_eq + hflip with vs without lighting-altered copies of the training
frames, held out 110535 / 171101 / 135059 (control). Copy runs use 1/3 of the epochs and LR milestones (3x the
frames) so every run gets the same gradient steps and schedule. Inference without hist_eq (see review).

- DLC has no gamma/contrast/exposure augmentation, so `src/photometric.py` writes copies as extra labeled-data
  folders, in a separate project (`dlc/pupil-photometric/`, see why in the review)
- the copies use milder ranges than the CPU study: on full frames the CPU ranges made some frames all black or all
  white (eye invisible); the CPU study standardized every crop, which undid that. See
  `results/sebastien/photometric_examples.png`. So the DLC run tests a related, not identical, augmentation.
- tested in the sandbox with DLC 3.0.2 (no training): splits, image paths, DLC's own data loader, notebook cells,
  scoring with `src/metrics.py` on fake predictions

## built on Edouard's work vs new

reused: his leave-one-session-out protocol, held-out sessions, 135059 label fix, `src/data.py`, `src/metrics.py`,
and his notebook helpers (moved into `src/shuffles.py`, his docstrings kept).

changed in his code:
- `src/data.py`: fail loudly on label files without anatomical names (they became all-NaN silently);
  `session_conditions`, `unseen_condition_sessions`
- `src/metrics.py`: `find_predictions` zero-pads the epoch like DLC does (`snapshot_033`)
- `src/shuffles.py` (his helpers): split from a fresh merge, copies of the held-out session never trained on,
  `snapshot-best-<epoch>` handled, `lr_milestones` option

new: `src/canonicalize.py` (lab export -> layout his code expects, incl. 135059 and 110535 fixes),
`src/photometric.py`, `experiments_sebastien.ipynb`, the CPU study in `experiments/sebastien/`.
