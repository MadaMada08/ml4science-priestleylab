# notes - edouard

## hist_eq on 110535:

shuffle 2 = same split as baseline, hist_eq true (train + inference), 100 epochs, eval snapshot-100

|               | baseline | hist_eq |
|---------------|----------|---------|
| mean err      | 238 px   | 93 px   |
| detection     | 0%       | 7%      |

- lighting is def part of the problem
- pupil top/bottom now ~5px
- anterior/posterior are SWAPPED: model puts anterior on the right (5/7 train sessions are like that), 110535 has it on the left
- eye-ant / eye-post still ~150px off, idk why yet

## hist_eq + hflip on 110535

shuffle 3 = same as shuffle 2 + hflip 0.5 in train only (no keypoint swap, labels are anatomical)

|               | baseline | hist_eq | hist_eq + hflip |
|---------------|----------|---------|-----------------|
| mean err      | 238 px   | 86 px   | 21 px           |
| median err    | 196 px   | 48 px   | 13 px           |
| detection     | 0%       | 8%      | 36%             |

- swap FIXED: anterior on the left in 97% of frames (truth = 100%)
- detected points are all good now (~11px), no more confident wrong points
- train err barely moves (3.2 px)
- still not great: 36% detection, 0/39 frames with all 4 pupil pts
- problem is now confidence not position: 32% of points are well placed but under 0.6
- eye-posterior is the worst one now (27px median)
- only tested on 110535 (and 3 runs judged on it already) -> need 135059 and 171101

next: run baseline vs hist_eq + hflip with 135059 and 171101 held out

## baseline vs hist_eq + hflip on 135059 and 171101

shuffles 4-7, 100 epochs, baseline = dlc defaults

|               | baseline | hist_eq + hflip |
|---------------|----------|-----------------|
| 135059        | 62 px    | 60 px           |
| 171101        | 15 px    | 10 px           |

- 171101: hist_eq + hflip helps (3px median, 71% detection, no confident wrong pts)
- 135059: model puts anterior/posterior the other way in 100% of frames, and it's confident about it

## 135059 labels are wrong

- checked 1 frame per session by eye (nose side vs labeled anterior): only 135059 is off. nose is on the left, labels say anterior = right (all 40 frames, eye + pupil, dorsal/ventral ok)
- prob comes from the right/left -> anterior/posterior renaming. to confirm with dani
- re-scored sh4/sh5 with 135059 swapped back (no retraining): 61 -> 8 px and 60 -> 7 px, orientation 0 -> 1
- so the model was right on 135059, it was a label problem
- fixed the labels (backup in data/backup_135059_labels/)
- sh1, 2, 3, 6, 7 trained with the bad 135059 labels -> redoing everything as shuffles 11-17

## shuffles 11-17 - same experiments with fixed 135059 labels

all 100 epochs, baseline = dlc defaults

| held out | baseline | hist_eq | hist_eq + hflip |
|----------|----------|---------|-----------------|
| 110535   | 220 px   | 68 px   | 20 px           |
| 135059   | 8 px     | -       | 7 px            |
| 171101   | 32 px    | -       | 3.5 px          |

- hist_eq + hflip is the best on all 3 sessions, and the only one with no orientation problem
- detection with hist_eq + hflip: 45% (110535), 98% (135059), 85% (171101)
- without hflip the model still swaps ant/post on 110535 and 171101 even with fixed labels -> the flip is really needed
- 135059 was never the problem: 7-8 px once the labels are right
- sh14 = sh4 exactly (neither trained on 135059) -> good sanity check
- 110535 still the hard one: 20 px and only 45% detected (lighting)
- sh17: last epoch = best epoch so dlc saved it as snapshot-best-100, had to evaluate it by hand

next: work on 110535 (lighting), then run all 8 held-out sessions