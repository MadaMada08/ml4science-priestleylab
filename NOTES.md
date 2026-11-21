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