# Robust Cross-Session Eye & Pupil Tracking for Head-Fixed Mice

CS-433 Machine Learning — Project 2 (ML4Science), in collaboration with the **Priestley Lab** (EPFL).

---

## Context

The Priestley Lab studies the hippocampus using a head-fixed behavioral setup: a mouse runs on a wheel while facing screens that present a virtual-reality environment, with a licking spout that dispenses sweet-water rewards. A two-photon (2p) microscope records hippocampal activity from above, and side cameras film every session, capturing the head and front of the mouse's body.

Alongside the neural data, the lab uses behavioral signals : running speed, licking rate, and **eye and pupil position/movement**. The eye and pupil are tracked with [DeepLabCut](https://www.mackenziemathislab.org/deeplabcut) (DLC), a markerless pose-estimation library, trained to detect **8 keypoints**:

- Eye: ventral, dorsal, posterior, anterior
- Pupil: ventral, dorsal, posterior, anterior

**The problem.** The trained model generalizes poorly to new recordings. As soon as the conditions shift — a different mouse, a slightly different camera angle (including a mirrored view of the other eye), a new session, or different lighting — predictions degrade badly: keypoints land far from the real positions, or are not detected at all (frames returned with no keypoints). The lab's current workaround is to label a handful of new frames and retrain the model for each new session, which does not scale.

**Our goal.** Produce a tracking pipeline that generalizes across mice, sessions, angles, and lighting conditions : trained once, usable afterward with no per-session retraining.

## Method

*Work in progress.* This section will be filled in as the project progresses and finalized near the end, documenting the full pipeline: preprocessing, feature/image processing, training setup, hyper-parameters, and evaluation.

## Repository structure

```
.
├── README.md
├── .github/
│   └── pull_request_template.md
├── data/              # raw & processed data — gitignored, never pushed
├── src/               # reusable code, importable as a package
├── results/           # figures and outputs
└── requirements.txt   # pinned dependencies
```

## Setup

<!-- Provisional we will finalize it once the environment is fixed, with exact version numbers. -->

DeepLabCut and all other libraries are installed as **pinned dependencies in an isolated environment**; they are never copied into this repository.

```bash
conda create -n ml4science python=3.10
conda activate ml4science
pip install -r requirements.txt
```

## Data

The data is provided by the Priestley Lab and is **not** included in this repository (`data/` is gitignored). Instructions on how to obtain and place the data will be added here later after discussion with the lab and the CS-433 team.

## Code style

We keep code readable through clear naming and type hints rather than heavy commenting. Inline comments are reserved for the non-obvious *why* — never to restate what the code already says.

Functions and classes are documented with concise docstrings:
- a one-line summary of what it does, in the imperative;
- an `Args:` / `Returns:` block only when the signature isn't self-explanatory.

Trivial helpers may rely on a descriptive name and type hints alone.

```python
def normalize_contrast(frame: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply CLAHE contrast normalization to a grayscale frame."""
    ...
```

## Team process

Our team follows a consistent GitHub workflow so that the project history stays readable, every change is traceable to a reason, and `main` always remains in a working state. The process is as follows.

**1. Every change starts with an issue.**
Before any code is written, the change is described in a GitHub issue with a clear, specific title, a description of the motivation and intended outcome, and at least one label (e.g. `enhancement`, `preprocessing`, `model`, `documentation`). The title should read like an action — for example *"Add CLAHE contrast normalization to preprocessing"* — because it becomes the branch name. Issues are the backbone of the project: they serve both as our to-do list and as a record of why each change exists.

**2. Branches are created from their issue.**
We do not create branches by hand with arbitrary names. We open the issue and use GitHub's *"Create a branch"* link, which produces a branch already linked to the issue and named with the `<issue-number>-issue-title` convention (e.g. `18-add-clahe-contrast-normalization`). Because the branch is tied to its issue, any line of code can be traced back to the reason it exists — which is also why issue titles must be meaningful.

**Commit messages.**
Commits are written in the imperative mood and capture one logical change: *"Add CLAHE contrast normalization"*. The subject line is short (~50 characters), capitalized, with no trailing period. When the reason for a change isn't obvious from the subject, a short body explains the *why*. This keeps `git log` readable as a record of decisions rather than noise.

**3. Bug fixes may skip the issue, but not the naming.**
When a bug reaches the repository and needs a direct correction, a fixing branch can be created without opening an issue first. It must still follow the `fix/<short-description>` convention (e.g. `fix/flipped-eye-keypoint-order`). Everything below still applies.

**4. All work enters `main` through a Pull Request.**
When a branch is ready, we open a Pull Request that links its issue with a closing keyword (`Closes #18`), so that merging the PR closes the issue automatically. Every PR follows the repository's PR template and requires at least one reviewer's approval before it can be merged.

**5. Reviews are thorough and constructive.**
A review is never a rubber stamp. The reviewer checks out the branch and tests the claims made in the PR description, thanks the author and highlights what was done well, and requests changes where needed — on code quality and/or documentation. Approval is given only once the PR does what it claims and meets our standards.

**6. Merge and clean up.**
Once approved, the PR is merged into `main`, the linked issue closes automatically, and the branch is deleted (the repository is set to delete merged branches automatically). No stale branches are left behind.

**7. Personal branches.**
Each of the three team member has a `personal/<member_name>` branch in which they can test any code and method they want. These branches are never merged and can serve as exploratory process for future issues.

## Authors & supervision

- **Team:** Adam Cohen, Edouard Fousson, Sebastien Nyiri
- **Lab supervisor:** Daniel Molinuevo Gomez, Priestley Lab
- **Course:** CS-433 Machine Learning, EPFL — Fall 2026
