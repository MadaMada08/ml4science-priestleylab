"""Train/test split scenarios over the merged DLC frame table.

Each scenario returns (train_indices, test_indices) into the merged
CollectedData table. Shuffle numbers are fixed so the same scenario is
comparable across projects: 1=hard, 2=easy, 3=mixed.
"""

import numpy as np

HARD_SESSION = "110535"   # glare / contaminated — generalization worst case
EASY_SESSION = "094124"   # clean, same orientation as hard — best case


def _by_session(df, test_sessions):
    test  = [i for i, idx in enumerate(df.index) if idx[1] in test_sessions]
    train = [i for i, idx in enumerate(df.index) if idx[1] not in test_sessions]
    return train, test


def hard(df):
    """Leave-one-session-out on the hard session."""
    return _by_session(df, {HARD_SESSION})


def easy(df):
    """Leave-one-session-out on the easy session."""
    return _by_session(df, {EASY_SESSION})


def mixed(df, frac=0.5, seed=42):
    """Half the frames of every session in train, the other half in test (no unseen session)."""
    rng = np.random.default_rng(seed)
    by_session = {}
    for i, idx in enumerate(df.index):
        by_session.setdefault(idx[1], []).append(i)
    train, test = [], []
    for idxs in by_session.values():
        idxs = list(idxs); rng.shuffle(idxs)
        k = round(len(idxs) * frac)
        train += idxs[:k]; test += idxs[k:]
    return sorted(train), sorted(test)


SCENARIOS = {1: ("hard", hard), 2: ("easy", easy), 3: ("mixed", mixed)}
