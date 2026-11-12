import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from train_sign_model import assign_own_subsets


def make_own(takes_per_sign, signs=("man", "none")):
    rows = []
    for sign in signs:
        for take in range(takes_per_sign):
            for frame in range(4):
                rows.append({"video_id": f"mine_{sign}_{take}",
                             "sign": sign, "subset": "own"})
    return pd.DataFrame(rows)


def test_every_recording_gets_a_subset():
    out = assign_own_subsets(make_own(15))
    assert out["subset"].isin(["train", "val", "test"]).all()


def test_a_recording_is_never_split_across_subsets():
    out = assign_own_subsets(make_own(15))
    assert (out.groupby("video_id")["subset"].nunique() == 1).all()


def test_each_sign_has_train_val_and_test():
    out = assign_own_subsets(make_own(15))
    for sign, group in out.groupby("sign"):
        assert set(group["subset"]) == {"train", "val", "test"}


def test_few_recordings_all_go_to_training():
    out = assign_own_subsets(make_own(2))
    assert set(out["subset"]) == {"train"}


def test_split_is_repeatable():
    a = assign_own_subsets(make_own(15))
    b = assign_own_subsets(make_own(15))
    assert a["subset"].tolist() == b["subset"].tolist()