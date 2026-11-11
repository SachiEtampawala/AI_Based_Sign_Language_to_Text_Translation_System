import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from features import hands_to_vector, normalize_landmarks, FEATURE_SIZE


def make_hand(seed=0, shift=(0.0, 0.0, 0.0), scale=1.0):
    rng = np.random.default_rng(seed)
    points = rng.random((21, 3)).astype(np.float32) * scale
    points += np.array(shift, dtype=np.float32)
    return points


def to_landmarks(points):
    return [SimpleNamespace(x=p[0], y=p[1], z=p[2]) for p in points]


def test_one_hand_ignores_position_and_size():
    a = make_hand(1)
    b = a * 2.5 + np.array([0.3, 0.2, 0.1], dtype=np.float32)
    na = normalize_landmarks(a.flatten())
    nb = normalize_landmarks(b.flatten())
    assert np.allclose(na, nb, atol=1e-4)


def test_two_hands_output_width():
    raw = np.concatenate([make_hand(1).flatten(), make_hand(2).flatten()])
    assert normalize_landmarks(raw).shape == (FEATURE_SIZE,)
    assert normalize_landmarks(np.stack([raw, raw])).shape == (2, FEATURE_SIZE)


def test_missing_second_hand_stays_zero():
    raw = np.concatenate([make_hand(1).flatten(), np.zeros(63)])
    out = normalize_landmarks(raw)
    assert np.allclose(out[63:126], 0)
    assert np.allclose(out[126:], 0)       # no offset without two hands


def test_offset_tells_where_hands_are():
    first = make_hand(1)
    near = np.concatenate([first.flatten(), make_hand(2, shift=(0.1, 0, 0)).flatten()])
    far = np.concatenate([first.flatten(), make_hand(2, shift=(0.9, 0, 0)).flatten()])
    assert normalize_landmarks(near)[126] != normalize_landmarks(far)[126]


def test_hands_sorted_left_to_right():
    left = to_landmarks(make_hand(1, shift=(0.0, 0, 0)))
    right = to_landmarks(make_hand(2, shift=(5.0, 0, 0)))
    a = hands_to_vector([right, left])
    b = hands_to_vector([left, right])
    assert np.allclose(a, b)
    assert a[0] == left[0].x


def test_single_hand_goes_in_first_slot():
    hand = to_landmarks(make_hand(3, shift=(5.0, 0, 0)))
    vector = hands_to_vector([hand])
    assert vector.shape == (126,)
    assert np.allclose(vector[63:], 0)
    assert vector[0] == hand[0].x


def test_wrong_width_raises():
    try:
        normalize_landmarks(np.zeros(10))
        assert False, "should have raised"
    except ValueError:
        pass