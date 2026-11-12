import numpy as np

NUM_LANDMARKS = 21                       # points per hand
HAND_SIZE = NUM_LANDMARKS * 3            # x, y, z for each point = 63
MAX_HANDS = 2
RAW_SIZE = HAND_SIZE * MAX_HANDS         # 126 raw numbers for two hands
FEATURE_SIZE = RAW_SIZE + 3              # 126 normalized + 3 hand-to-hand offset
NO_SIGN_LABEL = "none"                   # class for resting hand / no sign


def hands_to_vector(hands):
    """Turn detected hands into one raw vector of 126 numbers.

    `hands` is a list of hands from MediaPipe. Each hand is a list of
    21 landmarks that have .x, .y and .z.

    Hands are sorted left-to-right by the wrist position in the image,
    so the order does not depend on MediaPipe's left/right labels.
    A missing hand is filled with zeros. One hand always goes in slot 0.
    """
    ordered = sorted(hands, key=lambda hand: hand[0].x)[:MAX_HANDS]

    vector = np.zeros(RAW_SIZE, dtype=np.float32)

    for slot, hand in enumerate(ordered):
        values = []
        for landmark in hand:
            values.extend([landmark.x, landmark.y, landmark.z])
        vector[slot * HAND_SIZE:(slot + 1) * HAND_SIZE] = values

    return vector


def _normalize_one_hand(points):
    """points: (N, 21, 3). Move the wrist to the origin and scale the hand
    so its size is 1. Returns (N, 63) features and the (N,) hand size."""
    points = points - points[:, 0:1, :]
    size = np.linalg.norm(points[:, 9, :2], axis=1)   # wrist -> middle finger base
    points = points / (size[:, None, None] + 1e-6)
    return points.reshape(len(points), -1), size


def normalize_landmarks(flat):
    """Make landmarks independent of hand position and distance to camera.

    Input : (N, 63) or (63,)   -> one hand,  output (N, 63)
            (N, 126) or (126,) -> two hands, output (N, 129)
    For two hands, the last 3 numbers are the wrist-to-wrist offset
    (so the model can tell where the hands are relative to each other).
    A hand that is all zeros (not detected) stays all zeros.
    """
    array = np.asarray(flat, dtype=np.float32)
    single = array.ndim == 1
    array = array.reshape(1, -1) if single else array

    if array.shape[1] == HAND_SIZE:
        points = array.reshape(-1, NUM_LANDMARKS, 3)
        output, _ = _normalize_one_hand(points)

    elif array.shape[1] == RAW_SIZE:
        first = array[:, :HAND_SIZE]
        second = array[:, HAND_SIZE:]

        first_out, first_size = _normalize_one_hand(
            first.reshape(-1, NUM_LANDMARKS, 3)
        )
        second_out, second_size = _normalize_one_hand(
            second.reshape(-1, NUM_LANDMARKS, 3)
        )

        both_present = (np.abs(first).sum(axis=1) > 0) & \
                       (np.abs(second).sum(axis=1) > 0)

        reference_size = np.maximum(first_size, second_size)[:, None]
        offset = (second[:, 0:3] - first[:, 0:3]) / (reference_size + 1e-6)
        offset = np.where(both_present[:, None], offset, 0.0)

        output = np.concatenate([first_out, second_out, offset], axis=1)

    else:
        raise ValueError(
            f"Expected 63 or 126 landmark values, got {array.shape[1]}"
        )

    output = output.astype(np.float32)
    return output[0] if single else output