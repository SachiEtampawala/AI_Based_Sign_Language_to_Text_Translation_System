"""Build a two-hand landmark dataset from the WLASL videos.

Output : outputs/landmark_dataset_2hands.csv
Each row is one video frame: video_id, sign, subset, landmark_0 ... landmark_125
(hand 1 = 63 numbers, hand 2 = 63 numbers, zeros if the hand is missing).

The old dataset (landmark_dataset.csv) is NOT changed.
"""
import csv
import json
import random
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd

from features import hands_to_vector, RAW_SIZE

# Paths
DATASET_PATH = Path.home() / "Downloads" / "archive"
JSON_FILE = DATASET_PATH / "nslt_100.json"
CLASS_FILE = DATASET_PATH / "wlasl_class_list.txt"
VIDEOS_FOLDER = DATASET_PATH / "videos"

PROJECT_PATH = Path(__file__).resolve().parent.parent
OUTPUT_FOLDER = PROJECT_PATH / "outputs"
OLD_DATASET_FILE = OUTPUT_FOLDER / "landmark_dataset.csv"
NEW_DATASET_FILE = OUTPUT_FOLDER / "landmark_dataset_2hands.csv"
MODEL_FILE = PROJECT_PATH / "models" / "hand_landmarker.task"

# Settings
USE_ALL_SIGNS = False        # False = same signs as the current dataset
MAX_VIDEOS_PER_SIGN = 30
MIN_VIDEOS_PER_SIGN = 5      # signs with fewer videos are skipped
FRAME_STEP = 3               # use every 3rd frame
RANDOM_SEED = 42


def load_labels():
    labels = {}
    with open(CLASS_FILE, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                labels[int(parts[0])] = parts[1].strip()
    return labels


def split_videos(video_ids):
    """Split whole videos so frames from one video never leak across subsets."""
    rng = random.Random(RANDOM_SEED)
    video_ids = list(video_ids)
    rng.shuffle(video_ids)
    count = len(video_ids)

    train_count = max(3, int(count * 0.70))
    train_count = min(train_count, count - 2)
    val_count = max(1, int(count * 0.15))
    if train_count + val_count >= count:
        val_count = 1

    train = video_ids[:train_count]
    val = video_ids[train_count:train_count + val_count]
    test = video_ids[train_count + val_count:]
    return train, val, test


def extract_rows(video_id, sign, subset, landmarker):
    cap = cv2.VideoCapture(str(VIDEOS_FOLDER / f"{video_id}.mp4"))
    if not cap.isOpened():
        return []

    rows = []
    frame_index = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        use_frame = frame_index % FRAME_STEP == 0
        frame_index += 1
        if not use_frame:
            continue

        # Videos are NOT mirrored here. The webcam app must match this.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if not result.hand_landmarks:
            continue

        vector = hands_to_vector(result.hand_landmarks)
        rows.append([video_id, sign, subset, *vector.tolist()])

    cap.release()
    return rows


def main():
    print("Building the two-hand landmark dataset...\n")

    required = [JSON_FILE, CLASS_FILE, MODEL_FILE]
    if not USE_ALL_SIGNS:
        required.append(OLD_DATASET_FILE)

    for path in required:
        if not path.exists():
            print(f"ERROR : Required file not found.\n{path}")
            return

    if not VIDEOS_FOLDER.exists():
        print(f"ERROR : Videos folder not found.\n{VIDEOS_FOLDER}")
        return

    labels = load_labels()

    with open(JSON_FILE, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    # Which signs to use
    if USE_ALL_SIGNS:
        selected_signs = sorted({
            labels[info["action"][0]]
            for info in dataset.values()
            if info.get("action") and info["action"][0] in labels
        })
    else:
        old = pd.read_csv(OLD_DATASET_FILE)
        selected_signs = sorted(old["sign"].dropna().unique().tolist())

    print(f"Signs ({len(selected_signs)}): {', '.join(selected_signs)}")

    # Find the videos that exist on this computer
    candidates = {sign: [] for sign in selected_signs}

    for video_id, info in dataset.items():
        if not (VIDEOS_FOLDER / f"{video_id}.mp4").exists():
            continue
        if not info.get("action"):
            continue
        sign = labels.get(info["action"][0])
        if sign in candidates:
            candidates[sign].append(video_id)

    rng = random.Random(RANDOM_SEED)
    split_records = []
    skipped_signs = []

    for sign in selected_signs:
        ids = sorted(set(candidates[sign]))
        rng.shuffle(ids)
        ids = ids[:MAX_VIDEOS_PER_SIGN]

        if len(ids) < MIN_VIDEOS_PER_SIGN:
            skipped_signs.append((sign, len(ids)))
            continue

        train_ids, val_ids, test_ids = split_videos(ids)
        split_records += [(v, sign, "train") for v in train_ids]
        split_records += [(v, sign, "val") for v in val_ids]
        split_records += [(v, sign, "test") for v in test_ids]

    if skipped_signs:
        print("\nSkipped (not enough videos on this computer):")
        for sign, count in skipped_signs:
            print(f"  {sign}: {count} videos")

    if not split_records:
        print("\nSTOPPED : No usable videos found.")
        return

    print(f"\nVideos to process : {len(split_records)}")

    # MediaPipe: up to 2 hands
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_FILE)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    header = ["video_id", "sign", "subset"] + [
        f"landmark_{i}" for i in range(RAW_SIZE)
    ]

    total_rows = 0
    two_hand_rows = 0

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        with open(NEW_DATASET_FILE, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)

            for index, (video_id, sign, subset) in enumerate(split_records, 1):
                rows = extract_rows(video_id, sign, subset, landmarker)
                writer.writerows(rows)
                csv_file.flush()

                total_rows += len(rows)
                two_hand_rows += sum(1 for r in rows if any(r[3 + 63:]))

                print(
                    f"[{index}/{len(split_records)}] {video_id} -> "
                    f"{sign} ({subset}), frames kept: {len(rows)}"
                )

    print("\n----------")
    print("Dataset creation completed.")
    print(f"Rows saved           : {total_rows}")
    print(f"Rows with two hands  : {two_hand_rows}")
    print(f"Saved to             : {NEW_DATASET_FILE}")

if __name__ == "__main__":
    main()