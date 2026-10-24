import json
import random
import shutil
import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path

# Dataset and project paths
DATASET_PATH = Path.home() / "Downloads" / "archive"
JSON_FILE = DATASET_PATH / "nslt_100.json"
CLASS_FILE = DATASET_PATH / "wlasl_class_list.txt"
VIDEOS_FOLDER = DATASET_PATH / "videos"

PROJECT_PATH = Path(__file__).resolve().parent.parent
OUTPUT_FOLDER = PROJECT_PATH / "outputs"
DATASET_FILE = OUTPUT_FOLDER / "landmark_dataset.csv"
TEMP_FILE = OUTPUT_FOLDER / "landmark_dataset_new.csv"
BACKUP_FILE = OUTPUT_FOLDER / "landmark_dataset_backup.csv"
MODEL_FILE = PROJECT_PATH / "models" / "hand_landmarker.task"

# Keep the existing 10-sign vocabulary and use up to this many videos per sign.
MAX_VIDEOS_PER_SIGN = 30
FRAME_STEP = 3
RANDOM_SEED = 42
FEATURE_COLUMNS = [f"landmark_{i}" for i in range(63)]


def load_labels():
    labels = {}
    with open(CLASS_FILE, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                labels[int(parts[0])] = parts[1].strip()
    return labels


def split_videos(video_items):
    """Split whole videos so frames from one video never leak across subsets."""
    rng = random.Random(RANDOM_SEED)
    video_items = list(video_items)
    rng.shuffle(video_items)
    count = len(video_items)

    # At least 3 train videos and one video for validation and test.
    train_count = max(3, int(count * 0.70))
    train_count = min(train_count, count - 2)
    val_count = max(1, int(count * 0.15))
    if train_count + val_count >= count:
        val_count = 1
    train = video_items[:train_count]
    val = video_items[train_count:train_count + val_count]
    test = video_items[train_count + val_count:]
    return train, val, test


def extract_video_rows(video_id, sign, subset, landmarker):
    video_file = VIDEOS_FOLDER / f"{video_id}.mp4"
    cap = cv2.VideoCapture(str(video_file))
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

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect(mp_image)
        if not result.hand_landmarks:
            continue

        hand = result.hand_landmarks[0]
        values = []
        for landmark in hand:
            values.extend([landmark.x, landmark.y, landmark.z])

        if len(values) != 63:
            continue

        rows.append({
            "video_id": video_id,
            "sign": sign,
            "subset": subset,
            **{FEATURE_COLUMNS[i]: values[i] for i in range(63)}
        })

    cap.release()
    return rows


def main():
    print("Building a balanced dataset for the existing sign classes...\n")

    for required_file in [DATASET_FILE, JSON_FILE, CLASS_FILE, MODEL_FILE]:
        if not required_file.exists():
            print(f"ERROR : Required file not found. {required_file}")
            return
    if not VIDEOS_FOLDER.exists():
        print(f"ERROR : Videos folder not found. {VIDEOS_FOLDER}")
        return

    current_data = pd.read_csv(DATASET_FILE)
    selected_signs = sorted(current_data["sign"].dropna().unique().tolist())
    print(f"Signs to keep ({len(selected_signs)}): {', '.join(selected_signs)}")

    labels = load_labels()
    with open(JSON_FILE, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    # Gather available videos only for signs already used in this project.
    candidates = {sign: [] for sign in selected_signs}
    for video_id, info in dataset.items():
        video_file = VIDEOS_FOLDER / f"{video_id}.mp4"
        if not video_file.exists():
            continue
        action = info.get("action", [])
        if not action:
            continue
        sign = labels.get(action[0])
        if sign in candidates:
            candidates[sign].append(video_id)

    # Deterministically cap the number of videos per sign.
    rng = random.Random(RANDOM_SEED)
    selected_videos = {}
    insufficient = {}
    for sign in selected_signs:
        ids = sorted(set(candidates[sign]))
        rng.shuffle(ids)
        ids = ids[:MAX_VIDEOS_PER_SIGN]
        selected_videos[sign] = ids
        if len(ids) < 5:
            insufficient[sign] = len(ids)

    print("\nAvailable videos per sign:")
    for sign in selected_signs:
        print(f"  {sign}: {len(selected_videos[sign])}")

    if insufficient:
        print("\nSTOPPED : At least 5 available videos per sign are required for a useful split.")
        print("These signs do not have enough available videos.")
        for sign, count in insufficient.items():
            print(f"  {sign}: {count} videos")
        print("Your existing CSV has NOT been changed.")
        return

    split_records = []
    for sign in selected_signs:
        train_ids, val_ids, test_ids = split_videos(selected_videos[sign])
        split_records.extend((video_id, sign, "train") for video_id in train_ids)
        split_records.extend((video_id, sign, "val") for video_id in val_ids)
        split_records.extend((video_id, sign, "test") for video_id in test_ids)

    print("\nVideo counts per sign and subset:")
    print(pd.crosstab(
        pd.Series([record[1] for record in split_records], name="sign"),
        pd.Series([record[2] for record in split_records], name="subset")
    ).to_string())

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_FILE)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5
    )

    all_rows = []
    total = len(split_records)
    with HandLandmarker.create_from_options(options) as landmarker:
        for index, (video_id, sign, subset) in enumerate(split_records, start=1):
            rows = extract_video_rows(video_id, sign, subset, landmarker)
            all_rows.extend(rows)
            print(f"[{index}/{total}] {video_id} -> {sign} ({subset}), frames kept: {len(rows)}")

    if not all_rows:
        print("\nERROR : No hand landmarks were extracted. Existing CSV has NOT been changed.")
        return

    new_data = pd.DataFrame(all_rows, columns=["video_id", "sign", "subset"] + FEATURE_COLUMNS)
    videos_with_rows = new_data[["video_id", "sign", "subset"]].drop_duplicates()
    coverage = pd.crosstab(videos_with_rows["sign"], videos_with_rows["subset"])

    missing_train = set(selected_signs) - set(new_data.loc[new_data["subset"] == "train", "sign"])
    missing_val = set(selected_signs) - set(new_data.loc[new_data["subset"] == "val", "sign"])
    missing_test = set(selected_signs) - set(new_data.loc[new_data["subset"] == "test", "sign"])
    if missing_train or missing_val or missing_test:
        print("\nSTOPPED: Some signs have no extracted frames in one or more subsets.")
        print(f"Missing from train : {sorted(missing_train)}")
        print(f"Missing from validation : {sorted(missing_val)}")
        print(f"Missing from test : {sorted(missing_test)}")
        print("Existing CSV has NOT been changed.")
        return

    # Save and verify the new CSV before replacing the original.
    new_data.to_csv(TEMP_FILE, index=False)
    check = pd.read_csv(TEMP_FILE)
    if check.empty or check[FEATURE_COLUMNS].isna().any().any():
        TEMP_FILE.unlink(missing_ok=True)
        print("ERROR : New CSV failed verification. Existing CSV has NOT been changed.")
        return

    if not BACKUP_FILE.exists():
        shutil.copy2(DATASET_FILE, BACKUP_FILE)
        print(f"\nOriginal dataset backup: {BACKUP_FILE}")

    TEMP_FILE.replace(DATASET_FILE)
    print("\nDataset updated successfully.")
    print(f"Rows : {len(new_data)}")
    print(f"Unique videos : {new_data['video_id'].nunique()}")
    print(f"CSV : {DATASET_FILE}")
    print("\nFinal video coverage :")
    print(coverage.to_string())
    print("\nNext : run python src/check_class_coverage.py and share its output before retraining.")


if __name__ == "__main__":
    main()
