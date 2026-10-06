import json
import pandas as pd
import cv2
import mediapipe as mp
from pathlib import Path
from sklearn.model_selection import train_test_split


DATASET_PATH = Path.home() / "Downloads" / "archive"
JSON_FILE = DATASET_PATH / "nslt_100.json"
VIDEOS_FOLDER = DATASET_PATH / "videos"

PROJECT_PATH = Path(__file__).resolve().parent.parent
DATASET_FILE = PROJECT_PATH / "outputs" / "landmark_dataset.csv"
MODEL_FILE = PROJECT_PATH / "models" / "hand_landmarker.task"


TARGET_TRAIN_VIDEOS = 5
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


def load_labels():
    class_file = DATASET_PATH / "wlasl_class_list.txt"

    labels = {}

    with open(class_file, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(maxsplit=1)

            if len(parts) == 2:
                action_id = int(parts[0])
                labels[action_id] = parts[1]

    return labels


def extract_landmarks(video_id, sign, subset):
    video_file = VIDEOS_FOLDER / f"{video_id}.mp4"

    if not video_file.exists():
        return []

    cap = cv2.VideoCapture(str(video_file))

    if not cap.isOpened():
        return []

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_FILE)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    rows = []

    with HandLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = cap.read()

            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            result = landmarker.detect(mp_image)

            if not result.hand_landmarks:
                continue

            hand = result.hand_landmarks[0]

            row = {
                "video_id": video_id,
                "sign": sign,
                "subset": subset
            }

            for landmark_number, landmark in enumerate(hand):
                row[f"landmark_{landmark_number * 3}"] = landmark.x
                row[f"landmark_{landmark_number * 3 + 1}"] = landmark.y
                row[f"landmark_{landmark_number * 3 + 2}"] = landmark.z

            rows.append(row)

    cap.release()

    return rows


def main():

    print("Creating balanced video-level dataset...\n")

    if not DATASET_FILE.exists():
        print("ERROR: Existing landmark dataset not found:")
        print(DATASET_FILE)
        return

    if not MODEL_FILE.exists():
        print("ERROR: MediaPipe model not found:")
        print(MODEL_FILE)
        return

    labels = load_labels()

    with open(JSON_FILE, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    # ---------------------------------------------------------
    # Get one record per video
    # ---------------------------------------------------------

    videos = []

    for video_id, information in dataset.items():

        video_file = VIDEOS_FOLDER / f"{video_id}.mp4"

        if not video_file.exists():
            continue

        action_id = information["action"][0]
        sign = labels.get(action_id)

        if sign is None:
            continue

        videos.append({
            "video_id": video_id,
            "sign": sign
        })

    videos_df = pd.DataFrame(videos)

    print(f"Available videos: {len(videos_df)}")

    # ---------------------------------------------------------
    # Create video-level split
    # ---------------------------------------------------------

    train_videos = []
    val_videos = []
    test_videos = []

    for sign, group in videos_df.groupby("sign"):

        video_ids = group["video_id"].tolist()

        # Shuffle videos for this sign
        random_state = 42

        if len(video_ids) >= 3:

            train_ids, remaining_ids = train_test_split(
                video_ids,
                test_size=VALIDATION_RATIO + TEST_RATIO,
                random_state=random_state
            )

            relative_test_ratio = TEST_RATIO / (
                VALIDATION_RATIO + TEST_RATIO
            )

            val_ids, test_ids = train_test_split(
                remaining_ids,
                test_size=relative_test_ratio,
                random_state=random_state
            )

        else:
            # Very small classes cannot have all three subsets.
            train_ids = video_ids
            val_ids = []
            test_ids = []

        for video_id in train_ids:
            train_videos.append((video_id, sign))

        for video_id in val_ids:
            val_videos.append((video_id, sign))

        for video_id in test_ids:
            test_videos.append((video_id, sign))

    print("\nVideo-level split:")
    print(f"Training videos   : {len(train_videos)}")
    print(f"Validation videos : {len(val_videos)}")
    print(f"Testing videos    : {len(test_videos)}")

    # ---------------------------------------------------------
    # Show class distribution
    # ---------------------------------------------------------

    train_df = pd.DataFrame(train_videos, columns=["video_id", "sign"])
    val_df = pd.DataFrame(val_videos, columns=["video_id", "sign"])
    test_df = pd.DataFrame(test_videos, columns=["video_id", "sign"])

    print("\nTraining videos by class:")
    print(train_df["sign"].value_counts().sort_index())

    print("\nValidation videos by class:")
    print(val_df["sign"].value_counts().sort_index())

    print("\nTesting videos by class:")
    print(test_df["sign"].value_counts().sort_index())

    # ---------------------------------------------------------
    # Extract landmarks again using the new video-level split
    # ---------------------------------------------------------

    all_rows = []

    split_groups = [
        ("train", train_videos),
        ("val", val_videos),
        ("test", test_videos)
    ]

    for subset, video_list in split_groups:

        print(f"\nProcessing {subset} videos...")

        for index, (video_id, sign) in enumerate(video_list, start=1):

            rows = extract_landmarks(
                video_id,
                sign,
                subset
            )

            all_rows.extend(rows)

            print(
                f"[{index}/{len(video_list)}] "
                f"{video_id} -> {sign} "
                f"({len(rows)} frames)"
            )

    # ---------------------------------------------------------
    # Save new dataset
    # ---------------------------------------------------------

    new_data = pd.DataFrame(all_rows)

    if new_data.empty:
        print("\nERROR: No landmark data was created.")
        return

    feature_columns = [
        f"landmark_{i}"
        for i in range(63)
    ]

    columns = [
        "video_id",
        "sign",
        "subset"
    ] + feature_columns

    new_data = new_data[columns]

    new_data.to_csv(
        DATASET_FILE,
        index=False
    )

    print("\n-----------------------------------")
    print("New dataset created successfully.")
    print(f"Total rows: {len(new_data)}")
    print(f"Total videos: {new_data['video_id'].nunique()}")
    print(f"Saved to: {DATASET_FILE}")

    print("\nRows by subset:")
    print(new_data["subset"].value_counts())

    print("\nVideo-level dataset creation completed.")

if __name__ == "__main__":
    main()