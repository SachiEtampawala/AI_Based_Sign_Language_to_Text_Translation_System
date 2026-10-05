import json
import pandas as pd
from pathlib import Path
import cv2
import mediapipe as mp

DATASET_PATH = Path.home() / "Downloads" / "archive"

JSON_FILE = DATASET_PATH / "nslt_100.json"
CLASS_FILE = DATASET_PATH / "wlasl_class_list.txt"
VIDEOS_FOLDER = DATASET_PATH / "videos"

PROJECT_PATH = Path(__file__).resolve().parent.parent
OUTPUT_FOLDER = PROJECT_PATH / "outputs"
DATASET_FILE = OUTPUT_FOLDER / "landmark_dataset.csv"
MODEL_FILE = PROJECT_PATH / "models" / "hand_landmarker.task"

TARGET_CLASSES = [
    "africa",
    "all",
    "birthday",
    "cheat",
    "give",
    "help",
    "mother",
    "pink",
    "play",
    "purple",
    "son",
    "tell",
    "what",
]

TARGET_TRAIN_VIDEOS = 5

def load_class_labels():
    labels = {}

    with open(CLASS_FILE, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) == 2:
                action_id = int(parts[0])
                word = parts[1].strip()
                labels[action_id] = word

    return labels

def extract_landmarks(video_file, landmarker):
    cap = cv2.VideoCapture(str(video_file))

    if not cap.isOpened():
        return []

    rows = []
    frame_number = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = landmarker.detect(mp_image)

        if not result.hand_landmarks:
            continue

        hand = result.hand_landmarks[0]

        features = []

        for landmark in hand:
            features.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        rows.append(features)

    cap.release()

    return rows


def main():
    print("Balancing underrepresented training classes...\n")

    if not JSON_FILE.exists():
        print("ERROR : JSON file not found.")
        print(JSON_FILE)
        return

    if not CLASS_FILE.exists():
        print("ERROR : Class list file not found.")
        print(CLASS_FILE)
        return

    if not VIDEOS_FOLDER.exists():
        print("ERROR : Videos folder not found.")
        print(VIDEOS_FOLDER)
        return

    if not DATASET_FILE.exists():
        print("ERROR : Landmark dataset not found.")
        print(DATASET_FILE)
        return

    if not MODEL_FILE.exists():
        print("ERROR : MediaPipe model not found.")
        print(MODEL_FILE)
        return

    labels = load_class_labels()

    with open(JSON_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    data = pd.read_csv(DATASET_FILE)

    existing_video_ids = set(
        data["video_id"].astype(str)
    )

    existing_video_data = (
        data[["video_id", "sign", "subset"]]
        .drop_duplicates()
    )

    print(
        f"Existing unique videos : "
        f"{len(existing_video_data)}"
    )

    print("\nCurrent training video counts :")

    for sign in TARGET_CLASSES:
        count = len(
            existing_video_data[
                (existing_video_data["sign"] == sign)
                & (existing_video_data["subset"] == "train")
            ]
        )

        print(f"{sign}: {count}")

    candidates = {}

    for video_id, information in metadata.items():

        video_id = str(video_id)

        if video_id in existing_video_ids:
            continue

        if information.get("subset") != "train":
            continue

        action_id = information["action"][0]

        sign = labels.get(action_id)

        if sign not in TARGET_CLASSES:
            continue

        video_file = VIDEOS_FOLDER / f"{video_id}.mp4"

        if not video_file.exists():
            continue

        candidates.setdefault(sign, []).append(video_id)

    selected_videos = {}

    print("\nVideos that will be added :")

    total_selected = 0

    for sign in TARGET_CLASSES:

        current_count = len(
            existing_video_data[
                (existing_video_data["sign"] == sign)
                & (existing_video_data["subset"] == "train")
            ]
        )

        needed = max(
            TARGET_TRAIN_VIDEOS - current_count,
            0
        )

        available = candidates.get(sign, [])

        selected = available[:needed]

        selected_videos[sign] = selected

        print(
            f"{sign}: "
            f"current={current_count}, "
            f"needed={needed}, "
            f"available={len(available)}, "
            f"selected={len(selected)}"
        )

        total_selected += len(selected)

    if total_selected == 0:
        print("\nNo additional videos are needed.")
        return

    print(
        f"\nTotal new videos : {total_selected}"
    )

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = (
        mp.tasks.vision.HandLandmarkerOptions
    )
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_FILE)
        ),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    new_rows = []

    with HandLandmarker.create_from_options(options) as landmarker:

        for sign in TARGET_CLASSES:

            videos = selected_videos[sign]

            if not videos:
                continue

            print(f"\nProcessing: {sign}")

            for index, video_id in enumerate(
                videos,
                start=1
            ):

                video_file = (
                    VIDEOS_FOLDER
                    / f"{video_id}.mp4"
                )

                print(
                    f"  [{index}/{len(videos)}] "
                    f"{video_id}.mp4"
                )

                landmark_rows = extract_landmarks(
                    video_file,
                    landmarker
                )

                for features in landmark_rows:

                    row = {
                        "video_id": video_id,
                        "sign": sign,
                        "subset": "train",
                    }

                    for feature_index, value in enumerate(
                        features
                    ):
                        row[
                            f"landmark_{feature_index}"
                        ] = value

                    new_rows.append(row)

    if not new_rows:
        print(
            "\nNo landmark data was extracted."
        )
        return

    new_data = pd.DataFrame(new_rows)

    combined_data = pd.concat(
        [data, new_data],
        ignore_index=True
    )

    combined_data.to_csv(
        DATASET_FILE,
        index=False
    )

    print("\nDataset updated successfully.")

    print(
        f"Previous rows : {len(data)}"
    )

    print(
        f"New rows : {len(new_data)}"
    )

    print(
        f"Total rows : {len(combined_data)}"
    )

    print(
        f"\nSaved to :\n{DATASET_FILE}"
    )

    print(
        "\nBalanced dataset expansion completed."
    )

if __name__ == "__main__":
    main()