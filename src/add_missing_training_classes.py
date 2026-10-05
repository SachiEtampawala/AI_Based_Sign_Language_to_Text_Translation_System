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

TARGET_CLASSES = [
    "decide",
    "go",
    "jacket",
    "man",
    "many",
    "no",
    "orange",
]

TARGET_VIDEOS_PER_CLASS = 20

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

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

        rows.append({
            "frame_number": frame_number,
            "features": features
        })

    cap.release()

    return rows

def main():

    print("Adding missing training classes...\n")

    if not JSON_FILE.exists():
        print("ERROR : JSON file not found.")
        print(JSON_FILE)
        return

    if not CLASS_FILE.exists():
        print("ERROR : Class list not found.")
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

    labels = load_class_labels()

    with open(JSON_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    existing_data = pd.read_csv(DATASET_FILE)

    existing_video_ids = set(
        existing_data["video_id"].astype(str)
    )

    print(f"Existing processed videos : {len(existing_video_ids)}")

    print("\nTarget classes :")

    for target in TARGET_CLASSES:
        print(f"- {target}")

    candidates = {}

    for video_id, information in metadata.items():

        video_id = str(video_id)

        if video_id in existing_video_ids:
            continue

        action = information["action"]

        action_id = action[0]

        sign = labels.get(action_id)

        if sign not in TARGET_CLASSES:
            continue

        if information.get("subset") != "train":
            continue

        video_file = VIDEOS_FOLDER / f"{video_id}.mp4"

        if not video_file.exists():
            continue

        if sign not in candidates:
            candidates[sign] = []

        candidates[sign].append(video_id)

    print("\nAvailable training videos :")

    for target in TARGET_CLASSES:
        count = len(candidates.get(target, []))
        print(f"{target}: {count}")

    selected_videos = {}

    for target in TARGET_CLASSES:

        available = candidates.get(target, [])

        selected_videos[target] = available[
            :TARGET_VIDEOS_PER_CLASS
        ]

    print("\nVideos selected :")

    total_selected = 0

    for target in TARGET_CLASSES:

        selected = selected_videos[target]

        print(
            f"{target}: {len(selected)} videos"
        )

        total_selected += len(selected)

    if total_selected == 0:
        print("\nNo new videos were found.")
        return

    print(
        f"\nTotal new videos to process : "
        f"{total_selected}"
    )

    BaseOptions = mp.tasks.BaseOptions

    HandLandmarker = mp.tasks.vision.HandLandmarker

    HandLandmarkerOptions = (
        mp.tasks.vision.HandLandmarkerOptions
    )

    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(
                PROJECT_PATH
                / "models"
                / "hand_landmarker.task"
            )
        ),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    new_rows = []

    with HandLandmarker.create_from_options(options) as landmarker:

        for target in TARGET_CLASSES:

            selected = selected_videos[target]

            print(
                f"\nProcessing class: {target}"
            )

            for index, video_id in enumerate(
                selected,
                start=1
            ):

                video_file = (
                    VIDEOS_FOLDER
                    / f"{video_id}.mp4"
                )

                print(
                    f"  [{index}/{len(selected)}] "
                    f"{video_id}.mp4"
                )

                landmark_frames = extract_landmarks(
                    video_file,
                    landmarker
                )

                subset = "train"

                for frame_data in landmark_frames:

                    row = {
                        "video_id": video_id,
                        "sign": target,
                        "subset": subset,
                    }

                    for feature_index, value in enumerate(
                        frame_data["features"]
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
        [existing_data, new_data],
        ignore_index=True
    )

    combined_data.to_csv(
        DATASET_FILE,
        index=False
    )

    print("\nDataset updated successfully.")

    print(
        f"Previous rows : {len(existing_data)}"
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
        "\nMissing training class expansion completed."
    )

if __name__ == "__main__":
    main()