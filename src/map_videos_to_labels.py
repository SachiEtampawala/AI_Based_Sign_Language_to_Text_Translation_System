import json
from pathlib import Path

DATASET_PATH = Path.home() / "Downloads" / "archive"

JSON_FILE = DATASET_PATH / "nslt_100.json"
CLASS_FILE = DATASET_PATH / "wlasl_class_list.txt"

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

def main():
    print("Mapping videos to sign-language labels...\n")

    # Load class labels
    labels = load_class_labels()

    # Load video information
    with open(JSON_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"Total videos : {len(data)}")
    print(f"Total classes : {len(labels)}\n")

    print("First 20 video-label mappings :\n")

    count = 0

    for video_id, information in data.items():

        action = information["action"]
        action_id = action[0]

        word = labels.get(action_id, "Unknown")

        print(f"Video ID : {video_id}")
        print(f"Subset : {information['subset']}")
        print(f"Action ID : {action_id}")
        print(f"Sign : {word}")
        print("-" * 40)

        count += 1

        if count == 20:
            break

    print("\nMapping completed.")

if __name__ == "__main__":
    main()