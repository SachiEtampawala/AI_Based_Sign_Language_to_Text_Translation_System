import cv2
import json
from pathlib import Path

DATASET_PATH = Path.home() / "Downloads" / "archive"

JSON_FILE = DATASET_PATH / "nslt_100.json"
VIDEOS_FOLDER = DATASET_PATH / "videos"

def main():
    print("Inspecting WLASL video...\n")

    # Load dataset information

    with open(JSON_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Find the first video that actually exists

    video_id = None

    for current_id in data:
        video_file = VIDEOS_FOLDER / f"{current_id}.mp4"

        if video_file.exists():
            video_id = current_id
            break

    if video_id is None:
        print("ERROR : No matching video file was found.")
        return

    video_file = VIDEOS_FOLDER / f"{video_id}.mp4"

    print(f"Video ID : {video_id}")
    print(f"Video path : {video_file}")

    cap = cv2.VideoCapture(str(video_file))

    if not cap.isOpened():
        print("\nERROR : Could not open the video.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps > 0 else 0

    print("\nVideo information :")
    print(f"Frames : {frame_count}")
    print(f"FPS : {fps:.2f}")
    print(f"Width : {width}")
    print(f"Height : {height}")
    print(f"Duration : {duration:.2f} seconds")

    cap.release()

    print("\nVideo inspection completed.")

if __name__ == "__main__":
    main()