"""Record your own sign samples so the model learns YOUR hands and camera.

Run :  python src/collect_samples.py
Saves : outputs/my_samples.csv   (same columns as the WLASL dataset)

How it works
  1. Press a number key to choose a word (N = "no sign" / resting hand).
  2. Press SPACE. After a 3 second countdown it records for 2.5 seconds.
  3. Repeat about 15 times for each word, moving a little each time.
  U = undo the last recording     Q or Esc (or the X button) = quit
"""
import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd

from features import hands_to_vector, RAW_SIZE, NO_SIGN_LABEL
from webcam_sign_to_text import draw_hands

# Paths
PROJECT_PATH = Path(__file__).resolve().parent.parent
OUTPUT_FOLDER = PROJECT_PATH / "outputs"
WLASL_FILE = OUTPUT_FOLDER / "landmark_dataset_2hands.csv"
OLD_WLASL_FILE = OUTPUT_FOLDER / "landmark_dataset.csv"
OWN_FILE = OUTPUT_FOLDER / "my_samples.csv"
HAND_MODEL_FILE = PROJECT_PATH / "models" / "hand_landmarker.task"

# Settings
COUNTDOWN_SECONDS = 3
RECORD_SECONDS = 2.5
FRAME_STEP = 3            # keep every 3rd frame (same as the WLASL dataset)
MIN_FRAMES = 5            # a recording with fewer hand frames is thrown away
WINDOW_NAME = "Record my signs"

HEADER = ["video_id", "sign", "subset"] + [
    f"landmark_{i}" for i in range(RAW_SIZE)
]

DARK = (35, 35, 35)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (60, 60, 255)
YELLOW = (0, 220, 255)
GREY = (180, 180, 180)


# ---------- data helpers (no camera needed) ----------

def load_labels():
    """The words to record: the signs in the WLASL dataset + 'none'."""
    source = WLASL_FILE if WLASL_FILE.exists() else OLD_WLASL_FILE
    signs = []

    if source.exists():
        signs = sorted(
            pd.read_csv(source, usecols=["sign"])["sign"].dropna().unique()
        )

    return list(signs) + [NO_SIGN_LABEL]


def count_takes():
    """How many recordings are saved for each word."""
    if not OWN_FILE.exists():
        return {}

    data = pd.read_csv(OWN_FILE, usecols=["video_id", "sign"])
    data = data.drop_duplicates()
    return data["sign"].value_counts().to_dict()


def append_take(video_id, sign, vectors):
    """Add one recording (a list of 126-number vectors) to the CSV."""
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    new_file = not OWN_FILE.exists()

    with open(OWN_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if new_file:
            writer.writerow(HEADER)

        for vector in vectors:
            writer.writerow([video_id, sign, "own", *vector.tolist()])


def remove_take(video_id):
    """Delete one recording from the CSV."""
    if not OWN_FILE.exists():
        return

    data = pd.read_csv(OWN_FILE)
    data = data[data["video_id"] != video_id]
    data.to_csv(OWN_FILE, index=False)


def key_for_index(index):
    """Words 1-9 use keys 1-9, the 10th word uses key 0."""
    return str((index + 1) % 10)


# ---------- screen drawing ----------

def draw_ui(frame, labels, counts, selected, state, seconds_left, message):
    height, width = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (width, 70), DARK, -1)

    if selected is None:
        title = "Press a number key to choose a word"
    else:
        title = f"Word: {selected}   (recordings: {counts.get(selected, 0)})"

    cv2.putText(frame, title, (20, 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, WHITE, 2)

    # Word list on the right
    box_w = 190
    box_h = 12 + 24 * len(labels)
    overlay = frame.copy()
    cv2.rectangle(frame, (width - box_w, 80), (width, 80 + box_h), DARK, -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    for index, label in enumerate(labels):
        key = key_for_index(index) if index < 10 else "-"
        if label == NO_SIGN_LABEL:
            key = "N"

        color = YELLOW if label == selected else WHITE
        cv2.putText(
            frame, f"{key}  {label}: {counts.get(label, 0)}",
            (width - box_w + 10, 102 + 24 * index),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1
        )

    # Big status in the middle
    if state == "countdown":
        cv2.putText(frame, str(int(seconds_left) + 1),
                    (width // 2 - 30, height // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, YELLOW, 6)
    elif state == "recording":
        cv2.circle(frame, (40, 105), 12, RED, -1)
        cv2.putText(frame, f"RECORDING {seconds_left:.1f}s", (65, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)

    # Bottom bar
    cv2.rectangle(frame, (0, height - 70), (width, height), DARK, -1)
    cv2.putText(frame, message, (20, height - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
    cv2.putText(
        frame, "SPACE = record   U = undo last   Q or Esc = quit",
        (20, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1
    )


# ---------- main ----------

def main():
    if not HAND_MODEL_FILE.exists():
        print("ERROR : MediaPipe hand model not found.")
        print(HAND_MODEL_FILE)
        return

    labels = load_labels()

    if len(labels) < 2:
        print("ERROR : No dataset found to read the word list from.")
        print("Run  python src/build_dataset.py  first.")
        return

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(HAND_MODEL_FILE)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open the camera.")
        return

    print("Recorder started. Choose a word, then press SPACE.")

    counts = count_takes()
    selected = None
    state = "idle"                 # idle -> countdown -> recording -> idle
    phase_start = 0.0
    vectors = []
    frame_counter = 0
    last_take = None
    message = "Ready"
    message_until = 0.0

    start_time = time.time()
    last_timestamp = 0

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:
                success, frame = camera.read()

                if not success:
                    print("ERROR : Could not read a camera frame.")
                    break

                # Detect on the ORIGINAL image (same as the webcam app)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb
                )

                timestamp = max(
                    int((time.time() - start_time) * 1000), last_timestamp + 1
                )
                last_timestamp = timestamp

                result = landmarker.detect_for_video(mp_image, timestamp)
                hands = result.hand_landmarks or []

                frame = cv2.flip(frame, 1)

                if hands:
                    draw_hands(frame, hands)

                now = time.time()
                seconds_left = 0.0

                if state == "countdown":
                    seconds_left = COUNTDOWN_SECONDS - (now - phase_start)

                    if seconds_left <= 0:
                        state = "recording"
                        phase_start = now
                        vectors = []
                        frame_counter = 0

                elif state == "recording":
                    seconds_left = RECORD_SECONDS - (now - phase_start)

                    frame_counter += 1
                    if hands and frame_counter % FRAME_STEP == 0:
                        vectors.append(hands_to_vector(hands))

                    if seconds_left <= 0:
                        state = "idle"

                        if len(vectors) >= MIN_FRAMES:
                            video_id = f"mine_{selected}_{int(now)}"
                            append_take(video_id, selected, vectors)
                            last_take = video_id
                            counts = count_takes()
                            message = (
                                f"Saved {len(vectors)} frames for '{selected}'"
                            )
                        else:
                            message = ("Not enough hand frames. "
                                       "Keep your hand in view and try again.")

                        message_until = now + 4

                if state == "idle" and now > message_until and \
                        message != "Ready":
                    message = "Ready"

                draw_ui(frame, labels, counts, selected, state,
                        max(seconds_left, 0.0), message)

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break

                if cv2.getWindowProperty(
                    WINDOW_NAME, cv2.WND_PROP_VISIBLE
                ) < 1:
                    break

                if state != "idle":
                    continue

                # Choose a word
                char = chr(key).lower() if 0 < key < 128 else ""

                if char == "n":
                    selected = NO_SIGN_LABEL
                elif char.isdigit():
                    index = (int(char) - 1) % 10
                    if index < len(labels) - 1:
                        selected = labels[index]

                elif key == 32:                              # SPACE
                    if selected is None:
                        message = "Choose a word first (press a number key)"
                        message_until = now + 3
                    else:
                        state = "countdown"
                        phase_start = now

                elif char == "u":
                    if last_take:
                        remove_take(last_take)
                        counts = count_takes()
                        last_take = None
                        message = "Last recording removed"
                    else:
                        message = "Nothing to undo"
                    message_until = now + 3

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Recorder stopped.")
        print(f"Recordings per word: {count_takes()}")


if __name__ == "__main__":
    main()