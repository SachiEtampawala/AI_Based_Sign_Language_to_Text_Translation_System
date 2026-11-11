import cv2
import mediapipe as mp
import joblib
import numpy as np

from pathlib import Path
from collections import deque, Counter

from features import hands_to_vector, normalize_landmarks, FEATURE_SIZE
from sentence_builder import WordBuffer

# Project paths
PROJECT_PATH = Path(__file__).resolve().parent.parent
MODEL_FILE = PROJECT_PATH / "models" / "sign_recognition_model.pkl"
HAND_MODEL_FILE = PROJECT_PATH / "models" / "hand_landmarker.task"

# Settings
CONFIDENCE_THRESHOLD = 0.40
STABLE_FRAMES = 10          # frames a sign must be held to become a word
PREDICTION_HISTORY = deque(maxlen=7)

# Colors (BGR)
DARK = (35, 35, 35)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
PURPLE = (200, 100, 150)
GREY = (90, 90, 90)
HAND_COLORS = [(0, 255, 0), (255, 200, 0)]   # first hand, second hand

# Lines that connect the 21 hand points into a skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # little
    (0, 17),                                   # palm edge
]


def draw_hands(frame, hands):
    """Draw hand skeletons on the DISPLAY frame.

    Detection runs on the un-mirrored camera image, but the screen shows a
    mirrored image, so x is flipped here (x -> 1 - x).
    """
    height, width = frame.shape[:2]
    ordered = sorted(hands, key=lambda hand: hand[0].x)

    for index, hand in enumerate(ordered):
        color = HAND_COLORS[index % len(HAND_COLORS)]

        points = [
            (int((1.0 - landmark.x) * width), int(landmark.y * height))
            for landmark in hand
        ]

        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], color, 2)

        for point in points:
            cv2.circle(frame, point, 4, color, -1)


def draw_overlay(frame, sign, confidence, word_buffer, hand_count=0):
    """Draw the top bar (current sign) and bottom bar (sentence)."""
    height, width = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (width, 100), DARK, -1)

    cv2.putText(
        frame, "Recognized Sign: " + sign, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2
    )
    cv2.putText(
        frame, f"Confidence: {confidence:.0%}", (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, GREEN, 2
    )

    # Confidence bar
    conf_x, conf_y, conf_w, conf_h = 215, 62, 150, 16
    cv2.rectangle(
        frame, (conf_x, conf_y), (conf_x + conf_w, conf_y + conf_h), GREY, 1
    )
    conf_fill = int(conf_w * max(0.0, min(confidence, 1.0)))
    if conf_fill > 0:
        cv2.rectangle(
            frame, (conf_x, conf_y), (conf_x + conf_fill, conf_y + conf_h),
            GREEN, -1
        )

    # Hold progress bar (fills up until the word is added)
    bar_x, bar_y, bar_w, bar_h = width - 220, 60, 200, 16
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), GREY, 1)
    filled = int(bar_w * word_buffer.progress)
    if filled > 0:
        cv2.rectangle(
            frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), PURPLE, -1
        )
    cv2.putText(
        frame, "Hold sign", (bar_x, bar_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1
    )
    cv2.putText(
        frame, f"Hands: {hand_count}", (bar_x, bar_y - 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1
    )

    # Bottom bar: the sentence built so far
    cv2.rectangle(frame, (0, height - 80), (width, height), DARK, -1)

    words = list(word_buffer.words)
    text = " ".join(words) if words else "(no words yet)"

    # Keep the text inside the window: drop the oldest words if too long
    while words and cv2.getTextSize(
        "Sentence: " + text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    )[0][0] > width - 40:
        words.pop(0)
        text = "... " + " ".join(words)

    cv2.putText(
        frame, "Sentence: " + text, (20, height - 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2
    )
    cv2.putText(
        frame, "Backspace = delete last word   C = clear   Q = quit",
        (20, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1
    )


def main():
    print("Starting Sign Language Recognition System...\n")

    # Check required files
    if not MODEL_FILE.exists():
        print("ERROR : Sign recognition model not found.")
        print(MODEL_FILE)
        return

    if not HAND_MODEL_FILE.exists():
        print("ERROR : MediaPipe hand model not found.")
        print(HAND_MODEL_FILE)
        return

    # Load the trained sign recognition model
    print("Loading trained model...")
    model = joblib.load(MODEL_FILE)

    expected = getattr(model, "n_features_in_", FEATURE_SIZE)
    if expected != FEATURE_SIZE:
        print(f"ERROR : The saved model expects {expected} features, "
              f"but this app uses {FEATURE_SIZE} (two hands).")
        print("Retrain it with:")
        print("  python src/build_dataset.py")
        print("  python src/train_sign_model.py")
        return

    print("Sign recognition model loaded.")

    # Set up MediaPipe Tasks API (up to 2 hands)
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(HAND_MODEL_FILE)
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Open the laptop camera
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open the camera.")
        print("Check that your camera is connected and not being used by another app.")
        return

    print("\nCamera started.")
    print("Show one or two hands to the camera.")
    print("Hold a sign steady to add it to the sentence.")
    print("Backspace = delete last word | C = clear | Q = quit\n")

    last_prediction = "Show your hand"
    last_confidence = 0.0
    word_buffer = WordBuffer(stable_frames=STABLE_FRAMES)

    # VIDEO mode needs steadily increasing timestamps (milliseconds)
    timestamp_ms = 0

    try:
        with HandLandmarker.create_from_options(options) as landmarker:

            while True:
                success, frame = camera.read()

                if not success:
                    print("ERROR : Could not read a camera frame.")
                    break

                # Detect on the ORIGINAL (not mirrored) camera image.
                # The training videos are not mirrored, so this matches them.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                timestamp_ms += 33

                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                # Mirror the picture only for display
                frame = cv2.flip(frame, 1)

                hands = result.hand_landmarks or []

                if hands:
                    draw_hands(frame, hands)

                    # 126 raw numbers (two hands), then normalize -> 129
                    raw = hands_to_vector(hands)
                    features = normalize_landmarks(raw).reshape(1, -1)

                    prediction = model.predict(features)[0]

                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(features)[0]
                        confidence = float(np.max(probabilities))
                    else:
                        confidence = 1.0

                    if confidence >= CONFIDENCE_THRESHOLD:
                        PREDICTION_HISTORY.append(str(prediction))

                        # Most frequent recent prediction (smoothing)
                        last_prediction = Counter(
                            PREDICTION_HISTORY
                        ).most_common(1)[0][0]

                        last_confidence = confidence

                        word_buffer.update(last_prediction)
                    else:
                        PREDICTION_HISTORY.clear()
                        last_prediction = "Uncertain sign"
                        last_confidence = confidence

                        word_buffer.update(None)

                else:
                    PREDICTION_HISTORY.clear()
                    last_prediction = "Show your hand"
                    last_confidence = 0.0

                    # Hands left the view: the same word can be signed again
                    word_buffer.hand_lost()

                draw_overlay(
                    frame, last_prediction, last_confidence,
                    word_buffer, hand_count=len(hands)
                )

                cv2.imshow("AI Sign Language Recognition", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                elif key == 8:                 # Backspace
                    word_buffer.remove_last()
                elif key == ord("c"):
                    word_buffer.clear()

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("\nWebcam recognition stopped.")

if __name__ == "__main__":
    main()