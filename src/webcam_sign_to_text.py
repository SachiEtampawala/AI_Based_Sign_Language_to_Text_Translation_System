import cv2
import mediapipe as mp
import joblib
import numpy as np

from pathlib import Path
from collections import deque, Counter

# Project paths
PROJECT_PATH = Path(__file__).resolve().parent.parent
MODEL_FILE = PROJECT_PATH / "models" / "sign_recognition_model.pkl"
HAND_MODEL_FILE = PROJECT_PATH / "models" / "hand_landmarker.task"

# Settings
CONFIDENCE_THRESHOLD = 0.40
PREDICTION_HISTORY = deque(maxlen=7)


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
    print("Sign recognition model loaded.")

    # Set up MediaPipe Tasks API
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(HAND_MODEL_FILE)
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
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
    print("Show your hand to the camera.")
    print("Press Q to close the program.\n")

    last_prediction = "Show your hand"
    last_confidence = 0.0

    try:
        with HandLandmarker.create_from_options(options) as landmarker:

            while True:
                success, frame = camera.read()

                if not success:
                    print("ERROR : Could not read a camera frame.")
                    break

                # Mirror the camera view
                frame = cv2.flip(frame, 1)

                # Convert the frame to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # Create a MediaPipe image
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                # VIDEO mode requires a timestamp in milliseconds
                timestamp_ms = int(
                    camera.get(cv2.CAP_PROP_POS_MSEC)
                )

                # Use a steadily increasing timestamp for live camera frames
                if not hasattr(main, "frame_timestamp"):
                    main.frame_timestamp = 0

                main.frame_timestamp += 33
                timestamp_ms = main.frame_timestamp

                # Detect the hand
                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                if result.hand_landmarks:
                    hand = result.hand_landmarks[0]

                    # Draw the hand landmarks
                    height, width = frame.shape[:2]

                    for landmark in hand:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)

                        cv2.circle(
                            frame,
                            (x, y),
                            4,
                            (0, 255, 0),
                            -1
                        )

                    # Build the same 63 features used during training
                    features = []

                    for landmark in hand:
                        features.extend([
                            landmark.x,
                            landmark.y,
                            landmark.z
                        ])

                    features = np.array(
                        features,
                        dtype=np.float32
                    ).reshape(1, -1)

                    # Predict the sign
                    prediction = model.predict(features)[0]

                    # Get prediction confidence when available
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(features)[0]
                        confidence = float(np.max(probabilities))
                    else:
                        confidence = 1.0

                    if confidence >= CONFIDENCE_THRESHOLD:
                        PREDICTION_HISTORY.append(str(prediction))

                        # Display the most frequent recent prediction
                        last_prediction = Counter(
                            PREDICTION_HISTORY
                        ).most_common(1)[0][0]

                        last_confidence = confidence
                    else:
                        PREDICTION_HISTORY.clear()
                        last_prediction = "Uncertain sign"
                        last_confidence = confidence

                else:
                    PREDICTION_HISTORY.clear()
                    last_prediction = "Show your hand"
                    last_confidence = 0.0

                # Display the recognized text below the camera image
                cv2.rectangle(
                    frame,
                    (0, 0),
                    (frame.shape[1], 100),
                    (35, 35, 35),
                    -1
                )

                cv2.putText(
                    frame,
                    "Recognized Sign: " + last_prediction,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Confidence: {last_confidence:.0%}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                cv2.imshow(
                    "AI Sign Language Recognition",
                    frame
                )

                # Press Q to exit
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("\nWebcam recognition stopped.")


if __name__ == "__main__":
    main()