import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

from features import normalize_landmarks

# The CSV has many columns; pandas can warn about speed. It is harmless.
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

PROJECT_PATH = Path(__file__).resolve().parent.parent
DATASET_FILE = PROJECT_PATH / "outputs" / "landmark_dataset_2hands.csv"
OWN_FILE = PROJECT_PATH / "outputs" / "my_samples.csv"
MODEL_FOLDER = PROJECT_PATH / "models"
MODEL_FILE = MODEL_FOLDER / "sign_recognition_model.pkl"

# Your own recordings count more, because they match your camera and hands
OWN_SAMPLE_WEIGHT = 3.0
RANDOM_SEED = 42


def assign_own_subsets(own):
    """Split YOUR recordings into train / val / test, one whole recording at a
    time, so frames from one recording never appear in two subsets."""
    own = own.copy()
    rng = random.Random(RANDOM_SEED)
    subset_by_video = {}

    for sign, group in own.groupby("sign"):
        video_ids = sorted(group["video_id"].unique())
        rng.shuffle(video_ids)
        count = len(video_ids)

        test_count = max(1, int(count * 0.20)) if count >= 4 else 0
        val_count = max(1, int(count * 0.15)) if count >= 3 else 0

        for index, video_id in enumerate(video_ids):
            if index < test_count:
                subset_by_video[video_id] = "test"
            elif index < test_count + val_count:
                subset_by_video[video_id] = "val"
            else:
                subset_by_video[video_id] = "train"

    own["subset"] = own["video_id"].map(subset_by_video)
    return own


def main():

    print("Training sign recognition model...\n")

    if not DATASET_FILE.exists():
        print("ERROR : Two-hand landmark dataset not found.")
        print(DATASET_FILE)
        print("Run  python src/build_dataset.py  first.")
        return

    data = pd.read_csv(DATASET_FILE).assign(source="wlasl")

    # Add your own recordings (from collect_samples.py) if they exist
    if OWN_FILE.exists():
        own = assign_own_subsets(pd.read_csv(OWN_FILE)).assign(source="own")
        data = pd.concat([data, own], ignore_index=True)
        print(f"Added your own recordings : {own['video_id'].nunique()} "
              f"recordings, {len(own)} frames")
    else:
        print("No own recordings found (run collect_samples.py to add some).")

    feature_columns = [
        column for column in data.columns
        if column.startswith("landmark_")
    ]

    train_data = data[data["subset"] == "train"]
    val_data = data[data["subset"] == "val"]
    test_data = data[data["subset"] == "test"]

    # Make every hand the same position and size before training
    X_train = normalize_landmarks(train_data[feature_columns].values)
    y_train = train_data["sign"]

    X_val = normalize_landmarks(val_data[feature_columns].values)
    y_val = val_data["sign"]

    X_test = normalize_landmarks(test_data[feature_columns].values)
    y_test = test_data["sign"]

    print(f"\nTraining samples : {len(X_train)} "
          f"({train_data['video_id'].nunique()} videos/recordings)")
    print(f"Validation samples : {len(X_val)} "
          f"({val_data['video_id'].nunique()} videos/recordings)")
    print(f"Testing samples : {len(X_test)} "
          f"({test_data['video_id'].nunique()} videos/recordings)")
    print(f"Features : {X_train.shape[1]}")

    print("\nTraining classes :")
    print(sorted(y_train.unique()))

    print("\nCreating Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    weights = np.where(
        train_data["source"].values == "own", OWN_SAMPLE_WEIGHT, 1.0
    )

    model.fit(X_train, y_train, sample_weight=weights)

    print("Model training completed.")

    print("\nEvaluating model...")

    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)

    print(f"\nValidation accuracy : {val_accuracy:.4f}")

    if len(X_test) > 0:
        test_predictions = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        print(f"Test accuracy : {test_accuracy:.4f}")

        # Separate scores: WLASL videos vs YOUR recordings
        for source, name in [("wlasl", "WLASL videos"), ("own", "Your recordings")]:
            mask = (test_data["source"].values == source)
            if mask.sum() > 0:
                accuracy = accuracy_score(
                    y_test[mask], test_predictions[mask]
                )
                print(f"  Test accuracy on {name} : {accuracy:.4f} "
                      f"({mask.sum()} frames)")

    print("\nClassification report (validation) :")

    print(
        classification_report(
            y_val,
            val_predictions,
            zero_division=0
        )
    )

    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    print("\nModel saved successfully :")
    print(MODEL_FILE)

    print("\nSign recognition model training completed.")

if __name__ == "__main__":
    main()