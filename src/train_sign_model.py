import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

from features import normalize_landmarks

PROJECT_PATH = Path(__file__).resolve().parent.parent
DATASET_FILE = PROJECT_PATH / "outputs" / "landmark_dataset.csv"
MODEL_FOLDER = PROJECT_PATH / "models"
MODEL_FILE = MODEL_FOLDER / "sign_recognition_model.pkl"


def main():

    print("Training sign recognition model...\n")

    if not DATASET_FILE.exists():
        print("ERROR : Landmark dataset not found.")
        print(DATASET_FILE)
        return

    data = pd.read_csv(DATASET_FILE)

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

    print(f"Training samples : {len(X_train)}")
    print(f"Validation samples : {len(X_val)}")
    print(f"Testing samples : {len(X_test)}")
    print(f"Features : {len(feature_columns)}")

    print("\nTraining classes :")
    print(sorted(y_train.unique()))

    print("\nCreating Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("Model training completed.")

    print("\nEvaluating model...")

    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)

    print(f"\nValidation accuracy : {val_accuracy:.4f}")

    if len(X_test) > 0:
        test_predictions = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        print(f"Test accuracy : {test_accuracy:.4f}")

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