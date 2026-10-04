import pandas as pd
from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parent.parent
DATASET_FILE = PROJECT_PATH / "outputs" / "landmark_dataset.csv"

def main():

    print("Preparing landmark data for machine learning...\n")

    if not DATASET_FILE.exists():
        print("ERROR : Landmark dataset not found.")
        print(DATASET_FILE)
        return

    data = pd.read_csv(DATASET_FILE)

    # Landmark columns

    feature_columns = [
        column for column in data.columns
        if column.startswith("landmark_")
    ]

    print(f"Total rows : {len(data)}")
    print(f"Landmark features : {len(feature_columns)}")

    # Separate features and labels

    X = data[feature_columns]
    y = data["sign"]

    # Keep the original dataset split

    train_data = data[data["subset"] == "train"]
    val_data = data[data["subset"] == "val"]
    test_data = data[data["subset"] == "test"]

    X_train = train_data[feature_columns]
    y_train = train_data["sign"]

    X_val = val_data[feature_columns]
    y_val = val_data["sign"]

    X_test = test_data[feature_columns]
    y_test = test_data["sign"]

    print("\nData split :")
    print(f"Training samples : {len(X_train)}")
    print(f"Validation samples : {len(X_val)}")
    print(f"Testing samples : {len(X_test)}")

    print("\nTraining classes :")
    print(sorted(y_train.unique()))

    print("\nFeature shape :")
    print(f"X_train : {X_train.shape}")
    print(f"X_val :   {X_val.shape}")
    print(f"X_test :  {X_test.shape}")

    print("\nLabel shape :")
    print(f"y_train : {y_train.shape}")
    print(f"y_val :   {y_val.shape}")
    print(f"y_test :  {y_test.shape}")

    print("\nPreparing data completed.")

if __name__ == "__main__":
    main()