import pandas as pd
from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parent.parent
DATASET_FILE = PROJECT_PATH / "outputs" / "landmark_dataset.csv"

def main():

    print("Checking the class coverage...\n")

    if not DATASET_FILE.exists():
        print("ERROR : Dataset file not found.")
        print(DATASET_FILE)
        return

    data = pd.read_csv(DATASET_FILE)

    print("Classes in complete dataset :")
    print(sorted(data["sign"].unique()))

    print("\nClass distribution by subset :\n")

    coverage = pd.crosstab(
        data["sign"],
        data["subset"]
    )

    print(coverage)

    print("\nTraining classes :")
    train_classes = set(
        data[data["subset"] == "train"]["sign"]
    )
    print(sorted(train_classes))

    print("\nValidation classes :")
    val_classes = set(
        data[data["subset"] == "val"]["sign"]
    )
    print(sorted(val_classes))

    print("\nTest classes :")
    test_classes = set(
        data[data["subset"] == "test"]["sign"]
    )
    print(sorted(test_classes))

    print("\nClasses missing from training :")

    missing_from_training = sorted(
        set(data["sign"].unique()) - train_classes
    )

    if missing_from_training:
        for sign in missing_from_training:
            print(f"- {sign}")
    else:
        print("None")

    print("\nClass coverage check completed.")

if __name__ == "__main__":
    main()