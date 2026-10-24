import pandas as pd
from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parent.parent
DATASET_FILE = PROJECT_PATH / "outputs" / "landmark_dataset.csv"


def main():
    print("Checking class coverage...\n")

    if not DATASET_FILE.exists():
        print("ERROR : Dataset file not found.")
        return

    data = pd.read_csv(DATASET_FILE)

    videos = data[["video_id", "sign", "subset"]].drop_duplicates()

    print("Videos per sign and subset :")
    print(
        pd.crosstab(
            videos["sign"],
            videos["subset"]
        ).to_string()
    )

    train_classes = set(videos.loc[videos["subset"] == "train", "sign"])
    val_classes = set(videos.loc[videos["subset"] == "val", "sign"])
    test_classes = set(videos.loc[videos["subset"] == "test", "sign"])
    all_classes = set(videos["sign"])

    print("\nSigns missing from training :", sorted(all_classes - train_classes))
    print("Signs missing from validation :", sorted(all_classes - val_classes))
    print("Signs missing from testing :", sorted(all_classes - test_classes))

    print("\nClass coverage check completed.")

if __name__ == "__main__":
    main()