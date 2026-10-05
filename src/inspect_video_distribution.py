import pandas as pd
from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parent.parent
DATASET_FILE = PROJECT_PATH / "outputs" / "landmark_dataset.csv"

def main():

    print("Inspecting video distribution...\n")

    if not DATASET_FILE.exists():
        print("ERROR : Dataset file not found.")
        print(DATASET_FILE)
        return

    data = pd.read_csv(DATASET_FILE)

    video_data = data[
        ["video_id", "sign", "subset"]
    ].drop_duplicates()

    print(
        f"Total unique videos : "
        f"{len(video_data)}"
    )

    print("\nUnique videos by subset :")

    subset_counts = (
        video_data["subset"]
        .value_counts()
        .sort_index()
    )

    print(subset_counts)

    print("\nUnique videos by sign :")

    sign_counts = (
        video_data.groupby("sign")["video_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    print(sign_counts)

    print("\nUnique videos by sign and subset :")

    distribution = pd.crosstab(
        video_data["sign"],
        video_data["subset"]
    )

    print(distribution)

    print("\nVideo distribution inspection completed.")

if __name__ == "__main__":
    main()