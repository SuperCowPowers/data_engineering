"""
pull_data.py - download the Palmer Penguins dataset into this folder.

Run it from anywhere (the repo root is fine):

    uv run python project_5/data/pull_data.py

Project 5 reuses the same penguins as Project 3, but now we do *supervised*
learning: train models to predict a label (a penguin's sex, or its body mass)
from the other measurements.
"""

from pathlib import Path
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
DEST = Path(__file__).parent / "penguins.csv"


def main() -> None:
    print("Downloading the Palmer Penguins dataset...")
    print(f"  from: {DATA_URL}")
    print(f"  to:   {DEST}")

    urllib.request.urlretrieve(DATA_URL, DEST)

    size_kb = DEST.stat().st_size / 1024
    print(f"Done - saved {size_kb:.0f} KB.")

    try:
        import pandas as pd

        df = pd.read_csv(DEST)
        print(f"Loaded OK: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"Columns: {', '.join(df.columns)}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
