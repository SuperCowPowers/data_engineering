"""
pull_data.py - download the Palmer Penguins dataset into this folder.

Run it from anywhere (the repo root is fine):

    uv run python project_3/data/pull_data.py

It saves `penguins.csv` right next to this script. Project 3 reuses the same
dataset as Project 2, but now we'll treat the body measurements as points in a
multi-dimensional space, cluster them, and project the result down to 2D.
"""

from pathlib import Path
import urllib.request

# Palmer Penguins, served as a plain CSV from the seaborn-data mirror.
# The four numeric measurement columns (bill length/depth, flipper length,
# body mass) are the dimensions we'll cluster in.
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"

# Save the file next to THIS script (not wherever you happened to run it from),
# so it always lands in project_3/data/.
DEST = Path(__file__).parent / "penguins.csv"


def main() -> None:
    print("Downloading the Palmer Penguins dataset...")
    print(f"  from: {DATA_URL}")
    print(f"  to:   {DEST}")

    urllib.request.urlretrieve(DATA_URL, DEST)

    size_kb = DEST.stat().st_size / 1024
    print(f"Done - saved {size_kb:.0f} KB.")

    # Optional sanity check: if pandas is installed, report the shape.
    try:
        import pandas as pd

        df = pd.read_csv(DEST)
        print(f"Loaded OK: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"Columns: {', '.join(df.columns)}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
