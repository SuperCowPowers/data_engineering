"""
pull_data.py - download the Palmer Penguins dataset into this folder.

Run it from anywhere (the repo root is fine):

    uv run python project_2/data/pull_data.py

It saves `penguins.csv` right next to this script, so your analysis code can
load it with a simple relative path.
"""

from pathlib import Path
import urllib.request

# Palmer Penguins, served as a plain CSV from the seaborn-data mirror.
# 344 penguins of 3 species, with body measurements. It has a few genuine
# outliers AND some missing values - which is exactly what makes it a good
# dataset to practice statistics and outlier detection on.
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"

# Save the file next to THIS script (not wherever you happened to run it from),
# so it always lands in project_2/data/.
DEST = Path(__file__).parent / "penguins.csv"


def main() -> None:
    print("Downloading the Palmer Penguins dataset...")
    print(f"  from: {DATA_URL}")
    print(f"  to:   {DEST}")

    urllib.request.urlretrieve(DATA_URL, DEST)

    size_kb = DEST.stat().st_size / 1024
    print(f"Done - saved {size_kb:.0f} KB.")

    # Optional sanity check: if pandas is installed, report the shape so you
    # can see the download worked before you start the real analysis.
    try:
        import pandas as pd

        df = pd.read_csv(DEST)
        print(f"Loaded OK: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"Columns: {', '.join(df.columns)}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
