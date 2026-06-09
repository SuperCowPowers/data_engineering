# Project 1 — Reading CSV Files with Pandas

A first project: grab a real dataset, load it into a
[pandas](https://pandas.pydata.org/) DataFrame, and poke around.

## The dataset: AQSolDB

[AQSolDB](https://www.kaggle.com/datasets/sorkun/aqsoldb-a-curated-aqueous-solubility-dataset)
is a curated set of ~10,000 compounds with measured **aqueous solubility**
(how well a molecule dissolves in water), plus molecular descriptors. It's a
nice, clean CSV to learn on.

## 1. Get a Kaggle account

Datasets on Kaggle require a (free) login.

1. Sign up at [kaggle.com](https://www.kaggle.com/).
2. Open the dataset page:
   <https://www.kaggle.com/datasets/sorkun/aqsoldb-a-curated-aqueous-solubility-dataset>
3. Click **Download** to grab the zip, unzip it, and you'll get
   `curated-solubility-dataset.csv`.

Put the CSV in a `data/` folder next to this README:

```
project_1/
├── README.md
└── data/
    └── curated-solubility-dataset.csv
```

> **Don't commit the data.** Keep raw data out of git — add a line to the
> repo's `.gitignore`:
> ```bash
> echo 'project_1/data/' >> ../.gitignore
> ```

### (Optional) Download from the command line

If you'd rather script it, the [Kaggle CLI](https://github.com/Kaggle/kaggle-api)
works too. Create an API token (Kaggle → *Account* → *Create New Token*, saves
`kaggle.json` to `~/.kaggle/`), then:

```bash
uv add --dev kaggle
uv run kaggle datasets download -d sorkun/aqsoldb-a-curated-aqueous-solubility-dataset -p data --unzip
```

## 2. Add pandas to the project

From the repo root:

```bash
uv add pandas
```

This adds pandas to `dependencies` in `pyproject.toml` and updates `uv.lock`.

## 3. Read the CSV into a DataFrame

```python
import pandas as pd

df = pd.read_csv("data/curated-solubility-dataset.csv")

print(df.shape)        # (rows, columns)
print(df.head())       # first 5 rows
```

Run it with:

```bash
uv run python read_data.py
```

## 4. Explore the data

A few first moves once it's loaded:

```python
df.columns            # list every column name
df.dtypes             # data type of each column
df.info()             # columns, non-null counts, memory use
df.describe()         # summary stats for numeric columns
df.isna().sum()       # count missing values per column
```

Selecting and filtering:

```python
df["Solubility"]                      # one column (a Series)
df[["Name", "Solubility"]]            # several columns
df[df["Solubility"] > 0]              # rows where solubility is positive
df.sort_values("Solubility").head()   # least-soluble compounds
```

## Handy `read_csv` options

`pd.read_csv` has a flag for almost everything:

```python
pd.read_csv(
    "data/curated-solubility-dataset.csv",
    usecols=["ID", "Name", "SMILES", "Solubility"],  # only the columns you need
    index_col="ID",                                  # use a column as the row index
    nrows=1000,                                       # read just the first N rows
)
```

## Next steps

- Plot the solubility distribution (`df["Solubility"].hist()`).
- Group and aggregate (`df.groupby(...).mean()`).
- Save a cleaned subset back out (`df.to_csv("data/clean.csv", index=False)`).
