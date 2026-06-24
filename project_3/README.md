# Project 3 — Clustering & Dimensionality Reduction

In Project 2 you *described* the penguins and hunted for outliers. Now you'll do
something more powerful: hand the raw measurements to an algorithm and let it
**discover the groups on its own** — no species labels, no supervision. Then
you'll take those groups, living in 4-dimensional measurement space, and
**squash them down to 2D so you can actually see them**.

> **The shape of this project is a pipeline of DataFrame operations, where each
> step *adds a column*.** You start with the raw measurements; clustering adds a
> `cluster_id`; the 2D projection adds an `x` and a `y`. The DataFrame grows as
> the analysis progresses, and at the end you just plot the columns you've
> accumulated. That column-growing pipeline *is* the project:
>
> ```python
> df_with_clusters = cluster_data(df, k=3)          # adds: cluster_id
> df_projected     = dimensional_reduction(df_with_clusters)   # adds: x, y
> plot_clusters(df_projected)                        # uses: x, y, cluster_id
> ```

## The two ideas

**Clustering** — given points in space, find natural groups. We'll use
**k-means**, which you tell "there are `k` groups" and it finds the `k` cluster
centers that best explain the data. Our "space" is 4-dimensional: each penguin
is a point with coordinates (bill length, bill depth, flipper length, body
mass).

**Dimensionality reduction** — you can't plot 4 dimensions on a screen. **PCA
(Principal Component Analysis)** finds the 2 directions through the 4D cloud
that capture the most spread, and projects every point onto them. You lose some
information, but on this data those 2 directions keep about **88%** of it — so
the flattened picture is honest. PCA is the *classic, linear* method; in Step 2
you'll also meet **t-SNE**, a *modern, non-linear* projection that tends to
separate groups more dramatically.

## The dataset

Same Palmer Penguins as Project 2 — 344 penguins, but now we care about the four
numeric measurement columns as a single 4D point per penguin:

```
bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g
```

## 1. Pull the data

```bash
uv run python project_3/data/pull_data.py
```

That drops `penguins.csv` into `project_3/data/`.

## 2. Dependencies

This project uses [scikit-learn](https://scikit-learn.org/) (for k-means and
PCA) alongside pandas and matplotlib. They're all declared in the repo's
`pyproject.toml`, so from the repo root just:

```bash
uv sync
```

## 3. The mental model: a DataFrame that grows

Keep this table in your head — it's the whole project. Each function takes a
DataFrame and returns a new one with extra columns:

| Stage | Call | Columns it adds |
|---|---|---|
| raw | `pd.read_csv(...)` | the 7 original columns |
| cluster | `cluster_data(df, k=3)` | **`cluster_id`** |
| project | `dimensional_reduction(df)` | **`x`, `y`** |
| plot | `plot_clusters(df)` | *(nothing — it reads `x`, `y`, `cluster_id`)* |

## 4. Step 1 — cluster in the native (4D) space

### Watch out: scale your features first

k-means measures distance between points. Body mass is in the **thousands** of
grams; bill length is in the **tens** of millimeters. If you cluster the raw
numbers, body mass completely dominates the distance and the other three
measurements barely count. The fix is to **standardize** every feature to a
comparable scale (mean 0, standard deviation 1) before clustering:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

clean = df.dropna(subset=features).copy()   # a couple of penguins are missing measurements
X = StandardScaler().fit_transform(clean[features])

labels = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X)
clean["cluster_id"] = labels
```

A few things worth understanding:

- **`dropna`**: 2 of the 344 penguins have missing measurements; drop them so
  the math doesn't choke. You'll have **342 rows**.
- **`random_state=42`**: k-means starts from random guesses, so pin the seed to
  get the same answer every run (reproducibility).
- **`cluster_id` is an arbitrary label.** k-means returns `0`, `1`, `2` — these
  are *not* species. Cluster `0` isn't "Adélie"; it's just "the first group the
  algorithm happened to number 0." The algorithm finds groups; *you* interpret
  them.

### Did it rediscover the species?

Here's the satisfying part. We never told k-means about species — but compare
its clusters to the real species with a crosstab:

```python
import pandas as pd
print(pd.crosstab(clean["cluster_id"], clean["species"]))
```

You'll see that **Gentoo penguins fall almost perfectly into their own
cluster**, while **Adélie and Chinstrap overlap** more (they're more similar in
body measurements). That's a real result: clustering recovered most of the
biological structure from measurements alone — but not perfectly, and the messy
part is itself informative.

### Don't want to guess `k`? Use HDBSCAN

k-means has a catch we glossed over: **you have to tell it how many clusters to
look for.** We picked `k=3` because we happen to know there are 3 species — but
in real clustering you don't know the answer in advance. That's cheating.

**HDBSCAN** is a modern, *density-based* clustering algorithm built right into
scikit-learn. You don't give it a number of clusters — it finds however many
dense groups actually exist, and labels leftover points in the sparse gaps as
**noise** (`cluster_id == -1`). All it needs is a `min_cluster_size`:

```python
from sklearn.cluster import HDBSCAN

# same scaled X from Step 1
labels = HDBSCAN(min_cluster_size=15).fit_predict(X)
clean["cluster_id"] = labels
# labels of -1 mean "noise" — points HDBSCAN refused to assign to any cluster
```

Here's the fascinating part: run it on the penguins and **HDBSCAN finds only 2
clusters, not 3** (with a handful of noise points). It looked at the geometry
and decided Adélie and Chinstrap aren't separated by a real density gap, so it
merged them — and split Gentoo off on its own. k-means gave you 3 clusters only
because *you demanded 3*. HDBSCAN, asked nothing, says the data really has 2
well-separated blobs. Neither is "wrong" — they're answering different
questions, and comparing them is the whole lesson.

> (On `sklearn 1.9`, `HDBSCAN` prints a harmless `FutureWarning` about a `copy`
> default changing in a future version — pass `copy=True` to silence it.)

There's also **DBSCAN**, the older cousin. It also doesn't take `k`, but it
needs an `eps` (a neighborhood radius) that's notoriously finicky to tune —
HDBSCAN exists precisely to remove that headache, so prefer it.

## 5. Step 2 — project 4D down to 2D

Now collapse the four measurements into two coordinates you can plot, using PCA:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
coords = pca.fit_transform(X)         # X = the SAME scaled features from Step 1
clean["x"] = coords[:, 0]
clean["y"] = coords[:, 1]

print(pca.explained_variance_ratio_)  # how much spread each new axis captures
```

On this data the two components keep about **69%** and **19%** of the variance —
roughly **88% together**. That number matters: it tells you the 2D picture isn't
throwing away the story. (If it were 40%, you'd distrust the plot.)

> Note the `x` / `y` axes are no longer "millimeters" or "grams." They're
> *combinations* of all four measurements — abstract directions through the
> cloud. That's the trade: you gain a 2D view, you lose the original units.

### A modern alternative: t-SNE

PCA is linear — it can only rotate and flatten the cloud. **t-SNE**
(`sklearn.manifold.TSNE`) is a *non-linear* projection that can bend space to
pull apart groups that PCA leaves overlapping. It's the modern default for
*visualizing* clusters, and it's a drop-in swap for the projection step:

```python
from sklearn.manifold import TSNE

coords = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)
clean["x"] = coords[:, 0]
clean["y"] = coords[:, 1]
```

Run it and the clusters usually look crisper than under PCA. But understand the
trade-offs before you trust the prettier picture:

- **No "variance explained."** t-SNE gives you no honesty number like PCA does.
- **It's stochastic** — different `random_state` gives different-looking maps.
- **Global geometry is meaningless.** Distances *between* clusters and the
  *sizes* of clusters in a t-SNE plot don't mean anything; only "these points
  are near each other" is trustworthy. PCA's axes, by contrast, are real
  directions you can interpret.
- **`perplexity`** (roughly, how many neighbors each point considers) is a knob
  you have to tune; try values from 5 to 50.

Rule of thumb: **PCA to understand and sanity-check, t-SNE to present.** And the
genuinely most-modern option, **UMAP**, isn't in scikit-learn at all — it's the
separate `umap-learn` package (a stretch goal below).

## 6. Step 3 — plot it, colored by cluster

```python
import matplotlib.pyplot as plt

plt.scatter(clean["x"], clean["y"], c=clean["cluster_id"], cmap="viridis", s=20)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("Penguins clustered in 4D, projected to 2D")
plt.show()
```

You should see three blobs of color — your clusters, found in 4D, now visible in
2D.

## 7. The assignment

Write `project_3/analysis.py` built around two functions that each **add columns
to a DataFrame and return it**:

1. `cluster_data(df, method="kmeans", k=3)` — standardize the four measurement
   columns, cluster them, and return a copy of `df` with a new `cluster_id`
   column. Support both `method="kmeans"` (uses `k`) and `method="hdbscan"`
   (ignores `k`, finds its own number of groups, may produce `-1` noise).
2. `dimensional_reduction(df, method="pca")` — project the same four (scaled)
   columns to 2D and return a copy with new `x` and `y` columns. Support
   `method="pca"` (and print the explained-variance ratio) and `method="tsne"`.
3. `plot_clusters(df)` — scatter `x` vs `y`, colored by `cluster_id`.

Then wire them into the pipeline — and notice you can swap any stage without
touching the others, because they only communicate through columns:

```python
df = pd.read_csv("data/penguins.csv")
df = cluster_data(df, method="kmeans", k=3)   # try method="hdbscan" too
df = dimensional_reduction(df, method="pca")  # try method="tsne" too
plot_clusters(df)
```

And answer, in a comment or printout: **does `cluster_id` line up with
`species`?** (Use a crosstab.) Then compare: does **k-means (k=3)** or
**HDBSCAN** agree better with the real species — and does that even mean one is
"better"?

## 8. Stretch goals

- **Make k-means pick `k` itself.** HDBSCAN finds its own cluster count, but
  k-means can be coaxed there too: run `k = 2..8` and plot the **inertia** (the
  "elbow method") or the **silhouette score**, and see whether the data votes
  for 2 (like HDBSCAN) or 3 (like the species).
- **Color by species instead of cluster** in a second plot, side-by-side with
  the cluster-colored one. Where do the clusters and the real species disagree?
- **Try UMAP**, the genuinely most-modern projection — it's *not* in
  scikit-learn, so you'll `uv add umap-learn` and `from umap import UMAP`.
  Compare its 2D map to PCA and t-SNE. Does it separate the groups better, and
  how would you even know?
- **Add `sex` or `island`** to the feature set (you'll have to convert text to
  numbers first — look up one-hot encoding). Does the clustering change?

## What you'll have learned

- How to cluster points in their native high-dimensional space — with **k-means**
  (you choose `k`) and **HDBSCAN** (it chooses for you) — and why **feature
  scaling** matters before you measure distance.
- How to project high-dimensional data to 2D with **PCA** (linear, interpretable,
  with an explained-variance honesty check) and **t-SNE** (modern, non-linear,
  great for presentation but not for measuring).
- The pipeline mindset: an analysis as a sequence of **DataFrame transforms**,
  each adding the columns the next step needs — and how cleanly that lets you
  swap one algorithm for another.
