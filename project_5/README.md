# Project 5 — Supervised Modeling (XGBoost & PyTorch)

The sequel to Project 3. There you did **unsupervised** learning — you handed the
algorithm penguin measurements with *no labels* and it found structure on its own
(clusters that roughly matched species). Now you'll do **supervised** learning:
give the model the *right answers* during training so it learns to predict them
on data it's never seen.

You'll train **two different kinds of model** — **XGBoost** (gradient-boosted
decision trees) and a **PyTorch** neural network — on **two different kinds of
task**, and put them head-to-head:

|  | What it predicts | Kind of task |
|---|---|---|
| **Classification** | a penguin's **sex** (male / female) | pick a category |
| **Regression** | a penguin's **body mass** (grams) | predict a number |

## The core idea: features → label

Supervised learning splits your columns into two roles:

- **Features** (`X`) — the inputs the model gets to look at (bill length, flipper
  length, species, …).
- **Label / target** (`y`) — the answer you want it to predict (sex, or body mass).

Training means showing the model many `(X, y)` pairs so it learns the mapping.

## 1. Pull the data & sync

```bash
uv run python project_5/data/pull_data.py   # same penguins as Project 3
uv sync                                     # xgboost + torch are in pyproject.toml
```

## 2. The cardinal rule: split into train and test

Here's the single most important idea in all of supervised learning:

> **Never judge a model on the data it trained on.**

A model can *memorize* its training data and look perfect, yet be useless on
anything new. So you hold back a chunk of data the model never sees during
training — the **test set** — and grade it only on that. It's the difference
between "let me re-take the exam I already have the answer key for" and a real
exam.

We do this in a small shared file, **`data_prep.py`** (provided). Both model
scripts import it, so they train and test on the *exact same* split — otherwise
the comparison wouldn't be fair:

```python
import pandas as pd
from sklearn.model_selection import train_test_split

NUM = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

def _load():
    df = pd.read_csv("data/penguins.csv")
    return pd.get_dummies(df, columns=["species"], dtype=float)   # text -> 0/1 columns

def classification_split():                      # predict SEX (1 = male, 0 = female)
    d = _load().dropna(subset=NUM + ["sex"])
    species = [c for c in d.columns if c.startswith("species_")]
    X = d[NUM + species]
    y = (d["sex"].str.upper() == "MALE").astype(int)
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

def regression_split():                          # predict BODY MASS (grams)
    d = _load().dropna(subset=NUM)
    species = [c for c in d.columns if c.startswith("species_")]
    features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm"] + species
    return train_test_split(d[features], d["body_mass_g"], test_size=0.25, random_state=42)
```

Two details worth noticing:
- **Models need numbers, not text.** `pd.get_dummies` turns the `species` column
  into `species_Adelie` / `species_Chinstrap` / `species_Gentoo` (0/1) columns.
- **The target can't be a feature.** In `regression_split`, `body_mass_g` is what
  we're predicting, so it's kept *out* of the feature list.

## 3. How you grade each task

Different tasks need different scorecards:

| Task | Metric | Meaning |
|---|---|---|
| Classification | **accuracy** | fraction predicted correctly |
| Classification | **confusion matrix** | *where* it's wrong (false males vs false females) |
| Regression | **R²** | fraction of the variation explained (1.0 = perfect, 0 = useless) |
| Regression | **MAE** | mean absolute error — average miss, in real units (grams) |

## 4. Model 1 — XGBoost (`xgboost_models.py`)

XGBoost builds an ensemble of decision trees, each one fixing the previous ones'
mistakes. It's the workhorse of tabular data. The lovely part: **you just call
`.fit()`** — no feature scaling, no training loop.

Below is the **skeleton**. The structure and the fiddly bits are given; the
modeling steps are yours to fill in (the `YOUR CODE` gaps). Work out each line
rather than pasting a finished answer — that's the whole point.

```python
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_absolute_error
import data_prep

# ===== Classification: predict sex =====
Xtr, Xte, ytr, yte = data_prep.classification_split()

# YOUR CODE: create an XGBClassifier and fit it on (Xtr, ytr).
#   Reasonable settings: n_estimators=200, max_depth=3, learning_rate=0.1, eval_metric="logloss"
#   (trees don't need scaled features — fit the raw X)
clf = ...

# YOUR CODE: predict on Xte, then print the accuracy and the confusion matrix.
#   -> accuracy_score(yte, pred)   and   confusion_matrix(yte, pred)

# Given for you — which measurements mattered most (ties back to Project 3's PCA):
importance = pd.Series(clf.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
print(importance.round(3).to_string())

# ===== Regression: predict body mass =====
Xtr, Xte, ytr, yte = data_prep.regression_split()

# YOUR CODE: this mirrors the classification block, but with XGBRegressor.
#   1. create + fit an XGBRegressor  (try n_estimators=300, max_depth=3, learning_rate=0.1)
#   2. predict on Xte
#   3. print R^2 (r2_score) and mean absolute error (mean_absolute_error), in grams
```

Run it on its own:

```bash
uv run python xgboost_models.py     # from inside project_5/
```

You should see about **89% accuracy** on sex and **R² ≈ 0.85** on body mass —
and the feature importances will show **body mass and bill depth** carry the most
signal for sex (males are heavier with deeper bills — the model rediscovered
penguin biology).

## 5. Model 2 — PyTorch neural net (`pytorch_models.py`)

Same two tasks, but now a neural network you train yourself. This is more code —
and that's the point of the lesson. A net needs its inputs **scaled**, and it
learns through a **training loop**: predict → measure loss → adjust weights,
repeated for many *epochs*.

The net and the training loop are **given** below — study them, because this is
what "a neural network" and "a training loop" actually look like. Your job is to
wire them up (the `YOUR CODE` gaps): scale the data, train, predict, evaluate.

```python
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import data_prep

torch.manual_seed(42)

# ----- GIVEN: the model and the training loop (read these carefully) -----
def make_net(n_inputs):
    return nn.Sequential(nn.Linear(n_inputs, 16), nn.ReLU(), nn.Linear(16, 1))

def train(net, X, y, epochs, loss_fn):
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    for _ in range(epochs):
        opt.zero_grad()            # reset gradients
        loss = loss_fn(net(X), y)  # how wrong are we?
        loss.backward()            # backprop: compute gradients
        opt.step()                 # nudge the weights to reduce the loss
    return net

# ===== Classification: predict sex =====
Xtr, Xte, ytr, yte = data_prep.classification_split()

# YOUR CODE:
#   1. Fit a StandardScaler on Xtr and use it to scale Xtr / Xte (nets need scaled
#      inputs; the trees didn't).
#   2. net = make_net(Xtr.shape[1]); then call
#      train(net, <scaled Xtr>, ytr.values, epochs=300, loss_fn=nn.BCEWithLogitsLoss())
#   3. Predict: run the net on scaled Xte, apply torch.sigmoid, threshold at 0.5.
#      (do the prediction inside `with torch.no_grad():`)
#   4. print accuracy_score(yte, pred)

# ===== Regression: predict body mass =====
Xtr, Xte, ytr, yte = data_prep.regression_split()

# YOUR CODE: like the classification block, with three differences:
#   - scale the FEATURES *and* the TARGET. Body mass is in the thousands; an
#     unscaled target makes the loss explode. Use a SECOND StandardScaler for y.
#   - train with nn.MSELoss(), epochs=500
#   - after predicting, use the target scaler's .inverse_transform(...) to turn the
#     scaled prediction back into grams, then print R^2 and MAE
```

Run it on its own:

```bash
uv run python pytorch_models.py     # from inside project_5/
```

## 6. Why *two separate scripts*? (a real data-engineering lesson)

You might wonder why XGBoost and PyTorch live in **separate files** instead of one
tidy comparison script. Here's the war story, because it's a genuine
data-engineering lesson:

Both libraries ship their own private copy of **OpenMP**, the library that spreads
work across CPU cores. On macOS, if you `import` *both* into one Python process,
you end up with two OpenMP runtimes both trying to own the CPU threads — and they
**crash the program** (a segfault), even if you run the models one after another.

The important lesson: **the code was correct — the failure was about how the
libraries are packaged, not the math.** And the fix isn't a code change, it's an
*architectural* one: give each library its own process. `xgboost_models.py`
imports xgboost but never torch; `pytorch_models.py` does the reverse. Neither
process ever holds two OpenMP runtimes, so both just work.

> (This particular clash is a macOS + pip/uv-wheels quirk. On Linux, or with a
> conda environment that shares one OpenMP, they coexist fine — but "structure
> your program around your environment's constraints" is exactly the kind of
> thing real data engineers do all day.)

## 7. The head-to-head — and the real lessons

Run both scripts and line up the numbers. You'll see something like:

| Task | XGBoost | PyTorch |
|---|---|---|
| Sex (accuracy) | ~89% | ~92% |
| Body mass (R²) | ~0.85 | ~0.89 |

Two things to take away — and the second is the one most people get wrong:

**a) On small, clean tabular data, they're neck-and-neck.** The neural net edged
ahead here, which is a useful myth-buster: you'll hear "trees always beat neural
nets on tables," and while that's a real *tendency* on big messy datasets, it is
**not a law** — on this small tidy dataset a well-scaled net does great.

**b) A few points of difference is often just NOISE.** The test set is only ~84
penguins. Change the split and the numbers jump around. Try it: in `data_prep.py`
change `random_state=42` to a few other numbers and re-run. When I did this across
5 splits, XGBoost's sex accuracy ranged from **89% to 95%** — a 6-point swing from
*nothing but luck of the draw*. **Never crown a winner from a single split.** Real
evaluation averages over several (that's what "cross-validation" does).

**c) The honest practical difference is effort, not accuracy.** Look back at the
two scripts: XGBoost was `.fit()` and done. The net needed scaled inputs, a
training loop, a learning rate, and a choice of epochs — more knobs, more ways to
get it wrong. **That's** why XGBoost is most people's first reach for tabular
data: not because it always wins, but because it gets you a strong result with far
less that can break. Neural networks earn their keep on **big, unstructured** data
— images, audio, text — which is where a future project would take them.

## 8. The assignment

`data_prep.py` is given. Your job is to complete the two model scripts by filling
in the `YOUR CODE` gaps from Sections 4 and 5:

1. **`xgboost_models.py`** — fill in the classifier and the regressor
   (create → fit → predict → print metrics).
2. **`pytorch_models.py`** — wire up the given net + training loop for both tasks,
   including the target-scaling trick in the regression half.
3. Run each script **separately** and write down the comparison.
4. Change `random_state` in `data_prep.py` to 2–3 other values, re-run, and watch
   how much the "winner" moves — that's §7b in action.

## 9. Stretch goals

- **Cross-validation.** Instead of one split, use `sklearn.model_selection.cross_val_score`
  (for XGBoost) to average over 5 folds — a trustworthy score instead of a noisy one.
- **Tune XGBoost.** Try `max_depth`, `n_estimators`, `learning_rate`. Can you beat
  the net? Does deeper always help, or does it start overfitting?
- **Grow the net.** Add a second hidden layer, or more units. Does it help on 342
  penguins, or just overfit faster?
- **Predict species instead** and watch *both* models hit ~100% — a lesson in why
  an "easy" target teaches you nothing about which model is better.

## What you'll have learned

- The supervised setup — **features vs. labels**, and the cardinal rule of a
  held-out **train/test split**.
- **Classification vs. regression**, and the right scorecard for each (accuracy /
  confusion matrix vs. R² / MAE).
- Two model families: **XGBoost** (`.fit()` and go) and a **PyTorch** neural net
  (scaling + a training loop), and the honest trade-off between them.
- That model comparisons are **noisy** on small data — evaluate over several
  splits before believing a winner.
- A real engineering lesson: when two libraries can't share a process, you fix it
  by **structuring the program**, not by changing the math.
