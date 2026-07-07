# Project 6 — Dog Breed Image Classifier (CNN + Transfer Learning)

This is the project that cashes in Project 5's promise. Remember the honest
lesson there: on small *tabular* data, trees tie or beat neural nets — deep
learning's real edge is **big, unstructured data**. Dog photos are exactly that,
and here a neural net isn't just better, it's the *only* sensible tool. You'll
build a model that looks at a photo and names the breed.

The catch: training an image model *from scratch* needs millions of images and a
datacenter. So we do what real practitioners do — **transfer learning**.

## The big idea: transfer learning

> Take a CNN already trained on millions of images (ImageNet). It *already* knows
> edges, fur, ears, snouts, shapes. **Freeze all of that, throw away only its
> final layer, and train a fresh final layer for our 5 dog breeds.**

<p align="center">
  <img src="images/transfer_learning.svg" width="720"
       alt="A dog photo enters a frozen pretrained ResNet-18 backbone (weights reused, never trained), which feeds a new trainable head that outputs 5 breed probabilities.">
</p>

You reuse the pretrained network's "eyes" and only teach it the last step —
"given everything you already see, which of *these 5 breeds* is it?" That's why
this trains in a couple of minutes on a laptop instead of needing a server farm.

The backbone isn't special to dogs — it's a general-purpose *feature extractor*.
That's the whole reason freezing works: you keep the reusable vision and only
swap in a small head for your particular task.

<p align="center">
  <img src="images/backbone_reuse.svg" width="660"
       alt="The same frozen backbone can feed its original 1000-category ImageNet head (which we remove) or a new 5-breed head that we train.">
</p>

We'll use **ResNet18**, a well-known pretrained CNN that ships with torchvision.

## Our 5 breeds

**Beagle, Boxer, Pug, Samoyed, Shiba Inu** — visually distinct, so results are
high and the confusion matrix is easy to read. (All five live in the Oxford-IIIT
Pet dataset.)

## 1. Pull the data & sync

```bash
uv sync                                    # torch + torchvision are in pyproject.toml
uv run python -c "import data_setup; data_setup.train_loader()"   # triggers the download
```

The first run downloads the Oxford-IIIT Pet dataset (~800 MB) into
`project_6/data/` (git-ignored), then keeps just our 5 breeds. All the download +
filtering + image preprocessing lives in the provided **`data_setup.py`** — read
it, but you don't write it.

> **Run the scripts from inside `project_6/`** (they use relative paths and
> `import data_setup`).

## 2. Images as data, and why a CNN

A tabular row was a handful of numbers. An image is a **grid of pixels** — for us,
3 color channels × 224 × 224 ≈ 150,000 numbers per photo. You *could* flatten that
into a plain neural net (like Project 5's), but you'd throw away all the spatial
structure — that an ear is next to a head, that fur has texture.

A **CNN (convolutional neural network)** instead slides small filters across the
image looking for *local* patterns, and stacks them: edges → textures → parts →
whole dogs. That's why CNNs dominate images. The good news: ResNet already *is*
that CNN, fully trained — you just reuse it.

## 3. The provided plumbing

- **`data_setup.py`** — `train_loader()`, `test_loader()`, and `BREEDS`. It
  applies the *exact* preprocessing ResNet expects (resize → crop 224 →
  normalize), which is what lets us reuse the pretrained network.
- **`viz.py`** — `show_grid(images, titles)` to display dogs in a labeled grid
  (great for seeing mistakes).
- **`predict.py`** — classify a single photo (Section 6).

You write two scripts — **`train_classifier.py`** and **`evaluate.py`** — from the
skeletons below.

## 4. Write `train_classifier.py`

Same scaffolding rule as Project 5: some of the structure is given, and each
`TODO` is yours. You'll build the transfer model in three small steps — **load** a
pretrained network, **freeze** it, **swap** its final layer — then **train** the
new head. (Most TODOs are only a line or two.)

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import data_setup

# Apple Silicon GPU if available, else CPU (given).
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("training on:", device)

train_loader = data_setup.train_loader()
num_classes = len(data_setup.BREEDS)          # 5

# ===== TODO 1: load the pretrained network =====
# ResNet18, already trained on ImageNet — it arrives knowing how to "see".
#     model = resnet18(weights=ResNet18_Weights.DEFAULT)
model = ...

# ===== TODO 2: freeze the backbone =====
# `requires_grad = False` tells PyTorch NOT to compute gradients for these weights.
# No gradients means the optimizer can't nudge them — they're "frozen" and reused
# exactly as they came from ImageNet.
#     for p in model.parameters():
#         p.requires_grad = False

# ===== TODO 3: swap in a new head =====
# `fc` is ResNet's final "fully connected" layer — the classifier that sits on top
# of the features (a plain `nn.Linear`). The pretrained one has 1000 outputs
# (ImageNet's categories); replace it with a fresh Linear that outputs num_classes.
# A brand-new layer has requires_grad = True by default, so this head is the ONLY
# part that trains.
#     model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

# Keep the frozen backbone in eval() mode while training, so its
# BatchNorm stays fixed on the pretrained stats. (See "Why eval()?" below.)
model.eval()

# ===== TODO 4: train the new head =====
# Same shape as Project 5's training loop, with three image tweaks:
#   - optimize ONLY the head:   opt = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
#   - loss is nn.CrossEntropyLoss()   (multi-class)
#   - move each batch to the device:  xb, yb = xb.to(device), yb.to(device)
#   - 5 epochs is plenty; print the loss each epoch and watch it fall
#     (zero_grad -> forward -> loss -> backward -> step)

# ===== Save the trained model (given) =====
torch.save(model, "dog_model.pt")     # evaluate.py and Project 7 both load this
print("saved dog_model.pt")
```

> **Why `eval()` during training?** "Freezing" the backbone (the blue part of the
> diagrams above) means its weights don't change — and `eval()` extends that to its
> **BatchNorm** layers, keeping them on the pretrained ImageNet statistics instead
> of recomputing them from our few hundred images. It only touches BatchNorm/Dropout,
> not your head's learning. (Plain `train()` mode also works, ~1% lower here; `eval()`
> is the standard, slightly-steadier choice for a frozen feature extractor.)

Run it (from inside `project_6/`):

```bash
uv run python train_classifier.py
```

The loss should drop fast, and you'll get a **`dog_model.pt`** file — your trained
model, saved as an artifact. (Training is quick: only the tiny head is learning.)

## 5. Write `evaluate.py`

A separate script that **loads** the model you just saved and grades it on the
test set — the flip side of "a model is an artifact you reload."

```python
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
import data_setup
import viz

device = "mps" if torch.backends.mps.is_available() else "cpu"
test_loader = data_setup.test_loader()
BREEDS = data_setup.BREEDS

# Load the model train_classifier.py saved (given).
model = torch.load("dog_model.pt", map_location=device, weights_only=False)
model.eval()

# ===== TODO 1: accuracy + confusion matrix =====
# Collect the model's guess for every test image, then score them. The new piece is
# `with torch.no_grad():` — a *context manager*. Everything you indent beneath the
# `with` line runs as one block with gradient tracking switched off (the callout
# below says why). Type it out and watch the indentation:
#
#     preds, trues = [], []
#     with torch.no_grad():                       # opens the "no-gradients" block
#         for images, labels in test_loader:      # indented -> runs inside the block
#             outputs = model(images.to(device))  # one score per breed, per image
#             preds += outputs.argmax(1).cpu().tolist()   # index of the top score
#             trues += labels.tolist()                    # the correct answers
#     # dedent back to here and the block is closed again
#
#     print("accuracy:", accuracy_score(trues, preds))
#     print(confusion_matrix(trues, preds))       # rows = true breed, cols = predicted

# ===== TODO 2: see what it got wrong (fun) =====
# Gather a few test images where prediction != true label and show them:
#     viz.show_grid(images, [f"pred: {BREEDS[p]} / true: {BREEDS[t]}" for ...])
```

> **What's `with torch.no_grad()`?** While training, PyTorch quietly records every
> operation so it can compute gradients for backprop. At evaluation you're only
> *predicting* — no learning happens — so all that bookkeeping is wasted effort.
> `with torch.no_grad():` switches gradient tracking off for everything inside the
> block: it uses less memory, runs a bit faster, and clearly signals "just run the
> model, don't train it."

Run it (from inside `project_6/`):

```bash
uv run python evaluate.py
```

**What you should see:** test accuracy around **~99%**, with a nearly diagonal
confusion matrix. The main slip is Pug↔Boxer (a few Pugs get called Boxers);
everything else is near-perfect. Five distinct breeds + a pretrained network is an
easy, satisfying win — that's the power of transfer learning.

### Reading the confusion matrix: accuracy, precision, recall

The confusion matrix is a grid — **rows are the true breed, columns are what the
model predicted.** The diagonal is where they agree (correct); everything off the
diagonal is a mistake, and *which* off-diagonal cell tells you *which* breeds got
confused. Ours looks about like this:

```
              predicted →
              Beagle  Boxer  Pug  Samoyed  Shiba
true  Beagle    99      1     0      0       0
      Boxer      0     99     0      0       0
      Pug        0      3    97      0       0
      Samoyed    0      0     0     99       1
      Shiba      0      0     0      1      99
```

Three scores, all read straight off the grid:

- **Accuracy** — overall fraction correct: the diagonal divided by the total
  (here ≈ 493/499 ≈ **99%**). One number for the whole model. Handy, but it can
  hide a single breed the model is quietly bad at.
- **Recall** (for one breed) — of all the *real* photos of that breed, how many did
  the model catch? Read **across its row**: `diagonal ÷ row total`. Pug's recall is
  97/100 = **97%** — it *missed* 3 pugs (calling them Boxers).
- **Precision** (for one breed) — when the model *says* a breed, how often is it
  right? Read **down its column**: `diagonal ÷ column total`. Boxer's precision is
  99/103 = **96%** — 4 non-Boxers (3 Pugs + 1 Beagle) got *mislabeled* Boxer.

Notice the pattern in that Pug↔Boxer mix-up: it shows up as **Pug having lower
recall** (it misses some) *and* **Boxer having lower precision** (it over-claims) —
the same mistake seen from two sides. A miss for one breed is a false alarm for
another. Accuracy alone would've hidden which breed was the weak spot; precision
and recall pin it down.

> `sklearn` computes all of these at once — add
> `from sklearn.metrics import classification_report` and
> `print(classification_report(trues, preds, target_names=BREEDS))`.

## 6. Classify your own dog 🐕

The reward. `predict.py` (provided) loads your saved model and classifies any
photo:

```bash
uv run python predict.py path/to/your_dog.jpg
```

It prints a probability for *every* breed, e.g.:

```
Pug         97.8%  #############################
Boxer        1.3%
Shiba Inu    0.6%
Samoyed      0.2%
Beagle       0.1%
=> Pug
```

Point it at a photo of your own dog! It only *knows* these 5 breeds, so a dog
that's none of them gets sorted into the nearest look-alike — a French bulldog
will lean **Pug**, a husky will lean **Shiba Inu**. That "spread the probability
over similar breeds" behavior is exactly what Project 7 turns into a live chart.

## 7. The assignment

1. Complete **`train_classifier.py`** — the freeze + swap-the-final-layer step is
   the whole lesson; make sure you understand *why* only the new layer learns.
2. Complete **`evaluate.py`** — print the confusion matrix and **look at a grid of
   the dogs it got wrong**.
3. Run `predict.py` on a photo of your own dog (or any dog off the internet).

## 8. Stretch goals

- **Add more breeds.** Edit `BREEDS` in `data_setup.py` (Oxford Pets has ~25 dog
  breeds — try `data_setup` breeds you like). Does accuracy drop? Which breeds get
  confused, and do they *look* alike?
- **Fine-tune, don't just freeze.** Unfreeze the last block of the ResNet (set
  `requires_grad = True` on `model.layer4`) and train with a small learning rate.
  Does it help on so few images, or overfit?
- **Data augmentation.** Swap in random flips/crops for the training transform —
  more variety from the same photos.
- **Confidence.** Print the model's probability for each prediction; find the dogs
  it was *least* sure about and see if they're genuinely ambiguous.

## What you'll have learned

- What a **CNN** is and why it beats a plain net on images.
- **Transfer learning** — freeze a pretrained network, retrain only the final
  layer — and why that's how image classification is really done.
- Training with **DataLoaders**, batches, and the **Apple Silicon GPU (MPS)**.
- A couple of practical details: running a frozen backbone in **`eval()` mode** so
  its BatchNorm uses pretrained statistics, and that a **saved model is an
  artifact** you reload later (here, for Project 7).
