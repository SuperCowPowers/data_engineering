import sys
import torch
from PIL import Image
from data_setup import BREEDS, TRANSFORM

path = sys.argv[1]     # the photo to classify: uv run python predict.py my_dog.jpg

# Load the saved model and switch to eval() — identical to evaluate.py (given).
model = torch.load("dog_model.pt", map_location="cpu", weights_only=False)
model.eval()

# ===== TODO 1: turn ONE image file into a model-ready batch =====
# evaluate.py got ready-made batches from the DataLoader. Here the "batch" is the
# single new photo you supply, so you build it yourself: open the file, apply the
# SAME preprocessing the model was trained with (TRANSFORM), then add a batch
# dimension of 1 so the shape matches what the model expects.
#     image = Image.open(path).convert("RGB")
#     x = TRANSFORM(image).unsqueeze(0)          # shape (1, 3, 224, 224): a batch of one

image = Image.open(path).convert("RGB")
x = TRANSFORM(image).unsqueeze(0)

# ===== TODO 2: run inference and show every breed's probability =====
# Same no-gradients block as evaluate.py — you're predicting, not training. But
# evaluate.py only needed argmax (the single top guess) to score accuracy; here
# there's no answer key, so show the FULL distribution with softmax and rank it.
#     with torch.no_grad():
#         probs = torch.softmax(model(x), dim=1)[0]      # one probability per breed
#     ranked = sorted(zip(BREEDS, probs.tolist()), key=lambda t: t[1], reverse=True)
#     print(f"\nPrediction for {path}:")
#     for breed, p in ranked:
#         print(f"  {breed:10s} {p:6.1%}  {'#' * round(p * 30)}")
#     print(f"\n=> {ranked[0][0]}")

with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)[0]
    ranked = sorted(zip(BREEDS, probs.tolist()), key=lambda t: t[1], reverse=True)
    print(f"\nPrediction for {path}:")
    for breed, p in ranked:
        print(f"  {breed:10s} {p:6.1%}  {'#' * round(p * 30)}")
    print(f"\n=> {ranked[0][0]}")