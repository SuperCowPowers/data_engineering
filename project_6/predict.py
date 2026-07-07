"""
predict.py - classify a single dog photo with the trained model.

    uv run python predict.py path/to/your_dog.jpg

Run classifier.py first to create dog_model.pt. This prints a probability for
every breed (a text preview of what Project 7 will show as a real bar chart).
"""

import sys
import torch
from PIL import Image

from data_setup import BREEDS, TRANSFORM


def main(path):
    # weights_only=False because we saved the whole model object, not just weights.
    model = torch.load("dog_model.pt", map_location="cpu", weights_only=False)
    model.eval()

    image = Image.open(path).convert("RGB")
    x = TRANSFORM(image).unsqueeze(0)  # add a batch dimension of 1

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]  # a probability per breed

    ranked = sorted(zip(BREEDS, probs.tolist()), key=lambda t: t[1], reverse=True)
    print(f"\nPrediction for {path}:")
    for breed, p in ranked:
        print(f"  {breed:10s} {p:6.1%}  {'#' * round(p * 30)}")
    print(f"\n=> {ranked[0][0]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: uv run python predict.py path/to/dog.jpg")
        sys.exit(1)
    main(sys.argv[1])
