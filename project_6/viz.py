"""
viz.py - display dog images in a labeled grid. Handy for eyeballing which dogs
the model got wrong (pass the misclassified images and titles like
"pred: Pug / true: Boxer").
"""

import torch
import matplotlib.pyplot as plt

# The ResNet transforms normalize with these ImageNet stats; undo them to display.
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _unnormalize(img):
    img = img.cpu() * _STD + _MEAN
    return img.clamp(0, 1).permute(1, 2, 0).numpy()  # CxHxW tensor -> HxWxC image


def show_grid(images, titles, cols=4):
    """images: list of normalized CxHxW tensors. titles: list of strings."""
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(cols * 2.4, rows * 2.6))
    for i, (img, title) in enumerate(zip(images, titles)):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(_unnormalize(img))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
