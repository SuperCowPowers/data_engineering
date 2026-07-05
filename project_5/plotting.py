"""
plotting.py - a small visualization helper for Project 5.

`plot_predictions` draws the classic regression diagnostic: predicted vs actual,
with a diagonal "perfect prediction" line. A point sitting on the line was
predicted exactly right; how far the cloud of points scatters off the line is the
model's error, made visible. A good model hugs the diagonal.

Only uses matplotlib, so it's safe for either model script to import.
"""

import matplotlib.pyplot as plt


def plot_predictions(y_true, y_pred, title="Predicted vs actual"):
    """Scatter predicted vs actual body mass, with the y = x reference line."""
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="black", linewidth=0.3)

    # the diagonal: where predicted == actual. A perfect model lands every point here.
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", label="perfect prediction")

    plt.xlabel("actual body mass (g)")
    plt.ylabel("predicted body mass (g)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
