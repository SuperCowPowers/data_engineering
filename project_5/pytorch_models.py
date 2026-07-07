import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from plotting import plot_predictions
import data_prep

torch.manual_seed(42)


# ----- GIVEN: the model and the training loop (read these carefully) -----
def make_net(n_inputs):
    return nn.Sequential(
        nn.Linear(n_inputs, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )


def train(net, X, y, epochs, loss_fn):
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    for _ in range(epochs):
        opt.zero_grad()  # reset gradients
        loss = loss_fn(net(X), y)  # how wrong are we?
        loss.backward()  # backprop: compute gradients
        opt.step()  # nudge the weights to reduce the loss
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

scaler = StandardScaler()
scaled_Xtr = scaler.fit_transform(Xtr)
scaled_Xte = scaler.transform(Xte)

net = make_net(Xtr.shape[1])

train(net, scaled_Xtr, ytr.values, epochs=300, loss_fn=nn.BCEWithLogitsLoss())

with torch.no_grad():
    logits = net(torch.tensor(scaled_Xte, dtype=torch.float32))
    probs = torch.sigmoid(logits)
    pred = (probs >= 0.5).int().numpy()

print("Accuracy:", accuracy_score(yte, pred))

# ===== Regression: predict body mass =====
Xtr, Xte, ytr, yte = data_prep.regression_split()

# YOUR CODE: like the classification block, with three differences:
#   - scale the FEATURES *and* the TARGET. Body mass is in the thousands; an
#     unscaled target makes the loss explode. Use a SECOND StandardScaler for y.
#   - train with nn.MSELoss(), epochs=500
#   - after predicting, use the target scaler's .inverse_transform(...) to turn the
#     scaled prediction back into grams, then print R^2 and MAE
#   - finally, plot_predictions(yte, pred, "PyTorch: predicted vs actual") to SEE the error (Section 6)

x_scaler = StandardScaler()
y_scaler = StandardScaler()

scaled_Xtr = x_scaler.fit_transform(Xtr)
scaled_Xte = x_scaler.transform(Xte)

scaled_ytr = y_scaler.fit_transform(ytr.values.reshape(-1, 1)).ravel()

net = make_net(Xtr.shape[1])

train(net, scaled_Xtr, scaled_ytr, epochs=500, loss_fn=nn.MSELoss())

with torch.no_grad():
    scaled_pred = net(torch.tensor(scaled_Xte, dtype=torch.float32)).numpy()

pred = y_scaler.inverse_transform(scaled_pred)

print("R2:", r2_score(yte, pred))
print("MAE:", mean_absolute_error(yte, pred))

plot_predictions(yte, pred, "PyTorch: predicted vs actual")
