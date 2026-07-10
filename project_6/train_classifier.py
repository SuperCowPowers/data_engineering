import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import data_setup

# Apple Silicon GPU if available, else CPU (given).
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("training on:", device)

train_loader = data_setup.train_loader()
num_classes = len(data_setup.BREEDS)          # 25 (however many you listed)

# ===== TODO 1: load the pretrained network =====
# ResNet18, already trained on ImageNet — it arrives knowing how to "see".
#     model = resnet18(weights=ResNet18_Weights.DEFAULT)
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# ===== TODO 2: freeze the backbone =====
# `requires_grad = False` tells PyTorch NOT to compute gradients for these weights.
# No gradients means the optimizer can't nudge them — they're "frozen" and reused
# exactly as they came from ImageNet.
#     for p in model.parameters():
#         p.requires_grad = False

for p in model.parameters():
    p.requires_grad = False

# ===== TODO 3: swap in a new head =====
# `fc` is ResNet's final "fully connected" layer — the classifier that sits on top
# of the features (a plain `nn.Linear`). The pretrained one has 1000 outputs
# (ImageNet's categories); replace it with a fresh Linear that outputs num_classes.
# A brand-new layer has requires_grad = True by default, so this head is the ONLY
# part that trains.
#     model.fc = nn.Linear(model.fc.in_features, num_classes)

model.fc = nn.Linear(model.fc.in_features, num_classes)
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

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

epochs = 5

for epoch in range(epochs):
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        opt.zero_grad()

        preds = model(xb)
        loss = loss_fn(preds, yb)

        loss.backward()
        opt.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

# ===== Save the trained model (given) =====
model.breeds = data_setup.BREEDS      # remember the class names WITH the model, so
#                                       predict.py and Project 7 never need their own copy
torch.save(model, "dog_model.pt")     # evaluate.py and Project 7 both load this
print("saved dog_model.pt")