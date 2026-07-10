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

preds, trues = [], []
with torch.no_grad():                       
    for images, labels in test_loader:     
        outputs = model(images.to(device)) 
        preds += outputs.argmax(1).cpu().tolist() 
        trues += labels.tolist()                  

print("accuracy:", accuracy_score(trues, preds))
print(confusion_matrix(trues, preds))

# ===== TODO 2: see what it got wrong (fun) =====
# Gather a few test images where prediction != true label and show them:
#     viz.show_grid(images, [f"pred: {BREEDS[p]} / true: {BREEDS[t]}" for ...])

wrong_images = []
wrong_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(1)

        for img, pred, true in zip(images.cpu(), preds.cpu(), labels):
            if pred != true:
                wrong_images.append(img)
                wrong_labels.append(
                    f"pred: {BREEDS[pred]} / true: {BREEDS[true]}"
                )

viz.show_grid(wrong_images[:9], wrong_labels[:9])