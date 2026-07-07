import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
)
from plotting import plot_predictions
import data_prep
from sklearn.model_selection import cross_val_score

# ===== Classification: predict sex =====
Xtr, Xte, ytr, yte = data_prep.classification_split()

# YOUR CODE: create an XGBClassifier and fit it on (Xtr, ytr).
#   Reasonable settings: n_estimators=200, max_depth=3, learning_rate=0.1, eval_metric="logloss"
#   (trees don't need scaled features — fit the raw X)
clf = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    eval_metric="logloss",
    max_depth=3,
    random_state=42,
)

clf.fit(Xtr, ytr)

# YOUR CODE: predict on Xte, then print the accuracy and the confusion matrix.
#   -> accuracy_score(yte, pred)   and   confusion_matrix(yte, pred)
predictions = clf.predict(Xte)

accuracy = accuracy_score(yte, predictions)
cm = confusion_matrix(yte, predictions)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(cm)

cv_scores = cross_val_score(clf, pd.concat([Xtr, Xte]), pd.concat([ytr, yte]), cv=5, scoring="accuracy")

print("CV accuracy scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())
# Given for you — which measurements mattered most (ties back to Project 3's PCA):
importance = pd.Series(clf.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
print(importance.round(3).to_string())

# ===== Regression: predict body mass =====
Xtr, Xte, ytr, yte = data_prep.regression_split()

# YOUR CODE: this mirrors the classification block, but with XGBRegressor.
#   1. create + fit an XGBRegressor  (try n_estimators=300, max_depth=3, learning_rate=0.1)
#   2. predict on Xte
#   3. print R^2 (r2_score) and mean absolute error (mean_absolute_error), in grams
#   4. call plot_predictions(yte, pred, "XGBoost: predicted vs actual") to SEE the error (Section 6)

regressor = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.1, random_state=42)

regressor.fit(Xtr, ytr)

predictions = regressor.predict(Xte)

r2 = r2_score(yte, predictions)
mae = mean_absolute_error(yte, predictions)

print("R^2:", r2)
print("MAE:", mae)

plot_predictions(yte, predictions, "XGBoost: predicted vs actual")
