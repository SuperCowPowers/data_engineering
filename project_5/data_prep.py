"""
data_prep.py - shared data loading and train/test splitting for Project 5.

Both xgboost_models.py and pytorch_models.py import from here, so they train and
test on the *exact same* split. That's what makes the head-to-head comparison
fair: if each script made its own split, they'd be graded on different penguins
and you couldn't trust the numbers.

This file only uses pandas + scikit-learn (no xgboost, no torch), so it's safe
for either model script to import.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

DATA = "data/penguins.csv"  # run the scripts from inside project_5/
NUM = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]


def _load():
    """Load penguins and turn the text `species` column into 0/1 columns."""
    df = pd.read_csv(DATA)
    # Models need numbers, not text — one-hot encode species into
    # species_Adelie / species_Chinstrap / species_Gentoo columns.
    return pd.get_dummies(df, columns=["species"], dtype=float)


def classification_split():
    """Features + label for predicting SEX (1 = male, 0 = female)."""
    d = _load().dropna(subset=NUM + ["sex"])
    species = [c for c in d.columns if c.startswith("species_")]
    X = d[NUM + species]
    y = (d["sex"].str.upper() == "MALE").astype(int)
    # stratify keeps the male/female ratio the same in train and test
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


def regression_split():
    """Features + target for predicting BODY MASS (grams)."""
    d = _load().dropna(subset=NUM)
    species = [c for c in d.columns if c.startswith("species_")]
    # note: body_mass_g is the TARGET, so it must not be a feature
    features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm"] + species
    X = d[features]
    y = d["body_mass_g"]
    return train_test_split(X, y, test_size=0.25, random_state=42)
