import pandas as pd
import matplotlib.pyplot as plt

# Base dataframe
df = pd.read_csv("data/curated-solubility-dataset.csv")

# Historgram of solubility
df["Solubility"].hist(bins=20)

plt.title("Distribution of Solubility")
plt.xlabel("Solubility")
plt.ylabel("Number of Compounds")

plt.show()

# Filtered and sorted dataframe
sorted_solubility_df = df[["ID", "Name", "Solubility"]].sort_values("Solubility", ascending=False)
sorted_solubility_df.to_csv("data/descending_solubility.csv", index=False)