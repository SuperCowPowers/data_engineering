import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/penguins.csv")

outlier_indices = set()

# Df of only quantitative columns
measurments_df = df.select_dtypes(include="number")
print(measurments_df)
print("-------------------------------")

# Find outliers IQR method
def find_outliers_iqr(series):
    series = series.dropna()

    q1 = series.quantile(0.25)
    q2 = series.quantile(0.50)
    q3 = series.quantile(0.75)

    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    
    outliers = series[(series < lower_fence) | (series > upper_fence)]

    print("IQR: ", iqr)
    print("Lower Fence: ", lower_fence)
    print("Upper Fence: ", upper_fence)

    print("Outliers:", len(outliers))
    for index, value in outliers.items():
        print(f"Penguin #{index} {value}")
    print()

    return outliers

# Find outliers z-score method
def find_outliers_zscore(series):
    series = series.dropna()

    mean = series.mean()
    std = series.std()

    lower_fence = mean - std * 3
    upper_fence = mean + std * 3

    outliers = series[(series < lower_fence) | (series > upper_fence)]

    print("Mean:", mean)
    print("Standard Deviation:", std)
    print("Lower Fence:", lower_fence)
    print("Upper Fence:", upper_fence)

    print("Outliers:", len(outliers))
    for index, value in outliers.items():
        print(f"Penguin #{index} {value}")
    print()

    return outliers


# Analyze series of data
def series_data_analysis_iqr(series, title, axes=None):
    print("DF: ", title)
    print()
    print(series.describe())
    print()

    outliers = find_outliers_iqr(series)
    find_outliers_zscore(series)

    outlier_indices.update(outliers.index)

    print("-------------------------------")

    series.plot.box(ax=axes)
    axes.set_title(title)

# Analyze multiple series (dataframe) of data
def dataframe_data_analysis(dataframe, title):
    figure, axes = plt.subplots(1, len(dataframe.columns), figsize=(12, 4))

    for i, col in enumerate(dataframe.columns):
        series_data_analysis_iqr(dataframe[col], title, axes[i])

    plt.tight_layout()
    plt.show()

dataframe_data_analysis(measurments_df, "measurments_df")

for species, group in df.groupby("species"):
    numeric_group = group.select_dtypes(include="number")
    dataframe_data_analysis(numeric_group, species)

for (species, sex), group in df.groupby(["species", "sex"]):
    numeric_group = group.select_dtypes(include="number")
    dataframe_data_analysis(numeric_group, species)

df.boxplot(column="body_mass_g", by="species")
plt.show()

outlier_rows = df.loc[sorted(outlier_indices)]
outlier_rows.to_csv("data/penguin_outliers.csv", index=False)