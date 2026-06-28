import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from umap import UMAP

base_features = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

features = base_features + [
    "sex_FEMALE",
    "sex_MALE"
]


def cluster_data(df, method="kmeans", k=3):
    df = df.copy()

    X = StandardScaler().fit_transform(df[features])

    if method == "kmeans":
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)

    elif method == "hdbscan":
        labels = HDBSCAN(min_cluster_size=15).fit_predict(X)

    else:
        raise ValueError("method must be either 'kmeans' or 'hdbscan'")
    
    df["cluster_id"] = labels

    return df


def test_k_values(df):
    X = StandardScaler().fit_transform(df[features])

    k_values = range(2, 9)
    inertias = []
    silhouette_scores = []

    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(X)

        inertias.append(model.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))
    
    plt.plot(k_values, inertias, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for KMeans")
    plt.show()

    plt.plot(k_values, silhouette_scores, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Scores for KMeans")
    plt.show()


def dimensional_reduction(df, method="pca"):
    df = df.copy()

    X = StandardScaler().fit_transform(df[features])

    if method == "pca":
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X)
        print(pca.explained_variance_ratio_)
    
    elif method == "tsne":
        coords = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)
    
    elif method == "umap":
        coords = UMAP(n_components=2, random_state=42).fit_transform(X)
    
    else:
        raise ValueError("method must be either 'pca', 'tsne', or 'umap'")
    
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]

    return df


def plot_clusters(df):
    plt.scatter(
        df["x"], 
        df["y"], 
        c=df["cluster_id"], 
        cmap="viridis", 
        s=20
    )

    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("Penguins clustered in 4D, projected to 2D")
    plt.show()


def plot_cluster_vs_species(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(
        df["x"],
        df["y"],
        c=df["cluster_id"],
        cmap="viridis",
        s=20
    )
    axes[0].set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")
    axes[0].set_title("Colored by Cluster")

    species_codes = df["species"].astype("category").cat.codes

    axes[1].scatter(
        df["x"],
        df["y"],
        c=species_codes,
        cmap="viridis",
        s=20
    )
    axes[1].set_xlabel("PC 1")
    axes[1].set_ylabel("PC 2")
    axes[1].set_title("Colored by Species")

    plt.tight_layout()
    plt.show()


def main():
    df = pd.read_csv("data/penguins.csv")

    df = df.dropna(subset=base_features + ["sex"]).copy()
    
    df = pd.get_dummies(df, columns=["sex"], dtype=int)

    test_k_values(df)

    kmeans_df = cluster_data(df, method="kmeans", k=3)
    kmeans_df = dimensional_reduction(kmeans_df, method="pca")

    print("KMeans:")
    print(pd.crosstab(kmeans_df["species"], kmeans_df["cluster_id"]))
    print()

    hdbscan_df = cluster_data(df, method="hdbscan")
    hdbscan_df = dimensional_reduction(hdbscan_df, method="pca")

    print("HDBSCAN:")
    print(pd.crosstab(hdbscan_df["species"], hdbscan_df["cluster_id"]))
    print()

    umap_df = cluster_data(df, method="kmeans", k=3)
    umap_df = dimensional_reduction(umap_df, method="umap")

    print("KMeans with UMAP:")
    print(pd.crosstab(umap_df["species"], umap_df["cluster_id"]))
    print()

    print("""
    KMeans produced three clusters that corresponded fairly well to the three
    penguin species, although there was some overlap between the Adelie and
    Chinstrap species.

    HDBSCAN found two main clusters and classified a few penguins as noise
    (cluster -1). It grouped the Adelie and Chinstrap penguins together because,
    based on the four measurements, it did not detect a strong density boundary
    between them.

    In this case, KMeans agreed better with the known species labels. However,
    that does not necessarily make it the better algorithm. KMeans performs well
    when we already know the expected number of clusters, while HDBSCAN is useful
    when we want the algorithm to discover the natural cluster structure and
    identify potential outliers automatically.
    """)

    plot_cluster_vs_species(kmeans_df)
    plot_cluster_vs_species(hdbscan_df)
    plot_cluster_vs_species(umap_df)


if __name__ == "__main__":
    main()