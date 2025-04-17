import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture


def trip_duration_clustering(df):
    cluster_features = ["trip_duration", "trip_distance"]
    X = df[cluster_features]

    # Normalize
    X_scaled = StandardScaler().fit_transform(X)

    # Define models
    models = {
        "KMeans (k=4)": KMeans(n_clusters=4, random_state=0),
        "Gaussian Mixture": GaussianMixture(n_components=4, random_state=0),
    }

    for name, model in models.items():
        if name == "Gaussian Mixture":
            clusters = model.fit_predict(X_scaled)
        else:
            clusters = model.fit_predict(X_scaled)

        df["cluster"] = clusters

        plt.figure(figsize=(8, 5))
        sns.scatterplot(
            data=df,
            x="trip_distance",
            y="trip_duration",
            hue="cluster",
            palette="Set2",
            legend="full",
        )
        plt.title(f"Clustering with {name}")
        plt.xlabel("Trip Distance")
        plt.ylabel("Trip Duration")
        plt.legend(title="Cluster")
        plt.tight_layout()
        plt.show()
