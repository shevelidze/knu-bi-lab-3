import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def trip_duration_clustering(df):
    cluster_features = ["trip_duration", "trip_distance"]
    X = df[cluster_features]

    # Normalize
    X_scaled = StandardScaler().fit_transform(X)

    # KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=0)
    clusters = kmeans.fit_predict(X_scaled)

    df["cluster"] = clusters

    # Visualize
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="trip_distance",
        y="trip_duration",
        hue="cluster",
        palette="Set2",
    )
    plt.title("Trip Clustering")
    plt.show()
