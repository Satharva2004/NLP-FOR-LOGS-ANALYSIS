import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


# Function to generate sample log data
def generate_sample_logs(num_logs=1000):
    event_types = ["INFO", "ERROR", "WARNING", "LOGIN"]
    user_ids = [f"user_{i}" for i in range(1, 101)]  # 100 unique user IDs
    ip_addresses = [f"192.168.0.{i}" for i in range(1, 101)]  # 100 unique IP addresses

    data = {
        "timestamp": pd.date_range(start="1/1/2024", periods=num_logs, freq="H"),
        "event": np.random.choice(event_types, size=num_logs),
        "user_id": np.random.choice(user_ids, size=num_logs),
        "ip_address": np.random.choice(ip_addresses, size=num_logs),
        "message": np.random.choice(
            [
                "Connection error",
                "Authentication failure",
                "Session timeout",
                "Unknown error",
            ],
            size=num_logs,
        ),
    }

    df = pd.DataFrame(data)
    df.to_csv("sample_logs_clustering.csv", index=False)
    return df


df_logs = generate_sample_logs()


def perform_clustering(df):
    if "message" not in df.columns:
        raise ValueError("The 'message' column is missing from the DataFrame.")

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(
        df["message"].astype(str)
    )  # Convert to string type

    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(tfidf_matrix)
    df["kmeans_cluster"] = kmeans.labels_

    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(tfidf_matrix)
    df["dbscan_cluster"] = dbscan.labels_

    return df, kmeans, dbscan


def evaluate_clustering(df, algorithm):
    if algorithm == "kmeans":
        cluster_column = "kmeans_cluster"
    elif algorithm == "dbscan":
        cluster_column = "dbscan_cluster"
    else:
        raise ValueError("Algorithm must be 'kmeans' or 'dbscan'.")

    # Select numerical columns for silhouette score calculation
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    X = df[numerical_columns]

    silhouette_avg = silhouette_score(X, df[cluster_column])
    print(f"Silhouette Score ({algorithm.capitalize()}): {silhouette_avg}")


df_logs_clustered, kmeans, dbscan = perform_clustering(df_logs)

evaluate_clustering(df_logs_clustered, "kmeans")
evaluate_clustering(df_logs_clustered, "dbscan")

kmeans_cluster_dist = (
    df_logs_clustered.groupby(["kmeans_cluster", "event"])
    .size()
    .reset_index(name="count")
)
dbscan_cluster_dist = (
    df_logs_clustered.groupby(["dbscan_cluster", "event"])
    .size()
    .reset_index(name="count")
)

fig_kmeans = px.bar(
    kmeans_cluster_dist,
    x="event",
    y="count",
    color="kmeans_cluster",
    title="K-Means Clustering Distribution of Log Entries",
    labels={"event": "Event Type", "count": "Count", "kmeans_cluster": "Cluster"},
)
fig_kmeans.update_layout(
    xaxis_title="Event Type", yaxis_title="Count", legend_title="Cluster"
)
fig_kmeans2 = px.scatter(
    kmeans_cluster_dist,
    x="event",
    y="count",
    color="kmeans_cluster",
    title="K-Means Clustering Distribution of Log Entries",
    labels={"event": "Event Type", "count": "Count", "kmeans_cluster": "Cluster"},
    template="plotly",
)
fig_kmeans2.update_layout(
    xaxis_title="Event Type", yaxis_title="Count", legend_title="Cluster"
)

fig_dbscan = px.bar(
    dbscan_cluster_dist,
    x="event",
    y="count",
    color="dbscan_cluster",
    title="DBSCAN Clustering Distribution of Log Entries",
    labels={"event": "Event Type", "count": "Count", "dbscan_cluster": "Cluster"},
    template="plotly",
)
fig_dbscan.update_layout(
    xaxis_title="Event Type", yaxis_title="Count", legend_title="Cluster"
)
fig_dbscan2 = px.scatter(
    dbscan_cluster_dist,
    x="event",
    y="count",
    color="dbscan_cluster",
    title="DBSCAN Clustering Distribution of Log Entries",
    labels={"event": "Event Type", "count": "Count", "dbscan_cluster": "Cluster"},
)
fig_dbscan2.update_layout(
    xaxis_title="Event Type", yaxis_title="Count", legend_title="Cluster"
)
fig_kmeans.show()
fig_dbscan.show()
fig_kmeans2.show()
fig_dbscan2.show()
