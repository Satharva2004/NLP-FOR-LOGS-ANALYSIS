import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np

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

    # Use TfidfVectorizer to convert text data to numerical features
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(
        df["message"].astype(str)
    )

    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(tfidf_matrix)
    df["kmeans_cluster"] = kmeans.labels_

    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(tfidf_matrix)
    df["dbscan_cluster"] = dbscan.labels_

    return df, kmeans, dbscan


# Function to evaluate clustering quality
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


# Perform clustering on log data
df_logs_clustered, kmeans, dbscan = perform_clustering(df_logs)

# Evaluate clustering quality
evaluate_clustering(df_logs_clustered, "kmeans")
evaluate_clustering(df_logs_clustered, "dbscan")

# Interactive visualization with Plotly
# K-Means clustering plot
fig_kmeans = px.scatter_3d(
    df_logs_clustered,
    x="message",
    y="event",
    z="user_id",
    color="kmeans_cluster",
    title="K-Means Clustering of Log Entries",
    hover_name="message",
    hover_data=["event", "user_id"],
    labels={
        "message": "Message",
        "event": "Event Type",
        "user_id": "User ID",
        "kmeans_cluster": "Cluster",
    },
)
fig_kmeans.update_traces(marker=dict(size=5))
fig_kmeans.update_layout(
    scene=dict(xaxis_title="Message", yaxis_title="Event Type", zaxis_title="User ID")
)

# DBSCAN clustering plot
fig_dbscan = px.scatter_3d(
    df_logs_clustered,
    x="message",
    y="event",
    z="user_id",
    color="dbscan_cluster",
    title="DBSCAN Clustering of Log Entries",
    hover_name="message",
    hover_data=["event", "user_id"],
    labels={
        "message": "Message",
        "event": "Event Type",
        "user_id": "User ID",
        "dbscan_cluster": "Cluster",
    },
)
fig_dbscan.update_traces(marker=dict(size=5))
fig_dbscan.update_layout(
    scene=dict(xaxis_title="Message", yaxis_title="Event Type", zaxis_title="User ID")
)

# Show the plots
fig_kmeans.show()
fig_dbscan.show()
