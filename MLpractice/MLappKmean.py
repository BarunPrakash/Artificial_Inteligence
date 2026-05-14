from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
%matplotlib inline

df = pd.read_csv("kmeans_machine_sensor_practice.csv")
df

features = [
    "rpm",
    "temperature_c",
    "vibration_mm_s",
    "current_amp",
    "quality_score"
]

X = df[features]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X_scaled)

# -------------------------------------------------
# 5. Convert cluster number into business meaning
# -------------------------------------------------
# K-Means gives only cluster numbers: 0, 1, 2.
# We convert them into Normal, Warning, Fault_Risk.

scaled_df = pd.DataFrame(X_scaled, columns=features)
scaled_df["cluster"] = df["cluster"]

# Higher temp + vibration + current = more risk
# Lower quality_score = more risk
scaled_df["risk_score"] = (
    scaled_df["temperature_c"]
    + scaled_df["vibration_mm_s"]
    + scaled_df["current_amp"]
    - scaled_df["quality_score"]
)

cluster_risk = scaled_df.groupby("cluster")["risk_score"].mean().sort_values()

ordered_clusters = cluster_risk.index.tolist()

cluster_to_health = {
    ordered_clusters[0]: "Normal",
    ordered_clusters[1]: "Warning",
    ordered_clusters[2]: "Fault_Risk"
}

df["kmeans_health"] = df["cluster"].map(cluster_to_health)

# -------------------------------------------------
# 6. Recommended action
# -------------------------------------------------

def recommended_action(health):
    if health == "Normal":
        return "Continue running"
    elif health == "Warning":
        return "Schedule inspection / monitor closely"
    elif health == "Fault_Risk":
        return "Stop machine and inspect immediately"
    else:
        return "Unknown"

df["recommended_action"] = df["kmeans_health"].apply(recommended_action)


# -------------------------------------------------
# 7. Show cluster summary
# -------------------------------------------------

summary = df.groupby(["cluster", "kmeans_health"])[features].mean().round(2)

print("\nCluster Summary:")
print(summary)

print("\nCluster Mapping:")
print(cluster_to_health)


# -------------------------------------------------
# 8. Compare with learning-only condition
# -------------------------------------------------
# This column is only for checking/demo.
# K-Means does not need this column for training.

if "true_condition_for_learning_only" in df.columns:
    print("\nComparison with actual condition:")
    print(pd.crosstab(
        df["kmeans_health"],
        df["true_condition_for_learning_only"]
    ))


# -------------------------------------------------
# 9. Final output
# -------------------------------------------------

final_columns = [
    "machine_id",
    "sample_no",
    "rpm",
    "temperature_c",
    "vibration_mm_s",
    "current_amp",
    "quality_score",
    "cluster",
    "kmeans_health",
    "recommended_action"
]

print("\nFinal Output:")
print(df[final_columns].head(20))

# -------------------------------------------------
# 10. Save output
# -------------------------------------------------

df.to_csv("machine_health_kmeans_output.csv", index=False)

print("\nSaved file: machine_health_kmeans_output.csv")



# -------------------------------------------------
# 11. Plot result
# -------------------------------------------------

plt.figure(figsize=(8, 6))

for health in ["Normal", "Warning", "Fault_Risk"]:
    temp_df = df[df["kmeans_health"] == health]
    plt.scatter(
        temp_df["temperature_c"],
        temp_df["vibration_mm_s"],
        label=health
    )

plt.xlabel("Temperature C")
plt.ylabel("Vibration mm/s")
plt.title("K-Means Machine Health Clustering")
plt.legend()
plt.grid(True)
plt.show()

