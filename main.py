import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve, ConfusionMatrixDisplay)
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.neighbors import NearestNeighbors

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from pathlib import Path
from scipy.stats import zscore
from sklearn.covariance import EllipticEnvelope
import os

pd.set_option('display.max_columns', None)

# Ensure the 'figs' directory exists
output_dir = "figs/"
os.makedirs(output_dir, exist_ok=True)

file_path = 'dataset.csv'
data = pd.read_csv(file_path)
print(data['NObeyesdad'].unique())

# # Priting Data Info
print(data.info())
# print(data.isnull().sum())
data = data.drop_duplicates()
print(data.info())

data['Age'] = data["Age"].astype(int)
numerical_columns = ['Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
data[numerical_columns] = data[numerical_columns].round(2)
categorical_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

# --------------------------------------Data Preprocessing-----------------------------------------
####   Normalization ####
#     # Min-Max Sacling 
# scaler = MinMaxScaler()
# data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
# print(data)

#     # Z-score Scaling 
# Standard_Scaler = StandardScaler()
# data[numerical_columns] = Standard_Scaler.fit_transform(data[numerical_columns])
# print(data)

# num_bins = 4
# # Apply equal-depth (quantile) binning
# for col in numerical_columns:
#     data[f'{col}_equal_depth'] = pd.qcut(data[col], q=num_bins, labels=False, duplicates='drop')

# print(data['Age_equal_depth'].unique())

# data.to_csv('binned_data.csv', index=False)

# print(data['NObeyesdad'].value_counts(normalize=True))

# # data['NObeyesdad'] = data['NObeyesdad'].replace(['Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II',
# #                                                  'Obesity_Type_I', 'Insufficient_Weight', 'Obesity_Type_II',
# #                                                  'Obesity_Type_III'], [0, 1, 1, 1, 0, 1, 1])

# print(data['NObeyesdad'].unique())
# print(data['NObeyesdad'].value_counts(normalize=True))

# normal_weight_count = data[data['NObeyesdad'] == 0].shape[0]
# print(normal_weight_count)

data['BMI'] = round(data['Weight'] / (data['Height']) ** 2, 2)

# ----------------------------------------EDA---------------------------------------
# target_column = 'NObeyesdad'
# for col in categorical_columns:
#     if col != target_column:  # Skip the target column itself
#         # Calculate mean of the target column grouped by the current categorical column
#         group_mean = data.groupby(col)[target_column].mean()

#         # Plotting
#         group_mean.plot(kind='bar', title=f"Mean '{target_column}' by {col}")
#         plt.xlabel(col)
#         plt.ylabel(f"Mean {target_column}")
#         plt.xticks(rotation=45)
#         plt.show()

# plt.figure(figsize=(10, 8))
# correlation_matrix = data[numerical_columns + ['NObeyesdad']].corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# plt.title('Correlation Heatmap')
# plt.show()

data_copy = data.copy()
columns = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

for column in columns:
    data_copy[column] = data_copy[column].round().astype(int)

mapping = {
    'NCP': {
        '1': 'Between 1 and 2',
        '2': 'Three',
        '3': 'More than three',
        '4': 'More than three'
    },
    'CH2O': {
        '1': 'Less than a liter',
        '2': 'Between 1 and 2 L',
        '3': 'More than 2 L',
    },
    'FAF': {
        '0': 'I do not have',
        '1': '1 or 2 days',
        '2': '2 or 4 days',
        '3': '4 or 5 days'
    },
    'TUE': {
        '1': '0-2 hours',
        '2': '3-5 hours',
        '3': 'More than 5 hours',
    },
    'FCVC': {
        '1': 'Never',
        '2': 'Sometimes',
        '3': 'Always',
    }
}

for column in columns:
    if column in mapping:
        data_copy[column] = data_copy[column].astype(str).replace(mapping[column])

new_column_names = {
    'FCVC': 'Frequency of consumption of vegetables (FCVC)',
    'NCP': 'Number of main meals (NCP)',
    'CH2O': 'Consumption of water daily (CH2O)',
    'FAF': 'Physical activity frequency (FAF)',
    'TUE': 'Time using technology devices (TUE)',
    'CALC': 'Consumption of alcohol (CALC)',
    'CAEC': 'Consumption of food between meals (CAEC)',
    'FAVC': 'Frequent consumption of high caloric food (FAVC)',
    'SCC': 'Calories consumption monitoring (SCC)',
}

data_copy.rename(columns=new_column_names, inplace=True)
print(data_copy.info())
data_copy.to_csv('refined_data.csv', index=False)

# LabelEncoder = LabelEncoder()
# for col in categorical_columns:
#     data[col] = LabelEncoder.fit_transform(data[col])
# # # print(data)


# ####### Average age of each obesity type #########
print(data.groupby("NObeyesdad")['Age'].median())
data.groupby("NObeyesdad")["Age"].median().sort_values(ascending=False).plot(kind="bar",
                                                                             color=sns.color_palette("Set1"))
plt.title("Average age of each obesity type")
plt.savefig("figs/average_age_obesity_type.png")
plt.show()

# ####### Average weight of each obesity type #########
print(data.groupby("NObeyesdad")['Weight'].median())
data.groupby("NObeyesdad")["Weight"].median().sort_values(ascending=False).plot(kind="bar",
                                                                                color=sns.color_palette("Set2"))
plt.title("Average Weight of each obesity type")
plt.savefig("figs/average_weight_obesity_type.png")
plt.show()

# ####### How is obesity type affected by eating high calorie food? #########
print(data.groupby(['NObeyesdad', 'FAVC'])["FAVC"].count())
plt.figure(figsize=(10, 7))
sns.countplot(data=data, x=data.NObeyesdad, hue=data.FAVC, palette=sns.color_palette("Dark2"))
plt.xticks(rotation=-20)
plt.title("How is obesity type affected by eating high calorie food?")
plt.savefig("figs/obesity_type_eating_high_calorie_food.png")
plt.show()

# ####### Does family history with overweight affect obesity type? #########
plt.figure(figsize=(10, 7))
sns.countplot(data=data, x=data.NObeyesdad, hue=data.family_history_with_overweight, palette=sns.color_palette("Dark2"))
plt.xticks(rotation=-20)
plt.title("Does family history with overweight affect obesity type?")
plt.savefig("figs/family_history_with_overweight_obesity_type.png")
plt.show()

# ####### Correlation between data atributes #########
corr_data = data.copy()
encoder = LabelEncoder()
for col in corr_data.select_dtypes(include="object").columns:
    corr_data[col] = encoder.fit_transform(corr_data[col])

plt.figure(figsize=(16, 13))
sns.heatmap(data=corr_data.corr(), annot=True)
plt.title("Correlation between data atributes")
plt.savefig("figs/correlation_between_data_attributes.png")
plt.show()

plt.figure(figsize=(10, 7))
sns.boxplot(data=data, x='NObeyesdad', y='BMI', palette='Set3')
plt.title('BMI Distribution by Obesity Type')
plt.xticks(rotation=-20)
plt.ylabel('BMI')
plt.xlabel('Obesity Type')
plt.savefig('figs/bmi_distribution_by_obesity_type.png')
plt.show()

data = pd.get_dummies(data, columns=categorical_columns, drop_first=False)
data.to_csv('refined_data.csv', index=False)

# LabelEncoder = LabelEncoder()
# data['NObeyesdad'] = LabelEncoder.fit_transform(data['NObeyesdad'])


# # # # -------------------------------------Clustering-------------------------------------
# # # K-Means Clustering without target column 
X = data.drop(columns='NObeyesdad', inplace=False)

# # Apply PCA for dimensionality reduction (2 components for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# K-Means Clustering without target column 
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X_pca)

cluster_range = range(2, 10)
eps_values = [0.3, 0.5, 0.7, 0.9, 1.1]

kmeans_silhouette_scores = []
kmeans_calinski_scores = []
kmeans_davies_scores = []


def plot_clusters(X, labels, title, save_path=None):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter)
    if save_path:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


base_dir = "figs/k_Means_Clustering/cluster"
Path(base_dir).mkdir(parents=True, exist_ok=True)
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    silhouette = silhouette_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    davies = davies_bouldin_score(X, labels)
    kmeans_silhouette_scores.append(silhouette)
    kmeans_calinski_scores.append(calinski)
    kmeans_davies_scores.append(davies)

    filename = f"cluster_k{k}.png"
    filepath = os.path.join(base_dir, filename)

    plot_clusters(X_pca, labels, f"K-Means Clustering (n_clusters={k})", save_path=file_path)

silhouette_array = np.array(kmeans_silhouette_scores)
calinski_array = np.array(kmeans_calinski_scores)
davies_array = np.array(kmeans_davies_scores)

scaler = MinMaxScaler()
silhouette_normalized = scaler.fit_transform(silhouette_array.reshape(-1, 1)).flatten()
calinski_normalized = scaler.fit_transform(calinski_array.reshape(-1, 1)).flatten()
davies_normalized = scaler.fit_transform((1 / davies_array).reshape(-1, 1)).flatten()

# Compute average score for each k
average_scores = (silhouette_normalized + calinski_normalized + davies_normalized) / 3

# Find the best k
best_k_index = np.argmax(average_scores)
best_k = cluster_range[best_k_index]

print("Normalized Scores:")
print(f"Silhouette: {silhouette_normalized}")
print(f"Calinski-Harabasz: {calinski_normalized}")
print(f"Inverted Davies-Bouldin: {davies_normalized}")
print(f"Average Scores: {average_scores}")
print(f"Best number of clusters: {best_k}")

# Create a DataFrame to summarize the evaluation metrics
results_df = pd.DataFrame({
    'Number of Clusters': list(cluster_range),
    'Silhouette Score': kmeans_silhouette_scores,
    'Calinski-Harabasz Index': kmeans_calinski_scores,
    'Davies-Bouldin Index': kmeans_davies_scores
})

print(results_df)

# # K-Means Clustering with target column
X = data.drop(columns='NObeyesdad', inplace=False)

# Apply PCA for dimensionality reduction (2 components for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# K-Means Clustering with target column 
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

cluster_range = range(2, 10)
eps_values = [0.3, 0.5, 0.7, 0.9, 1.1]

kmeans_silhouette_scores = []
kmeans_calinski_scores = []
kmeans_davies_scores = []


def plot_clusters(X, labels, title):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter)
    plt.show()


base_dir = "figs/k_Means_Clustering/cluster"
Path(base_dir).mkdir(parents=True, exist_ok=True)
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    silhouette = silhouette_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    davies = davies_bouldin_score(X, labels)
    kmeans_silhouette_scores.append(silhouette)
    kmeans_calinski_scores.append(calinski)
    kmeans_davies_scores.append(davies)

    plot_clusters(X_pca, labels, f"K-Means Clustering (n_clusters={k})")
    filename = f"cluster_k{k}.png"
    filepath = os.path.join(base_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

silhouette_array = np.array(kmeans_silhouette_scores)
calinski_array = np.array(kmeans_calinski_scores)
davies_array = np.array(kmeans_davies_scores)

scaler = MinMaxScaler()
silhouette_normalized = scaler.fit_transform(silhouette_array.reshape(-1, 1)).flatten()
calinski_normalized = scaler.fit_transform(calinski_array.reshape(-1, 1)).flatten()
davies_normalized = scaler.fit_transform((1 / davies_array).reshape(-1, 1)).flatten()

# Compute average score for each k
average_scores = (silhouette_normalized + calinski_normalized + davies_normalized) / 3

# Find the best k
best_k_index = np.argmax(average_scores)
best_k = cluster_range[best_k_index]

print("Normalized Scores:")
print(f"Silhouette: {silhouette_normalized}")
print(f"Calinski-Harabasz: {calinski_normalized}")
print(f"Inverted Davies-Bouldin: {davies_normalized}")
print(f"Average Scores: {average_scores}")
print(f"Best number of clusters: {best_k}")

# Create a DataFrame to summarize the evaluation metrics
results_df = pd.DataFrame({
    'Number of Clusters': list(cluster_range),
    'Silhouette Score': kmeans_silhouette_scores,
    'Calinski-Harabasz Index': kmeans_calinski_scores,
    'Davies-Bouldin Index': kmeans_davies_scores
})

print(results_df)


## DBSCAN Clustering
# Analyze kNN distances for estimating optimal eps
neighbors = NearestNeighbors(n_neighbors=10)
neighbors_fit = neighbors.fit(X_pca)
distances, _ = neighbors_fit.kneighbors(X_pca)
distances = np.sort(distances[:, -1])

# Plot kNN distances
plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.title("kNN Distance Plot")
plt.xlabel("Points Sorted by Distance")
plt.ylabel("Distance to 10th Nearest Neighbor")
plt.grid(True)
plt.show()

# Define the range of eps values for DBSCAN
eps_values = [4.0, 4.2, 4.5, 4.8, 5.0]  # Adjust this based on kNN plot
min_samples = 10

dbscan_silhouette_scores = []
dbscan_calinski_scores = []
dbscan_davies_scores = []

dbscan_base_dir = "figs/dbscan_clustering/cluster"
Path(dbscan_base_dir).mkdir(parents=True, exist_ok=True)

for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_pca)

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if num_clusters <= 1:
        print(f"DBSCAN skipped for eps={eps} (all points classified as noise).")
        dbscan_silhouette_scores.append(None)
        dbscan_calinski_scores.append(None)
        dbscan_davies_scores.append(None)
        continue

    silhouette = silhouette_score(X_pca, labels)
    calinski = calinski_harabasz_score(X_pca, labels)
    davies = davies_bouldin_score(X_pca, labels)

    dbscan_silhouette_scores.append(silhouette)
    dbscan_calinski_scores.append(calinski)
    dbscan_davies_scores.append(davies)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.title(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})")
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter)
    filename = f"dbscan_eps_{eps}.png"
    filepath = os.path.join(dbscan_base_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# Normalize the scores for comparison
dbscan_silhouette_array = np.array([score if score is not None else 0 for score in dbscan_silhouette_scores])
dbscan_calinski_array = np.array([score if score is not None else 0 for score in dbscan_calinski_scores])
dbscan_davies_array = np.array([score if score is not None else np.inf for score in dbscan_davies_scores])

scaler = MinMaxScaler()
dbscan_silhouette_normalized = scaler.fit_transform(dbscan_silhouette_array.reshape(-1, 1)).flatten()
dbscan_calinski_normalized = scaler.fit_transform(dbscan_calinski_array.reshape(-1, 1)).flatten()
dbscan_davies_normalized = scaler.fit_transform((1 / dbscan_davies_array).reshape(-1, 1)).flatten()

dbscan_average_scores = (dbscan_silhouette_normalized + dbscan_calinski_normalized + dbscan_davies_normalized) / 3

dbscan_best_eps_index = np.argmax(dbscan_average_scores)
dbscan_best_eps = eps_values[dbscan_best_eps_index]

print("DBSCAN Normalized Scores:")
print(f"Silhouette: {dbscan_silhouette_normalized}")
print(f"Calinski-Harabasz: {dbscan_calinski_normalized}")
print(f"Inverted Davies-Bouldin: {dbscan_davies_normalized}")
print(f"Average Scores: {dbscan_average_scores}")
print(f"Best eps value: {dbscan_best_eps}")

dbscan_results_df = pd.DataFrame({
    'Eps': eps_values,
    'Silhouette Score': dbscan_silhouette_scores,
    'Calinski-Harabasz Index': dbscan_calinski_scores,
    'Davies-Bouldin Index': dbscan_davies_scores,
    'Normalized Silhouette Score': dbscan_silhouette_normalized,
    'Normalized Calinski-Harabasz Index': dbscan_calinski_normalized,
    'Normalized Davies-Bouldin Index (Inverted)': dbscan_davies_normalized,
    'Average Normalized Score': dbscan_average_scores
})

print("DBSCAN Results Summary:")
print(dbscan_results_df)


# Hierarchical Clustering

print("Hierarchical Clustering Results")


def plot_dendrogram(X, method='ward', truncate_level=5, title="Hierarchical Clustering Dendrogram"):
    """Plots a dendrogram for hierarchical clustering."""
    plt.figure(figsize=(12, 8))
    linkage_matrix = linkage(X, method=method)
    dendrogram(linkage_matrix, truncate_mode='level', p=truncate_level)
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.show()


def evaluate_hierarchical_clustering(X, n_clusters_range, linkage_method='ward'):
    """Evaluates hierarchical clustering using multiple metrics."""
    hierarchical_results = []

    for n_clusters in n_clusters_range:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        labels = model.fit_predict(X)

        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        davies = davies_bouldin_score(X, labels)

        hierarchical_results.append({
            'Number of Clusters': n_clusters,
            'Silhouette Score': silhouette,
            'Calinski-Harabasz Index': calinski,
            'Davies-Bouldin Index': davies
        })

    return pd.DataFrame(hierarchical_results)


def print_hierarchical_results(df):
    """Prints hierarchical clustering results in a tabular format."""
    print(df)
    print("\nBest Results:")
    best_silhouette_idx = df['Silhouette Score'].idxmax()
    print(f"Best Number of Clusters (Silhouette): {df.loc[best_silhouette_idx, 'Number of Clusters']}")
    print(f"Silhouette Score: {df.loc[best_silhouette_idx, 'Silhouette Score']:.4f}")
    print(f"Calinski-Harabasz Index: {df.loc[best_silhouette_idx, 'Calinski-Harabasz Index']:.4f}")
    print(f"Davies-Bouldin Index: {df.loc[best_silhouette_idx, 'Davies-Bouldin Index']:.4f}")


def plot_evaluation_metrics(df):
    """Plots evaluation metrics for hierarchical clustering."""
    plt.figure(figsize=(15, 5))

    # Silhouette Score
    plt.subplot(1, 3, 1)
    plt.plot(df['Number of Clusters'], df['Silhouette Score'], marker='o')
    plt.title("Silhouette Score vs. Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")

    # Calinski-Harabasz Index
    plt.subplot(1, 3, 2)
    plt.plot(df['Number of Clusters'], df['Calinski-Harabasz Index'], marker='o', color='orange')
    plt.title("Calinski-Harabasz Index vs. Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Calinski-Harabasz Index")

    # Davies-Bouldin Index
    plt.subplot(1, 3, 3)
    plt.plot(df['Number of Clusters'], df['Davies-Bouldin Index'], marker='o', color='green')
    plt.title("Davies-Bouldin Index vs. Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Davies-Bouldin Index")

    plt.tight_layout()
    plt.show()


def plot_clusters(X_pca, labels, title="Hierarchical Clustering"):
    """Plots clustering results using PCA-reduced data."""
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(scatter)
    plt.show()


# Perform hierarchical clustering
X = data.drop(columns='NObeyesdad')
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot dendrogram
plot_dendrogram(X, method='ward', truncate_level=5)

# Evaluate clustering performance
n_clusters_range = range(2, 10)
results_df = evaluate_hierarchical_clustering(X, n_clusters_range, linkage_method='ward')

# Print clustering results
print_hierarchical_results(results_df)

# Plot evaluation metrics
plot_evaluation_metrics(results_df)

# Visualize clustering with the best number of clusters
best_n_clusters = results_df.loc[results_df['Silhouette Score'].idxmax(), 'Number of Clusters']
model = AgglomerativeClustering(n_clusters=int(best_n_clusters), linkage='ward')
labels = model.fit_predict(X)
plot_clusters(X_pca, labels, title=f"Agglomerative Clustering with {best_n_clusters} Clusters")

output_dir_outlier_dectection = "figs/outlier_detection/"
os.makedirs(output_dir_outlier_dectection, exist_ok=True)
# # -------------------------------------Outlier detection------------------------------------------------
feature_data = data.drop(columns='NObeyesdad', inplace=False)
iso_forest = IsolationForest()

# Isolation Forest 
iso_forest = IsolationForest(contamination=0.05, random_state=42)
data['Outlier_ISO'] = iso_forest.fit_predict(feature_data)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Age', y='Weight', hue='Outlier_ISO', palette='coolwarm')
plt.title("Outlier Detection using Isolation Forest")
plt.xlabel("Age")
plt.ylabel("Weight")
plt.legend(title="Outlier")
plt.savefig("figs/outlier_detection/ISOLation_Forest.png")
plt.show()

print("Isolation Forest Outlier Counts:")
print(data['Outlier_ISO'].value_counts())

# Print the outliers
# outliers = data[data['Outlier_ISO'] == -1]
# print("Outliers detected by Isolation Forest:")
# print(outliers)

outliers = data[data['Outlier_ISO'] == -1]
print(f"Number of outliers detected: {len(outliers)}")

# Calculate z-scores for numerical features
numerical_features = feature_data.columns  # Exclude categorical or target features

# Compute how far each outlier lies from the mean for selected features
deviation_from_mean = outliers[numerical_features] - feature_data[numerical_features].mean()
for col in numerical_features:
    print(f"Outliers for {col}:")
    print(deviation_from_mean[col].abs().sort_values(ascending=False).head())
    print("-" * 40)

# Visualize deviations for one feature (e.g., Weight)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Outlier_ISO', y='Weight', data=data, palette='coolwarm')
plt.title("Outliers vs Non-Outliers for Weight")
plt.xlabel("Outlier (ISO Forest)")
plt.ylabel("Weight")
plt.savefig("figs/outlier_detection/ISO_variation_in_oultiers")
plt.show()

# data = data[data['Outlier_ISO'] == 1]
# data = data.drop(columns='Outlier_ISO', inplace=False)


# # ---- LOF ------
# Extract numerical data for LOF
feature_data = data.drop(columns='NObeyesdad', inplace=False)

# Apply Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
outlier_labels = lof.fit_predict(feature_data)

# Add outlier labels to the dataframe
data['Outlier_LOF'] = outlier_labels

outliers_LOF = data[data['Outlier_LOF'] == -1]

numerical_features = feature_data.columns
deviation_from_mean = outliers_LOF[numerical_features] - feature_data[numerical_features].mean()
for col in numerical_features:
    print(f"Outliers for {col}:")
    print(deviation_from_mean[col].abs().sort_values(ascending=False).head())
    print("-" * 40)

# Visualize deviations for one feature (e.g., Weight)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Outlier_LOF', y='Weight', data=data, palette='coolwarm')
plt.title("Outliers vs Non-Outliers for Weight")
plt.xlabel("Outlier (LOF)")
plt.ylabel("Weight")
plt.savefig("figs/outlier_detection/LOF_variation_in_oultiers")
plt.show()

# Visualize the results using a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Age', y='Weight', hue='Outlier_LOF', palette='coolwarm')
plt.title("Outlier Detection using Local Outlier Factor")
plt.xlabel("Age")
plt.ylabel("Weight")
plt.legend(title="Outlier")
plt.savefig("figs/outlier_detection/LOF_outliers")
plt.show()

# Count and display the number of outliers detected
n_outliers = sum(outlier_labels == -1)
print(f"Local Outlier Factor detected {n_outliers} outliers out of {feature_data.shape[0]} samples.")
# data = data.drop(columns='Outlier_LOF', inplace=False)


## EllipticEnvelope
feature_data = data.drop(columns='NObeyesdad', inplace=False)

# Apply EllipticEnvelope
elliptic_env = EllipticEnvelope(contamination=0.05, random_state=42)
elliptic_env.fit(feature_data)
outlier_labels_elliptic = elliptic_env.predict(feature_data)

data['Outlier_Elliptic'] = outlier_labels_elliptic
outliers_elliptic = data[data['Outlier_Elliptic'] == -1]
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Age', y='Weight', hue='Outlier_Elliptic', palette='coolwarm')
plt.title("Outlier Detection using Elliptic Envelope")
plt.xlabel("Age")
plt.ylabel("Weight")
plt.legend(title="Outlier")
plt.savefig("figs/outlier_detection/EllipticEnvelope_outliers.png")
plt.show()

numerical_features = feature_data.columns
deviation_from_mean = outliers_elliptic[numerical_features] - feature_data[numerical_features].mean()

# Print the largest deviations for each feature
for col in numerical_features:
    print(f"Outliers for {col}:")
    print(deviation_from_mean[col].abs().sort_values(ascending=False).head())
    print("-" * 40)

# Visualize deviations for one feature (e.g., Weight)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Outlier_Elliptic', y='Weight', data=data, palette='coolwarm')
plt.title("Outliers vs Non-Outliers for Weight (Elliptic Envelope)")
plt.xlabel("Outlier (Elliptic Envelope)")
plt.ylabel("Weight")
plt.savefig("figs/outlier_detection/EllipticEnvelope_variation_in_outliers.png")
plt.show()

# Count and display the number of outliers detected
n_outliers_elliptic = sum(outlier_labels_elliptic == -1)
print(f"Elliptic Envelope detected {n_outliers_elliptic} outliers out of {feature_data.shape[0]} samples.")

##Removal of outliers based on above results

common_outliers = (
        set(data[data['Outlier_Elliptic'] == -1].index) &
        set(data[data['Outlier_ISO'] == -1].index) &
        set(data[data['Outlier_LOF'] == -1].index)
)
print(f"Number of common outliers: {len(common_outliers)}")

data = data.drop(index=common_outliers)

data.to_csv('cleaned_data.csv', index=False)
feature_data = data.drop(columns='Outlier_Elliptic', inplace=False)
feature_data = data.drop(columns='Outlier_ISO', inplace=False)
feature_data = data.drop(columns='Outlier_LOF', inplace=False)
print(data.info())

# ----------------------------------------Feature Selection----------------------------------------

# Mutual Information
# mi_scores = mutual_info_classif(resampled_data[numerical_columns], resampled_data['NObeyesdad'])
# mi_scores_df = pd.DataFrame({'Feature': numerical_columns, 'MI Score': mi_scores}).sort_values(by='MI Score',
#                                                                                                ascending=False)

# print("Mutual Information Scores:")
# print(mi_scores_df)

# # Top feature based on the scores
# selected_features_mi = mi_scores_df[mi_scores_df['MI Score'] > 0.01]['Feature'].tolist()  # Example threshold
# print("Selected Features by Mutual Information:", selected_features_mi)


# ----------------------- Classification ---------------------
X = data.drop('NObeyesdad', axis=1)
y = data['NObeyesdad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'gamma': [0.1, 1, 10],
        'kernel': ['rbf', 'linear']
    },
    'KNN': {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    },
    'Decision Tree': {
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
}

classifiers = {
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, class_weight='balanced', random_state=42),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42)
}

best_classifiers = {}
best_score = 0.0

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# f = KFold(n_splits=5, shuffle=True, random_state=42)
# default_scores = {}
# for name, clf in classifiers.items():
#     clf.fit(X_train, y_train)  # Train with default hyperparameters
#     y_pred = clf.predict(X_test)
#     f1 = f1_score(y_test, y_pred, average='weighted')
#     default_scores[name] = f1
#     print(f"Model: {name}")
#     print("\nClassification Report:\n", classification_report(y_test, y_pred))

######## Hyperparameter Tuning ########
for name, clf in classifiers.items():
    grid_search = GridSearchCV(clf, param_grid[name], cv=kf, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_clf = grid_search.best_estimator_
    best_classifiers[name] = best_clf  # Store best classifier

    # Predict and calculate metrics
    y_pred = best_clf.predict(X_test)
    y_proba = best_clf.predict_proba(X_test)[:, 1] if hasattr(best_clf, "predict_proba") else None
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Model: {name}")
    print("\nClassification Report (ss):\n", classification_report(y_test, y_pred))

    # Display metrics
    # print(f"Model: {name}")
    # print(f"Recall: {recall}")
    # print(f"Accuracy: {accuracy}")
    # print(f"Precision: {precision}")
    # print(f"F1-Score: {f1}")
    # print("-" * 50)
    # print()

class_labels = ['Insuf', 'Normal', 'Obesi I', 'Obesi II', 'Obesi III', 'OverW I', 'OverW II']

fig, axes = plt.subplots(1, len(best_classifiers), figsize=(15, 4))

for ax, (model_name, best_model) in zip(axes, best_classifiers.items()):
    predictions = best_model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax,
                xticklabels=class_labels, yticklabels=class_labels)
    ax.set_title(f"{model_name}", weight='bold', size=13)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

plt.tight_layout()
plt.savefig('figs/Confusion_matrix_classification.png')
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(18, 12))
axs = axs.ravel()  # Flatten the axes for easy indexing

# Loop through best classifiers to plot ROC curves
for idx, (model_name, model) in enumerate(best_classifiers.items()):
    if hasattr(model, "predict_proba"):
        y_pred_prob = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        y_pred_prob = model.decision_function(X_test)
    else:
        print(f"{model_name} does not support probability or decision function output.")
        continue

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i, class_label in enumerate(model.classes_):
        fpr[i], tpr[i], _ = roc_curve(y_test == class_label, y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        axs[idx].plot(fpr[i], tpr[i], lw=2, label=f'Class {class_label} (AUC = {roc_auc[i]:.3f})')
        axs[idx].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        axs[idx].set_xlabel("False Positive Rate")
        axs[idx].set_ylabel("True Positive Rate")
        axs[idx].set_title(f"ROC Curve - {model_name}")
        axs[idx].legend(loc="best")

# Adjust layout and display
plt.tight_layout()
plt.title('ROC Curves')
plt.savefig('figs/ROC_Curves.png')
plt.show()
