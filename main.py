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

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
import os

pd.set_option('display.max_columns', None)

file_path = 'dataset.csv'
data = pd.read_csv(file_path)
print(data['NObeyesdad'].unique())

# # Priting Data Info
print(data.info())
# print(data.isnull().sum())
data = data.drop_duplicates()

numerical_columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
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

num_bins = 4
# Apply equal-depth (quantile) binning
for col in numerical_columns:
    data[f'{col}_equal_depth'] = pd.qcut(data[col], q=num_bins, labels=False, duplicates='drop')

print(data['Age_equal_depth'].unique())

data.to_csv('binned_data.csv', index=False)

# Checking for null values and handling categorical values
print(data.isnull().sum())
print(data['Gender'].unique())

print(data['family_history_with_overweight'].unique())
print(data['FAVC'].unique())
print(data['CAEC'].unique())
print(data['SMOKE'].unique())
print(data['SCC'].unique())
print(data['CALC'].unique())
print(data['MTRANS'].unique())
print(data['NObeyesdad'].unique())

print(data['NObeyesdad'].value_counts(normalize=True))

data['NObeyesdad'] = data['NObeyesdad'].replace(['Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II',
                                                 'Obesity_Type_I', 'Insufficient_Weight', 'Obesity_Type_II',
                                                 'Obesity_Type_III'], [0, 1, 1, 1, 0, 1, 1])

print(data['NObeyesdad'].unique())
print(data['NObeyesdad'].value_counts(normalize=True))

normal_weight_count = data[data['NObeyesdad'] == 0].shape[0]
print(normal_weight_count)

# ----------------------------------------EDA---------------------------------------
target_column = 'NObeyesdad'
for col in categorical_columns:
    if col != target_column:  # Skip the target column itself
        # Calculate mean of the target column grouped by the current categorical column
        group_mean = data.groupby(col)[target_column].mean()

        # Plotting
        group_mean.plot(kind='bar', title=f"Mean '{target_column}' by {col}")
        plt.xlabel(col)
        plt.ylabel(f"Mean {target_column}")
        plt.xticks(rotation=45)
        plt.show()

plt.figure(figsize=(10, 8))
correlation_matrix = data[numerical_columns + ['NObeyesdad']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

LabelEncoder = LabelEncoder()
for col in categorical_columns:
    data[col] = LabelEncoder.fit_transform(data[col])
# # print(data)

data = pd.get_dummies(data, columns=categorical_columns, drop_first=False)
data.to_csv('refined_data.csv', index=False)

# -------------------------------------Clustering-------------------------------------
X = data[numerical_columns]

# Apply PCA for dimensionality reduction (2 components for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# K-Means Clustering
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

# Create a DataFrame to summarize the evaluation metrics
results_df = pd.DataFrame({
    'Number of Clusters': list(cluster_range),
    'Silhouette Score': kmeans_silhouette_scores,
    'Calinski-Harabasz Index': kmeans_calinski_scores,
    'Davies-Bouldin Index': kmeans_davies_scores
})

print(results_df)

# DBSCAN clustering
# dbscan_silhouette_scores = []
# dbscan_calinski_scores = []
# dbscan_davies_scores = []
# dbscan_eps_values = []

# def plot_dbscan_clusters(X, labels, eps):
#     plt.figure(figsize=(10, 6))
#     scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
#     plt.title(f"DBSCAN Clustering (eps={eps})")
#     plt.xlabel('PCA Component 1')
#     plt.ylabel('PCA Component 2')
#     plt.colorbar(scatter)
#     plt.show()

# for eps in eps_values:
#     dbscan = DBSCAN(eps=eps, min_samples=10)
#     labels = dbscan.fit_predict(X)

#     # Skip evaluation if all points are classified as noise (-1)
#     if len(set(labels)) <= 1:
#         continue

#     # Evaluate the clustering performance
#     silhouette = silhouette_score(X, labels)
#     calinski = calinski_harabasz_score(X, labels)
#     davies = davies_bouldin_score(X, labels)

#     dbscan_silhouette_scores.append(silhouette)
#     dbscan_calinski_scores.append(calinski)
#     dbscan_davies_scores.append(davies)
#     dbscan_eps_values.append(eps)
#     plot_dbscan_clusters(X_pca, labels, eps)

# dbscan_results_df = pd.DataFrame({
#     'Eps Value': dbscan_eps_values,
#     'Silhouette Score': dbscan_silhouette_scores,
#     'Calinski-Harabasz Index': dbscan_calinski_scores,
#     'Davies-Bouldin Index': dbscan_davies_scores
# })

# print(dbscan_results_df)


# Hierarchical Clustering
n_clusters_range = range(2, 11)  # This will try 2 to 10 clusters

# Initialize lists to store results
silhouette_scores = []
calinski_scores = []
davies_scores = []

# Compute linkage matrix for the dendrogram
plt.figure(figsize=(12, 8))
linkage_matrix = linkage(X, method='ward')  # 'ward' linkage minimizes variance
dendrogram(linkage_matrix, truncate_mode="level", p=5)  # Show the first 5 levels of the dendrogram
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample index")
plt.ylabel("Distance")
plt.show()

# Perform Hierarchical Clustering for each number of clusters
for n_clusters in n_clusters_range:
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    agg_labels = agg_clustering.fit_predict(X)

    # Calculate and store the evaluation metrics
    silhouette_scores.append(silhouette_score(X, agg_labels))
    calinski_scores.append(calinski_harabasz_score(X, agg_labels))
    davies_scores.append(davies_bouldin_score(X, agg_labels))

# Create a DataFrame with the results
hierarchical_results_df = pd.DataFrame({
    'Number of Clusters': list(n_clusters_range),
    'Silhouette Score': silhouette_scores,
    'Calinski-Harabasz Index': calinski_scores,
    'Davies-Bouldin Index': davies_scores
})

# Plot the evaluation metrics
plt.figure(figsize=(15, 5))

# Silhouette Score plot
plt.subplot(1, 3, 1)
plt.plot(n_clusters_range, silhouette_scores, marker='o')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')

# Calinski-Harabasz Index plot
plt.subplot(1, 3, 2)
plt.plot(n_clusters_range, calinski_scores, marker='o', color='orange')
plt.title('Calinski-Harabasz Index vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Calinski-Harabasz Index')

# Davies-Bouldin Index plot
plt.subplot(1, 3, 3)
plt.plot(n_clusters_range, davies_scores, marker='o', color='green')
plt.title('Davies-Bouldin Index vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Davies-Bouldin Index')

plt.tight_layout()
plt.show()

# Visualize Agglomerative Clustering for a specific number of clusters (e.g., 4)
n_clusters_visualization = 4
agg_clustering_visualization = AgglomerativeClustering(n_clusters=n_clusters_visualization, metric='euclidean',
                                                       linkage='ward')
agg_labels_visualization = agg_clustering_visualization.fit_predict(X)

# Visualize Agglomerative Clustering results using PCA components
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agg_labels_visualization, cmap='viridis', s=50, alpha=0.7)
plt.title(f'Agglomerative Clustering (n_clusters={n_clusters_visualization})')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter)
plt.show()

# Display the DataFrame
print(hierarchical_results_df)

# -------------------------------------Outlier detection------------------------------------------------
numerical_data = data[numerical_columns]

# Isolation Forest 
iso_forest = IsolationForest(contamination=0.05, random_state=42)
data['Outlier_ISO'] = iso_forest.fit_predict(numerical_data)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Age', y='Weight', hue='Outlier_ISO', palette='coolwarm')
plt.title("Outlier Detection using Isolation Forest")
plt.xlabel("Age")
plt.ylabel("Weight")
plt.legend(title="Outlier")
plt.show()

print("Isolation Forest Outlier Counts:")
print(data['Outlier_ISO'].value_counts())

# Print the outliers
# outliers = data[data['Outlier_ISO'] == -1]
# print("Outliers detected by Isolation Forest:")
# print(outliers)

# Remove outliers
data = data[data['Outlier_ISO'] == 1].drop(columns=['Outlier_ISO'])
data.to_csv('cleaned_data.csv', index=False)

# ---- LOF ------
# Extract numerical data for LOF
X = data[numerical_columns]

# Apply Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
outlier_labels = lof.fit_predict(X)

# Add outlier labels to the dataframe
data['Outlier_LOF'] = outlier_labels

# Visualize the results using a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Age', y='Weight', hue='Outlier_LOF', palette='coolwarm')
plt.title("Outlier Detection using Local Outlier Factor")
plt.xlabel("Age")
plt.ylabel("Weight")
plt.legend(title="Outlier")
plt.show()

# Count and display the number of outliers detected
n_outliers = sum(outlier_labels == -1)
print(f"Local Outlier Factor detected {n_outliers} outliers out of {X.shape[0]} samples.")

# Filter outliers if needed
cleaned_data = data[data['Outlier_LOF'] == 1].drop(columns=['Outlier_LOF'])
print(f"Remaining data after outlier removal: {cleaned_data.shape}")

# Save cleaned data if necessary
cleaned_data.to_csv('cleaned_data.csv', index=False)

# ----------------------------------------Feature Selection----------------------------------------

# Mutual Information
mi_scores = mutual_info_classif(data[numerical_columns], data['NObeyesdad'])
mi_scores_df = pd.DataFrame({'Feature': numerical_columns, 'MI Score': mi_scores}).sort_values(by='MI Score',
                                                                                               ascending=False)

print("Mutual Information Scores:")
print(mi_scores_df)

# Top feature based on the scores
selected_features_mi = mi_scores_df[mi_scores_df['MI Score'] > 0.01]['Feature'].tolist()  # Example threshold
print("Selected Features by Mutual Information:", selected_features_mi)

# # Recursive Feature Elimination 
# rf = RandomForestClassifier(random_state=42)
# rfe = RFE(estimator=rf, n_features_to_select=5)  
# rfe.fit(data[numerical_columns], data['NObeyesdad'])
# selected_features_rfe = [feature for feature, selected in zip(numerical_columns, rfe.support_) if selected]
# print("Selected Features by RFE:", selected_features_rfe)


# ----------------------------------------Classification-------------------------------------------
# KNN classification
# ------------------------- Dataset Splitting -------------------------
X_selected = data[selected_features_mi]
y = data['NObeyesdad']

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

# ------------------------- Train k-NN Classifier -------------------------
knn = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", cv_scores.mean())

# Train the k-NN model on the training set
knn.fit(X_train, y_train)

# ------------------------- Make Predictions -------------------------
# Predict on the test set
y_pred = knn.predict(X_test)
y_proba = knn.predict_proba(X_test)[:, 1] if len(set(y)) == 2 else None  # Use probabilities for binary classification

# ------------------------- Evaluate the Model -------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nModel Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Calculate AUC-ROC if binary classification
if y_proba is not None:
    auc = roc_auc_score(y_test, y_proba)
    print(f"AUC-ROC: {auc:.4f}")

# ------------------------- Visualization -------------------------
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# ROC Curve (for binary classification)
if y_proba is not None:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle='--', color='red', label="Random Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

# X = data.drop(columns=['NObeyesdad'])
# print(X.info)
# y = data['NObeyesdad']

# os.makedirs("figs", exist_ok=True)
# classifiers = {
#     'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
# }

# # Define the parameter grid for hyperparameter tuning
# param_grid = {
#     'Random Forest': {
#         'n_estimators': [50, 100, 200],
#         'max_depth': [5, 10, None]
#     }
# }


# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)


# # Initialize best classifier and score
# best_classifier = None
# best_score = 0.0
# results = []
# kf = KFold(n_splits=5, shuffle=True, random_state=42)


# # Iterate over classifiers and perform grid search for hyperparameter tuning
# for name, clf in classifiers.items():
#     grid_search = GridSearchCV(clf, param_grid[name], cv=kf, scoring='f1_weighted', n_jobs=-1)
#     grid_search.fit(X_train, y_train)
#     best_clf = grid_search.best_estimator_
#     y_pred = best_clf.predict(X_test)
#     y_proba = best_clf.predict_proba(X_test)[:, 1] if hasattr(best_clf, "predict_proba") else None

#     # Calculate metrics
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, average='weighted')
#     recall = recall_score(y_test, y_pred, average='weighted')
#     f1 = f1_score(y_test, y_pred, average='weighted')
#     auc = roc_auc_score(y_test, y_proba, multi_class='ovr') if y_proba is not None else None

#     # Store results
#     results.append({
#         "Model": name,
#         "Accuracy": accuracy,
#         "Precision": precision,
#         "Recall": recall,
#         "F1-Score": f1,
#         "AUC-ROC": auc
#     })

#     # Update the best classifier if current model performs better
#     if accuracy > best_score:
#         best_score = accuracy
#         best_classifier = best_clf

#     # Plot confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#     disp.plot(cmap=plt.cm.Blues)
#     plt.title(f"Confusion Matrix - {name}")
#     plt.savefig(f'figs/confusion_matrix_{name}.png')
#     plt.show()

#     # Plot ROC curve if probabilities are available
#     if y_proba is not None:
#         fpr, tpr, _ = roc_curve(y_test, y_proba)
#         plt.figure(figsize=(10, 6))
#         plt.plot(fpr, tpr, color='blue', label='ROC curve')
#         plt.plot([0, 1], [0, 1], color='red', linestyle='--')
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title(f'ROC Curve - {name}')
#         plt.legend(loc='lower right')
#         plt.savefig(f'figs/roc_curve_{name}.png')
#         plt.show()

# # Print best classifier and evaluation metrics
# for result in results:
#     print(f"Model: {result['Model']}")
#     print(f"Accuracy: {result['Accuracy']:.4f}")
#     print(f"Precision: {result['Precision']:.4f}")
#     print(f"Recall: {result['Recall']:.4f}")
#     print(f"F1-Score: {result['F1-Score']:.4f}")
#     print(f"AUC-ROC: {result['AUC-ROC']:.4f}\n")

# print(f"Best Classifier: {best_classifier}\nBest Accuracy Score: {best_score:.4f}")

# ----------------------------------------Hyperparameter Tuning-------------------------------------------
# KNN classifier after tuning
# Define the parameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# Perform Grid Search
grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and cross-validation accuracy
print("\nBest Parameters from Grid Search:", grid_search.best_params_)
print("Best Cross-Validation Accuracy from Grid Search:", grid_search.best_score_)

# ----------------------------------- Tuned k-NN Model -----------------------------------
# Use the best model from Grid Search
best_knn = grid_search.best_estimator_

# Evaluate the tuned model
y_pred_tuned = best_knn.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
precision_tuned = precision_score(y_test, y_pred_tuned, average='weighted')
recall_tuned = recall_score(y_test, y_pred_tuned, average='weighted')
f1_tuned = f1_score(y_test, y_pred_tuned, average='weighted')

print("\nTuned Model Evaluation Metrics:")
print(f"Accuracy: {accuracy_tuned:.4f}")
print(f"Precision: {precision_tuned:.4f}")
print(f"Recall: {recall_tuned:.4f}")
print(f"F1-Score: {f1_tuned:.4f}")

# ----------------------------------- Visualization -----------------------------------
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_tuned)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_knn.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Tuned Model)")
plt.show()

# ROC Curve (for binary classification)
if len(set(y)) == 2:
    y_proba_tuned = best_knn.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba_tuned)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label="ROC Curve (Tuned)")
    plt.plot([0, 1], [0, 1], linestyle='--', color='red', label="Random Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Tuned Model)")
    plt.legend()
    plt.show()
