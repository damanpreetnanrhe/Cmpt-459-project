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
from imblearn.over_sampling import SMOTE,RandomOverSampler
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import os

pd.set_option('display.max_columns', None)

file_path = 'dataset.csv'
data = pd.read_csv(file_path)
print(data['NObeyesdad'].unique())

# # Priting Data Info
print(data.info())
# print(data.isnull().sum())
data = data.drop_duplicates()


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
    'SCC' : 'Calories consumption monitoring (SCC)',    
}

data_copy.rename(columns=new_column_names, inplace=True)
print(data_copy.info())
data_copy.to_csv('refined_data.csv', index = False)

# LabelEncoder = LabelEncoder()
# for col in categorical_columns:
#     data[col] = LabelEncoder.fit_transform(data[col])
# # print(data)


####### Average age of each obesity type #########
# print(data.groupby("NObeyesdad")['Age'].median())
# data.groupby("NObeyesdad")["Age"].median().sort_values(ascending=False).plot(kind="bar",color = sns.color_palette("Set1"))
# plt.title("Average age of each obesity type")
# plt.savefig("figs/average_age_obesity_type.png")
# plt.show()


####### Average weight of each obesity type #########
print(data.groupby("NObeyesdad")['Weight'].median())
data.groupby("NObeyesdad")["Weight"].median().sort_values(ascending=False).plot(kind="bar",color=sns.color_palette("Set2"))
plt.title("Average Weight of each obesity type")
plt.savefig("figs/average_weight_obesity_type.png")
plt.show()

####### How is obesity type affected by eating high calorie food? #########
print(data.groupby(['NObeyesdad', 'FAVC'])["FAVC"].count())
plt.figure(figsize=(10,7))
sns.countplot(data=data,x=data.NObeyesdad,hue=data.FAVC,palette=sns.color_palette("Dark2"))
plt.xticks(rotation=-20)
plt.title("How is obesity type affected by eating high calorie food?")
plt.savefig("figs/obesity_type_eating_high_calorie_food.png")
plt.show()

####### Does family history with overweight affect obesity type? #########
plt.figure(figsize=(10,7))
sns.countplot(data=data,x=data.NObeyesdad,hue=data.family_history_with_overweight,palette=sns.color_palette("Dark2"))
plt.xticks(rotation=-20)
plt.title("Does family history with overweight affect obesity type?")
plt.savefig("figs/family_history_with_overweight_obesity_type.png")
plt.show()

####### Correlation between data atributes #########
corr_data = data.copy()
encoder  = LabelEncoder()
for col in corr_data.select_dtypes(include="object").columns:
    corr_data[col] =encoder.fit_transform(corr_data[col])

plt.figure(figsize=(16,13))
sns.heatmap(data=corr_data.corr(),annot=True)
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

# # -------------------------------------Clustering-------------------------------------
# X = data

# # Apply PCA for dimensionality reduction (2 components for visualization)
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)

# # K-Means Clustering
# kmeans = KMeans(n_clusters=4, random_state=42)
# kmeans_labels = kmeans.fit_predict(X)

# cluster_range = range(2, 10)
# eps_values = [0.3, 0.5, 0.7, 0.9, 1.1]

# kmeans_silhouette_scores = []
# kmeans_calinski_scores = []
# kmeans_davies_scores = []


# def plot_clusters(X, labels, title):
#     plt.figure(figsize=(10, 6))
#     scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
#     plt.title(title)
#     plt.xlabel('PCA Component 1')
#     plt.ylabel('PCA Component 2')
#     plt.colorbar(scatter)
#     plt.show()


# for k in cluster_range:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     labels = kmeans.fit_predict(X)
#     silhouette = silhouette_score(X, labels)
#     calinski = calinski_harabasz_score(X, labels)
#     davies = davies_bouldin_score(X, labels)
#     kmeans_silhouette_scores.append(silhouette)
#     kmeans_calinski_scores.append(calinski)
#     kmeans_davies_scores.append(davies)

#     plot_clusters(X_pca, labels, f"K-Means Clustering (n_clusters={k})")

# # Create a DataFrame to summarize the evaluation metrics
# results_df = pd.DataFrame({
#     'Number of Clusters': list(cluster_range),
#     'Silhouette Score': kmeans_silhouette_scores,
#     'Calinski-Harabasz Index': kmeans_calinski_scores,
#     'Davies-Bouldin Index': kmeans_davies_scores
# })

# print(results_df)

# # DBSCAN clustering
# # Analyze kNN distances for estimating optimal eps
# neighbors = NearestNeighbors(n_neighbors=10) 
# neighbors_fit = neighbors.fit(X)
# distances, _ = neighbors_fit.kneighbors(X)
# distances = np.sort(distances[:, -1])  

# # Plot kNN distances
# plt.figure(figsize=(10, 6))
# plt.plot(distances)
# plt.title("kNN Distance Plot")
# plt.xlabel("Points Sorted by Distance")
# plt.ylabel("Distance to 10th Nearest Neighbor")
# plt.grid(True)
# plt.show()

# eps_values = [4.0, 4.2, 4.5, 4.8, 5.0] # Adjust this based on kNN plot
# min_samples = 10  
# dbscan_results = []

# for eps in eps_values:
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#     labels = dbscan.fit_predict(X)

#     num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

#     if num_clusters <= 1:
#         print(f"DBSCAN skipped for eps={eps} (all points classified as noise).")
#         dbscan_results.append({
#             'Eps': eps,
#             'Min Samples': min_samples,
#             'Clusters': num_clusters,
#             'Silhouette Score': None,
#             'Calinski-Harabasz Index': None,
#             'Davies-Bouldin Index': None
#         })
#         continue

#     silhouette = silhouette_score(X, labels)
#     calinski = calinski_harabasz_score(X, labels)
#     davies = davies_bouldin_score(X, labels)

#     dbscan_results.append({
#         'Eps': eps,
#         'Min Samples': min_samples,
#         'Clusters': num_clusters,
#         'Silhouette Score': silhouette,
#         'Calinski-Harabasz Index': calinski,
#         'Davies-Bouldin Index': davies
#     })

#     plt.figure(figsize=(10, 6))
#     scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
#     plt.title(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})")
#     plt.xlabel('PCA Component 1')
#     plt.ylabel('PCA Component 2')
#     plt.colorbar(scatter)
#     plt.grid(True)
#     plt.show()

# dbscan_results_df = pd.DataFrame(dbscan_results)
# print("DBSCAN Tuning Results:")
# print(dbscan_results_df)


# # Hierarchical Clustering
# n_clusters_range = range(2, 11)  # This will try 2 to 10 clusters

# # Initialize lists to store results
# silhouette_scores = []
# calinski_scores = []
# davies_scores = []

# # Compute linkage matrix for the dendrogram
# plt.figure(figsize=(12, 8))
# linkage_matrix = linkage(X, method='ward')  # 'ward' linkage minimizes variance
# dendrogram(linkage_matrix, truncate_mode="level", p=5)  # Show the first 5 levels of the dendrogram
# plt.title("Hierarchical Clustering Dendrogram")
# plt.xlabel("Sample index")
# plt.ylabel("Distance")
# plt.show()

# # Perform Hierarchical Clustering for each number of clusters
# for n_clusters in n_clusters_range:
#     agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
#     agg_labels = agg_clustering.fit_predict(X)

#     # Calculate and store the evaluation metrics
#     silhouette_scores.append(silhouette_score(X, agg_labels))
#     calinski_scores.append(calinski_harabasz_score(X, agg_labels))
#     davies_scores.append(davies_bouldin_score(X, agg_labels))

# # Create a DataFrame with the results
# hierarchical_results_df = pd.DataFrame({
#     'Number of Clusters': list(n_clusters_range),
#     'Silhouette Score': silhouette_scores,
#     'Calinski-Harabasz Index': calinski_scores,
#     'Davies-Bouldin Index': davies_scores
# })

# # Plot the evaluation metrics
# plt.figure(figsize=(15, 5))

# # Silhouette Score plot
# plt.subplot(1, 3, 1)
# plt.plot(n_clusters_range, silhouette_scores, marker='o')
# plt.title('Silhouette Score vs. Number of Clusters')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Silhouette Score')

# # Calinski-Harabasz Index plot
# plt.subplot(1, 3, 2)
# plt.plot(n_clusters_range, calinski_scores, marker='o', color='orange')
# plt.title('Calinski-Harabasz Index vs. Number of Clusters')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Calinski-Harabasz Index')

# # Davies-Bouldin Index plot
# plt.subplot(1, 3, 3)
# plt.plot(n_clusters_range, davies_scores, marker='o', color='green')
# plt.title('Davies-Bouldin Index vs. Number of Clusters')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Davies-Bouldin Index')

# plt.tight_layout()
# plt.show()

# # Visualize Agglomerative Clustering for a specific number of clusters (e.g., 4)
# n_clusters_visualization = 4
# agg_clustering_visualization = AgglomerativeClustering(n_clusters=n_clusters_visualization, metric='euclidean',
#                                                        linkage='ward')
# agg_labels_visualization = agg_clustering_visualization.fit_predict(X)

# # Visualize Agglomerative Clustering results using PCA components
# plt.figure(figsize=(10, 6))
# scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=agg_labels_visualization, cmap='viridis', s=50, alpha=0.7)
# plt.title(f'Agglomerative Clustering (n_clusters={n_clusters_visualization})')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.colorbar(scatter)
# plt.show()

# # Display the DataFrame
# print(hierarchical_results_df)

# -------------------------------------Outlier detection------------------------------------------------
# numerical_data = data
# iso_forest = IsolationForest()

# # Isolation Forest 
# iso_forest = IsolationForest(contamination=0.05, random_state=42)
# data['Outlier_ISO'] = iso_forest.fit_predict(numerical_data)

# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=data, x='Age', y='Weight', hue='Outlier_ISO', palette='coolwarm')
# plt.title("Outlier Detection using Isolation Forest")
# plt.xlabel("Age")
# plt.ylabel("Weight")
# plt.legend(title="Outlier")
# plt.show()

# print("Isolation Forest Outlier Counts:")
# print(data['Outlier_ISO'].value_counts())

# # Print the outliers
# # outliers = data[data['Outlier_ISO'] == -1]
# # print("Outliers detected by Isolation Forest:")
# # print(outliers)

# # Remove outliers
# data = data[data['Outlier_ISO'] == 1].drop(columns=['Outlier_ISO'])
# data.to_csv('cleaned_data.csv', index=False)

# # ---- LOF ------
# # Extract numerical data for LOF
# X = data[numerical_columns]

# # Apply Local Outlier Factor
# lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
# outlier_labels = lof.fit_predict(X)

# # Add outlier labels to the dataframe
# data['Outlier_LOF'] = outlier_labels

# # Visualize the results using a scatter plot
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=data, x='Age', y='Weight', hue='Outlier_LOF', palette='coolwarm')
# plt.title("Outlier Detection using Local Outlier Factor")
# plt.xlabel("Age")
# plt.ylabel("Weight")
# plt.legend(title="Outlier")
# plt.show()

# # Count and display the number of outliers detected
# n_outliers = sum(outlier_labels == -1)
# print(f"Local Outlier Factor detected {n_outliers} outliers out of {X.shape[0]} samples.")

# # Filter outliers if needed
# data = data[data['Outlier_LOF'] == 1].drop(columns=['Outlier_LOF'])
# print(f"Remaining data after outlier removal: {data.shape}")

# # Save cleaned data if necessary
# data.to_csv('cleaned_data.csv', index=False)


# ----------------------------------------SMOTE----------------------------------------------------
#uplifting minority
# X_temp = data.drop(columns = 'NObeyesdad')
# y = data.NObeyesdad
# ros = RandomOverSampler(sampling_strategy = 'minority')
# X_resampled, y_resampled = ros.fit_resample(X_temp, y)
# resampled_data = pd.concat([pd.DataFrame(X_resampled, columns=X_temp.columns), pd.Series(y_resampled, name='NObeyesdad')], axis=1)
# print(resampled_data.info())

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
# X = data.drop('NObeyesdad', axis=1)
# y = data['NObeyesdad']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# param_grid = {
#     'Random Forest': {
#         'n_estimators': [50, 100, 200],
#         'max_depth': [5, 10, None]
#     },
#     'Gradient Boosting': {
#         'n_estimators': [50, 100, 200],
#         'learning_rate': [0.01, 0.1, 1.0]
#     },
#     'SVM': {
#         'C': [0.1, 1, 10],
#         'gamma': [0.1, 1, 10],
#         'kernel': ['rbf', 'linear']
#     },
#     'KNN': {
#         'n_neighbors': [3, 5, 7],
#         'weights': ['uniform', 'distance']
#     },
#     'Decision Tree': {
#         'max_depth': [5, 10, None],
#         'min_samples_split': [2, 5, 10]
#     }
# }

# classifiers = {
#     'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
#     'Gradient Boosting': GradientBoostingClassifier(random_state=42),
#     'SVM': SVC(probability=True, class_weight='balanced', random_state=42),
#     'KNN': KNeighborsClassifier(),
#     'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42)
# }

# best_classifiers = {}
# best_score = 0.0

# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# # f = KFold(n_splits=5, shuffle=True, random_state=42)
# # default_scores = {}
# # for name, clf in classifiers.items():
# #     clf.fit(X_train, y_train)  # Train with default hyperparameters
# #     y_pred = clf.predict(X_test)
# #     f1 = f1_score(y_test, y_pred, average='weighted')
# #     default_scores[name] = f1
# #     print(f"Model: {name}")
# #     print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ######## Hyperparameter Tuning ########
# for name, clf in classifiers.items():
#     grid_search = GridSearchCV(clf, param_grid[name], cv=kf, scoring='f1_weighted', n_jobs=-1)
#     grid_search.fit(X_train, y_train)
#     best_clf = grid_search.best_estimator_
#     best_classifiers[name] = best_clf  # Store best classifier

#     # Predict and calculate metrics
#     y_pred = best_clf.predict(X_test)
#     y_proba = best_clf.predict_proba(X_test)[:, 1] if hasattr(best_clf, "predict_proba") else None
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, average='weighted')
#     recall = recall_score(y_test, y_pred, average='weighted')
#     f1 = f1_score(y_test, y_pred, average='weighted')
#     print(f"Model: {name}")
#     print("\nClassification Report (ss):\n", classification_report(y_test, y_pred))


#     # Display metrics
#     # print(f"Model: {name}")
#     # print(f"Recall: {recall}")
#     # print(f"Accuracy: {accuracy}")
#     # print(f"Precision: {precision}")
#     # print(f"F1-Score: {f1}")
#     # print("-" * 50)
#     # print()




# class_labels = ['Insuf', 'Normal', 'Obesi I', 'Obesi II', 'Obesi III', 'OverW I', 'OverW II']

# fig, axes = plt.subplots(1, len(best_classifiers), figsize=(15, 4))

# for ax, (model_name, best_model) in zip(axes, best_classifiers.items()):
#     predictions = best_model.predict(X_test)
#     cm = confusion_matrix(y_test, predictions)
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax, 
#                 xticklabels=class_labels, yticklabels=class_labels)
#     ax.set_title(f"{model_name}", weight='bold', size=13)
#     ax.set_xlabel("Predicted")
#     ax.set_ylabel("True")

# plt.tight_layout()
# plt.savefig('figs/Confusion_matrix_classification.png')
# plt.show()


# fig, axs = plt.subplots(2, 3, figsize=(18, 12))
# axs = axs.ravel()  # Flatten the axes for easy indexing

# # Loop through best classifiers to plot ROC curves
# for idx, (model_name, model) in enumerate(best_classifiers.items()):
#     if hasattr(model, "predict_proba"):
#         y_pred_prob = model.predict_proba(X_test)
#     elif hasattr(model, "decision_function"):
#         y_pred_prob = model.decision_function(X_test)
#     else:
#         print(f"{model_name} does not support probability or decision function output.")
#         continue

#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()

#     for i, class_label in enumerate(model.classes_):
#         fpr[i], tpr[i], _ = roc_curve(y_test == class_label, y_pred_prob[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#         axs[idx].plot(fpr[i], tpr[i], lw=2, label=f'Class {class_label} (AUC = {roc_auc[i]:.3f})')

#     axs[idx].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
#     axs[idx].set_xlabel("False Positive Rate")
#     axs[idx].set_ylabel("True Positive Rate")
#     axs[idx].set_title(f"ROC Curve - {model_name}")
#     axs[idx].legend(loc="best")

# # Adjust layout and display
# plt.tight_layout()
# plt.title('ROC Curves')
# plt.savefig('figs/ROC_Curves.png')
# plt.show()
