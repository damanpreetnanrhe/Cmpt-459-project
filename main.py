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

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve, ConfusionMatrixDisplay)

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
#print(data.isnull().sum())
data = data.drop_duplicates()

numerical_columns = ['Age', 'Height', 'Weight', 'FCVC','NCP', 'CH2O', 'FAF', 'TUE']
categorical_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE' , 'SCC', 'CALC', 'MTRANS']
# ####   Normalization ####
    ## Min-Max Sacling 
# scaler = MinMaxScaler()
# data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
# print(data)

    ## Z-score Scaling 
# Standard_Scaler = StandardScaler()
# data[numerical_columns] = Standard_Scaler.fit_transform(data[numerical_columns])
# print(data)

# # num_bins = 4
# # # Apply equal-depth (quantile) binning
# # for col in numerical_columns:
# #     data[f'{col}_equal_depth'] = pd.qcut(data[col], q=num_bins, labels=False, duplicates='drop')

# # print(data['Age_equal_depth'].unique())

# # data.to_csv('binned_data.csv', index=False)


# # # Checking for null values and handling categorical values
# # print(data.isnull().sum())
# # print(data['Gender'].unique())

# # print(data['family_history_with_overweight'].unique())

# # print(data['FAVC'].unique())

# # print(data['CAEC'].unique())

# # print(data['SMOKE'].unique())

# # print(data['SCC'].unique())

# # print(data['CALC'].unique())

# # print(data['MTRANS'].unique())

# # print(data['NObeyesdad'].unique())



# print(data['NObeyesdad'].value_counts(normalize=True))

data['NObeyesdad'] = data['NObeyesdad'].replace(['Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II',
 'Obesity_Type_I', 'Insufficient_Weight', 'Obesity_Type_II',
 'Obesity_Type_III'], [0, 1,1, 1, 0, 1, 1])

print(data['NObeyesdad'].unique())
print(data['NObeyesdad'].value_counts(normalize=True))

normal_weight_count = data[data['NObeyesdad'] == 0].shape[0]
print(normal_weight_count)


# ##EDA
# # target_column = 'NObeyesdad'
# # for col in categorical_columns:
# #     if col != target_column:  # Skip the target column itself
# #         # Calculate mean of the target column grouped by the current categorical column
# #         group_mean = data.groupby(col)[target_column].mean()

# #         # Plotting
# #         group_mean.plot(kind='bar', title=f"Mean '{target_column}' by {col}")
# #         plt.xlabel(col)
# #         plt.ylabel(f"Mean {target_column}")
# #         plt.xticks(rotation=45)
# #         plt.show()


# # plt.figure(figsize=(10, 8))
# # correlation_matrix = data[numerical_columns + ['NObeyesdad']].corr()
# # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# # plt.title('Correlation Heatmap')
# # plt.show()


LabelEncoder = LabelEncoder()
for col in categorical_columns:
    data[col] = LabelEncoder.fit_transform(data[col])
# # print(data)

data = pd.get_dummies(data, columns=categorical_columns, drop_first=False)
# data.to_csv('refined_data.csv', index=False)


##Clustering
# X = data[numerical_columns]

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


## Classification 


# target_column = 'NObeyesdad'
# X = data.drop(columns=[target_column])
# y_resampled = data[target_column]

# scaler_dump = MinMaxScaler().fit(X)
# scaler = MinMaxScaler().fit_transform(X)
# X = pd.DataFrame(scaler, columns=X.columns)
# X = X.astype(np.float64)

# classifiers = {
#     'Random Forest': RandomForestClassifier(),
#     'Gradient Boosting': GradientBoostingClassifier(),
#     'SVM': SVC(),
#     'KNN': KNeighborsClassifier(),
#     'Logistic Regression': LogisticRegression(),
#     'Decision Tree': DecisionTreeClassifier()
# }

# # Define the parameter grid for hyperparameter tuning
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
#     'Logistic Regression': {
#     'C': [0.1, 1, 10],
#     'solver': ['liblinear', 'lbfgs'],
#     'max_iter': [200, 500]  # Increase max_iter
# },
#     'Decision Tree': {
#         'max_depth': [5, 10, None],
#         'min_samples_split': [2, 5, 10]
#     }
# }

# X_train, X_test, y_train, y_test = train_test_split(X, y_resampled, random_state=42, test_size=0.25)


# # Initialize best classifier and score
# best_classifier = None
# best_score = 0.0

# # Iterate over classifiers and perform grid search for hyperparameter tuning
# for name, clf in classifiers.items():
#     grid_search = GridSearchCV(clf, param_grid[name], cv=5, scoring='f1_weighted')
#     grid_search.fit(X_train, y_train)
#     best_clf = grid_search.best_estimator_
#     y_pred = best_clf.predict(X_test)
#     score = accuracy_score(y_test, y_pred)
#     if score > best_score:
#         best_score = score
#         best_classifier = best_clf


# from sklearn.preprocessing import label_binarize
# from sklearn.metrics import roc_auc_score, roc_curve, auc
# ## Binarize the output labels
# y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
# n_classes = y_test_binarized.shape[1]

# # Compute ROC curve and AUC for each class
# fpr = {}
# tpr = {}
# roc_auc = {}

# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], best_classifier.predict_proba(X_test)[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Plot ROC curve for each class
# plt.figure(figsize=(10, 8))
# colors = ['blue', 'green', 'orange', 'red']
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

# plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve (One-vs-Rest)')
# plt.legend(loc='lower right')
# plt.show()


X = data.drop(columns=['NObeyesdad'])
print(X.info)
y = data['NObeyesdad']

# # Split the dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define models
# models = {
#     "Random Forest": RandomForestClassifier(random_state=42),
#     "K-Nearest Neighbors": KNeighborsClassifier(),
#     "Support Vector Machine": SVC(probability=True, random_state=42)
# }

# # Initialize metrics storage
# results = []

# # Cross-validation setup
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# # Model evaluation
# for model_name, model in models.items():
#     # Cross-validation scores
#     cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
    
#     # Fit model and make predictions
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

#     # Metrics calculation
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, average='weighted')
#     recall = recall_score(y_test, y_pred, average='weighted')
#     f1 = f1_score(y_test, y_pred, average='weighted')
#     auc = roc_auc_score(y_test, y_proba, multi_class='ovr') if y_proba is not None else None

#     # Store the results
#     results.append({
#         "Model": model_name,
#         "Cross-Validation Accuracy": np.mean(cv_scores),
#         "Accuracy": accuracy,
#         "Precision": precision,
#         "Recall": recall,
#         "F1-Score": f1,
#         "AUC-ROC": auc
#     })

#     # Plot confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#     disp.plot()
#     plt.title(f"Confusion Matrix - {model_name}")
#     plt.show()

# # Convert results to DataFrame and display
# results_df = pd.DataFrame(results)
# print(results_df)

os.makedirs("figs", exist_ok=True)
classifiers = {
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, class_weight='balanced', random_state=42),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42)
}

# Define the parameter grid for hyperparameter tuning
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


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)


# Initialize best classifier and score
best_classifier = None
best_score = 0.0
results = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)


# Iterate over classifiers and perform grid search for hyperparameter tuning
for name, clf in classifiers.items():
    grid_search = GridSearchCV(clf, param_grid[name], cv=kf, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(X_test)
    y_proba = best_clf.predict_proba(X_test)[:, 1] if hasattr(best_clf, "predict_proba") else None

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_proba, multi_class='ovr') if y_proba is not None else None

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "AUC-ROC": auc
    })

    # Update the best classifier if current model performs better
    if accuracy > best_score:
        best_score = accuracy
        best_classifier = best_clf

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(f'figs/confusion_matrix_{name}.png')
    plt.show()

    # Plot ROC curve if probabilities are available
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='blue', label='ROC curve')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc='lower right')
        plt.savefig(f'figs/roc_curve_{name}.png')
        plt.show()

# Print best classifier and evaluation metrics
for result in results:
    print(f"Model: {result['Model']}")
    print(f"Accuracy: {result['Accuracy']:.4f}")
    print(f"Precision: {result['Precision']:.4f}")
    print(f"Recall: {result['Recall']:.4f}")
    print(f"F1-Score: {result['F1-Score']:.4f}")
    print(f"AUC-ROC: {result['AUC-ROC']:.4f}\n")

print(f"Best Classifier: {best_classifier}\nBest Accuracy Score: {best_score:.4f}")