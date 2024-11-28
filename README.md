# Cmpt-459-project - Obesity Analysis in Latin America

This dataset focuses on estimating obesity levels in individuals from Mexico, Peru, and Colombia, based on their eating habits and physical condition. It consists of 2,111 records with 16 features and a target variable (`NObesity`) that represents obesity levels.

## Key Features:
- **Global Relevance:** Addresses obesity, a critical global health issue, with data sourced from diverse populations in three countries.
- **Data Integrity:** Nearly complete dataset with minimal missing values, ensuring reliability and straightforward preprocessing.
- **Feature Diversity:** Includes categorical, binary, and continuous variables, offering rich opportunities for analysis and feature engineering.
- **Synthetic Data:** 77% of the data is synthetically generated, preserving realistic patterns while addressing privacy concerns.
- **Cross-Cultural Analysis:** Covers data from multiple countries, enabling comparative and cross-cultural studies on obesity factors.
- **Machine Learning Ready:** With a clear target variable (`NObesity`), it supports various machine learning tasks such as classification, regression, and clustering.

### Language:
- Python 

### Libraries Used:
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Pathlib
- SciPy
- OS

## Data Preprocessing

#### Dataset Loading
- The dataset was loaded using **Pandas** for efficient data manipulation.
- Configured to display all columns for easier inspection during analysis.

#### Duplicate Removal
- Duplicate rows were identified and removed to maintain data integrity:
  ```python
  data = data.drop_duplicates()
  ```

#### Handling Missing Values
- Checked for missing values using:
  ```python
  data.isnull().sum()
  ```
- The analysis confirmed that there are no missing values in the dataset, as all columns returned a sum of `0`.

#### Normalization and Standardization
- After analyzing the dataset, feature scaling techniques such as **Min-Max Scaling** and **StandardScaler** were considered.
- However, the dataset's numerical features were already appropriately scaled, eliminating the need for additional scaling transformations.









