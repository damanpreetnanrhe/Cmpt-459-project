import pandas as pd
from sklearn.preprocessing import OneHotEncoder

file_path = 'dataset.csv'
data = pd.read_csv(file_path)

# Priting Data Info
print(data.info())

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

# One hot coding
ohe = OneHotEncoder(handle_unknown = 'error', sparse_output = False).set_output(transform = 'pandas')
ohetransform = ohe.fit_transform(data)
