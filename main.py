import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

file_path = 'dataset.csv'
data = pd.read_csv(file_path)

# Priting Data Info
print(data.info())
#print(data.isnull().sum())
data = data.drop_duplicates()

numerical_columns = ['Age', 'Height', 'Weight']

####   Normalization ####
    ## Min-Max Sacling 
scaler = MinMaxScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    ## Z-score Scaling 
#Standard_Scaler = StandardScaler()
#data[numerical_columns] = Standard_Scaler.fit_transform(data[numerical_columns])
print(data)

num_bins = 4
# Apply equal-depth (quantile) binning
for col in numerical_columns:
    data[f'{col}_equal_depth'] = pd.qcut(data[col], q=num_bins, labels=False, duplicates='drop')

print(data['Age_equal_depth'].unique())

data.to_csv('binned_data.csv', index=False)


# # Checking for null values and handling categorical values
# print(data.isnull().sum())
# print(data['Gender'].unique())

# print(data['family_history_with_overweight'].unique())

# print(data['FAVC'].unique())

# print(data['CAEC'].unique())

# print(data['SMOKE'].unique())

# print(data['SCC'].unique())

# print(data['CALC'].unique())

# print(data['MTRANS'].unique())

# print(data['NObeyesdad'].unique())



# # One hot coding
# ohe = OneHotEncoder(handle_unknown = 'error', sparse_output = False).set_output(transform = 'pandas')
# ohetransform = ohe.fit_transform(data)
