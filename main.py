import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

file_path = 'dataset.csv'
data = pd.read_csv(file_path)
print(data['NObeyesdad'].unique())

# Priting Data Info
print(data.info())
#print(data.isnull().sum())
data = data.drop_duplicates()

numerical_columns = ['Age', 'Height', 'Weight', 'FCVC','NCP', 'CH2O', 'FAF', 'TUE']
categorical_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE' , 'SCC', 'CALC', 'MTRANS']
####   Normalization ####
    ## Min-Max Sacling 
scaler = MinMaxScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
print(data)

#     ## Z-score Scaling 
# #Standard_Scaler = StandardScaler()
# #data[numerical_columns] = Standard_Scaler.fit_transform(data[numerical_columns])
# print(data)

# num_bins = 4
# # Apply equal-depth (quantile) binning
# for col in numerical_columns:
#     data[f'{col}_equal_depth'] = pd.qcut(data[col], q=num_bins, labels=False, duplicates='drop')

# print(data['Age_equal_depth'].unique())

# data.to_csv('binned_data.csv', index=False)


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



print(data['NObeyesdad'].value_counts(normalize=True))

data['NObeyesdad'] = data['NObeyesdad'].replace(['Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II',
 'Obesity_Type_I', 'Insufficient_Weight', 'Obesity_Type_II',
 'Obesity_Type_III'], [1, 2 ,2, 3, 0, 3, 3])

print(data['NObeyesdad'].unique())
print(data['NObeyesdad'].value_counts(normalize=True))

target_column = 'NObeyesdad'

LabelEncoder = LabelEncoder()
for col in categorical_columns:
    data[col] = LabelEncoder.fit_transform(data[col])
print(data)

##EDA
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




# LabelEncoder = LabelEncoder()
# for col in categorical_columns:
#     data[col] = LabelEncoder.fit_transform(data[col])
# print(data)

# data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
# print(data)

# data.to_csv('refined_data.csv', index=False)







