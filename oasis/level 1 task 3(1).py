# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load the dataset
# Replace with your actual dataset path
dataset_path = 'C:/Users/hp/Downloads/New York City Airbnb Open Data/archive/AB_NYC_2019.csv'  # Change this path
data = pd.read_csv(dataset_path)

# Step 3: Data Exploration
print("Dataset Overview:")
print(data.head())
print("\nDataset Info:")
print(data.info())

# Step 4: Missing Data Handling
print("\nMissing Values Before Handling:")
print(data.isnull().sum())

# Impute missing values (mean imputation for numerical columns)
data.fillna(data.mean(), inplace=True)

# Alternatively, you could drop rows with missing values
# data.dropna(inplace=True)

print("\nMissing Values After Handling:")
print(data.isnull().sum())

# Step 5: Duplicate Removal
initial_count = len(data)
data.drop_duplicates(inplace=True)
final_count = len(data)
print(f"\nRemoved {initial_count - final_count} duplicate rows.")

# Step 6: Standardization
# Standardizing numerical columns (e.g., if you have columns like 'Age', 'Salary')
numerical_cols = ['Age', 'Salary']  # Replace with your actual numerical columns
data[numerical_cols] = (data[numerical_cols] - data[numerical_cols].mean()) / data[numerical_cols].std()

# Step 7: Outlier Detection using IQR
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

outliers = detect_outliers_iqr(data, 'Salary')  # Replace 'Salary' with your numerical column

# Step 8: Visualization of Outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['Salary'])  # Replace 'Salary' with your numerical column
plt.title('Box Plot of Salary with Outliers')
plt.savefig('C:/Users/hp/Downloads/outliers_visualization.png')  # Change the path if needed
plt.show()

# Print outlier details
print("\nOutliers Detected:")
print(outliers)

# Step 9: Final Data Overview
print("\nFinal Dataset Overview:")
print(data.describe())
