# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Data Collection: Load the dataset
# Replace 'your_dataset.csv' with your actual file path
df = pd.read_csv('C:/Users/hp/Downloads/Housing.xls')

# 2. Data Exploration and Cleaning
# Display basic information about the dataset
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values (if any)
df = df.dropna()

# 3. Feature Selection
# Correlation matrix to see relationships between features and target
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Assuming the target variable is 'target_column' and features are 'feature1', 'feature2', etc.
X = df[['feature1', 'feature2', 'feature3']]  # Replace with your feature columns
y = df['target_column']  # Replace with your target column

# 4. Model Training
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# 5. Model Evaluation
# Predict the target values for the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# 6. Visualization: Plot the actual vs predicted values
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Ideal Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()
