# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Load the dataset (assuming 'creditcard.csv' is the dataset)
df = pd.read_csv('C:/Users/hp/Downloads/65165165/archive(2)/creditcard.csv')

# Inspecting the dataset
print(df.head())
print(df.info())

# Since the dataset is highly imbalanced, let's look at the class distribution
fraud_cases = df[df['Class'] == 1]
non_fraud_cases = df[df['Class'] == 0]
print(f"Number of fraud cases: {len(fraud_cases)}")
print(f"Number of non-fraud cases: {len(non_fraud_cases)}")

# Visualizing the imbalance in the dataset
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution (0: Non-Fraud, 1: Fraud)')
plt.show()

# Feature Engineering: Scaling the features
# We need to scale the features for better performance of models like Logistic Regression
scaler = StandardScaler()
df['scaled_Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Dropping the original 'Time' and 'Amount' columns and using the scaled versions
df = df.drop(['Time', 'Amount'], axis=1)

# Features and target
X = df.drop('Class', axis=1)  # Feature variables
y = df['Class']  # Target variable (Class: 0 = Non-Fraud, 1 = Fraud)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 1. Logistic Regression for Anomaly Detection
lr = LogisticRegression(solver='liblinear')
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Evaluating the Logistic Regression model
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("ROC AUC Score:", roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]))
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# 2. Decision Tree Classifier for Anomaly Detection
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Evaluating the Decision Tree model
print("\nDecision Tree Classifier Accuracy:", accuracy_score(y_test, y_pred_dt))
print("ROC AUC Score:", roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1]))
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# Visualizing the decision tree
from sklearn.tree import plot_tree

plt.figure(figsize=(12, 8))
plot_tree(dt, filled=True, feature_names=X.columns, class_names=['Non-Fraud', 'Fraud'], max_depth=2, fontsize=10)
plt.title('Decision Tree (depth=2 for simplicity)')
plt.show()

# Feature Importance from Decision Tree
importances = dt.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10, 6))
plt.title("Feature Importance (Decision Tree)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.show()
