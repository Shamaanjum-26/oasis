# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 2: Load the dataset (CSV file)
# Replace 'ifood_df.csv' with the path to your dataset file
data = pd.read_csv('C:/Users/hp/Downloads/ifood_df.csv')

# Step 3: Data Exploration
print("Dataset Overview:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Handle missing values by filling with mean (you can modify the method based on your needs)
data.fillna(data.mean(), inplace=True)

# Step 4: Descriptive Statistics
print("\nDescriptive Statistics:")
print(data.describe())

# Key metrics like average purchase value, frequency of purchases
average_purchase_value = data['PurchaseValue'].mean()
purchase_frequency = data['PurchaseFrequency'].mean()

print(f"\nAverage Purchase Value: {average_purchase_value}")
print(f"Purchase Frequency: {purchase_frequency}")

# Step 5: Customer Segmentation using K-means
# Selecting relevant features for clustering
features = ['PurchaseValue', 'PurchaseFrequency']

# Standardizing the features for better clustering performance
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[features])

# Apply K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)  # You can adjust the number of clusters
data['Cluster'] = kmeans.fit_predict(scaled_features)

# Step 6: Visualization of the Segments

# Enhanced scatter plot with cluster centroids
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PurchaseValue', y='PurchaseFrequency', hue='Cluster', data=data, palette='viridis', s=100, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids', marker='X')
plt.title('Customer Segmentation by Purchase Patterns')
plt.xlabel('Purchase Value')
plt.ylabel('Purchase Frequency')
plt.legend()
plt.show()

# Visualizing the distribution of customers per cluster
plt.figure(figsize=(8, 5))
sns.countplot(x='Cluster', data=data, palette='viridis')
plt.title('Customer Count per Segment')
plt.show()

# Step 7: Correlation Heatmap

# Set the figure size
plt.figure(figsize=(10, 6))

# Compute the correlation matrix
corr_matrix = data[['PurchaseValue', 'PurchaseFrequency', 'Age', 'LoyaltyScore']].corr()

# Create a mask to hide the upper triangle of the heatmap (optional, for cleaner visualization)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Draw the heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, mask=mask, linewidths=0.5, fmt=".2f")

# Add titles and labels
plt.title('Correlation Heatmap of Features', fontsize=15)
plt.show()

# Step 8: Pairplot to explore relationships between all numerical features and clusters
sns.pairplot(data[['PurchaseValue', 'PurchaseFrequency', 'Age', 'LoyaltyScore', 'Cluster']], hue='Cluster', palette='viridis', diag_kind='kde')
plt.suptitle('Pairplot of Customer Features by Cluster', y=1.02)
plt.show()

# Step 9: Insights and Recommendations
print("\nCluster Insights:")
for cluster_num in sorted(data['Cluster'].unique()):
    cluster_data = data[data['Cluster'] == cluster_num]
    print(f"\nCluster {cluster_num} Summary:")
    print(cluster_data.describe())

# Further analysis could include more detailed segment insights
