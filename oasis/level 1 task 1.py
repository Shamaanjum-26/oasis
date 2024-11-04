import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



dataset1_path = 'C:/Users/hp/Downloads/retail sales dataset/archive/retail_sales_dataset.csv'
dataset2_path = 'C:/Users/hp/Downloads/nutrition facts/archive/menu.csv'        

try:
    dataset1 = pd.read_csv(dataset1_path)
    print(f"Successfully loaded Dataset1 from {dataset1_path}\n")
except FileNotFoundError:
    print(f"Error: Dataset1 not found at {dataset1_path}")
    exit(1)

try:
    dataset2 = pd.read_csv(dataset2_path)
    print(f"Successfully loaded Dataset2 from {dataset2_path}\n")
except FileNotFoundError:
    print(f"Error: Dataset2 not found at {dataset2_path}")
    exit(1)

print("Dataset1 - Initial Rows:")
print(dataset1.head(), "\n")

print("Dataset1 - Data Types and Non-Null Counts:")
print(dataset1.info(), "\n")

print("Dataset2 - Initial Rows:")
print(dataset2.head(), "\n")

print("Dataset2 - Data Types and Non-Null Counts:")
print(dataset2.info(), "\n")

print("Dataset1 Columns:")
print(dataset1.columns.tolist(), "\n")

print("Dataset2 Columns:")
print(dataset2.columns.tolist(), "\n")

print("Handling Missing Values in Dataset1:")
missing_values = dataset1.isnull().sum()
print(missing_values, "\n")

dataset1.ffill(inplace=True)

print("Missing Values After Forward Fill:")
print(dataset1.isnull().sum(), "\n")

initial_duplicates = dataset1.duplicated().sum()
dataset1.drop_duplicates(inplace=True)
final_duplicates = dataset1.duplicated().sum()
print(f"Number of duplicates before removal: {initial_duplicates}")
print(f"Number of duplicates after removal: {final_duplicates}\n")

if 'Date' in dataset1.columns:
    dataset1['Date'] = pd.to_datetime(dataset1['Date'], errors='coerce')

    if dataset1['Date'].isnull().any():
        print("Warning: Some dates could not be converted and are set as NaT.\n")
else:
    print("Error: 'Date' column not found in Dataset1.")
    exit(1)

numeric_columns = dataset1.select_dtypes(include=[np.number]).columns.tolist()
print("Numeric Columns in Dataset1:")
print(numeric_columns, "\n")

for col in numeric_columns:
    if dataset1[col].dtype == 'object':
        print(f"Column '{col}' has non-numeric data. Attempting to convert.")
        dataset1[col] = pd.to_numeric(dataset1[col], errors='coerce')
        print(f"Conversion completed for column '{col}'.\n")

numeric_columns = dataset1.select_dtypes(include=[np.number]).columns.tolist()
print("Updated Numeric Columns in Dataset1:")
print(numeric_columns, "\n")


# ----------------------- Descriptive Statistics -----------------------

descriptive_stats = dataset1[numeric_columns].describe()
print("Basic Descriptive Statistics:")
print(descriptive_stats, "\n")

median_values = dataset1[numeric_columns].median()
print("Median Values:")
print(median_values, "\n")

mode_values = dataset1.mode().iloc[0]
print("Mode Values:")
print(mode_values, "\n")

std_dev = dataset1[numeric_columns].std()
print("Standard Deviation:")
print(std_dev, "\n")


# ----------------------- Time Series Analysis -----------------------

sales_over_time = None  

if {'Date', 'Sales'}.issubset(dataset1.columns):

    sales_over_time = dataset1.groupby('Date')['Sales'].sum().reset_index()
    print("Sales aggregated over time:\n")
    print(sales_over_time.head(), "\n")

    plt.figure(figsize=(12,6))
    sns.lineplot(x='Date', y='Sales', data=sales_over_time, marker='o')
    plt.title('Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    try:
        decomposition = seasonal_decompose(sales_over_time['Sales'], model='additive', period=12)
        decomposition.plot()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Time Series Decomposition Error:", e, "\n")
else:
    print("Warning: Either 'Date' or 'Sales' column is missing in Dataset1. Skipping Time Series Analysis.\n")

# ----------------------- Customer and Product Analysis -----------------------

customer_stats = None
if 'CustomerID' in dataset2.columns:
    customer_stats = dataset2.groupby('CustomerID').agg({
        'Age': 'mean',
        'Gender': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
        'Location': lambda x: x.mode()[0] if not x.mode().empty else np.nan
    }).reset_index()
    print("Customer Statistics (Demographics):")
    print(customer_stats.head(), "\n")
else:
    print("Warning: 'CustomerID' column not found in Dataset2. Skipping Customer Demographics Analysis.\n")

purchase_behavior = None
if {'CustomerID', 'Sales', 'Transaction ID'}.issubset(dataset1.columns):
    purchase_behavior = dataset1.groupby('CustomerID').agg({
        'Sales': 'sum',
        'Transaction ID': 'count'
    }).rename(columns={'Transaction ID': 'PurchaseCount'}).reset_index()
    print("Purchase Behavior:")
    print(purchase_behavior.head(), "\n")
else:
    print("Warning: One or more required columns ('CustomerID', 'Sales', 'Transaction ID') not found in Dataset1. Skipping Purchase Behavior Analysis.\n")

top_products = None
if {'ProductID', 'Sales', 'Quantity'}.issubset(dataset1.columns):
    product_sales = dataset1.groupby('ProductID').agg({
        'Sales': 'sum',
        'Quantity': 'sum'
    }).reset_index().sort_values(by='Sales', ascending=False)
    print("Product Sales Performance:")
    print(product_sales.head(), "\n")

    top_products = product_sales.head(10)
    print("Top 10 Products by Sales:")
    print(top_products, "\n")
else:
    print("Warning: One or more required columns ('ProductID', 'Sales', 'Quantity') not found in Dataset1. Skipping Product Performance Analysis.\n")
# ----------------------- Visualization of Insights -----------------------

if top_products is not None:
    plt.figure(figsize=(12,8))
    sns.barplot(x='Sales', y='ProductID', data=top_products, palette='viridis')
    plt.title('Top 10 Products by Sales')
    plt.xlabel('Sales')
    plt.ylabel('Product ID')
    plt.tight_layout()
    plt.show()
else:
    print("Warning: Top products data not available. Skipping Top 10 Products by Sales visualization.\n")

if {'Category', 'Sales'}.issubset(dataset1.columns):
    plt.figure(figsize=(12,8))
    sns.barplot(x='Category', y='Sales', data=dataset1, palette='Set2')
    plt.title('Sales by Category')
    plt.xlabel('Category')
    plt.ylabel('Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Warning: 'Category' or 'Sales' column not found in Dataset1. Skipping Sales by Category visualization.\n")

if sales_over_time is not None:
    plt.figure(figsize=(12,6))
    sns.lineplot(x='Date', y='Sales', data=sales_over_time, marker='o')
    plt.title('Sales Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Warning: sales_over_time DataFrame is not defined. Skipping Sales Trend visualization.\n")

if len(numeric_columns) > 1:
    plt.figure(figsize=(14,12))
    correlation = dataset1[numeric_columns].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()
else:
    print("Warning: Not enough numeric columns to compute correlation heatmap.\n")

if 'Sales' in dataset1.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(y='Sales', data=dataset1)
    plt.title('Boxplot of Sales')
    plt.tight_layout()
    plt.show()
else:
    print("Warning: 'Sales' column not found in Dataset1. Skipping boxplot.\n")

if {'Category', 'Sales'}.issubset(dataset1.columns):
    category_sales = dataset1.groupby('Category')['Sales'].sum().reset_index()
    plt.figure(figsize=(8,8))
    plt.pie(category_sales['Sales'], labels=category_sales['Category'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set3'))
    plt.title('Market Share by Category')
    plt.axis('equal') 
    plt.tight_layout()
    plt.show()
else:
    print("Warning: 'Category' or 'Sales' column not found in Dataset1. Skipping Market Share Pie Chart.\n")

if customer_stats is not None:

    plt.figure(figsize=(10,6))
    sns.histplot(dataset2['Age'], bins=20, kde=True, color='skyblue')
    plt.title('Distribution of Customer Ages')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    if 'Gender' in dataset2.columns:
        plt.figure(figsize=(6,6))
        sns.countplot(x='Gender', data=dataset2, palette='pastel')
        plt.title('Gender Distribution')
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()
    else:
        print("Warning: 'Gender' column not found in Dataset2. Skipping Gender Distribution Visualization.\n")

    if 'Location' in dataset2.columns:
        plt.figure(figsize=(12,6))
        sns.countplot(x='Location', data=dataset2, palette='viridis')
        plt.title('Location Distribution')
        plt.xlabel('Location')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("Warning: 'Location' column not found in Dataset2. Skipping Location Distribution Visualization.\n")
else:
    print("Warning: Customer statistics not available. Skipping Customer Demographics Visualizations.\n")

if purchase_behavior is not None:

    plt.figure(figsize=(10,6))
    sns.scatterplot(x='PurchaseCount', y='Sales', data=purchase_behavior, alpha=0.6)
    plt.title('Purchase Count vs. Total Sales')
    plt.xlabel('Purchase Count')
    plt.ylabel('Total Sales')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,6))
    sns.histplot(purchase_behavior['Sales'], bins=30, kde=True, color='salmon')
    plt.title('Distribution of Total Sales per Customer')
    plt.xlabel('Total Sales')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
else:
    print("Warning: Purchase behavior data not available. Skipping Purchase Behavior Visualizations.\n")
