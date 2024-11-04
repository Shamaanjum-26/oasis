import pandas as pd
import numpy as np
from scipy import stats
import glob
import json
import chardet

def detect_encoding(file_path):
    """Detects the encoding of a file."""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def clean_csv(file_path):
    try:
        # Try detecting encoding
        encoding = detect_encoding(file_path)
        print(f"Detected encoding for {file_path}: {encoding}")
        
        # Try reading the CSV with the detected encoding
        try:
            df = pd.read_csv(file_path, encoding=encoding)
        except Exception as e:
            print(f"Error with encoding {encoding}, trying 'utf-8' with error handling: {e}")
            df = pd.read_csv(file_path, encoding='utf-8', errors='replace')  # Replace problematic characters
        
        # Step 1: Check dataset info (Data Integrity)
        print(f"Dataset Information for {file_path}:")
        print(df.info())

        # Step 2: Check for duplicates based on a relevant column (Duplicate Removal)
        duplicate_records = df[df.duplicated(keep=False)]
        print(f"\nNumber of duplicate records in {file_path}: {duplicate_records.shape[0]}")
        df.drop_duplicates(inplace=True)

        # Step 3: Summary of missing data (Missing Data Handling)
        missing_data_summary = df.isnull().sum()
        print("\nMissing Data Summary:")
        print(missing_data_summary)

        # Step 4: Handling missing data
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            df[col].fillna('No Data Provided', inplace=True)

        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df.dropna(subset=numerical_cols, inplace=True)

        # Step 5: Detect and handle outliers (Outlier Detection)
        def remove_outliers(df, column):
            z_scores = np.abs(stats.zscore(df[column]))
            return df[z_scores < 3]

        for col in numerical_cols:
            df = remove_outliers(df, col)

        # Step 6: Standardization of specific columns (Standardization)
        if 'price' in df.columns:
            df['price'] = df['price'].abs()

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')  # Standardize date format

        # Step 7: Check final dataset shape after cleaning
        print("\nDataset shape after cleaning:", df.shape)

        # Export cleaned dataset
        cleaned_file_path = file_path.replace('.csv', '_cleaned.csv')
        df.to_csv(cleaned_file_path, index=False)
        print(f"Cleaned data exported to {cleaned_file_path}\n")

        return df

    except Exception as e:
        print(f"An error occurred while cleaning {file_path}: {e}")


def clean_json(file_path):
    # Load the JSON dataset
    with open(file_path, 'r') as f:
        data = json.load(f)

    df = pd.json_normalize(data)

    # Similar cleaning steps as in the clean_csv function
    print(f"Dataset Information for {file_path}:")
    print(df.info())

    duplicate_records = df[df.duplicated(keep=False)]
    print(f"\nNumber of duplicate records in {file_path}: {duplicate_records.shape[0]}")
    df.drop_duplicates(inplace=True)

    missing_data_summary = df.isnull().sum()
    print("\nMissing Data Summary:")
    print(missing_data_summary)

    string_cols = df.select_dtypes(include=['object']).columns
    for col in string_cols:
        df[col].fillna('No Data Provided', inplace=True)

    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df.dropna(subset=numerical_cols, inplace=True)

    for col in numerical_cols:
        df = remove_outliers(df, col) #type:ignore

    if 'price' in df.columns:
        df['price'] = df['price'].abs()

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    print("\nDataset shape after cleaning:", df.shape)

    cleaned_file_path = file_path.replace('.json', '_cleaned.json')
    df.to_json(cleaned_file_path, orient='records', lines=True)
    print(f"Cleaned data exported to {cleaned_file_path}\n")

    return df

csv_files = [
    'C:/Users/hp/Downloads/trending youtube/archive/USvideos.csv',
    'C:/Users/hp/Downloads/trending youtube/archive/RUvideos.csv',
    'C:/Users/hp/Downloads/trending youtube/archive/MXvideos.csv',
    'C:/Users/hp/Downloads/trending youtube/archive/INvideos.csv',
    'C:/Users/hp/Downloads/trending youtube/archive/JPvideos.csv',
    'C:/Users/hp/Downloads/trending youtube/archive/KRvideos.csv',
    'C:/Users/hp/Downloads/trending youtube/archive/GBvideos.csv',
    'C:/Users/hp/Downloads/trending youtube/archive/DEvideos.csv',
    'C:/Users/hp/Downloads/trending youtube/archive/FRvideos.csv',
    'C:/Users/hp/Downloads/trending youtube/archive/CAvideos.csv'
]

for file in csv_files:
    clean_csv(file)


# Process JSON files
json_files = [
    'C:/Users/hp/Downloads/trending youtube/archive/US_category_id.json',
    'C:/Users/hp/Downloads/trending youtube/archive/RU_category_id.json',
    'C:/Users/hp/Downloads/trending youtube/archive/JP_category_id.json',
    'C:/Users/hp/Downloads/trending youtube/archive/KR_category_id.json',
    'C:/Users/hp/Downloads/trending youtube/archive/MX_category_id.json',
    'C:/Users/hp/Downloads/trending youtube/archive/IN_category_id.json',
    'C:/Users/hp/Downloads/trending youtube/archive/FR_category_id.json',
    'C:/Users/hp/Downloads/trending youtube/archive/GB_category_id.json',
    'C:/Users/hp/Downloads/trending youtube/archive/CA_category_id.json',
    'C:/Users/hp/Downloads/trending youtube/archive/DE_category_id.json'
]

for file in json_files:
    clean_json(file)
