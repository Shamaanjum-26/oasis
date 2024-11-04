# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Step 2: Load the datasets
try:
    data1 = pd.read_csv('C:/Users/hp/Downloads/level 1 4th/user_reviews.csv/user_reviews.csv')
    data2 = pd.read_csv('C:/Users/hp/Downloads/level 1 4th/archive(1)/Twitter_Data.csv')
except Exception as e:
    print("Error loading datasets:", e)

# Ensure that both datasets have the 'text' and 'label' columns
if 'text' not in data1.columns or 'label' not in data1.columns:
    print("Dataset 1 is missing 'text' or 'label' columns.")
if 'text' not in data2.columns or 'label' not in data2.columns:
    print("Dataset 2 is missing 'text' or 'label' columns.")

# Combine the datasets
data = pd.concat([data1, data2], ignore_index=True)

# Step 3: Data Exploration
print("Combined Dataset Overview:")
print(data.head())

# Step 4: Text Preprocessing
# Tokenization and lemmatization
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize
    words = nltk.word_tokenize(text.lower())
    # Remove stop words and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

# Apply preprocessing
data['cleaned_text'] = data['text'].apply(preprocess_text)

# Step 5: Feature Engineering
# Bag-of-Words model
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(data['cleaned_text'])

# TF-IDF model
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(data['cleaned_text'])

# Step 6: Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, data['label'], test_size=0.2, random_state=42)

# Step 7: Machine Learning Algorithm - Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 8: Predictions
y_pred = model.predict(X_test)

# Step 9: Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 10: Data Visualization
# Visualization of Confusion Matrix
confusion_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Bar plot for sentiment distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='label', data=data, palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
