import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud

df = pd.read_csv('C:/Users/hp/Downloads/archive/creditcard.csv')

print("Columns in the DataFrame:", df.columns)

plt.figure(figsize=(10, 5))
plt.hist(df['Amount'], bins=50, color='blue', alpha=0.7)
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')
plt.title('Distribution of Transaction Amounts')
plt.show()

class_distribution = df['Class'].value_counts()
plt.figure(figsize=(6, 4))
class_distribution.plot(kind='bar', color=['green', 'red'])
plt.xlabel('Transaction Class')
plt.ylabel('Number of Transactions')
plt.title('Distribution of Transaction Classes')
plt.xticks(ticks=[0, 1], labels=['Non-Fraudulent', 'Fraudulent'], rotation=0)
plt.show()

y_true = [1, 0, 1] 
y_pred = [1, 0, 0]  
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy}')
