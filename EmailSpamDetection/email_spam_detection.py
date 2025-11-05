# 📧 Email Spam Detection using Naive Bayes
# Importing required libraries
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load dataset
# Dataset: SMS Spam Collection Dataset from UCI ML Repository
data = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'message']

# Step 2: Data preprocessing
nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()                                   # Convert to lowercase
    text = ''.join([c for c in text if c not in string.punctuation])  # Remove punctuation
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]   # Remove stopwords and stem
    return " ".join(words)

data['clean_message'] = data['message'].apply(clean_text)

# Step 3: Encode labels
data['label_num'] = data.label.map({'ham': 0, 'spam': 1})

# Step 4: Data visualization
plt.figure(figsize=(6,4))
data['label'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Spam vs Ham Message Distribution")
plt.xlabel("Message Type")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# Step 5: Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    data['clean_message'], data['label_num'], test_size=0.2, random_state=42
)

# Step 6: Convert text to numerical form
vectorizer = TfidfVectorizer(max_features=2500)
X_train_tf = vectorizer.fit_transform(X_train)
X_test_tf = vectorizer.transform(X_test)

# Step 7: Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tf, y_train)

# Step 8: Model prediction and evaluation
y_pred = model.predict(X_test_tf)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("=== 📈 Model Evaluation ===")
print(f"Accuracy: {acc*100:.2f}%")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 9: Custom sample predictions
sample_emails = [
    "Congratulations! You've won a $500 gift card. Claim now!",
    "Are we still meeting tomorrow at 10 AM?",
    "Your account has been compromised. Click here to reset your password!",
]

sample_features = vectorizer.transform(sample_emails)
sample_preds = model.predict(sample_features)

for i, mail in enumerate(sample_emails):
    label = "SPAM" if sample_preds[i] == 1 else "HAM"
    print(f"\nEmail: {mail}\n→ Classified as: {label}")

# Step 10: Accuracy visualization
plt.figure(figsize=(5,3))
plt.bar(['Accuracy'], [acc*100], color='seagreen')
plt.title("Model Accuracy (%)")
plt.ylim(90, 100)
plt.ylabel("Accuracy %")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
