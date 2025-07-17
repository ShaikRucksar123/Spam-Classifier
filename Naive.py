import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
 # Generate synthetic dataset
np.random.seed(42)
spam_samples = [
    "Win a free iPhone now! Click the link to claim.",
    "Congratulations! You've won a lottery. Call now!",
    "Limited offer! Get 50% off on all items. Hurry!",
    "Your account is suspended. Verify immediately.",
    "Earn money from home! Work just 2 hours a day."
 ]
ham_samples = [
    "Hey, how are you doing today?",
    "Let's meet for lunch tomorrow at noon.",
    "Can you send me the project report?",
    "Don't forget about the meeting at 3 PM.",
    "Happy birthday! Have a great day ahead."]
messages = spam_samples * 20 + ham_samples * 20  # Duplicate samples for balance
labels = ['spam'] * 100 + ['ham'] * 100  # Corresponding labels
 # Create DataFrame
df = pd.DataFrame({'message': messages, 'label': labels})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    return text
df['message'] = df['message'].apply(preprocess_text)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
 # Feature extraction
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['message'])
y = df['label']
 # Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 # Train Naïve Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)
 # Predictions
y_pred = model.predict(X_test)
 # Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))
# Function for prediction
def predict_spam(text):
    text = preprocess_text(text)
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return "Spam" if prediction[0] == 1 else "Ham"
 # User input
user_input = input("Enter a message: ")
print("Prediction:", predict_spam(user_input))
