
# 📧 Spam Message Classifier using Naive Bayes

This is a simple Python-based machine learning project that classifies text messages as **Spam** or **Ham (Not Spam)** using the **Naive Bayes algorithm**. It uses a small synthetic dataset, text preprocessing, and TF-IDF feature extraction for model training and prediction.

---

## 🚀 Features

- Detects whether a message is spam or not
- Uses a **Multinomial Naive Bayes** classifier
- **TF-IDF Vectorization** for converting text into numeric features
- Simple preprocessing of text using regex
- Real-time message input and prediction

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn (for ML model and metrics)
- Regular Expressions (`re` and `string` for text cleaning)

---

## 📦 How It Works

1. **Dataset Creation**: 
   - Synthetic dataset of spam and ham messages is generated and balanced.
2. **Text Preprocessing**: 
   - Converts to lowercase, removes punctuation and numbers.
3. **Vectorization**:
   - TF-IDF is used to convert text into numerical form.
4. **Model Training**:
   - A Naive Bayes classifier is trained using scikit-learn.
5. **Prediction**:
   - Users can enter a custom message to check if it's spam or not.

---

## 📈 Model Evaluation

- Accuracy is printed on the console.
- Detailed classification report includes precision, recall, and F1-score.

---

## 🧪 How to Run

1. Make sure you have Python installed.
2. Install required libraries:
   ```bash
   pip install pandas numpy scikit-learn
   ```
3. Run the script:
   ```bash
   python spam_classifier.py
   ```
4. Enter a message when prompted to check if it's spam.

---

## 📝 Example

```text
Enter a message: Congratulations! You've won a prize.
Prediction: Spam
```

---

## 📚 License

This project is open for educational and personal use.
