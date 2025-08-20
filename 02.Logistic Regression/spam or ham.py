import numpy as np
import re
import nltk
from collections import Counter
from sklearn.model_selection import train_test_split

# Download stopwords + tokenizer (first time only)
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab") 

from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

# 1. Load dataset (your path)
file_path = "/Users/anand/Desktop/Ml_deep_notes_for_myself/Logistic Regression/smsspamcollection/SMSSpamCollection"

with open(file_path, "r", encoding="utf-8") as f:
    data = f.readlines()

labels, texts = [], []
for line in data:
    label, text = line.split("\t", 1)
    labels.append(1 if label == "spam" else 0)  # spam=1, ham=0
    texts.append(text.strip())

y = np.array(labels)

# 2. Preprocess text
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = text.split()  # just split on whitespace
    tokens = [t for t in tokens if t not in stop_words]
    return tokens


tokenized_texts = [preprocess(t) for t in texts]

# Build Vocabulary (top 3000 words)
all_words = Counter([word for tokens in tokenized_texts for word in tokens])
vocab = {word: i for i, (word, _) in enumerate(all_words.most_common(3000))}

# Convert texts to Bag-of-Words vectors
def text_to_vector(tokens):
    vec = np.zeros(len(vocab))
    for t in tokens:
        if t in vocab:
            vec[vocab[t]] += 1
    return vec

X = np.array([text_to_vector(t) for t in tokenized_texts])

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Logistic Regression (from scratch)
class LogisticRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            
            # Gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update weights
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return np.where(y_pred >= 0.5, 1, 0)

# Train model
model = LogisticRegressionScratch(lr=0.1, epochs=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)
