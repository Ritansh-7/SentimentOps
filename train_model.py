from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

# Create models directory
os.makedirs("models", exist_ok=True)

# Sample training data
texts = [
    "This is great!",
    "I love this!",
    "Amazing product!",
    "So bad",
    "I hate this",
    "Terrible experience"
]
labels = [1, 1, 1, 0, 0, 0]  # 1 = positive, 0 = negative

# Train vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Save models
joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("✅ Models saved!")