import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Load datasets
train_df = pd.read_csv("dataset/training.csv")
val_df = pd.read_csv("dataset/validation.csv")
test_df = pd.read_csv("dataset/test.csv")

# Check structure
assert 'text' in train_df.columns and 'emotion' in train_df.columns

# Combine training + validation (optional but improves accuracy slightly)
full_train_df = pd.concat([train_df, val_df], ignore_index=True)

# Create ML pipeline
model = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Train
model.fit(full_train_df["text"], full_train_df["emotion"])

# Evaluate
y_test_pred = model.predict(test_df["text"])
print("Accuracy:", accuracy_score(test_df["emotion"], y_test_pred))
print("\nClassification Report:\n", classification_report(test_df["emotion"], y_test_pred))

# --- Predict your own input ---
def predict_emotion(text):
    pred = model.predict([text])[0]
    print(f"'{text}' ➜ Predicted Emotion: {pred}")

# Example
predict_emotion("I feel great and happy today")
predict_emotion("I want to cry all day")
