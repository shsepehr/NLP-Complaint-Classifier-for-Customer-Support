import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump
from preprocess import clean_text

df = pd.read_csv("data/complaints.csv")

df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, pred))

dump(model, "model/complaint_model.pkl")
print("\n🎉 Model Trained & Saved in model/complaint_model.pkl")
