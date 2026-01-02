import pandas as pd
from joblib import load
from preprocess import clean_text

model = load("model/complaint_model.pkl")

def predict_single(text):
    text = clean_text(text)
    return model.predict([text])[0]

if __name__ == "__main__":
    print(predict_single("I want a refund!"))
