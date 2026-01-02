from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
from preprocess import clean_text

app = FastAPI(title="NLP Complaint Classifier API")

# Load model
model = load("model/complaint_model.pkl")

# Define input schema
class ComplaintRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    category: str

@app.post("/predict", response_model=PredictionResponse)
def classify_complaint(item: ComplaintRequest):
    cleaned_text = clean_text(item.text)
    prediction = model.predict([cleaned_text])[0]
    return {"category": prediction}
