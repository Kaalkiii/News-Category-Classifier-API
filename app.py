from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import string
# Load model and vectorizer
model = joblib.load("model/news_classifier.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

app = FastAPI()

# Request body structure
class NewsRequest(BaseModel):
    text: str

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the News Classifier API!"}

# Clean and preprocess the input text
def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

# Prediction endpoint
@app.post("/predict")
def predict_category(request: NewsRequest):
    cleaned_text = preprocess(request.text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    return {"category": prediction}