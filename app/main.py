from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from typing import List
from pydantic import BaseModel
import joblib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

sys.path.append(PROJECT_ROOT) 

MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
MODEL_CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'tfidf_classifier.pkl')
THRESHOLDS_PATH = os.path.join(MODEL_DIR, 'thresholds.pkl')

CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

model = None
vectorizer = None
thresholds = None

templates = Jinja2Templates(directory="app/templates")
app = FastAPI(
    title="Toxic Comment Detector API",
    description="API для классификации токсичных комментариев с использованием TF-IDF + Logistic Regression",
    version="1.0"
)





class Text(BaseModel):
    content: str

@app.on_event("startup")
def load_model():
    global model, vectorizer, thresholds
    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
        model = joblib.load(MODEL_CLASSIFIER_PATH)
        thresholds = joblib.load(THRESHOLDS_PATH)
        print("Model, vectorizer, and thresholds loaded successfully.")
    except FileNotFoundError:
        print("Model files not found.", VECTORIZER_PATH, MODEL_CLASSIFIER_PATH)
        pass




@app.get("/", tags=["Home"], summary="Home Page")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.post("/api/classify", tags=["Text Classification"], summary="Classify Text")
async def classify_text(text: Text) -> dict:
    global model, vectorizer, thresholds
    
    if not model or not vectorizer or not thresholds:
        raise HTTPException(status_code=500, detail="Model not loaded. Please try again later.")
    
    if not text.content.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    
    try:
        text_vectorized = vectorizer.transform([text.content])
        prediction = model.predict_proba(text_vectorized)
        pred = prediction.tolist()[0]
        response = {}
        for i, cls in enumerate(CLASSES):
            score = pred[i]
            threshold_for_cls = thresholds[cls]

            response[cls] = {
                "score": round(score, 4),
                "threshold": round(threshold_for_cls, 4),
                "is_toxic": score >= threshold_for_cls
            }

        return response
    except Exception:
        raise HTTPException(status_code=400, detail="Text cannot be classified")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)