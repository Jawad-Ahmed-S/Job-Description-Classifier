from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import re
import numpy as np

# ── Load Model Artifacts ───────────────────────────────────────
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
config = pickle.load(open('config.pkl', 'rb'))
threshold = config['threshold']

# ── FastAPI App ────────────────────────────────────────────────
app = FastAPI(title="Fake Job Detector API")

# ── CORS — allows React frontend to talk to this API ──────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request Schema ─────────────────────────────────────────────
class JobRequest(BaseModel):
    job_text: str

# ── Text Cleaning ──────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ── Risk Level ─────────────────────────────────────────────────
def get_risk_level(prob):
    if prob >= 0.75:   return "High"
    elif prob >= 0.55: return "Medium"
    elif prob >= 0.15: return "Low"
    else:              return "Safe"

# ── Health Check ───────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Fake Job Detector API is running"}

# ── Main Prediction Endpoint ───────────────────────────────────
@app.post("/analyze")
def analyze_job(request: JobRequest):
    cleaned = clean_text(request.job_text)
    vectorized = vectorizer.transform([cleaned])
    prob = model.predict_proba(vectorized)[0][1]
    risk = get_risk_level(prob)
    verdict = "FAKE" if prob >= threshold else "REAL"

    return {
        "verdict": verdict,
        "fraud_probability": round(prob * 100, 2),
        "risk_level": risk,
        "escalate_to_llm": bool(prob >= 0.15)  # cost gate flag
    }