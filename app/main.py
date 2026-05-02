from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.services.llm_service import analyze_with_llm;
from app.services.ml_service import model,vectorizer,get_top_features;
from app.services.rag_service import get_similar_scams

from app.utils.text_cleaner import clean_text
import os 

from dotenv import load_dotenv


os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
load_dotenv()



# ── App Setup ──────────────────────────────────────────────────
app = FastAPI(title="Fake Job Detector API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class JobRequest(BaseModel):
    job_text: str




@app.get("/")
def root():
    return {"status": "Fake Job Detector API is running"}



@app.post("/analyze")
async def analyze_job(request: JobRequest):
    processed_text = clean_text(request.job_text)
    ml_prob = model.predict_proba(vectorizer.transform([processed_text]))[0][1]

    # ZONE 1: Obvious Safe (Dynamic Green Flags)
    if ml_prob < 0.2:
        green_flags = get_top_features(processed_text)
        return {
            "verdict": "LEGITIMATE",
            "ml_prob": round(ml_prob, 2),
            "explanation": "High-confidence legitimate structural signals detected.",
            "flags": green_flags, # These are now words the model likes
            "method": "ML_ONLY"
        }

    # ZONE 2: Obvious Fraud (Dynamic Red Flags)
    elif ml_prob > 0.6:
        red_flags = get_top_features(processed_text)
        return {
            "verdict": "FRAUDULENT",
            "ml_prob": round(ml_prob, 2),
            "explanation": "ML model detected high-certainty fraud markers.",
            "flags": red_flags, # These are now words the model hates
            "method": "ML_ONLY"
        }

    # ZONE 3: The Grey Area (Stays the same - RAG + LLM)
    else:
        context_data = get_similar_scams(request.job_text)
        llm_result = analyze_with_llm(request.job_text, ml_prob, context_data)
        llm_result["ml_prob"] = round(ml_prob, 2)
        llm_result["method"] = "RAG_LLM_HYBRID"
        return llm_result