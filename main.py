from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import re
import os
import json
from google import genai
from dotenv import load_dotenv

# RAG Imports
import chromadb
from sentence_transformers import SentenceTransformer

# ── Load Environment ───────────────────────────────────────────
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── Load Model Artifacts ───────────────────────────────────────
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
config = pickle.load(open('config.pkl', 'rb'))
threshold = config['threshold']

# ── RAG Setup (Pakistani Scam Database) ────────────────────────
# This connects to the local folder 'pak_scam_db'
chroma_client = chromadb.PersistentClient(path="./pak_scam_db")
collection = chroma_client.get_or_create_collection(name="local_fraud_patterns")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_similar_scams(user_job_text):
    """Retrieve top 2 similar local patterns from ChromaDB."""
    try:
        query_emb = embed_model.encode([user_job_text]).tolist()
        results = collection.query(query_embeddings=query_emb, n_results=2)
        return results['documents'][0] if results['documents'] else []
    except Exception as e:
        print(f"RAG Error: {e}")
        return []

# ── App Setup ──────────────────────────────────────────────────
app = FastAPI(title="Fake Job Detector API")
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

# ── LLM Analysis with RAG Context ──────────────────────────────
def analyze_with_llm(job_text: str, ml_probability: float, context_scams: list) -> dict:
    context_str = "\n".join(context_scams) if context_scams else "No specific similar patterns found."
    
    prompt = f"""
You are a fraud detection assistant helping Pakistani job seekers avoid scams.

Job posting submitted:
{job_text}

Known similar scams in Pakistan (for context):
{context_str}

ML model fraud probability: {ml_probability * 100:.1f}%

Your task:
1. Compare the job posting with the provided local scam patterns.
2. Specifically look for JazzCash/EasyPaisa mentions, "training fees", or WhatsApp-only contact.
3. Give a verdict: FRAUDULENT, LEGITIMATE, or UNCERTAIN.
4. Explain in 2-3 sentences in Hinglish (Roman Urdu/English mix) why it is suspicious.
5. List 2-3 specific red flags.

Respond in this exact JSON format:
{{
    "verdict": "FRAUDULENT/LEGITIMATE/UNCERTAIN",
    "confidence": "High/Medium/Low",
    "explanation": "your Hinglish explanation here",
    "flags": ["flag 1", "flag 2"]
}}
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    
    # Parse JSON cleanly
    raw_text = response.text.strip()
    clean_json = raw_text.replace("```json", "").replace("```", "").strip()
    return json.loads(clean_json)

# ── Routes ─────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Fake Job Detector API is running"}

@app.post("/analyze")
async def analyze_job(request: JobRequest):
    # ML Prediction
    processed_text = clean_text(request.job_text)
    # transform takes a list, predict_proba returns [ [prob_real, prob_fake] ]
    ml_prob = model.predict_proba(vectorizer.transform([processed_text]))[0][1]

    # ZONE 1: Obvious Safe (Fast Response)
    if ml_prob < 0.2:
        return {
            "verdict": "LEGITIMATE",
            "ml_prob": round(ml_prob, 2),
            "explanation": "This job looks safe based on structural analysis.",
            "flags": ["Professional formatting", "Standard terminology"],
            "method": "ML_ONLY"
        }

    # ZONE 2: Obvious Fraud (Fast Response)
    elif ml_prob > 0.6:
        # Check specific keywords for flags
        detected_flags = []
        if "whatsapp" in processed_text or "telegram" in processed_text:
            detected_flags.append("Unprofessional communication (WhatsApp/Telegram)")
        if "urgent" in processed_text or "immediately" in processed_text:
            detected_flags.append("High pressure/Urgency tactics")
        
        return {
            "verdict": "FRAUDULENT",
            "ml_prob": round(ml_prob, 2),
            "flags": detected_flags if detected_flags else ["Highly suspicious patterns"],
            "explanation": "ML model detected high-certainty fraud markers.",
            "method": "ML_ONLY"
        }

    # ZONE 3: The "Grey Area" (RAG + LLM Trigger)
    else:
        # Step 1: Pull local context
        context_data = get_similar_scams(request.job_text)
        
        # Step 2: Call LLM
        llm_result = analyze_with_llm(request.job_text, ml_prob, context_data)
        
        # Add metadata
        llm_result["ml_prob"] = round(ml_prob, 2)
        llm_result["method"] = "RAG_LLM_HYBRID"
        return llm_result