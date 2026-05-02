from groq import Groq
import os
import json

from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def analyze_with_llm(job_text: str, ml_probability: float, context_scams: list) -> dict:
    context_str = "\n".join(context_scams) if context_scams else "No local patterns found."
    
    prompt = f"""
    Analyze this Pakistani job posting: {job_text}
    
    Local Scam Context: {context_str}
    ML Risk Score: {ml_probability * 100:.1f}%
    
    Task: Identify fraud signals (EasyPaisa/JazzCash, 'security fees', WhatsApp-only).
    Output valid JSON only. Explain in Hinglish.
    Format:
    {{
        "verdict": "FRAUDULENT/LEGITIMATE/UNCERTAIN",
        "confidence": "High/Medium/Low",
        "explanation": "Hinglish explanation here",
        "flags": ["flag 1", "flag 2"]
    }}
    """

    # Groq API Call
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile", # Great for reasoning & JSON
        messages=[
            {"role": "system", "content": "You are a fraud detection expert. Output JSON only."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"} # Forces JSON output
    )
    
    return json.loads(completion.choices[0].message.content)
