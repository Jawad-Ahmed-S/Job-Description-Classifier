import chromadb
from sentence_transformers import SentenceTransformer

# 1. Initialize DB and Model
chroma_client = chromadb.PersistentClient(path="./pak_scam_db")
collection = chroma_client.get_or_create_collection(name="local_fraud_patterns")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def seed_pakistani_data(scam_list):
    """Call this once to populate your DB."""
    ids = [f"pak_scam_{i}" for i in range(len(scam_list))]
    embeddings = embed_model.encode(scam_list).tolist()
    collection.add(ids=ids, embeddings=embeddings, documents=scam_list)

def get_similar_scams(user_job_text):
    """Retrieve top 2 similar local patterns for LLM context."""
    query_emb = embed_model.encode([user_job_text]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=2)
    return results['documents'][0]