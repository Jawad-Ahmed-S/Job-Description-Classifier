import chromadb

from sentence_transformers import SentenceTransformer


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
