import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="./pak_scam_db")
collection = client.get_or_create_collection(name="local_fraud_patterns")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Add your 100-200 records here
scams = [
    "Urgent hiring for Dubai. Pay 5000 for medical via EasyPaisa to Mr. Ahmed.",
    "Like YouTube videos and earn 5000 daily. WhatsApp us for details.",
    "Data entry work from home. Registration fee 2000 mandatory.",
    # ... add more
]

ids = [f"id{i}" for i in range(len(scams))]
embeddings = model.encode(scams).tolist()
collection.add(ids=ids, embeddings=embeddings, documents=scams)
print("Database Seeded!")