from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from ollama import chat
import faiss
import numpy as np

# Connect MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["careersproject"]
collection = db["careers"]

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Read careers from MongoDB
careers = list(collection.find())

if len(careers) == 0:
    print("No career data found in MongoDB")
    exit()

documents = []

for career in careers:
    text = f"""
    Career: {career.get('career','')}
    Description: {career.get('description','')}
    Skills: {', '.join(career.get('skills',[]))}
    """
    documents.append(text)

print("Generating embeddings...")

embeddings = model.encode(documents)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings).astype("float32"))

print("FAISS Index Created")

# User Query
query = input("Ask Career Question: ")

query_embedding = model.encode([query])

distances, indices = index.search(
    np.array(query_embedding).astype("float32"),
    3
)

context = ""

for idx in indices[0]:
    context += documents[idx] + "\n"

prompt = f"""
You are a Career Guidance Assistant.

Career Database:
{context}

User Question:
{query}

Provide:
1. Best Career Match
2. Required Skills
3. Skill Gap Analysis
4. Learning Roadmap
"""

response = chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\n===== Career Recommendation =====\n")
print(response["message"]["content"])