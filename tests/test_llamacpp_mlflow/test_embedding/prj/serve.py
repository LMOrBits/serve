# serve.py
from typing import List
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import uvicorn

# Load model once on startup
model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI()

@app.post("/embed")
async def embed(sentences: List[str]):
    """
    Receive a list of sentences and return their embeddings.
    """
    embeddings = model.encode(sentences).tolist()
    return {"embeddings": embeddings}

if __name__ == "__main__":
    # Run with auto-reload for development
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)