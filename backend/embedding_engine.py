from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

DATA_PATH = "data/first_aid_data.txt"
INDEX_PATH = "vector_store/firstaid_index.bin"
CORPUS_PATH = "vector_store/corpus.npy"

os.makedirs("vector_store", exist_ok=True)

class EmbeddingEngine:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def load_data(self):
        with open(DATA_PATH, "r") as f:
            return f.read().split("\n\n")

    def create_index(self):
        docs = self.load_data()
        embeddings = self.model.encode(docs)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings))

        faiss.write_index(index, INDEX_PATH)
        np.save(CORPUS_PATH, docs)
        print("✅ Vector index created successfully.")

if __name__ == "__main__":
    engine = EmbeddingEngine()
    engine.create_index()
