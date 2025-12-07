import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from backend.llm_engine import LLMEngine
import torch # <-- MAKE SURE TORCH IS IMPORTED
from backend.llm_engine import LLMEngine
from .utils import get_sentence_embedder

class RAGEngine:
    def __init__(self): # <-- Ensure the spelling is __init__
        
        # Replace direct loading with cached function call
        self.embedder = get_sentence_embedder() 
        
        # --- NO MORE DEVICE CHECK/LOADING HERE ---
        
        self.index = faiss.read_index("vector_store/firstaid_index.bin")
        self.corpus = np.load("vector_store/corpus.npy", allow_pickle=True)
        self.llm = LLMEngine()
        self.similarity_threshold = 1.2

    def search_context(self, user_query):
        emb = self.embedder.encode([user_query])
        distances, indices = self.index.search(emb, k=1)
        return distances[0][0], self.corpus[indices[0][0]]

    # For known emergencies (RAG)
    def build_rag_prompt(self, context, user_query):
        return f"""
Use ONLY the steps from the context.

Context:
{context}

User emergency:
{user_query}

TASK:
Output ONLY the numbered steps.
Each step on a new line.

End with:
Call emergency services immediately.
"""

    # For unknown emergencies (fallback)
    def build_safety_warning_prompt(self, user_query):
        return f"""


Strict Instruction: You are an AI First Aid System. The user has requested guidance for an unknown emergency: '{user_query}'.
You are strictly forbidden from providing any first aid steps or descriptive text.
Your SOLE and ONLY task is to output the critical safety instruction below.

Output ONLY the following sentence, and nothing else:
Call emergency services immediately (911/999/000) and seek professional medical help. Do not wait.
"""

    def get_response(self, user_query):
        distance, context = self.search_context(user_query)

        if distance < self.similarity_threshold:
            prompt = self.build_rag_prompt(context, user_query)
        else:
            prompt = self.build_fallback_prompt(user_query)

        return self.llm.generate_answer(prompt)
