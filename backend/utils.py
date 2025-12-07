# backend/utils.py
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch

@st.cache_resource
def get_sentence_embedder():
    """Initializes and caches the SentenceTransformer model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing SentenceTransformer ONCE on device: {device}")
    # Pass the device here for the one-time load
    return SentenceTransformer("all-MiniLM-L6-v2", device=device)

@st.cache_resource
def get_llm_pipeline():
    """Initializes and caches the Flan-T5 pipeline."""
    # -1 for CPU, 0 for first GPU
    device_id = 0 if torch.cuda.is_available() else -1 
    print(f"Initializing Flan-T5 Pipeline ONCE on device ID: {device_id}")
    
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=device_id
    )