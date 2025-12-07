# backend/llm_engine.py (Updated)

from transformers import pipeline
# Remove direct model imports
# import torch 

from .utils import get_llm_pipeline # <-- NEW IMPORT

class LLMEngine:
    def __init__(self):
        # Replace direct loading with cached function call
        self.generator = get_llm_pipeline() 
        
        # --- NO MORE DEVICE CHECK/LOADING HERE ---

    def generate_answer(self, prompt):
        result = self.generator(
            prompt,
            max_length=400,
            temperature=0.3,
            do_sample=True,
            # --- CRITICAL FIX FOR REPETITION ---
            no_repeat_ngram_size=3, # Prevents repeating any 3-word sequence
            num_beams=3             # Helps produce a more coherent, less impulsive output
            # ----------------------------------
        )
        return result[0]["generated_text"]