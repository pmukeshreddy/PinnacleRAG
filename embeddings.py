# embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def compute_embeddings(self, data):
        judgment_embeddings = np.array([self.model.encode(entry["judgment"]) for entry in data])
        summary_embeddings = np.array([self.model.encode(entry["summary"]) for entry in data])
        return judgment_embeddings, summary_embeddings
    
