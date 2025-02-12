# retriever.py
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, index, model_name='all-MiniLM-L6-v2'):
        self.index = index
        self.model = SentenceTransformer(model_name)

    def retrieve(self, query, top_k=10):
        query_embedding = self.model.encode(query).tolist()
        response = self.index.query(
            vector=query_embedding + query_embedding,  # Match combined vector dim
            top_k=top_k,
            include_metadata=True
        )
        return [match.metadata for match in response.matches]