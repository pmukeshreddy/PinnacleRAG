# pinecone_manager.py
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

class PineconeManager:
    def __init__(self, api_key, index_name):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.index = self.pc.Index(index_name)

    def create_index(self, dimension=1536, metric="cosine", cloud="aws", region="us-east-1"):
        try:
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region)
            )
        except Exception as e:
            print(f"Index already exists or error occurred: {e}")
            
    def batched_upsert(self, vectors, batch_size=100):
        for i in tqdm(range(0, len(vectors), batch_size)):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch)