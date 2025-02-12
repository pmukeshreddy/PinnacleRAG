from fastapi import FastAPI , HTTPException
from pydantic import BaseModel



from data_loader import DataLoader
from embeddings import Embedder
from analysis import DataAnalyzer
from pinecone_manager import PineconeManager
from retriever import Retriever
from summarizer import DocumentSummarizer
import numpy as np

app = FastAPI(title="Legal Document Summarization API")

JUDGMENT_DIR = "/Users/mukeshreddypochamreddy/Documents/rag deployment law/dataset/IN-Abs/train-data/judgement"
SUMMARY_DIR = "/Users/mukeshreddypochamreddy/Documents/rag deployment law/dataset/IN-Abs/train-data/summary"
PINECONE_API_KEY = "pcsk_5FgdQP_B2AX3uc2wBgxvAnHNSg3NDjTdaeDssPzmbZH2e9AULEdPNSdG1FLCoSCRT2PgJs"
INDEX_NAME = "legal-docs-2"


pinecone_manager = None
retriever = None
summarizer = None


class QueryRequest(BaseModel):
    query : str

@app.on_event("startup") # decorator that cleans code fastAPI have set of cycles startup is one of them

def startup_event():
    global pineconemanager, retriever, summarizer

    data_loader = DataLoader(JUDGMENT_DIR, SUMMARY_DIR)
    embedder = Embedder()
    analyzer = DataAnalyzer()
    pineconemanager = PineconeManager(PINECONE_API_KEY, INDEX_NAME)
    summarizer = DocumentSummarizer()

    raw_data = data_loader.prepare_data()
    filtered_data = data_loader.filter_data(raw_data)

    judgment_embeddings , summary_embedding = embedder.compute_embeddings(filtered_data)
    vectors = []
    for idx, entry in enumerate(filtered_data):
        vector = (
            str(idx),
            np.concatenate((judgment_embeddings[idx], summary_embedding[idx])).tolist(),
            {"summary": entry["summary"], "judgment_text": entry["judgment"]}
        )
        vectors.append(vector)

    # Batch upsert vectors to the Pinecone index
    pineconemanager.batched_upsert(vectors)

    retriever = Retriever(pineconemanager.index)
    print("Server startup initialization complete.")


@app.post("/summarize") # creates a spefic adress where we can send and receive data

def summarize(query_request: QueryRequest):
    global retriever, summarizer
    if retriever is None or summarizer is None:
        raise HTTPException(status_code=500, detail="Server not fully initialized.")
    

    query = query_request.query
    results = retriever.retrieve(query)
    if not results:
        raise HTTPException(status_code=404, detail="No documents found for the query.")

    documents = [doc['judgment_text'] for doc in results]
    final_summary = summarizer.summarize_collection(documents)

    return {"query": query, "summary": final_summary}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


