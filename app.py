import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

from data_loader import DataLoader
from embeddings import Embedder
from analysis import DataAnalyzer
from pinecone_manager import PineconeManager
from retriever import Retriever
from summarizer import DocumentSummarizer

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="Legal Document Summarization API")

JUDGMENT_DIR = "./dataset/IN-Abs/train-data/judgement"
SUMMARY_DIR = "./dataset/IN-Abs/train-data/summary"
PINECONE_API_KEY = "pcsk_5FgdQP_B2AX3uc2wBgxvAnHNSg3NDjTdaeDssPzmbZH2e9AULEdPNSdG1FLCoSCRT2PgJs"
INDEX_NAME = "legal-docs-2"

pineconemanager = None
retriever = None
summarizer = None


class QueryRequest(BaseModel):
    query: str


@app.on_event("startup")
def startup_event():
    global pineconemanager, retriever, summarizer

    try:
        logging.info("Starting server initialization...")

        logging.info("Initializing DataLoader...")
        data_loader = DataLoader(JUDGMENT_DIR, SUMMARY_DIR)

        logging.info("Initializing Embedder...")
        embedder = Embedder()

        logging.info("Initializing DataAnalyzer...")
        analyzer = DataAnalyzer()

        logging.info("Initializing PineconeManager...")
        pineconemanager = PineconeManager(PINECONE_API_KEY, INDEX_NAME)
        logging.info(f"Pinecone API Key: {PINECONE_API_KEY[:10]}********")
        logging.info(f"Pinecone Index Name: {INDEX_NAME}")

        logging.info("Initializing DocumentSummarizer...")
        summarizer = DocumentSummarizer()

        logging.info("Preparing and filtering data...")
        raw_data = data_loader.prepare_data()
        filtered_data = data_loader.filter_data(raw_data)
        logging.info(f"Total filtered records: {len(filtered_data)}")

        logging.info("Computing embeddings...")
        judgment_embeddings, summary_embedding = embedder.compute_embeddings(filtered_data)

        vectors = []
        for idx, entry in enumerate(filtered_data):
            vector = (
                str(idx),
                np.concatenate((judgment_embeddings[idx], summary_embedding[idx])).tolist(),
                {"summary": entry["summary"], "judgment_text": entry["judgment"]}
            )
            vectors.append(vector)

        logging.info("Upserting vectors to Pinecone index...")
        pineconemanager.batched_upsert(vectors)

        logging.info("Initializing Retriever...")
        retriever = Retriever(pineconemanager.index)

        logging.info("Server startup initialization complete.")

    except Exception as e:
        logging.error(f"Error during startup: {e}", exc_info=True)


@app.post("/summarize")
def summarize(query_request: QueryRequest):
    global retriever, summarizer

    if retriever is None or summarizer is None:
        logging.error("Server not fully initialized.")
        raise HTTPException(status_code=500, detail="Server not fully initialized.")

    query = query_request.query
    logging.info(f"Received query: {query}")

    results = retriever.retrieve(query)
    if not results:
        logging.warning("No documents found for the query.")
        raise HTTPException(status_code=404, detail="No documents found for the query.")

    logging.info(f"Found {len(results)} documents for query.")
    documents = [doc["judgment_text"] for doc in results]

    logging.info("Generating summary...")
    final_summary = summarizer.summarize_collection(documents)

    logging.info("Summary generation complete.")
    return {"query": query, "summary": final_summary}


if __name__ == "__main__":
    import uvicorn
    logging.info("Starting Uvicorn server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
