# main.py
from data_loader import DataLoader
from embeddings import Embedder
from analysis import DataAnalyzer
from pinecone_manager import PineconeManager
from retriever import Retriever
from summarizer import DocumentSummarizer
import numpy as np

def main():
    # Configuration
    JUDGMENT_DIR = "/Users/mukeshreddypochamreddy/Documents/rag deployment law/dataset/IN-Abs/train-data/judgement"
    SUMMARY_DIR = "/Users/mukeshreddypochamreddy/Documents/rag deployment law/dataset/IN-Abs/train-data/summary"
    PINECONE_API_KEY = "pcsk_5FgdQP_B2AX3uc2wBgxvAnHNSg3NDjTdaeDssPzmbZH2e9AULEdPNSdG1FLCoSCRT2PgJs"
    INDEX_NAME = "legal-docs-2"
    
    # Initialize components
    data_loader = DataLoader(JUDGMENT_DIR, SUMMARY_DIR)
    embedder = Embedder()
    analyzer = DataAnalyzer()
    pinecone_manager = PineconeManager(PINECONE_API_KEY, INDEX_NAME)
    summarizer = DocumentSummarizer()

    # Data preparation pipeline
    raw_data = data_loader.prepare_data()
    filtered_data = data_loader.filter_data(raw_data)
    
    # Data analysis
    stats = analyzer.compute_word_stats(filtered_data)
    print(f"Average judgment length: {stats['avg_judgment']:.2f} words")
    print(f"Average summary length: {stats['avg_summary']:.2f} words")
    analyzer.plot_word_counts(stats)

    # Embedding generation
    judgment_embeddings, summary_embeddings = embedder.compute_embeddings(filtered_data)

    # Pinecone index setup
    pinecone_manager.create_index(dimension=1536)  # 768*2 for concatenated embeddings
    
    # Prepare and upload vectors
    vectors = [(str(idx), 
                np.concatenate((judgment_embeddings[idx], summary_embeddings[idx])).tolist(),
                {"summary": entry["summary"], "judgment_text": entry["judgment"]})
               for idx, entry in enumerate(filtered_data)]
    
    pinecone_manager.batched_upsert(vectors)

    # Query and summarization pipeline
    query = "Patna"
    retriever = Retriever(pinecone_manager.index)
    results = retriever.retrieve(query)
    documents = [doc['judgment_text'] for doc in results]
    
    final_summary = summarizer.summarize_collection(documents)
    print("\nFinal Summary:", final_summary)

if __name__ == "__main__":
    main()