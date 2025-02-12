# summarizer.py
from transformers import pipeline

class DocumentSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.summarizer = pipeline("summarization", model=model_name)
        self.max_input_length = 1024

    def summarize_single(self, text):
        return self.summarizer(
            text[:self.max_input_length],
            max_length=100,
            min_length=30,
            do_sample=False
        )[0]['summary_text']

    def summarize_collection(self, documents):
        individual = [self.summarize_single(doc) for doc in documents]
        combined = " ".join(individual)
        return self.summarize_single(combined)