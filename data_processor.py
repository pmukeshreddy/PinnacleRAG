# data_processor.py
class DataProcessor:
    @staticmethod
    def create_chunks(filter_data):
        return [{
            "summary": entry["summay"],  
            "judgment_text": entry["judgment"]
        } for entry in filter_data]

    @staticmethod
    def prepare_vectors(filter_data, judgment_embeddings, summary_embeddings):
        vectors = []
        for idx, entry in enumerate(filter_data):
            combined_vector = (
                judgment_embeddings[idx].tolist() + 
                summary_embeddings[idx].tolist()
            )
            vectors.append((
                str(idx),
                combined_vector,
                {
                    "summary": entry["summay"],
                    "judgment_text": entry["judgment"]
                }
            ))
        return vectors
