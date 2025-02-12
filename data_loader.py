# data_loader.py
import os

class DataLoader:
    def __init__(self, judgment_dir, summary_dir):
        self.judgment_dir = judgment_dir
        self.summary_dir = summary_dir

    def load_text_files(self, directory):
        data = {}
        for file in os.listdir(directory):
            if file.endswith(".txt"):
                with open(os.path.join(directory, file), "r", encoding="utf-8") as f:
                    data[file] = f.read()
        return data

    def prepare_data(self, prefix="[IN-Abs]"):
        judgments = self.load_text_files(self.judgment_dir)
        summaries = self.load_text_files(self.summary_dir)
        
        data = []
        for key in judgments:
            if key in summaries:
                data.append({
                    "judgment": prefix + judgments[key],
                    "summary": summaries[key]
                })
        return data

    def filter_data(self, data, max_judgment_len=5000, max_summary_len=1000):
        return [entry for entry in data if 
                len(entry["judgment"].split()) < max_judgment_len and 
                len(entry["summary"].split()) < max_summary_len]