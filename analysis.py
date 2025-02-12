# analysis.py
import matplotlib.pyplot as plt

class DataAnalyzer:
    @staticmethod
    def compute_word_stats(data):
        judgment_counts = [len(entry["judgment"].split()) for entry in data]
        summary_counts = [len(entry["summary"].split()) for entry in data]
        
        return {
            "avg_judgment": sum(judgment_counts) / len(judgment_counts),
            "avg_summary": sum(summary_counts) / len(summary_counts),
            "judgment_counts": judgment_counts,
            "summary_counts": summary_counts
        }

    @staticmethod
    def plot_word_counts(stats):
        plt.figure(figsize=(10, 5))
        plt.plot(stats["judgment_counts"], label="Judgment Word Counts", marker="o")
        plt.plot(stats["summary_counts"], label="Summary Word Counts", marker="s")
        plt.xlabel("Index")
        plt.ylabel("Word Count")
        plt.legend()
        plt.show()