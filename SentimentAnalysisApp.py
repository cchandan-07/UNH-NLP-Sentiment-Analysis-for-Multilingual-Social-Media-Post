import tkinter as tk
from tkinter import filedialog
import pandas as pd
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch
import seaborn as sns
from nltk import ngrams
from collections import Counter

class SentimentAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentiment Analysis GUI")

        self.label = tk.Label(root, text="Select File or Enter Text:")
        self.label.pack(pady=10)

        self.text_input = tk.Text(root, height=5, width=50)
        self.text_input.pack(pady=10)

        self.browse_button = tk.Button(root, text="Browse", command=self.load_data)
        self.browse_button.pack(pady=5)

        self.file_type_var = tk.StringVar(value="Text")
        self.file_type_menu = tk.OptionMenu(root, self.file_type_var, "Text", "CSV", "Other")
        self.file_type_menu.pack(pady=5)

        self.language_label = tk.Label(root, text="Language:")
        self.language_label.pack(pady=5)

        self.language_var = tk.StringVar(value="English")
        self.language_menu = tk.OptionMenu(root, self.language_var, "English", "French", "Spanish", "Other")
        self.language_menu.pack(pady=5)

        self.analyze_button = tk.Button(root, text="Analyze Sentiment", command=self.perform_sentiment_analysis)
        self.analyze_button.pack(pady=10)

        self.result_label = tk.Label(root, text="")
        self.result_label.pack(pady=10)

        self.listbox_label = tk.Label(root, text="Sentiment Analysis Outputs:")
        self.listbox_label.pack(pady=10)

        self.listbox = tk.Listbox(root, selectmode=tk.SINGLE, width=50, height=10)
        self.listbox.pack(pady=10)

        self.table_label = tk.Label(root, text="Overall Words Analysis:")
        self.table_label.pack(pady=5)

        self.table = ttk.Treeview(root, columns=("Word", "Count"))
        self.table.heading("#0", text="Word")
        self.table.heading("#1", text="Count")
        self.table.pack(pady=5)

        self.clear_button = tk.Button(root, text="Clear Input", command=self.clear_input)
        self.clear_button.pack(pady=10)

        self.exit_button = tk.Button(root, text="Exit", command=root.destroy)
        self.exit_button.pack(pady=10)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.df = self.load_dataframe(file_path)
            self.result_label.config(text=f"Loaded {len(self.df)} entries from {file_path}")

    def load_dataframe(self, file_path):
        if self.file_type_var.get() == "CSV":
            return pd.read_csv(file_path)
        elif self.file_type_var.get() == "Text":
            return pd.DataFrame({"Text": [line.strip() for line in open(file_path, 'r', encoding='utf-8') if line.strip()]})
        else:
            return pd.DataFrame()

    def perform_sentiment_analysis(self):
        input_text = self.text_input.get("1.0", tk.END).strip()

        if not input_text and not hasattr(self, 'df'):
            self.result_label.config(text="Please enter text or load a CSV file.")
            return

        text_data = [input_text] if input_text else self.df['Text'].astype(str).tolist()

        sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

        sentiments = sentiment_analyzer(text_data)

        self.listbox.delete(0, tk.END)
        for i, sentiment in enumerate(sentiments):
            self.listbox.insert(tk.END, f"Entry {i + 1}: {sentiment['label']} ({sentiment['score']:.4f})")

        overall_sentiment = max(sentiments, key=lambda x: x['score'])
        self.result_label.config(text=f"Overall Sentiment: {overall_sentiment['label']} ({overall_sentiment['score']:.4f})")

        if not hasattr(self, 'df'):
            self.df = pd.DataFrame({"Text": text_data})

        self.df['Sentiment'] = [result['label'] for result in sentiments]

        self.df.to_csv("analyzed_sentiments.csv", index=False)

        overall_words_analysis = self.get_overall_words_analysis(text_data)
        self.display_overall_words_analysis(overall_words_analysis)

    def get_overall_words_analysis(self, text_data):
        combined_text = ' '.join(text_data)

        words = combined_text.split()

        n = 2
        ngrams_list = list(ngrams(words, n))

        ngrams_counter = Counter(ngrams_list)

        return ngrams_counter

    def display_overall_words_analysis(self, overall_words_analysis):
        for item in self.table.get_children():
            self.table.delete(item)

        for ngram, count in overall_words_analysis.items():
            self.table.insert("", tk.END, values=(f"{ngram[0]} {ngram[1]}", count))

    def clear_input(self):
        self.text_input.delete("1.0", tk.END)
        self.result_label.config(text="")
        self.listbox.delete(0, tk.END)
        self.df = None 
        self.clear_table()

    def clear_table(self):
        for item in self.table.get_children():
            self.table.delete(item)

if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentAnalyzerGUI(root)
    root.mainloop()
