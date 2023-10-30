import Anlamsal
import CumleSkor
import MetinOzetleme
import RougeSkor
import re
import networkx as nx
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Arayuzun olusturulmasi ve main akis (1)
class DocumentUploaderApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("KOU Folder Desktop Application")
        self.geometry("600x400")  # Width x Height
        self.file_path1 = ""
        self.file_path2 = ""
        self.selected_model = tk.StringVar(value="BERT")  # Varsayılan olarak "BERT" seçili
        self.score_threshold = tk.StringVar()
        self.similarity_threshold = tk.StringVar()

        self.upload1_button = tk.Button(self, text="Select Text", command=self.select_document1)
        self.upload1_button.grid(row=1, column=1, pady=5)

        self.upload2_button = tk.Button(self, text="Select Summary", command=self.select_document2)
        self.upload2_button.grid(row=2, column=1, pady=5)

        self.score_algorithm_label = tk.Label(self, text="Select Algorithm:")
        self.score_algorithm_label.grid(row=0, column=3, pady=5)

        self.model_bert = tk.Radiobutton(self, text="BERT", variable=self.selected_model, value="BERT")
        self.model_bert.grid(row=1, column=3, pady=5)

        self.model_word2vec = tk.Radiobutton(self, text="Word2Vec", variable=self.selected_model, value="Word2Vec")
        self.model_word2vec.grid(row=2, column=3, pady=5)

        self.score_threshold_label = tk.Label(self, text="Score Threshold:")
        self.score_threshold_label.grid(row=7, column=1, pady=5)

        self.score_threshold_entry = tk.Entry(self, textvariable=self.score_threshold)
        self.score_threshold_entry.grid(row=7, column=2, pady=5)

        self.similarity_threshold_label = tk.Label(self, text="Similarity Threshold:")
        self.similarity_threshold_label.grid(row=9, column=1, pady=5)

        self.similarity_threshold_entry = tk.Entry(self, textvariable=self.similarity_threshold)
        self.similarity_threshold_entry.grid(row=9, column=2, pady=5)

        self.calculate_button = tk.Button(self, text="Calculate", command=self.calculate)
        self.calculate_button.grid(row=10, column=3, pady=5)

        self.result_label = tk.Label(self, text="")
        self.result_label.grid(row=7, column=0, pady=5)

    def select_document1(self):
        self.file_path1 = filedialog.askopenfilename(filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        print("Metin:", self.file_path1)

    def select_document2(self):
        self.file_path2 = filedialog.askopenfilename(filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        print("Özet:", self.file_path2)

    def calculate(self):
        selected_model = self.selected_model.get()
        score_threshold = self.score_threshold.get()
        similarity_threshold = self.similarity_threshold.get()
        print("Selected model:", selected_model)
        if self.file_path1 and self.file_path2:
            with open(self.file_path2, 'r') as file2:
                real_summary = file2.read()
            with open(self.file_path1, 'r') as file1:
                text = file1.read()

                # Split the text into sentences
                sentences = re.split(r'(?<=[.!?])\s+', text)

                # Remove whitespace from each sentence
                sentences = [s.strip() for s in sentences]

                # Setting the title and main_text variables
                title = sentences[0]
                sentences = sentences[1:]

                # Cumle skorlari ve onislemeler
                main_text = ''
                for sentence in sentences:
                    main_text += sentence + ''

                sentence_scores = {}
                for sentence in sentences:
                    p1_p2_score = CumleSkor.calculate_sentence_score_p1_p2(sentence)
                    sentence = Anlamsal.preprocess_sentence(sentence)
                    p3_p4_p5_score = CumleSkor.calculate_sentence_score_p3_p4_p5(main_text, sentence,
                                                                                 title, score_threshold)
                    sentence_scores[sentence] = p1_p2_score + p3_p4_p5_score

                # Cumleler arasi iliski (BERT or Word2Vec)
                similarities = {}
                if selected_model == 'BERT':
                    similarities = Anlamsal.bert(main_text)

                if selected_model == 'Word2Vec':
                    similarities = Anlamsal.word2vec(main_text)

                # Metnin ozetinin olusturulmasi
                summary = MetinOzetleme.generate_summary(main_text, title, similarities,
                                                         score_threshold, similarity_threshold)
                print(summary)
                print(real_summary)

                # Rouge skoru hesaplanmasi
                #rouge_skor = RougeSkor.Rouge(str(summary), str(real_summary))
                #print(rouge_skor)

                # Create a document-term matrix using bag-of-words representation
                vectorizer = CountVectorizer()
                doc_term_matrix = vectorizer.fit_transform(sentences)

                # Calculate the cosine similarity between each pair of sentences
                similarity_matrix = cosine_similarity(doc_term_matrix)

                # Create a network graph of the sentences based on their similarity
                G = nx.DiGraph()
                G.add_nodes_from(range(len(sentences)))
                for i in range(len(sentences)):
                    for j in range(i + 1, len(sentences)):
                        if similarity_matrix[i][j] > 0.3:
                            G.add_edge(i, j)

                # Make the first sentence (title) special
                G.nodes[0]['color'] = 'red'

                # Draw the graph with custom node color
                pos = nx.spring_layout(G)
                nx.draw_networkx_nodes(G, pos, node_color=[G.nodes[n].get('color', 'blue') for n in G.nodes()])
                nx.draw_networkx_labels(G, pos)

                # Print the title
                if title:
                    title_label = tk.Label(self, text=title)
                    #title_label.pack()
                    title_label.config(fg="red")

                # Print sentences on another screen
                top = tk.Toplevel()
                top.title("Sentences")
                for i, sentence in enumerate(sentences):
                    label = tk.Label(top, text=sentence, fg="black")
                    label.pack()

                nx.draw_networkx_edges(G, pos, edge_color='black', arrows=True)
                plt.show()

        else:
            print("Dosya(lar) seçilmedi!")


app = DocumentUploaderApp()
app.mainloop()
