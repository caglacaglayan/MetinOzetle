import string
import torch
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel


# Cumleler icin metin onisleme adimlari (2)
def preprocess_sentence(cumle):
    # Tokenization
    kelimeler = word_tokenize(cumle)

    # Stemming
    stemmer = SnowballStemmer("english")
    kelimeler = [stemmer.stem(kelime) for kelime in kelimeler]

    # Stop-word Elimination
    stop_words = set(stopwords.words("english"))
    kelimeler = [kelime for kelime in kelimeler if kelime.lower() not in stop_words]

    # Punctuation
    kelimeler = [kelime for kelime in kelimeler if kelime not in string.punctuation]

    sentence = ''
    for kelime in kelimeler:
        sentence += kelime

    return sentence


# Word Embedding (Word2Vec):
def word2vec(main_text):
    # Cümleleri tokenizasyon yapın
    sentences = sent_tokenize(main_text)
    tokens = [word_tokenize(sentence) for sentence in sentences]

    # Word2Vec modelini eğitin
    model = Word2Vec(tokens, min_count=1)

    # Cümle vektörlerini hesaplayın
    vectors = [np.mean([model.wv[token] for token in token_list], axis=0) for token_list in tokens]

    # Kosinus benzerliği matrisini hesaplayın
    similarity_matrix = cosine_similarity(vectors)

    return similarity_matrix


# BERT:
def bert(main_text):
    # BERT modelünü yükleyin
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    # Cümleleri tokenizasyon yapın
    sentences = sent_tokenize(main_text)
    encoded_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

    # Cümleleri BERT'e göndererek kodlayın
    with torch.no_grad():
        outputs = model(**encoded_inputs)
        sentence_embeddings = torch.mean(outputs.last_hidden_state, dim=1)

    # Kosinus benzerliği matrisini hesaplayın
    similarity_matrix = cosine_similarity(sentence_embeddings)

    return similarity_matrix
