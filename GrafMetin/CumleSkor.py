import math
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer


# Skor hesaplamalari (3)

# Özel isimleri kontrol etmek için gerekli fonksiyon (P1)
def count_special_names(sentence):
    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    count = 0
    for word, tag in pos_tags:
        if tag == 'NNP':  # Özel isim etiketi
            count += 1
    return count


# Cümledeki numerik verileri kontrol etmek için gerekli fonksiyon (P2)
def count_numerical_data(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    count = 0
    for word in sentence_words:
        if word.isdigit():
            count += 1
    return count


# Cümledeki başlıkta geçen kelimeleri kontrol etmek için gerekli fonksiyon (P4)
def count_title_words(sentence, title):
    title_words = nltk.word_tokenize(title)
    sentence_words = nltk.word_tokenize(sentence)
    count = 0
    for word in sentence_words:
        if word.lower() in title_words:
            count += 1
    return count


# Cümlede metnin tema kelimelerinin olup olmadığını kontrol etmek için gerekli fonksiyon (P5)
def count_theme_words(sentence, text):
    theme_words = get_theme_words(text)
    sentence_words = sentence.split()

    count = 0
    for word in sentence_words:
        if word.lower() in theme_words:
            count += 1
    return count


def calculate_tfidf(text):
    vectorizer = TfidfVectorizer()
    text_vector = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()

    word_tfidf = {}
    feature_index = text_vector.nonzero()[1]
    tfidf_scores = zip(feature_index, [text_vector[0, x] for x in feature_index])
    for word_index, tfidf_score in tfidf_scores:
        word = feature_names[word_index]
        word_tfidf[word] = tfidf_score

    return word_tfidf


def get_theme_words(text):
    word_tfidf_scores = calculate_tfidf(text)
    total_words = len(text.split())
    theme_words_count = math.ceil(total_words * 0.1)
    theme_words = []
    sorted_words = sorted(word_tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    for word, tfidf in sorted_words[:theme_words_count]:
        theme_words.append(word)

    return theme_words


# Cümle skoru hesaplama fonksiyonu 1
def calculate_sentence_score_p1_p2(sentence):
    sentence_length = len(sentence.split())
    p1 = count_special_names(sentence) / sentence_length
    p2 = count_numerical_data(sentence) / sentence_length

    sentence_score = p1 + p2
    return sentence_score


# Cümle skoru hesaplama fonksiyonu 2
def calculate_sentence_score_p3_p4_p5(text, sentence, title, score_threshold):
    tfidf_scores = calculate_tfidf(text)
    sentence_length = len(sentence.split())
    # p3 = sum(1 for word in sentence.split() if tfidf_scores[sentence].get(word, 0) > score_threshold) / len(tfidf_scores)
    p4 = count_title_words(sentence, title) / sentence_length
    p5 = count_theme_words(sentence, text) / sentence_length

    sentence_score = p4 + p5 #+p3
    return sentence_score
