import Anlamsal
import CumleSkor
from nltk.tokenize import sent_tokenize


# Cumle skorlarina gore secerek ozetleme (4)
def generate_summary(main_text, title, similarities, score_threshold, similarity_threshold):
    sentences = sent_tokenize(main_text)
    sentence_scores = []

    for i, sentence in enumerate(sentences):
        score1 = CumleSkor.calculate_sentence_score_p1_p2(sentence)
        sentence2 = Anlamsal.preprocess_sentence(sentence)
        threshold1 = float(score_threshold)
        score2 = CumleSkor.calculate_sentence_score_p3_p4_p5(main_text, sentence2, title, threshold1)
        score = score1 + score2
        sentence_scores.append((sentence, score, similarities[i]))

    # Cümleleri skorlarına ve benzerlik oranlarına göre sırala
    sorted_sentences = sorted(sentence_scores, key=lambda x: (x[1], x[2]), reverse=True)

    # Cümleleri benzerlik oranı eşik değeriyle filtrele
    threshold2 = float(similarity_threshold)
    filtered_sentences = [sentence for sentence, score, similarity in sorted_sentences if similarity >= threshold2]

    # Özet oluştur
    summary = " ".join(filtered_sentences[:5])  # İlk 5 cümleyi özet olarak al

    return summary
