from rouge import Rouge


# Algoritma ile hesaplanan ve verilen gercek ozetin rouge skoru (5)
def rouge_score(cal_sum, real_sum):
    rouge = Rouge()
    scores = rouge.get_scores(cal_sum, real_sum, avg=True)

    rouge_1_score = scores["rouge-1"]
    rouge_2_score = scores["rouge-2"]
    rouge_l_score = scores["rouge-l"] #Longest Common Subsequence (LCS) tabanlı skor

    return rouge_l_score

cumle= 'jherjewrew. jekrhewjr.'
cumle2= 'evhewreuew. jekrhewjr.'
skore= rouge_score(cumle,cumle2)
print(skore)