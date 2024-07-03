import sacrebleu
import matplotlib.pyplot as plt
from nltk.corpus import wordnet

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def replace_with_synonyms(text):
    words = text.split()
    new_text = []
    for word in words:
        synonyms = get_synonyms(word)
        if synonyms:
            new_text.append(synonyms[0])  # Pilih sinonim pertama
        else:
            new_text.append(word)
    return ' '.join(new_text)

def calculate_bleu(references, hypothesis):
    return sacrebleu.corpus_bleu(hypothesis, [references]).score

def plot_bleu_scores(scores, titles, labels):
    plt.figure(figsize=(10, 6))
    for score, label in zip(scores, labels):
        plt.plot(titles, score, label=label, marker='o')
    plt.xlabel('Kasus Uji')
    plt.ylabel('Skor BLEU')
    plt.title('Perbandingan Skor BLEU dengan dan tanpa Pengenalan Sinonim')
    plt.legend()
    plt.ylim(0, 1)
    plt.show()

if __name__ == "__main__":
    # Contoh kalimat referensi dan hipotesis
    references = [
        'masa pengabdian komcad ada masa aktif dan tidak aktif masa aktif yaitu saat mengikuti pelatihan dan penyegaran dimana pelatihan penyegaran setiap tahun selama maksimal 90 hari dan minimal 12 hari sementara masa tidak aktif yaitu setelah selesai masa aktif dan kembali ke profesi semula',
        # Tambahkan lebih banyak referensi sesuai kebutuhan
    ]
    hypotheses = [
        'komcad memungkinkan memperbesar dan memperkuat komponen utama tni secara efisien tanpa harus memperbesar kekuatan tni yang membutuhkan anggaran jauh lebih besar praktik semacam ini juga dilakukan oleh negara negara yang memiliki anggaran serta kekuatan militer yang besar seperti as china rusia dan india',
        # Tambahkan lebih banyak hipotesis sesuai kebutuhan
    ]

    # Menghitung BLEU scores tanpa pengenalan sinonim
    bleu_scores_no_synonyms = [calculate_bleu(references, [hypothesis]) for hypothesis in hypotheses]

    # Menghitung BLEU scores dengan pengenalan sinonim
    hypotheses_with_synonyms = [replace_with_synonyms(hypothesis) for hypothesis in hypotheses]
    bleu_scores_with_synonyms = [calculate_bleu(references, [hypothesis]) for hypothesis in hypotheses_with_synonyms]

    # Menampilkan hasil BLEU scores
    print("BLEU Scores Tanpa Pengenalan Sinonim:", bleu_scores_no_synonyms)
    print("BLEU Scores Dengan Pengenalan Sinonim:", bleu_scores_with_synonyms)

    # Plotting the BLEU scores
    plot_bleu_scores([bleu_scores_no_synonyms, bleu_scores_with_synonyms], 
                     list(range(1, len(hypotheses) + 1)), 
                     ["Tanpa Pengenalan Sinonim", "Dengan Pengenalan Sinonim"])
