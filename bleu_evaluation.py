import sacrebleu
import matplotlib.pyplot as plt

def calculate_bleu(references, hypothesis):
    # SacreBLEU expects references to be a list of list of strings and hypothesis to be a list of strings
    return sacrebleu.corpus_bleu([hypothesis], references).score

def plot_bleu_scores(with_synonym_scores, without_synonym_scores, titles):
    plt.figure(figsize=(10, 6))
    plt.plot(titles, with_synonym_scores, 'o-', label='Dengan Pengenalan Sinonim')
    plt.plot(titles, without_synonym_scores, 'o-', label='Tanpa Pengenalan Sinonim')
    plt.xlabel('Kasus Uji')
    plt.ylabel('Skor BLEU')
    plt.title('Perbandingan Skor BLEU dengan dan tanpa Pengenalan Sinonim')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Data contoh untuk kasus uji
    references = [
        ['menurut anda, bagaimana peran tni dan polri dalam kestabilan politik dalam negeri'],
        ['menurut anda, bagaimana peran tni dan polri dalam menjaga stabilitas politik dalam negeri'],
        ['peran tni dan polri sangat penting dalam menjaga stabilitas politik'],
        ['bagaimana menurut anda peran tni dan polri dalam stabilitas politik'],
        ['seberapa penting peran tni dan polri dalam menjaga stabilitas politik dalam negeri']
    ]
    
    hypotheses_with_synonym = [
        'Bagaimana pandangan Anda tentang peran TNI dan Polri dalam menjaga stabilitas politik dalam negeri?',
        'Bagaimana pandangan Anda tentang peran TNI dan Polri dalam menjaga stabilitas politik dalam negeri?',
        'Peran TNI dan Polri sangat penting dalam menjaga stabilitas politik dalam negeri.',
        'Bagaimana pandangan Anda tentang peran TNI dan Polri dalam menjaga stabilitas politik dalam negeri?',
        'Seberapa penting peran TNI dan Polri dalam menjaga stabilitas politik dalam negeri?'
    ]
    
    hypotheses_without_synonym = [
        'menurut pandangan anda peran TNI Polri kestabilan politik negeri',
        'bagaimana pandangan Anda peran TNI Polri menjaga stabilitas politik dalam negeri',
        'peran TNI Polri sangat penting dalam menjaga kestabilan politik',
        'bagaimana menurut anda peran TNI Polri dalam kestabilan politik',
        'seberapa penting peran TNI Polri menjaga kestabilan politik negeri'
    ]
    
    # Menghitung BLEU scores dengan dan tanpa sinonim
    with_synonym_scores = [calculate_bleu(references, hyp) for hyp in hypotheses_with_synonym]
    without_synonym_scores = [calculate_bleu(references, hyp) for hyp in hypotheses_without_synonym]
    
    # Menampilkan hasil BLEU score
    for i, (with_syn, without_syn) in enumerate(zip(with_synonym_scores, without_synonym_scores)):
        print(f"Kasus {i+1} - Dengan Pengenalan Sinonim: {with_syn:.2f}, Tanpa Pengenalan Sinonim: {without_syn:.2f}")
    
    # Plotting the BLEU scores
    plot_bleu_scores(with_synonym_scores, without_synonym_scores, list(range(1, len(references) + 1)))
