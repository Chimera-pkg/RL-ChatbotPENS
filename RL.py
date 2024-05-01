import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 1. Membaca Dataset dari CSV
dataset = pd.read_csv('train_new.csv')

# 2. Inisialisasi Chatbot
pertanyaan = dataset['question'].tolist()
jawaban = dataset['answer'].tolist()

# Inisialisasi TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(pertanyaan)

# Inisialisasi reward dan punishment menggunakan NumPy
reward_punishment = np.zeros((len(jawaban), 2))  # Matriks untuk menyimpan reward dan punishment
np.save('reward_punishment.npy', reward_punishment)  # Menyimpan data ke file npy

# 3. Menerima Pertanyaan
@app.route("/chat", methods=["POST"])
def chat():
    pertanyaan_pengguna = request.json.get("pertanyaan")

    # 4. Menghitung Similarity
    vektor_tfidf_pertanyaan = tfidf_vectorizer.transform([pertanyaan_pengguna])
    cosine_similarities = cosine_similarity(vektor_tfidf_pertanyaan, tfidf_matrix)
    best_match_index = cosine_similarities.argmax()

    # 5. Pilih Jawaban Terbaik
    jawaban_terbaik = jawaban[best_match_index]

    return jsonify({"jawaban": jawaban_terbaik})

# 6. Menerima Feedback
@app.route("/feedback", methods=["POST"])
def feedback():
    feedback_data = request.json
    jawaban_terpilih = feedback_data.get("jawaban")
    feedback = feedback_data.get("feedback")

    # 7. Mengupdate Reward
    reward_punishment = np.load('reward_punishment.npy')
    if feedback == "ya":
        reward_punishment[jawaban.index(jawaban_terpilih), 0] += 1  # Increment reward
    elif feedback == "tidak":
        reward_punishment[jawaban.index(jawaban_terpilih), 1] += 1  # Increment punishment

    np.save('reward_punishment.npy', reward_punishment)  # Menyimpan data ke file npy

    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True)
