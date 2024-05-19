from flask import Blueprint, request, jsonify, render_template
from database import execute_query, fetch_last_question_id, fetch_all
from preprocess import text_preprocessing, text_tokenizing, text_filtering, text_stemming, preprocess_texts
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack
import joblib
import os
import pandas as pd
import numpy as np
import logging

routes = Blueprint('routes', __name__)

# Load dataset
dataset = pd.read_csv('train_new.csv')
texts = dataset['question'].tolist()
processed_texts = preprocess_texts(texts)

@routes.route("/")
def home():
    return render_template("index.html")

@routes.route("/get", methods=['GET', 'POST'])
def get_bot_response():
    query = request.args.get('msg')
    sql = "INSERT INTO pertanyaan (pertanyaan) VALUES (%s)"
    execute_query(sql, (query,))

    processed_query = ' '.join(text_stemming(text_filtering(text_tokenizing(text_preprocessing(query)))))
    vectorizer = TfidfVectorizer()

    tfidf_matrix_file = 'tfidf_matrix_dataset.pkl'
    if os.path.exists(tfidf_matrix_file):
        tfidf_matrix_dataset = joblib.load(tfidf_matrix_file)
        vectorizer.fit(processed_texts)
    else:
        tfidf_matrix_dataset = vectorizer.fit_transform(processed_texts)
        joblib.dump(tfidf_matrix_dataset, tfidf_matrix_file)

    tfidf_matrix_query = vectorizer.transform([processed_query])
    tfidf_matrix = vstack([tfidf_matrix_dataset, tfidf_matrix_query])
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    most_similar_idx = np.argmax(cosine_similarities)
    jawaban = dataset['answer'][most_similar_idx]
    cosine_similarity_value = float(cosine_similarity(tfidf_matrix_dataset, tfidf_matrix_query)[most_similar_idx])

    validateRL(cosine_similarity_value, jawaban)
    pertanyaan_id = fetch_last_question_id()
    sql = "INSERT INTO jawaban (jawaban, cosine, score, pertanyaanId) VALUES (%s, %s, 3, %s)"
    execute_query(sql, (jawaban, cosine_similarity_value, pertanyaan_id))

    logging.info(f"Answer: {jawaban}")
    return jsonify({'jawaban': jawaban})

@routes.route("/response", methods=['GET', 'POST'])
def handle_response():
    if request.method == 'POST':
        data = request.get_json()
        response = data.get('response')
        jawaban = data.get("jawaban")
        if response == "yes":
            reward(jawaban)
        else:
            punish(jawaban)
        return "Response handled successfully"
    return "Invalid method"

def reward(jawaban_id):
    """Increase the score of the last inserted answer."""
    last_id = fetch_last_question_id()
    if last_id:
        sql_update_score = "UPDATE jawaban SET score = score + (score + (0.1 * score)) WHERE id = %s"
        execute_query(sql_update_score, (last_id,))
        logging.info("Reward Score berhasil diperbarui.")
    else:
        logging.info("Reward Gagal Masuk.")

def punish(jawaban_id):
    """Decrease the score of the last inserted answer."""
    last_id = fetch_last_question_id()
    if last_id:
        sql_update_score = "UPDATE jawaban SET score = score - (score - (0.1 * score)) WHERE id = %s"
        execute_query(sql_update_score, (last_id,))
        logging.info("Punish Score diperbarui.")
    else:
        logging.info("Punish Gagal dimasukkan.")

def validateRL(cosine_similarity_value, jawaban, score=None):
    """Validate and display top answers based on cosine similarity and score."""
    sql = "SELECT jawaban, cosine, score FROM jawaban ORDER BY score DESC, cosine DESC LIMIT 5"
    results = fetch_all(sql)

    if results:
        top_answers = []
        for result in results:
            best_jawaban, best_cosine_similarity_value, best_score = result
            logging.info(f"\nTop Answer:\n{best_jawaban}\nCosine Similarity: {best_cosine_similarity_value}\nScore: {best_score}")
            top_answers.append({'jawaban': best_jawaban, 'cosine_similarity': best_cosine_similarity_value, 'score': best_score})
        return jsonify(top_answers)
    return jsonify({'error': 'No answer found'})
