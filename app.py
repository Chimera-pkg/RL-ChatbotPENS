import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack

import os
import pandas as pd
import joblib
import numpy as np

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS, cross_origin
import mysql.connector

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="chatbot"
)
mycursor = mydb.cursor()

dataset = pd.read_csv('train_new.csv')
texts = dataset['question'].tolist()

def text_preprocessing(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()

    return text

def text_tokenizing(text):
    tokens = word_tokenize(text)

    return tokens

def text_filtering(tokens):
    stop_words = set(stopwords.words('indonesian'))
    tokens = [word for word in tokens if word not in stop_words]

    return tokens

def text_stemming(tokens):
    stemmer = StemmerFactory().create_stemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]

    return stemmed_tokens

if os.path.exists('processed_texts.pkl'):
    processed_texts = joblib.load('processed_texts.pkl')
else:
    texts = dataset['question'].tolist()
    processed_texts = []

    for text in texts:
        text = text_preprocessing(text)
        tokens = text_tokenizing(text)
        filtered_tokens = text_filtering(tokens)
        stemmed_tokens = text_stemming(filtered_tokens)
        processed_text = ' '.join(stemmed_tokens)
        processed_texts.append(processed_text)

    # menyimpan preprocessing dataset
    joblib.dump(processed_texts, 'processed_texts.pkl')


app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=['GET','POST'])
def get_bot_response():
        query = request.args.get('msg')
        sql = "INSERT INTO pertanyaan (pertanyaan) VALUES (%s)"
        val = (query,)
        mycursor.execute(sql,val)
        processed_query = text_preprocessing(query)
        tokens_query = text_tokenizing(processed_query)
        filtered_tokens_query = text_filtering(tokens_query)
        stemmed_tokens_query = text_stemming(filtered_tokens_query)
        processed_query = ' '.join(stemmed_tokens_query)
        vectorizer = TfidfVectorizer()

        if os.path.exists('tfidf_matrix_dataset.pkl'):
            tfidf_matrix_dataset = joblib.load('tfidf_matrix_dataset.pkl')
            vectorizer.fit(processed_texts)
            tfidf_matrix_query = vectorizer.transform([processed_query])

        else:
            tfidf_matrix_dataset = vectorizer.fit_transform(processed_texts)
            tfidf_matrix_query = vectorizer.transform([processed_query])
            joblib.dump(tfidf_matrix_dataset, 'tfidf_matrix_dataset.pkl')

        tfidf_matrix = vstack([tfidf_matrix_dataset, tfidf_matrix_query])
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        cosine_sim = cosine_similarity(tfidf_matrix_dataset, tfidf_matrix_query)
        most_similar_idx = np.argmax(cosine_similarities)
        jawaban = dataset['answer'][most_similar_idx]
        cosine_similarity_value = float(cosine_sim[most_similar_idx])
        validateRL(cosine_similarity_value,jawaban)
        pertanyaan_id = 1
        sql = "INSERT INTO jawaban (jawaban, cosine, score, pertanyaanId) VALUES (%s, %s, 1, %s)"
        val = (jawaban, cosine_similarity_value,pertanyaan_id)
        mycursor.execute(sql, val)
        mydb.commit()
        print("\nAnswer:")
        print(jawaban)
        # print("Cosine Similarity:", cosine_similarity_value)
        response = {
            'jawaban': jawaban
        }
        return jsonify(response)

def response_path():
    data = request.get_json()
    if data is not None and 'yes' in data and 'no' in data:
        print('json masuk')
    else:
        print('Invalid JSON data')



@app.route("/response", methods=['GET', 'POST'])
def handle_response():
    if request.method == 'POST':
        data = request.get_json()
        response = data.get('response')

        if response == "yes":
            jawaban = data.get("jawaban")
            reward(jawaban)
        else:
            jawaban = data.get("jawaban")
            punish(jawaban)

        return "Response handled successfully"
    else:
        return "Invalid method"


def reward(jawaban_id):
    sql_select_last_id = "SELECT id FROM jawaban ORDER BY createdAt DESC LIMIT 1"
    mycursor.execute(sql_select_last_id)
    result = mycursor.fetchone()
    last_id = result[0] if result else None

    if last_id:
        sql_update_score = "UPDATE jawaban SET score = score + (score + (0.1 * score)) WHERE id = (SELECT id FROM jawaban ORDER BY createdAt DESC LIMIT 1)"
        mycursor.execute(sql_update_score)
        mydb.commit()
        print("Reward Score berhasil diperbarui.")
        answerRL()
        
    else:
        print("Reward Gagal Masuk.")


def punish(jawaban_id):
    sql_select_last_id = "SELECT id FROM jawaban ORDER BY createdAt DESC LIMIT 1"
    mycursor.execute(sql_select_last_id)
    result = mycursor.fetchone()
    last_id = result[0] if result else None

    if last_id:
        sql_update_score = "UPDATE jawaban SET score = score - (score - (0.1 * score)) WHERE id = (SELECT id FROM jawaban ORDER BY createdAt DESC LIMIT 1)"
        print(sql_update_score)
        mycursor.execute(sql_update_score)
        mydb.commit()
        print("Punish Score diperbarui.")
    else:
        print("Punish Gagal dimasukkan.")

def validateRL(cosine_similarity_value,jawaban):
    sql = "SELECT jawaban, cosine, score FROM jawaban ORDER BY score DESC, cosine DESC LIMIT 1"
    mycursor.execute(sql)
    result = mycursor.fetchone()
    if result:
        jawaban, cosine_similarity_value, score = result
        print("\nAnswer:")
        print(jawaban)
        print("Cosine Similarity:", cosine_similarity_value)
        print("Score:", score)

        response = {
            'jawaban': jawaban,
            'cosine_similarity': cosine_similarity_value,
            'score': score
        }
        return jsonify(response)
    else:
        return jsonify({'error': 'No answer found'})

def answerRL():
    result = mycursor.fetchone()
    if result:
        jawaban, cosine_similarity_value, score = result
        print("\nAnswer:")
        print(jawaban)
        print("Cosine Similarity:", cosine_similarity_value)
        print("Score:", score)

        response = {
            'jawaban': jawaban,
            'cosine_similarity': cosine_similarity_value,
            'score': score
        }
        return jsonify(response)
    else:
        return jsonify({'error': 'No answer found'})

if __name__ == "__main__":
    app.run()