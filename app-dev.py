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

dataset = pd.read_csv('reinforcement.csv')
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
        sql_check = "SELECT pertanyaan FROM pertanyaan WHERE pertanyaan = %s"
        val_check = (query,)
        mycursor.execute(sql_check, val_check)
        existing_question = mycursor.fetchone()

        if existing_question:
            print("Pertanyaan sudah ada dalam database")
        else:
            sql_insert = "INSERT INTO pertanyaan (pertanyaan) VALUES (%s)"
            val_insert = (query,)
            mycursor.execute(sql_insert, val_insert)
            print("Pertanyaan berhasil dimasukkan")

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
        top_3_indices = np.argsort(cosine_similarities[0])[-3:][::-1]
        cosine_sim = cosine_similarity(tfidf_matrix_dataset, tfidf_matrix_query)
        most_similar_idx = np.argmax(cosine_similarities)
        jawaban = dataset['answer'][most_similar_idx]
        cosine_similarity_value = float(cosine_sim[most_similar_idx])
        # validateRL(cosine_similarity_value,jawaban)  
        
        ambil_pertanyaan = "SELECT id FROM pertanyaan ORDER BY createdAt DESC LIMIT 1"
        mycursor.execute(ambil_pertanyaan)
        result = mycursor.fetchone()
        print(result)
        pertanyaan_id = result[0] if result else 1
        
        # Prepare the values for executemany
        jawaban_values = []
        existing_answers = set()  # Membuat set untuk menyimpan jawaban yang sudah ada

        for idx in top_3_indices:
            answer = dataset.iloc[idx]['answer']
            answer2 = dataset.iloc[idx].get('answer2', '')
            answer3 = dataset.iloc[idx].get('answer3', '')
            cosine_similarity_value = float(cosine_similarities[0][idx])

            # Pengecekan jawaban 1
            mycursor.execute("""
                SELECT COUNT(*) FROM jawaban 
                WHERE jawaban = %s AND pertanyaanId = %s
            """, (answer, pertanyaan_id))
            count = mycursor.fetchone()[0]

            if count == 0:
                jawaban_values.append((answer, cosine_similarity_value, pertanyaan_id))

            # Pengecekan jawaban 2
            if answer2:
                mycursor.execute("""
                    SELECT COUNT(*) FROM jawaban 
                    WHERE jawaban = %s AND pertanyaanId = %s
                """, (answer2, pertanyaan_id))
                count = mycursor.fetchone()[0]

                if count == 0:
                    jawaban_values.append((answer2, cosine_similarity_value, pertanyaan_id))

            # Pengecekan jawaban 3
            if answer3:
                mycursor.execute("""
                    SELECT COUNT(*) FROM jawaban 
                    WHERE jawaban = %s AND pertanyaanId = %s
                """, (answer3, pertanyaan_id))
                count = mycursor.fetchone()[0]

                if count == 0:
                    jawaban_values.append((answer3, cosine_similarity_value, pertanyaan_id))


        sql = "INSERT INTO jawaban (jawaban, cosine, score, pertanyaanId) VALUES (%s, %s, 3, %s)"
        mycursor.executemany(sql, jawaban_values)
        mydb.commit()

        # Fetch the top 3 responses based on cosine similarity from the database
        mycursor.execute("""
            SELECT jawaban FROM jawaban 
            WHERE pertanyaanId = %s 
            ORDER BY score DESC 
            LIMIT 1
        """, (pertanyaan_id,))
        top_responses = mycursor.fetchall()

        response = {
            'jawaban': [row[0] for row in top_responses]
        }
        print(jawaban)

        return jsonify(response)


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

def validateRL(cosine_similarity_value, jawaban, score=None):
    # Fetch the top three highest scored answers considering both cosine similarity and score
    sql = "SELECT jawaban, cosine, score FROM jawaban ORDER BY score DESC, cosine DESC LIMIT 3"
    mycursor.execute(sql)
    results = mycursor.fetchall()

    if results:
        top_answers = []
        for result in results:
            # Extract the values from each fetched row
            best_jawaban, best_cosine_similarity_value, best_score = result

            # Print and construct response for each of the top answers
            print("\nTop Answer:")
            print(best_jawaban)
            print("Cosine Similarity:", best_cosine_similarity_value)
            print("Score:", best_score)

            response = {
                'jawaban': best_jawaban,
                'cosine_similarity': best_cosine_similarity_value,
                'score': best_score
            }
            top_answers.append(response)
        
        return jsonify(top_answers)
    else:
        return jsonify({'error': 'No answer found'})



if __name__ == "__main__":
    app.run()