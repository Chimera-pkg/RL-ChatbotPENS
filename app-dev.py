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

dataset = pd.read_csv('SampleDataset.csv')
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

class QuestionCheck:
    def __init__(self, query):
        self.query = query

    def check_question(self):
        sql_check = "SELECT pertanyaan FROM pertanyaan WHERE pertanyaan = %s"
        val_check = (self.query,)
        mycursor.execute(sql_check, val_check)
        existing_question = mycursor.fetchone()

        if existing_question:
            print("Pertanyaan sudah ada dalam database")
        else:
            sql_insert = "INSERT INTO pertanyaan (pertanyaan) VALUES (%s)"
            val_insert = (self.query,)
            mycursor.execute(sql_insert, val_insert)
            print("Pertanyaan berhasil dimasukkan")

@app.route("/get", methods=['GET','POST'])
def get_bot_response():
    query = request.args.get('msg')
    print(f"Query: {query}")
    
    question_checker = QuestionCheck(query)
    question_checker.check_question()
    
    # Preprocess
    processed_query = text_preprocessing(query)
    tokens_query = text_tokenizing(processed_query)
    filtered_tokens_query = text_filtering(tokens_query)
    stemmed_tokens_query = text_stemming(filtered_tokens_query)
    processed_query = ' '.join(stemmed_tokens_query)
    vectorizer = TfidfVectorizer()

    # TF IDF
    if os.path.exists('tfidf_matrix_dataset.pkl'):
        tfidf_matrix_dataset = joblib.load('tfidf_matrix_dataset.pkl')
        vectorizer.fit(processed_texts)
        tfidf_matrix_query = vectorizer.transform([processed_query])
    else:
        tfidf_matrix_dataset = vectorizer.fit_transform(processed_texts)
        tfidf_matrix_query = vectorizer.transform([processed_query])
        joblib.dump(tfidf_matrix_dataset, 'tfidf_matrix_dataset.pkl')

    # Cosine
    tfidf_matrix = vstack([tfidf_matrix_dataset, tfidf_matrix_query])
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    cosine_sim = cosine_similarity(tfidf_matrix_dataset, tfidf_matrix_query)
    most_similar_idx = np.argmax(cosine_similarities)

    jawaban = dataset['answer'][most_similar_idx]
    cosine_similarity_value = float(cosine_sim[most_similar_idx])

    # Get ID by pertanyaan
    mycursor.execute("SELECT id FROM pertanyaan WHERE pertanyaan LIKE %s ORDER BY createdAt DESC LIMIT 1", (f"%{query}%",))
    result = mycursor.fetchone()
    pertanyaan_id = result[0] if result else 1
    print(f"Pertanyaan ID: {pertanyaan_id}")

    # Convert dataset questions to lowercase for case-insensitive comparison
    dataset['question'] = dataset['question'].str.lower()
    query_lower = query.lower()

    # Print the full dataset for debugging
    print(f"Dataset: {dataset}")

    # Filter dataset for the input query (case-insensitive)
    filtered_dataset = dataset[dataset['question'] == query_lower]
    print(f"Filtered Dataset: {filtered_dataset}")

    # Check Multiple Answer existed in DB
    jawaban_values = []
    checked_answers = set()

    for idx, row in filtered_dataset.iterrows():
        answer = row['answer']
        answer2 = row.get('answer2', '')
        answer3 = row.get('answer3', '')
        cosine_similarity_value = float(cosine_similarities[0][idx])

        for ans in [answer, answer2, answer3]:
            if ans and (ans, pertanyaan_id) not in checked_answers:
                mycursor.execute("""
                    SELECT COUNT(*) FROM jawaban 
                    WHERE jawaban = %s AND pertanyaanId = %s
                """, (ans, pertanyaan_id))
                count = mycursor.fetchone()[0]

                if count == 0:
                    jawaban_values.append((ans, cosine_similarity_value, pertanyaan_id))
                checked_answers.add((ans, pertanyaan_id))

    # Debugging: Check the answers to be inserted
    print(f"Jawaban Values: {jawaban_values}")

    # Store Answer DB
    if jawaban_values:  # Ensure there are values to insert
        sql = "INSERT INTO jawaban (jawaban, cosine, score, pertanyaanId) VALUES (%s, %s, 3, %s)"
        mycursor.executemany(sql, jawaban_values)
        mydb.commit()

    # Checking Answer To Show Bubble Chat HTML
    mycursor.execute("""
        SELECT jawaban FROM jawaban 
        WHERE pertanyaanId = %s 
        ORDER BY score DESC 
        LIMIT 1
    """, (pertanyaan_id,))
    top_responses = mycursor.fetchall()
    print(f"Top Responses: {top_responses}")

    # Show Answer Bubble Chat HTML
    response = {
        'jawaban': [row[0] for row in top_responses]
    }
    print(f"Response: {response}")

    return jsonify(response)


@app.route("/response", methods=['GET', 'POST'])
def handle_response():
    if request.method == 'POST':
        data = request.get_json()
        response = data.get('response')

        if response == "yes":
            jawaban = data.get("jawaban")
            print("ISI HANDLE RESPONSE")
            print(jawaban)
            reward(jawaban)
        else:
            jawaban = data.get("jawaban")
            punish(jawaban)

        return "Response handled successfully"
    else:
        return "Invalid method"


def reward(jawaban_id): #how to Parsing jawaban_id
    print("ISI JAWABAN ID dari parsing an")
    print(jawaban_id)
    sql_select_id = "SELECT id FROM jawaban WHERE id = %s" # Belum adannya jawaban id yang diget
    mycursor.execute(sql_select_id, (jawaban_id,))
    result = mycursor.fetchone()
    selected_id = result[0] if result else None

    if selected_id:
        sql_update_score = "UPDATE jawaban SET score = score + (0.1 * score) WHERE id = %s"
        mycursor.execute(sql_update_score, (jawaban_id,))
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


if __name__ == "__main__":
    app.run()