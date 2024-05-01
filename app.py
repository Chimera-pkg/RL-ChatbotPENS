import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack
from flask import Flask, jsonify, render_template, request
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
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=['GET','POST'])


def get_bot_response():
    while True:
        query = request.args.get('msg')
        sql = "INSERT INTO pertanyaan (pertanyaan) VALUES (%s)"
        val = (query,)
        mycursor.execute(sql,val)
        mydb.commit()
        processed_query = text_preprocessing(query)
        tokens_query = text_tokenizing(processed_query)
        filtered_tokens_query = text_filtering(tokens_query)
        stemmed_tokens_query = text_stemming(filtered_tokens_query)
        processed_query = ' '.join(stemmed_tokens_query)
        vectorizer = TfidfVectorizer()

        if os.path.exists('tfidf_matrix_dataset.pkl'):
            tfidf_matrix_dataset = joblib.load('tfidf_matrix_dataset.pkl')
            vectorizer.fit(processed_texts)
            tfidf_df = pd.DataFrame(tfidf_matrix_dataset.toarray(), columns=vectorizer.get_feature_names_out())
            tfidf_matrix_query = vectorizer.transform([processed_query])

        else:
            tfidf_matrix_dataset = vectorizer.fit_transform(processed_texts)
            tfidf_matrix_query = vectorizer.transform([processed_query])
            tfidf_df = pd.DataFrame(tfidf_matrix_dataset.toarray(), columns=vectorizer.get_feature_names_out())
            joblib.dump(tfidf_matrix_dataset, 'tfidf_matrix_dataset.pkl')

        while True:
            tfidf_matrix = vstack([tfidf_matrix_dataset, tfidf_matrix_query])
            cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
            most_similar_idx = np.argmax(cosine_similarities)
            most_similar_idx_str = str(most_similar_idx)
            jawaban = dataset['answer'][most_similar_idx]
            sql = "INSERT INTO jawaban (jawaban) VALUES (%s)"
            val = (jawaban,)
            mycursor.execute(sql,val)
            mydb.commit()
            print("\nAnswer:")
            print(jawaban)
            response = {
                'jawaban': jawaban
            }
        
            return jsonify(response)

@app.route("/response", methods=["POST"])
def handle_response():
    # Ambil respons dari permintaan POST
    response = request.args.get("response")
    
    # Lakukan sesuatu dengan respons, seperti memanggil fungsi sendResponse
    # Pastikan untuk menentukan logika bisnis Anda di sini

    # Kembalikan respons HTTP yang sesuai, misalnya 200 OK
    return "Response received successfully"

@app.route("/reward", methods=["POST"])
def reward():
    score = request.args.get("score")
    jawaban = request.args.get("jawaban")

    # Tambahkan reward ke skor jawaban dalam database
    sql = "UPDATE jawaban SET (score) = score + %s WHERE jawaban = %s"
    val = (score, jawaban)
    mycursor.execute(sql, val)
    mydb.commit()

    return "Reward berhasil diberikan"

@app.route("/punish", methods=["POST"])
def punish():
    score = request.args.get("score")
    jawaban = request.args.get("jawaban")

    # Kurangi skor jawaban dalam database
    sql = "UPDATE jawaban SET (score) = score - %s WHERE jawaban = %s"
    val = (score, jawaban)
    mycursor.execute(sql, val)
    mydb.commit()

    return "Punishment berhasil diberikan"

if __name__ == "__main__":
    app.run()