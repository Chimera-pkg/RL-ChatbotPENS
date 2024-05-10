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
       # Menyimpan jawaban dan cosine similarity ke dalam database
        sql = "INSERT INTO jawaban (jawaban, cosine) VALUES (%s, %s)"
        val = (jawaban, cosine_similarity_value)
        mycursor.execute(sql, val)
        mydb.commit()

        # Menampilkan jawaban dan cosine similarity
        print("\nAnswer:")
        print(jawaban)
        print("Cosine Similarity:", cosine_similarity_value)
        response = {
            'jawaban': jawaban
        }
        return jsonify(response)


@app.route("/response", methods=['GET','POST'])
def handle_response():
    response = request.args.get('sendResponse')

    if response == "yes":
        score = 10
        jawaban = request.args.get("jawaban")
        reward(jawaban, score)
        
    else:
        # Jika response bukan "benar", berikan punishment 5
        score = 5
        jawaban = request.args.get("jawaban")
        punish(jawaban, score)

    return "Response handled successfully"  # Pastikan untuk memberikan respon dari fungsi rute

def reward(jawaban_id, score):
    global mycursor
    global mydb
    sql_select_last_id = "SELECT id FROM jawaban ORDER BY createdAt DESC LIMIT 1"
    mycursor.execute(sql_select_last_id)
    result = mycursor.fetchone()
    last_id = result[0] if result else None

    if last_id:
        sql_update_score = "UPDATE jawaban SET score = score + 500 WHERE id = (SELECT id FROM jawaban ORDER BY createdAt DESC LIMIT 1)"
        mydb.commit()
        mycursor.execute(sql_update_score)
        print("Score berhasil diperbarui.")
    else:
        print("Tidak ada data jawaban untuk diperbarui.")
    
    return "Reward handled successfully"  # Pastikan untuk memberikan respon dari fungsi rute


def punish(jawaban, score):
    try:
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="chatbot"
        )
        mycursor = mydb.cursor()

        # Query SQL dengan placeholder untuk parameter
        sql = "UPDATE jawaban SET score = score + 500 WHERE id = (SELECT id FROM jawaban ORDER BY createdAt DESC LIMIT 1)"
        val = (jawaban, score)

        # Eksekusi query dengan parameter yang diberikan
        mycursor.execute(sql, val)

        mydb.commit()

        print(mycursor.rowcount, "record inserted.")

    except mysql.connector.Error as error:
        print("Failed to insert record into MySQL table:", error)

    finally:
        if mydb.is_connected():
            mycursor.close()
            mydb.close()



if __name__ == "__main__":
    app.run()