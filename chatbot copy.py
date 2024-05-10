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

#MENGAMBIL DATASET
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



query = input("Masukkan pertanyaan Anda: ")

# Text mining untuk query input
processed_query = text_preprocessing(query)
tokens_query = text_tokenizing(processed_query)
filtered_tokens_query = text_filtering(tokens_query)
stemmed_tokens_query = text_stemming(filtered_tokens_query)
processed_query = ' '.join(stemmed_tokens_query)

# gabungkan dataset dengan query
# all_texts = processed_texts + [processed_query]

# tf idf vectorizer
vectorizer = TfidfVectorizer()
if os.path.exists('tfidf_matrix_dataset.pkl'):
    tfidf_matrix_dataset = joblib.load('tfidf_matrix_dataset.pkl')
    vectorizer.fit(processed_texts)
    tfidf_df = pd.DataFrame(tfidf_matrix_dataset.toarray(), columns=vectorizer.get_feature_names_out())
    print(tfidf_df)
    tfidf_matrix_query = vectorizer.transform([processed_query])

else:
    tfidf_matrix_dataset = vectorizer.fit_transform(processed_texts)
    tfidf_matrix_query = vectorizer.transform([processed_query])
    tfidf_df = pd.DataFrame(tfidf_matrix_dataset.toarray(), columns=vectorizer.get_feature_names_out())
    print(tfidf_df)
    joblib.dump(tfidf_matrix_dataset, 'tfidf_matrix_dataset.pkl')
    

# Menampilkan vektor TF-IDF untuk setiap kata dalam teks input
input_tokens = text_tokenizing(processed_query)
input_tfidf_per_word = {word: tfidf_matrix_query[0, vectorizer.vocabulary_[word]] for word in input_tokens}

# print("Vektor TF-IDF per kata untuk Teks Input:")
# for word, tfidf_value in input_tfidf_per_word.items():
#     print(f"{word}: {tfidf_value}")

# menggabungkan vektor dataset dan query
from scipy.sparse import vstack
tfidf_matrix = vstack([tfidf_matrix_dataset, tfidf_matrix_query])

cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

# mendapatkan indeks dokumen terdekat dengan teks input berdasarkan cosine similarity
most_similar_idx = np.argmax(cosine_similarities)
most_similar_idx_str = str(most_similar_idx)
# print("most similiar index = " + most_similar_idx_str)

# most_similar_text = texts[most_similar_idx]
# print("Pertanyaan yang paling mirip dengan teks input:")
# print(most_similar_text)

jawaban = dataset['answer'][most_similar_idx]
print("\nanswer:")
print(jawaban)



# # Preprocessing
# preprocessed_text = text_preprocessing(most_similar_text)

# # Tokenizing
# tokenized_text = text_tokenizing(preprocessed_text)

# # Filtering
# filtered_text = text_filtering(tokenized_text)

# # Stemming
# stemmed_text = text_stemming(filtered_text)

# Print hasil
# print("Teks Awal:", most_similar_text)
# print("Hasil Preprocessing:", preprocessed_text)
# print("Hasil Tokenizing:", tokenized_text)
# print("Hasil Filtering:", filtered_text)
# print("Hasil Stemming:", stemmed_text)

# Menampilkan jawaban dari kolom "answer"
# answer_text = dataset['answer'][most_similar_idx]
# print("Jawaban Awal:", answer_text)

# Preprocessing pada jawaban
# preprocessed_answer = text_preprocessing(answer_text)
# tokenized_answer = text_tokenizing(preprocessed_answer)
# filtered_answer = text_filtering(tokenized_answer)
# stemmed_answer = text_stemming(filtered_answer)

# # Print hasil Tokenizing
# print("\nHasil Preprocessing Jawaban:", preprocessed_answer)
# print("Hasil Tokenizing Jawaban:", tokenized_answer)
# print("Hasil Filtering Jawaban:", filtered_answer)
# print("Hasil Stemming Jawaban:", stemmed_answer)