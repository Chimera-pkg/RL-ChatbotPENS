import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# MENGAMBIL DATASET
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
    joblib.dump(processed_texts, 'processed_texts.pkl')

query = input("Masukkan pertanyaan Anda: ")

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

from scipy.sparse import vstack
tfidf_matrix = vstack([tfidf_matrix_dataset, tfidf_matrix_query])
cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
top_3_indices = np.argsort(cosine_similarities[0])[-3:][::-1]  # Get top 3 indices with highest similarity scores

print("\nPertanyaan Anda:")
print(query)

# Prepare the answers
answers = []
for idx in top_3_indices:
    answer = dataset.iloc[idx]['answer']
    answer2 = dataset.iloc[idx].get('answer2', '')
    answer3 = dataset.iloc[idx].get('answer3', '')
    answers.append({
        'answer': answer,
        'answer2': answer2,
        'answer3': answer3
    })

# Display the answers
for i, ans in enumerate(answers):
    print(f"\nJawaban {i + 1}:")
    print("Answer:", ans['answer'])
    if ans['answer2']:
        print("Answer2:", ans['answer2'])
    if ans['answer3']:
        print("Answer3:", ans['answer3'])
